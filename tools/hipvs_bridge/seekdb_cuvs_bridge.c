/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

/* ---------------------------------------------------------------------------
 * Tuning knobs and phase instrumentation.
 *
 * Every knob keeps the frozen V5 baseline value as its default, so a run with an
 * untouched environment behaves exactly like the baseline. Overrides are read
 * from the process environment, which lets a single binary be swept on the
 * cloud instance without recompiling and without touching the evaluation SQL.
 *
 *   OB_CUVS_NN_DESCENT_NITER          CAGRA nn-descent iterations   (default 100)
 *   OB_CUVS_GRAPH_DEGREE               final graph degree             (default 128)
 *   OB_CUVS_INTERMEDIATE_GRAPH_DEGREE  intermediate graph degree     (default 256)
 *   OB_CUVS_SEARCH_WIDTH               CAGRA search width             (default 4)
 *   OB_CUVS_ITOPK_SIZE                 search itopk_size (0 = baseline rule)
 *   OB_CUVS_TIMING                     "1" -> print phase timings to stderr
 * ------------------------------------------------------------------------- */
#define OB_CUVS_DEFAULT_NN_DESCENT_NITER         100
#define OB_CUVS_DEFAULT_GRAPH_DEGREE              128
#define OB_CUVS_DEFAULT_INTERMEDIATE_GRAPH_DEGREE 256
#define OB_CUVS_DEFAULT_SEARCH_WIDTH              4
#define OB_CUVS_DEFAULT_ITOPK_SIZE                0

static long ob_cuvs_env_long(const char *name, long fallback)
{
  const char *raw = getenv(name);
  if (raw == NULL || raw[0] == '\0') { return fallback; }
  char *end = NULL;
  long value = strtol(raw, &end, 10);
  if (end == raw || value < 0) {
    fprintf(stderr, "[seekdb_cuvs] ignoring invalid %s=%s\n", name, raw);
    return fallback;
  }
  return value;
}

static int ob_cuvs_env_flag(const char *name)
{
  const char *raw = getenv(name);
  return (raw != NULL && raw[0] == '1') ? 1 : 0;
}

static double ob_cuvs_now_ms(void)
{
#if defined(_WIN32)
  LARGE_INTEGER freq;
  LARGE_INTEGER now;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&now);
  return ((double)now.QuadPart * 1000.0) / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((double)ts.tv_sec) * 1000.0 + ((double)ts.tv_nsec) / 1000000.0;
#endif
}

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
} sk_cuvs_index;

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  const int niter = (int)ob_cuvs_env_long("OB_CUVS_NN_DESCENT_NITER",
                                           OB_CUVS_DEFAULT_NN_DESCENT_NITER);
  const int graph_degree = (int)ob_cuvs_env_long("OB_CUVS_GRAPH_DEGREE",
                                                 OB_CUVS_DEFAULT_GRAPH_DEGREE);
  const int inter_degree = (int)ob_cuvs_env_long("OB_CUVS_INTERMEDIATE_GRAPH_DEGREE",
                                                 OB_CUVS_DEFAULT_INTERMEDIATE_GRAPH_DEGREE);
  const int timing = ob_cuvs_env_flag("OB_CUVS_TIMING");
  const double t_start = ob_cuvs_now_ms();
  if (base == NULL || n <= 0 || dim <= 0) { return NULL; }
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; int64_t ds[2] = {(int64_t)n, (int64_t)dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = graph_degree;
  h->ip->intermediate_graph_degree = inter_degree;
  h->ip->nn_descent_niter = niter;
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
  h->n = n; h->dim = dim;
  if (timing) {
    fprintf(stderr, "[seekdb_cuvs] build n=%ld dim=%ld niter=%d gd=%d igd=%d took=%.1f ms\n",
            n, dim, niter, graph_degree, inter_degree, ob_cuvs_now_ms() - t_start);
    fflush(stderr);
  }
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  const int timing = ob_cuvs_env_flag("OB_CUVS_TIMING");
  const double t_start = ob_cuvs_now_ms();
  int rc = 0;
  float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
  if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float)*nq*h->dim)  != CUVS_SUCCESS) { rc=10; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned)*nq*topk) != CUVS_SUCCESS) { rc=11; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float)*nq*topk)    != CUVS_SUCCESS) { rc=12; goto done; }
  if (cudaMemcpy(qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc=13; goto done; }
  {
    DLManagedTensor qt, nt, dt; int64_t qs[2]={(int64_t)nq,(int64_t)h->dim}, nsh[2]={(int64_t)nq,(int64_t)topk}, dsh[2]={(int64_t)nq,(int64_t)topk};
    qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc=14; goto done; }
    {
      size_t required_itopk = (size_t)topk * 4;
      if (required_itopk < 512) required_itopk = 512;
      if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
      sp->search_width = OB_CUVS_DEFAULT_SEARCH_WIDTH;
      {
        const long itopk = ob_cuvs_env_long("OB_CUVS_ITOPK_SIZE", OB_CUVS_DEFAULT_ITOPK_SIZE);
        const long width = ob_cuvs_env_long("OB_CUVS_SEARCH_WIDTH", OB_CUVS_DEFAULT_SEARCH_WIDTH);
        if (itopk > 0) { sp->itopk_size = (size_t)itopk; }
        sp->search_width = (size_t)width;
      }
    }
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc=15; goto done; }
    if (cudaMemcpy(out_ids, nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) { rc=16; goto done; }
    if (out_dist) cudaMemcpy(out_dist, dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
done:
  if (sp) cuvsCagraSearchParamsDestroy(sp);
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float)*nq*topk);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(unsigned)*nq*topk);
  if (qd) cuvsRMMFree(h->res, qd, sizeof(float)*nq*h->dim);
  if (rc) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  if (timing) {
    fprintf(stderr, "[seekdb_cuvs] search nq=%ld topk=%ld took=%.1f ms\n",
            nq, topk, ob_cuvs_now_ms() - t_start);
    fflush(stderr);
  }
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  const int timing = ob_cuvs_env_flag("OB_CUVS_TIMING");
  const double t_start = ob_cuvs_now_ms();
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  if (timing) {
    fprintf(stderr, "[seekdb_cuvs] serialize n=%ld dim=%ld took=%.1f ms\n",
            index->n, index->dim, ob_cuvs_now_ms() - t_start);
    fflush(stderr);
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  const int timing = ob_cuvs_env_flag("OB_CUVS_TIMING");
  const double t_start = ob_cuvs_now_ms();
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  if (timing) {
    fprintf(stderr, "[seekdb_cuvs] deserialize n=%ld dim=%ld took=%.1f ms\n",
            n, dim, ob_cuvs_now_ms() - t_start);
    fflush(stderr);
  }
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
