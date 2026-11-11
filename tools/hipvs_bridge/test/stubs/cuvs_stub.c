/*
 * Offline stand-in for the hipVS/cuVS C API.
 *
 * It implements just enough of cuvsCagra* to compile and exercise
 * tools/hipvs_bridge/seekdb_cuvs_bridge.c on a machine without a GPU:
 * a brute-force exact kNN instead of the real approximate CAGRA search.
 * The point is to validate the bridge plumbing (argument wiring, parameter
 * overrides, serialize/deserialize round trip, error paths), never the
 * performance.
 *
 * The stub also records the parameters it was handed so the test can assert
 * that the environment overrides really reach cuVS.
 */
#include "cuvs/core/c_api.h"
#include "cuvs/neighbors/cagra.h"
#include "cuda_runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUB_FILE_MAGIC 0x53435631u /* "SCV1" */

static const char *g_last_error = "no error";
const char *ob_cuvs_stub_last_error(void) { return g_last_error; }
const char *cuvsGetLastErrorText(void)
{
  return g_last_error;
}

struct cuvsCagraIndex {
  int64_t n;
  int64_t dim;
  float *data;      /* row-major, n * dim */
  size_t graph_degree;
  size_t intermediate_graph_degree;
  size_t nn_descent_niter;
  int build_algo;
};

struct cuvsResources {
  int unused;
};

/* --- recorded state, read by the test ------------------------------------ */
static int g_last_niter = -1;
static int g_last_graph_degree = -1;
static int g_last_intermediate_degree = -1;
static int g_last_build_algo = -1;
static size_t g_last_itopk = 0;
static size_t g_last_search_width = 0;
static long g_build_calls = 0;
static long g_search_calls = 0;

int ob_cuvs_stub_last_niter(void) { return g_last_niter; }
int ob_cuvs_stub_last_graph_degree(void) { return g_last_graph_degree; }
int ob_cuvs_stub_last_intermediate_degree(void) { return g_last_intermediate_degree; }
int ob_cuvs_stub_last_build_algo(void) { return g_last_build_algo; }
size_t ob_cuvs_stub_last_itopk(void) { return g_last_itopk; }
size_t ob_cuvs_stub_last_search_width(void) { return g_last_search_width; }
long ob_cuvs_stub_build_calls(void) { return g_build_calls; }
long ob_cuvs_stub_search_calls(void) { return g_search_calls; }

cuvsStatus_t cuvsResourcesCreate(cuvsResources_t *res)
{
  if (res == NULL) { g_last_error = "resourcesCreate"; return CUVS_INVALID_VALUE; }
  *res = (cuvsResources_t)calloc(1, sizeof(struct cuvsResources));
  return (*res == NULL) ? CUVS_OUT_OF_MEMORY : CUVS_SUCCESS;
}

cuvsStatus_t cuvsResourcesDestroy(cuvsResources_t res)
{
  free(res);
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsRMMAlloc(cuvsResources_t res, void **addr, size_t bytes)
{
  (void)res;
  if (addr == NULL) { return CUVS_INVALID_VALUE; }
  *addr = calloc(1, bytes == 0 ? 1 : bytes);
  return (*addr == NULL) ? CUVS_OUT_OF_MEMORY : CUVS_SUCCESS;
}

cuvsStatus_t cuvsRMMFree(cuvsResources_t res, void *addr, size_t bytes)
{
  (void)res;
  (void)bytes;
  free(addr);
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t *params)
{
  if (params == NULL) { g_last_error = "paramsCreate"; return CUVS_INVALID_VALUE; }
  *params = (cuvsCagraIndexParams_t)calloc(1, sizeof(struct cuvsCagraIndexParams));
  return (*params == NULL) ? CUVS_OUT_OF_MEMORY : CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params)
{
  free(params);
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraIndexCreate(cuvsCagraIndex_t *index)
{
  if (index == NULL) { g_last_error = "indexCreate"; return CUVS_INVALID_VALUE; }
  *index = (cuvsCagraIndex_t)calloc(1, sizeof(struct cuvsCagraIndex));
  return (*index == NULL) ? CUVS_OUT_OF_MEMORY : CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index)
{
  if (index == NULL) { return CUVS_INVALID_VALUE; }
  free(index->data);
  free(index);
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraBuild(cuvsResources_t res,
                             cuvsCagraIndexParams_t params,
                             DLManagedTensor *dataset,
                             cuvsCagraIndex_t index)
{
  (void)res;
  if (params == NULL || dataset == NULL || index == NULL) { g_last_error = "build: null argument"; return CUVS_INVALID_VALUE; }
  const DLTensor *t = &dataset->dl_tensor;
  if (t->ndim != 2 || t->dtype.code != kDLFloat || t->dtype.bits != 32) {
    g_last_error = "build: unexpected tensor layout";
    return CUVS_INVALID_VALUE;
  }
  const int64_t n = t->shape[0];
  const int64_t dim = t->shape[1];
  if (n <= 0 || dim <= 0) { g_last_error = "build: empty shape"; return CUVS_INVALID_VALUE; }
  free(index->data);
  index->data = (float *)malloc((size_t)(n * dim) * sizeof(float));
  if (index->data == NULL) { return CUVS_OUT_OF_MEMORY; }
  memcpy(index->data, t->data, (size_t)(n * dim) * sizeof(float));
  index->n = n;
  index->dim = dim;
  index->graph_degree = params->graph_degree;
  index->intermediate_graph_degree = params->intermediate_graph_degree;
  index->nn_descent_niter = params->nn_descent_niter;
  index->build_algo = (int)params->build_algo;
  g_last_niter = (int)params->nn_descent_niter;
  g_last_graph_degree = (int)params->graph_degree;
  g_last_intermediate_degree = (int)params->intermediate_graph_degree;
  g_last_build_algo = (int)params->build_algo;
  g_build_calls++;
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraSearch(cuvsResources_t res,
                              cuvsCagraSearchParams_t params,
                              cuvsCagraIndex_t index,
                              DLManagedTensor *queries,
                              DLManagedTensor *neighbors,
                              DLManagedTensor *distances,
                              cuvsFilter filter)
{
  (void)res;
  (void)filter;
  if (params == NULL || index == NULL || queries == NULL || neighbors == NULL) { return CUVS_INVALID_VALUE; }
  if (index->data == NULL) { return CUVS_INVALID_VALUE; }
  const DLTensor *qt = &queries->dl_tensor;
  const DLTensor *nt = &neighbors->dl_tensor;
  const DLTensor *dt = (distances != NULL) ? &distances->dl_tensor : NULL;
  if (qt->ndim != 2 || nt->ndim != 2) { return CUVS_INVALID_VALUE; }
  const int64_t nq = qt->shape[0];
  const int64_t dim = qt->shape[1];
  const int64_t topk = nt->shape[1];
  if (dim != index->dim || nq <= 0 || topk <= 0 || topk > index->n) { return CUVS_INVALID_VALUE; }

  const float *q = (const float *)qt->data;
  unsigned int *ids = (unsigned int *)nt->data;
  float *dists = (dt != NULL) ? (float *)dt->data : NULL;

  for (int64_t qi = 0; qi < nq; ++qi) {
    const float *query = q + qi * dim;
    /* partial selection sort over the (small) stub corpus */
    int64_t *order = (int64_t *)malloc((size_t)index->n * sizeof(int64_t));
    float *dist = (float *)malloc((size_t)index->n * sizeof(float));
    if (order == NULL || dist == NULL) { free(order); free(dist); return CUVS_OUT_OF_MEMORY; }
    for (int64_t i = 0; i < index->n; ++i) {
      const float *row = index->data + i * dim;
      float acc = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        const float diff = query[d] - row[d];
        acc += diff * diff;
      }
      dist[i] = acc;
      order[i] = i;
    }
    for (int64_t k = 0; k < topk; ++k) {
      int64_t best = k;
      for (int64_t i = k + 1; i < index->n; ++i) {
        if (dist[order[i]] < dist[order[best]]) { best = i; }
      }
      const int64_t tmp = order[k];
      order[k] = order[best];
      order[best] = tmp;
      ids[qi * topk + k] = (unsigned int)order[k];
      if (dists != NULL) { dists[qi * topk + k] = dist[order[k]]; }
    }
    free(order);
    free(dist);
  }
  g_last_itopk = params->itopk_size;
  g_last_search_width = params->search_width;
  g_search_calls++;
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraSerialize(cuvsResources_t res,
                                 const char *filename,
                                 cuvsCagraIndex_t index,
                                 bool include_dataset)
{
  (void)res;
  (void)include_dataset;
  if (filename == NULL || index == NULL || index->data == NULL) { return CUVS_INVALID_VALUE; }
  FILE *f = fopen(filename, "wb");
  if (f == NULL) { return CUVS_INVALID_VALUE; }
  const unsigned int magic = STUB_FILE_MAGIC;
  const int64_t n = index->n;
  const int64_t dim = index->dim;
  int ok = fwrite(&magic, sizeof(magic), 1, f) == 1
        && fwrite(&n, sizeof(n), 1, f) == 1
        && fwrite(&dim, sizeof(dim), 1, f) == 1
        && fwrite(index->data, sizeof(float), (size_t)(n * dim), f) == (size_t)(n * dim);
  fclose(f);
  return ok ? CUVS_SUCCESS : CUVS_INVALID_VALUE;
}

cuvsStatus_t cuvsCagraDeserialize(cuvsResources_t res,
                                   const char *filename,
                                   cuvsCagraIndex_t index)
{
  (void)res;
  if (filename == NULL || index == NULL) { return CUVS_INVALID_VALUE; }
  FILE *f = fopen(filename, "rb");
  if (f == NULL) { return CUVS_INVALID_VALUE; }
  unsigned int magic = 0;
  int64_t n = 0;
  int64_t dim = 0;
  int ok = fread(&magic, sizeof(magic), 1, f) == 1
        && fread(&n, sizeof(n), 1, f) == 1
        && fread(&dim, sizeof(dim), 1, f) == 1
        && magic == STUB_FILE_MAGIC && n > 0 && dim > 0;
  if (ok) {
    free(index->data);
    index->data = (float *)malloc((size_t)(n * dim) * sizeof(float));
    if (index->data == NULL) { ok = 0; }
    else {
      ok = fread(index->data, sizeof(float), (size_t)(n * dim), f) == (size_t)(n * dim);
      index->n = n;
      index->dim = dim;
      index->graph_degree = 0;
      index->intermediate_graph_degree = 0;
      index->nn_descent_niter = 0;
    }
  }
  fclose(f);
  return ok ? CUVS_SUCCESS : CUVS_INVALID_VALUE;
}

cuvsStatus_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t *size)
{
  if (index == NULL || size == NULL) { return CUVS_INVALID_VALUE; }
  *size = index->n;
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t *dims)
{
  if (index == NULL || dims == NULL) { return CUVS_INVALID_VALUE; }
  *dims = index->dim;
  return CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t *params)
{
  if (params == NULL) { return CUVS_INVALID_VALUE; }
  *params = (cuvsCagraSearchParams_t)calloc(1, sizeof(struct cuvsCagraSearchParams));
  return (*params == NULL) ? CUVS_OUT_OF_MEMORY : CUVS_SUCCESS;
}

cuvsStatus_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params)
{
  free(params);
  return CUVS_SUCCESS;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
  (void)kind;
  memcpy(dst, src, count);
  return cudaSuccess;
}
