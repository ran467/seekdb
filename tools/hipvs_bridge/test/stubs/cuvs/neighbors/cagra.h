/*
 * Minimal cuVS CAGRA C API declarations for the offline bridge test.
 * See dlpack/dlpack.h in this directory for the rationale.
 */
#ifndef CUVS_NEIGHBORS_CAGRA_H_
#define CUVS_NEIGHBORS_CAGRA_H_

#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NN_DESCENT = 0,
  IVF_PQ = 1,
} cuvsCagraBuildAlgo;

struct cuvsCagraIndexParams {
  cuvsCagraBuildAlgo build_algo;
  size_t graph_degree;
  size_t intermediate_graph_degree;
  size_t nn_descent_niter;
  size_t nn_descent_itopk_size;
  size_t nn_descent_max_degree;
  size_t kmeans_niter;
  size_t num_pq_centroids;
  size_t pq_code_size;
};
typedef struct cuvsCagraIndexParams *cuvsCagraIndexParams_t;

typedef struct cuvsCagraIndex *cuvsCagraIndex_t;

struct cuvsCagraSearchParams {
  size_t itopk_size;
  size_t search_width;
  size_t min_iterations;
  size_t max_iterations;
};
typedef struct cuvsCagraSearchParams *cuvsCagraSearchParams_t;

typedef enum {
  NO_FILTER = 0,
} cuvsFilterType;

typedef struct cuvsFilter {
  cuvsFilterType type;
  uintptr_t addr;
} cuvsFilter;

cuvsStatus_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t *params);
cuvsStatus_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params);
cuvsStatus_t cuvsCagraIndexCreate(cuvsCagraIndex_t *index);
cuvsStatus_t cuvsCagraIndexDestroy(cuvsCagraIndex_t index);
cuvsStatus_t cuvsCagraBuild(cuvsResources_t res,
                             cuvsCagraIndexParams_t params,
                             DLManagedTensor *dataset,
                             cuvsCagraIndex_t index);
cuvsStatus_t cuvsCagraSearch(cuvsResources_t res,
                              cuvsCagraSearchParams_t params,
                              cuvsCagraIndex_t index,
                              DLManagedTensor *queries,
                              DLManagedTensor *neighbors,
                              DLManagedTensor *distances,
                              cuvsFilter filter);
cuvsStatus_t cuvsCagraSerialize(cuvsResources_t res,
                                 const char *filename,
                                 cuvsCagraIndex_t index,
                                 bool include_dataset);
cuvsStatus_t cuvsCagraDeserialize(cuvsResources_t res,
                                   const char *filename,
                                   cuvsCagraIndex_t index);
cuvsStatus_t cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t *size);
cuvsStatus_t cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t *dims);
cuvsStatus_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t *params);
cuvsStatus_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params);

#ifdef __cplusplus
}
#endif

#endif /* CUVS_NEIGHBORS_CAGRA_H_ */
