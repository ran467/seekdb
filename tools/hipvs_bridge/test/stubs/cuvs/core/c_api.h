/*
 * Minimal cuVS core C API declarations for the offline bridge test.
 * See dlpack/dlpack.h in this directory for the rationale.
 */
#ifndef CUVS_CORE_C_API_H_
#define CUVS_CORE_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuvsResources *cuvsResources_t;

typedef enum {
  CUVS_SUCCESS = 0,
  CUVS_INVALID_VALUE = 1,
  CUVS_OUT_OF_MEMORY = 2,
  CUVS_NOT_IMPLEMENTED = 3,
  CUVS_UNKNOWN_ERROR = 4,
} cuvsStatus_t;

cuvsStatus_t cuvsResourcesCreate(cuvsResources_t *res);
cuvsStatus_t cuvsResourcesDestroy(cuvsResources_t res);
cuvsStatus_t cuvsRMMAlloc(cuvsResources_t res, void **addr, size_t bytes);
cuvsStatus_t cuvsRMMFree(cuvsResources_t res, void *addr, size_t bytes);
const char *cuvsGetLastErrorText(void);

#ifdef __cplusplus
}
#endif

#endif /* CUVS_CORE_C_API_H_ */
