/*
 * Minimal CUDA runtime declarations for the offline bridge test.
 * The real bridge includes the CUDA/ROCm runtime shipped with hipVS.
 */
#ifndef CUDA_RUNTIME_H_STUB_
#define CUDA_RUNTIME_H_STUB_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
} cudaError_t;

typedef enum {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
} cudaMemcpyKind;

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_RUNTIME_H_STUB_ */
