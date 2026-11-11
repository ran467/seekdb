/*
 * Minimal DLPack declarations used by the offline bridge test.
 *
 * This is NOT a copy of the upstream dlpack header: it only declares the
 * handful of types and enum values that tools/hipvs_bridge/seekdb_cuvs_bridge.c
 * touches, so the bridge can be compiled and exercised on a machine without
 * hipVS/cuVS installed. The real build always uses the headers shipped with
 * hipVS (see tools/hipvs_bridge/build.sh).
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLROCM = 10,
} DLDeviceType;

typedef struct {
  int32_t device_type;
  int32_t device_id;
} DLDevice;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct DLPackManagedTensor {
  DLTensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(struct DLPackManagedTensor *self);
} DLManagedTensor;

enum {
  kDLInt = 0,
  kDLUInt = 1,
  kDLFloat = 2,
  kDLBfloat = 4,
  kDLBool = 5,
};

#ifdef __cplusplus
}
#endif

#endif /* DLPACK_DLPACK_H_ */
