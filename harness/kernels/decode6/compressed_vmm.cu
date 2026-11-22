// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <ATen/dlpack.h>
#include <new>

// Keep allocator setup outside the timed graph. Return the actual allocation
// property, since the driver may decline a compression request.
extern "C" int stream_vmm_create(size_t bytes, int compressed, CUdeviceptr* ptr,
                                  size_t* reserved, int* actual) {
  CUdevice device;
  CUresult rc = cuCtxGetDevice(&device);
  if (rc != CUDA_SUCCESS) return rc;
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
  prop.allocFlags.compressionType = compressed ? CU_MEM_ALLOCATION_COMP_GENERIC
                                               : CU_MEM_ALLOCATION_COMP_NONE;
  size_t granularity;
  rc = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (rc != CUDA_SUCCESS) return rc;
  const size_t capacity = ((bytes + granularity - 1) / granularity) * granularity;
  CUmemGenericAllocationHandle handle;
  rc = cuMemCreate(&handle, capacity, &prop, 0);
  if (rc != CUDA_SUCCESS) return rc;
  CUmemAllocationProp obtained{};
  CUdeviceptr address = 0;
  bool mapped = false;
  rc = cuMemGetAllocationPropertiesFromHandle(&obtained, handle);
  if (rc == CUDA_SUCCESS) rc = cuMemAddressReserve(&address, capacity, 0, 0, 0);
  if (rc == CUDA_SUCCESS) {
    rc = cuMemMap(address, capacity, 0, handle, 0);
    mapped = (rc == CUDA_SUCCESS);
  }
  if (rc == CUDA_SUCCESS) {
    CUmemAccessDesc access{prop.location, CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    rc = cuMemSetAccess(address, capacity, &access, 1);
  }
  cuMemRelease(handle);
  if (rc != CUDA_SUCCESS) {
    if (mapped) cuMemUnmap(address, capacity);
    if (address) cuMemAddressFree(address, capacity);
    return rc;
  }
  *ptr = address;
  *reserved = capacity;
  *actual = obtained.allocFlags.compressionType;
  return CUDA_SUCCESS;
}

extern "C" int stream_vmm_destroy(CUdeviceptr ptr, size_t reserved) {
  CUresult rc = cuMemUnmap(ptr, reserved);
  return rc == CUDA_SUCCESS ? cuMemAddressFree(ptr, reserved) : rc;
}

namespace {
struct OwnedBuffer {
  DLManagedTensor tensor{};
  CUdeviceptr pointer = 0;
  size_t reserved = 0;
  int64_t elements = 0;
};
}  // namespace

extern "C" void muse_release_compressed(DLManagedTensor* tensor) {
  auto* owner = static_cast<OwnedBuffer*>(tensor->manager_ctx);
  stream_vmm_destroy(owner->pointer, owner->reserved);
  delete owner;
}

extern "C" DLManagedTensor* muse_alloc_compressed(size_t bytes) {
  if (bytes == 0) return nullptr;
  CUdevice device;
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS) return nullptr;
  int supported = 0;
  if (cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
                           device) != CUDA_SUCCESS || !supported) return nullptr;
  auto* owner = new (std::nothrow) OwnedBuffer;
  if (!owner) return nullptr;
  int actual = 0;
  int rc = stream_vmm_create(bytes, 1, &owner->pointer, &owner->reserved, &actual);
  if (rc || actual != CU_MEM_ALLOCATION_COMP_GENERIC) {
    if (!rc) stream_vmm_destroy(owner->pointer, owner->reserved);
    delete owner;
    return nullptr;
  }
  owner->elements = bytes;
  owner->tensor.manager_ctx = owner;
  owner->tensor.deleter = muse_release_compressed;
  owner->tensor.dl_tensor = {reinterpret_cast<void*>(owner->pointer), {kDLCUDA, device},
                             1, {kDLUInt, 8, 1}, &owner->elements, nullptr, 0};
  return &owner->tensor;
}
