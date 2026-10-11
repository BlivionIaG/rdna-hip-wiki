// Device-only shim so the probes can be compiled to ISA without a ROCm install
// (clang -x hip --cuda-device-only -nogpulib -nogpuinc). Used only by
// `build.sh --device-only`; a real build uses <hip/hip_runtime.h> via hipcc.
#pragma once
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __restrict__ __restrict
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned short uint16_t;
struct mmaprobe_dim3 { unsigned x, y, z; };
#define threadIdx (mmaprobe_dim3{__builtin_amdgcn_workitem_id_x(), 0u, 0u})
#define blockIdx (mmaprobe_dim3{__builtin_amdgcn_workgroup_id_x(), 0u, 0u})
#define blockDim (mmaprobe_dim3{(unsigned)__builtin_amdgcn_workgroup_size_x(), 1u, 1u})
