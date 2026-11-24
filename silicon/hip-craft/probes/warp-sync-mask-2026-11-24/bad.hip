#include <hip/hip_runtime.h>
__global__ void bad(float* o, const float* i) {
  float v = i[threadIdx.x];
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  o[threadIdx.x] = v;
}
