#include <hip/hip_runtime.h>
__global__ void red_sync(float* o, const float* i) {
  float v = i[threadIdx.x];
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffffffffffULL, v, off);
  o[threadIdx.x] = v;
}
__global__ void red_nosync(float* o, const float* i) {
  float v = i[threadIdx.x];
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor(v, off);
  o[threadIdx.x] = v;
}
__global__ void vote(unsigned long long* o, const int* p) {
  o[2 * threadIdx.x] = __ballot(p[threadIdx.x]);
  o[2 * threadIdx.x + 1] = __activemask();
}
__global__ void ws(int* o) { o[threadIdx.x] = warpSize; }
