// Family: RDNA3 WMMA, FP32 accumulate (gfx11 only, wave32).
//   v_wmma_f32_16x16x16_f16   : A, B = 16 x f16 per lane, C/D = 8 x f32 per lane
//   v_wmma_f32_16x16x16_bf16  : A, B = 16 x bf16 per lane
// The kernel broadcasts one K-vector to every row of A and every column of B,
// so each of the 256 outputs is the same dot product (see probe_common.h).
#include "probe_common.h"

#if MMAPROBE_ARCH_ID != 1100
#error "wmma.hip is gfx1100-only: gfx1030 has no WMMA"
#endif

static __device__ inline mmaprobe_v8u mmaprobe_c8(uint32_t c) {
  mmaprobe_v8u v;
  for (int i = 0; i < 8; ++i) v[i] = c;
  return v;
}
static __device__ inline mmaprobe_v8u mmaprobe_op_wmma_f32_16x16x16_f16(mmaprobe_v8u A, mmaprobe_v8u B, uint32_t c) {
  mmaprobe_v8f C = __builtin_bit_cast(mmaprobe_v8f, mmaprobe_c8(c));
  mmaprobe_v8f D = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
      __builtin_bit_cast(mmaprobe_h16, A), __builtin_bit_cast(mmaprobe_h16, B), C);
  return __builtin_bit_cast(mmaprobe_v8u, D);
}
static __device__ inline mmaprobe_v8u mmaprobe_op_wmma_f32_16x16x16_bf16(mmaprobe_v8u A, mmaprobe_v8u B, uint32_t c) {
  mmaprobe_v8f C = __builtin_bit_cast(mmaprobe_v8f, mmaprobe_c8(c));
  mmaprobe_v8f D = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
      __builtin_bit_cast(mmaprobe_s16, A), __builtin_bit_cast(mmaprobe_s16, B), C);
  return __builtin_bit_cast(mmaprobe_v8u, D);
}

MMAPROBE_FP_MODES(MMAPROBE_DEF_WMMA_KERNEL, wmma_f32_16x16x16_f16)
MMAPROBE_FP_MODES(MMAPROBE_DEF_WMMA_KERNEL, wmma_f32_16x16x16_bf16)

#ifndef MMAPROBE_DEVICE_ONLY
static const mmaprobe_entry kEntries[] = {
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_WMMA, wmma_f32_16x16x16_f16)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_WMMA, wmma_f32_16x16x16_bf16)};
int main(int argc, char** argv) {
  return mmaprobe_main(argc, argv, kEntries, sizeof(kEntries) / sizeof(kEntries[0]), "wmma");
}
#endif
