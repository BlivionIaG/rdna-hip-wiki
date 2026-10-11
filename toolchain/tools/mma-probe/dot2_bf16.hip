// Family: packed BF16 dot product into FP32 (gfx11 only).
//   gfx1100: V_DOT2_F32_BF16 (VOP3P). Not present on gfx1030 (no dot12-insts).
// Record: a = bf16x2 bits, b = bf16x2 bits, c = f32 bits.  d = f32 bits.
#include "probe_common.h"

#if MMAPROBE_ARCH_ID != 1100
#error "dot2_bf16.hip is gfx1100-only: gfx1030 has no V_DOT2_F32_BF16"
#endif

static __device__ inline uint32_t mmaprobe_op_dot2_f32_bf16(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("v_dot2_f32_bf16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}
static __device__ inline uint32_t mmaprobe_op_fdot2_f32_bf16_builtin(uint32_t a, uint32_t b, uint32_t c) {
  float r = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(mmaprobe_s2, a),
                                           __builtin_bit_cast(mmaprobe_s2, b),
                                           __builtin_bit_cast(float, c), false);
  return __builtin_bit_cast(uint32_t, r);
}

MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, dot2_f32_bf16)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, fdot2_f32_bf16_builtin)

#ifndef MMAPROBE_DEVICE_ONLY
static const mmaprobe_entry kEntries[] = {
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, dot2_f32_bf16)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, fdot2_f32_bf16_builtin)};
int main(int argc, char** argv) {
  return mmaprobe_main(argc, argv, kEntries, sizeof(kEntries) / sizeof(kEntries[0]), "dot2_bf16");
}
#endif
