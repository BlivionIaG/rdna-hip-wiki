// Family: packed FP16 dot product into FP32.
//   gfx1030: V_DOT2_F32_F16 (VOP3P) and V_DOT2C_F32_F16 (VOP2, dst = accumulator)
//   gfx1100: V_DOT2_F32_F16 (VOP3P) and V_DOT2ACC_F32_F16 (VOP2, gfx11 name)
// Record: a = half2 bits, b = half2 bits, c = f32 bits.  d = f32 bits.
// d = a.lo*b.lo + a.hi*b.hi + c   (ISA pseudocode; precision is what we probe)
#include "probe_common.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1030__) && !defined(__gfx1100__)
#error "dot2_f16.hip targets gfx1030 or gfx1100 only"
#endif

// Forced VOP3P encoding: three sources, separate destination.
static __device__ inline uint32_t mmaprobe_op_dot2_f32_f16(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}

// Forced VOP2 accumulate-in-place encoding.
static __device__ inline uint32_t mmaprobe_op_dot2c_f32_f16(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d = c;
#if defined(__gfx1100__)
  asm volatile("v_dot2acc_f32_f16 %0, %1, %2" : "+v"(d) : "v"(a), "v"(b));
#else
  asm volatile("v_dot2c_f32_f16 %0, %1, %2" : "+v"(d) : "v"(a), "v"(b));
#endif
  return d;
}

// What kernels in the fork actually call. The compiler picks the encoding;
// check_isa.py records which one it picked.
static __device__ inline uint32_t mmaprobe_op_fdot2_builtin(uint32_t a, uint32_t b, uint32_t c) {
  float r = __builtin_amdgcn_fdot2(__builtin_bit_cast(mmaprobe_h2, a),
                                   __builtin_bit_cast(mmaprobe_h2, b),
                                   __builtin_bit_cast(float, c), false);
  return __builtin_bit_cast(uint32_t, r);
}

static __device__ inline uint32_t mmaprobe_op_fdot2_builtin_clamp(uint32_t a, uint32_t b, uint32_t c) {
  float r = __builtin_amdgcn_fdot2(__builtin_bit_cast(mmaprobe_h2, a),
                                   __builtin_bit_cast(mmaprobe_h2, b),
                                   __builtin_bit_cast(float, c), true);
  return __builtin_bit_cast(uint32_t, r);
}

MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, dot2_f32_f16)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, dot2c_f32_f16)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, fdot2_builtin)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, fdot2_builtin_clamp)

#ifndef MMAPROBE_DEVICE_ONLY
static const mmaprobe_entry kEntries[] = {
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, dot2_f32_f16)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, dot2c_f32_f16)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, fdot2_builtin)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, fdot2_builtin_clamp)};
int main(int argc, char** argv) {
  return mmaprobe_main(argc, argv, kEntries, sizeof(kEntries) / sizeof(kEntries[0]), "dot2_f16");
}
#endif
