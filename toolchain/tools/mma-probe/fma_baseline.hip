// Family: non-DOT baselines on the same VALU, built with -ffp-contract=fast.
//   fma_mix_f32 : V_FMA_MIX_F32, d = f32(a.lo f16) * f32(b.lo f16) + c(f32)
//   pk_fma_f16  : V_PK_FMA_F16, two FP16 FMAs; record words are half2 bits
//   fma_f32     : V_FMA_F32 on f32 bits (FP32 denorm-mode sanity check)
//   fma_c       : __builtin_fmaf, compiler-chosen FP32 FMA encoding
//   mul_add_c   : a*b + c with `#pragma clang fp contract(off)` -> separate
//                 v_mul_f32 + v_add_f32 (the unfused reference)
//   fma_mix_c   : (float)h*(float)h + c, contract fast; records whether the
//                 compiler selects v_fma_mix_f32 (LLVM may also fold a pair of
//                 these into v_dot2; see README "fp-contract")
#include "probe_common.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1030__) && !defined(__gfx1100__)
#error "fma_baseline.hip targets gfx1030 or gfx1100 only"
#endif

static __device__ inline uint32_t mmaprobe_op_fma_mix_f32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("v_fma_mix_f32 %0, %1, %2, %3 op_sel_hi:[1,1,0]"
               : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}
static __device__ inline uint32_t mmaprobe_op_pk_fma_f16(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("v_pk_fma_f16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}
static __device__ inline uint32_t mmaprobe_op_fma_f32(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("v_fma_f32 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}
static __device__ inline uint32_t mmaprobe_op_fma_c(uint32_t a, uint32_t b, uint32_t c) {
  float r = __builtin_fmaf(__builtin_bit_cast(float, a), __builtin_bit_cast(float, b),
                           __builtin_bit_cast(float, c));
  return __builtin_bit_cast(uint32_t, r);
}
static __device__ inline uint32_t mmaprobe_op_mul_add_c(uint32_t a, uint32_t b, uint32_t c) {
#pragma clang fp contract(off)
  float r = __builtin_bit_cast(float, a) * __builtin_bit_cast(float, b) +
            __builtin_bit_cast(float, c);
  return __builtin_bit_cast(uint32_t, r);
}
static __device__ inline uint32_t mmaprobe_op_fma_mix_c(uint32_t a, uint32_t b, uint32_t c) {
  mmaprobe_h2 ha = __builtin_bit_cast(mmaprobe_h2, a);
  mmaprobe_h2 hb = __builtin_bit_cast(mmaprobe_h2, b);
  float r = (float)ha.x * (float)hb.x + __builtin_bit_cast(float, c);
  return __builtin_bit_cast(uint32_t, r);
}

MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, fma_mix_f32)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, pk_fma_f16)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, fma_f32)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, fma_c)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, mul_add_c)
MMAPROBE_FP_MODES(MMAPROBE_DEF_SCALAR_KERNEL, fma_mix_c)

#ifndef MMAPROBE_DEVICE_ONLY
static const mmaprobe_entry kEntries[] = {
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, fma_mix_f32)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, pk_fma_f16)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, fma_f32)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, fma_c)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, mul_add_c)
    MMAPROBE_FP_MODES(MMAPROBE_ENTRY_SCALAR, fma_mix_c)};
int main(int argc, char** argv) {
  return mmaprobe_main(argc, argv, kEntries, sizeof(kEntries) / sizeof(kEntries[0]), "fma_baseline");
}
#endif
