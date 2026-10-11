// Family: packed INT8 dot product into INT32 (sdot4).
//   gfx1030: V_DOT4_I32_I8 (VOP3P), V_DOT4C_I32_I8 (VOP2), builtin sdot4
//   gfx1100: no V_DOT4_I32_I8; signed i8 x i8 goes through V_DOT4_I32_IU8
//            with neg_lo:[1,1,0] (builtin sudot4 with both signs set).
// Record: a = 4 x i8, b = 4 x i8, c = i32.  d = i32.
// Integer ops do not read MODE, so only the "desc" variant is built.
#include "probe_common.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1030__) && !defined(__gfx1100__)
#error "dot4_i8.hip targets gfx1030 or gfx1100 only"
#endif

#if MMAPROBE_ARCH_ID == 1030
static __device__ inline uint32_t mmaprobe_op_dot4_i32_i8(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}
static __device__ inline uint32_t mmaprobe_op_dot4_i32_i8_clamp(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("v_dot4_i32_i8 %0, %1, %2, %3 clamp" : "=v"(d) : "v"(a), "v"(b), "v"(c));
  return d;
}
static __device__ inline uint32_t mmaprobe_op_dot4c_i32_i8(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d = c;
  asm volatile("v_dot4c_i32_i8 %0, %1, %2" : "+v"(d) : "v"(a), "v"(b));
  return d;
}
static __device__ inline uint32_t mmaprobe_op_sdot4_builtin(uint32_t a, uint32_t b, uint32_t c) {
  return (uint32_t)__builtin_amdgcn_sdot4((int)a, (int)b, (int)c, false);
}
MMAPROBE_INT_MODES(MMAPROBE_DEF_SCALAR_KERNEL, dot4_i32_i8)
MMAPROBE_INT_MODES(MMAPROBE_DEF_SCALAR_KERNEL, dot4_i32_i8_clamp)
MMAPROBE_INT_MODES(MMAPROBE_DEF_SCALAR_KERNEL, dot4c_i32_i8)
MMAPROBE_INT_MODES(MMAPROBE_DEF_SCALAR_KERNEL, sdot4_builtin)
#ifndef MMAPROBE_DEVICE_ONLY
static const mmaprobe_entry kEntries[] = {
    MMAPROBE_INT_MODES(MMAPROBE_ENTRY_SCALAR, dot4_i32_i8)
    MMAPROBE_INT_MODES(MMAPROBE_ENTRY_SCALAR, dot4_i32_i8_clamp)
    MMAPROBE_INT_MODES(MMAPROBE_ENTRY_SCALAR, dot4c_i32_i8)
    MMAPROBE_INT_MODES(MMAPROBE_ENTRY_SCALAR, sdot4_builtin)};
#endif
#else  // gfx1100
static __device__ inline uint32_t mmaprobe_op_sudot4_ss_builtin(uint32_t a, uint32_t b, uint32_t c) {
  return (uint32_t)__builtin_amdgcn_sudot4(true, (int)a, true, (int)b, (int)c, false);
}
static __device__ inline uint32_t mmaprobe_op_sudot4_ss_builtin_clamp(uint32_t a, uint32_t b, uint32_t c) {
  return (uint32_t)__builtin_amdgcn_sudot4(true, (int)a, true, (int)b, (int)c, true);
}
MMAPROBE_INT_MODES(MMAPROBE_DEF_SCALAR_KERNEL, sudot4_ss_builtin)
MMAPROBE_INT_MODES(MMAPROBE_DEF_SCALAR_KERNEL, sudot4_ss_builtin_clamp)
#ifndef MMAPROBE_DEVICE_ONLY
static const mmaprobe_entry kEntries[] = {
    MMAPROBE_INT_MODES(MMAPROBE_ENTRY_SCALAR, sudot4_ss_builtin)
    MMAPROBE_INT_MODES(MMAPROBE_ENTRY_SCALAR, sudot4_ss_builtin_clamp)};
#endif
#endif

#ifndef MMAPROBE_DEVICE_ONLY
int main(int argc, char** argv) {
  return mmaprobe_main(argc, argv, kEntries, sizeof(kEntries) / sizeof(kEntries[0]), "dot4_i8");
}
#endif
