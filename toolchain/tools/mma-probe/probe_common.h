// mma-probe: shared kernel scaffolding and host runner.
//
// Every probe kernel reads fixed-size records of raw bit patterns, optionally
// rewrites the wave's MODE register (FP_ROUND [3:0], FP_DENORM [7:4]) with
// s_setreg_imm32_b32, runs exactly one instance of the instruction under test,
// reads MODE back with s_getreg_b32 and stores the raw result bits.
// All interpretation happens in harness.py. See README.md.
//
// One binary per instruction family per arch. Never build one object for both
// gfx1030 and gfx1100: build.sh passes exactly one --offload-arch and
// -DMMAPROBE_ARCH_ID=<1030|1100>, checked below in the device pass.
#pragma once

#ifdef MMAPROBE_DEVICE_ONLY
#include "shim/hip_shim.h"
#else
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#endif

#ifndef MMAPROBE_ARCH_ID
#error "build with -DMMAPROBE_ARCH_ID=1030 or 1100 (see build.sh)"
#endif

#if defined(__HIP_DEVICE_COMPILE__)
#if MMAPROBE_ARCH_ID == 1030 && !defined(__gfx1030__)
#error "MMAPROBE_ARCH_ID=1030 but the device pass is not gfx1030"
#endif
#if MMAPROBE_ARCH_ID == 1100 && !defined(__gfx1100__)
#error "MMAPROBE_ARCH_ID=1100 but the device pass is not gfx1100"
#endif
#endif

#define MMAPROBE_STR2(x) #x
#define MMAPROBE_STR(x) MMAPROBE_STR2(x)

typedef float mmaprobe_v8f __attribute__((ext_vector_type(8)));
typedef unsigned int mmaprobe_v8u __attribute__((ext_vector_type(8)));
typedef _Float16 mmaprobe_h2 __attribute__((ext_vector_type(2)));
typedef _Float16 mmaprobe_h16 __attribute__((ext_vector_type(16)));
typedef short mmaprobe_s2 __attribute__((ext_vector_type(2)));
typedef short mmaprobe_s16 __attribute__((ext_vector_type(16)));

// ---- MODE register control -------------------------------------------------
// MODE byte = FP_DENORM << 4 | FP_ROUND (RDNA2 ISA 70648 §3.5 / §6.4).
//   FP_ROUND  [1:0] f32, [3:2] f16/f64: 0 RNE, 1 +inf, 2 -inf, 3 toward zero.
//   FP_DENORM [5:4] f32, [7:6] f16/f64: 0 flush in+out, 1 allow in / flush out,
//                                       2 flush in / allow out, 3 allow both.
// The "+v" operands make the instruction under test data-dependent on the
// s_setreg, so the compiler cannot hoist it above the mode write.
#define MMAPROBE_SET(MV, x, y, z)                                              \
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 8), " #MV            \
               "\n\ts_nop 7\n\ts_nop 7"                                        \
               : "+v"(x), "+v"(y), "+v"(z));
#define MMAPROBE_NOSET(MV, x, y, z)

// Mode sweep for floating-point ops. "desc" = whatever the kernel descriptor
// (FLOAT_DENORM_MODE_32 / _16_64, FLOAT_ROUND_MODE_*) set at wave launch.
#define MMAPROBE_FP_MODES(DEF, OP)                                             \
  DEF(OP, desc, 0, MMAPROBE_NOSET)                                             \
  DEF(OP, m00, 0x00, MMAPROBE_SET) DEF(OP, m10, 0x10, MMAPROBE_SET)            \
  DEF(OP, m20, 0x20, MMAPROBE_SET) DEF(OP, m30, 0x30, MMAPROBE_SET)            \
  DEF(OP, m40, 0x40, MMAPROBE_SET) DEF(OP, m50, 0x50, MMAPROBE_SET)            \
  DEF(OP, m60, 0x60, MMAPROBE_SET) DEF(OP, m70, 0x70, MMAPROBE_SET)            \
  DEF(OP, m80, 0x80, MMAPROBE_SET) DEF(OP, m90, 0x90, MMAPROBE_SET)            \
  DEF(OP, ma0, 0xa0, MMAPROBE_SET) DEF(OP, mb0, 0xb0, MMAPROBE_SET)            \
  DEF(OP, mc0, 0xc0, MMAPROBE_SET) DEF(OP, md0, 0xd0, MMAPROBE_SET)            \
  DEF(OP, me0, 0xe0, MMAPROBE_SET) DEF(OP, mf0, 0xf0, MMAPROBE_SET)            \
  DEF(OP, mf1, 0xf1, MMAPROBE_SET) DEF(OP, mf2, 0xf2, MMAPROBE_SET)            \
  DEF(OP, mf3, 0xf3, MMAPROBE_SET) DEF(OP, mf4, 0xf4, MMAPROBE_SET)            \
  DEF(OP, mf8, 0xf8, MMAPROBE_SET) DEF(OP, mfc, 0xfc, MMAPROBE_SET)

// Integer ops do not read MODE; one variant is enough.
#define MMAPROBE_INT_MODES(DEF, OP) DEF(OP, desc, 0, MMAPROBE_NOSET)

static __device__ inline unsigned mmaprobe_getmode(unsigned dep) {
  unsigned m;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 8)" : "=s"(m) : "v"(dep));
  return m;
}

// ---- scalar record kernels: in = {a, b, c} words, out = {d, mode} ----------
#define MMAPROBE_DEF_SCALAR_KERNEL(OP, MNAME, MVAL, SETTER)                    \
  extern "C" __global__ void mmaprobe_##OP##__##MNAME(                         \
      const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {    \
    int t = blockIdx.x * blockDim.x + threadIdx.x;                             \
    uint32_t a = 0, b = 0, c = 0;                                              \
    if (t < n) { a = in[3 * t]; b = in[3 * t + 1]; c = in[3 * t + 2]; }        \
    SETTER(MVAL, a, b, c)                                                      \
    uint32_t d = mmaprobe_op_##OP(a, b, c);                                    \
    uint32_t m = mmaprobe_getmode(d);                                          \
    if (t < n) { out[2 * t] = d; out[2 * t + 1] = m; }                         \
  }

// ---- WMMA record kernels (one wave32 per record) ---------------------------
// in  = 8 words A (16 x 16-bit along K), 8 words B (16 x 16-bit along K), c
// out = {D[0] of lane 0, 1 if all 256 outputs are bit-identical, mode}
// Every row of A gets the same K-vector and every column of B the same
// K-vector, so all 256 outputs compute the same c + sum_k a_k b_k
// (MMA-Sim CLFP Step 1 cross-position consistency).
#define MMAPROBE_DEF_WMMA_KERNEL(OP, MNAME, MVAL, SETTER)                      \
  extern "C" __global__ void mmaprobe_##OP##__##MNAME(                         \
      const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {    \
    int t = blockIdx.x;                                                        \
    if (t >= n) return;                                                        \
    const uint32_t* r = in + 17 * t;                                           \
    mmaprobe_v8u A, B;                                                         \
    for (int i = 0; i < 8; ++i) { A[i] = r[i]; B[i] = r[8 + i]; }             \
    uint32_t c = r[16];                                                        \
    SETTER(MVAL, A, B, c)                                                      \
    mmaprobe_v8u D = mmaprobe_op_##OP(A, B, c);                               \
    uint32_t m = mmaprobe_getmode(D[0]);                                       \
    uint32_t d0 = (uint32_t)__builtin_amdgcn_readfirstlane((int)D[0]);         \
    bool same = true;                                                          \
    for (int i = 0; i < 8; ++i) same = same && (D[i] == d0);                  \
    unsigned bad = __builtin_amdgcn_ballot_w32(!same);                         \
    if (threadIdx.x == 0) {                                                    \
      out[3 * t] = d0; out[3 * t + 1] = (bad == 0u) ? 1u : 0u;                 \
      out[3 * t + 2] = m;                                                      \
    }                                                                          \
  }

// ---- host side ---------------------------------------------------------------
#ifndef MMAPROBE_DEVICE_ONLY
struct mmaprobe_entry {
  const char* op;
  const char* mode;
  const void* fn;
  int in_words;
  int out_words;
  int threads_per_record;  // 1 = one lane per record, 32 = one wave per record
};

#define MMAPROBE_ENTRY_SCALAR(OP, MNAME, MVAL, SETTER)                         \
  {#OP, #MNAME, (const void*)mmaprobe_##OP##__##MNAME, 3, 2, 1},
#define MMAPROBE_ENTRY_WMMA(OP, MNAME, MVAL, SETTER)                           \
  {#OP, #MNAME, (const void*)mmaprobe_##OP##__##MNAME, 17, 3, 32},

#define MMAPROBE_CHECK(x)                                                      \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_),    \
              __FILE__, __LINE__, #x);                                         \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

static int mmaprobe_main(int argc, char** argv, const mmaprobe_entry* ent,
                         size_t nent, const char* family) {
  if (argc >= 2 && !strcmp(argv[1], "info")) {
    hipDeviceProp_t p;
    MMAPROBE_CHECK(hipGetDeviceProperties(&p, 0));
    int rt = 0;
    MMAPROBE_CHECK(hipRuntimeGetVersion(&rt));
    printf("{\"family\": \"%s\", \"arch_id\": %d, \"device\": \"%s\", "
           "\"gcnArchName\": \"%s\", \"hip_runtime\": %d, \"entries\": [",
           family, MMAPROBE_ARCH_ID, p.name, p.gcnArchName, rt);
    for (size_t i = 0; i < nent; ++i)
      printf("%s{\"op\": \"%s\", \"mode\": \"%s\", \"in\": %d, \"out\": %d, "
             "\"tpr\": %d}", i ? ", " : "", ent[i].op, ent[i].mode,
             ent[i].in_words, ent[i].out_words, ent[i].threads_per_record);
    printf("]}\n");
    return 0;
  }
  if (argc != 6 || strcmp(argv[1], "run")) {
    fprintf(stderr, "usage: %s info | run <op> <mode> <in.bin> <out.bin>\n",
            argv[0]);
    return 1;
  }
  {
    hipDeviceProp_t p;
    MMAPROBE_CHECK(hipGetDeviceProperties(&p, 0));
    std::string want = "gfx" MMAPROBE_STR(MMAPROBE_ARCH_ID);
    if (std::string(p.gcnArchName).rfind(want, 0) != 0) {
      fprintf(stderr, "device 0 is %s, this binary was built for %s only\n",
              p.gcnArchName, want.c_str());
      return 3;
    }
  }
  const mmaprobe_entry* e = nullptr;
  for (size_t i = 0; i < nent; ++i)
    if (!strcmp(ent[i].op, argv[2]) && !strcmp(ent[i].mode, argv[3])) e = &ent[i];
  if (!e) { fprintf(stderr, "unknown op/mode %s/%s\n", argv[2], argv[3]); return 1; }
  FILE* f = fopen(argv[4], "rb");
  if (!f) { perror(argv[4]); return 1; }
  std::vector<uint32_t> in;
  uint32_t w;
  while (fread(&w, 4, 1, f) == 1) in.push_back(w);
  fclose(f);
  int n = (int)(in.size() / e->in_words);
  if (n == 0 || in.size() % e->in_words) { fprintf(stderr, "bad input size\n"); return 1; }
  std::vector<uint32_t> out((size_t)n * e->out_words, 0xDEADBEEFu);
  uint32_t *din, *dout;
  MMAPROBE_CHECK(hipMalloc(&din, in.size() * 4));
  MMAPROBE_CHECK(hipMalloc(&dout, out.size() * 4));
  MMAPROBE_CHECK(hipMemcpy(din, in.data(), in.size() * 4, hipMemcpyHostToDevice));
  MMAPROBE_CHECK(hipMemcpy(dout, out.data(), out.size() * 4, hipMemcpyHostToDevice));
  void* args[] = {&din, &dout, &n};
  dim3 block(e->threads_per_record == 1 ? 256 : 32);
  dim3 grid(e->threads_per_record == 1 ? (n + 255) / 256 : n);
  MMAPROBE_CHECK(hipLaunchKernel(e->fn, grid, block, args, 0, 0));
  MMAPROBE_CHECK(hipDeviceSynchronize());
  MMAPROBE_CHECK(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost));
  f = fopen(argv[5], "wb");
  if (!f) { perror(argv[5]); return 1; }
  fwrite(out.data(), 4, out.size(), f);
  fclose(f);
  MMAPROBE_CHECK(hipFree(din));
  MMAPROBE_CHECK(hipFree(dout));
  return 0;
}
#endif  // MMAPROBE_DEVICE_ONLY
