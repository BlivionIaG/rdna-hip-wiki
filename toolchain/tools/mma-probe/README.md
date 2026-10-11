---
title: "mma-probe: MMA-Sim-style numerics probes for RDNA2/RDNA3 DOT and WMMA"
owner: RDNA2_Researcher
updated: 2026-10-11
sources:
  - https://arxiv.org/abs/2511.10909
  - https://github.com/microsoft/MMA-Sim/tree/aab7f2ae9f06ffbee1e7e86cb21b708e4c6f9254
  - https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/docs/AMDGPUUsage.rst
  - https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/clang/docs/UsersManual.md
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---

# mma-probe

Date: **2026-10-11**. A standalone HIP kit plus a Python harness. It measures how the gfx1030 (V620) and gfx1100 (W7800 / 7900 XTX) dot and matrix instructions actually round, sum and flush. The method follows MMA-Sim's closed-loop feature probing (CLFP, arXiv 2511.10909 §III-A)[^mmasim-paper]. The findings and the reasons for them are in [silicon/arch-and-memory/dot-wmma-numerics.md](../../../silicon/arch-and-memory/dot-wmma-numerics.md). This page only covers how to build, run and read the kit.

**Status:** it compiles and the ISA is checked on the box (no GPU). **Nothing has been measured on hardware yet.** Every behaviour this kit reports is "to be measured" until someone runs it on a V620 and a gfx1100 card.

Folder note: this sits under `toolchain/tools/`, which ROCM_specialist owns. Their [README](../README.md) is not edited here.

## Files

| File | What it is |
|---|---|
| `probe_common.h` | Kernel and host scaffolding. Records are raw bit patterns in, `{d, MODE readback}` out. Each op is compiled once per MODE setting (`s_setreg_imm32_b32 hwreg(HW_REG_MODE,0,8)`), plus one `desc` kernel that leaves MODE as the kernel descriptor set it. |
| `dot2_f16.hip` | `v_dot2_f32_f16` (inline asm, VOP3P), `v_dot2c_f32_f16` (gfx1030) / `v_dot2acc_f32_f16` (gfx1100), `__builtin_amdgcn_fdot2` with and without clamp |
| `dot4_i8.hip` | gfx1030: `v_dot4_i32_i8` (± clamp), `v_dot4c_i32_i8`, `__builtin_amdgcn_sdot4`. gfx1100: `__builtin_amdgcn_sudot4` signed×signed (`v_dot4_i32_iu8`), because gfx11 has no `dot1-insts`[^td-features] |
| `fma_baseline.hip` | Baselines: `v_fma_mix_f32`, `v_pk_fma_f16`, `v_fma_f32` (asm); `__builtin_fmaf`; `a*b+c` with contraction off; `(float)h*(float)h+c` |
| `dot2_bf16.hip` | gfx1100 only: `v_dot2_f32_bf16` (asm and builtin) |
| `wmma.hip` | gfx1100 only: `v_wmma_f32_16x16x16_f16` and `_bf16`. One wave per record. Every row of A and every column of B is the same, so all 256 outputs must be equal (MMA-Sim Step 1). The result does not depend on the lane layout. |
| `fpmodel.py` | Exact (Fraction) reference models: exact fused, FMA chain, pairwise, E-FDPA, CDNA2 FTZ-Mul/Add, T-FDPA with F fractional bits. Each crossed with rounding mode and input/accumulator/output flush. |
| `harness.py` | Builds the test vectors, runs the binaries, infers the tree, precision, rounding, flushing and specials, and writes `results.md` and `results.json` |
| `build.sh` | Pinned-flag build. One `--offload-arch` per run, two denorm variants. Writes `.s`, the code object, `readelf --notes` output, the disassembly and `isa-check.txt` |
| `check_isa.py` | Checks each kernel contains exactly the intended FP/DOT/WMMA instruction and the right `s_setreg` immediate. Records the descriptor float fields from the `.s` and from the code object's `.kd` |
| `box-build/<arch>-isa-check.txt` | Evidence from a device-only build on the box (clang 19.1.7, no ROCm) |

## Build flags, and why

All of this is in `build.sh`. The exact flags are printed into `isa-check.txt`.

- `-O2 -fno-fast-math -fno-finite-math-only -fno-unsafe-math-optimizations`. Never fast-math: it changes descriptor denorm modes and lets the compiler rewrite FP.
- `-ffp-contract=off` for DOT, WMMA and bf16. `-ffp-contract=fast-honor-pragmas` for `fma_baseline`. Plain `fast` ignores `#pragma clang fp contract`, which `mul_add_c` depends on[^clang-contract]. On the box, plain `fast` turned `mul_add_c` into `v_fmac_f32`, so `check_isa.py` would have failed.
- Two variants of every family:
  - `default`: `-fno-gpu-flush-denormals-to-zero`
  - `ftz`: `-fgpu-flush-denormals-to-zero`[^clang-ftz]

  The descriptor fields `FLOAT_DENORM_MODE_32` (bits 17:16) and `FLOAT_DENORM_MODE_16_64` (19:18) set the wave's starting MODE[^amdgpuusage-rsrc1]. On the box (clang 19.1.7), `ftz` changed only `FLOAT_DENORM_MODE_32` from 3 to 0. `FLOAT_DENORM_MODE_16_64` stayed 3 in both variants (see `box-build/`). Comparing the two variants' `desc` rows separates "the descriptor flushes" from "the instruction flushes". The `m..` rows overwrite MODE at run time and show what the hardware does with each setting.
- `--offload-arch` is always exactly one of `gfx1030` and `gfx1100`. gfx1030 and gfx1100 objects are never built together.

## What the box build confirmed (no GPU)

`./build.sh --device-only gfx1030` and `gfx1100` with clang 19.1.7 → **`ISA CHECK PASS`** on both. That is 468 kernels for gfx1030 and 648 for gfx1100. Each contains exactly one intended instruction and no fallback. The disassembled code object agrees with the `.s`. The descriptor fields decoded from each `.kd` agree with the `.amdhsa_*` directives. Full listing: [box-build/gfx1030-isa-check.txt](box-build/gfx1030-isa-check.txt), [box-build/gfx1100-isa-check.txt](box-build/gfx1100-isa-check.txt).

| op | gfx1030 | gfx1100 |
|---|---|---|
| `dot2_f32_f16` | `v_dot2_f32_f16` | `v_dot2_f32_f16` |
| `dot2c_f32_f16` | `v_dot2c_f32_f16` | `v_dot2acc_f32_f16` |
| `fdot2_builtin` | `v_dot2c_f32_f16` | `v_dot2acc_f32_f16` |
| `fdot2_builtin_clamp` | `v_dot2_f32_f16 … clamp` | `v_dot2_f32_f16 … clamp` |
| `dot4_i32_i8` / `_clamp` / `dot4c_i32_i8` / `sdot4_builtin` | `v_dot4_i32_i8` / `… clamp` / `v_dot4c_i32_i8` / `v_dot4c_i32_i8` | n/a |
| `sudot4_ss_builtin` / `_clamp` | n/a | `v_dot4_i32_iu8 … neg_lo:[1,1,0]` / `… clamp` |
| `dot2_f32_bf16`, `fdot2_f32_bf16_builtin` | n/a | `v_dot2_f32_bf16` |
| `wmma_f32_16x16x16_f16` / `_bf16` | n/a | `v_wmma_f32_16x16x16_f16` / `_bf16` |
| `fma_mix_f32`, `fma_mix_c` | `v_fma_mix_f32 … op_sel_hi:[1,1,0]` | same |
| `fma_f32` / `fma_c` / `mul_add_c` / `pk_fma_f16` | `v_fma_f32` / `v_fmac_f32` / `v_mul_f32`+`v_add_f32` / `v_pk_fma_f16` | same |

ROCm's hipcc ships a different clang. **Run `build.sh` on each GPU box and read its own `isa-check.txt` before trusting results.** The harness warns if that file does not say `ISA CHECK PASS`.

## Run it on the V620 box (gfx1030)

```bash
git clone -b wiki/rdna-mma-probe https://github.com/BlivionIaG/rdna-hip-wiki.git mma-probe-wiki
cd mma-probe-wiki/toolchain/tools/mma-probe
python3 harness.py --selftest                       # model logic, no GPU; must print SELFTEST PASS
./build.sh gfx1030                                  # hipcc from /opt/rocm; HIPCC=... / ROCM_PATH=... to override
grep -E "^TOTAL" build/gfx1030/isa-check.txt        # must say ISA CHECK PASS
HIP_VISIBLE_DEVICES=0 python3 harness.py --arch gfx1030 --build build/gfx1030 --out results/gfx1030
```

## Run it on the W7800 / 7900 XTX box (gfx1100)

Both cards are gfx1100 in LLVM's processor table[^amdgpuusage-procs].

```bash
git clone -b wiki/rdna-mma-probe https://github.com/BlivionIaG/rdna-hip-wiki.git mma-probe-wiki
cd mma-probe-wiki/toolchain/tools/mma-probe
python3 harness.py --selftest
./build.sh gfx1100
grep -E "^TOTAL" build/gfx1100/isa-check.txt
HIP_VISIBLE_DEVICES=0 python3 harness.py --arch gfx1100 --build build/gfx1100 --out results/gfx1100
```

Notes:

- Each binary refuses to run if the device's `gcnArchName` does not start with the arch it was built for. On a mixed box, choose the card with `HIP_VISIBLE_DEVICES`. `build/<arch>/default/dot2_f16 info` prints the device it sees.
- `--ops dot2_f32_f16,wmma_f32_16x16x16_f16` limits the run. `--random N` (default 4000) and `--random-wmma N` (default 600) set the number of random differential tests. Use `--random 300 --random-wmma 60` for a quick first pass.
- Runtime has not been measured. The GPU time is small. Most of the time is Python evaluating exact-arithmetic candidate models. A fake 200-test run of 9 ops took about 20 s on the box.
- Send back `results/<arch>/results.md`, `results.json` and `build/<arch>/isa-check.txt`.

## Reading `results/<arch>/results.md`

The header repeats `toolchain.txt` (clang/hipcc and ROCm versions), the device name and the ISA-check verdict. Then there is one section per op.

| Line / column | Meaning |
|---|---|
| `deterministic`, `all positions identical` | MMA-Sim Step 1[^mmasim-paper]. Repeated launches give the same bits. For WMMA, all 256 outputs of a broadcast input are bitwise equal (the `uniform` flag comes from a wave ballot). A `False` here invalidates everything below. |
| `MODE at launch, build default/ftz` | `s_getreg` of MODE in the `desc` kernel, decoded per ISA §3.5 Table 4[^isa70648]. It shows what the descriptor actually gave the wave. |
| `summation tree` | Step 2 (FPRev-style). Inputs are `U=2^30` and `-U` at two positions with small `v` elsewhere. Possible results: "fused N-term sum, no intermediate rounding visible", a pairwise or sequential tree, or a block size L for WMMA. |
| `fused precision` | FPT1. How far below a cancelled `2^30` a term survives. `fused_bits_below_leading` ≈ F, the number of fractional bits kept. "probe limit" means at least that many. |
| `sticky / extra bits` | Whether bits shifted out below F still affect rounding (sticky). `24 / 24` means exact-then-round, as far as the probe can tell. |
| rounding `(c=U / p=U)`, `ties` | FPT2 with the large value in the accumulator or in a product. `RN*` is round-to-nearest; `ties` gives RNE, RNA and so on. `RZ`, `RU` and `RD` are reported as such. |
| `input sub` | Is a subnormal f16/bf16 input kept? `flushed` plus `flushed sign` (`flush_to_+0` like CDNA2[^mmasim-paper], or sign-keeping). |
| `acc sub` | Is a subnormal f32 `c` kept? |
| `output sub` | Is a subnormal f32 result kept? For f16 dot2 this cannot be separated from `acc sub`, because the smallest normal f16 product is 2^-28. bf16 can separate them. |
| the MODE rows | `desc` is the descriptor's mode. `m<D><R>` means MODE byte `DENORM<<4 | ROUND` was written with `s_setreg`. If every row is identical, the instruction ignores MODE. If the rows follow the DENORM bits, the documented MODE semantics apply[^isa70648]. `ftz:desc` is the `-fgpu-flush-denormals-to-zero` build. |
| `models matching all N random tests` | Step 4. Candidate models from `fpmodel.py` that reproduce every random output bit for bit, under `desc`, `mf0` (allow everything) and `m00` (flush everything). The name encodes `base[rho, in16, c32, out]`. **NONE** means the hardware does something none of the candidates do. Read the sample failures in `results.json` and extend `fpmodel.py` (Step 5). |
| `Specials` | Inf/NaN propagation and the NaN encoding returned |
| `odd symmetry violations` | `d(-a,b,-c) == -d(a,b,c)`. Violations point to directed rounding or asymmetric truncation (MMA-Sim reports such asymmetry for CDNA3[^mmasim-paper]). |
| `fused vs unfused` (fma ops) | A single-rounding test value. Tells `v_fma_*` from mul+add. |
| `arith` (integer dot) | "exact, wraps mod 2^32", "exact, saturating on i32 overflow", or neither. Overflow edge cases decide between wrap and saturate. |

Known blind spots:

- An f16×f16 or bf16×bf16 product is exact in f32, so for dot2/fma_mix the question "is the product rounded before the add" only shows up as summation precision.
- The WMMA probe reads lane 0 plus a uniformity flag, not the full layout.
- NaN payloads beyond the returned encoding are not studied.

## Self-test

`python3 harness.py --selftest` runs the whole inference against fake hardware made from known models. These are exact, FMA chain with input FTZ, `tfdpa_L16_F24` RZ for WMMA and CDNA2 `ftz_muladd_P4` for WMMA. It checks that the harness recovers each one. The test is about the harness logic only. It says nothing about the hardware.

## Sources

[^mmasim-paper]: Xie, Xu, Wang et al. (Microsoft Research), "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units: Demystifying Numerical Discrepancy and Accuracy" (MMA-Sim), arXiv 2511.10909v3 (2026-09-28): §III-A CLFP workflow (Steps 1–5, FPT1–4), §IV-A Alg. 1–2 (CDNA2 FTZ-Mul/FTZ-Add, input subnormals flushed to +0), Table VI. https://arxiv.org/abs/2511.10909 (checked 2026-10-11)
[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30, §3.5 Table 4 (MODE FP_ROUND / FP_DENORM) and §6.4 Table 24. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-11)
[^amdgpuusage-rsrc1]: LLVM AMDGPUUsage at `f3653503`, compute_pgm_rsrc1 FLOAT_ROUND_MODE_32 13:12, FLOAT_DENORM_MODE_32 17:16, FLOAT_DENORM_MODE_16_64 19:18 ("Wavefront starts execution with specified … mode"); denorm enum 0–3; "Denormal flushing is sign respecting". https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/docs/AMDGPUUsage.rst#L6477-L6569 (checked 2026-10-11)
[^amdgpuusage-procs]: LLVM AMDGPUUsage processor table: gfx1030 lists Radeon PRO V620; gfx1100 lists Radeon PRO W7800 and Radeon RX 7900 XTX. https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/docs/AMDGPUUsage.rst#L526-L574 (checked 2026-10-11)
[^td-features]: LLVM AMDGPU.td at `f3653503`: Dot1 (v_dot4_i32_i8) etc. at L888–L934; FeatureISAVersion10_3_0 (L2278) vs FeatureISAVersion11_Common (L2298) / 11_0_Common (L2343). https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/lib/Target/AMDGPU/AMDGPU.td#L888-L934 (checked 2026-10-11)
[^clang-contract]: Clang Users Manual at `f3653503`, `-ffp-contract`: "pragmas will be ignored with -ffp-contract=fast"; `fast-honor-pragmas` is the HIP default. https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/clang/docs/UsersManual.md (checked 2026-10-11)
[^clang-ftz]: Clang Users Manual at `f3653503`: `-fgpu-flush-denormals-to-zero` "Flush denormal floating point values to zero in CUDA/HIP device mode." https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/clang/docs/UsersManual.md (checked 2026-10-11)
