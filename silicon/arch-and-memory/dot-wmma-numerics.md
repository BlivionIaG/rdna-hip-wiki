---
title: "DOT and WMMA numerics on gfx1030 / gfx1100: what is documented, what the MMA-Sim method can measure"
owner: RDNA2_Researcher
updated: 2026-10-11
sources:
  - https://arxiv.org/abs/2511.10909
  - https://github.com/microsoft/MMA-Sim/tree/aab7f2ae9f06ffbee1e7e86cb21b708e4c6f9254
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/shared/machine-readable-isa/isa/amdgpu_isa_rdna3.xml
  - https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/docs/AMDGPUUsage.rst
  - https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/lib/Target/AMDGPU/AMDGPU.td
  - https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/lib/Target/AMDGPU/GCNSubtarget.h
  - https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/lib/Target/AMDGPU/SIISelLowering.cpp
  - https://github.com/llvm/llvm-project/pull/205101
  - https://github.com/SemiAnalysisAI/FuzzX/blob/ee2d7ca879b541e76c824fb8bb19e1eb7e2c41ab/amdgpu/known-miscompiles/m100-performfmacombine-fdot2-ignores-denormal-mode/NOTES.md
  - https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/clang/docs/UsersManual.md
  - https://github.com/opengfx1030/vllm-rdna/pull/40
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC8225237/
---

# DOT and WMMA numerics on gfx1030 / gfx1100

Date: **2026-10-11**. Up: [README.md](README.md). Instruction sheet: [valu.md](valu.md). Probe kit: [toolchain/tools/mma-probe/](../../toolchain/tools/mma-probe/README.md).

**Bottom line.** AMD's ISA guides give the formulas for `V_DOT2*`, `V_DOT4*` and `V_WMMA_*`. They say nothing about summation order, internal precision, rounding or subnormal handling for these instructions. The LLVM backend claims, from hardware testing, that only gfx90a's dot2 flushes f16 subnormal inputs. It does not name gfx1030 or gfx1100, and an older LLVM comment and the PR's own first description say the opposite about f32. So for gfx1030 and gfx1100 **everything about dot2/WMMA rounding and flushing is still to be measured**. The kit in `toolchain/tools/mma-probe/` measures it with the MMA-Sim method. Until it has run on a V620 and a gfx1100 card, every RDNA behaviour on this page is labelled documented, hypothesis or *(unsourced)*.

## 1. The MMA-Sim method (CLFP)

MMA-Sim (Microsoft, arXiv 2511.10909, v3 2026-09-28, MIT) builds bit-exact models of GPU matrix instructions by *closed-loop feature probing*. The workflow is in §III-A[^mmasim-paper]:

| Step | What it does | How it decides |
|---|---|---|
| 1. Determinism and cross-position consistency | Make every row of A identical and every column of B identical, so every `d_ij` should be the same. Repeat runs. | Bitwise-equal outputs mean every element shares one arithmetic path, so a single dot-product-accumulate model is enough.[^mmasim-paper] |
| 2. Summation order and arity | FPRev-style. Put a large `U` at product `k` and `-U` at product `j`, with small `v` everywhere else. `d/v` counts the summands that were not swamped. | The swamped set reveals the tree. A high-precision fused sum (Eq. 9) is told apart from low-precision chained adds (Eq. 8).[^mmasim-paper] |
| 3. Feature-probing tests | FPT1 summation precision: `Add(U, ε)` and `FusedSum(U, −U, ε)`, halving ε; `F = e_U − log2 ε − 1`. FPT2 rounding: `U ± 1.5ε`, `U ± 0.5ε`, ties at `U ± ε`, `U ± 3ε`, giving RNE/RNA/RZ/RU/RD/RNO. FPT3 internal order. FPT4 product renormalisation. | Each edge-case output maps to one feature.[^mmasim-paper] |
| 4. Randomised differential test | Random bit streams, normal values and large-condition-number inputs, run against the candidate model | Any bit mismatch is a model bug.[^mmasim-paper] |
| 5. Minimum failing subset | Shrink the mismatch into a new FPT and go back to Step 3 | Repeat until Step 4 is clean (Table I: 2–12 h per instruction).[^mmasim-paper] |

The repository at `aab7f2ae` (2026-09-02) holds the models and kernels:
- `src/mmasim/mmasim/amd/sim.py` maps CDNA1 to `MMA_E_FDPA`, CDNA2 to `MMA_FTZ_MUL_ADD(P)` and CDNA3 to `MMA_TR_FDPA`.
- `src/mmasim/mmasim/arithmetic/ftz_mul_add.py` has `flush_subnormal`, `ftz_mul`, `ftz_add` and `MMA_FTZ_MUL_ADD`. `arithmetic/fdpa.py` has the FDPA family.
- `src/mmasim_kernels/mmasim_kernels/amd/kernels/` has `mfma_cdna{1,2,3}.hip`, and the `Makefile` builds with one `--offload-arch` each: gfx908, gfx90a and gfx942.[^mmasim-repo]

The paper's appendix names `tests/probe/clfp_example_step<1-4>.py`, but those files are **not** in the repository at that commit. The probe logic in our kit is therefore a re-implementation from the paper text, not a port.[^mmasim-repo]

**RDNA is not covered.** The paper models MI100, MI250X and MI300X (gfx908, gfx90a, gfx942) and NVIDIA tensor cores.[^mmasim-paper]

## 2. CDNA results we use as reference points (not RDNA)

| GPU | Model | Key facts |
|---|---|---|
| CDNA2 (gfx90a) FP16/BF16 MFMA | Φ_FTZ-MulAdd (Alg. 1–2) | Input subnormals are flushed to **+0**. Subnormal outputs of each mul/add are flushed with the sign kept. Groups of L ∈ {2, 4} products are summed pairwise, then `c` and the partial sums are added in sequence. Each step rounds to f32 RNE.[^mmasim-paper][^mmasim-repo] |
| CDNA2 consequence | Table IV, Table VI, §VI-B1 | For the paper's Eq. 10 input, CDNA2 FP16 returns 0.0. Input FTZ costs up to 2^-14 per FP16 input. That is the "CDNA2 FP16 input FTZ" concern. §VI-C Workaround 1 scales the inputs out of the subnormal range.[^mmasim-paper] |
| CDNA1 | E-FDPA | Exact fused sum, RNE.[^mmasim-paper] |
| CDNA3 | TR-FDPA, F = 24 | Truncation inside the block, RD internally, so the result is asymmetric. FP8 uses GTR-FDPA.[^mmasim-paper] |
| NVIDIA (for scale) | T-FDPA | RZ, F = 23/24/25 depending on generation; FP8 F = 13.[^mmasim-paper] |

Table V gives per-operation error bounds: input FTZ up to 2^-126 or 2^-14, add/mul 0.5 ulp, output FTZ 2^-126, fused summation (L+1)·2^(emax−F), output rounding 0.5 or 1 ulp.[^mmasim-paper]

## 3. What primary sources say about RDNA2 / RDNA3 (documented)

| # | Claim | Source |
|---|---|---|
| D1 | `V_DOT2C_F32_F16` (VOP2 op 2): `D.f32 = S0.f16[0]*S1.f16[0] + S0.f16[1]*S1.f16[1] + D.f32`. `V_DOT2_F32_F16` (VOP3P op 19) uses the same formula with `S2.f32`. `V_DOT4_I32_I8` (VOP3P op 22) is the signed-byte equivalent. The descriptions give **no** precision, order, rounding or denorm statement. Plain VALU ops such as `V_ADD_F32` *do* state "0.5ULP precision, denormals supported". | RDNA2 ISA 70648, pp. 131, 164, 165[^isa70648] |
| D2 | "Rounding and Denormal modes apply to all floating-point operations unless otherwise specified in the instruction description." | ISA 70648 ch. 12 intro, p. 101[^isa70648] |
| D3 | MODE register: FP_ROUND [3:0] ([1:0] single, [3:2] double/half; 0 = RNE, 1 = +inf, 2 = −inf, 3 = RZ). FP_DENORM [7:4] ([1:0] single, [3:2] double/half; 0 = flush in and out, 1 = allow in / flush out, 2 = flush in / allow out, 3 = allow both). Settable with `S_SETREG`, `S_ROUND_MODE` and `S_DENORM_MODE`. | ISA 70648 §3.5 Table 4 (p. 15), §6.4 Table 24 (p. 47)[^isa70648] |
| D4 | The kernel descriptor's `COMPUTE_PGM_RSRC1` sets the starting MODE: FLOAT_ROUND_MODE_32 13:12, FLOAT_DENORM_MODE_32 17:16, FLOAT_DENORM_MODE_16_64 19:18, ENABLE_DX10_CLAMP 21, ENABLE_IEEE_MODE 23. Flushing done this way is "sign respecting". | LLVM AMDGPUUsage[^amdgpuusage] |
| D5 | RDNA3 `V_WMMA_F32_16X16X16_F16` / `_BF16`: "Multiply the 16x16 matrix in the first input by the 16x16 matrix in the second input and add the 16x16 matrix in the third input using fused multiply add." `V_DOT2_F32_F16` / `_BF16`: dot product "in the single-precision float domain". There is no pseudocode and no order, width, rounding or denorm statement. | AMD machine-readable ISA XML (RDNA3)[^xml-rdna3] |
| D6 | Instruction availability: gfx1030 (FeatureISAVersion10_3_0) has Dot1, 2, 5, 6, 7, 10, so `v_dot4_i32_i8`, `v_dot2c_f32_f16`, `v_dot4c_i32_i8` and `v_dot2_f32_f16`, but no Dot12 (`v_dot2_f32_bf16`) and no WMMA. gfx1100 has Dot5/7/8/9/10/12 and WMMA256bInsts, but no Dot1. | LLVM AMDGPU.td[^td] |
| D7 | The LLVM backend says: "v_dot2c_f32_f16 unconditionally flushes f16 subnormal inputs to zero regardless of the MODE register" **on gfx90a only** (`dot2UnconditionalFlush`), and "All other GPUs: v_dot2 does NOT flush f16 subnormal inputs." The fma_mix→dot2 fold is allowed only when the f32 denorm mode is IEEE (PreserveSign on gfx90a). | `GCNSubtarget.h` L585–589, `SIISelLowering.cpp` L18458–18486[^llvm-dot2] |
| D8 | That change (PR #205101, merged 2026-08-14) rests on "hardware testing across multiple GPU generations". **The GPUs are not named.** Earlier, the PR description said dot2 flushes the *f32 accumulator* input and output, and an older source comment said "fdot2_f32_f16 always flushes fp32 denormal operand and output to zero". These contradict each other. | PR #205101[^llvm-pr], FuzzX note quoting the older comment[^fuzzx] |
| D9 | `-fgpu-flush-denormals-to-zero` flushes device denormals. `-ffp-contract=fast` ignores `#pragma clang fp contract`, and `fast-honor-pragmas` is the HIP default. | Clang Users Manual[^clang] |
| D10 | Box observation (compiler only, no GPU), clang 19.1.7. HIP defaults give `.amdhsa_float_denorm_mode_32 3`, `_16_64 3`, `ieee_mode 1` and `dx10_clamp 1`. `-fgpu-flush-denormals-to-zero` changes only `_32` (to 0); `_16_64` stays 3. `__builtin_amdgcn_fdot2` lowers to `v_dot2c_f32_f16` on gfx1030 and `v_dot2acc_f32_f16` on gfx1100. | [box-build/](../../toolchain/tools/mma-probe/box-build/)[^kit] |

Consistency with existing pages: [valu.md](valu.md) maps `fdot2` to `V_DOT2C_F32_F16`, which D10 matches on gfx1030. It also says gfx1030 has no bf16 DOT and no WMMA, which matches D6. [petit-kernel.md](../external-projects/petit-kernel.md) already says CDNA2 flush behaviour must be measured on V620, not assumed. No `silicon/` page on main (`c54a72c`) states how gfx1030/gfx1100 dot2 or WMMA round or flush.

## 4. What the docs do not settle (to be measured)

Each row is a hypothesis until the kit has run on hardware. The kit result that decides it is named.

| # | Open question | Hypotheses | Decided by (kit row) |
|---|---|---|---|
| O1 | gfx1030 `v_dot2(c)_f32_f16`: is the sum `p0 + p1 + c` fused with one rounding, or chained? | Fused/exact (H-a) vs `fma(p1, fma(p0, c))` (H-b). Both are possible: the products are exact in f32 either way, so only the order of adds differs. | `summation tree`, `fused precision`, `sticky` |
| O2 | Does dot2 follow MODE.FP_ROUND? | D2 says yes unless stated otherwise. Not verified for DOT. | rounding column across the `mf1/mf2/mf3` rows |
| O3 | f16 input subnormals on gfx1030 / gfx1100 dot2 | LLVM D7 says not flushed. gfx1030 is not named. Does MODE.FP_DENORM[3:2] change it? | `input sub` across the `m..` rows |
| O4 | f32 accumulator and output subnormals for dot2 | D8 has contradictory claims. Does MODE.FP_DENORM[1:0] apply? | `acc sub`, `output sub`, `ftz:desc` vs `desc` |
| O5 | `v_fma_mix_f32` f16 input subnormals: governed by the f32 or the f16 denorm field? | The LLVM comment says it flushes "when f32 denorm = PreserveSign" *(LLVM comment, not AMD doc)*. | `fma_mix_f32` `input sub` across the `m..` rows |
| O6 | gfx1100 WMMA f16/bf16: block size L, F, internal rounding, input/output FTZ | "Fused multiply add" (D5) does not fix any of these. CDNA2-like, CDNA3-like and exact are all candidates. | WMMA `summation tree`, `fused precision`, `models matching` |
| O7 | Does `v_dot2_f32_bf16` behave like the bf16 WMMA? | Unknown | compare the two sections |
| O8 | NaN encoding, clamp on NaN, DX10 clamp | Unknown | `Specials`, the clamp ops |
| O9 | Integer dot overflow (`clamp` off vs on) | Wraps without clamp, saturates with clamp *(unsourced)* | `arith` |
| O10 | Does ROCm's hipcc emit the same instructions as box clang 19? | Unknown | `build/<arch>/isa-check.txt` on each GPU box |

## 5. The kit

[`toolchain/tools/mma-probe/`](../../toolchain/tools/mma-probe/README.md) contains:
- one `.hip` per instruction family;
- `fpmodel.py`, exact reference models with the MMA-Sim model classes (E-FDPA, FTZ-MulAdd, T-FDPA) plus FMA chains, crossed with rounding and flush options;
- `harness.py`, which implements Steps 1–4 and writes a verdict table.

Every op is built once per MODE byte (16 denorm combinations × RNE, plus RU/RD/RZ) and once more under the descriptor's own MODE. Each family is built twice: default, and `-fgpu-flush-denormals-to-zero`. Comparing the two separates what the descriptor does from what the hardware does.

On the box, both arches give `ISA CHECK PASS`. Every kernel contains exactly the intended instruction, with no fallback.[^kit]

| Instruction under test | gfx1030 | gfx1100 |
|---|---|---|
| `v_dot2_f32_f16` (VOP3P) | confirmed | confirmed |
| `v_dot2c_f32_f16` / gfx11 `v_dot2acc_f32_f16` | confirmed | confirmed (`v_dot2acc`) |
| `v_dot4_i32_i8`, `v_dot4c_i32_i8`, `sdot4` | confirmed | not on gfx11 (D6) → `v_dot4_i32_iu8 neg_lo:[1,1,0]` confirmed |
| `v_fma_mix_f32`, `v_pk_fma_f16`, `v_fma_f32` | confirmed | confirmed |
| `v_wmma_f32_16x16x16_f16` / `_bf16` | n/a (D6) | confirmed |
| `v_dot2_f32_bf16` | n/a (D6) | confirmed |

The exact commands for the V620 box and the W7800/7900 XTX box are in the [kit README](../../toolchain/tools/mma-probe/README.md#run-it-on-the-v620-box-gfx1030).

## 6. Why this matters for hetero MoE (vllm-rdna PR #40)

PR #40 is "feat(rdna): default-off hetero MoE for gfx1100 hot experts and gfx1030 cold experts". It is OPEN at head `4144f2c6`.[^pr40]
- Hot experts run on the gfx1100 tier under the fast W4A16 kernel name `triton_wna16`.
- Cold experts run on the gfx1030 cards with `moe_gptq_gemm_rdna2` (`fdot2`).
- The combine accumulates in float64 (`combine.py`).[^pr40]

So a single token's MoE output is a weighted sum of expert outputs from **two different arithmetic paths**:
- gfx1030 `v_dot2c_f32_f16` chains (O1–O4);
- whatever the gfx1100 Triton kernel lowers to. Whether `triton_wna16` uses `v_wmma_*` on gfx1100 at all is *(unsourced)*: the PR names the kernel but no ISA dump shows WMMA. Dump its ISA before applying the WMMA rows (O6).

These paths do not have to agree bit for bit. If dot2 on one tier flushes f16 subnormals (as CDNA2 MFMA does[^mmasim-paper]) and the other does not, small activations times small dequantised weights will diverge systematically, not randomly. A test therefore needs a tolerance that comes from the measured arithmetic, not a guessed `atol`.

The wiki's placement page records that the W7800 fast tier was retipped "dead" on 2026-08-27 in favour of Spark + 8×V620 ([hetero-moe-w7800-v620.md](../interconnect/hetero-moe-w7800-v620.md)), while PR #40 targets gfx1100. The tolerance method below does not depend on which fast tier is used. Only the γ of the fast tier changes (NVIDIA T-FDPA parameters are in MMA-Sim Table III[^mmasim-paper]).

## 7. Tolerance method (proposal)

**Routing rule.** This comes from the LLM_Arch_specialist brief for this task, not from a primary source:
- The router runs **once**, on the fast tier, in fp32.
- The V620s receive the expert ids and gate weights exactly as sent and **do not recompute top-k**. This keeps a near-tie in the router from turning into different expert sets on the two tiers.
- Tests compare **each expert's output before the weighted combine** against the probe-derived tolerance.
- They then compare the combined layer output against the combined tolerance, and finally the greedy tokens end to end.
- **Never compare bit for bit after the combine.**

**Per-expert bound (proposal, derived; constants come from measurement).** Take each GEMM output element `y = Σ_k w_k x_k` computed on tier T with K terms. The standard forward-error bound for a recursively summed inner product is `|ŷ − y| ≤ γ_n Σ|w_k x_k|` with `γ_n = n·u/(1 − n·u)`, where n is the number of roundings on the longest path[^inner-product]. For the two tiers:

```
tau_elem = (gamma_A + gamma_B) * S  +  F_A + F_B  +  u_out * |y_ref|
  S       = sum_k |w_k x_k|            (computed once in fp64 from the dequantised weights)
  gamma_T = n_T * u_T / (1 - n_T * u_T)
  F_T     = flush term: sum over flushed inputs of |other operand| * (flush threshold)
            (0 if the probe says "preserved"; MMA-Sim Table V gives 2^-14 for FP16 input FTZ)
  u_out   = unit roundoff of the expert's output dtype (fp16: 2^-11, bf16: 2^-8)
```

How `n_T` and `u_T` come from the kit's verdicts:

| Probe verdict for tier T | n_T (K = reduction length) | u_T |
|---|---|---|
| dot2 fused per step, RNE (O1 H-a) | K/2 | 2^-24 |
| dot2 chained FMA (O1 H-b) | K | 2^-24 |
| Rounding RZ/RD instead of RNE | same n | 2^-23 (one full ulp)[^mmasim-paper] |
| WMMA T-FDPA with block L, F bits | K/L blocks, plus a (L+1)·2^(emax−F) term per block instead of u·S (Table V)[^mmasim-paper] | — |

Apply this per GEMM in the expert: gate/up, then down. Push the gate/up bound through the activation using its derivative evaluated on the fp64 reference, not an assumed constant. That gives `tau_e`, a per-element vector for expert e.

**Combined layer.** With gate weights `g_e` taken from the router as sent, the combined bound is `Σ_e g_e · tau_e`, plus fp64-combine rounding, which is negligible. The layer-output check is `|y_A − y_B| ≤ Σ_e g_e tau_e` element-wise. It does not use a global `atol`.

**Two thresholds.**
- **fail**: the deterministic bound above.
- **warn**: a probabilistic bound with γ proportional to √n·u. It holds with high probability when the rounding errors are independent and zero-mean[^inner-product]. It catches regressions far inside the hard bound. The independence assumption fails for directed rounding and for systematic flushing, so do not use the warn threshold on a tier where the probe reports RZ/RD or flushed inputs.

**End to end.** Greedy tokens must match for N steps. If they diverge, find the first token whose top-2 logit margin at that step is above the propagated logit bound. A divergence there is a bug. A divergence inside the margin is an expected near tie, so log it.

**What this needs from the kit.** For each tier: O1 (tree), O2 (rounding), O3/O4 (input and accumulator flush), and O6 if the gfx1100 path really is WMMA. Until those are measured, use the most pessimistic row of the table (n = K, u = 2^-23, input flush on) and record the test as provisional.

## Sources

[^mmasim-paper]: Peichen Xie, Shuotao Xu, Yang Wang et al. (Microsoft Research), "Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate Units: Demystifying Numerical Discrepancy and Accuracy" (MMA-Sim), arXiv 2511.10909v3 (2026-09-28; v1 2025-11-14). §III-A CLFP workflow (Steps 1–5, FPT1–4, Table I); §IV-A Alg. 1–2 (CDNA2 FTZ-Add/FTZ-Mul, "subnormal MMA inputs are flushed to positive zero"); Tables II–VI; §VI-B1 input FTZ; §VI-C Workaround 1. https://arxiv.org/abs/2511.10909 (checked 2026-10-11)
[^mmasim-repo]: microsoft/MMA-Sim at `aab7f2ae9f06ffbee1e7e86cb21b708e4c6f9254` (2026-09-02): `src/mmasim/mmasim/amd/sim.py` L15–30, `src/mmasim/mmasim/arithmetic/ftz_mul_add.py`, `arithmetic/fdpa.py`, `src/mmasim_kernels/mmasim_kernels/amd/kernels/{mfma_cdna1,2,3}.hip` and `Makefile` L15–21; `tests/probe/` absent at this commit. https://github.com/microsoft/MMA-Sim/tree/aab7f2ae9f06ffbee1e7e86cb21b708e4c6f9254 (checked 2026-10-11)
[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30: §3.5 Table 4 p. 15; §6.4 Table 24 p. 47; ch. 12 note p. 101; V_DOT2C_F32_F16 p. 131; V_DOT2_F32_F16 p. 164; V_DOT4_I32_I8 p. 165. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-11)
[^xml-rdna3]: AMD machine-readable ISA, `amdgpu_isa_rdna3.xml` (ROCm/rocm-systems `65575fbb`), entries V_WMMA_F32_16X16X16_F16, V_WMMA_F32_16X16X16_BF16, V_DOT2_F32_F16, V_DOT2_F32_BF16, V_DOT2ACC_F32_F16. The RDNA3 ISA PDF could not be downloaded from amd.com on 2026-10-11, so the XML is the AMD source used. https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/shared/machine-readable-isa/isa/amdgpu_isa_rdna3.xml (checked 2026-10-11)
[^amdgpuusage]: LLVM AMDGPUUsage at `f3653503`: compute_pgm_rsrc1 L6477–6569; denorm enum and "Denormal flushing is sign respecting" L7044–7056; `.amdhsa_float_*` defaults L21916–21935. https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/docs/AMDGPUUsage.rst#L6477-L6569 (checked 2026-10-11)
[^td]: LLVM AMDGPU.td at `f3653503`: Dot feature definitions L888–934, WMMA256bInsts L994, FeatureISAVersion10_3_0 L2278, FeatureISAVersion11_Common L2298, FeatureISAVersion11_0_Common L2343. https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/lib/Target/AMDGPU/AMDGPU.td#L888-L934 (checked 2026-10-11)
[^llvm-dot2]: LLVM at `f3653503`: `GCNSubtarget.h` L585–589 (`dot2UnconditionalFlush`), `SIISelLowering.cpp` L18458–18486 (performFMACombine fdot2 fold). https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/llvm/lib/Target/AMDGPU/SIISelLowering.cpp#L18458-L18486 (checked 2026-10-11)
[^llvm-pr]: llvm/llvm-project PR #205101 (merged 2026-08-14 as `ff5e7802`): "gfx90a (CDNA2) is the sole outlier … All other tested GPUs with dot2 instruction do not flush f16 subnormal inputs." https://github.com/llvm/llvm-project/pull/205101 (checked 2026-10-11)
[^fuzzx]: SemiAnalysisAI/FuzzX at `ee2d7ca8`, `amdgpu/known-miscompiles/m100-performfmacombine-fdot2-ignores-denormal-mode/NOTES.md`, quoting the earlier LLVM comment. https://github.com/SemiAnalysisAI/FuzzX/blob/ee2d7ca879b541e76c824fb8bb19e1eb7e2c41ab/amdgpu/known-miscompiles/m100-performfmacombine-fdot2-ignores-denormal-mode/NOTES.md (checked 2026-10-11)
[^clang]: Clang Users Manual at `f3653503`: `-ffp-contract` values and pragma interaction; `-fgpu-flush-denormals-to-zero`. https://github.com/llvm/llvm-project/blob/f3653503a4ab638cd50b4b120e57f58bc50eab22/clang/docs/UsersManual.md (checked 2026-10-11)
[^kit]: This branch: `toolchain/tools/mma-probe/box-build/gfx1030-isa-check.txt` and `gfx1100-isa-check.txt`, device-only build with clang 19.1.7 on 2026-10-11. https://github.com/BlivionIaG/rdna-hip-wiki/tree/wiki/rdna-mma-probe/toolchain/tools/mma-probe/box-build (checked 2026-10-11)
[^pr40]: opengfx1030/vllm-rdna PR #40 at `4144f2c642353e4bfb773c16fe684dcdfafd164c`: `vllm/distributed/hetero_moe/kernels.py` (`FAST_W4A16_KERNEL = "triton_wna16"`, `COLD_W4A16_KERNEL = "moe_gptq_gemm_rdna2"`), `combine.py` (float64 accumulate). https://github.com/opengfx1030/vllm-rdna/pull/40 (checked 2026-10-11)
[^inner-product]: Ipsen & Zhou, "Probabilistic error analysis for inner products", SIAM J. Matrix Anal. Appl. 41(4), 2020, 1726–1741 (PMC8225237): Theorem 3.2, the deterministic bound γ_n·|x|ᵀ|y| (Higham 2002 §3.1); Theorem 3.3, a probabilistic bound ∝ √n·u for independent zero-mean errors. https://pmc.ncbi.nlm.nih.gov/articles/PMC8225237/ (checked 2026-10-11)
