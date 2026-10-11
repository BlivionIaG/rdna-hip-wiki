# Fastest FP16 path on gfx1030 (RDNA2)

Date: 2026-08-18. Engine study. Occupancy on `fa_rdna2` + `skinny_gemms.cu` still first. Do not invent tok/s.

Silicon contract: [silicon/fp16-rdna2.md](../../silicon/fp16-rdna2.md) (peak numbers: [silicon/valu.md](../../silicon/valu.md)). MoE: [fp16-moe.md](../moe/fp16-moe.md) + [kernels/fp16-moe.md](../../kernels/fp16-moe.md). Baselines: [baseline-order.md](../stock-baseline/baseline-order.md), [kernels/triton-skinny-gemm.md](../../kernels/triton-skinny-gemm.md). Attention: [attention-dispatch.md](../attention/attention-dispatch.md).

## The only inner op

```text
acc_f32 = fdot2(a_f16x2, b_f16x2, acc_f32)   // prefer V_DOT2C_F32_F16
```

Ceiling is still **256 FP16 FLOP/clk/CU** *(derived: 2 SIMD32 × 32 lanes × DOT2 = 4 FLOP per lane; the ISA guide gives no rates — see [silicon/valu.md](../../silicon/valu.md))*. `V_DOT2C_F32_F16` VOP2 form: ISA guide[^rdna2-isa]. No WMMA, no MFMA, no AGPR, no bf16 dot/ALU (bf16 goes through f32 conversion)[^llvm-amdgpu-td]. Prefer the **C** form (VOP2, dest=accum). `V_PK_FMA_F16` can also hit 256 but accumulates **fp16** — epilogue only, never the GEMM/attention reduction. Scalar FMA after unpack is bring-up, not the roofline.

HIP: `__builtin_amdgcn_fdot2`, K even, `half2`-aligned. hipcc will **not** peephole `__hfma2` into DOT. ISA-dump it.

## Split by phase (do not one-tile it)

| Phase | Bound | Fastest honest path | Stock hole |
|---|---|---|---|
| **Decode** skinny `M∈{1,2,4,8}` | bandwidth + launch | Custom HIP GEMV, stream B, A in LDS, `fdot2`, specialize M | `F.linear` / rocBLAS. Skinny HIP gated off (`on_gfx9() or on_gfx1x()`). |
| **Messy** `8<M<32` | fill | `M=8` specialized first; 32×32 only if CUs starve | Do not launch 64×64 at M=1 |
| **Prefill** fat `M≥32` | can be compute | **rocBLAS first.** HIP `64×64×32` `fdot2` only if it wins | Keep BLAS if the tile loses |
| **FA** D=128/256 | mixed | `fa_rdna2` already `fdot2` on QK — **occupancy flip only**, not a new FA | Triton `TRITON_ATTN` is the A/B |
| **MLA prefill** | both sides already half | **First attention `fdot2` rewrite**, after `load_row` OOB | Current HIP prefill is scalar FMA |
| **MLA decode** | unpack then math | Stay scalar until E4M3 unpack is gone | `fdot2` later |
| **FP16 MoE** | many tiny GEMMs | Same two-kernel contract. [fp16-moe.md](../moe/fp16-moe.md) | Triton MoE excludes gfx10xx |

## Occupancy is part of the path

`amdgpu_waves_per_eu(1,1)` / HIP second `__launch_bounds__` arg is a **min-waves** trap. Decode-class: drop min-blocks, `waves_per_eu(4, 8)` only if VGPR ≤64 and no scratch. A perfect `fdot2` loop at 1 wave/EU is not the fastest path.

## Engine rules that keep FP16 fast

1. **Continuous batching + chunked prefill** so decode stays skinny and prefill gets fat tiles.
2. **`--dtype half`.** gfx1030 has no bf16 dot/ALU, so bf16 runs through f32 conversion[^llvm-amdgpu-td][^vllm-issue38107].
3. **No Triton on the hot decode path** once HIP exists (compile tax).
4. **Collectives stay outside GEMM.** `0c59068e` is transport only.
5. **W7800 is a different ISA.** [multi-tier.md](../moe/multi-tier.md).
6. **V340L/gfx900 is not this path.** `mad_mix`/`pk_fma`, no DOT.

## Study order (after current occupancy card)

Matches silicon proof order + [baseline-order.md](../stock-baseline/baseline-order.md).

1. Occupancy flip — decode kernels landed at `1dc3bac4` (`amdgpu_waves_per_eu(4, 8)`); prefill (`__launch_bounds__(N, 1)`) and `skinny_gemms.cu` `wvSplitKrc_` (`(1, 1)`) are what is left[^dest-occupancy].
2. Microbench `fdot2` vs `__hfma2` vs scalar (must show `v_dot2c`).
3. Skinny HIP vs `F.linear` at `M=1,2,4,8`.
4. Prefill: rocBLAS vs 64×64×32; **keep BLAS if it wins**.
5. MLA prefill: OOB, then `fdot2` (first attention rewrite).
6. Occupancy-fixed `fa_rdna2` vs `TRITON_ATTN`.
7. Native FP16 MoE uses the skinny contract.

FlyDSL later (needs clean `v_dot2`). No invented tok/s.

## Cards

Reuse occupancy + baseline epic. Not a new first kernel ticket.

## Sources

- [silicon/fp16-rdna2.md](../../silicon/fp16-rdna2.md)
- [baseline-order.md](../stock-baseline/baseline-order.md), [fp16-moe.md](../moe/fp16-moe.md)
- Note 2026-08-18

Added in the 2026-10-11 sourcing pass:

- LLVM AMDGPU backend usage (processor feature table, dot insts per target) — <https://llvm.org/docs/AMDGPUUsage.html>
- vLLM PR #34709 — wvSplitK skinny GEMM for gfx1x (gfx1030 not included) — <https://github.com/vllm-project/vllm/pull/34709>

[^rdna2-isa]: AMD, *RDNA 2 Shader Instruction Set Architecture* (ISA 70648) — <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf>
[^vllm-issue38107]: vLLM issue #38107 — gfx1030 bf16 is emulated — <https://github.com/vllm-project/vllm/issues/38107>
[^dest-occupancy]: dest occupancy state at `1dc3bac4` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L283-L284>, <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L535-L536> (decode); <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L1054> (prefill); <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/skinny_gemms.cu#L1344-L1346> (`wvSplitKrc_`)
[^llvm-amdgpu-td]: LLVM `llvm/lib/Target/AMDGPU/AMDGPU.td` @ `694d1a69de0d`: `FeatureISAVersion10_3_0` (gfx1030) has dot1/2/5/6/7/10 only; `v_dot2_bf16_bf16` is `dot9-insts`, `v_dot2_f32_bf16` is `dot12-insts` (gfx11+) — <https://github.com/llvm/llvm-project/blob/694d1a69de0d/llvm/lib/Target/AMDGPU/AMDGPU.td>
