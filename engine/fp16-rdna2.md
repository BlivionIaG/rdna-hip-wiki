# Fastest FP16 path on gfx1030 (RDNA2)

Date: 2026-08-18. Engine study. Occupancy on `fa_rdna2` + `skinny_gemms.cu` still first. Do not invent tok/s.

Silicon: [silicon/valu.md](../silicon/valu.md). MoE kernels: [fp16-moe.md](fp16-moe.md) + [kernels/fp16-moe.md](../kernels/fp16-moe.md). Baselines: [baseline-order.md](baseline-order.md), [kernels/triton-skinny-gemm.md](../kernels/triton-skinny-gemm.md). Attention: [attention-dispatch.md](attention-dispatch.md).

## The only inner op

```text
acc_f32 += fdot2(a_f16x2, b_f16x2)   // V_DOT2_F32_F16
```

Peak from the RX 6950 XT / V620 table: **FP16 256 FLOPS/clock/CU**. No WMMA, no MFMA, no AGPR, no hardware bf16 (`v_dot2_f32_bf16` is RDNA3+). Scalar fp32 FMA after unpack is a **correctness/bring-up** path, not the roofline.

HIP: `__builtin_amdgcn_fdot2`, K even, `half2`-aligned. Do not hope Triton/LLVM peepholes `__hfma2` into DOT. Do not call this “INT8 compute.”

## Split by phase (do not one-tile it)

| Phase | Bound | Fastest honest path | Stock hole |
|---|---|---|---|
| **Decode** skinny `M∈{1,2,4,8}` | bandwidth + launch | Custom HIP GEMV, stream B, A in LDS/regs, `fdot2` | `F.linear` / rocBLAS. `wvSplitK`/`LLMM1` not compiled in (`on_gfx9() or on_gfx1x()`). |
| **Prefill** fat GEMM | can be compute | Measure **rocBLAS** vs HIP `64×64×32` `fdot2` tile | Vendor library is the baseline, not a religious “never beat BLAS” for skinny |
| **FA prefill** D=128/256 | compute-ish | `fa_rdna2` `fdot2` after occupancy flip | Triton `TRITON_ATTN` (`--dtype half`) is the A/B |
| **FA decode** paged | bandwidth | `fa_rdna2` split-K, occupancy `waves_per_eu(4,8)` | `ROCM_ATTN` on gfx1030 is Triton/Triton, not HIP |
| **MLA prefill** | both sides already half | **First place `fdot2` pays.** OOB `load_row` is the gate | Current HIP prefill is scalar FMA |
| **MLA decode** | unpack then math | Stay scalar until E4M3 unpack is gone | `fdot2` later |
| **FP16 MoE** | decode many tiny GEMMs | Two kernels: skinny + grouped. [fp16-moe.md](fp16-moe.md) | Triton MoE excludes gfx10xx |

## Occupancy is part of the path

`amdgpu_waves_per_eu(1,1)` / HIP second `__launch_bounds__` arg is a **min-waves** trap. Decode-class: drop min-blocks, `amdgpu_waves_per_eu(4, 8)`, target ≤64 VGPR, no scratch. Prefill HIP MLA is `__launch_bounds__(32)` — no `(1,1)` trap.

A perfect `fdot2` inner loop at 1 wave/EU is not the fastest path.

## Engine rules that keep FP16 fast

1. **Continuous batching + chunked prefill** so decode stays skinny and prefill gets fat tiles. Mixing them in one launch kills both.
2. **`--dtype half`.** bf16 is emulated on gfx1030.
3. **No Triton on the hot decode path** once HIP exists (compile tax).
4. **Collectives stay outside GEMM.** `0c59068e` is transport only.
5. **W7800 is a different ISA.** Fastest *box* path later may put attn on gfx1100 WMMA and experts on V620 `fdot2` ([multi-tier.md](multi-tier.md)). Fastest *V620 FP16* path is this page.
6. **V340L/gfx900 is not this path.** `mad_mix`/`pk_fma`, no DOT.

## Study order (after current occupancy card)

Matches [baseline-order.md](baseline-order.md). Do not skip the stock map.

1. Occupancy flip on `fa_rdna2` + `skinny_gemms.cu` — **now**.
2. Harness: same shapes, warm/JIT/graph, no tok/s invention.
3. Stock map: Triton FA + rocBLAS linear on V620.
4. Skinny HIP `fdot2` vs BLAS at `M=1,2,4,8`.
5. Triton FA vs `fa_rdna2` (occupancy-fixed).
6. Prefill fat: rocBLAS vs HIP `fdot2` 64×64×32.
7. MLA prefill: OOB, then `fdot2`.
8. Native FP16 MoE two kernels.

FlyDSL later (gate 0+). No MFMA kernels.

## Cards

Reuse occupancy + baseline epic. Optional Later: “FP16 roofline study” pointing here — not a new first ticket.

## Sources

- [silicon/valu.md](../silicon/valu.md), GPUOpen RX 6950 XT peak table
- [baseline-order.md](baseline-order.md), [fp16-moe.md](fp16-moe.md), [attention-dispatch.md](attention-dispatch.md)
- Room 2026-08-18: study fastest FP16 RDNA2
