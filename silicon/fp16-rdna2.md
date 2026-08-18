# Fastest gfx1030 FP16 path — silicon contract

Date: 2026-08-18. **Study / after occupancy.** Peak math from [valu.md](valu.md). Engine: [engine/fp16-rdna2.md](../engine/fp16-rdna2.md). Native MoE tiles: [../kernels/fp16-moe.md](../kernels/fp16-moe.md). Stock hole: [../kernels/triton-skinny-gemm.md](../kernels/triton-skinny-gemm.md).

## One-line verdict

Fastest legal FP16 compute on gfx1030 is **explicit `__builtin_amdgcn_fdot2` → `V_DOT2C_F32_F16`**, fp32 accum, wave32, occupancy that actually launches. Not WMMA, not `tl.dot`, not `__hfma2` hoping hipcc peels a DOT, not gfx900 `mad_mix`.

GPUOpen RX 6950 XT class: **256 FP16 FLOP/clk/CU** via packed DOT/FMA. That is already the RDNA2 ceiling. gfx1100 WMMA doubles it; V620 does not have WMMA.

## Inner opcode

```cpp
acc = __builtin_amdgcn_fdot2(a_h2, b_h2, acc, /*clamp=*/false);
```

- Prefer the **C** form (`V_DOT2C_F32_F16`, VOP2, 1 dword, dest=accum).
- K even, operands aligned `half2` / 32-bit words.
- Accumulators stay **fp32** until the store/epilogue.
- hipcc 7.2 will **not** rewrite `__hfma2` + add into DOT2. Issue the builtin and ISA-dump it.
- `V_PK_FMA_F16` can also hit 256 FLOP/clk/CU but accumulates **fp16**. Use it only for bandwidth-bound epilogues, never as the GEMM/attention reduction.
- Scalar `v_fma_f32` after `__half2float` is half-width or worse (two converts + one FMA per pair). That is today's HIP MLA / indexer tax.

## Two kernels, not one tile

| Regime | Shape | Fast path |
|---|---|---|
| Decode / skinny | `M=1,2,4,8` | HIP GEMV: A in LDS, stream B as `half2`, `fdot2`, specialize M |
| Prefill / fat | `M≥32` | First **measure rocBLAS / hipBLASLt**. Then a tiled `fdot2` seed **64×64×32**, WG256, 16 fp32 acc/thread |
| Messy band | `8<M<32` | `M_COUNT=8` scalar-DOT2 first; write 32×32 only if CU fill is bad |

Stock vLLM on gfx1030 does **not** fire `LLMM1`/`wvSplitK` (`on_gfx9() || on_gfx1x()`). Decode GEMM today is `F.linear` / rocBLAS. That is the hole a skinny `fdot2` kernel is allowed to attack.

Do not launch a 64×64 tile at `M=1`. Do not copy gfx1100 WMMA 16×16 fragments.

## Occupancy is part of the peak

256 FLOP/clk/CU is unreachable if the kernel is occupancy-1 or occupancy-0.

- Current trap: `__launch_bounds__(*, 1)` / `amdgpu_waves_per_eu(1,1)` on `fa_rdna2` and `wvSplitKrc_`. HIP's second bound is **min waves/EU**, not CUDA min-blocks.
- Target after the flip: `amdgpu_waves_per_eu(4, 8)` only if VGPR ≤64 (decode) and no scratch.
- `llvm-calc-occupancy -mcpu=gfx1030` **and** `hipOccupancyMaxActiveBlocksPerMultiprocessor` must be ≥1.
- This occupancy fix is still **first**, before any new FP16 kernel.

## Attention

| Kernel | Today | Fast FP16 |
|---|---|---|
| `fa_rdna2` | occupancy-trapped; QK already `fdot2` | flip waves/EU, keep `fdot2`, leave Sage/INT8 for later |
| HIP sparse MLA decode | scalar fp32 FMA (FP8 K_nope unpack) | later: stop promoting Q pairs |
| HIP sparse MLA prefill @ `66bb24d7` | scalar FMA, both sides already `half` | **first** `fdot2` rewrite after `load_row` OOB fix |
| Stock `TRITON_ATTN` / `ROCM_ATTN` | gfx1030 decode is Triton fallback | baseline only; ISA often scalar. Tune on a separate page |

`fa_rdna2` vs Triton A/B only after occupancy + identical cache/mask/graph contract.

## Memory rules that decide whether FLOP shows up

- LDS as `half2`/dword, 64-bank WGP, pad A rows (`+8` seed) and measure conflicts.
- 16-byte global loads when aligned.
- Software-pipeline weight/A loads over the 5-cycle VALU dest latency.
- Infinity Cache is 128 MB: a 7B fp16 layer does **not** fit unsharded. TP=4 may fit; nontemporal the miss.
- No persist/bypass bit. Fit + reuse is the whole policy.

## What is not the fast path

- Triton `tl.dot` / `tl.sum` without an ISA dump proving `v_dot2*`.
- rocBLAS at `M=1..8` as “good enough” without measuring (it is the stock baseline, not the claimed peak).
- FlyDSL until Gate 2 emits clean `v_dot2_f32_f16` ([flydsl-dot-atoms.md](flydsl-dot-atoms.md)).
- gfx900 `mad_mix` (V340L) or gfx906 DOT objects.
- Dequant-in-the-K-loop formats when the question is **pure FP16**. Those are W4A16/W8A16; they still use `fdot2` after unpack but they are not the FP16 peak.

## Proof order

1. Occupancy flip on `fa_rdna2` (+ skinny if we open that gate). ISA + runtime occ.
2. Microbench: one `fdot2` vs `__hfma2` vs scalar FMA on 4096-long GEMV. Must show `v_dot2c`/`v_dot2`.
3. Skinny HIP FP16 `M=1,2,4,8` vs stock `F.linear` on real q/o/MLP shapes.
4. Prefill: rocBLAS vs 64×64×32 `fdot2` tile; keep BLAS if it wins.
5. MLA prefill: fix `load_row`, then `fdot2` both-sides-half.
6. `fa_rdna2` vs forced `TRITON_ATTN` after step 1.
7. Native FP16 MoE decode uses the same skinny contract ([../kernels/fp16-moe.md](../kernels/fp16-moe.md)).

Record cold compile, warm, median/p10/p90, VGPR/LDS/spills/waves, DRAM bytes, IC/L2 hits, and the profiler symbol. No invented tok/s.

## Sources

- [valu.md](valu.md) GPUOpen 256 FP16 / clk / CU
- RDNA 2 ISA 70648: `V_DOT2_F32_F16` / `V_DOT2C_F32_F16`
- hip-craft: hipcc does not peephole `hfma2` → DOT2
- vLLM `rocm_unquantized_gemm_impl`: skinny gated off gfx1030
