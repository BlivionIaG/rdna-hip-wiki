# hipfire — silicon review

Date: 2026-08-21. Engine: [../engine/hipfire.md](../engine/hipfire.md). Repo: [warpfront/hipfire](https://github.com/warpfront/hipfire). Occupancy still first. Do **not** copy their tok/s.

Standalone Rust + HIP. Tuned family is **WMMA on gfx11/12**. gfx1030 is a **capability-gated portable** path.

## Capability lock (`arch_caps.rs`, sourced)

Atoms → molecules → `has_*`. This is the steal: **feature predicates, not `on_rdna()`**.

| Cap | gfx1030 | Notes |
|---|---|---|
| `is_rdna2` | yes (`1030/1031/1032`) | Wave32. |
| `has_dot2_f32_f16` | **yes** | Also gfx1011/12 + gfx11/12. **gfx1010 and gfx906: no.** |
| `has_wmma` / `has_wmma_w32` | **no** | `is_rdna3 \| is_rdna4` only. |
| `has_hfq3_sdot4` | **yes** | `rdna1p1 \| rdna2`. `v_dot4_i32_i8`. |
| `has_hfq3_mmq` / `has_hfq4_mmq` | **yes** (default-on) | Prefill MMQ on the sdot4 allowlist. Escape hatches exist. |
| `has_mmq` (generic Q8) | **no** | gfx906 + RDNA3 only. Do not assume Q8 MMQ exists on V620. |
| `should_use_mmq` cutoff | **256** | gfx11/12 = 128; comment says RDNA2 untested lower. Sweep, don't paste. |
| `gemv_rows_default` | **1** | Same as wave64-native and RDNA3 dGPU. |
| `is_wave32` | yes | Native. |

ANTIBLEED they already paid for: mb4 / WMMA sources **panic** if admitted on a chip with no variant. gfx1030 must never take a gfx1100 `.gfx1100.hip` / `_wmma` object.

## Inner ops on V620

Same ISA we already fire ([valu.md](valu.md), [fp16-rdna2.md](fp16-rdna2.md)):

- Decode skinny: `fdot2` / `V_DOT2C` (their HFQ3 GEMV is arch-agnostic HIP, not a gfx1030-special TU).
- Prefill fat: `sdot4` MMQ **if** both sides are i8 and the batch clears 256. Else per-token GEMV fallback.
- No WMMA. No rocBLAS-required for their portable path (they `dlopen` HIP; rocBLAS is MI300-lazy).
- Lloyd codebook-in-LDS is **out of scope on gfx10** in their own MQ3 plan (RDNA2 vs RDNA3 cvt). Do not steal Lloyd-on-gfx10.

HIP does not need their Triton byte-slice dodge (that was leapdragon `fd_rdna2`). Packed load → VGPR cvt → `fdot2` stays ours ([../kernels/kv-int8.md](../kernels/kv-int8.md)).

## Occupancy

Their plan mentioned `__launch_bounds__(32, 16)` on `gemv_hfq3g256` as a *spill risk on gfx10*, not a recipe. extras still sits on `__launch_bounds__(N, 1)` / `waves_per_eu(1,1)`. **Our occupancy card is not closed by their portable path.**

Redline (ROCr retained-replay / PM4) is dispatch overhead, not a CU-occupancy fix. hippih Later. Fail-closed to HIP is the only part to keep.

## Numbers (sourced, not ours)

`tests/speed-baselines/gfx1030.txt` is a **speed-gate floor** for *their* MQ4 stack, captured 2026-04-28. 0.8B/4B/9B = RX 6900 XT 16 GB; 27B = V620 Pro 32 GB and **not enforced** on gfx1030 CI. `docs/BENCHMARKS.md` historical row is a different fixture (“truth state: historical”). Neither is extras W4 / TP=4 / ROCm 7.14 / 88096 PIX. Do not write them into [coverage.md](../engine/coverage.md).

## Not a dead end for us

Portable ≠ tuned. Their gfx1030 path being “correct, slower prefill” does **not** make extras `fa_rdna2` / `skinny_gemms.cu` `(1,1)` a dead end. Occupancy flip still first.

## Sources

- https://github.com/warpfront/hipfire/blob/master/crates/rdna-compute/src/arch_caps.rs
- https://github.com/warpfront/hipfire/blob/master/docs/plans/mq3_gfx10.md
- https://github.com/warpfront/hipfire/blob/master/tests/speed-baselines/gfx1030.txt
- [../engine/hipfire.md](../engine/hipfire.md), [valu.md](valu.md), [hippih.md](hippih.md)
