# causalflow-ai/petit-kernel — silicon Take / Leave (gfx1030)

Date: 2026-09-05. [causalflow-ai/petit-kernel](https://github.com/causalflow-ai/petit-kernel) (BSD), tip skimmed via README + `RELEASE_NOTES` 0.0.5 + blog [Optimizing FP4 Mixed-Precision Inference on AMD GPUs](https://www.causalflow.ai/blogs/2025-08-optimizing-fp4-mixed-precision-inference-on-amd-gpus) (Aug 2025) + `lib/gemm/rocm/quantization/{dequant.cuh,types.h,fp4/*}`. Do not invent tok/s. Engine dispatch stays on [../kernels/mxfp4.md](../kernels/mxfp4.md) / [../kernels/nvfp4.md](../kernels/nvfp4.md).

Petit = **CDNA2/3/4** (MI200 / MI300 / MI350) FP16/BF16 × FP4 dense + fused MoE. Requirements say CDNA only; MegaMoE needs `gfx950`. Inner path is Marlin-style offline shuffle → dequant to fp16/bf16 → **MFMA MatrixCore**, not `fdot2`.

Dest lock: gfx1030 V620, ROCm 7.14, wave32, 64 KiB LDS/WG, 64 banks × 4 B. **No WMMA/MFMA/FP8 unit.** Dest mxfp4 = unpack E2M1+UE8M0 → `fdot2`. Produce stays EXL3 grain v2 `bits=3` `M≤8`.

## Take

| Idea | Why it transfers (or partially) |
|---|---|
| **Offline shuffle / pack for dequant** | Marlin-class: rearrange FP4 nibbles so GPU-side unpack is fewer VALU ops. Their CDNA pack uses `v_bfrev_b32` + `v_cvt_pk_f32_bf8` / SDWA (15 inst vs 31). On gfx1030 keep the *idea* (pack for cheap unpack) but rewrite against our E2M1 bit-trick / EXL3 3-inst codebook — **not** their bf8 convert path. |
| **LDS bank-conflict awareness + tile hyperparams** | Blog: CDNA LDS is **32 banks**, wave64 → conflicts bite hard; permute layouts; shared-memory tile shapes are sensitive — run benches. Transfer: same discipline on gfx1030 with **64 banks / wave32** ([lds-tiles.md](lds-tiles.md)). Do not copy their 32-bank XOR. |
| **CDNA2 (MI200) denormal flush caveat** | README: MFMA on MI200 flushes denorm in/out → accuracy hit; their corrective path ~+10% overhead. Transfer as a **numerics warning**: gfx1030 `fdot2` is a different pipe, but any fold that relies on e4m3/fp16 subnormals (cf. radiance `kMag` into subnormals) must be **measured on V620**, not assumed from CDNA2 or gfx12 WMMA. |
| **Bounded / clamp loads over predicated staging** | Blog: buffer loads with range discard OOB instead of branches. Aligns with radiance “clamp, never predicate” staging — useful for skinny M. |
| **Scale-type split already in our contract** | Their `DataType` enum has `Fp4e2m1`, `MxFp4e2m1`, `Fp8e4m3`, `Fp8e8m0`, `Fp8e5m2Fnuz`. Confirms NVFP4 (E4M3 mul) ≠ MXFP4 (E8M0 exp) — same split as [../kernels/nvfp4.md](../kernels/nvfp4.md) / [../kernels/mxfp4.md](../kernels/mxfp4.md). |

## Leave

| Item | Why |
|---|---|
| **Entire MFMA / MatrixCore GEMM grid** | `gemm_fp4_fp16_grid*.cuh`, warp schedules, solution maps — CDNA `v_mfma_*`. Will not load / is wrong ISA on gfx1030. Dest path is `mxfp4_dot2_*` / `fdot2`. |
| **CDNA-only arch gate / MI300X tok/s / hipBLASLt A/B** | Requirements + blog numbers are Instinct. Do not cite as V620 evidence. |
| **Chiplet / XCD L2–L3 topology scheduling** | MI300 XCD interconnect. V620 is single-die RDNA2 + Infinity Cache — different hierarchy ([cache-policy.md](cache-policy.md)). |
| **`v_cvt_pk_f32_bf8` / FNUZ e5m2 bias quirks as our dequant** | CDNA3 FNUZ (`kIntermediateConvertBias`) and gfx950/gfx12 OCP branches in `dequant.cuh`. gfx1030 has no bf8 MatrixCore path; we stay on E2M1→fp16 bit-trick + E8M0 exp add or E4M3 mul. |
| **MegaMoE / A8W4/A4W4 fused MoE as drop-in** | `gfx950` experimental; produce on dest is EXL3 / ticket-26, not Petit's MoE pack. |
| **Importing Petit as the gfx1030 NVFP4/MXFP4 engine** | Useful reference for shuffle + numerics footnotes only. Implementation ownership stays extras HIP. |

## CDNA2 accuracy / perf → RDNA2 `fdot2`

| CDNA2 fact | Transfers? |
|---|---|
| MFMA denorm flush needs a corrective upscale of scales (`kFp8ScaleBias`, `kUpscale`) | **Concept yes, code no.** If a fold pushes e4m3/fp16 into the subnormal range, measure flush-vs-honor on gfx1030 VALU; do not ship Petit's MFMA upscale table. |
| Dequant + LDS layout dominate small-`m` wins vs BF16 BLAS | **Yes as a priority ordering** (memory-bound decode). Our skinny path is already `fdot2` + LDS A tile, not MFMA occupancy. |
| 32-bank / wave64 conflict model | **No** — rewrite with 64-bank / wave32 ([lds-tiles.md](lds-tiles.md) §1). |
| hipBLASLt / MFMA peak % claims | **Leave.** |

## Sources

- https://github.com/causalflow-ai/petit-kernel (README, RELEASE_NOTES 0.0.5 2026-09-04)
- https://www.causalflow.ai/blogs/2025-08-optimizing-fp4-mixed-precision-inference-on-amd-gpus
- `lib/gemm/rocm/quantization/dequant.cuh`, `types.h`, `fp4/gemm_fp4.h`
- [../kernels/mxfp4.md](../kernels/mxfp4.md), [../kernels/nvfp4.md](../kernels/nvfp4.md), [lds-tiles.md](lds-tiles.md), [exl3.md](exl3.md)
