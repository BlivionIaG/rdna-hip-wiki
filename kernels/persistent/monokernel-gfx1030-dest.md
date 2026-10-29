---
title: "MonoKernel on gfx1030: what dest built"
sources:
  - https://github.com/opengfx1030/vllm-rdna/pull/94
  - https://github.com/opengfx1030/vllm-rdna/commit/1c1426d585c9b2653265947734225721cfc7ea6e
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_gdn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_qsa_decode_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/rdna_allreduce.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/rdna_allreduce.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/platforms/rocm.py
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/models/qwen4_exp/amd/mono/rdna2/__init__.py
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/models/qwen4_exp/amd/mono/rdna2/README.md
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/docs/rdna/monokernel/p0-results.md
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/models/qwen4_exp/amd/mono/rdna2/gdn.py
  - https://github.com/opengfx1030/vllm-rdna/commit/655cde1ce20838cf86f6a2f071db53cac089727b
  - https://github.com/opengfx1030/vllm-rdna/commit/2436df23358ef8b9397076c0663707ec3dce69bf
---
# MonoKernel on gfx1030: what dest built

Silicon locks from the MonoKernel merge into dest `rdna_extras` (PR #94, tip `1c1426d5` = `1c1426d585c9b2653265947734225721cfc7ea6e`).[^pr94] This page supersedes the open questions in [monokernel-gfx1030-plan.md](monokernel-gfx1030-plan.md), which stays as the historical plan read at `e245a469`. No tok/s here; the fork's own results live under `docs/rdna/monokernel/`.[^readme]

## Update at dest `rdna_extras` @ `b7402893` (checked 2026-10-29)

Read at tip `b7402893` (`b74028935393d1e21f73f80f65f506c9ea4639af`). Two silicon changes since this page was locked at `1c1426d5`: the GDN norm reduction (`655cde1c`) and a staged K2 for MoE expert offload (PR #146, `2436df23`).[^gdnfix][^c2436] The rows below supersede the matching bullets further down, which stay as history.

| Item | State at `b7402893` |
|---|---|
| GDN gated-RMSNorm sum of squares | **FP32 per-tile partials, not Q32.** Each of the `kV / kBV` = 4 V tiles of a (token, head) has one writer and stores its squared-sum partial as a plain `volatile float` store into `ssq[(tok·HV + hv)·4 + tile]`; the last tile sums the four partials in fixed tile order and computes `rsqrtf(tot / kV + eps)`. The source says Q32 "overflowed above RMS4096 and discarded small squared sums", while FP32 covers the finite FP16 range, and the fixed order keeps the result deterministic regardless of arrival order. The old int64 `atomicAdd` and the per-call `*ps = 0` clear are gone.[^gdnpart][^gdnsum][^gdnconst2] |
| GDN scratch layout | FP32 `ssq` partials, then `cnt`, then `cntqk`: `kCapRows · (HV·20 + H·4)` bytes (was `HV·12 + H·4`), then the fp16 `obuf`, then one `int` diagnostic flag; `mono_gdn_rdna2_scratch_bytes` grows to match.[^gdnscratch] |
| GDN accepted-index guard | Padded sequences may carry `nacc = 0`, so the conv slot is checked first and the accepted-state index is read only after `1 ≤ accepted ≤ tmax`; invalid metadata writes NaN to `y` instead of reading outside the slot row.[^gdnacc] |
| GDN diagnostics | `gdn_fused_k<KMAP, DIAG>`: `DIAG = true` only when `VLLM_RDNA_GDN_DIAGNOSTICS=1` (read once, process-static). It adds bit-test finiteness checks (`finite32`, works under fast-math) and a first-error `atomicCAS` flag plus `printf`; the default `DIAG = false` specialization compiles none of it.[^gdndiag][^gdnenv][^gdnpy] |
| GDN occupancy | Unchanged: `__launch_bounds__(kThreads)` with `kThreads = 256`, grid `(N·HV, kV/kBV)`, dynamic LDS `tmax · (2·kK + kBV) · 4` B.[^gdnlb][^gdnconst2][^gdnlds2] |
| K2 offload stages | `mono_ffn_kernel<MT, WPE, Stage>` (Stage 0 = the resident kernel, unchanged). Per the source: stage 1 (prefix) runs P1–P4 including router / top-k and the shared expert and writes `route_ids` / `route_weights`; stage 2 (expert pass) reads them and computes only pairs whose local expert maps to a resident slot via `local_to_slot`, then returns before P7; stage 3 (finish) runs only P7. "Separate specializations leave the resident kernel's generated code intact."[^k2stage][^k2offload][^k2launch] |
| K2 offload limits | `__launch_bounds__(256)` and `amdgpu_waves_per_eu(WPE, 16)` as before (WPE 8 for every M because `MONO_FFN_TILED_BIG` is 0). Offload stages raise `LocalCapacity` from `kLocalE` = 128 to 256 local experts and allow `Ise ≤ 320` (resident: 160); the entry requires symmetric Flash-Next INT4 geometry (`wr` 512 rows, H 2560, w13 `[slots, 320, 1280]`, w2 `[slots, 80, 2560]`, `zero_offset == 1`) and an int32 `local_to_slot[128 or 256]`.[^k2lb][^k2cap][^k2checks] |
| K2 offload LDS | The four expert-list arrays (`s_used`, `s_ucnt`, `s_slot` int, `s_urow` short × `kMaxM` = 24) are sized by `LocalCapacity`: 7680 B at Stage 0, 15360 B in the offload stages (computed from the declarations). Each specialization gets its own slot in the code-object residency cap (`cap[16]`, slots 4–15), so the grid is re-derived from that object's VGPR / LDS rather than reused from the resident kernel.[^k2lds][^k2launch][^k2capslot] Whether the larger static LDS crosses the 32 KiB rounding step noted below is (unsourced): it needs the code object. |

## What is on dest

| Part | Kernel | Shape | Width gate (default) |
|---|---|---|---|
| K1 | `mono_ffn_kernel` with `hc_only` (attention-side HC preamble, stops after P3) | persistent, 256 threads | ≤ 24 rows[^k1][^opts] |
| K2 | `mono_ffn_kernel` (whole FFN block P1–P7: HC combine + norm, HC down/up GEMVs, fp32 router, top-k, shared expert, routed W4A16 experts) | persistent, 256 threads | ≤ 8 rows[^k2hdr][^opts] |
| GDN | `gdn_fused_k` (conv1d update + gated delta rule + gated RMSNorm) | one WG per (seq, value head, 32-row V tile), 256 threads | ≤ 64 rows[^gdnhdr][^opts] |
| QSA | `qsa_dec_k` (sparse paged GQA decode + split merge) | grid (rows × kv heads, splits), 256 threads = 8 waves | decode / verify[^qsahdr][^readme] |

All three sources are in the gfx1030 HIP list in `CMakeLists.txt`.[^cmake] Off by default: `--monokernel` (`KernelConfig.monokernel`), with `VLLM_ROCM_MONO_DECODE=1` as an alias read by the ROCm platform; startup refuses non-gfx10x, non-Qwen4Exp, MRV2 off, PP > 1, or a non-symmetric-W4A16 MoE.[^env][^rocm][^readme] The workers need `GPU_MAX_HW_QUEUES=4`; the engine raises it to 4 by default (`hw_queues` knob).[^readme][^opts]

## Grid sync and residency (occupancy first)

- **Plain launch, own barrier, not cooperative launch.** K2 separates phases with a grid barrier on a generation counter in global memory (agent-scope atomics, bounded wait, no `grid.sync()`); a timeout writes a sticky `{code, phase, wg}` record, every WG leaves, and a later launch that sees the record writes NaN and exits.[^k2hdr][^barrier]
- **Why:** the P0 probe found HIP refuses `hipLaunchCooperativeKernel` for 72 × 64 KiB WGs in the default WGP build (`hipErrorCooperativeLaunchTooLarge`, also at capture) although 72 are co-resident; it works only with `-mcumode`. `grid.sync()` is unbounded and hung when one WG bailed.[^p0coop][^p0bound]
- **Occupancy API under-reports 2× in WGP mode.** `hipOccupancy*` reports 36 units ("CUs" = WGPs) at 64 KiB each; the V620 held 72 × 64 KiB × 256-thread WGs co-resident in both WGP and CU mode.[^p0res]
- **So dest computes the cap from the code object:** waves per SIMD = min(16, 1024 / VGPRs) (wave32), 2 SIMDs per CU, 64 KiB LDS per CU, CUs = 2 × the reported WGP count; grid = min(by-waves, by-LDS) WGs per CU × CUs.[^cap]
- **K2 target:** `__launch_bounds__(256)` plus `amdgpu_waves_per_eu(8, 16)` on the M ≤ 8 row path, i.e. ≤ 128 VGPRs and 2 WGs (16 waves) per CU, which the header gives as 144 WGs on a V620.[^lb][^k2hdr] The M > 8 4×8 register-tiled variant (one WG per CU) is compiled out by default (`MONO_FFN_TILED_BIG 0`), so M = 12 / 24 also launch with WPE 8.[^tiled][^dispatch]
- **LDS rounding trap:** at grid 144 with 32 KiB + 4 B static, the WGP build held only 108 WGs (3 per WGP) because allocation rounds up past 32 KiB; keep static + dynamic LDS at exactly 64 or 32 KiB.[^p0res][^p0concl]
- **Starvation, not deadlock:** a wave-filling kernel on another stream kept every WG of the persistent kernel from starting for ~6 s; the device-side bound starts at WG entry and cannot see that wait. Keep the mono step on one stream.[^p0fp]

## ISA and memory locks

- **Dense paths are fp16 `__builtin_amdgcn_fdot2` with fp32 accumulate** (HC GEMVs, router, shared expert); routed int4 experts dequantize to half and also use `fdot2`. No sdot4 in the MonoKernel sources.[^fdot2][^fdot2q]
- **QSA decode** also uses `fdot2` for q·k, lane l owns head dims 8l..8l+7 (head_dim 256), ≤ 8 query heads per group, probabilities rounded to fp16 before P·V like the Triton kernel.[^qsahdr][^qsadot]
- **Cross-WG sums are 64-bit fixed point** (2^-32 resolution) through native `global_atomic_add_x2`, because gfx1030 has no global fp32 atomic add (`atomicAdd(float*)` is a CAS loop); integer sums are order-independent, so reductions are deterministic.[^fx] *(Historical @ `1c1426d5`, still true at `8015ea48`; superseded at `b7402893` by FP32 per-tile partials, see the update table above.)* The GDN gated RMSNorm uses the same int64 fixed-point sums of squares.[^gdnhdr]
- **Barrier wait:** wave 0 arrives and polls with `s_sleep` backoff (2 to 127) while waves 1–7 prefetch the next phase's weights into L2 / Infinity Cache.[^barrier]
- **LDS tile:** `wg_tile_tn` ports the `hc_prefill_rdna2.cu` split-K layout to 256 threads: k-pair-major `half2`, 32-element slabs, A stride `BM + 4`, B stride `BN + 2`, next slab prefetched into registers.[^tile]
- **GDN LDS** is dynamic, `tmax × (2·128 + 32) × 4` bytes (tmax ≤ 8 MTP verify tokens).[^gdnconst][^gdnlds]

## In-kernel all-reduce (Stage D, opt-in)

`rdna_ar_device()` exports the one-shot AR's peers, sequence buffer and slot size as `RdnaArDev` so K2 can run the same protocol (push to every peer's slot, bump seq, publish flag, wait, sum in rank order).[^ardev][^arhost] K2 does one `__threadfence_system()` per WG (not per thread) before publishing.[^stageD] It only engages when the FFN payload fits `max_bytes` and is off by default (`k2_ar: false`).[^arcap][^opts]

## Superseded plan claims

Marked historical in the plan page (`e245a469`): "plan, nothing built"; "cooperative launch under hipGraph capture … not tested" (answered: only with `-mcumode`, dest uses plain launch);[^p0coop] "grid sizing via `hipOccupancyMaxActiveBlocksPerMultiprocessor` × 72" (under-reports 2× in WGP mode);[^p0res] "safe persistent grid = 72 WGs" (K2 targets 2 WGs per CU at ≤ 128 VGPRs);[^k2hdr][^lb] "sticky error word … land with 2b" (already in K2's barrier).[^barrier]

## Sources

[^pr94]: [opengfx1030/vllm-rdna PR #94](https://github.com/opengfx1030/vllm-rdna/pull/94), merge commit [`1c1426d585c9b2653265947734225721cfc7ea6e`](https://github.com/opengfx1030/vllm-rdna/commit/1c1426d585c9b2653265947734225721cfc7ea6e) (51 commits over `e84c2eb2`).
[^readme]: [mono/rdna2/README.md:1–50](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/models/qwen4_exp/amd/mono/rdna2/README.md#L1-L50) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^opts]: [mono/rdna2/__init__.py:15–35](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/models/qwen4_exp/amd/mono/rdna2/__init__.py#L15-L35) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^k1]: [mono_k2_rdna2.cu:150, 1113, 1923, 2171–2173](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L2171-L2173) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^k2hdr]: [mono_k2_rdna2.cu:4–36](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L4-L36) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^barrier]: [mono_k2_rdna2.cu:225–324](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L225-L324) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^cap]: [mono_k2_rdna2.cu:2089–2115](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L2089-L2115) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^lb]: [mono_k2_rdna2.cu:775–783](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L775-L783) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^tiled]: [mono_k2_rdna2.cu:759–774](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L759-L774) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^dispatch]: [mono_k2_rdna2.cu:2116–2127](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L2116-L2127) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^fx]: [mono_k2_rdna2.cu:91–109](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L91-L109) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^fdot2]: [mono_k2_rdna2.cu:360–372, 450–453, 494–495, 588](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L360-L372) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^fdot2q]: [mono_k2_rdna2.cu:674–677](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L674-L677) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^tile]: [mono_k2_rdna2.cu:507–530](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L507-L530) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^stageD]: [mono_k2_rdna2.cu:1703–1764](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L1703-L1764) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^arcap]: [mono_k2_rdna2.cu:2080–2085](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L2080-L2085) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^gdnhdr]: [mono_gdn_rdna2.cu:4–29](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_gdn_rdna2.cu#L4-L29) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^gdnconst]: [mono_gdn_rdna2.cu:41–44, 158](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_gdn_rdna2.cu#L41-L44) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^gdnlds]: [mono_gdn_rdna2.cu:436–441](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_gdn_rdna2.cu#L436-L441) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^qsahdr]: [mono_qsa_decode_rdna2.cu:4–17, 31–40, 69](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_qsa_decode_rdna2.cu#L4-L40) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^qsadot]: [mono_qsa_decode_rdna2.cu:161](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_qsa_decode_rdna2.cu#L161) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^ardev]: [rdna_allreduce.cuh:163–174](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/rdna_allreduce.cuh#L163-L174) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^arhost]: [rdna_allreduce.cu:69–80](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/rdna_allreduce.cu#L69-L80) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^cmake]: [CMakeLists.txt:1527–1529](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/CMakeLists.txt#L1527-L1529) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^env]: [envs.py:156, 1409–1414](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/envs.py#L1409-L1414) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^rocm]: [platforms/rocm.py:1134–1143](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/vllm/platforms/rocm.py#L1134-L1143) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^p0res]: [p0-results.md:46–58](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/docs/rdna/monokernel/p0-results.md#L46-L58) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^p0coop]: [p0-results.md:60–76, 279–282](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/docs/rdna/monokernel/p0-results.md#L60-L76) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^p0fp]: [p0-results.md:78–92](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/docs/rdna/monokernel/p0-results.md#L78-L92) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^p0bound]: [p0-results.md:94–105](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/docs/rdna/monokernel/p0-results.md#L94-L105) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^p0concl]: [p0-results.md:107–122](https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/docs/rdna/monokernel/p0-results.md#L107-L122) @ 1c1426d585c9b2653265947734225721cfc7ea6e.
[^gdnfix]: opengfx1030/vllm-rdna commit [`655cde1c`](https://github.com/opengfx1030/vllm-rdna/commit/655cde1ce20838cf86f6a2f071db53cac089727b) "fix(rdna2): preserve finite GDN normalization across FP16 range": "Replace overflowing Q32 squared sums with fixed-order FP32 tile partials. Add opt-in device diagnostics and reject invalid accepted indices before state reads."
[^c2436]: opengfx1030/vllm-rdna commit [`2436df23`](https://github.com/opengfx1030/vllm-rdna/commit/2436df23358ef8b9397076c0663707ec3dce69bf) "[ROCm] Add graph-safe RDNA2 MoE expert offloading" (PR #146).
[^gdnpart]: [mono_gdn_rdna2.cu:352–361](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L352-L361) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnsum]: [mono_gdn_rdna2.cu:370–381](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L370-L381) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnconst2]: [mono_gdn_rdna2.cu:43–46, 75](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L43-L46) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnscratch]: [mono_gdn_rdna2.cu:467–476, 513–523, 545–548](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L467-L548) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnacc]: [mono_gdn_rdna2.cu:202–231](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L202-L231) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdndiag]: [mono_gdn_rdna2.cu:83–102, 190–191](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L83-L102) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnenv]: [mono_gdn_rdna2.cu:469–475, 526–541](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L469-L541) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnpy]: [mono/rdna2/gdn.py:93–102](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/models/qwen4_exp/amd/mono/rdna2/gdn.py#L93-L102) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnlb]: [mono_gdn_rdna2.cu:191](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L191) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gdnlds2]: [mono_gdn_rdna2.cu:525–527](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_gdn_rdna2.cu#L525-L527) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2stage]: [mono_k2_rdna2.cu:949–950, 1425, 1668–1676, 1690–1697, 1858–1861, 1992](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L1668-L1697) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2offload]: [mono_k2_rdna2.cu:2511–2570](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L2511-L2570) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2launch]: [mono_k2_rdna2.cu:2456–2478](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L2456-L2478) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2lb]: [mono_k2_rdna2.cu:872–894](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L872-L894) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2cap]: [mono_k2_rdna2.cu:66–69, 2241–2260](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L2241-L2260) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2checks]: [mono_k2_rdna2.cu:2525–2544](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L2525-L2544) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2lds]: [mono_k2_rdna2.cu:917–922](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L917-L922) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^k2capslot]: [mono_k2_rdna2.cu:2429–2436](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/mono_k2_rdna2.cu#L2429-L2436) @ b74028935393d1e21f73f80f65f506c9ea4639af.
