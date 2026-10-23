---
title: "MonoKernel on gfx1030: silicon plan"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/1c1426d585c9b2653265947734225721cfc7ea6e
  - https://github.com/vllm-project/vllm/issues/60904
  - https://github.com/vllm-project/vllm/pull/61001
  - https://github.com/vllm-project/vllm/pull/61325
  - https://github.com/vllm-project/vllm/pull/60731
  - https://github.com/vllm-project/vllm/pull/60397
  - https://github.com/vllm-project/vllm/pull/60889
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/VOP3PInstructions.td
  - https://github.com/ROCm/TheRock/issues/9306
  - https://github.com/ROCm/rocm-systems/issues/13234
  - https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/ockl/src/cg.cl
  - https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_fused_glue.cu
  - https://github.com/opengfx1030/vllm-rdna/tree/e245a4697a139fe8d3bbec4b9c837c37279674e9/bench_results/2026-09-29_tunableop-rows/curated
---
# MonoKernel on gfx1030: silicon plan

> **Historical (plan read at `e245a469`).** Dest built K1/K2, the fused GDN core and QSA decode in PR #94 at `1c1426d5`; see [monokernel-gfx1030-dest.md](monokernel-gfx1030-dest.md) for the current silicon locks. Claims below marked *(historical @ e245a469)* are superseded there.[^built]

Status at `e245a469`: **plan, nothing built** *(historical @ e245a469)*. Step 0 of Project 4 card `PVTI_lAHOAOnQn84BgjK3zhAaHOk`. Read against dest `rdna_extras` tip `e245a469` (`e245a4697a139fe8d3bbec4b9c837c37279674e9`) and the upstream PRs as of 2026-10-13. No tok/s here. Lines marked **[unsourced]** are my derivations and need a V620 check.

## What AMD built

- A MonoKernel is one persistent kernel running most or all of a decoder layer's decode step; off by default behind `--monokernel`; decode only; prefill and mixed steps run vLLM's layers.[^rfc]
- Flash-Next template (#61325): two persistent launches per layer (K1, K2), pure decode, 1–8 rows, no spec decode, TP2 on MI355X; K1 uses up to 55 KB LDS and K2 up to 68 KB at gfx950.[^61325]
- DSv4.1 (#60397, merged 2026-10-10 UTC): K1/K2 on 30 layers, an FFN launch on 10; all TP all-reduces in-kernel over symmetric peer memory; ≤48 rows; CTA hand-offs are epoch-tagged mailbox pairs in per-width scratch, no per-step clears, graph-safe.[^60397]
- gfx942 port (#60731): ≤42 rows because 64 KB LDS per workgroup (vs 160 KB on gfx950) means a GEMV pass takes 6 tokens instead of 16.[^60731]
- #61001 (WIP draft) generalises to `MonoOp` / `MonoSpec` / `MonoKernel` / `MonoRuntime`, with arch-specific specs (`max_rows` per gfx) and arbitrary partial-layer windows.[^61001]
- #60889 (GLM-5.2) has an empty description; only the RFC's numbers describe it.[^60889][^rfc]

## 1. Grid-wide sync

**AMD:** not cooperative launch. Cross-CTA hand-offs are tagged mailboxes with a device epoch bumped once per step.[^60397][^61325] The RFC requires every cross-CTA / cross-rank wait to be bounded, a sticky device error word on expiry, and a release fence before further data; it admits the cross-rank flag relies on a release fence ordering remote stores, "not formally guaranteed".[^rfc] The grid must fit the CUs actually present, or a partly resident grid deadlocks on its own barriers.[^rfc]

**HIP cooperative groups on gfx1030:** `grid.sync()` lowers to `__ockl_grid_sync`, which is a workgroup release fence + `s_barrier`, then (one lane) an agent-scope release, an atomic counter barrier over `num_wg`, agent-scope acquire, `s_barrier`.[^ockl] It is plain global atomics, so nothing gfx1030-specific blocks it; but it needs `hipLaunchCooperativeKernel` (grid capped at co-resident size) and is a full-grid barrier per phase. **Recommendation:** use our own epoch-tagged flags (producer→consumer), not `grid.sync` — same choice as AMD, and graph-capturable without cooperative launch. **[unsourced: cooperative launch under hipGraph capture on our ROCm not tested]** *(historical @ e245a469: P0 answered it, cooperative launch is refused in the WGP build and only captures with `-mcumode`; dest uses a plain launch with bounded barriers)*[^built]

**Room proposal (cooperative launch + `this_grid().sync()`):** workable on gfx1030 in principle, since `__ockl_grid_sync` is the same agent-scope fence + atomic-counter pattern a hand-rolled barrier would use; the difference is a full-grid barrier per phase versus point-to-point flags.[^ockl] Grid sizing via `hipOccupancyMaxActiveBlocksPerMultiprocessor` × 72, cross-checked against VGPR/LDS from `llvm-readelf --notes`, is the right check. *(historical @ e245a469: the occupancy API under-reports 2× in WGP mode; dest derives the cap from the code object)*[^built] **(untested P0 item, no source: whether `hipLaunchCooperativeKernel` can be stream-captured into a hipGraph on ROCm 7.14 — no source found; must be tested on the V620 before choosing it.)** If it can't be captured, flags are mandatory for FULL graphs.

**Hang triage:** ROCm/legacy-rocm-build#6698 (2× V620, compute queues stop being serviced) reports **no waves resident** at the hang (`umr --waves` empty, `SPI_CSQ_WF_ACTIVE_STATUS` = 0) and pending AQL packets on idle MEC pipes.[^6698] A grid-sync deadlock would leave spinning waves, so #6698 is a CP/queue-scheduling stall, not a barrier deadlock; it is still a risk because a persistent kernel cannot time out of a dispatch that never starts. rocm-systems#13234 (hipGraphLaunch hangs at ≥ 2 × `ROC_AQL_QUEUE_SIZE` packets in one segment) is not a blocker at the ~210 launches per step the RFC quotes for GLM-5.2.[^13234][^rfc] Design rule: every wait bounded. Sticky error word and per-timeout phase-ID logging land with 2b (cross-GPU waits) *(historical @ e245a469: K2's same-GPU barrier already writes a sticky {code, phase, wg} record)*[^built]; dest's AR report already encodes a phase code.[^ar-abort]

**Scopes on RDNA2 (GFX10 memory model):**[^llvm]
- One L0 per CU; `buffer_gl0_inv` needed for coherence between waves of different work-groups (and same WG in WGP mode).
- GL1 per shader array; `buffer_gl1_inv` needed between work-groups on different SAs.
- L2 shared by the agent; agent-scope atomic loads are `glc dlc` on gfx10.
- So a cross-WG flag wait is: `__hip_atomic_load(..., __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)` (compiler emits the glc/dlc load + `buffer_gl1_inv; buffer_gl0_inv`), and the producer does an agent-scope release store. That is exactly the pattern the dest AR already uses.[^ar-acq]
- **Cross-GPU:** a peer's write into coarse-grained memory is invisible to the owner's L2; dest therefore allocates staging and flags `hipDeviceMallocUncached` and polls locally.[^ar-uc][^ar-alloc]

## 2. Resident grid on a 72-CU V620

Inputs: wave32; max 16 waves per SIMD on gfx1030 (`FeatureMaxWavesPerEU16`);[^td-1030] at most 256 VGPRs addressable per wave (no `Feature1024AddressableVGPRs` on gfx10.3);[^bi-vgpr] occupancy = min(max waves, total VGPRs / rounded VGPRs).[^bi-occ] Total VGPR file of 1024 per lane per SIMD in wave32 **(unsourced: the per-GPU total comes from a table I did not locate; the AMD RDNA2 ISA PDF URL returned an HTML page)**. 2 SIMDs per CU, 72 CUs = 36 WGPs = 144 SIMDs **(unsourced)**. LDS 64 KiB per workgroup / 128 KiB per WGP (wiki silicon pages).

| Block | Waves/WG | VGPR/wave | waves/SIMD by VGPR | LDS/WG | WGs resident (CU mode, by LDS) | Bound |
|---|---|---|---|---|---|---|
| 128 | 4 | 128 | 8 | ≤32 KiB | 144 (2/CU) | LDS |
| 256 | 8 | 128 | 8 | ≤32 KiB | 144 | LDS |
| 256 | 8 | 256 | 4 | ≤32 KiB | 144 (4 waves/SIMD = 1 WG/CU by VGPR → 72) | VGPR → **72** |
| 256 | 8 | any | — | 64 KiB | **72** (1/CU, 2/WGP) | LDS |

Derivation **[unsourced arithmetic]**: waves/SIMD = min(16, ⌊1024/VGPR⌋); a 256-thread WG puts 4 waves on each of a CU's 2 SIMDs. **Safe persistent grid = 72 WGs × 256 threads** (one per CU), which leaves no room for a co-stream (RFC rule).[^rfc] *(historical @ e245a469: K2 targets ≤ 128 VGPRs, 2 WGs / 16 waves per CU, 144 WGs)*[^built]

**Row cap:** gfx1030 has the same 64 KiB per-WG LDS as gfx942, so by #60731's reasoning a GEMV pass holds ~6 rows, not 16.[^60731] #61325's K2 needs up to 68 KB LDS at gfx950 — **over** gfx1030's 64 KiB, so K2 must be re-tiled.[^61325] **With MTP (production recipe `flashnext-mtp2`, k=2):** each request carries 3 query rows, so batch 1/4/8 = 3/12/24 rows and widths must be multiples of 3. #61325 refuses spec decode, so this is beyond AMD's template.[^61325] **[unsourced]** At ~6 rows per 64 KiB GEMV pass, 24 rows = 4 passes per GEMV.

| Mode | Widths | Rows max |
|---|---|---|
| No MTP | 1–8 | 8 |
| MTP k=2 | 3, 6, …, 24 | 24 |

**Proposed cap (without MTP): widths 1–8 (Flash-Next decode without spec), each GEMV in 2 passes above 6 rows [unsourced; to measure].**

## 3. Per-phase map, Flash-Next decode

| Phase | Reuse (dest @ e245a469) | LDS / VGPR | ISA |
|---|---|---|---|
| HC front, norms | `hc_rdna2.cu`, `layernorm.cu` scalar fp32 | small | OK |
| QKV / in_proj (W4A16) | skinny int4 GEMV, `v_dot2_f32_f16` inner loop[^skinny] | LDS holds x rows (`s[max_lds_len]`)[^skinny-lds] | OK (fdot2) |
| QKV (W4A8) | `w4a8_sdot4_rdna2.cuh` `sdot4`[^sdot4] | `lds_bytes(k_per_split)`[^sdot4-lds] | OK (sdot4) |
| GDN recurrent | `gdn_decode_rdna2.cu`, `__launch_bounds__(GDN_THREADS)`[^gdn] | fp32 state | OK |
| QSA attention (every 4th layer) | `fa_rdna2.cu` split-K decode, 256 threads[^fa] | `extern __shared__` + block tables | OK; QSA indexer `qsa_rdna2.cu` |
| o_proj | as QKV | — | OK |
| Router + top-k | scalar fp32 | tiny | OK |
| Shared expert | `moe_resident_decode.cu` fdot2, 4-way K split[^res] | 2× `[4][32]` fp32 | OK |
| Routed experts | resident/skinny W4A16 or `moe_w4a8_rdna2.cu` sdot4[^moe-a8] | `lds_a[BLOCK_M*kKBlock]` | OK |
| AR | `rdna_allreduce.cuh` one-shot | flags + staging | see §5 |

**ISA-blocked on gfx1030:** anything reading AMD's MXFP4/MXFP8 via MFMA or scaled FP8 MFMA (#61325 reads `amd/Qwen3.8-Flash-Next-Quark-MXFP4` with AITER layouts).[^61325] No MFMA, WMMA, FP8 dot, or bf16 dot2 path to use — **[source: wiki silicon/ISA pages]**. We reuse only the contract, not their kernels. Note `skinny_gemms_int4.cu` contains a `__builtin_amdgcn_fdot2_f32_bf16` branch[^skinny-bf16]; that branch must stay unreachable on gfx1030.

## 4. FFN-only first (2a, then 2b)

The RFC's FFN layer: vLLM runs attention with its output all-reduce turned off; the kernel sums the partials and runs the rest.[^rfc] #60397 does this on 10 of 40 layers.[^60397] We split it:

- **2a — compute only.** The kernel runs residual add + norm → router → shared expert (dense fp16 on dest) → routed experts (W4A16/W4A8) → combine. vLLM's existing all-reduce runs **before and after** it, unchanged; the `VLLM_RDNA_AR` lock is untouched. The only flag waits are between workgroups on the **same GPU** (agent scope, §1). In = reduced hidden + residual; out = `(hidden, residual)` unreduced for vLLM's AR. Attention, GDN, conv and QSA stay on dest kernels. FULL graphs only; PIECEWISE refused.[^rfc]
- **2b — in-kernel all-reduce, conditional.** Only if 2a gains on time per output token **and** profiling shows the AR launches are worth removing. 2b moves both ARs inside (§5) and brings the cross-GPU safety layer: bounded waits, sticky error word, phase-ID logging of every timeout.[^rfc][^ar-abort]

## 5. All-reduce over PCIe

Dest already runs in-kernel peer writes: one-shot, uncached staging, a peer announces by one posted P2P store into our local uncached flag slot, we poll locally; bounded by `RDNA_AR_WAIT_MS` = 120 s, sticky abort via `atomicCAS` and a posted host-memory report (no PCIe atomics).[^ar-uc][^ar-wait][^ar-abort] Before announce, `__threadfence_system()`.[^ar-fence]

Risks:
- **Ordering:** data then flag over PCIe relies on posted-write ordering on the same path; RFC calls the remote-fence assumption unguaranteed.[^rfc] Keep data+flag in the same uncached allocation (dest does: flag page appended).[^ar-uc]
- **Timeouts inside a persistent kernel:** 120 s is engine-scale; in a mono step it must be a sticky error that drains every phase (RFC rule).[^rfc]
- **Bus stress:** host-page polling previously coincided with cards dropping off the bus; keep local polling.[^ar-uc]
- **Residency:** AR blocks spin while other WGs work; the grid must leave them resident (§2).

**Recommendation (2b only):** reuse the `rdna_allreduce.cuh` device function as the in-kernel AR stage, unchanged protocol, behind a startup check; full-layer mode only after it passes a fault-injection test.

## Flash-Next arch contract (LLM_Arch_specialist, room 2026-10-13)

- **FFN-only:** only routed experts use W4A16/W4A8 layouts. Router, shared expert and gated residual stay **dense** (not quantized). The official Flash-Next checkpoint stores those tensors as BF16 on disk, but **dest serves them in fp16**: `rdna_fused_glue.cu` accepts `at::kHalf` only on x, weights, scales and activations,[^glue-half] and the #81 TunableOp CSVs are `GemmTunableOp_Half_*` with no BFloat16 rows.[^tunable-half] So the §3 shared-expert `moe_resident_decode.cu` fdot2 loop already matches dest's serving dtype.[^res] Router runs once; combine reuses its ids/weights.
- **Whole layer:** pattern GDN, GDN, GDN, QSA → one spec per layer type (`gdn_layer`, `qsa_layer`). GDN state fp32. QSA KV, compressed keys and indexer share one heap, separate from GDN state. **MTP drafter layers get no mono spec in P1–P3**: they run vLLM's path, as #60397 does for its DSpark draft layers.[^60397]

### Dense GEMV: dest fp16 path vs true bf16

Dest's serving path for the dense router, shared expert and gated residual is **fp16**, not bf16:

- `rdna_fused_glue.cu` checks `at::kHalf` on activations and weights (and fp16 scales for the int8 shadow); there is no bf16 entry point.[^glue-half]
- #81 TunableOp results under `bench_results/2026-09-29_tunableop-rows/curated/tunableop_results{0..3}.csv` (gfx1030, HIP 714): 2,876 `GemmTunableOp_Half_*` rows, **zero** BFloat16 rows; every row selects `Gemm_Rocblas`. Decode-size shapes are present — token counts 1–9, 12, 16 and 24 (covers MTP widths 3/6/12/24).[^tunable-half]

| Deployment | Dense GEMV ISA | Exact vs serving cast? |
|---|---|---|
| **Dest today (fp16)** | `v_dot2_f32_f16` / `__builtin_amdgcn_fdot2` (fdot2), fp32 accumulate | **Yes** — same dtype as dest's rocBLAS fp16 GEMMs and `rdna_fused_glue`[^glue-half][^tunable-half] |
| True bf16 (not dest) | Path A: widen (`v_lshlrev_b32 16` / `v_and_b32 0xffff0000`) + 2× `v_fma_f32` | Yes vs bf16→fp32 bit-exact widen |
| True bf16 via cast (avoid) | bf16→fp16 + `v_dot2_f32_f16` | **No** — fp16 range ±65504 / 10-bit mantissa vs bf16 exponent |

`V_DOT2_F32_BF16` needs `HasDot12Insts` (gfx11/gfx12 encodings only);[^v3p-bf16] gfx1030 has Dot1/2/5/6/7/10, not Dot12.[^td-1030] The dest `__builtin_amdgcn_fdot2_f32_bf16` branch[^skinny-bf16] must stay unreachable on gfx1030. `V_PK_FMA_F32` needs `HasPackedFP32Ops`,[^v3p-pk] absent on gfx1030.[^td-1030]

**Recommendation for MonoKernel on current dest: fdot2 on the fp16 tensors.** Decode-size rocBLAS shapes are already tuned,[^tunable-half] so a custom dense GEMV is a fusion/dispatch win over stock library launches, not a missing-tune gap. Keep the bf16 shift-widen + `v_fma_f32` route only for a true bf16 deployment.

**Numerics on the fp16 path:** fdot2 keeps **fp32 accumulation**, so router logits stay fp32 and the sigmoid/softmax, bias and top-k run once in fp32; combine reuses those ids/weights. Residual and gated-residual activations can exceed the fp16 max (±65504), so the KL check needs long-context runs. GDN state stays fp32. **[unsourced: overflow magnitude on Flash-Next gated residual / residual at long context — confirm with KL on V620]**

## Model Runner V2 dependency

The Flash-Next MonoKernel requires Model Runner V2 (room report); MRV2 on dest is **unproven**. Silicon assumptions that depend on it: FULL-graph capture at the mono widths (capture sizes forced to widths),[^rfc] rank-uniform per-step decision from attention metadata,[^rfc] and MTP row layout (3 rows/request contiguous). If dest stays on V1, P1 must hook the V1 runner's graph capture instead. **(unsourced: the room's MRV2 requirement does not appear in #61325's diff — no `model_runner`/V2 reference in the 14,679-line diff read 2026-10-13. #61325 picks the path in the uncompiled outer `forward`.[^61325] Treat MRV2 as unverified.)**

## 6. Risks and phases

Risks: LDS overflow vs #61325's K2 (68 KB);[^61325] spills (#61325 already spills SGPRs at 8 rows on gfx950);[^61325] co-tenant starvation;[^rfc] PCIe ordering.

1. **P0** — Test cooperative-launch graph capture on ROCm 7.14 and MRV2 on dest; spec + refusals only (MonoSpec for gfx1030, widths 1–8, FULL graphs). No kernel.
2. **P2a** — FFN compute-only kernel at TP4 between vLLM's two ARs; same-GPU flags only; fdot2/sdot4 loops reused.
   **P2b** — in-kernel ARs + sticky error + phase-ID logs, only if P2a gains on TPOT and the profile justifies it. Gate: stage numerics vs vLLM ops, graph replay bit-identical ×300, GSM8K parity (RFC CI).[^rfc]
3. **P2** — K1 (HC + QKV + GDN/QSA front), re-tiled to ≤64 KiB LDS.
4. **P3** — K2 full layer; spec-decode widths later.

## Sources

[^built]: Superseded by dest `rdna_extras` PR #94 merge [`1c1426d585c9b2653265947734225721cfc7ea6e`](https://github.com/opengfx1030/vllm-rdna/commit/1c1426d585c9b2653265947734225721cfc7ea6e); details and line-pinned sources in [monokernel-gfx1030-dest.md](monokernel-gfx1030-dest.md).
[^td-1030]: LLVM [AMDGPU.td:2278–2292](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292) `FeatureISAVersion10_3_0` @ eb61ef62; [AMDGPU.td:451](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L451).
[^bi-vgpr]: LLVM [AMDGPUBaseInfo.cpp:1325–1330](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1325-L1330) @ eb61ef62.
[^bi-occ]: LLVM [AMDGPUBaseInfo.cpp:1356–1363](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1356-L1363) @ eb61ef62.
[^v3p-bf16]: LLVM [VOP3PInstructions.td:832–839, 2960](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/VOP3PInstructions.td#L832-L839) @ eb61ef62.
[^v3p-pk]: LLVM [VOP3PInstructions.td:1549–1550](https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/VOP3PInstructions.td#L1549-L1550) @ eb61ef62.
[^rfc]: vLLM RFC [#60904](https://github.com/vllm-project/vllm/issues/60904) (open; read 2026-10-13).
[^61001]: vLLM PR [#61001](https://github.com/vllm-project/vllm/pull/61001) "[ROCm] MonoSpec + abstractions" (WIP, open).
[^61325]: vLLM PR [#61325](https://github.com/vllm-project/vllm/pull/61325) Qwen3.8-Flash-Next mono decode, MI355X TP2 (open).
[^60731]: vLLM PR [#60731](https://github.com/vllm-project/vllm/pull/60731) DSv4.1 mono on gfx942, "Limits on gfx942".
[^60397]: vLLM PR [#60397](https://github.com/vllm-project/vllm/pull/60397) DSv4.1 mono decode, MI355X (merged 2026-10-10 01:53 UTC).
[^6698]: [ROCm/legacy-rocm-build#6698](https://github.com/ROCm/TheRock/issues/9306) (open, read 2026-10-13; moved to TheRock#9306, link updated 2026-10-23).
[^13234]: [ROCm/rocm-systems#13234](https://github.com/ROCm/rocm-systems/issues/13234) (open, read 2026-10-13).
[^60889]: vLLM PR [#60889](https://github.com/vllm-project/vllm/pull/60889) GLM-5.2 MonoKernel (open, empty description).
[^ockl]: ROCm device-libs [ockl/src/cg.cl](https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/ockl/src/cg.cl) `__ockl_grid_sync` (~L143–154, amd-staging, read 2026-10-13).
[^llvm]: LLVM [AMDGPUUsage, Memory Model GFX10-GFX11](https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx10-gfx11).
[^ar-acq]: [rdna_allreduce.cuh:181, 213–248](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L181-L248) @ e245a469.
[^ar-uc]: [rdna_allreduce.cuh:11–25](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L11-L25) @ e245a469.
[^ar-alloc]: [rdna_allreduce.cu:115–120](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cu#L115-L120) @ e245a469.
[^ar-wait]: [rdna_allreduce.cuh:60–62](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L60-L62) @ e245a469.
[^ar-abort]: [rdna_allreduce.cuh:76–103](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L76-L103); [rdna_allreduce.cu:128–129](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cu#L128-L129) @ e245a469.
[^ar-fence]: [rdna_allreduce.cuh:213–214](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_allreduce.cuh#L213-L214) @ e245a469.
[^skinny]: [skinny_gemms_int4.cu:1040–1084](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/skinny_gemms_int4.cu#L1040-L1084) @ e245a469.
[^skinny-lds]: [skinny_gemms_int4.cu:166](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/skinny_gemms_int4.cu#L166) @ e245a469.
[^skinny-bf16]: [skinny_gemms_int4.cu:92](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/skinny_gemms_int4.cu#L92) @ e245a469.
[^sdot4]: [w4a8_sdot4_rdna2.cuh:125–126](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/w4a8_sdot4_rdna2.cuh#L125-L126) @ e245a469.
[^sdot4-lds]: [w4a8_sdot4_rdna2.cuh:97–100](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/w4a8_sdot4_rdna2.cuh#L97-L100) @ e245a469.
[^gdn]: [gdn_decode_rdna2.cu:53](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/gdn_decode_rdna2.cu#L53) @ e245a469.
[^fa]: [fa_rdna2.cu:776](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/fa_rdna2.cu#L776) @ e245a469.
[^res]: [moe_resident_decode.cu:83–114](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/moe_resident_decode.cu#L83-L114) @ e245a469.
[^moe-a8]: [moe_w4a8_rdna2.cu:130–131, 222–223](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/moe_w4a8_rdna2.cu#L130-L223) @ e245a469.
[^glue-half]: [rdna_fused_glue.cu:366–373](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_fused_glue.cu#L366-L373) (`check_weight`: `at::kHalf` or int8+fp16 scale); host entry points [L447–449](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_fused_glue.cu#L447-L449), [L475–478](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_fused_glue.cu#L475-L478), [L504–505](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_fused_glue.cu#L504-L505), [L527–533](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/csrc/rocm/rdna_fused_glue.cu#L527-L533) — all `at::kHalf`; device dots use `__builtin_amdgcn_fdot2` on `half2` @ e245a4697a139fe8d3bbec4b9c837c37279674e9.
[^tunable-half]: [tunableop_results0.csv](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/bench_results/2026-09-29_tunableop-rows/curated/tunableop_results0.csv) (identical [1](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/bench_results/2026-09-29_tunableop-rows/curated/tunableop_results1.csv)/[2](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/bench_results/2026-09-29_tunableop-rows/curated/tunableop_results2.csv)/[3](https://github.com/opengfx1030/vllm-rdna/blob/e245a4697a139fe8d3bbec4b9c837c37279674e9/bench_results/2026-09-29_tunableop-rows/curated/tunableop_results3.csv)); validators `GCN_ARCH_NAME=gfx1030`, `HIP_VERSION=714`; 4×719 = 2,876 `GemmTunableOp_Half_*` rows, 0 BFloat16, all `Gemm_Rocblas`; shape field `tn_*_*_*` / `nn_*_*_*` includes decode token counts 1–9, 12, 16, 24 @ e245a4697a139fe8d3bbec4b9c837c37279674e9.
