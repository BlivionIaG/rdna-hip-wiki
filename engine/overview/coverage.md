# vLLM / HIP coverage on gfx1030

> **Dest today (2026-10-11):** [`opengfx1030/vllm-rdna` `rdna_extras`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras) = vLLM **v0.31.0** port ([PR #47](https://github.com/opengfx1030/vllm-rdna/pull/47)), tip `fbe84e94` (contains `8abf4c97`; engine code claims are pinned at `1dc3bac4` unless a footnote says otherwise, since dest #52 changed only PLE Python/tools/tests). Tips and branch names quoted below (`rdna2_extras` on BlivionIaG/vllm, `perf/rdna2_w4a16`) are **historical snapshots**. Serve locks (ROCm pin, all-reduce path, prefix caching) live in [fork/serve-locks.md](../../fork/serve-locks.md)10)#10) and [toolchain/pin/README.md](../../toolchain/pin/README.md); this page only says what the engine does.

Date: 2026-08-22. Progress map. Tip of human branch: **`rdna2_extras`** @ **`d24f6c25`** (read-only). Overlay of vLLM **v0.27.1** + gfx1030 work (`9ff87936` merge). Historical source: `perf/rdna2_w4a16` (stale @ `3baecdb516`). Review: [rdna2-extras.md](../serving-stack/rdna2-extras.md). Tickets stay on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not invent tok/s. Do not edit that branch from this page.

Session dump: [notes/session-2026-08-17.md](../notes/session-2026-08-17.md).

**Live** = in the fork today. **Must** = HIP we write. **Fallback** = Triton / `torch.nn.functional.linear` / rocBLAS, not a win. **Dead** = no unit, CUDA-only, or wrong physics. **Later** = possible after Must.

Inner ops we actually have: `fdot2` (FP16 256), `sdot4` (IU8 512), `V_DOT8_I32_I4` (IU4 1024). No WMMA / MFMA / FP8 / FP4 / bf16 matrix. `supports_fp8()` and `supports_mx()` are **false** (CDNA / gfx12 / gfx95). `v_dot2_f32_bf16` is RDNA3+ — gfx1030 dots stay fp16.

**Dispatch watch (v0.27.1):** `on_rdna()` is gfx11/12 only. V620 is `on_gfx10x()`. New upstream `on_rdna()` gates skip gfx1030. At `1dc3bac4` (v0.31.0) upstream `on_rdna()` is still gfx11/12; the dest adds `on_rdna_family()` (gfx10x or RDNA3/4) and `on_rdna2()` (gfx10x) in `vllm/platforms/rdna.py`, and fork behaviour in shared code is gated on them. Example: the EXL3 fp16 cast used `on_rdna()` and never fired on gfx1030; it now uses `on_rdna_family()`[^d-rdnapy][^d-exl3cast]

## ROCm allowlist ≠ V620 fire list

`rocm.py` still *names* `fp8`, `mxfp4`, `mxfp8`, `bitsandbytes`, `modelopt_fp4`. Those strings are not a ship promise.

What actually HIP-fires: **W4A16**, **W8A16** (i8→half→`fdot2`), **W8A16-FP8** / leftover **MXFP8** cvt, **W8A8-FP8** dense, **mxfp4** unpack. INT8 that’s “not too bad” is **W8A16** — same DOT as W4, twice the bytes. **W8A8 `sdot4`** is the real INT8 compute and is not shipped. INT8 KV is still the scalar `__hmul` kernel ([kv-int8.md](../kv-cache/kv-int8.md)). EXL3 HIP is live WIP @ `a2c8d5cf` (compile fix `e268c7d3`; unmarked 2/3/4-bit 3inst → `fdot2`; mul1/mcg-marked fold fp16). `d24f6c25` FULL graphs are rebase glue, not this list.

Silicon contracts: [kernels/](../../kernels/README.md). `sdot4` explore: [kernels/quant/sdot4-explore.md](../../kernels/quant/sdot4-explore.md) (W8A8, Sage QK, W4A8). W4A4 integer: [w4a4.md](../quant-formats/w4a4.md). Dispatch: [attention-dispatch.md](../attention/attention-dispatch.md), [sage-attention.md](../attention/sage-attention.md). NVFP4: [nvfp4.md](../quant-formats/nvfp4.md). INT8 KV: [kv-int8.md](../kv-cache/kv-int8.md). INT2: [int2.md](../quant-formats/int2.md). MTP / DFlash / DSpark: [mtp.md](../spec-decoding/mtp.md), [dflash.md](../spec-decoding/dflash.md), [dspark.md](../spec-decoding/dspark.md). Native MoE: [fp16-moe.md](../moe/fp16-moe.md), [int8-moe.md](../moe/int8-moe.md). Stock baselines: [baseline-order.md](../stock-baseline/baseline-order.md), [triton-rocm.md](../stock-baseline/triton-rocm.md). FlyDSL: [flydsl.md](../alt-engines/flydsl.md). DeepEP: [deepep.md](../multi-gpu/deepep.md). ROCmFPX: [rocmfpx.md](../quant-formats/rocmfpx.md). EXL3: [exl3.md](../quant-formats/exl3.md). DSv4 run: [dsv4-flash-run.md](../run-recipes/dsv4-flash-run.md).

## Weight × activation GEMM

| Scheme | Inner op | Status | Notes |
|---|---|---|---|
| FP16 × FP16 | `fdot2` / rocBLAS | Fallback | Stock linear on gfx1030 is `F.linear` / rocBLAS. `wvSplitK` / `LLMM1` require `on_gfx9() or on_gfx1x()`. `VLLM_ROCM_USE_SKINNY_GEMM` is a **no-op** here. [kernels/gemm-moe/triton-skinny-gemm.md](../../kernels/gemm-moe/triton-skinny-gemm.md) |
| **W4A16** | dequant → `fdot2` | **Live** | Dense + MoE. [kernels/quant/w4a16.md](../../kernels/quant/w4a16.md) |
| **W8A16** | i8→half → `fdot2` | **Live** | “INT8 that’s not too bad.” Same DOT as W4, 2× bytes. Not IU8. Do not call this W8A8. |
| **W8A16-FP8** | LUT → `fdot2` | **Live** dense; MoE **not built** @ `1dc3bac4` | Dense `q_gemm_w8a16_fp8_rdna2.cu` is compiled; `moe_w8a16_fp8_rdna2.cu` is excluded from the gfx1030 build, so FP8-weight MoE experts fall back. *(2026-08-22: "Shipping".)*[^d-cmake] Weight storage is FP8-looking; compute is still LUT+`fdot2`. Leftover DSv4 MXFP8 cvt is this family. |
| **W8A8-FP8** (dense) | FP8 bytes → fp16 bit-trick → `fdot2` | **Live** | `750ca545`. Act dequant at LDS staging, not inner loop. Not `sdot4`. Not Instinct FP8 MMA. GPU verify pending per commit. |
| **W8A8** INT8×INT8 | `sdot4`, i32 through K, scale epilogue | Must / explore | Prefill 64×64×64 i8. [kernels/quant/w8a8-mxfp4.md](../../kernels/quant/w8a8-mxfp4.md) + [sdot4-explore.md](../../kernels/quant/sdot4-explore.md). No `sudot4`. |
| **Native HIP FP16 MoE** | `fdot2` | Must / queued | New unquantized backend. Decode skinny `M∈{1,2,4,8}` + prefill 64×64×32. Not a rewrite of `moe_q_gemm_rdna2`. [fp16-moe.md](../moe/fp16-moe.md) |
| **Native HIP INT8 MoE** | W8A16 `fdot2` / W8A8 `sdot4` | Must / queued | Dual route. Dispatch by `(rows/expert, K, N, scale)`. Never call W8A16 “INT8 compute.” [int8-moe.md](../moe/int8-moe.md) |
| **mxfp4** (E2M1+UE8M0) | unpack → `fdot2` | **Fixed, unit-tested** @ `1dc3bac4`; DSv4 serve pending | On the 0.28 line every call was rejected and the kernels never ran; three dequant bugs were fixed in `523eb846e0` and `test_mxfp4_rdna2` passes 50/50. History: `290715e6` + `d67577d6`.[^d-port-mxfp4] RDNA2_Researcher: `mxfp4_dot2_common.cuh` is 4× `fdot2` over 8 K. GPU smoke pending per commit. |
| **NVFP4** (E2M1+E4M3×16) | unpack → `fdot2` | Must / queued | Same E2M1 LUT as mxfp4; scale is E4M3/16 + optional FP32. Spec: [nvfp4.md](../quant-formats/nvfp4.md). Not Blackwell MMA. |
| **INT2 / W2A16** | unpack 16×i2 → `fdot2` | Must / queued | No i2 DOT. Spec: [int2.md](../quant-formats/int2.md) + [kernels/quant/int2.md](../../kernels/quant/int2.md). Not INT2 KV. |
| **Mixed INT2/INT4 MoE** | two unpackers, one `fdot2` | Must / queued | Bitwidth grouped outside K. Same [int2.md](../quant-formats/int2.md). |
| W4A8 | W4→i8 + `sdot4` A8 | **Live opt-in** (dense prefill) @ `1dc3bac4` | `VLLM_RDNA2_W4A8_SDOT4=1` (recipe `W4A8=1`): dense W4A16 prefill GEMMs with M ≥ 33 run int8 activations on `v_dot4`; decode stays W4A16; off by default. `moe_w4a8_rdna2.cu` is added only when `VLLM_GPU_ARCHES` includes gfx1030, but into the shared `VLLM_ROCM_EXT_SRC`, so a mixed build compiles it for every arch (`8abf4c97` fixed that build). Only the `__gfx1030__` device pass has the kernel body; other archs get an empty stub. The host guard `on_gfx1030()` (reads `gcnArchName`) is the first condition of `eligible`, otherwise `fallback()` runs before any launch[^d-w4a8moe]. MoE W4A8 in serving is not validated. *(2026-08-22: "Later / explore, after W8A8".)*[^d-port-w4a8][^d-cmake] `sdot8` is W4A4 only. [sdot4-explore.md](../../kernels/quant/sdot4-explore.md) |
| **W4A4 integer** i4×i4 | `sdot8`, i32 through K | Later / explore | After W8A8. Prefill first; decode stays W4A16 unless measured. Spec: [w4a4.md](../quant-formats/w4a4.md). Not E2M1. |
| W4A4-native / MXFP4-native | FP4 MMA | **Dead** | No FP4 unit. Quark/NVFP4 W4A4: dequant A to fp16 or refuse. Do not `sdot8` E2M1. |
| FP8 W8A8 / PTPC-FP8 (Instinct) | FP8 MMA | **Dead** | No FP8 unit. Do not confuse with W8A8-FP8 `fdot2` above. |
| AWQ / GPTQ / WNA16 Triton | Triton | Fallback | **Do not rewrite.** Dispatch to `q_gemm_rdna2` / `moe_q_gemm_rdna2`. |
| compressed-tensors W8A8 INT8 | CUDA / CDNA | Must (same as W8A8 sdot4) | Format is the checkpoint; kernel is sdot4. |
| Marlin / Machete / FlashInfer | CUDA MMA | **Dead** | Includes vendor NVFP4. |
| bitsandbytes | CUDA | **Dead** on this box | Official AMD column is ❌ |
| GGUF (stock Q4_0 / K / IQ*) | vLLM loader / plugin | Live loader | Not ggml MMVQ. Custom ROCmFPX types are not this. [rocmfpx.md](../quant-formats/rocmfpx.md) |
| ROCmFPX (`Q4_0_ROCMFP4` …) | codebook → `perm` → `sdot4` | **No vLLM port** | llama.cpp side project only. [llamacpp-rocmfpx.md](../alt-engines/llamacpp-rocmfpx.md) |
| **EXL3** (QTIP trellis) | 3-inst codebook → `fdot2` | **Live** @ `1dc3bac4` (kernels built RDNA-generic, gfx1030 + gfx1100; `27b-exl3` recipe); history `a2c8d5cf` / `e268c7d3` | Produce `3inst`, compile `mcg`, no `mul1`. HIP kernel for unmarked 2/3/4-bit; marked mul1/mcg fold fp16. Occupancy still first. [exl3.md](../quant-formats/exl3.md). |
| Ternary / BitNet 1.58 | LUT or pack + `V_DOT8`? | Later | No ternary unit. Research after DOT kernels exist. Not a first ticket. |

## Attention

| Path | Status | Notes |
|---|---|---|
| `fa_rdna2` FA2 + `fdot2` D=128/256 | **Live**, opt-in | Prefill Br=16/32, decode paged split-K. Selected only with `VLLM_USE_RDNA2_FA=1` (env default off; the launcher's `ATTN=fa` sets it). Decode occupancy landed; prefill ticket stands. |
| Occupancy flip | **Decode landed** @ `1dc3bac4`; prefill + skinny still Ticket[^dest-occupancy] | `fa_rdna2` paged decode D=128/256 now `amdgpu_waves_per_eu(4, 8)`; prefill kernels, including INT8 split-K and GQA prefill, still `__launch_bounds__(N, 1)`; `skinny_gemms.cu` `wvSplitKrc_` still `amdgpu_waves_per_eu(1, 1)`. Original note: Same trap: `amdgpu_waves_per_eu(1, 1)` / HIP second `launch_bounds` arg. Fix: drop min-blocks, `amdgpu_waves_per_eu(4, 8)` on decode-class kernels. |
| FULL cudagraphs | **Upstream capture** @ `1dc3bac4` | Recipes run `FULL_AND_PIECEWISE`. On 2026-10-10 the RDNA replay-copy `CUDAGraphWrapper`, the piecewise split at TP collectives and the capture-freeze flags (`rdna2_freeze_capture_persist` etc.) were removed; RDNA now uses upstream capture/replay. *(2026-08-22: "Live glue @ `d24f6c25`, recaptures the `(1,1)` kernel".)* |
| Sage INT8 QK `sdot4` | Ticket / Must | Prefill only. **v2 write #2** after occupancy. [sage-attention.md](../attention/sage-attention.md) + [kernels/attention/sage-qk.md](../../kernels/attention/sage-qk.md) |
| Head-64 FA2 tile | Ticket / Must | Triton hole. **v2 write #3** after Sage. |
| Short vs split-K | Ticket (blocked) | Fill vs LDS, not occupancy |
| HIP paged-decode (`attention.cu`) | **Dead** stock | `on_gfx1x` = gfx11/12. fa_rdna2 is the replacement |
| Stock `TRITON_ATTN` | Fallback / baseline | Cleanest gfx1030 FA baseline. `--dtype half`. [triton-flash-attention.md](../stock-baseline/triton-flash-attention.md) |
| Stock `ROCM_ATTN` on gfx1030 | Fallback / baseline | Triton prefill **and** Triton `kernel_paged_attention_2d`. Native `paged_attention_rocm` is CDNA or gfx11/12. Do **not** call this HIP-decode. [triton-rocm.md](../stock-baseline/triton-rocm.md) |
| Skinny GEMM / `skinny_gemms.cu` | **Live sources** | In-tree fork. Occupancy ticket, not a from-scratch Must. Stock gfx1030 is BLAS (row above). |
| `reshape_and_cache` match | Must | Writer for the layout we gather. INT8 writer is [kv-int8.md](../kv-cache/kv-int8.md). |
| AITER / CK FA / shuffle | **Dead** | CDNA |
| FA3 / Sage2 / Sage3 | **Dead** | Hopper / INT4 / FP4 |
| MLA sparse (Triton fp16) | Fallback @ `1dc3bac4` | Used only when the RDNA HIP MLA ops are not registered. *(2026-08-22: "Live, default if env unset".)* |
| MLA sparse HIP decode | **Default on RDNA** @ `1dc3bac4` (no env gate on the DSv4 path; kernel in `csrc/rocm/rdna/dsv4/`)[^d-dsv4path]. *(2026-08-22: opt-in, `VLLM_USE_RDNA2_MLA=1`.)* | `9a344444`: fp16 q/out, H-generic. ISA: scalar fp32 FMA after FP8 unpack, not `fdot2`. `VLLM_USE_RDNA2_MLA=1`. Decode `fdot2` after unpack is gone. |
| MLA sparse HIP prefill | **Default on RDNA** @ `1dc3bac4` (same selection as decode)[^d-dsv4path]. *(2026-08-22: opt-in on branch.)* | `66bb24d7`: `sparse_mla_prefill_rdna2`. `__launch_bounds__(32)` (no `(1,1)` trap). Grid `(T, H/4)`, 14+2 split, scalar FMA. KV plain fp16 `[skv,512]`. **OOB in `load_row`** (gate before `fdot2`). First place `fdot2` pays (both sides already half). Not fat tile `q>1`. |
| Indexer HIP radix top-k | Opt-in **on branch** | `8496f4ca`: `ops.top_k_per_row_decode` replaces `torch.topk` on the RDNA2 indexer decode path. |
| MLA fat tile q>1 | Must / Later | Before mix, MTP, DFlash, DSpark. HIP prefill does not unlock this. |
| MTP | Later / queued | Native heads. Fat tile first. [mtp.md](../spec-decoding/mtp.md) |
| DFlash | Later / queued | Parallel block draft. Same gate. [dflash.md](../spec-decoding/dflash.md) |
| DSpark | Later / queued | DFlash + Markov + confidence. Same gate. [dspark.md](../spec-decoding/dspark.md) |
| INT8 KV + fused dequant | Must / queued | `int8_per_token_head` only. Fused into `fa_rdna2`. Spec: [kv-int8.md](../kv-cache/kv-int8.md). After occupancy. `4cc1fe59` is **not** this. |
| FP8 KV | **Dead** as a vLLM dtype path | `supports_fp8()` is false. Software fuse on extras is occupancy-blocked. |
| INT4 KV | Later | After INT8 KV |

## Collectives / engine glue

| Path | Status | Notes |
|---|---|---|
| gfx1030/gfx1100 all-reduce bypass | **Live** @ `3e05abc9` *(historical; current path: [fork/serve-locks.md](../../fork/serve-locks.md))* | `RocmPlatform.use_custom_op_collectives()` returns False (ROCm-wide) → PYNCCL `_all_reduce_out_place`. Transport only. Same intent as `0c59068e`. Verified TP=4 Qwen3.6-35B-A3B-FP16 on the historical branch. |
| RCCL inside MoE GEMM | **Dead** | Engine owns `combine_mode`. Default under TP/EP = unreduced routed rows. `3e05abc9` does not change that. [moe.md](../moe/moe.md) |
| DeepEP / IBGDA / MORI | **Later** | Steal asymmetry only. On 4×V620 the analogue is GPU-initiated PCIe P2P if peer access is real. [deepep.md](../multi-gpu/deepep.md) |
| FlyDSL as a backend | Gate 0 | Compiler may emit gfx1030. Every shipped fast GEMM/MoE/FA is MFMA or gfx11/12 WMMA. [flydsl.md](../alt-engines/flydsl.md) |

## Triton → HIP (compile tax)

Triton is fine for bring-up. Autotune/compile makes it slow to *use*. Stock map first: [baseline-order.md](../stock-baseline/baseline-order.md). Tuning knobs only: [triton-tuning.md](../stock-baseline/triton-tuning.md).

| Still Triton | HIP replacement | When |
|---|---|---|
| Sparse MLA Triton decode | HIP MLA decode fp16 @ `9a344444` | No env flip at `1dc3bac4`: HIP decode is selected whenever the RDNA ops are registered[^d-dsv4path]. `fdot2` after unpack is gone. |
| Sparse MLA Triton prefill | HIP prefill @ `66bb24d7` | Same selection as decode (no env flip at `1dc3bac4`)[^d-dsv4path]. **OOB first** *(status unverified at tip)*, then `fdot2` (both sides half). |
| `kernel_paged_attention_2d` (head-64) | Head-64 `fa_rdna2` tile | v2 write #3 |
| Triton prefill `_fwd_page` / `_fwd_kernel` | Extend `fa_rdna2_prefill_*` | After occupancy |
| Triton AWQ / GPTQ / WNA16 / MoE | `q_gemm_rdna2` / `moe_q_gemm_rdna2` | **Do not rewrite those GEMMs** |
| Triton INT8 reshape / attn | HIP writer + `fa_rdna2` gather | After occupancy. [kv-int8.md](../kv-cache/kv-int8.md) |

Do not HIP-rewrite unused Triton (AITER FA, FA3, Marlin).

## v2 write order (locked with silicon)

1. Occupancy flip: `fa_rdna2` + `skinny_gemms.cu`. *(2026-08-22 text: "Current subject. Not fixed by the extras rebase or by `d24f6c25` graphs.")* At `1dc3bac4` the decode half has landed; prefill and `wvSplitKrc_` remain[^dest-occupancy].
2. Sage QK prefill (`sdot4`).
3. Head-64 paged HIP.
4. Then: W8A8 `sdot4`, INT8 KV, MLA fat tile, W4A8, NVFP4 unpack→`fdot2`. HIP MLA: **OOB on prefill `load_row`**, then `fdot2` on prefill first (both sides half). Decode `fdot2` after FP8 unpack is gone. Note on the MLA card, not this list.
5. After fat tile: MTP / DFlash / DSpark (engine only — no new DOT). INT2 / mixed MoE is a GEMM unpack, not this attention list.
6. After W8A8: integer W4A4 `sdot8` ([w4a4.md](../quant-formats/w4a4.md)). MXFP4/NVFP4 A4 is not this.

Native HIP FP16 / INT8 MoE sit after occupancy (and after a measured stock baseline). Never rewrite `q_gemm_rdna2` / `moe_q_gemm_rdna2`. Never: Instinct FP8 MMA, FA3, Marlin, AITER, W4A4-native FP4 MMA, streaming decode-KV over PCIe.

## Progress

Locked 2026-08-22 *(historical)*: human branch was **`rdna2_extras`** @ **`d24f6c25`**. Overlay merge `9ff87936` onto v0.27.1. `4cc1fe59` INT8 KV is not the fused contract. `d24f6c25` FULL graphs are glue. Occupancy still first. Allowlist ≠ fire list. EXL3 HIP live WIP ([exl3.md](../quant-formats/exl3.md); tip `e268c7d3`, ISA `a2c8d5cf`). Occupancy still first. Engine tickets live on GitHub project 4 (V620 inference); this page is the index.

## Sources

Linked inline on this page (each URL checked 2026-10-11):

- <https://github.com/users/BlivionIaG/projects/4>

Added in the 2026-10-11 sourcing pass:

- dest `docs/rdna2/v0.31.0-port-notes.md` — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/v0.31.0-port-notes.md>
- dest `csrc/rocm/q_gemm_rdna2.cu` (W4A16 dequant → fdot2) — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/csrc/rocm/q_gemm_rdna2.cu>
- dest `csrc/rocm/moe_q_gemm_rdna2.cu` — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/csrc/rocm/moe_q_gemm_rdna2.cu>
- dest `csrc/rocm/skinny_gemms.cu` — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/csrc/rocm/skinny_gemms.cu>
- dest `csrc/rocm/fa_rdna2.cu` — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/csrc/rocm/fa_rdna2.cu>
- dest `csrc/rocm/q_gemm_rdna3_wmma.cu` (gfx1100-only WMMA path, separate fatbin target) — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/csrc/rocm/q_gemm_rdna3_wmma.cu>
- vLLM `vllm/platforms/rocm.py` (arch gates, `supports_fp8`, custom paged-attn, custom AR) — <https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py>
- AMD, *RDNA 2 Shader Instruction Set Architecture* (ISA 70648) — <https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf>
- LLVM AMDGPU backend usage (processor feature table, dot insts per target) — <https://llvm.org/docs/AMDGPUUsage.html>

[^dest-occupancy]: dest occupancy state at `1dc3bac4` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L283-L284>, <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L535-L536> (decode); <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L1054> (prefill); <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/skinny_gemms.cu#L1344-L1346> (`wvSplitKrc_`); INT8 / GQA prefill — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L2447>, <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L2695>, <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L3810>
[^d-rdnapy]: dest `vllm/platforms/rdna.py` (`on_rdna_family`, `on_rdna2`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/platforms/rdna.py>
[^d-exl3cast]: dest `vllm/model_executor/models/qwen3_5.py` `_exl3_needs_fp16_cast()` uses `on_rdna_family()` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/model_executor/models/qwen3_5.py#L108-L114>
[^d-cmake]: dest `CMakeLists.txt` gfx1030 source list (incl. the `moe_w8a16_fp8_rdna2.cu` exclusion) and RDNA-generic EXL3 list at `1dc3bac4` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1517-L1572>. The gfx1030 CMake block appends to the shared `VLLM_ROCM_EXT_SRC`, which is compiled for every arch in `VLLM_GPU_ARCHES`, so these files also compile for every arch in a mixed build — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1577-L1583>
[^d-port-mxfp4]: dest v0.31.0 port notes, "MXFP4 kernels — fixed `523eb846e0`" — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L509-L523>
[^d-port-w4a8]: dest v0.31.0 port notes, "W4A8 prefill (opt-in)" — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L279-L297>
[^d-recipes]: dest recipes at `1dc3bac4` — <https://github.com/opengfx1030/vllm-rdna/tree/1dc3bac4cda3201a1b4fb345e044578e84a272cf/tools/rdna/recipes>
[^d-port-replay]: dest v0.31.0 port notes, "Replay-copy wrapper removed (2026-10-10)" — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L772-L829>
[^d-faopt]: dest `vllm/envs.py` `VLLM_USE_RDNA2_FA` (default `"False"`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/envs.py#L1442-L1444>; `vllm/platforms/rocm.py` RDNA_ATTN pin — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/platforms/rocm.py#L1134-L1147>; launcher `rdna_select_attention` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/tools/rdna/rdna_launcher_common.sh#L106-L118>
[^d-w4a8moe]: dest `csrc/rocm/moe_w4a8_rdna2.cu` at `1dc3bac4`: kernel body only in the `__gfx1030__` device pass — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/moe_w4a8_rdna2.cu#L58>; empty stub for other archs — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/moe_w4a8_rdna2.cu#L271>; host `on_gfx1030()` reads `gcnArchName` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/moe_w4a8_rdna2.cu#L312-L322>; first condition of `eligible` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/moe_w4a8_rdna2.cu#L487-L488>; `fallback()` before any launch — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/moe_w4a8_rdna2.cu#L501-L503>. CMake adds it only when `VLLM_GPU_ARCHES` matches gfx1030 — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1530> — but into the shared `VLLM_ROCM_EXT_SRC` compiled for all `VLLM_GPU_ARCHES` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1577-L1583>; mixed-arch build fixed by `8abf4c97` — <https://github.com/opengfx1030/vllm-rdna/commit/8abf4c9719d588349b6ed5afc75e9d0133a43984>
[^d-dsv4path]: dest at `1dc3bac4`: `deepseek_v4/amd/model.py` `_attention_cls()` picks `DeepseekV4RDNAAttention` on `on_rdna_family()` when both `sparse_mla_decode_rdna2` and `sparse_mla_prefill_rdna2` are registered (no env check) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/models/deepseek_v4/amd/model.py#L226-L239>; op presence checks — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/models/deepseek_v4/amd/rdna/ops.py#L24-L37>; HIP decode/prefill routing — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/models/deepseek_v4/amd/rdna/attention.py#L3-L22>; kernels in `csrc/rocm/rdna/dsv4/` (decode L251, prefill L581 of `sparse_mla_decode.cu`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna/dsv4/sparse_mla_decode.cu#L251>, <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna/dsv4/sparse_mla_decode.cu#L581>; indexer `__builtin_amdgcn_fdot2` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu#L179>, <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu#L415>; `VLLM_USE_RDNA2_MLA` only in the generic wrapper — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/v1/attention/ops/rocm_rdna2_mla_sparse.py#L44>
