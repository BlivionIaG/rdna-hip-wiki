# vLLM / HIP coverage on gfx1030

Date: 2026-08-22. Progress map. Tip of human branch: **`rdna2_extras`** @ **`d24f6c25`** (read-only). Overlay of vLLM **v0.27.1** + gfx1030 work (`9ff87936` merge). Historical source: `perf/rdna2_w4a16` (stale @ `3baecdb516`). Review: [rdna2-extras.md](rdna2-extras.md). Tickets stay on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not invent tok/s. Do not edit that branch from this page.

Session dump: [notes/session-2026-08-17.md](notes/session-2026-08-17.md).

**Live** = in the fork today. **Must** = HIP we write. **Fallback** = Triton / `torch.nn.functional.linear` / rocBLAS, not a win. **Dead** = no unit, CUDA-only, or wrong physics. **Later** = possible after Must.

Inner ops we actually have: `fdot2` (FP16 256), `sdot4` (IU8 512), `V_DOT8_I32_I4` (IU4 1024). No WMMA / MFMA / FP8 / FP4 / bf16 matrix. `supports_fp8()` and `supports_mx()` are **false** (CDNA / gfx12 / gfx95). `v_dot2_f32_bf16` is RDNA3+ — gfx1030 dots stay fp16.

**Dispatch watch (v0.27.1):** `on_rdna()` is gfx11/12 only. V620 is `on_gfx10x()`. New upstream `on_rdna()` gates skip gfx1030.

## ROCm allowlist ≠ V620 fire list

`rocm.py` still *names* `fp8`, `mxfp4`, `mxfp8`, `bitsandbytes`, `modelopt_fp4`. Those strings are not a ship promise.

What actually HIP-fires: **W4A16**, **W8A16** (i8→half→`fdot2`), **W8A16-FP8** / leftover **MXFP8** cvt, **W8A8-FP8** dense, **mxfp4** unpack. INT8 that’s “not too bad” is **W8A16** — same DOT as W4, twice the bytes. **W8A8 `sdot4`** is the real INT8 compute and is not shipped. INT8 KV is still the scalar `__hmul` kernel ([kv-int8.md](kv-int8.md)). EXL3 HIP is live WIP @ `a2c8d5cf` (compile fix `e268c7d3`; unmarked 2/3/4-bit 3inst → `fdot2`; mul1/mcg-marked fold fp16). `d24f6c25` FULL graphs are rebase glue, not this list.

Silicon contracts: [kernels/](../kernels/README.md). `sdot4` explore: [kernels/sdot4-explore.md](../kernels/sdot4-explore.md) (W8A8, Sage QK, W4A8). W4A4 integer: [w4a4.md](w4a4.md). Dispatch: [attention-dispatch.md](attention-dispatch.md), [sage-attention.md](sage-attention.md). NVFP4: [nvfp4.md](nvfp4.md). INT8 KV: [kv-int8.md](kv-int8.md). INT2: [int2.md](int2.md). MTP / DFlash / DSpark: [mtp.md](mtp.md), [dflash.md](dflash.md), [dspark.md](dspark.md). Native MoE: [fp16-moe.md](fp16-moe.md), [int8-moe.md](int8-moe.md). Stock baselines: [baseline-order.md](baseline-order.md), [triton-rocm.md](triton-rocm.md). FlyDSL: [flydsl.md](flydsl.md). DeepEP: [deepep.md](deepep.md). ROCmFPX: [rocmfpx.md](rocmfpx.md). EXL3: [exl3.md](exl3.md). DSv4 run: [dsv4-flash-run.md](dsv4-flash-run.md).

## Weight × activation GEMM

| Scheme | Inner op | Status | Notes |
|---|---|---|---|
| FP16 × FP16 | `fdot2` / rocBLAS | Fallback | Stock linear on gfx1030 is `F.linear` / rocBLAS. `wvSplitK` / `LLMM1` require `on_gfx9() or on_gfx1x()`. `VLLM_ROCM_USE_SKINNY_GEMM` is a **no-op** here. [kernels/triton-skinny-gemm.md](../kernels/triton-skinny-gemm.md) |
| **W4A16** | dequant → `fdot2` | **Live** | Dense + MoE. [kernels/w4a16.md](../kernels/w4a16.md) |
| **W8A16** | i8→half → `fdot2` | **Live** | “INT8 that’s not too bad.” Same DOT as W4, 2× bytes. Not IU8. Do not call this W8A8. |
| **W8A16-FP8** | LUT → `fdot2` | **Live** | Shipping. Weight storage is FP8-looking; compute is still LUT+`fdot2`. Leftover DSv4 MXFP8 cvt is this family. |
| **W8A8-FP8** (dense) | FP8 bytes → fp16 bit-trick → `fdot2` | **Live** | `750ca545`. Act dequant at LDS staging, not inner loop. Not `sdot4`. Not Instinct FP8 MMA. GPU verify pending per commit. |
| **W8A8** INT8×INT8 | `sdot4`, i32 through K, scale epilogue | Must / explore | Prefill 64×64×64 i8. [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md) + [sdot4-explore.md](../kernels/sdot4-explore.md). No `sudot4`. |
| **Native HIP FP16 MoE** | `fdot2` | Must / queued | New unquantized backend. Decode skinny `M∈{1,2,4,8}` + prefill 64×64×32. Not a rewrite of `moe_q_gemm_rdna2`. [fp16-moe.md](fp16-moe.md) |
| **Native HIP INT8 MoE** | W8A16 `fdot2` / W8A8 `sdot4` | Must / queued | Dual route. Dispatch by `(rows/expert, K, N, scale)`. Never call W8A16 “INT8 compute.” [int8-moe.md](int8-moe.md) |
| **mxfp4** (E2M1+UE8M0) | unpack → `fdot2` | **Live sources** | `290715e6` + `d67577d6`. RDNA2_Researcher: `mxfp4_dot2_common.cuh` is 4× `fdot2` over 8 K. GPU smoke pending per commit. |
| **NVFP4** (E2M1+E4M3×16) | unpack → `fdot2` | Must / queued | Same E2M1 LUT as mxfp4; scale is E4M3/16 + optional FP32. Spec: [nvfp4.md](nvfp4.md). Not Blackwell MMA. |
| **INT2 / W2A16** | unpack 16×i2 → `fdot2` | Must / queued | No i2 DOT. Spec: [int2.md](int2.md) + [kernels/int2.md](../kernels/int2.md). Not INT2 KV. |
| **Mixed INT2/INT4 MoE** | two unpackers, one `fdot2` | Must / queued | Bitwidth grouped outside K. Same [int2.md](int2.md). |
| W4A8 | W4→i8 + `sdot4` A8 | Later / explore | After W8A8. `sdot8` is W4A4 only. [sdot4-explore.md](../kernels/sdot4-explore.md) |
| **W4A4 integer** i4×i4 | `sdot8`, i32 through K | Later / explore | After W8A8. Prefill first; decode stays W4A16 unless measured. Spec: [w4a4.md](w4a4.md). Not E2M1. |
| W4A4-native / MXFP4-native | FP4 MMA | **Dead** | No FP4 unit. Quark/NVFP4 W4A4: dequant A to fp16 or refuse. Do not `sdot8` E2M1. |
| FP8 W8A8 / PTPC-FP8 (Instinct) | FP8 MMA | **Dead** | No FP8 unit. Do not confuse with W8A8-FP8 `fdot2` above. |
| AWQ / GPTQ / WNA16 Triton | Triton | Fallback | **Do not rewrite.** Dispatch to `q_gemm_rdna2` / `moe_q_gemm_rdna2`. |
| compressed-tensors W8A8 INT8 | CUDA / CDNA | Must (same as W8A8 sdot4) | Format is the checkpoint; kernel is sdot4. |
| Marlin / Machete / FlashInfer | CUDA MMA | **Dead** | Includes vendor NVFP4. |
| bitsandbytes | CUDA | **Dead** on this box | Official AMD column is ❌ |
| GGUF (stock Q4_0 / K / IQ*) | vLLM loader / plugin | Live loader | Not ggml MMVQ. Custom ROCmFPX types are not this. [rocmfpx.md](rocmfpx.md) |
| ROCmFPX (`Q4_0_ROCMFP4` …) | codebook → `perm` → `sdot4` | **No vLLM port** | llama.cpp side project only. [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md) |
| **EXL3** (QTIP trellis) | 3-inst codebook → `fdot2` | **Live WIP** @ `a2c8d5cf` / tip `e268c7d3` | Produce `3inst`, compile `mcg`, no `mul1`. HIP kernel for unmarked 2/3/4-bit; marked mul1/mcg fold fp16. Occupancy still first. [exl3.md](exl3.md). |
| Ternary / BitNet 1.58 | LUT or pack + `V_DOT8`? | Later | No ternary unit. Research after DOT kernels exist. Not a first ticket. |

## Attention

| Path | Status | Notes |
|---|---|---|
| `fa_rdna2` FA2 + `fdot2` D=128/256 | **Live** | Prefill Br=16/32, decode paged split-K. Occupancy ticket stands. |
| Occupancy flip | Ticket | `fa_rdna2` **and** `skinny_gemms.cu`. Same trap: `amdgpu_waves_per_eu(1, 1)` / HIP second `launch_bounds` arg. Fix: drop min-blocks, `amdgpu_waves_per_eu(4, 8)` on decode-class kernels. |
| FULL cudagraphs | **Live glue** @ `d24f6c25` | `_cudagraph_support = ALWAYS` on RDNA_ATTN. Recaptures the same `(1,1)` kernel. Rebase-must-keep, not occupancy. |
| Sage INT8 QK `sdot4` | Ticket / Must | Prefill only. **v2 write #2** after occupancy. [sage-attention.md](sage-attention.md) + [kernels/sage-qk.md](../kernels/sage-qk.md) |
| Head-64 FA2 tile | Ticket / Must | Triton hole. **v2 write #3** after Sage. |
| Short vs split-K | Ticket (blocked) | Fill vs LDS, not occupancy |
| HIP paged-decode (`attention.cu`) | **Dead** stock | `on_gfx1x` = gfx11/12. fa_rdna2 is the replacement |
| Stock `TRITON_ATTN` | Fallback / baseline | Cleanest gfx1030 FA baseline. `--dtype half`. [triton-flash-attention.md](triton-flash-attention.md) |
| Stock `ROCM_ATTN` on gfx1030 | Fallback / baseline | Triton prefill **and** Triton `kernel_paged_attention_2d`. Native `paged_attention_rocm` is CDNA or gfx11/12. Do **not** call this HIP-decode. [triton-rocm.md](triton-rocm.md) |
| Skinny GEMM / `skinny_gemms.cu` | **Live sources** | In-tree fork. Occupancy ticket, not a from-scratch Must. Stock gfx1030 is BLAS (row above). |
| `reshape_and_cache` match | Must | Writer for the layout we gather. INT8 writer is [kv-int8.md](kv-int8.md). |
| AITER / CK FA / shuffle | **Dead** | CDNA |
| FA3 / Sage2 / Sage3 | **Dead** | Hopper / INT4 / FP4 |
| MLA sparse (Triton fp16) | **Live** | Default if env unset. Mix/spec off. |
| MLA sparse HIP decode | Opt-in **silicon-ok** | `9a344444`: fp16 q/out, H-generic. ISA: scalar fp32 FMA after FP8 unpack, not `fdot2`. `VLLM_USE_RDNA2_MLA=1`. Decode `fdot2` after unpack is gone. |
| MLA sparse HIP prefill | Opt-in **on branch** | `66bb24d7`: `sparse_mla_prefill_rdna2`. `__launch_bounds__(32)` (no `(1,1)` trap). Grid `(T, H/4)`, 14+2 split, scalar FMA. KV plain fp16 `[skv,512]`. **OOB in `load_row`** (gate before `fdot2`). First place `fdot2` pays (both sides already half). Not fat tile `q>1`. |
| Indexer HIP radix top-k | Opt-in **on branch** | `8496f4ca`: `ops.top_k_per_row_decode` replaces `torch.topk` on the RDNA2 indexer decode path. |
| MLA fat tile q>1 | Must / Later | Before mix, MTP, DFlash, DSpark. HIP prefill does not unlock this. |
| MTP | Later / queued | Native heads. Fat tile first. [mtp.md](mtp.md) |
| DFlash | Later / queued | Parallel block draft. Same gate. [dflash.md](dflash.md) |
| DSpark | Later / queued | DFlash + Markov + confidence. Same gate. [dspark.md](dspark.md) |
| INT8 KV + fused dequant | Must / queued | `int8_per_token_head` only. Fused into `fa_rdna2`. Spec: [kv-int8.md](kv-int8.md). After occupancy. `4cc1fe59` is **not** this. |
| FP8 KV | **Dead** as a vLLM dtype path | `supports_fp8()` is false. Software fuse on extras is occupancy-blocked. |
| INT4 KV | Later | After INT8 KV |

## Collectives / engine glue

| Path | Status | Notes |
|---|---|---|
| gfx1030/gfx1100 all-reduce bypass | **Live** @ `3e05abc9` | `RocmPlatform.use_custom_op_collectives()` returns False (ROCm-wide) → PYNCCL `_all_reduce_out_place`. Transport only. Same intent as `0c59068e`. Verified TP=4 Qwen3.6-35B-A3B-FP16 on the historical branch. |
| RCCL inside MoE GEMM | **Dead** | Engine owns `combine_mode`. Default under TP/EP = unreduced routed rows. `3e05abc9` does not change that. [moe.md](moe.md) |
| DeepEP / IBGDA / MORI | **Later** | Steal asymmetry only. On 4×V620 the analogue is GPU-initiated PCIe P2P if peer access is real. [deepep.md](deepep.md) |
| FlyDSL as a backend | Gate 0 | Compiler may emit gfx1030. Every shipped fast GEMM/MoE/FA is MFMA or gfx11/12 WMMA. [flydsl.md](flydsl.md) |

## Triton → HIP (compile tax)

Triton is fine for bring-up. Autotune/compile makes it slow to *use*. Stock map first: [baseline-order.md](baseline-order.md). Tuning knobs only: [triton-tuning.md](triton-tuning.md).

| Still Triton | HIP replacement | When |
|---|---|---|
| Sparse MLA Triton decode | HIP MLA decode fp16 @ `9a344444` | Env flip is silicon-ok. `fdot2` after unpack is gone. |
| Sparse MLA Triton prefill | HIP prefill @ `66bb24d7` | Env flip same as decode. **OOB first**, then `fdot2` (both sides half). |
| `kernel_paged_attention_2d` (head-64) | Head-64 `fa_rdna2` tile | v2 write #3 |
| Triton prefill `_fwd_page` / `_fwd_kernel` | Extend `fa_rdna2_prefill_*` | After occupancy |
| Triton AWQ / GPTQ / WNA16 / MoE | `q_gemm_rdna2` / `moe_q_gemm_rdna2` | **Do not rewrite those GEMMs** |
| Triton INT8 reshape / attn | HIP writer + `fa_rdna2` gather | After occupancy. [kv-int8.md](kv-int8.md) |

Do not HIP-rewrite unused Triton (AITER FA, FA3, Marlin).

## v2 write order (locked with silicon)

1. Occupancy flip: `fa_rdna2` + `skinny_gemms.cu`. Current subject. **Not fixed by the extras rebase or by `d24f6c25` graphs.**
2. Sage QK prefill (`sdot4`).
3. Head-64 paged HIP.
4. Then: W8A8 `sdot4`, INT8 KV, MLA fat tile, W4A8, NVFP4 unpack→`fdot2`. HIP MLA: **OOB on prefill `load_row`**, then `fdot2` on prefill first (both sides half). Decode `fdot2` after FP8 unpack is gone. Note on the MLA card, not this list.
5. After fat tile: MTP / DFlash / DSpark (engine only — no new DOT). INT2 / mixed MoE is a GEMM unpack, not this attention list.
6. After W8A8: integer W4A4 `sdot8` ([w4a4.md](w4a4.md)). MXFP4/NVFP4 A4 is not this.

Native HIP FP16 / INT8 MoE sit after occupancy (and after a measured stock baseline). Never rewrite `q_gemm_rdna2` / `moe_q_gemm_rdna2`. Never: Instinct FP8 MMA, FA3, Marlin, AITER, W4A4-native FP4 MMA, streaming decode-KV over PCIe.

## Progress

Locked 2026-08-22: human branch is **`rdna2_extras`** @ **`d24f6c25`**. Overlay merge `9ff87936` onto v0.27.1. `4cc1fe59` INT8 KV is not the fused contract. `d24f6c25` FULL graphs are glue. Occupancy still first. Allowlist ≠ fire list. EXL3 HIP live WIP ([exl3.md](exl3.md); tip `e268c7d3`, ISA `a2c8d5cf`). Occupancy still first. Engine tickets live on GitHub project 4 (V620 inference); this page is the index.
