# vLLM / HIP coverage on gfx1030

Date: 2026-08-17. Progress map. Tip of human branch: `perf/rdna2_w4a16` @ `add17dd7` (read-only). Tickets stay on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not invent tok/s. Do not edit that branch from this page.

**Live** = in the fork today. **Must** = HIP we write. **Fallback** = Triton / `torch.nn.functional.linear` / rocBLAS, not a win. **Dead** = no unit, CUDA-only, or wrong physics. **Later** = possible after Must.

Inner ops we actually have: `fdot2` (FP16 256), `sdot4` (IU8 512), `V_DOT8_I32_I4` (IU4 1024). No WMMA / MFMA / FP8 / FP4 / bf16 matrix. `supports_fp8()` is false. `v_dot2_f32_bf16` is RDNA3+ — gfx1030 dots stay fp16.

Silicon contracts: [kernels/](../kernels/README.md). Dispatch: [attention-dispatch.md](attention-dispatch.md), [sage-attention.md](sage-attention.md). NVFP4 spec: [nvfp4.md](nvfp4.md).

## Weight × activation GEMM

| Scheme | Inner op | Status | Notes |
|---|---|---|---|
| FP16 × FP16 | `fdot2` / rocBLAS | Fallback | Stock linear. Skinny decode GEMV is in-tree — see Attention table. |
| **W4A16** | dequant → `fdot2` | **Live** | Dense + MoE. [kernels/w4a16.md](../kernels/w4a16.md) |
| **W8A16** | LUT → `fdot2` | **Live** | Not the IU8 path. Do not call this W8A8. |
| **W8A16-FP8** | LUT → `fdot2` | **Live** | Shipping. Weight storage is FP8-looking; compute is still LUT+`fdot2`. |
| **W8A8-FP8** (dense) | FP8 bytes → fp16 bit-trick → `fdot2` | **Live** | `750ca545`. Act dequant at LDS staging, not inner loop. Not `sdot4`. Not Instinct FP8 MMA. GPU verify pending per commit. |
| **W8A8** INT8×INT8 | `sdot4`, i32 through K, scale epilogue | Must | [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md). Tile seed 64×64×64. No `sudot4`. |
| **mxfp4** (E2M1+UE8M0) | unpack → `fdot2` | **Live sources** | `290715e6` + `d67577d6`. RDNA2_Researcher: `mxfp4_dot2_common.cuh` is 4× `fdot2` over 8 K. GPU smoke pending per commit. |
| **NVFP4** (E2M1+E4M3×16) | unpack → `fdot2` | Must / queued | Same E2M1 LUT as mxfp4; scale is E4M3/16 + optional FP32. Spec: [nvfp4.md](nvfp4.md). Not Blackwell MMA. |
| W4A8 | W4→i8 + `sdot4` A8 | Later | After W8A8 sdot4. Same IU8 pipe, half the weight bytes. |
| W4A4-native / MXFP4-native | FP4 MMA | **Dead** | No FP4 unit. Quark/NVFP4 W4A4: dequant A to fp16 or refuse. |
| FP8 W8A8 / PTPC-FP8 (Instinct) | FP8 MMA | **Dead** | No FP8 unit. Do not confuse with W8A8-FP8 `fdot2` above. |
| AWQ / GPTQ / WNA16 Triton | Triton | Fallback | **Do not rewrite.** Dispatch to `q_gemm_rdna2` / `moe_q_gemm_rdna2`. |
| compressed-tensors W8A8 INT8 | CUDA / CDNA | Must (same as W8A8 sdot4) | Format is the checkpoint; kernel is sdot4. |
| Marlin / Machete / FlashInfer | CUDA MMA | **Dead** | Includes vendor NVFP4. |
| bitsandbytes | CUDA | **Dead** on this box | Official AMD column is ❌ |
| GGUF | llama.cpp HIP | Out of vLLM | Steal MMVQ (fused dequant+dot). |
| Ternary / BitNet 1.58 | LUT or pack + `V_DOT8`? | Later | No ternary unit. Research after DOT kernels exist. Not a first ticket. |

## Attention

| Path | Status | Notes |
|---|---|---|
| `fa_rdna2` FA2 + `fdot2` D=128/256 | **Live** | Prefill Br=16/32, decode paged split-K. Occupancy ticket stands. |
| Occupancy flip | Ticket | `fa_rdna2` **and** `skinny_gemms.cu`. Same trap: `amdgpu_waves_per_eu(1, 1)` / HIP second `launch_bounds` arg. Fix: drop min-blocks, `amdgpu_waves_per_eu(4, 8)` on decode-class kernels. |
| Sage INT8 QK `sdot4` | Ticket / Must | Prefill only. **v2 write #2** after occupancy. [sage-attention.md](sage-attention.md) + [kernels/sage-qk.md](../kernels/sage-qk.md) |
| Head-64 FA2 tile | Ticket / Must | Triton hole. **v2 write #3** after Sage. |
| Short vs split-K | Ticket (blocked) | Fill vs LDS, not occupancy |
| HIP paged-decode (`attention.cu`) | **Dead** stock | `on_gfx1x` = gfx11/12. fa_rdna2 is the replacement |
| Skinny GEMM / `skinny_gemms.cu` | **Live sources** | In-tree. Occupancy ticket, not a from-scratch Must. |
| `reshape_and_cache` match | Must | Writer for the layout we gather |
| AITER / CK FA / shuffle | **Dead** | CDNA |
| FA3 / Sage2 / Sage3 | **Dead** | Hopper / INT4 / FP4 |
| MLA sparse (Triton fp16) | **Live** | `add17dd7`. Default DSv4 path. Mix/spec off. |
| MLA sparse HIP | Opt-in **blocked** | `sparse_mla_rdna2.cu` is bf16 in/out. Do **not** set `VLLM_USE_RDNA2_MLA=1` until that kernel is fp16. Not a promote. |
| MLA fat tile q>1 | Must / Later | Before mix or MTP |
| INT8 KV + fused dequant | Must / Later | After paged-decode is solid |
| FP8 KV | **Dead** as a vLLM dtype path | `supports_fp8()` false. |
| INT4 KV | Later | After INT8 KV |

## Triton → HIP (compile tax)

Triton is fine for bring-up. Autotune/compile makes it slow to *use*.

| Still Triton | HIP replacement | When |
|---|---|---|
| Sparse MLA Triton (default) | HIP MLA in tree, **bf16 today** | Blocked until HIP is fp16. Not a promote. |
| `kernel_paged_attention_2d` (head-64) | Head-64 `fa_rdna2` tile | v2 write #3 |
| Triton prefill `_fwd_kernel` | Extend `fa_rdna2_prefill_*` | After occupancy |
| Triton AWQ / GPTQ / WNA16 / MoE | `q_gemm_rdna2` / `moe_q_gemm_rdna2` | **Do not rewrite those GEMMs** |

Do not HIP-rewrite unused Triton (AITER FA, FA3, Marlin).

## v2 write order (locked with silicon)

1. Occupancy flip: `fa_rdna2` + `skinny_gemms.cu`. Current subject.
2. Sage QK prefill (`sdot4`).
3. Head-64 paged HIP.
4. Then: W8A8 `sdot4`, INT8 KV, MLA fat tile, W4A8, NVFP4 unpack→`fdot2`. HIP MLA fp16 dtype flip is a later rewrite, not this list.

Never rewrite `q_gemm_rdna2` / `moe_q_gemm_rdna2`. Never: Instinct FP8 MMA, FA3, Marlin, AITER, W4A4-native, streaming decode-KV over PCIe. Never flip `VLLM_USE_RDNA2_MLA=1` while the HIP kernel is bf16.

## Progress

Locked 2026-08-17 with RDNA2_Researcher v2 order. HIP MLA gate locked the same day: default Triton fp16; `VLLM_USE_RDNA2_MLA=1` blocked until HIP is fp16. NVFP4 spec queued: [nvfp4.md](nvfp4.md). VLLM_FORK_Manager owns the board; this page is the index.
