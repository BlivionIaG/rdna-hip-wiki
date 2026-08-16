# vLLM / HIP coverage on gfx1030

Date: 2026-08-17. Progress map. Not a ticket dump. Tickets stay on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not invent tok/s. Do not edit `perf/rdna2_w4a16` from this page.

**Live** = in the fork today. **Must** = HIP we write. **Fallback** = Triton / `torch.nn.functional.linear` / rocBLAS, not a win. **Dead** = no unit, CUDA-only, or wrong physics. **Later** = possible after Must.

Inner ops we actually have: `fdot2` (FP16 256), `sdot4` (IU8 512), `V_DOT8_I32_I4` (IU4 1024). No WMMA / MFMA / FP8 / FP4 / bf16 matrix. `supports_fp8()` is false.

Silicon contracts: [kernels/](../kernels/README.md). Dispatch: [attention-dispatch.md](attention-dispatch.md), [sage-attention.md](sage-attention.md).

## Weight × activation GEMM

| Scheme | Inner op | Status | Notes |
|---|---|---|---|
| FP16 × FP16 | `fdot2` / rocBLAS | Fallback | Stock linear. Skinny decode GEMV is a Must (gated `on_gfx1x`). |
| **W4A16** | dequant → `fdot2` | **Live** | Dense + MoE. [kernels/w4a16.md](../kernels/w4a16.md) |
| **W8A16** / W8A16-FP8 | LUT → `fdot2` | **Live** | Not the IU8 path. Do not call this W8A8. |
| **W8A8** | `sdot4`, i32 through K, scale epilogue | Must | [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md). Tile seed 64×64×64. No `sudot4`. |
| **mxfp4** (E2M1+E8M0) | unpack → `fdot2` | Must | No FP4 unit. `supports_mx()` is gfx950. |
| W4A8 | W4→i8 + `sdot4` A8 | Later | After W8A8. Same IU8 pipe, half the weight bytes. |
| W4A4 / NVFP4 / MXFP4-native | — | **Dead** | No FP4 unit. Quark W4A4: dequant A to fp16 or refuse. |
| FP8 W8A8 / PTPC-FP8 | — | **Dead** | No FP8 unit. Instinct path. |
| AWQ W4A16 | Triton dequant + GEMM | Fallback | Forced-build path. `VLLM_USE_TRITON_AWQ=1`. Same math as W4A16; win is a HIP kernel, not the checkpoint name. |
| GPTQ W4A16 / W8A16 | Triton / gfx1100 HIP | Fallback | `gptq_gemm_rdna3` is gfx1100. PR #52391 only opens Triton W4A16 via `on_gfx10x()`. |
| compressed-tensors WNA16 | ExllamaLinear / Triton | Fallback | ROCm enable is “not Marlin” (#27187). Same W4A16/W8A16 math. |
| compressed-tensors W8A8 INT8 | CUDA / CDNA | Must (same as W8A8) | Format is the checkpoint; kernel is sdot4. |
| Marlin / Machete / FlashInfer | CUDA MMA | **Dead** | |
| bitsandbytes | CUDA | **Dead** on this box | Official AMD column is ❌ |
| GGUF | llama.cpp HIP | Out of vLLM | Steal MMVQ (fused dequant+dot). |
| Ternary / BitNet 1.58 | LUT or pack + `V_DOT8`? | Later | No ternary unit. Research after DOT kernels exist. Not a first ticket. |

## Attention

| Path | Status | Notes |
|---|---|---|
| `fa_rdna2` FA2 + `fdot2` D=128/256 | **Live** | Prefill Br=16/32, decode paged split-K |
| Occupancy flip | Ticket | Drop second `launch_bounds`, `amdgpu_waves_per_eu(4,8)` |
| Head-64 FA2 tile | Ticket / Must | Triton hole |
| Short vs split-K | Ticket (blocked) | Fill vs LDS, not occupancy |
| Sage INT8 QK `sdot4` | Ticket / Must | Prefill only. [sage-attention.md](sage-attention.md) + [kernels/sage-qk.md](../kernels/sage-qk.md) |
| HIP paged-decode (`attention.cu`) | **Dead** stock | `on_gfx1x` = gfx11/12. fa_rdna2 is the replacement |
| Skinny GEMM / `wvSplitK` | Must | Decode QKV+FFN. Gated off gfx10 |
| `reshape_and_cache` match | Must | Writer for the layout we gather |
| AITER / CK FA / shuffle | **Dead** | CDNA |
| FA3 / Sage2 / Sage3 | **Dead** | Hopper / INT4 / FP4 |
| MLA sparse q=1 | **Live** | Mix/spec off |
| MLA fat tile q>1 | Must / Later | Before mix or MTP |
| INT8 KV + fused dequant | Must / Later | Capacity. Unfused erases the win. After paged-decode is solid |
| FP8 KV | **Dead** | `supports_fp8()` false |
| INT4 KV | Later | After INT8 KV |

## Engine features (not new GEMM ISA)

| Feature | Status |
|---|---|
| Continuous batching + chunked prefill | Use. Measure MBT. |
| Prefix cache | Use. First novel token ends the prefix. |
| Specdec | Later. Need q>1 / short-extend. |
| PD disagg | Skip as a win. |
| TP-for-speed | Skip. Capacity only. |
| MoE HIP grouped/skinny | Must (same W4/W8 kernels, token-major, small BLOCK_M) |
| Triton MoE | **Dead** whitelist (PR #37826 excludes gfx10xx) |
| AITER MoE | **Dead** |

## Write order (engine)

1. Occupancy flip (ticket).
2. Keep live W4A16 / W8A16 / fa_rdna2 128/256.
3. Head-64 FA2.
4. Skinny GEMM + matching cache writer.
5. W8A8 `sdot4` dense+MoE.
6. Sage QK prefill.
7. mxfp4 unpack.
8. INT8 KV fused into paged-decode.
9. W4A8, then maybe ternary / INT4 KV.

Never: FP8, FA3, Marlin, AITER, W4A4-native, streaming decode-KV over PCIe.

## Progress

Update the Status column when something lands. VLLM_FORK_Manager owns the board; this page is the map.
