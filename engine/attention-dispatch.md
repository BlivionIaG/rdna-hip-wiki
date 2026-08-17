# fa_rdna2 dispatch / head-64 / short vs split-K

Date: 2026-08-17. Engine slice. Occupancy / LDS / VGPR live in [silicon/fa-occupancy.md](../silicon/fa-occupancy.md). Live tree details from VLLM_FORK_Manager (not all pushed).

`VLLM_USE_RDNA2_FA=1` splits MHA: `fa_rdna2_decode_paged` vs `fa_rdna2_prefill_*`.

## Dispatch table

| Phase | q_len | head_dim | KV length vs 72 CU | Take |
|---|---|---|---|---|
| Decode | 1 | 128 or 256 | grid B×H << 72 | `fa_rdna2_decode_paged` split-K (Br=1) |
| Decode | 1 | 128 or 256 | grid already fills 72 CU | decode paged, fewer splits or no split |
| Decode | 1 | 64 (and 80/96) | any | **hole** → Triton `kernel_paged_attention_2d` until a D=64 tile exists |
| Short extend | small q (2–32) | 128/256 | — | fa_rdna2 short (not the 256-thread decode bounds) |
| Prefill | large q | 128/256 | — | fa_rdna2_prefill Br=16/32 |
| Prefill / decode | any | other | — | Triton |
| MLA decode | 1 | ragged / fp8_ds_mla | 16 CTAs/query, WG=32 | HIP `sparse_mla_decode_rdna2` if `VLLM_USE_RDNA2_MLA=1`, else Triton. Mix/spec off. |
| MLA prefill | T queries | plain fp16 `[skv,D]` | grid `(T, H/4)`, WG=32 | HIP `sparse_mla_prefill_rdna2` @ `66bb24d7` (same env). Many Q rows ≠ fat tile. |
| MLA | q>1 verify / draft | — | — | no fat tile yet. Blocks MTP / DFlash / DSpark. |

Stock HIP `attention.cu` instantiates 64 and 128 but gfx1030 never launches it (`on_gfx1x` = 11/12). gfx11 runtime gate is head==128 only. Head 256: HIP has no instantiation; fa_rdna2 claims D=256 decode (DeepSeek/MLA-adjacent). Keep it.

## Head-64 hole

fa_rdna2 decode/prefill as reported: D=128/256 only. Treat 64 as the first missing FA2 tile after 128/256 are solid. Confirm which served models are actually d=64 (many Llama/Gemma/Mistral 7B are d=128). GQA packing can look like 64.

## Short vs split-K (72 CU V620)

**Locked:** Split-K = grid-fill when B×H leaves 72 CUs idle. Short Br=32 = small-q. Do **not** use D=256 split-K to "fix occupancy." Occupancy is [silicon/fa-occupancy.md](../silicon/fa-occupancy.md): drop the second `__launch_bounds__` arg, add `amdgpu_waves_per_eu(4, 8)` on decode 128/256.

Split-K (Flash-Decoding): extra parallel axis = KV partitions + LSE reduce.

- B=1, GQA (e.g. 8 KV heads): 8 WGs. 16 CTAs/query on MLA is the same idea. Split-K or multi-CTA/query is mandatory.
- B×H ≳ 72: split-K adds reduce traffic and can lose.
- Short (small q, no KV split): wins when q_len is a few tokens (spec/chunk) and the Br tile already occupies, or when seqlen_kv is short enough that one pass is the whole working set (IC-hot pages).
- Prefill Br=16/32: never split-K the Q axis the decode way; occupancy comes from q tiles.

#44899 (DSv4 split-K) is gfx950. Copy the heuristic (`CUs / (B×H)`), not the kernel.

## Sage

Locked. Full page: [sage-attention.md](sage-attention.md).

INT8 QK via `sdot4` (pack 4×i8, `D%4==0`, i32 through K, scale in epilogue). PV stays `fdot2`. **Prefill only**, after occupancy. Decode stays `fa_rdna2`. Absent in the live tree today. Do not port FA3 / Sage2 / Sage3.

## Write order

1. Keep MHA 128/256 decode split-K + prefill Br tiles (already live). Occupancy flip is a ticket, not a direct edit of `perf/rdna2_w4a16`.
2. Head-64 decode tile.
3. Sage-style INT8 QK via sdot4 on prefill only. See [sage-attention.md](sage-attention.md).
4. MLA fat tile (q>1) before mix, MTP, DFlash, or DSpark. HIP sparse prefill (`66bb24d7`) does not unlock this. Specs: [mtp.md](mtp.md), [dflash.md](dflash.md), [dspark.md](dspark.md).

## Unknowns

- Measured crossover B×H where split-K loses on V620.
- Whether any served model is actually d=64.
- Exact fa_rdna2 source not all pushed.
