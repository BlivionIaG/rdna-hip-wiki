---
title: "INT4 KV cache with FP16 block scales (int4_fp16_block) on gfx1030 — read-path silicon notes"
sources:
  - https://github.com/opengfx1030/vllm-rdna/pull/138
  - https://github.com/opengfx1030/vllm-rdna/commit/2d461709e2bf864f85b23ede1f5171f122f13407
  - https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/int4_kv_layout_common.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/fa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/qsa_rdna2.cu
  - https://github.com/BlivionIaG/rdna-hip-wiki/pull/29
  - https://github.com/vllm-project/vllm/pull/41394
  - https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/moe_resident_decode.cu
---
# INT4 KV cache with FP16 block scales on gfx1030: read-path silicon notes

State at dest `rdna_extras` @ `2d461709` (merge of #138, 2026-10-25 13:12 UTC+1), checked 2026-10-25.[^c-2d461709] The format is opt-in; defaults are unchanged.[^pr138]

## Layout

One record per (block, slot, head): `D/2` payload bytes (two INT4 codes per byte, low nibble first) followed by `D/32` uint16 scale words. Each scale word is `(fp16_bits(scale) & 0xfff0) | zp`: an FP16 scale with its low 4 mantissa bits cleared, plus a 4-bit zero-point. One scale covers 32 contiguous head-dim channels. Dequant is `(code - zp) * scale`, clamped to the FP16 range. The scale words are a view on the tail of the same pages, not a separate allocation.[^f-layout]

The header warns that a kernel reading these bytes as FP16 gets finite, plausible, silently wrong numbers instead of a fault, so every consumer must identify the layout from the descriptor, never from a dtype.[^f-layout]

## What holds: FP16 LDS tile, fdot2 QK

Both readers turn INT4 back into FP16 while staging K/V into the same FP16 LDS tile the FP16 path fills, so the softmax and dot code is shared:

| Kernel | Staging | QK | PV |
|---|---|---|---|
| `fa_rdna2.cu` (RDNA_ATTN decode/prefill) | `int4_eight_from_rows` when `vector8`, else `int4_pair_from_rows`[^f-fa-stage] | `__builtin_amdgcn_fdot2` on half2 from LDS[^f-fa-qk] | scalar FP32 FMA (`sP[k] * __half2float(sV[...])`)[^f-fa-qk] |
| `qsa_rdna2.cu` (non-mono QSA sparse decode) | `int4_eight_from_rows` when `vector8`, else `int4_half_from_rows`[^f-qsa-stage] | four `fdot2` per 16-byte LDS read[^f-qsa-qk] | scalar FP32 FMA per half[^f-qsa-pv] |

So this format does not fall back to a scalar FP32 QK dot.

## Note 1: the unpack goes through FP32

Every helper does, per pair of channels: extract the nibbles, `float(int(code) - zp) * scale` in FP32 for each, clamp each to ±65504 with `fminf`/`fmaxf`, then `__floats2half2_rn`.[^f-pair][^f-eight] From the source, that is roughly 2 integer subtracts, 2 int-to-float converts, 2 FP32 multiplies, 4 min/max and a pack per pair (a source-level count, not an ISA dump).

The W4A16 GEMM path already uses the cheaper FP16 trick: OR the nibble into `0x6400` to get `half(1024 + q)`, then one packed `hfma2`/`hsub2` folds the zero-point and scale ([w4a16.md](../quant/w4a16.md)).[^vpr-41394][^f-moe-resident] Applying it here should cut the unpack to a few packed FP16 ops per pair. The clamp is only needed if `(15 - zp) * scale` can exceed 65504, which the writer could rule out instead. (unsourced: no gfx1030 measurement that the INT4 staging is VALU- or load-bound.)

## Note 2: when the 8-wide load is not legal

`make_layout` sets `vector8` only when the payload channel stride is 1, the data pointer is 4-byte aligned and the block/slot/head strides are multiples of 4.[^f-vector8] Then each thread loads one dword of payload plus one scale word per 8 channels.[^f-eight]

Otherwise:
- `fa_rdna2` falls back to `int4_pair_from_rows`, one byte load plus one scale-word load per 2 channels.[^f-pair][^f-fa-stage]
- QSA falls back to `int4_half_from_rows`, one channel at a time.[^f-qsa-stage]

Either fallback re-reads the same scale word up to 16 or 32 times per 32-channel group. `fa_rdna2` has the 8-wide path too; only its fallback is 2-wide.

## Status

Not on the board. VLLM_FORK_Manager will add it only if a V620 measurement shows the INT4 KV path is load-bound or VALU-bound in staging. The staging writes into LDS through VGPRs; the direct VMEM-to-LDS alternative and its limits (dword-or-narrower, no dwordx4, one M0 write per load) are covered in the open [wiki PR #29](https://github.com/BlivionIaG/rdna-hip-wiki/pull/29). Direct LDS loads can't dequantize, so they don't apply to this format unless the tile stays INT4 in LDS.

## Sources

[^pr138]: opengfx1030/vllm-rdna [PR #138](https://github.com/opengfx1030/vllm-rdna/pull/138), "feat/kv-int4-fp16-scales".
[^c-2d461709]: opengfx1030/vllm-rdna commit [`2d461709e2bf864f85b23ede1f5171f122f13407`](https://github.com/opengfx1030/vllm-rdna/commit/2d461709e2bf864f85b23ede1f5171f122f13407) (2026-10-25), "Merge pull request #138 from opengfx1030/feat/kv-int4-fp16-scales".
[^f-layout]: [`csrc/rocm/int4_kv_layout_common.cuh` L4-L29 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/int4_kv_layout_common.cuh#L4-L29).
[^f-pair]: [`int4_kv_layout_common.cuh` L122-L149 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/int4_kv_layout_common.cuh#L122-L149) (`int4_pair_from_rows`).
[^f-eight]: [`int4_kv_layout_common.cuh` L151-L180 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/int4_kv_layout_common.cuh#L151-L180) (`int4_eight_from_rows`).
[^f-vector8]: [`int4_kv_layout_common.cuh` L192-L210 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/int4_kv_layout_common.cuh#L192-L210) (`make_layout`, `vector8`).
[^f-fa-stage]: [`csrc/rocm/fa_rdna2.cu` L412-L450 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/fa_rdna2.cu#L412-L450); same pattern at L723-L760 for the 256 variant.
[^f-fa-qk]: [`fa_rdna2.cu` L547-L591 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/fa_rdna2.cu#L547-L591).
[^f-qsa-stage]: [`csrc/rocm/qsa_rdna2.cu` L1069-L1120 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/qsa_rdna2.cu#L1069-L1120).
[^f-qsa-qk]: [`qsa_rdna2.cu` L1168-L1197 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/qsa_rdna2.cu#L1168-L1197).
[^f-qsa-pv]: [`qsa_rdna2.cu` L1240-L1257 @ `2d461709`](https://github.com/opengfx1030/vllm-rdna/blob/2d461709e2bf864f85b23ede1f5171f122f13407/csrc/rocm/qsa_rdna2.cu#L1240-L1257).
[^vpr-41394]: vLLM [PR #41394](https://github.com/vllm-project/vllm/pull/41394), native W4A16 kernel for AMD RDNA3 (gfx1100), nibble `| 0x64006400` dequant.
[^f-moe-resident]: opengfx1030/vllm-rdna `csrc/rocm/moe_resident_decode.cu` @ [`e131562936f9c6f27c26449b00aaaeeb34159b4c`](https://github.com/opengfx1030/vllm-rdna/blob/e131562936f9c6f27c26449b00aaaeeb34159b4c/csrc/rocm/moe_resident_decode.cu) (`0x64006400` magic, `0x64086408` bias), also summarized in [skinny-gemm.md](../gemm-moe/skinny-gemm.md).
