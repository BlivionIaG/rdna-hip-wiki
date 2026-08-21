# DSv4 Flash 0731 on 4×V620 — silicon

Engine run plan: [../engine/dsv4-flash-run.md](../engine/dsv4-flash-run.md). Occupancy + live W4 still first. mxfp4 GPU smoke still pending (`290715e6`).

**Best kernel we already have is mxfp4, not EXL3, not a fresh INT4.** Leftover “FP8” is **MXFP8**, not W8A16-FP8 group-scale.

## What the bytes actually are

| Tensors | Pack | gfx1030 inner |
|---|---|---|
| 256 routed experts | MXFP4: E2M1 nibble + **E8M0 / 32** | live `mxfp4_dot2_moe`: bit-trick + exp-add + `dot22_8_f` (4× `fdot2` / 8 K) |
| Attn, shared experts, indexer, MTP `main_proj` | MXFP8: E4M3 + **E8M0 / 128×128 block** | leftover cvt: `fp8_e4m3_to_fp16_bits` + **same** `mxfp4_apply_e8m0_bits` (broadcast the block), then `fdot2` |
| Gate / embed / head / norms / `hc_*` | bf16/f32 | host `kFloat16` / existing |

Official QAT also act-quants A to e4m3/ue8m0 per 128. **Skip on first V620 run.** Weight-only unpack is the extras contract. Do not `sdot4` E4M3 bits.

## Decode: expert tile wins, leftover cvt does not

13B active / 284B. Routed experts are ~94% of weight bytes. Per token you fire a handful of mxfp4 expert GEMVs. That kernel is the bandwidth and occupancy problem.

Leftover MXFP8 is skinny `M∈{1,2,4,8}` on MLA projections, the always-on shared expert, and the indexer. Software e4m3 bit-trick is the same VALU class as E2M1 unpack. Those tensors are a few percent of the checkpoint. **They do not close 155 GB → 120 GB** and they are not the decode bottleneck.

| Kernel | Tile | Occupancy |
|---|---|---|
| `mxfp4_dot2_moe` | decode `BLOCK_M=1/2/4/8`, A in LDS, stream B, `BLOCK_KN=256` seed | **must not stay `(1,1)`** |
| leftover MXFP8 dense / shared | same skinny as `q_gemm_w8a16_fp8` / `q_gemm_rdna2` | same occupancy ticket |
| `fa_rdna2` / sparse MLA / indexer | existing HIP | same ticket |

Prefill: leftover cvt is more visible (fat M on attn + shared). Still expert-bound if many tokens hit many experts. Do not grow a third GEMM family — MXFP8 is the FP8 dense path with E8M0 block broadcast instead of a group fp16 scale.

TP=4: write unreduced expert rows (same mxfp4 TP bug as PR #46676).

## Do not requant

- MXFP4 QAT → GPTQ/AWQ INT4+scales: lose QAT, same ~4-bit expert size, no new fit.
- MXFP4 → EXL3/QTIP now: no infer kernel; Viterbi is CUDA-producer ([exl3.md](exl3.md)).
- Leftover MXFP8 → INT4: saves a sliver of the *small* tensors. Not the 120 GB gap.

Fit 120 GB (4×30 usable) by **REAP keeping mxfp4** or **host-offload cold experts**. K216×MXFP4 ~130 GB is an estimate, not measured.

## KV

Existing HIP MLA record, short `max_model_len`. Do not port Spark 432/584 NVFP4/FP8 units. DSpark off until fat tile.

## Done-when (this page)

- ISA dump: expert path is nibble unpack + `v_dot2*`. Leftover is e4m3 bit-trick + e8m0 exp-add + `v_dot2*`. No `sdot4` / `sdot8` / WMMA.
- mxfp4 MoE GPU smoke on a tiny expert tile.
- No tok/s from this page.

## Sources

- Vontra DSv4-Flash MXFP4-MLX card (routed = mxfp4, attn/shared/indexer = mxfp8 e4m3+e8m0/128²)
- 0xSero EXL3 3.0 bpw card: source was packed E2M1+UE8M0; 116.29 GiB; not e2e-validated
- Live extras: `mxfp4_dot2_common.cuh`, `qdq_fp8_rdna2.cuh` @ `290715e6`
- [kernels/mxfp4.md](../kernels/mxfp4.md), [kernels/w8a16-fp8.md](../kernels/w8a16-fp8.md)
