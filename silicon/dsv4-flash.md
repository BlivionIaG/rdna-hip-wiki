# DSv4 Flash 0731 on 4×V620 — silicon

Engine: [../engine/dsv4-flash-run.md](../engine/dsv4-flash-run.md). Occupancy still first. Pair: [exl3.md](exl3.md).

**Destination kernel is QTIP/EXL3 HIP on routed experts**, K216-class ~100 GiB so 120 GB still has KV. Producer input is **official QAT MXFP4**, not Int4-FP8. Leftover stays official **MXFP8**.

## Packs

| Tensors | Official QAT | Int4-FP8 (side) | Dest HIP |
|---|---|---|---|
| Routed experts | MXFP4 E2M1 + E8M0/32 | W4A16 i4 + group-32 (second quant of those nibbles) | EXL3 16×16 + `decode_3inst` → `fdot2` |
| Attn / shared / indexer / `main_proj` | MXFP8 e4m3 + **e8m0 / 128²** | e4m3 + **fp32** 128² blocks | keep official MXFP8 cvt (`e4m3` bit-trick + `mxfp4_apply_e8m0_bits`) |
| Gate / embed / head / norms / `hc_*` | bf16/f32 | bf16 | unchanged |

Int4-FP8 leftover is **not** official MXFP8 (fp32 block scale vs e8m0). 164.96 GiB full / 118.25 GiB REAP-216 — no KV on 120 GB. Do **not** Viterbi it.

Skip official act-quant (A→e4m3/ue8m0) on the first V620 run. Do not `sdot4` E4M3 or EXL3 halves.

## Same skinny tile

EXL3 expert GEMM is `mxfp4_dot2_moe` with a fatter unpack (`decode_3inst` vs E2M1+E8M0). Decode `BLOCK_M=1/2/4/8`, A in LDS, stream B, `BLOCK_KN=256` seed, 4×`fdot2`/8K after half exists. **Occupancy flip on that MoE launch is the EXL3 occupancy flip.** Do not grow a second `(1,1)` tree.

Leftover MXFP8 stays skinny like `q_gemm_w8a16_fp8`. Few percent of bytes. Not the 155→100 GiB cut.

TP=4: unreduced expert rows (PR #46676).

## Produce

1. REAP (or equivalent) **on official MXFP4** to K216-class, **then** CUDA Viterbi 3–4 bpw `mcg`/`mul1`. Don’t spend Viterbi on experts you will drop.
2. Cornell/ExLlama produce. We consume. One HIP kernel for a QTIP dump or EXL3 ([exl3.md](exl3.md)).
3. 0xSero 256-expert 116.29 GiB is the wrong size (no KV). 99.48 GiB K216 is the size target, not the runtime.

## KV

Existing HIP MLA, short ctx. No Spark 432/584. DSpark off until fat tile.

## Done-when (when the Later kernel lands)

- ISA dump: expert path is `decode_3inst` + `v_dot2*`. Leftover is e4m3 + e8m0 exp-add + `v_dot2*`. No MMA / `sdot*`.
- One named QAT→EXL3 checkpoint bit-matches codebook id.
- No tok/s from this page.

## Sources

- Official 0731 QAT (Vontra MXFP4-MLX split)
- [BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8](https://huggingface.co/BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8) — W4A16 of MXFP4, 164.96 GiB
- 0xSero K216 EXL3 99.48 GiB (size only)
- extras `mxfp4_dot2_common.cuh` / `qdq_fp8_rdna2.cuh`
- [kernels/exl3.md](../kernels/exl3.md), [kernels/mxfp4.md](../kernels/mxfp4.md)
