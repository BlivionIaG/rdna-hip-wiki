# DeepSeek-V4-Flash on 4	imesV620 (120 GB) — quality destination

Date: 2026-08-22. Engine contract. Human branch **`rdna2_extras`**. Occupancy still first. Pair with [exl3.md](exl3.md), [dspark.md](dspark.md), [silicon/dsv4-flash.md](../silicon/dsv4-flash.md).

**Goal (user lock):** maximize quality with **QTIP/EXL3** and a **native HIP** kernel that runs it. Size target is **K216-class ~100 GiB** so 4×30 GB (120 GB) still has KV room. Official 0731 QAT MXFP4/MXFP8 is the *producer input*, not the all-GPU serve. Int4-FP8 is a side checkpoint, not the quality path.

## Order (locked)

**REAP, then Viterbi.** Do not Viterbi first.

1. Start from **official QAT**. Leftover attn/shared/indexer **MXFP8** is never converted.
2. **REAP** on those official **MXFP4 experts** (keep ~216/256, remap router). Experts stay MXFP4 through this step.
3. **Viterbi / EXL3** only the *retained* experts (3–4 bpw, `mcg`/`mul1` + 16×16). CUDA producer. Do not spend the trellis on experts we drop.
4. HIP infer: `mxfp4_dot2_moe` with a fatter unpack (`decode_3inst` → half → `fdot2`). Occupancy flip on that launch is the EXL3 flip.

Do **not** MXFP4 → Int4/W4, then EXL3. Do **not** Viterbi all 256 then REAP. Do **not** Viterbi Int4-FP8.

## Checkpoints (measured)

| Artifact | What | Size | Role |
|---|---|---|---|
| Official 0731 QAT | MXFP4 experts (E2M1+UE8M0). Leftover is **MXFP8** (e4m3 + e8m0/128²), not W8A16-FP8. | ~155–167 GB | **Convert source.** |
| [BlivionIaG Int4-FP8](https://huggingface.co/BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8) | W4A16 of those MXFP4 nibbles + leftover e4m3+fp32-block. 46 shards. | **164.96 GiB** | Side path. Don’t Viterbi it. |
| [BlivionIaG Int4-FP8 REAP-216B](https://huggingface.co/BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8-REAP-216B) | Same Int4 + 216 experts. | **118.25 GiB** | No KV. Not the dest. |
| 0xSero EXL3 3.0 bpw | Full-expert EXL3. Not e2e-validated. | **116.29 GiB** | Wrong order (Viterbi before prune). |
| 0xSero / MiaAI Spark **K216 EXL3** | 216/256, top-k 6. | **99.48 GiB** | **Size target.** Runtime dead here. |

## Destination (locked)

| Piece | Take |
|---|---|
| Quality | QTIP/EXL3 3–4 bpw of **REAP’d official MXFP4 experts**. One HIP kernel. |
| Size | **~100 GiB** like K216 EXL3. |
| Leftover | Official MXFP8 → `fdot2` after bit-trick + `mxfp4_apply_e8m0_bits`. No `sdot4`. Skip act-quant on first run. |
| Decode tile | `mxfp4_dot2_moe` skinny `BLOCK_M` 1/2/4/8 — must not stay `(1,1)`. |

## Plan

1. Occupancy flip still first — EXL3 is a fatter unpack on that same launch.
2. REAP official MXFP4 experts → Viterbi retained only → K216-class EXL3.
3. One HIP kernel for QTIP dump or EXL3 ([exl3.md](exl3.md)).
4. KV: short ctx, existing HIP MLA record.
5. DSpark/MTP off until fat tile.
6. Later card is up: “QTIP/EXL3 HIP, K216-class DSv4 Flash.”

## Not this ticket

SparkInfer. Viterbi-then-REAP. EXL3-from-Int4. Copied tok/s. Official upstream PR.
