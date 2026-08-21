# DeepSeek-V4-Flash on 4×V620 (120 GB) — quality destination

Date: 2026-08-22. Engine contract. Human branch **`rdna2_extras`**. Occupancy still first. Pair with [exl3.md](exl3.md), [dspark.md](dspark.md), [silicon/dsv4-flash.md](../silicon/dsv4-flash.md).

**Goal (user lock):** maximize quality with **QTIP/EXL3** and a **native HIP** kernel that runs it. Size target is **K216-class ~100 GiB** so 4×30 GB (120 GB) still has KV room. Official 0731 QAT MXFP4/MXFP8 is the *producer input*, not the all-GPU serve. Int4-FP8 is a side checkpoint, not the quality path.

## Checkpoints (measured)

| Artifact | What | Size | Role |
|---|---|---|---|
| Official 0731 QAT | MXFP4 experts (E2M1+UE8M0). Leftover attn/shared/indexer is **MXFP8** (e4m3 + e8m0/128²), not W8A16-FP8. | ~155–167 GB | **Convert source.** Does not fit 120 GB. |
| [BlivionIaG Int4-FP8](https://huggingface.co/BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8) | compressed-tensors W4A16 of those MXFP4 experts. 46 shards. | **164.96 GiB** (177,126,472,096 B) | Side path. Larger than official. Don’t EXL3 this. |
| [BlivionIaG Int4-FP8 REAP-216B](https://huggingface.co/BlivionIaG/DeepSeek-V4-Flash-0731-Int4-FP8-REAP-216B) | Same Int4 + 216 experts. | **118.25 GiB** (126,967,193,328 B) | Touches 120 GB with **no KV**. Not the quality dest. |
| 0xSero EXL3 3.0 bpw | Rank-sliced TP4, `mcg`, from E2M1+UE8M0. Not e2e-validated. | **116.29 GiB** | Full-expert EXL3. No KV room. |
| 0xSero / MiaAI Spark **K216 EXL3** | 216/256, top-k 6, SparkInfer/GB10. | **99.48 GiB** (106,816,685,560 B) | **Size target.** Runtime is dead here. Layout/size reference. |

## Destination (locked)

| Piece | Take |
|---|---|
| Quality | QTIP/EXL3 3–4 bpw of **official QAT** (or REAP-keep-mxfp4, then EXL3). One HIP kernel: `decode_3inst` → half → `fdot2`. |
| Size | **~100 GiB** like K216 EXL3. Leaves ~20 GB for KV/acts/TP on 120 GB. |
| Convert | CUDA Viterbi stays the producer. Do **not** EXL3 the Int4-FP8 repos (second quant). |
| Leftover | MXFP8 → same `fdot2` after bit-trick + `mxfp4_apply_e8m0_bits`. No `sdot4`. Skip act-quant on first run. |
| Decode tile | mxfp4-shaped skinny `BLOCK_M` 1/2/4/8, 4×`fdot2`/8K — must not stay `(1,1)`. |
| Int4-FP8 | Keep as a W4 experiment. It does not beat QAT and does not hit the ~100 GiB hole. |
| Official mxfp4 HIP | Interim serve only if we offload/prune *without* requant. Does not replace EXL3 HIP. |

## Plan

1. Occupancy flip still first — the EXL3 GEMM inherits `q_gemm_rdna2` attrs.
2. Produce (or reuse) a **K216-class EXL3** from official 0731 QAT, 3.0–3.5 bpw, `mcg`/`mul1` + 16×16. Cornell/ExLlama produce; we consume.
3. One HIP kernel for QTIP dump or EXL3 ([exl3.md](exl3.md)). Loader remaps pack; GEMM does not change.
4. KV: short ctx, existing HIP MLA record. No Spark `stock432` NVFP4.
5. DSpark/MTP off until fat tile.
6. One Project 4 **Later** card: “QTIP/EXL3 HIP, K216-class DSv4 Flash.” Occupancy still blocks landing.

## Not this ticket

SparkInfer. Drop-in 0xSero as the runtime. EXL3-from-Int4. Requant QAT MXFP4 → W4 as the dest. Copied tok/s. Official upstream PR.
