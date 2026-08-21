# DeepSeek-V4-Flash on 4×V620 (120 GB) — run plan

Date: 2026-08-22. Engine contract. Human branch **`rdna2_extras`**. Occupancy still first. Pair with [exl3.md](exl3.md), [dspark.md](dspark.md), [moe.md](moe.md), [kv-int8.md](kv-int8.md).

**Budget:** 4×30 GB usable (V620 32 GB minus reserve) = **120 GB**. TP=4. DSv4 Flash 0731 is 284B total / 13B active. Official experts are already **MXFP4** (E2M1 + UE8M0); most other linears FP8. extras already has mxfp4 HIP (`unpack → fdot2`) and Flash HIP MLA. That is the run path. EXL3 is a *later* quality-for-size kernel, not a drop-in.

## What those three artifacts actually are

| Artifact | What it is | Size (sourced) | Runs here? |
|---|---|---|---|
| [0xSero EXL3 3.0 bpw](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-EXL3-3.0bpw) | Rank-sliced TP4 EXL3 (`mcg`) of routed experts. Source was packed E2M1+UE8M0. **Not e2e-validated** (H200 load, no generation). | **116.29 GiB** (124,867,114,600 B) | **No.** extras has no EXL3. Weights eat the 120 GB budget. |
| [0xSero Spark REAP K216](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) | Same EXL3, **216/256** experts, top-k 6. Carried FP8 stays FP8. Sized for one 128 GB DGX Spark. | **99.48 GiB** (106,816,685,560 B) | Weights *could* fit. Runtime is SparkInfer Trellis + GB10. **No HIP.** |
| [MiaAI One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) | Docker launcher for that K216 EXL3 on **GB10 / SM121 / aarch64**, NVFP4 KV `stock432`, DSpark K5, 384k ctx. | ~107 GB download, 128 GiB UMA | **Steal KV-record idea only.** Image and kernels are dead on gfx1030. |
| brandonmusic GLM-5.2 EXL3 TR3 3.0 bpw | Different model. CUDA SM120 + Sparkinfer Trellis recipe that 0xSero copied. | ~332 GB | Recipe notes only. |

Official 0731 mixed MXFP4/FP8 checkpoint is **~155–167 GB** (range across index vs DSpark-fused). It does **not** fit 120 GB.

## Format choice (locked)

| Option | Take |
|---|---|
| Drop-in those EXL3 repos | **No.** No loader, no Trellis HIP, CUDA `mma` dead. |
| Requant to our own EXL3/QTIP now | **No.** Viterbi is CUDA-producer; we have no infer kernel yet ([exl3.md](exl3.md)). |
| Requant MXFP4 experts → INT4+scales / W4A16 | **No.** Experts are already 4-bit E2M1. A GPTQ/AWQ pass loses QAT and does not beat official size. |
| Keep official **mxfp4** experts, extras HIP | **Yes — best kernel we already have.** |
| REAP-prune **keeping mxfp4** (harder than K216) or expert host-offload | **Yes — how mxfp4 fits 120 GB.** K216×MXFP4 is estimated ~130 GB (216/256 of ~94% expert bytes on a ~155 GB ckpt) — still over. Need more prune or offload. Do not treat that estimate as measured. |
| EXL3 HIP later, consume K216 99.48 GiB | **Later.** Only published all-GPU fit that leaves KV room. |

**Best way = mxfp4, not EXL3, not a fresh INT4.** EXL3 is the fit trick after the HIP kernel exists.

## KV

Spark’s 384k ctx is `nvfp4_ds_mla` **432 B/token** (or older 584 B FP8 padded) on a unit we do not have. extras `fp8_ds_mla` is a **576 B** uint8 layout; `supports_fp8()` is false; software e4m3→half is occupancy-blocked; INT8 KV is not the [kv-int8.md](kv-int8.md) contract yet.

Adjust: **short ctx first**, keep the existing HIP MLA cache, DSpark **off** (fat tile first). Do not port `stock432` / NVFP4 KV. Do not plan 384k on 120 GB.

## Plan

1. Occupancy flip still first. A load that decodes at `(1,1)` is not “running stuff.”
2. Serve **official MXFP4 experts** through extras DSv4 HIP (`on_gfx10x()`). Software-cvt leftover FP8 linears. TP=4. DSpark/MTP off.
3. Fit 120 GB by **REAP (or equivalent) while staying MXFP4**, or **expert host-offload** — not by EXL3. Measure the pruned MXFP4 byte count; do not ship on the ~130 GB estimate.
4. KV: short `max_model_len`, existing MLA record. INT8 fused gather after occupancy.
5. Later: one HIP EXL3 kernel (`decode_3inst → fdot2`), then the K216 99.48 GiB checkpoint is the all-GPU quality-for-size load. Cornell/ExLlama stay producers.

@RDNA2_Researcher: MXFP4 expert tile vs leftover FP8 cvt cost on V620. @VLLM_FORK_Manager: no new EXL3 card; occupancy + existing DSv4/mxfp4 cards stay the board.

## Not this ticket

SparkInfer. GB10 image. CUDA Trellis. Official upstream PR. 384k NVFP4 KV. Requant-to-W4 of QAT MXFP4. Copied tok/s.
