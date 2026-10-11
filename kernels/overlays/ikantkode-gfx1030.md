---
title: "ikantkode gfx1030 overlays — silicon review"
sources:
  - https://github.com/ikantkode/gfx1030-vllm-0.26
  - https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd
  - https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/fa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/skinny_gemms.cu
  - https://github.com/BlivionIaG/vllm/commit/8e35767f59830fbe373ea3c8df8d39421716fad7
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---
# ikantkode gfx1030 overlays — silicon review

Sourced from [ikantkode/gfx1030-vllm-0.26](https://github.com/ikantkode/gfx1030-vllm-0.26) (file-mount patches on `blivioniag/vllm-rdna:v0.26.0`) and the `-vd` checkpoint [ikantkode/Qwen3.5-4B-AWQ-vd](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd). Their numbers, not ours. Occupancy still first.[^ikantkode][^ikantkode-hf]

**Not our HIP path.** Triton AWQ GEMV + gate flips. No `fdot2` / `sdot4` / `fa_rdna2`. Do not port the Triton kernels. Take the **gates and occupancy lessons**.[^f-7779514b-fa-rdna2-cu][^ikantkode]

## What they actually changed (silicon)

| Rung | Change | Opcode / launch | Take [^ikantkode] |
|---|---|---|---|
| 3 | Unlock AMD **LLMM1** for gfx1030 (`on_gfx9 \|\| on_gfx1x \|\| gfx1030`) | in-tree skinny n==1, `k≤8192` | **Keep.** Highest-ROI line. Check the fork has this gate. [^f-7779514b-skinny-gemms-cu][^ikantkode] |
| 3 | **Disable `wvSplitK` on gfx1030** | device-assert | **Keep off.** Same skinny card. Do not “enable MFMA.” [^f-7779514b-skinny-gemms-cu] |
| 6–10, 15 | M==1 AWQ GEMV, then K-split + per-(N,K) table | Triton `tl.sum` fp32 FMA, **not** `tl.dot` / `fdot2` | Lesson only: fill 72 CUs when `cdiv(N,BN)` is tiny. HIP W4A16 already owns this job. [^ikantkode] |
| 7 | fp16 GEMV for `n==1, k>8192` (GDN `out_proj` 2560×9216) | Triton scalar FMA | LLMM1 cannot launch there. HIP skinny / W4A16 should cover k>8192. [^f-7779514b-skinny-gemms-cu][^ikantkode] |
| 13 | Fused Gemma RMSNorm | 1 Triton vs 10–13 ATen; IR gate wants `weight.dtype==x.dtype`, Gemma weight is fp32 | Llama HIP AOT landed extras `8e35767f` ([triton-jit-aot.md](../aot-glue/triton-jit-aot.md)). Gemma `(1+w)` still Todo. Do not port their Triton GEMV. [^c-8e35767f][^ikantkode] |
| 14 | paged-attn Triton `num_warps=8` | 4 WG on 72 CUs was latency-bound (226→112 µs) | Same occupancy ticket as `fa_rdna2`. [^f-7779514b-fa-rdna2-cu] |
| 8 | Re-quant attn + GDN to INT4 | checkpoint, not a kernel | Engine. `(1+w)` LN-fold, not Llama `w`. [^ikantkode] |

Their GEMV inner loop is `acc += tl.sum(w_f32 * x_f32)` after AWQ nibble unpack. Same “unpack then scalar FMA” we already refused to treat as a new DOT on MLA.[^ikantkode]

## Confirmed landmines (matches our ISA)

- LDS **64 KB/WG**: `TRITON_ATTN` asked 139264 and died. They stay on `ROCM_ATTN`.[^isa][^ikantkode]
- `wvSplitK` asserts on gfx1030 (our `skinny_gemms.cu` MFMA/`waves_per_eu(1,1)` trap).[^f-7779514b-skinny-gemms-cu]
- No bf16; `--dtype float16`.[^ikantkode]
- AITER off.[^ikantkode]
- MTP **slower** on this M=1-optimized decode — same fat-tile (`q>1`) blocker we already filed.[^ikantkode]
- torch.compile/inductor can freeze gfx1030 (ROCm #5572); they use breakable cudagraphs.[^ikantkode]

## Tickets

Reuse: occupancy (`fa_rdna2` + skinny), W4A16 dense (HIP GEMV, not their Triton), MTP Todo.[^f-7779514b-fa-rdna2-cu][^ikantkode]

**Add**
1. **LLMM1 gfx1030 gate** — one-line `on_gfx10x()` in `rocm_unquantized_gemm_impl`. Confirm whether `perf/rdna2_w4a16` already has it.[^f-7779514b-skinny-gemms-cu][^ikantkode]
2. **Fused Gemma RMSNorm** — fp32 gain `(1+w)`, one kernel, kill the native chain. Spec.[^ikantkode]

**Do not add:** a Triton AWQ GEMV card, a GDN/FLA HIP card (their FLA is 0.31 ms/token), copying their per-shape table into HIP.[^ikantkode]

## Sources

- https://github.com/ikantkode/gfx1030-vllm-0.26 README + CHANGELOG + `patches/utils.py` + `patches/awq_triton.py`
- v1.1.0 INT4 `lm_head` (untie + `quantization_config.lm_head`): decode bytes, not a new DOT. On extras HIP, M=1 K=2560 N≈248k already hits `select_config` ConfigA (`M<4 && N>4096`), not the fat-M leftover.
- Their `paged_attention_rocm` “triple-dead” (arch gate, no HEAD=256, non-pow2 528) is our FA occupancy / head-256 card (`03b2d91a`), not Triton `num_warps=8`.

- https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd
- RDNA 2 ISA: 64 KB LDS/WG; no bf16 DOT

[^ikantkode]: ikantkode/gfx1030-vllm-0.26 (README, CHANGELOG, `patches/`). <https://github.com/ikantkode/gfx1030-vllm-0.26>
[^ikantkode-hf]: ikantkode/Qwen3.5-4B-AWQ-vd model card. <https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd>
[^f-7779514b-fa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/fa_rdna2.cu` @ [`7779514b`](https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/fa_rdna2.cu).
[^f-7779514b-skinny-gemms-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms.cu` @ [`7779514b`](https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/skinny_gemms.cu).
[^c-8e35767f]: BlivionIaG/vllm commit [`8e35767f`](https://github.com/BlivionIaG/vllm/commit/8e35767f59830fbe373ea3c8df8d39421716fad7) (2026-09-01) — "feat(rocm): HIP AOT RMSNorm — cudagraph-safe replacement for Triton path".
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
