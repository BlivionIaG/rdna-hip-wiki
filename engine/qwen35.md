# Qwen3.5 on gfx1030

Date: 2026-08-17. Engine contract. Sourced from [ikantkode/gfx1030-vllm-0.26](https://github.com/ikantkode/gfx1030-vllm-0.26) + [Qwen3.5-4B-AWQ-vd](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd) (overlay on `blivioniag/vllm-rdna:v0.26.0`). Digest: [notes/ikantkode-qwen35.md](notes/ikantkode-qwen35.md). Tickets on [project 4](https://github.com/users/BlivionIaG/projects/4). Occupancy still first.

**Verdict:** Qwen3.5 is a **hybrid GDN + full-attn** model with **GemmaRMSNorm (`1+w`)**. Stock AWQ (QuantTrio) leaves attention in fp16. A full-attn INT4 requant is legal if the LN fold is stored as `w = s·(1+w_base)−1`, not Llama `s·w`. Serve path they proved is **Triton AWQ + ROCM_ATTN**, not our HIP GEMM/FA. Steal gates and the norm fuse; do **not** rewrite `q_gemm_rdna2`.

Their tok/s stay on the note page.

## Architecture (engine)

| Piece | Fact |
|---|---|
| Blocks | 8 full-attn + 24 gated-delta (linear-attn) |
| Norm | `Qwen3_5RMSNorm` = Gemma `(1+w)`. IR `rms_norm` rejects fp32 weight vs fp16 act → ATen chain. |
| Quant-safe | MLP + self-attn + GDN qkv/z/out as AWQ INT4 g128. |
| Stay fp16 | `in_proj_a`/`in_proj_b`, embeddings, MTP head, vision |
| MTP | Ships. On an M=1-optimized gfx1030 decode stack they measured it **slower** than plain decode. Matches [mtp.md](mtp.md). |
| Attn backend | `TRITON_ATTN` blows 64 KB LDS. Live: `ROCM_ATTN` → Triton paged (`paged_attention_rocm` dead: arch-gate, no HEAD=256, non-pow2 block). |

## Steal (tickets)

| Card | Action |
|---|---|
| LLMM1 gfx1030 + wvSplitK off | Flip skinny gate to include gfx1030. **Do not** enable `wvSplitK` (device-assert). `k>8192` is HIP skinny / existing Triton GEMV, not a new family. |
| Qwen3.5 / Gemma RMSNorm fuse | One kernel, `(1+w)` in fp32 from raw fp16 weight. After occupancy. |
| Qwen3.5 AWQ-vd recipe | Requant post-pass only. Not a DOT. |
| Qwen3.5 GDN | Later. No HIP linear-attn. |

Paged Triton `num_warps=8` folds into the existing occupancy / head-64 Triton-paged card.

## Do not

- Treat `-vd` as a new quant opcode.
- Port their AWQ Triton GEMV over live HIP W4A16.
- Copy base compose (TRITON_ATTN, TP=2, 32k, eager).
- Enable AITER, wvSplitK, or torch.compile/inductor (ROCm #5572).
- Put their 84.5 tok/s on [coverage.md](coverage.md).

## Done-when (if we take the cards)

LLMM1 either launches on gfx1030 or we document why HIP skinny supersedes it. wvSplitK stays off. A Qwen3.5 checkpoint with `(1+w)`-correct norms serves without garbage. No tok/s from this page.
