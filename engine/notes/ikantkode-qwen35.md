# ikantkode gfx1030 Qwen3.5 AWQ-vd — source digest

Date: 2026-08-17. Paraphrase + attribution only. Not a silicon contract. Their tok/s are **theirs** (one V620, vLLM 0.26.1.dev, file-mount on `blivioniag/vllm-rdna:v0.26.0`). Do not copy those numbers onto coverage / fork-delta.

**Sources**

- Checkpoint: [ikantkode/Qwen3.5-4B-AWQ-vd](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd)
- Patches + ladder: [ikantkode/gfx1030-vllm-0.26](https://github.com/ikantkode/gfx1030-vllm-0.26) (`README.md`, `CHANGELOG.md`, `patches/`, `requant/`)
- Image they overlay: `blivioniag/vllm-rdna:v0.26.0` (Blivion docker). No vLLM rebuild — bind-mounts into `/src/vllm/...`.

Contract takeaways: [qwen35.md](../qwen35.md).

## What they actually did

Two products, not one:

1. **Checkpoint recipe** (`requant/quant.py`) — re-AWQ of `Qwen/Qwen3.5-4B` so **self-attn + GDN linear-attn + MLP** are INT4 (group 128, asymmetric ZP, GEMM pack). `in_proj_a`/`in_proj_b`, norms, embeddings, MTP head stay fp16. They claim 3.8 GB vs QuantTrio 5.7 GB (attn left fp16).
2. **Triton / gate overlays** on the 0.26 image — stay on **stock AWQ Triton + ROCM_ATTN**, not our HIP `q_gemm_rdna2` / `fa_rdna2`.

Working serve (override, not the base compose): TP=1, `--dtype float16`, `--attention-backend ROCM_ATTN`, `--max-model-len 8192`, `--max-num-batched-tokens 2048`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, no `--enforce-eager`. Base yml still says TRITON_ATTN + TP=2 + eager — **do not copy that**; it is the broken starting point.

## Ladder (theirs, sourced)

They publish a 15-rung single-stream decode ladder on one V620, 256-token greedy, `Qwen3.5-4B-AWQ-vd` after rung 8. ~10 → **84.5** tok/s (tag `rung-15-84.5tps`). Hard ceiling they compute: 3.12 GB/token @ ~445 GB/s ≈ 142 tok/s. We do not re-measure here.

| Rung | What (engine view) |
|---|---|
| 1, 4–6, 9–11, 15 | Triton AWQ: `SPLIT_K` policy, M=1 GEMV (no `tl.dot` M-tile), K-split GEMV + reduce, per-(N,K) dispatch |
| 2 | Breakable CUDA graphs, no torch.compile (ROCm #5572 freeze) |
| 3, 7 | Unlock AMD **LLMM1** for gfx1030; Triton fp16 GEMV when `k>8192` (LLMM1 will not launch) |
| 8 | Full-attn + GDN INT4 requant |
| 13 | Fused Gemma-style RMSNorm (gain `1+w`) |
| 14 | Triton paged-attn `num_warps=8` (4-WG grid on 72 CUs was latency-bound) |

## Steal-worthy (vs our fork)

Our HIP path is a different stack. Steal facts and gates, not their Triton AWQ as a rewrite of `q_gemm_rdna2`.

| Item | Why it is real | Ticket? |
|---|---|---|
| **LLMM1 works on gfx1030** | One arch-gate: `use_skinny` was `on_gfx9() or on_gfx1x()` — gfx1030 matches neither. They add `or on_g1030`. Claim 3.6–5.6× vs rocBLAS at n==1, `k<=8192`. | **Yes** — confirm vs live `skinny_gemms.cu`. Same occupancy card if HIP skinny is the product. |
| **wvSplitK device-asserts** | They leave the `n<=5` branch **off** on gfx1030. | **Yes** — landmine on the same skinny card. |
| **LLMM1 `k>8192` hole** | GDN `out_proj` (2560, 9216) falls to rocBLAS. Their Triton fp16 GEMV is a stopgap; HIP skinny should cover it. | Note on skinny card, not a new kernel family. |
| **Qwen3.5 = GemmaRMSNorm (`1+w`)** | IR `rms_norm` wants `weight.dtype == x.dtype`; Gemma passes fp32 weight vs fp16 act → 10–13-launch ATen chain ×81/token. Fused Triton computes `(1+w)` in fp32 from the raw fp16 weight. | **Yes** — Qwen3.5 / Gemma RMSNorm fuse (HIP later). |
| **LN-fold Llama vs `(1+w)`** | AutoAWQ (`quivent/autoawq-qwen35`) stores `w = s·w_base`. Qwen3.5 applies `(1+w)` → garbage at serve, clean per-tensor checks. Fix: `w_new = s·(1+w_base)−1`, s from LS on a consumer. MLP up/down: identity AWQ scales (plain g128). fp16 `in_proj_a/b` divided by s. | **Yes** — checkpoint recipe card, not a DOT kernel. |
| **Paged-attn `num_warps=8`** | Stock launch passes no warps → JIT 4. Grid `(seqs, kv_heads)` = 4 WGs. They claim 226→112 µs/call. | Fold into occupancy / Triton paged (head-64 hole). Not a new FA. |
| **TRITON_ATTN LDS crash** | `OutOfResources` shared 139264 > 65536. Live path is `ROCM_ATTN` → Triton `kernel_paged_attention_2d` (custom `paged_attention_rocm` is arch-gated, no HEAD=256, non-pow2 528 block). | Landmine. Matches our “stock HIP paged is dead / Triton hole”. |
| **MTP slower on this stack** | Model card: MTP verify worse than plain decode with M=1-optimized kernels. | Confirms [mtp.md](../mtp.md). No extra card. |
| **Breakable CUDA graphs** | `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, never eager-without-it. | Deploy note, not a kernel ticket. |

## Do not steal as-is

- Their **Triton AWQ GEMV / split-K** as a replacement for live `q_gemm_rdna2`. Dispatch leftover stock AWQ there; do not rewrite the HIP GEMMs.
- Their **tok/s** onto our coverage map.
- **AITER**, `wvSplitK`, `TRITON_ATTN` default, `--enforce-eager` without the breakable-graph env.
- **TP=2 + 32k ctx** from the base compose (negative KV on this box).
- Treating `-vd` as a new quant scheme — it is AWQ INT4 with more layers quantized + a norm-storage fix.

## Qwen3.5 shape (why this model is annoying)

Hybrid: 8 full-attn layers + 24 gated-delta / linear-attn layers. GDN keeps `in_proj_a`/`in_proj_b` in fp16. MTP head ships as `model_mtp.safetensors` (`qwen3_next_mtp`). Vision excluded from quant. Head dim / block sizes push them onto Triton paged (HEAD=256, non-pow2 block).

No HIP GDN kernel in their tree or ours. Later.

## Landmines they named (keep)

- `VLLM_ROCM_USE_AITER=0`
- `--dtype float16` (no bf16 unit)
- After editing a bind-mount: `compose up --force-recreate` (inode replace leaves the old file)
- rocprof v1 cannot see the decode window; they use torch.profiler (`profiling/prof2.py`)
- Microbench L2 reuse lies; trust e2e / profiler

## Ticket list (for VLLM_FORK_Manager)

Reuse if a card already covers it. Occupancy still first.

1. **LLMM1 gfx1030 + wvSplitK off** — confirm AMD skinny vs `skinny_gemms.cu`; keep wvSplitK disabled. Same occupancy family if HIP skinny is the product.
2. **Qwen3.5 / Gemma RMSNorm (`1+w`) fuse** — after occupancy. Their Triton is the existence proof; HIP optional.
3. **Qwen3.5 AWQ-vd recipe** — `(1+w)` LN-fold + identity MLP up/down + `in_proj_a/b` 1/s. Checkpoint, not a kernel.
4. **Qwen3.5 GDN linear-attn** — Later. No HIP. Do not open until W4A16/FA occupancy is clean.

Paged `num_warps=8` = note on the existing Triton paged / occupancy card.
