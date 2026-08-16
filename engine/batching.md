# Continuous batching and chunked prefill

Date: 2026-08-17. Sourced only.

## Mechanics

- **Request-level** (FasterTransformer): admit a batch, all prefills, then decode until last finishes. New prefills wait.
- **Iteration-level** (Orca, OSDI'22): one scheduler decision per model iteration. Selective batching: flatten non-attention; attention per-request because KV lengths differ.
- Orca/vLLM-era were prefill-prioritizing → generation stalls. **Sarathi-Serve:** token budget τ; pack running decodes first, continue partial prefill, admit new only into leftover, chunk to fit.

## vLLM V1

- waiting / skipped_waiting / running. No swapped queue. GPU↔CPU swap removed. Default RECOMPUTE.
- Prefix cache on by default. Hash complete `block_size=16` chunks. Incomplete last block not reusable.
- `schedule()`: running first (decode priority) into token_budget; waiting only if no preemption this step.
- Chunked prefill always on. `max_num_partial_prefills` default 1.
- `scheduler_reserve_full_isl` default True: admit only if full sequence fits.
- Scheduling: FCFS default, or priority.

Tune `max_num_batched_tokens`: smaller (e.g. 2048) → better ITL; `>8192` for throughput on small models / large GPUs. ROCm guide online default 8192, offline 16384. Those are Instinct / HBM numbers.

## SGLang

- waiting_queue / chunked_req / running_batch. Retract on KV full (recompute, not swap).
- Mixed-chunk default **False**. Default does **not** mix P+D. Whole mixed batch still goes through the prefill kernel (inefficient).
- Chunk size heuristic: 8192 A100/H100, 16384 B200/MI300, 4096 fallback.
- Mixed-chunk: ITL win on some workloads, not peak throughput.
- Overlap schedule on by default (CPU preps N+1 while GPU runs N).

## Kernel shapes

Linear ops dominate even at long seq (Sarathi). Decode GEMV-like; prefill GEMM. A100: one decode token ≈ 128 prefill tokens in linear cost. Crossover ~500–600 tokens at higher TP — **A100, not RDNA**.

Tile quantization: chunk 257 vs 256 can add 32% prefill time.

Mixed batch = many `l_q=1` plus one `l_q=chunk`. vLLM V1 picked FA3 because it mixes P+D. AITER FA/MLA: uniform batches only.

DeepSeek V4 / sparse MLA: q=`[B,H,D]` only. Chunked prefill must be a **separate prefill step**, not V1 leftover-MBT mix, until an MLA fat tile exists. See [attention-dispatch.md](attention-dispatch.md).

## Measure MBT on V620

Sweep 512, 2048, 4096, 8192, 16384. Report p50/p99 TTFT, ITL, TPS, KV usage, preemption, graph hit. Do not copy 8192/16384.

## Unknowns

- Exact `get_batch_defaults()` on today's main for consumer GPUs.
- Whether V1 `max_num_partial_prefills > 1` is actually live.
- Whether SGLang two-kernel mixed-chunk is still roadmap.
- Consumer RDNA optimal MBT: not published.

## Sources

- [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [vLLM V1 guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)
- [SGLang mixed-chunk](https://github.com/sgl-project/sglang/discussions/1163)

## Kiely addendum (Jan 2026)

Reusable prefix **ends at the first novel token**. Put stable context first (system, repo, RAG, history) and unique user tokens last. Incomplete last page is not reusable (vLLM hashes complete 16-token blocks). Chunked prefill is also the long-ISL interleave so decode is not stalled. Still **measure MBT** on V620; do not copy Instinct 8k/16k.
