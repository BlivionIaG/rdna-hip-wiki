# Cache-aware scheduling

Date: 2026-08-20. Engine contract. Batching mechanics: [batching.md](batching.md). IC fit: [infinity-cache.md](infinity-cache.md). Occupancy still first. Do not invent tok/s.

**Two different caches.** Do not merge them into one “cache-aware” ticket.

| Cache | What it holds | Scheduler lever we already have |
|---|---|---|
| **KV prefix** (paged, `block_size=16`) | Reusable tokens | vLLM V1 **APC on by default**. Hash complete blocks. Incomplete last block is not reusable. |
| **Infinity Cache** (128 MB L3) | Hot **weight shard + this-step pages** | Working set + decode-first. **No persist bit.** |

extras has **no** custom cache-aware scheduler. Stock V1 is the lever. SGLang radix / cache-aware router is steal-later, not the next engine.

## What stock V1 already does

- **Prefix cache:** skip prefill for hashed complete 16-token blocks. Put stable context first (system / repo / RAG / history); unique user tokens last. Reuse **ends at the first novel token** ([batching.md](batching.md) Kiely).
- **Decode first:** `schedule()` packs running decodes into `token_budget`, then waiting only if no preemption. That is the IC-friendly order — same W4 shard reread every token.
- **Chunked prefill:** leftover budget, `max_num_partial_prefills` default **1**. One fat chunk + many `l_q=1` still fights the 128 MB.
- Policy: FCFS or priority. Not “route to the GPU that already has this prefix” (that is a **router**, multi-replica).

Measure `max_num_batched_tokens` on V620. Do not copy Instinct 8k/16k.

## IC-aware rule (this box)

Keep a **decode-only** step when we care about the 128 MB hit: 7B/27B W4 TP=4 shards fit; a fat prefill chunk does not. Two HIP streams / SDMA on the same V620 evict the shard ([infinity-cache.md](infinity-cache.md)).

Do **not** write a new scheduler to pin lines. Do:

1. Leave APC on.
2. Occupancy flip so the decode GEMV actually runs.
3. Sweep MBT; prefer decode-heavy mixed batches over huge leftover prefills.
4. Later: nontemporal only miss-path weights; INT8-KV shrinks the window ([kv-int8.md](kv-int8.md)).

Not a win here: PD-disagg ([pd-disagg.md](pd-disagg.md)), per-token KV H2D, SGLang cache-aware LB (needs a replica pool we are not running).

## Not a card

Same occupancy card. APC/MBT is tune-and-measure after occupancy, not a new first kernel.

## Sources

- [batching.md](batching.md), [infinity-cache.md](infinity-cache.md), [silicon/cache-policy.md](../silicon/cache-policy.md)
- Room 2026-08-20 (user: cache-aware scheduling)
