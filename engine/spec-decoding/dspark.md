# DSpark — engine spec (gfx1030)

Date: 2026-08-17. Engine contract. Card: DSpark on [project 4](https://github.com/users/BlivionIaG/projects/4). Not sdot4. Sibling cards: [mtp.md](mtp.md), [dflash.md](dflash.md). Family page: [specdec.md](specdec.md). Do not edit `perf/rdna2_w4a16` from this page. Occupancy still first.

**Verdict:** DSpark = **DFlash-style parallel backbone + lightweight Markov head + confidence-scheduled verify length**. Verify is still **extend (`q = 1+k`)**, and DSpark draft attn is **non-causal sliding-window** (upstream reuses SparseMLA with expanded topk so queries see each other). On gfx1030 it does **not** fire until a **fat MLA tile (`q>1`)** exists. No new DOT kernel. Same silicon blocker as MTP / DFlash / mix.

Fork today: DSv4 gfx10x routing keeps this off (`VLLM_DISABLE_DSPARK_MTP`). DSpark is **not** serial MTP even when the checkpoint stores the module under `mtp.*` — upstream refuses a silent MTP fallback (vLLM #46965).

## What DSpark is (engine)

| | DSpark |
|---|---|
| Draft | Parallel block (DFlash backbone) + rank-256 sequential Markov logit-bias. |
| Scheduler | Confidence head prunes tail tokens; verify length is load-aware. |
| vLLM method | `speculative-config.method = dspark` (needs a `dspark_*` block in `config.json`). |
| Typical k | 5–7 (`num_speculative_tokens`). `draft_sample_method` greedy or probabilistic. |
| vs MTP | Parallel + scheduled verify. Replaces MTP-1 in DeepSeek V4 prod. |
| vs DFlash | Adds Markov head + dedicated confidence head. |

Checkpoints: `DeepSeek-V4-Flash-DSpark` / `*-0731` carry the module. Preview / NVFP4 DSv4 without `dspark_*` is MTP, not DSpark.

## gfx1030 contract

```
// keep — after occupancy + fat tile
Verify: HIP MLA fat tile or fa_rdna2 short-extend
Draft attn: SparseMLA-shaped, non-causal topk (queries include each other)
fp16 only. Existing fdot2 GEMMs for the backbone / Markov head

// drop
AITER / TileLang DSpark sparse MLA
SM120 paged sparse MLA (verify batch is k tokens, kernel wants >64)
CUDA-graph of backbone + AR sample loop as first gfx1030 work
Silent fallback to MTP
A new DOT / sdot4 path "for DSpark"
```

The Markov head and confidence head are tiny linears. They are not the ticket. The ticket is **q>1 verify + non-causal draft attn** on the kernels we already have.

## Dispatch

| Piece | Spec |
|---|---|
| Gate | stay off (`VLLM_DISABLE_DSPARK_MTP`) until fat tile + occupancy. |
| Method | `dspark` only if `dspark_*` is in the checkpoint. Do not lie and run MTP. |
| Draft attn | reuse HIP / Triton sparse MLA with expanded topk (upstream CUDA design). No TileLang port. |
| Verify attn | fat tile. Short Br=32 is the MHA-shaped cousin. |
| Prefix cache | DSpark has a private rolling KV window — defer proposals after cache hits until it is rebuilt (upstream). |
| Parallelism | upstream DSv4 DSpark is PP=1, PCP=1, DCP=1. Do not invent TP=4 draft-state handling on this card. |

## Done-when

Fat tile exists. A DSpark checkpoint drafts + confidence-trims + verifies on gfx1030 without AITER/TileLang. No silent MTP fallback. No tok/s from this page.

## Not this ticket

Occupancy. Sage. INT2. MTP / DFlash (own cards). INT8 KV. Writing a Markov-head kernel.

## Sources

- [DSpark paper](https://arxiv.org/html/2607.05147v1)
- vLLM [PR #46995](https://github.com/vllm-project/vllm/pull/46995), [PR #46965](https://github.com/vllm-project/vllm/pull/46965)
- [DeepSeek-V4-Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- [specdec.md](specdec.md)
- Lock: HIP MLA still q=1; fat tile before any of these fire
