# DFlash — engine spec (gfx1030)

Date: 2026-08-17. Engine contract. Card: DFlash on [project 4](https://github.com/users/BlivionIaG/projects/4). Not sdot4. Sibling cards: [mtp.md](mtp.md), [dspark.md](dspark.md). Family page: [specdec.md](specdec.md). Do not edit `perf/rdna2_w4a16` from this page. Occupancy still first.

**Verdict:** DFlash is a **block-diffusion parallel drafter**. It projects target hidden states into the speculator KV and proposes a block in one pass. Verify is still **extend (`q = 1+k`)**. On gfx1030 it does **not** fire until a **fat MLA tile (`q>1`)** exists. No new DOT kernel. Same silicon blocker as MTP / mix / DSpark.

Not on the human branch as a working path. Upstream method: `speculative-config.method = dflash`.

## What DFlash is (engine)

| | DFlash |
|---|---|
| Draft | External / attached speculator. Parallel block, not serial MTP heads. |
| Conditioning | Hidden states → projected into speculator KV (not concatenated as extra tokens). |
| vLLM method | `dflash` (Speculators / DeepSpec family). |
| Verify | one target forward over the drafted block. Confidence proxy = `max(softmax(logits))` (head-free). |
| vs MTP | Parallel draft, extra module, higher single-user k, worse under high concurrency if you always verify the full block. |
| vs DSpark | No Markov head, no dedicated confidence head. DSpark **is** DFlash + those two. |

## gfx1030 contract

```
// keep — after occupancy + fat tile
Reuse HIP MLA / fa_rdna2 for verify (q>1)
Draft backbone is another forward — same fp16 GEMMs we already have
fp16 only

// drop
AITER / TileLang DFlash sparse MLA (CDNA / CUDA)
SM120 paged sparse MLA (aborts when num_tokens ≤ 64 — draft verify is 5–7)
CUDA-graph capture of the DFlash backbone as the first gfx1030 work
A new DOT "for DFlash"
```

Draft GEMMs are ordinary (or already-quant) linears. The missing piece is **verify attention shape**, not a quant kernel.

## Dispatch

| Piece | Spec |
|---|---|
| Gate | off until fat tile + occupancy. |
| Draft attn | SparseMLA-shaped, often non-causal / block. Do not take the CUDA TileLang op. |
| Verify attn | HIP MLA fat tile or fa_rdna2 short-extend. |
| k | `num_speculative_tokens` typically 5–7. Do not raise k to hide a q=1 kernel. |
| Checkpoint | needs a `dflash` / speculators module. Preview DSv4 without that block is MTP, not DFlash. |

## Done-when

Fat tile exists. A DFlash checkpoint drafts + verifies on gfx1030 without AITER/TileLang. Verify attn is the same inner op we already own. No tok/s from this page.

## Not this ticket

Occupancy. Sage. INT2. MTP / DSpark (own cards). INT8 KV. Writing a parallel-draft GEMM from scratch.

## Sources

- [vLLM Speculators blog](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting) (P-EAGLE / DFlash / DSpark)
- [specdec.md](specdec.md)
- Room lock: HIP MLA still q=1; fat tile before any of these fire
