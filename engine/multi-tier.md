# Multi-tier MoE — placement vs engine base

Date: 2026-08-18. Engine contract. **Product path (locked):** optimize **vLLM fork first**, then **SGLang**. Llaminar is a placement *model* to steal, **not** the repo we extend. Occupancy + V620 HIP MoE still first. Do not invent tok/s.

Silicon: [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md). Related: [moe.md](moe.md), [deepep.md](deepep.md), [alt-engines.md](alt-engines.md), [pd-disagg.md](pd-disagg.md).

## Topology (human plan)

- **Fast tier:** 2× W7800 48 GB (**gfx1100**) — attention, router, embeddings, LM head, **live KV**
- **Capacity tier:** 8× V620 32 GB (**gfx1030**) — expert GEMMs only
- **Bus rule:** ship **activations only** (fp16 on the hop unless both kernels consume a smaller dtype). Not expert-weight ping-pong.
- **Later SKU:** V340L is a possible test box. Unpinned. Do not collapse it into gfx1030/gfx1100.

This is **two HIP targets**, not one fat binary. gfx1100 may use WMMA/BF16 locally; V620 stays `fdot2`/`sdot4`. Do not park live KV on V620 — that makes decode a bigger PCIe read than the expert hop. Decode hop is tiny; **prefill is the bus**. Cross-SKU P2P (W7800↔V620) is **unmeasured** — assume host-staged copies until [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md) has a matrix.

## What this is (and is not)

Heterogeneous expert placement / compute-offload MoE (ktransformers-class: attn+shared on fast device, routed experts on capacity). Not:

- layer PP (whole layers on a stage),
- homogeneous EP (every GPU same role),
- PD disagg (already **skip as a win** on a single PCIe box).

Collective ownership stays in the engine. GEMM does not own RCCL. Intra-node A2A analogue is **mapped-peer scatter/combine over PCIe BARs**, not IBGDA/MORI/SDMA doorbells. Mixed-SKU may stay host-staged until peer access is measured.

## Engine base (locked 2026-08-18)

| Order | Engine | Why |
|---|---|---|
| **1. Now** | **vLLM fork** (`perf/rdna2_w4a16`) | Stack that already compiles gfx1030 HIP. Occupancy, FA, MoE DOT, then placement glue. |
| **2. Next** | **SGLang** | Radix / CB / serving after vLLM kernels exist. Same two HIP backends + a heterogeneous EP scheduler SGLang does not have for this topology. |
| **Not the base** | Llaminar | Closer *architecture* (heterogeneous domains, TP/PP, prefix-cache exists). ROCm is **gfx906 only**. Continuous batching still a plan. Contributing still means new gfx1030 + gfx1100 HIP. |

Do **not** block occupancy or HIP MoE on a runtime-shell choice.

## Llaminar (steal, do not extend)

[Llaminar/llaminar](https://github.com/Llaminar/llaminar) — C++ kernel-centric, alpha, GGUF.

| Piece | Status |
|---|---|
| Heterogeneous domains (CPU / CUDA / ROCm in one plan) | Native design — **steal** |
| TP / PP / MoE EP | EP WiP; TP/PP exercised |
| Prefix cache | **Exists** (`--prefix-cache`, MoE placement-fingerprint) |
| Continuous batching | **Plan / non-goal for V1 HTTP** |
| ROCm GPU targets | **`gfx906` only** |

Their CUDA+ROCm examples are PP dense layers or host-staged TP, not attn-on-RDNA3 / experts-on-RDNA2.

## SGLang (after vLLM)

| Piece | Status |
|---|---|
| Radix / prefix cache, continuous batching, serving | Strong |
| ROCm path | Instinct / AITER / MORI-centric; RDNA is bridge work |
| Heterogeneous EP (W7800 router + V620 experts) | **Not first-class** |

## Implementation order (engine + silicon)

1. `fa_rdna2` occupancy + V620 HIP MoE compute (already first).
2. gfx1100 attention/router baseline on W7800 (stock, then HIP if needed).
3. Measured W7800↔V620 peer/host activation matrix.
4. Asymmetric activation dispatch/combine (PCIe BAR or host).
5. Optional KV *overflow* park — never live-KV on V620.
6. SGLang serving shell after the vLLM kernels exist.

## Cards (for VLLM_FORK_Manager)

Reuse if present. Occupancy still first.

- Multi-tier MoE placement (W7800 active + V620 experts, activations only, KV stays on W7800) — Later
- W7800↔V620 activation matrix — Later, after occupancy + HIP MoE
- SGLang hetero EP — Later, after vLLM
- Llaminar contribution — **not a card** unless the product path changes
- V340L — Later / unpinned

## Sources

- Room lock 2026-08-18 (GFX1030 Inference): vLLM then SGLang; Llaminar not the base
- [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md)
- https://github.com/Llaminar/llaminar README
- [deepep.md](deepep.md), [moe.md](moe.md)
