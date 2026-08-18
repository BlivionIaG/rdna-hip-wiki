# Multi-tier MoE — Llaminar vs SGLang fork

Date: 2026-08-18. Engine contract. Target topology (human plan):

- **Fast tier:** 2× W7800 48 GB (gfx1100) — attention, routing, active path
- **Capacity tier:** 8× V620 32 GB (gfx1030) — parked experts
- **Bus rule:** ship **activations only** (and maybe KV offload), not expert weight ping-pong

Related: [moe.md](moe.md), [deepep.md](deepep.md) + [silicon/deepep-v620.md](../silicon/deepep-v620.md), [alt-engines.md](alt-engines.md), [pd-disagg.md](pd-disagg.md). Occupancy + HIP MoE on V620 still first. Do not invent tok/s.

## What this is (and is not)

This is **heterogeneous expert placement / compute-offload MoE**, closer to ktransformers (attn+shared on fast device, routed experts on capacity) than to:

- layer PP (whole layers on a stage),
- homogeneous EP (every GPU same role),
- PD disagg (already **skip as a win** on a single PCIe box).

Collective ownership stays in the engine. GEMM does not own RCCL. Intra-node A2A analogue is **mapped-peer scatter/combine over PCIe BARs**, not IBGDA/MORI/SDMA doorbells.

## Llaminar ([Llaminar/llaminar](https://github.com/Llaminar/llaminar))

C++ kernel-centric runtime. Alpha. GGUF. Explicit strengths for *this* topology:

| Piece | Status |
|---|---|
| Heterogeneous domains (CPU / CUDA / ROCm in one plan) | Native design |
| TP / PP / MoE EP | EP WiP; TP/PP exercised |
| Prefix cache | **Exists** (`--prefix-cache`, MoE placement-fingerprint policy) |
| Continuous batching | **Plan / non-goal for V1 HTTP** (single-request queue first) |
| ROCm GPU targets | **`gfx906` only** today |
| Mixed CUDA+ROCm PP | Documented |

**Verdict:** closest *orchestration skeleton* for “fast GPU = active/routing, capacity = experts.” Real cost is still gfx1030 + gfx1100 HIP backends (DOT / FA / MoE) and a multi-request scheduler. Contributing without those kernels does not unlock the box.

Do **not** assume “add prefix caching” is the Llaminar gap — that is outdated. The serving gap is continuous batching / dynamic batch.

## SGLang fork

| Piece | Status |
|---|---|
| Radix / prefix cache, continuous batching, serving | Strong |
| ROCm path | Instinct / AITER / MORI-centric; RDNA is bridge work |
| Heterogeneous EP (W7800 router + V620 experts) | **Not first-class** — you invent placement |
| Custom HIP from our vLLM fork | Same class of port as today |

**Verdict:** buy serving features, still build the tiering contract yourself. Prefer if **production multi-request** is the first product goal.

## Locked recommendation (2026-08-18)

1. **Prototype placement + activation A2A** on Llaminar (or a thin custom runtime that steals its domain/collective model).
2. **Keep SGLang / vLLM** for serving-kernel R&D and for shipping continuous batching / radix while the tiered path is immature.
3. **Do not** pick one fork as “the” engine until gfx1030 HIP MoE compute + a measured peer-store matrix exist.
4. **W7800 (gfx1100)** is a second HIP target (WMMA path exists upstream in places; our V620 work stays `fdot2`/`sdot4`). Do not collapse the two ISAs.
5. KV split/offload is optional and secondary to activation-only expert dispatch. Same PCIe physics as [kv-quant-offload.md](kv-quant-offload.md) / PD skip.

## Cards (for VLLM_FORK_Manager — Later / research)

Reuse if present. Occupancy still first.

- Multi-tier MoE placement contract (W7800 active + V620 experts, activations only) — Later
- Llaminar gfx1030 / gfx1100 backend spike — Later / side-project
- SGLang heterogeneous EP spike — Later, only if serving-first

## Sources

- https://github.com/Llaminar/llaminar README (prefix-cache flags; ROCm `gfx906`; CB plans in `docs/v2/projects/`)
- Room lock 2026-08-18 (GFX1030 Inference)
- [deepep.md](deepep.md), [moe.md](moe.md)
