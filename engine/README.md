# Engine

vLLM / SGLang dispatch, prefill vs decode, batching, PD, specdec, MoE, KV. Owned by LLM_Inference_specialist.

Target box: 4× V620 (gfx1030), ROCm 7.2. Hardware facts live in [silicon/](../silicon/README.md). Format contracts live in [kernels/](../kernels/README.md). VALU fire-list: [silicon/valu.md](../silicon/valu.md).

Do not invent tok/s, IC TB/s, or a P2P-works claim.

Session dump (2026-08-17/18): [notes/session-2026-08-17.md](notes/session-2026-08-17.md).

| Page | Contents |
|---|---|
| [full-map.md](full-map.md) | One-page verdict + technique table |
| [coverage.md](coverage.md) | Feature / quant / HIP kernel coverage + progress |
| [fork-delta.md](fork-delta.md) | Each added feature vs upstream — most incomplete, W4A16 dense is the bar |
| [rdna2-extras.md](rdna2-extras.md) | Release overlay — **`rdna2_extras`** = vLLM tag + gfx1030 work |
| [hippih.md](hippih.md) | In-house HIP engine — gfx1030 / gfx1100 / gfx900; after extras + Llaminar |
| [baseline-order.md](baseline-order.md) | Stock-first research order: harness → Triton map → skinny → FA → HIP A/B → FlyDSL |
| [fp16-rdna2.md](fp16-rdna2.md) | Fastest gfx1030 FP16 — explicit `fdot2` + occupancy, phase-split |
| [triton-rocm.md](triton-rocm.md) | Stock Triton / ROCM_ATTN / skinny dispatch on gfx1030 |
| [triton-flash-attention.md](triton-flash-attention.md) | Stock Triton FA configs (ROCM_ATTN is Triton/Triton here) |
| [triton-tuning.md](triton-tuning.md) | Tuning knobs only — never overwrite stock |
| [flydsl.md](flydsl.md) | FlyDSL gfx1030 gates — compiler yes, shipped MFMA/WMMA kernels no |
| [fp16-moe.md](fp16-moe.md) | Native HIP FP16 MoE draft — decode skinny + prefill grouped, `fdot2` |
| [int8-moe.md](int8-moe.md) | Dual-route INT8 MoE — W8A16 `fdot2` + W8A8 `sdot4` |
| [multi-tier.md](multi-tier.md) | Hetero MoE — vLLM first, Llaminar next, **hippih in-house**; gfx900 is mix/FMA |
| [rocmfpx.md](rocmfpx.md) | ROCmFPX leverage — codebook→`perm`→`sdot4`, not a vLLM port |
| [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md) | llama.cpp side project: `build-rdna2.sh` on V620 |
| [deepep.md](deepep.md) | DeepEP insight — mapped-peer scatter/combine, not MORI/IBGDA |
| [plx.md](plx.md) | PEX88096 / PEX8749 — ACS, PIX, Gen4 vs Gen3 ceiling, MMIO |
| [nvfp4.md](nvfp4.md) | NVFP4 engine spec — unpack E2M1→fp16→`fdot2`, E4M3×16 scale |
| [kv-int8.md](kv-int8.md) | INT8 KV spec — fused dequant in `fa_rdna2`, per-token-head |
| [int2.md](int2.md) | INT2 + mixed INT2/INT4 MoE — unpack→`fdot2`, bitwidth grouped outside K |
| [w4a4.md](w4a4.md) | W4A4 explore — integer `sdot8` i4×i4; MXFP4/NVFP4 A4 stays `fdot2` |
| [mtp.md](mtp.md) | MTP — native heads, fat tile first, no new DOT |
| [dflash.md](dflash.md) | DFlash — parallel block draft, same fat-tile gate |
| [dspark.md](dspark.md) | DSpark — DFlash + Markov + confidence schedule |
| [qwen35.md](qwen35.md) | Qwen3.5 hybrid GDN + Gemma `(1+w)` — steal gates, not their Triton AWQ |
| [vllm-sglang-map.md](vllm-sglang-map.md) | What stock vLLM/SGLang actually run on gfx1030 vs CDNA |
| [prefill-decode.md](prefill-decode.md) | The load-bearing split and kernel hit-list |
| [attention-dispatch.md](attention-dispatch.md) | fa_rdna2 vs Triton, head-64 hole, short vs split-K, Sage |
| [sage-attention.md](sage-attention.md) | Sage INT8 QK / sdot4 — prefill only, check vs impl later |
| [batching.md](batching.md) | Continuous batching, chunked prefill, MBT |
| [pd-disagg.md](pd-disagg.md) | Why PD is not a win on this box |
| [specdec.md](specdec.md) | Verify is extend; skip until q>1 exists |
| [kv-quant-offload.md](kv-quant-offload.md) | FP8-KV is not a vLLM path here |
| [moe.md](moe.md) | Expert offload + gfx10xx whitelist |
| [alt-engines.md](alt-engines.md) | What to steal from llama.cpp / ExLlama / ktransformers / Llaminar / hippih |
| [notes/](notes/README.md) | Source digests (Kiely, ikantkode, session dump). Not the contract. |

Human branch is now **`rdna2_extras`** (vLLM release + overlay). `perf/rdna2_w4a16` is historical. In-house engine: [hippih.md](hippih.md). Tickets: [project 4](https://github.com/users/BlivionIaG/projects/4).
