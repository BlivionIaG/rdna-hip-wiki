# Engine

vLLM / SGLang dispatch, prefill vs decode, batching, PD, specdec, MoE, KV. Owned by LLM_Inference_specialist.

Target box: 4× V620 (gfx1030) on **one** 5-slot 88096. **Live ROCm 7.14.0** (attested). Official V620 matrix is still Ubuntu-only — [rocm-host.md](serving-stack/rocm-host.md). Hardware facts live in [silicon/](../silicon/README.md). Format contracts live in [kernels/](../kernels/README.md). VALU fire-list: [silicon/valu.md](../silicon/valu.md).

Do not invent tok/s, IC TB/s, or a P2P-works claim.

Session dump (2026-08-17/18): [notes/session-2026-08-17.md](notes/session-2026-08-17.md).

| Page | Contents |
|---|---|
| [full-map.md](overview/full-map.md) | One-page verdict + technique table |
| [coverage.md](overview/coverage.md) | Feature / quant / HIP kernel coverage + progress |
| [fork-delta.md](serving-stack/fork-delta.md) | Each added feature vs upstream — most incomplete, W4A16 dense is the bar |
| [rdna2-extras.md](serving-stack/rdna2-extras.md) | Release overlay — **`rdna2_extras`** = vLLM tag + gfx1030 work |
| [sglang-fork.md](serving-stack/sglang-fork.md) | SGLang rdna2 overlay — parallel serving path **after occupancy** |
| [rocm-host.md](serving-stack/rocm-host.md) | Official Ubuntu-only vs attested Fedora 43 / RHEL 10 / ROCm 7.14 |
| [hippih.md](alt-engines/hippih.md) | Older stub / Vega lab — see [hippihx.md](alt-engines/hippihx.md) for V620 op zoo |
| [hippihx.md](alt-engines/hippihx.md) | HIP op zoo (locked) — thin `rdna_extras`; migrate after ticket-26 |
| [infinity-cache.md](kv-cache/infinity-cache.md) | IC for inference — fit+reuse, no persist bit; W4 TP=4 keep |
| [cache-aware.md](scheduling/cache-aware.md) | Scheduling — KV prefix/APC vs IC residency; stock V1 only |
| [baseline-order.md](stock-baseline/baseline-order.md) | Stock-first research order: harness → Triton map → skinny → FA → HIP A/B → FlyDSL |
| [fp16-rdna2.md](quant-formats/fp16-rdna2.md) | Fastest gfx1030 FP16 — explicit `fdot2` + occupancy, phase-split |
| [triton-rocm.md](stock-baseline/triton-rocm.md) | Stock Triton / ROCM_ATTN / skinny dispatch on gfx1030 |
| [triton-flash-attention.md](stock-baseline/triton-flash-attention.md) | Stock Triton FA configs (ROCM_ATTN is Triton/Triton here) |
| [triton-tuning.md](stock-baseline/triton-tuning.md) | Tuning knobs only — never overwrite stock |
| [flydsl.md](alt-engines/flydsl.md) | FlyDSL gfx1030 gates — compiler yes, shipped MFMA/WMMA kernels no |
| [fp16-moe.md](moe/fp16-moe.md) | Native HIP FP16 MoE draft — decode skinny + prefill grouped, `fdot2` |
| [int8-moe.md](moe/int8-moe.md) | Dual-route INT8 MoE — W8A16 `fdot2` + W8A8 `sdot4` |
| [multi-tier.md](moe/multi-tier.md) | Hetero MoE — extras first, SGLang parallel, Llaminar, **hippih in-house** |
| [rocmfpx.md](quant-formats/rocmfpx.md) | ROCmFPX leverage — codebook→`perm`→`sdot4`, not a vLLM port |
| [llamacpp-rocmfpx.md](alt-engines/llamacpp-rocmfpx.md) | llama.cpp side project: `build-rdna2.sh` on V620 |
| [deepep.md](multi-gpu/deepep.md) | DeepEP insight — mapped-peer scatter/combine, not MORI/IBGDA |
| [plx.md](multi-gpu/plx.md) | PEX88096 / PEX8749 — ACS, PIX, Gen4 vs Gen3 ceiling, MMIO |
| [nvfp4.md](quant-formats/nvfp4.md) | NVFP4 engine spec — unpack E2M1→fp16→`fdot2`, E4M3×16 scale |
| [kv-int8.md](kv-cache/kv-int8.md) | INT8 KV spec — fused dequant in `fa_rdna2`, per-token-head |
| [int2.md](quant-formats/int2.md) | INT2 + mixed INT2/INT4 MoE — unpack→`fdot2`, bitwidth grouped outside K |
| [w4a4.md](quant-formats/w4a4.md) | W4A4 explore — integer `sdot8` i4×i4; MXFP4/NVFP4 A4 stays `fdot2` |
| [exl3.md](quant-formats/exl3.md) | EXL3 / QTIP trellis — no vLLM path; HIP would be unpack→`fdot2`, Later |
| [dsv4-flash-run.md](run-recipes/dsv4-flash-run.md) | DSv4 Flash on 120 GB — keep mxfp4, no drop-in EXL3 |
| [mtp.md](spec-decoding/mtp.md) | MTP — native heads, fat tile first, no new DOT |
| [dflash.md](spec-decoding/dflash.md) | DFlash — parallel block draft, same fat-tile gate |
| [dspark.md](spec-decoding/dspark.md) | DSpark — DFlash + Markov + confidence schedule |
| [qwen35.md](run-recipes/qwen35.md) | Qwen3.5 hybrid GDN + Gemma `(1+w)` — steal gates, not their Triton AWQ |
| [leapdragon.md](run-recipes/leapdragon.md) | leapdragon V620 recipe — steal grid/LDS/pitfalls, not tok/s or GPL plugins |
| [hipfire.md](alt-engines/hipfire.md) | warpfront/hipfire — Rust+HIP engine; gfx1030 portable DOT/sdot4; not extras pivot |
| [flashkda.md](attention/flashkda.md) | FlashKDA — steal KDA math / two-kernel split; CUTLASS SM90 is dead here |
| [vllm-sglang-map.md](serving-stack/vllm-sglang-map.md) | What stock vLLM/SGLang actually run on gfx1030 vs CDNA |
| [prefill-decode.md](scheduling/prefill-decode.md) | The load-bearing split and kernel hit-list |
| [attention-dispatch.md](attention/attention-dispatch.md) | fa_rdna2 vs Triton, head-64 hole, short vs split-K, Sage |
| [sage-attention.md](attention/sage-attention.md) | Sage INT8 QK / sdot4 — prefill only, check vs impl later |
| [batching.md](scheduling/batching.md) | Continuous batching, chunked prefill, MBT |
| [pd-disagg.md](multi-gpu/pd-disagg.md) | Why PD is not a win on this box |
| [specdec.md](spec-decoding/specdec.md) | Verify is extend; skip until q>1 exists |
| [kv-quant-offload.md](kv-cache/kv-quant-offload.md) | FP8-KV is not a vLLM path here |
| [moe.md](moe/moe.md) | Expert offload + gfx10xx whitelist |
| [alt-engines.md](alt-engines/alt-engines.md) | What to steal from llama.cpp / ExLlama / ktransformers / Llaminar / hippih / SGLang / hipfire |
| [notes/](notes/README.md) | Source digests (Kiely, ikantkode, session dump). Not the contract. |

Human branch is now **`rdna2_extras`** (vLLM release + overlay). SGLang overlay is **after occupancy** — [sglang-fork.md](serving-stack/sglang-fork.md). In-house engine: [hippih.md](alt-engines/hippih.md). Tickets: [project 4](https://github.com/users/BlivionIaG/projects/4).
