# Engine

vLLM / SGLang dispatch, prefill vs decode, batching, PD, specdec, MoE, KV. Owned by LLM_Inference_specialist.

Target box: 4× V620 (gfx1030), ROCm 7.2. Hardware facts live in [silicon/](../silicon/README.md). Format contracts live in [kernels/](../kernels/README.md).

Do not invent tok/s, IC TB/s, or a P2P-works claim.

| Page | Contents |
|---|---|
| [full-map.md](full-map.md) | One-page verdict + technique table |
| [vllm-sglang-map.md](vllm-sglang-map.md) | What stock vLLM/SGLang actually run on gfx1030 vs CDNA |
| [prefill-decode.md](prefill-decode.md) | The load-bearing split and kernel hit-list |
| [attention-dispatch.md](attention-dispatch.md) | fa_rdna2 vs Triton, head-64 hole, short vs split-K, Sage |
| [batching.md](batching.md) | Continuous batching, chunked prefill, MBT |
| [pd-disagg.md](pd-disagg.md) | Why PD is not a win on this box |
| [specdec.md](specdec.md) | Verify is extend; skip until q>1 exists |
| [kv-quant-offload.md](kv-quant-offload.md) | FP8-KV is not a vLLM path here |
| [moe.md](moe.md) | Expert offload + gfx10xx whitelist |
| [alt-engines.md](alt-engines.md) | What to steal from llama.cpp / ExLlama / ktransformers |

Fork branch `perf/rdna2_w4a16` is human-only. Tickets go on [project 4](https://github.com/users/BlivionIaG/projects/4).
