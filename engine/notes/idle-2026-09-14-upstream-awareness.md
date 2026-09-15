# Idle upstream awareness 2026-09-14

Window ≈ 2026-09-11 → 2026-09-14. `gh search` / `gh pr view` only; no clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. No new UNC. PD remains **Dead** for PCIe gfx1030.

## Take Later (portable ideas only)

- Connectors / offload gate hash/store/lookup on `prefix_cacheable` (not merely transferable): [vLLM #55027](https://github.com/vllm-project/vllm/pull/55027) MERGED; siblings [#54743](https://github.com/vllm-project/vllm/pull/54743)/[#56404](https://github.com/vllm-project/vllm/pull/56404)/[#56810](https://github.com/vllm-project/vllm/pull/56810) still open.
- Breakable CG replaces PIECEWISE only — must not override FULL under `FULL_AND_PIECEWISE`: [vLLM #56312](https://github.com/vllm-project/vllm/pull/56312) MERGED.
- Graph-pool borrow is stream-owned; wrong-stream borrow strands segments → OOM; dedup updates live exec: [SGLang #39176](https://github.com/sgl-project/sglang/pull/39176)/[#39177](https://github.com/sgl-project/sglang/pull/39177)/[#39178](https://github.com/sgl-project/sglang/pull/39178)/[#39180](https://github.com/sgl-project/sglang/pull/39180) MERGED.
- Unified-pool PD page envelopes / v2p / relocate-under-transfer / SWA-tail pricing: [SGLang #37506](https://github.com/sgl-project/sglang/pull/37506) MERGED; early-send device-agnostic wait_event [#39417](https://github.com/sgl-project/sglang/pull/39417) open.
- Mid-page KV free coalesce under DCP; page-stride scratch zero-fill: [SGLang #38941](https://github.com/sgl-project/sglang/pull/38941)/[#38851](https://github.com/sgl-project/sglang/pull/38851) MERGED.
- Chunked-prefill exact τ fill (page-ceil must not starve budget): [SGLang #32888](https://github.com/sgl-project/sglang/pull/32888) MERGED.
- Spec draft topology: DSpark `d2t` only if draft vocab strictly < target [vLLM #55133](https://github.com/vllm-project/vllm/pull/55133); PARTIAL_ONLY MTP checkpoints [llama.cpp #28873](https://github.com/ggml-org/llama.cpp/pull/28873) open.
- MoE: opt-in fused path + M=16/32/64 tile-band dispatch ideas [ExL #356](https://github.com/turboderp-org/exllamav3/pull/356)/[#357](https://github.com/turboderp-org/exllamav3/pull/357); permute activations once per expert [KT #2205](https://github.com/kvcache-ai/ktransformers/pull/2205).

## Leave

- NIXL / Mooncake / Ascend / Blackwell / gfx95 / `nixl_rocm` product objects and claimed tok/s.
- CUDA/AVX KPool, p2b fused-MoE kernels, FlashInfer/trtllm-gen, community ExL HIP ports (#365–367), SM120 GDN (#369 Dead).
- Expert DRAM offload as a capacity plan when MoE already fits after quant (**Dead**).
