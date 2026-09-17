# Idle upstream awareness 2026-09-15

Window ≈ 2026-09-14 → 2026-09-15. `gh search` / `gh pr view` on prior scan artifacts; thin status refresh only. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant.

## Take Later (portable ideas only)

- Capture-order: record side-stream work immediately before join so replay stays on the first-recorded parent stream — [SGLang #39420](https://github.com/sgl-project/sglang/pull/39420) **MERGED**.
- MoE zero-copy: page-locked shared expert arena DMA without host staging memcpy — [ExL #341](https://github.com/turboderp-org/exllamav3/pull/341) **MERGED**; placement flag must live in `infer_params` [#376](https://github.com/turboderp-org/exllamav3/pull/376) open.
- Hybrid KV admission after remote resume must use physical Mamba extent, not logical null-gap count — [vLLM #57000](https://github.com/vllm-project/vllm/pull/57000) open; per-group CPU offload slot sizing [#56953](https://github.com/vllm-project/vllm/pull/56953); decode-boundary Mamba save + config-keyed store hashes [#56971](https://github.com/vllm-project/vllm/pull/56971) (Leave Mooncake).
- Rebuild fused-MoE around resident weights (layout-compatible pairs + dry_run) — [vLLM #57010](https://github.com/vllm-project/vllm/pull/57010) open.
- Host-staged all-reduce for two GPUs **without peer access** (large msgs outside graph; decode untouched) — [SGLang #39605](https://github.com/sgl-project/sglang/pull/39605) open; PCIe-no-P2P idea only.
- Resolve policies before graph/memory sizing — [SGLang #39587](https://github.com/sgl-project/sglang/pull/39587); QSA extend must tolerate unaligned chunk-cache prefixes [#39575](https://github.com/sgl-project/sglang/pull/39575); `page_unified` staged write-back [#39606](https://github.com/sgl-project/sglang/pull/39606).
- Spec: Gemma4 MTP must not swap draft into target `VllmConfig` under EP — [vLLM #56936](https://github.com/vllm-project/vllm/pull/56936); DSpark PP prefill-only + KV transfer [#56957](https://github.com/vllm-project/vllm/pull/56957) draft; PARTIAL_ONLY MTP checkpoints [llama.cpp #28873](https://github.com/ggml-org/llama.cpp/pull/28873) still open; causal-attn must not force sched re-reserve [#28927](https://github.com/ggml-org/llama.cpp/pull/28927).
- KT: gate pre-permuted activations to prefill-sized rows — [#2209](https://github.com/kvcache-ai/ktransformers/pull/2209) open (refinement of MERGED [#2205](https://github.com/kvcache-ai/ktransformers/pull/2205)).
- NIXL cancel≠release: keep pollable [#2244](https://github.com/ai-dynamo/nixl/pull/2244) + POSIX refuse-while-active [#2252](https://github.com/ai-dynamo/nixl/pull/2252); allocator split [#2196](https://github.com/ai-dynamo/nixl/pull/2196) still open.

## Leave

- Mega-mHC DeepGEMM / SM90–SM120 / claimed µs ([vLLM #56962](https://github.com/vllm-project/vllm/pull/56962)); Triton merge_attn_states; ROCm skinny-GEMM / MLA LSE HIP ISA; Ascend/NPU/AITER/FlyDSL/Hopper DeepEP; FlashInfer/Blackwell; Mooncake/NIXL/`nixl_rocm` product; ExL community HIP #365–367; SM120 GDN #369 Dead; AVX/CPU KT backends; claimed tok/s or ~2×.
