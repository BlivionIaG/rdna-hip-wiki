# Idle upstream awareness 2026-09-23

Window ≈ 2026-09-22 16:21 → 2026-09-23 ~16:24 Europe/Paris. Prior watches + topic PRs (`gh pr view` / GitHub search). Portable merges (AuxOutput PD narrow, eager JIT warmup skip, EP remote-expert skip, hybrid internal demote, MoE scatter row bounds, quantized-cache staging watch); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path) — open [#57807](https://github.com/vllm-project/vllm/pull/57807) / [#55507](https://github.com/vllm-project/vllm/pull/55507) / [#57908](https://github.com/vllm-project/vllm/pull/57908) **reinforce**, do not lift. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182).

## Take Later (portable ideas only)

- AuxOutput KV reject is a PD-connector allowlist, not a blanket AuxOutput ban — [vLLM #58150](https://github.com/vllm-project/vllm/pull/58150) **MERGED** (**Leave** PD product; PD **Dead** lab).
- Eager mode must not pay JIT warmup for short-lived processes — [vLLM #58197](https://github.com/vllm-project/vllm/pull/58197) **MERGED**.
- DCP ranks stay host scalars in attn-metadata helpers (no sync H2D) — [vLLM #58169](https://github.com/vllm-project/vllm/pull/58169) **MERGED**.
- EP launch must not bill M-blocks for remote experts (`ignore_invalid_experts`) — [vLLM #58051](https://github.com/vllm-project/vllm/pull/58051) **MERGED** (**Leave** DeepEP).
- Config feature disables that affect hybrid/Mamba must run before hooks that bake derived state — [vLLM #58287](https://github.com/vllm-project/vllm/pull/58287) **MERGED**.
- Process-global PrefetchOffloader resets on engine teardown after device sync — [vLLM #57834](https://github.com/vllm-project/vllm/pull/57834) **MERGED**.
- `NCCL_ALGO` allreduce prefix must not disable TP/PP Broadcast/AllGather — [vLLM #58179](https://github.com/vllm-project/vllm/pull/58179) **MERGED** (**Leave** NCCL version pin).
- Hybrid radix eviction demotes recurrent Mamba/KDA state on internal nodes too — [SGLang #40680](https://github.com/sgl-project/sglang/pull/40680) **MERGED**.
- Quantized KV block scales travel as page-indexed extra state with KV — [SGLang #40792](https://github.com/sgl-project/sglang/pull/40792) **MERGED** (**Leave** NIXL; PD **Dead** lab).
- MoE padded-row bounds keyed on scatter/gather mode — [SGLang #40672](https://github.com/sgl-project/sglang/pull/40672) **MERGED**.
- Prefill fp16 staging peak sized once to `max_num_tokens` (not power-of-two doubling) — [ExL #401](https://github.com/turboderp-org/exllamav3/pull/401) **OPEN** (**Leave** CUDA sizes).
- MoE top-k path must not skip fire — [llama.cpp #28432](https://github.com/ggml-org/llama.cpp/pull/28432) **MERGED** (awareness).
- APC reinforce (state≠KV): separate aligned-metadata vs state-copy contexts — [vLLM #57807](https://github.com/vllm-project/vllm/pull/57807) **OPEN**; seed align index with mamba block_size — [#55507](https://github.com/vllm-project/vllm/pull/55507) **OPEN**; align CoW — [#57908](https://github.com/vllm-project/vllm/pull/57908) **OPEN**. Offload watches: layout-namespaced cache [#58196](https://github.com/vllm-project/vllm/pull/58196); request-free prefetch [#58198](https://github.com/vllm-project/vllm/pull/58198); tiering off scheduler [#58168](https://github.com/vllm-project/vllm/pull/58168); breakable under BATCH_INVARIANT [#57586](https://github.com/vllm-project/vllm/pull/57586).
- Status flips: [#58150](https://github.com/vllm-project/vllm/pull/58150) / [#58197](https://github.com/vllm-project/vllm/pull/58197) / [#58169](https://github.com/vllm-project/vllm/pull/58169) / [#58051](https://github.com/vllm-project/vllm/pull/58051) / [#58287](https://github.com/vllm-project/vllm/pull/58287) / [#57834](https://github.com/vllm-project/vllm/pull/57834) / [#58179](https://github.com/vllm-project/vllm/pull/58179) / [SGLang #40680](https://github.com/sgl-project/sglang/pull/40680) / [#40792](https://github.com/sgl-project/sglang/pull/40792) / [#40672](https://github.com/sgl-project/sglang/pull/40672) / [llama #28432](https://github.com/ggml-org/llama.cpp/pull/28432) MERGED; [#58113](https://github.com/vllm-project/vllm/pull/58113) / [SGLang #39417](https://github.com/sgl-project/sglang/pull/39417) / [ExL #396](https://github.com/turboderp-org/exllamav3/pull/396)/[#402](https://github.com/turboderp-org/exllamav3/pull/402) CLOSED. Still open core: vLLM #57158/#57183/#57180/#57128/#56953/#56971/#57010/#56936/#56957/#57182/#56455/#56243/#56177/#56869/#57908/#57785/#57881/#57943/#57959/#57569/#54743/#56466/#53520/#56089/#57184/#57170/#58108/#57807/#55507/#58196/#58198/#58168/#57586; SGLang #39799/#39605/#39575/#39606/#39815/#38887/#38663/#40166/#40159/#39787/#39803/#39960/#39792/#40571/#40543; llama.cpp #28873/#28927/#29019/#29191/#29208; ExL #379/#315/#392/#401; KT #2190; NIXL #2196/#2244/#2252/#2241/#2225.

## Leave

- Custom ROCm AR [#57182](https://github.com/vllm-project/vllm/pull/57182); RDNA3/4 Triton prefill tile [#58225](https://github.com/vllm-project/vllm/pull/58225); AITER/ROCm CI/TheRock/TurboQuant/XPU; HiCache/Mooncake/NIXL/MoRIIO/UCX product ([#2278](https://github.com/ai-dynamo/nixl/pull/2278), [#1856](https://github.com/ai-dynamo/nixl/pull/1856)); Ascend/NPU; DeepGEMM/NVFP4/Blackwell; llama DSpark/DFlash model objects [#29226](https://github.com/ggml-org/llama.cpp/pull/29226)/[#28890](https://github.com/ggml-org/llama.cpp/pull/28890); HIP ISA; claimed tok/s.

## Sources

Linked inline on this page (each URL checked 2026-10-11):

- <https://github.com/vllm-project/vllm/pull/57807>
- <https://github.com/vllm-project/vllm/pull/55507>
- <https://github.com/vllm-project/vllm/pull/57908>
- <https://github.com/vllm-project/vllm/pull/57182>
- <https://github.com/vllm-project/vllm/pull/58150>
- <https://github.com/vllm-project/vllm/pull/58197>
- <https://github.com/vllm-project/vllm/pull/58169>
- <https://github.com/vllm-project/vllm/pull/58051>
- <https://github.com/vllm-project/vllm/pull/58287>
- <https://github.com/vllm-project/vllm/pull/57834>
- <https://github.com/vllm-project/vllm/pull/58179>
- <https://github.com/sgl-project/sglang/pull/40680>
- <https://github.com/sgl-project/sglang/pull/40792>
- <https://github.com/sgl-project/sglang/pull/40672>
- <https://github.com/turboderp-org/exllamav3/pull/401>
- <https://github.com/ggml-org/llama.cpp/pull/28432>
- <https://github.com/vllm-project/vllm/pull/58196>
- <https://github.com/vllm-project/vllm/pull/58198>
- <https://github.com/vllm-project/vllm/pull/58168>
- <https://github.com/vllm-project/vllm/pull/57586>
- <https://github.com/vllm-project/vllm/pull/58113>
- <https://github.com/sgl-project/sglang/pull/39417>
- <https://github.com/turboderp-org/exllamav3/pull/396>
- <https://github.com/turboderp-org/exllamav3/pull/402>
- <https://github.com/vllm-project/vllm/pull/58225>
- <https://github.com/ai-dynamo/nixl/pull/2278>
- <https://github.com/ai-dynamo/nixl/pull/1856>
- <https://github.com/ggml-org/llama.cpp/pull/29226>
- <https://github.com/ggml-org/llama.cpp/pull/28890>

