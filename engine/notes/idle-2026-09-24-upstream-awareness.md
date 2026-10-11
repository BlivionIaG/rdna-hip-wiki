# Idle upstream awareness 2026-09-24

Window ≈ 2026-09-23 16:24 → 2026-09-24 ~16:14 Europe/Paris. Prior watches + topic PRs (`gh` search API). Portable merges (BATCH_INVARIANT breakable CG, mixed FULL prefill capture, KV offload event batch, adaptive long-prefill share, multi-backend block_size, DSpark≠MTP, DP×spec prefill, HiCache order-stable reclaim, NCCL×DP-attn graph reg off); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path) — open [#57807](https://github.com/vllm-project/vllm/pull/57807) / [#55507](https://github.com/vllm-project/vllm/pull/55507) / [#57908](https://github.com/vllm-project/vllm/pull/57908) **still reinforce**, do not lift. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182).

## Take Later (portable ideas only)

- Breakable CG default under `VLLM_BATCH_INVARIANT` so tuned matmul sees runtime M — [vLLM #57586](https://github.com/vllm-project/vllm/pull/57586) **MERGED** (was open watch).
- Mixed FULL capture must cover prefill rows (query-length bound over whole token batch) — [vLLM #58275](https://github.com/vllm-project/vllm/pull/58275) **MERGED**.
- Offload event payloads survive full `take_events` batch translation — [vLLM #57453](https://github.com/vllm-project/vllm/pull/57453) **MERGED**.
- Long-prefill threshold shares remaining budget under contention — [vLLM #58459](https://github.com/vllm-project/vllm/pull/58459) **MERGED** (mates solo soften [#57951](https://github.com/vllm-project/vllm/pull/57951)).
- Multi-backend KV `block_size` = ∩ of supports — [vLLM #49845](https://github.com/vllm-project/vllm/pull/49845) **MERGED**.
- DSpark parallel width ≠ MTP stage / `n_predict` — [vLLM #54631](https://github.com/vllm-project/vllm/pull/54631) **MERGED**.
- Encoder-cache hit validates embedding count — [vLLM #57696](https://github.com/vllm-project/vllm/pull/57696) **MERGED**.
- Full prefill CG replay view carries captured-bucket `input_ids` — [SGLang #40851](https://github.com/sgl-project/sglang/pull/40851) **MERGED**.
- DP-attn decode graphs disable NCCL graph buffer registration — [SGLang #40649](https://github.com/sgl-project/sglang/pull/40649) **MERGED**.
- HiCache host reclaim order-stable across pools/ranks — [SGLang #40512](https://github.com/sgl-project/sglang/pull/40512) **MERGED** (**Leave** HiCache product).
- Preserve speculative during prefill across DP (+ TBO child flag) — [SGLang #40118](https://github.com/sgl-project/sglang/pull/40118) / [#41096](https://github.com/sgl-project/sglang/pull/41096) **MERGED**.
- Ragged full-attn prefill plans over `prefix_lens`; SWA free via `swa_dead_lo` — [SGLang #40796](https://github.com/sgl-project/sglang/pull/40796) / [#40798](https://github.com/sgl-project/sglang/pull/40798) **MERGED**.
- Runner-local hybrid SWA layer ids (do not mutate shared ModelConfig) — [SGLang #40983](https://github.com/sgl-project/sglang/pull/40983) **MERGED**.
- Status flips: [#57586](https://github.com/vllm-project/vllm/pull/57586) MERGED; [ExL #401](https://github.com/turboderp-org/exllamav3/pull/401) CLOSED unmerged (staging-peak idea may reappear). Still open core: vLLM #57158/#57183/#57180/#57128/#56953/#56971/#57010/#56936/#56957/#57182/#56455/#56243/#56177/#56869/#57908/#57785/#57881/#57943/#57959/#57569/#54743/#56466/#53520/#56089/#57184/#57170/#58108/#57807/#55507/#58196/#58198/#58168; SGLang #39799/#39605/#39575/#39606/#39815/#38887/#38663/#40166/#40159/#39787/#39803/#39960/#39792/#40571/#40543/#40857/#40907; llama.cpp #28873/#28927/#29019/#29191/#29208; ExL #379/#315/#392; KT #2190; NIXL #2196/#2244/#2252/#2241/#2225.

## Leave

- Custom ROCm AR [#57182](https://github.com/vllm-project/vllm/pull/57182); AITER/ROCm MXFP8 sparse / Mooncake+NIXL PD CI / MoRI / XPU GRAPH; HiCache/Mooncake/NIXL/MoRIIO/UCX product ([#58188](https://github.com/vllm-project/vllm/pull/58188), [#53300](https://github.com/vllm-project/vllm/pull/53300), [#2280](https://github.com/ai-dynamo/nixl/pull/2280), [#40238](https://github.com/sgl-project/sglang/pull/40238) as PD product); Ascend/NPU; Cute-DSL AR / gfx950 QSA/MXFP4; llama DFlash/DSpark convert [#29339](https://github.com/ggml-org/llama.cpp/pull/29339); ExL MemAvailable [#398](https://github.com/turboderp-org/exllamav3/pull/398); HIP ISA; claimed tok/s.
