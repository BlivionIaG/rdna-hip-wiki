# Idle upstream awareness 2026-09-22

Window ≈ 2026-09-21 16:23 → 2026-09-22 ~16:21 Europe/Paris. `gh search prs --updated` / GitHub `merged:>=2026-09-21` / `gh pr view` on prior watches + topic PRs. Portable merges (NIXL DCP×MLA, DFlash async sched, solo long-prefill soften, CG stream-explosion fixes, PD abort-ACK drain, ReplaySSM KDA gate, ExL stream-decode watch); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path) — open [#57908](https://github.com/vllm-project/vllm/pull/57908) (align CoW) **reinforces**, does not lift; SGLang [#40517](https://github.com/sgl-project/sglang/pull/40517) capability gate does not change that. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182).

## Take Later (portable ideas only)

- NIXL DCP pulls across MLA cache regions (per-rank reads + prefix skip) — [vLLM #57389](https://github.com/vllm-project/vllm/pull/57389) **MERGED** (**Leave** NIXL product; PD **Dead** lab).
- DFlash joins async-scheduling allowlist (shared GPU draft-token contract) — [vLLM #58065](https://github.com/vllm-project/vllm/pull/58065) **MERGED**.
- Soften `long_prefill_token_threshold` when scheduler has a single request — [vLLM #57951](https://github.com/vllm-project/vllm/pull/57951) **MERGED**.
- Capture-stream collectives / main-before-aux record order to stop CG stream explosion — [SGLang #40658](https://github.com/sgl-project/sglang/pull/40658) / [#40640](https://github.com/sgl-project/sglang/pull/40640) **MERGED**; side-stream join before breakable resume — [vLLM #57785](https://github.com/vllm-project/vllm/pull/57785) **OPEN**; Mamba eager-break under breakable CG — [#57881](https://github.com/vllm-project/vllm/pull/57881) **OPEN**.
- PD abort ACK lifecycle spans outstanding KV drain — [SGLang #40645](https://github.com/sgl-project/sglang/pull/40645) / [#40711](https://github.com/sgl-project/sglang/pull/40711) **MERGED** (**Leave** Mooncake; PD **Dead** lab).
- ReplaySSM/KDA predicates are capability-based (not Kimi-only config) — [SGLang #40517](https://github.com/sgl-project/sglang/pull/40517) **MERGED** (APC stays off).
- Streamed single-row MoE decode from pinned arena — [ExL #392](https://github.com/turboderp-org/exllamav3/pull/392) **OPEN**; DFlash SWA draft bidirectionality — [#396](https://github.com/turboderp-org/exllamav3/pull/396) **OPEN**; host-bank MoE — [vLLM #57943](https://github.com/vllm-project/vllm/pull/57943) **OPEN**; llama expert stream — [#29191](https://github.com/ggml-org/llama.cpp/pull/29191) **OPEN** — expert DRAM still **Dead** when MoE fits. Hash-routing reject on monolithic MoE — [vLLM #57867](https://github.com/vllm-project/vllm/pull/57867) **MERGED**.
- Align-mode Mamba CoW — [vLLM #57908](https://github.com/vllm-project/vllm/pull/57908) **OPEN** (APC stays off). AuxOutput KV reject narrowed to known PD connectors — [#58150](https://github.com/vllm-project/vllm/pull/58150) **OPEN**.
- POSIX partial async I/O retry — [NIXL #2224](https://github.com/ai-dynamo/nixl/pull/2224) **MERGED** (**Leave** product).
- Status flips: [#57389](https://github.com/vllm-project/vllm/pull/57389) / [#58065](https://github.com/vllm-project/vllm/pull/58065) / [#57951](https://github.com/vllm-project/vllm/pull/57951) / [SGLang #40658](https://github.com/sgl-project/sglang/pull/40658) / [#40640](https://github.com/sgl-project/sglang/pull/40640) / [#40645](https://github.com/sgl-project/sglang/pull/40645) / [#40517](https://github.com/sgl-project/sglang/pull/40517) / [NIXL #2224](https://github.com/ai-dynamo/nixl/pull/2224) MERGED; [ExL #380](https://github.com/turboderp-org/exllamav3/pull/380) / [SGLang #40563](https://github.com/sgl-project/sglang/pull/40563) CLOSED. Still open core: vLLM #57158/#57183/#57180/#57128/#56953/#56971/#57010/#56936/#56957/#57182/#56455/#56243/#56177/#56869/#57908/#57785/#57881/#57943 (+ #58150); SGLang #39799/#39605/#39575/#39606/#38887/#38663/#40166/#40159/#40571; llama.cpp #28873/#28927/#29019/#29191; ExL #379/#315/#392/#396; KT #2190; NIXL #2244/#2252/#2196.

## Leave

- Custom ROCm AR [#57182](https://github.com/vllm-project/vllm/pull/57182); AITER MXFP4 EP [#57866](https://github.com/vllm-project/vllm/pull/57866); HiCache/Mooncake/NIXL/MoRIIO/UCX product; Ascend/NPU/AITER/FlyDSL; DeepGEMM/NVFP4/Blackwell; RDNA3/4 kv-split [#58156](https://github.com/vllm-project/vllm/pull/58156); claimed tok/s.
