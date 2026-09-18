# Idle upstream awareness 2026-09-17

Window ≈ 2026-09-16 16:14 → 2026-09-17 ~16:06 Europe/Paris. `gh search prs --updated` / `gh pr view` on prior watches + new topic PRs. Substantive portable merges (breakable prefill, offload scratch skip, ROCm graph converge); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path). No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182).

## Take Later (portable ideas only)

- Breakable CG **prefill** landed upstream on SGLang ROCm DSV4 (opt-in) — [SGLang #37810](https://github.com/sgl-project/sglang/pull/37810): metadata-stable capture/replay, host-proven scalars, mixed PD under DP stays eager. **Leave** their HIP radix/FP4/gfx950; elevates prior "if breakable lands on ROCm" watch — still not a gfx1030 extras dest path.
- ROCm graph-path converge: restore side-stream capture profiling + drop obsolete DS V4→NONE under MRV2 — [vLLM #57229](https://github.com/vllm-project/vllm/pull/57229). FULL bucket cliff when `max_num_seqs` ∤ 8 — open [#57355](https://github.com/vllm-project/vllm/pull/57355). DCP-sharded DFlash capture needs rank align — open [#56869](https://github.com/vllm-project/vllm/pull/56869).
- KV/offload: scratch-group skip MERGED [#57145](https://github.com/vllm-project/vllm/pull/57145); `skip_reading_prefix_cache` for connector hits MERGED [#57269](https://github.com/vllm-project/vllm/pull/57269); chunked host-register [#51081](https://github.com/vllm-project/vllm/pull/51081); compact MLA rows [#56799](https://github.com/vllm-project/vllm/pull/56799). Mamba radix-page checkpoint depth MERGED [SGLang #39115](https://github.com/sgl-project/sglang/pull/39115). T-LRU opt-in radix eviction MERGED [#34012](https://github.com/sgl-project/sglang/pull/34012).
- Spec / batch: PP×EAGLE/MTP opt-in (PCIe motivation) MERGED [SGLang #30775](https://github.com/sgl-project/sglang/pull/30775); draft embed `is_token_ids` open [vLLM #57356](https://github.com/vllm-project/vllm/pull/57356); decode-priority cadence open [#56848](https://github.com/vllm-project/vllm/pull/56848); llama.cpp speculative batch-order open [#29019](https://github.com/ggml-org/llama.cpp/pull/29019).
- Status flips: [#57000](https://github.com/vllm-project/vllm/pull/57000) CLOSED (superseded by MERGED [#57050](https://github.com/vllm-project/vllm/pull/57050)); [SGLang #39587](https://github.com/sgl-project/sglang/pull/39587) CLOSED unmerged. Still open core watches: vLLM #56953/#56971/#57010/#56936/#56957/#57180/#57128/#57158/#57170/#57160/#57183/#57182; SGLang #39799/#39605/#39575/#39606/#39474(already MERGED prior)/#38887/#38663; llama.cpp #28873/#28927; ExL #379/#377/#376/#315; KT #2210/#2209; NIXL #2244/#2252/#2196.

## Leave

- Custom ROCm AR barrier [#57182](https://github.com/vllm-project/vllm/pull/57182); EPD image batching / CPU×NIXL affinity; HiCache/Mooncake/NIXL/`nixl_rocm` product; Ascend/NPU/AITER/FlyDSL; DeepGEMM/NVFP4/Blackwell/FlashMLA; gfx950 `vattn_asm` [#39513](https://github.com/sgl-project/sglang/pull/39513); ExL sc_measure [#364](https://github.com/turboderp-org/exllamav3/pull/364); claimed tok/s.
