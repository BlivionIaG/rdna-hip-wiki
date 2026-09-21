# Idle upstream awareness 2026-09-18

Window ≈ 2026-09-17 16:06 → 2026-09-18 ~16:10 Europe/Paris. `gh search prs --updated` / `--merged` / `gh pr view` on prior watches + new topic PRs. Substantive portable merges (FULL bucket grid, ROCm private KV pin, SPF, DP zero-token, KT placement map); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path). No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182).

## Take Later (portable ideas only)

- FULL cudagraph bucket grid at max load — [vLLM #57355](https://github.com/vllm-project/vllm/pull/57355) **MERGED**: `max_num_seqs` ∤ 8 drops last buckets to PIECEWISE. **Ops:** keep capture ladder aligned to ×8 when FULL matters.
- ROCm CPU KV offload private pins — [vLLM #57160](https://github.com/vllm-project/vllm/pull/57160) **MERGED**: shared mmap register fails on ROCm; private per-rank pin. Mates chunked [#51081](https://github.com/vllm-project/vllm/pull/51081). Awareness only if lab enables CPU offload.
- Draft embed mask under prompt_embeds+spec — [vLLM #57356](https://github.com/vllm-project/vllm/pull/57356) **MERGED**.
- SPF admission — [SGLang #40024](https://github.com/sgl-project/sglang/pull/40024) **MERGED**; DP idle-rank empty batch under breakable prefill — [#39899](https://github.com/sgl-project/sglang/pull/39899) **MERGED**; spec kernels JIT CUDA/ROCm — [#40033](https://github.com/sgl-project/sglang/pull/40033) **MERGED** (**Leave** JIT package).
- MoE placement map + prefill-only permute — [KT #2210](https://github.com/kvcache-ai/ktransformers/pull/2210) / [#2209](https://github.com/kvcache-ai/ktransformers/pull/2209) **MERGED**. Expert DRAM still **Dead**.
- Status flips: [#56458](https://github.com/vllm-project/vllm/pull/56458) / [#56848](https://github.com/vllm-project/vllm/pull/56848) / [SGLang #39387](https://github.com/sgl-project/sglang/pull/39387) CLOSED unmerged. Still open core: vLLM #57158/#57183/#57180/#57128/#56953/#56971/#57010/#56936/#56957/#57182/#56455/#56243/#56177/#56869; SGLang #39799/#39605/#39575/#39606/#38887/#38663/#40184/#40166/#40159; llama.cpp #28873/#28927/#29019; ExL #379/#380/#377/#376/#315; KT #2190; NIXL #2244/#2252/#2196.

## Leave

- Custom ROCm AR [#57182](https://github.com/vllm-project/vllm/pull/57182); ROCR segfault pin [#57328](https://github.com/vllm-project/vllm/pull/57328); HiCache/Mooncake/NIXL product; Ascend/NPU/AITER/FlyDSL; DeepGEMM/NVFP4/Blackwell; ExL sc_measure; claimed tok/s.
