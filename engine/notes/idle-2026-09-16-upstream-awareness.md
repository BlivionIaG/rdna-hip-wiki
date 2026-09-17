# Idle upstream awareness 2026-09-16

Window ≈ 2026-09-15 14:24 → 2026-09-16. `gh search` / `gh pr view` on prior scan artifacts + new updated PRs. Thin substantive window for extras; mostly status refresh + portable hygiene. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR barrier PRs.

## Take Later (portable ideas only)

- Hybrid APC × MTP: honor a single drop-margin / saved Mamba prompt-tail contract across attention and Mamba groups — [vLLM #57180](https://github.com/vllm-project/vllm/pull/57180) / [#57128](https://github.com/vllm-project/vllm/pull/57128) open; physical admit after remote load MERGED [#57050](https://github.com/vllm-project/vllm/pull/57050) (mates still-open [#57000](https://github.com/vllm-project/vllm/pull/57000)); transient checkpoints ≠ APC LRU MERGED [#56794](https://github.com/vllm-project/vllm/pull/56794).
- Graph hygiene: zero null KV block after capture ([#57158](https://github.com/vllm-project/vllm/pull/57158)); SparseMLA piecewise side-stream MERGED [#56825](https://github.com/vllm-project/vllm/pull/56825); padded-row ownership MERGED [SGLang #39574](https://github.com/sgl-project/sglang/pull/39574); reuse side streams [#39474](https://github.com/sgl-project/sglang/pull/39474); warn when decode coverage exceeds capture size [#57183](https://github.com/vllm-project/vllm/pull/57183).
- Chunked-prefill / offload: profile workspace from active backend [#57170](https://github.com/vllm-project/vllm/pull/57170); skip scratch offload groups [#57145](https://github.com/vllm-project/vllm/pull/57145); ROCm private pin offload path [#57160](https://github.com/vllm-project/vllm/pull/57160) (idea only — not a dest capacity plan).
- Spec / MoE watches: EAGLE draft pool × DCP replica [SGLang #39799](https://github.com/sgl-project/sglang/pull/39799); ExL DFlash2 [#379](https://github.com/turboderp-org/exllamav3/pull/379) + SWA ring [#377](https://github.com/turboderp-org/exllamav3/pull/377); KT BF16 placement map [#2210](https://github.com/kvcache-ai/ktransformers/pull/2210); ExL `#376` / `#315` / KT `#2209` still open.
- Sep-15 open watches **unchanged** (still open): vLLM #57000/#56953/#56971/#57010/#56936/#56957; SGLang #39605/#39587/#39575/#39606; llama.cpp #28873/#28927; NIXL #2244/#2252/#2196; ExL #376; KT #2209. Already MERGED prior: SGLang #39420, ExL #341.

## Leave

- Custom ROCm AR barrier [#57182](https://github.com/vllm-project/vllm/pull/57182); Mega-mHC/DeepGEMM/FlashMLA/NVFP4/Blackwell; Ascend/NPU/AITER/FlyDSL; Mooncake/NIXL/`nixl_rocm`/AIS product; ExL community HIP; claimed tok/s or accept-length tables; HIP ISA / RDNA3.5 MoE tile llama.cpp [#28935](https://github.com/ggml-org/llama.cpp/pull/28935).
