# Idle upstream awareness 2026-09-21

Window ≈ 2026-09-18 16:10 → 2026-09-21 ~16:23 Europe/Paris. `gh search prs --updated` / GitHub `merged:>=2026-09-18` / `gh pr view` on prior watches + topic PRs. Substantive portable merges (SimpleCPU `prefix_cacheable`, idle-DP dummy draft, NIXL notify≠receive, FULL-only mHC overlap, optimistic Mamba prefill, ExL SWA ring / moe_cpu_split / DFlash2); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path) — open [#57908](https://github.com/vllm-project/vllm/pull/57908) (align CoW) **reinforces**, does not lift. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182).

## Take Later (portable ideas only)

- SimpleCPU skip non-`prefix_cacheable` — [vLLM #56810](https://github.com/vllm-project/vllm/pull/56810) **MERGED**; [#56404](https://github.com/vllm-project/vllm/pull/56404) CLOSED. One gate across offload/connector/Mooncake.
- Idle-DP dummy draft must not write through stale block tables — [vLLM #56734](https://github.com/vllm-project/vllm/pull/56734) **MERGED**.
- NIXL notify ≠ receive — [vLLM #57570](https://github.com/vllm-project/vllm/pull/57570) **MERGED** (**Leave** NIXL product).
- FULL-only mHC overlap; capture `intermediate_tensors` parity — [vLLM #57874](https://github.com/vllm-project/vllm/pull/57874) / [#57745](https://github.com/vllm-project/vllm/pull/57745) **MERGED**. Side-stream join before breakable resume — [#57785](https://github.com/vllm-project/vllm/pull/57785) **OPEN**; Mamba eager-break under breakable CG — [#57881](https://github.com/vllm-project/vllm/pull/57881) **OPEN**.
- Align-mode Mamba CoW for shared APC snapshots — [vLLM #57908](https://github.com/vllm-project/vllm/pull/57908) **OPEN** (APC stays off).
- Optimistic Mamba prefill + pinned sampling meta + prefill-CG runner contracts — [SGLang #40184](https://github.com/sgl-project/sglang/pull/40184) / [#39777](https://github.com/sgl-project/sglang/pull/39777) / [#35452](https://github.com/sgl-project/sglang/pull/35452) **MERGED**. PD pack ≠ compute chunk — [#40376](https://github.com/sgl-project/sglang/pull/40376) **MERGED** (**Leave** PD product; PD **Dead** lab).
- ExL SWA ring + `infer_params.moe_cpu_split` + DFlash2 — [#377](https://github.com/turboderp-org/exllamav3/pull/377) / [#376](https://github.com/turboderp-org/exllamav3/pull/376) / [#334](https://github.com/turboderp-org/exllamav3/pull/334) **MERGED** (**Leave** CUDA / tok/s; UNC-26 `-cb 3inst` untouched). Host-bank MoE offload primitives — [vLLM #57943](https://github.com/vllm-project/vllm/pull/57943) **OPEN**; llama expert stream — [#29191](https://github.com/ggml-org/llama.cpp/pull/29191) **OPEN** — expert DRAM still **Dead** when MoE fits.
- Status flips: [#56810](https://github.com/vllm-project/vllm/pull/56810) / [#56734](https://github.com/vllm-project/vllm/pull/56734) / [#57570](https://github.com/vllm-project/vllm/pull/57570) / [SGLang #40184](https://github.com/sgl-project/sglang/pull/40184) / [#39777](https://github.com/sgl-project/sglang/pull/39777) MERGED; [#56404](https://github.com/vllm-project/vllm/pull/56404) CLOSED. Still open core: vLLM #57158/#57183/#57180/#57128/#56953/#56971/#57010/#56936/#56957/#57182/#56455/#56243/#56177/#56869 (+ #57908/#57785/#57881/#57943); SGLang #39799/#39605/#39575/#39606/#38887/#38663/#40166/#40159; llama.cpp #28873/#28927/#29019; ExL #379/#380/#315; KT #2190; NIXL #2244/#2252/#2196.

## Leave

- Custom ROCm AR [#57182](https://github.com/vllm-project/vllm/pull/57182); SWA bounded-replay ROCm disable [#57906](https://github.com/vllm-project/vllm/pull/57906); HiCache/Mooncake/NIXL/MoRIIO product; Ascend/NPU/AITER/FlyDSL; DeepGEMM/NVFP4/Blackwell; ExL sc_measure [#364](https://github.com/turboderp-org/exllamav3/pull/364); claimed tok/s.

## Sources

Linked inline on this page (each URL checked 2026-10-11):

- <https://github.com/vllm-project/vllm/pull/57908>
- <https://github.com/vllm-project/vllm/pull/57182>
- <https://github.com/vllm-project/vllm/pull/56810>
- <https://github.com/vllm-project/vllm/pull/56404>
- <https://github.com/vllm-project/vllm/pull/56734>
- <https://github.com/vllm-project/vllm/pull/57570>
- <https://github.com/vllm-project/vllm/pull/57874>
- <https://github.com/vllm-project/vllm/pull/57745>
- <https://github.com/vllm-project/vllm/pull/57785>
- <https://github.com/vllm-project/vllm/pull/57881>
- <https://github.com/sgl-project/sglang/pull/40184>
- <https://github.com/sgl-project/sglang/pull/39777>
- <https://github.com/sgl-project/sglang/pull/35452>
- <https://github.com/sgl-project/sglang/pull/40376>
- <https://github.com/turboderp-org/exllamav3/pull/377>
- <https://github.com/turboderp-org/exllamav3/pull/376>
- <https://github.com/turboderp-org/exllamav3/pull/334>
- <https://github.com/vllm-project/vllm/pull/57943>
- <https://github.com/ggml-org/llama.cpp/pull/29191>
- <https://github.com/vllm-project/vllm/pull/57906>
- <https://github.com/turboderp-org/exllamav3/pull/364>

