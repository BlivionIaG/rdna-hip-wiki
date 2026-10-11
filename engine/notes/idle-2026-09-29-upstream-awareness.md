# Idle upstream awareness 2026-09-29 (morning)

Window ≈ 2026-09-28 10:51 → 2026-09-29 10:58 Europe/Paris. Prior watches + topic PRs (`gh`). Portable merges (PLE meta hygiene, KV pointer refresh after re-bind, spec-driven mamba radix leaves, PCIe-IPC AR methodology, hc_post gate fuse idea); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path) — open [#57807](https://github.com/vllm-project/vllm/pull/57807) / [#55507](https://github.com/vllm-project/vllm/pull/55507) / [#57908](https://github.com/vllm-project/vllm/pull/57908) / RecoverSSM [#58863](https://github.com/vllm-project/vllm/pull/58863) / **new** align checkpoint [#59175](https://github.com/vllm-project/vllm/pull/59175) / [#59146](https://github.com/vllm-project/vllm/pull/59146) **still reinforce / watch**, do not lift. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182); PCIe-IPC [#34528](https://github.com/sgl-project/sglang/pull/34528) is methodology only. Mojo≠produce.

## Take Later (portable ideas only)

- PLE metadata construction cut under mixed speculative batches — [vLLM #58114](https://github.com/vllm-project/vllm/pull/58114) **MERGED**.
- Capture-profile KV rebinds refresh base pointers — [vLLM #58814](https://github.com/vllm-project/vllm/pull/58814) **MERGED**.
- Mamba align checkpoint reservation/eviction + sparse retention keep prompt-end — [vLLM #59175](https://github.com/vllm-project/vllm/pull/59175) / [#59146](https://github.com/vllm-project/vllm/pull/59146) **OPEN** (**Must** awareness; **not** APC lift).
- Hybrid GDN+DFlash draft pool collapse RFC — [vLLM #59117](https://github.com/vllm-project/vllm/pull/59117) **OPEN**.
- KDA state honors `--mamba-ssm-cache-dtype` — [vLLM #59147](https://github.com/vllm-project/vllm/pull/59147) **OPEN**.
- Spec-driven mamba radix leaves for linear-attn — [SGLang #41165](https://github.com/sgl-project/sglang/pull/41165) **MERGED** (**Import Later** HybridRadix).
- PCIe-IPC AR staging/island ideas — [SGLang #34528](https://github.com/sgl-project/sglang/pull/34528) **MERGED** (**Leave** FlashInfer; PYNCCL pin holds).
- HIP graph bf16 SSM/KDA pad+dtype — [SGLang #41570](https://github.com/sgl-project/sglang/pull/41570) **OPEN**.
- Fuse hc_post gate chain (qwen4exp) methodology — [llama.cpp #29520](https://github.com/ggml-org/llama.cpp/pull/29520) **MERGED** (**Leave** Vulkan; **not** APC).
- Status: prior-watch OPEN→MERGED flips **none** in window. Still open core: vLLM #58642/#58676/#58863/#57908/#57807/#55507/#57785/#57881/#57182 + prior; SGLang #41218/#41171/#41173/#41182/#39605/#39606 + prior; llama.cpp #29353/#29572 (Leave fattn); ExL #419/#379/#392/#315; KT #2190.

## Leave

- Custom ROCm AR [#57182](https://github.com/vllm-project/vllm/pull/57182); FlashInfer PCIe-IPC objects [#34528](https://github.com/sgl-project/sglang/pull/34528); NIXL/AITER shuffle PD [#59110](https://github.com/vllm-project/vllm/pull/59110)/[#59111](https://github.com/vllm-project/vllm/pull/59111); gfx950 aiter/mHC; DeepEP V2.5; FlashInfer SM100 Cake; HIP fattn [#29572](https://github.com/ggml-org/llama.cpp/pull/29572); claimed tok/s.
