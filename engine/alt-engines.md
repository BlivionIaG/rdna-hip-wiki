# What to steal from other engines

Date: 2026-08-20. Sourced only. Product path: **vLLM first, SGLang overlay parallel after occupancy, then Llaminar, then hippih**. [sglang-fork.md](sglang-fork.md), [multi-tier.md](multi-tier.md), [hippih.md](hippih.md).

## Copy

1. Split decode vs prefill attention (llama.cpp VEC vs TILE). gfx1030 has no WMMA/MFMA → VEC/TILE only. Live `fa_rdna2` already does this split.
2. Keep weights quantized on device; fuse dequant into GEMV (GGML MMVQ).
3. Symmetric KV quant if fused FA.
4. Cell/slot KV + padded n_kv **or** a block table with **your** page size (not FA's 256).
5. V non-transposed iff fused FA.
6. Compute-offload for MoE, not weight ping-pong (ktransformers).
7. ARI-based CPU kernel switch (4 tokens/expert threshold in the paper).
8. Layerwise GPU prefill as an equilibrium switch.
9. Prefix hashing / refcounted pages (ExLlama).
10. Write HIP-portable kernels from day one.
11. ROCmFPX codebook → `__builtin_amdgcn_perm` → i8 → `sdot4` vs A8 (lesson only). Contract: [rocmfpx.md](rocmfpx.md).
12. Llaminar heterogeneous domains + explicit collective routing — **hetero engine after the serving forks**, not just a model to skim.
13. hippih three-ISA HIP (gfx1030 DOT / gfx1100 local WMMA / gfx900 mix) — **in-house engine after extras + SGLang + Llaminar**.
14. SGLang radix + overlap schedule — **own overlay**, import extras HIP, no AITER. [sglang-fork.md](sglang-fork.md).

## Ignore

ExLlama / EXL3 / Marlin / FlashInfer / TRT-LLM kernels (CUDA). rocWMMA / MMA FA (RDNA3+/CDNA). FA page=256 as if optimal (Tri Dao: convenience). Intel AMX tiles. CUDA-graph-as-religion (HIP graph capture + alloc is fragile). Unified memory on dGPUs (llama.cpp: hurts non-iGPU). FlyDSL shipped MFMA/WMMA GEMM/MoE/FA ([flydsl.md](flydsl.md)). DeepEP IBGDA / MORI / NVLink ([deepep.md](deepep.md)). Porting ROCmFPX custom GGUF types into vLLM. Treating Llaminar gfx906 ROCm as a gfx1030/gfx1100/gfx900 backend. Loading gfx906 DOT or gfx1030 `fdot2`/`sdot4` on V340L. Starting hippih or the SGLang overlay before extras occupancy. SGLang AITER / MFMA / FlashKDA CUTLASS.

## llama.cpp

Graph engine, not a serving scheduler. Cell-pool KV, not a paged block table. FA: VEC (decode), TILE (no tensor cores / AMD fallback), MMA_F16 (NVIDIA TC / AMD MFMA or WMMA). gfx1030 → VEC/TILE. HIP is documented: `-DGGML_HIP=ON -DGPU_TARGETS=gfx1030`. Server continuous batching is slot scheduling, not kernel-level paging. GGUF: never materialize FP16 weights. `ggml_cuda_dp4a` → sdot4 on all RDNA2.

### ROCmFPX (charlie12345)

llama.cpp GGUF family, not a vLLM backend. Custom types `Q4_0_ROCMFP4` / `_FAST` / `Q2_0_ROCMFPX`. Inner: Codebook10 → `amdgcn_perm` → `sdot4` vs Q8, UE4M3 scale. **No vLLM port.** Stock vLLM GGUF loader is Q4_0/K/IQ*, not these types. Side project: [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md) (`build-rdna2.sh` on a V620). Closest existing path: W4A8 after W8A8.

## ExLlama

V2 archived, V3 exists, ROCm is a TODO. Page size 256 because FA API requires a multiple of 256. CUDA-only.

## ktransformers

SOSP'25: attention+shared on GPU, routed experts in DRAM. PCIe 4.0 they quote vs DDR5. Layerwise prefill when tokens > threshold. Expert deferral decode-only. Long-context: KV parked in DRAM, sparse CPU attn so KV is not swapped back over PCIe. Only engine in this set that names PCIe as the prefill bottleneck. Closest *policy* cousin of W7800-attn / V620-experts ([multi-tier.md](multi-tier.md)) — still CPU+CUDA, not a ROCm backend.

## Llaminar

C++ graph runtime. Heterogeneous CPU/CUDA/ROCm domains, TP/PP, prefix-cache exists, continuous batching still a plan (we add it), ROCm **gfx906 only** today. **Product path #3 after the serving forks.** We write gfx1030 + gfx1100 HIP. gfx900/V340L is a third ISA (packed `mad_mix` / `v_pk_fma_f16`, no DL DOT) — Later.

## hippih

[BlivionIaG/hippih](https://github.com/BlivionIaG/hippih) — in-house HIP engine. README stub only. **Product path #4.** Three ISA backends (gfx1030 DOT, gfx1100 local WMMA, gfx900 mix/FMA). Steal extras kernels + SGLang serving + Llaminar placement; do not start before occupancy. Contract: [hippih.md](hippih.md).

## SGLang

Serving-strong (radix, overlap schedule). Instinct/AITER kernels are **ignore**. **Own rebase-on-release overlay after extras occupancy** — import `fa_rdna2` / skinny `fdot2`, `on_gfx10x()` only. Contract: [sglang-fork.md](sglang-fork.md). Do not start a second kernel tree.

## Others

MLC-LLM: compile-to-target (steal: emit gfx1030 kernels). TRT-LLM / Sarathi: scheduler ideas only, not the CUDA. LightLLM TokenAttention: per-token table, less fragmentation.

## Sources

- llama.cpp `fattn.cu`, `llama-kv-cache.cpp`, `docs/build.md`
- ExLlamaV2 `dynamic.md`; FA issue #828
- ktransformers SOSP'25 + layerwise-prefill docs
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)
- [Llaminar/llaminar](https://github.com/Llaminar/llaminar)
- [sgl-project/sglang](https://github.com/sgl-project/sglang)
- [BlivionIaG/hippih](https://github.com/BlivionIaG/hippih)
- [sglang-fork.md](sglang-fork.md), [multi-tier.md](multi-tier.md), [hippih.md](hippih.md)
