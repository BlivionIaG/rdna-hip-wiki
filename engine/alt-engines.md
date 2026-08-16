# What to steal from other engines

Date: 2026-08-17. Sourced only.

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

## Ignore

ExLlama / EXL3 / Marlin / FlashInfer / TRT-LLM kernels (CUDA). rocWMMA / MMA FA (RDNA3+/CDNA). FA page=256 as if optimal (Tri Dao: convenience). Intel AMX tiles. CUDA-graph-as-religion (HIP graph capture + alloc is fragile). Unified memory on dGPUs (llama.cpp: hurts non-iGPU).

## llama.cpp

Graph engine, not a serving scheduler. Cell-pool KV, not a paged block table. FA: VEC (decode), TILE (no tensor cores / AMD fallback), MMA_F16 (NVIDIA TC / AMD MFMA or WMMA). gfx1030 → VEC/TILE. HIP is documented: `-DGGML_HIP=ON -DGPU_TARGETS=gfx1030`. Server continuous batching is slot scheduling, not kernel-level paging. GGUF: never materialize FP16 weights. `ggml_cuda_dp4a` → sdot4 on all RDNA2.

## ExLlama

V2 archived, V3 exists, ROCm is a TODO. Page size 256 because FA API requires a multiple of 256. CUDA-only.

## ktransformers

SOSP'25: attention+shared on GPU, routed experts in DRAM. PCIe 4.0 they quote vs DDR5. Layerwise prefill when tokens > threshold. Expert deferral decode-only. Long-context: KV parked in DRAM, sparse CPU attn so KV is not swapped back over PCIe. Only engine in this set that names PCIe as the prefill bottleneck.

## Others

MLC-LLM: compile-to-target (steal: emit gfx1030 kernels). TRT-LLM / Sarathi: scheduler ideas only, not the CUDA. LightLLM TokenAttention: per-token table, less fragmentation.

## Sources

- llama.cpp `fattn.cu`, `llama-kv-cache.cpp`, `docs/build.md`
- ExLlamaV2 `dynamic.md`; FA issue #828
- ktransformers SOSP'25 + layerwise-prefill docs
