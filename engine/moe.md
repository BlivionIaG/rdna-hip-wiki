# MoE expert offloading

Date: 2026-08-17. Sourced only.

## Verdict on gfx1030

No sourced MoE-on-gfx1030 notes. gfx1030 is outside AITER's matrix and outside vLLM's Triton MoE ROCm whitelist (PR #37826 explicitly excluded gfx10xx). Radeon MoE work that exists is gfx1100 / gfx1201.

Honest one-box huge-MoE design: CPU-resident experts + GPU attention/hot set (ktransformers / llama.cpp `--n-cpu-moe`). Layerwise GPU prefill when the batch is large enough that PCIe+GPU math beats CPU experts.

Decode MoE is many tiny GEMMs — terrible even on RDNA3 (7900 XTX: Triton-unfused 8.5 → native HIP 48.8 tok/s).

Live enablement path is the user's W4A16 / W8A8 / mxfp4 dense+MoE (PR #52391). Nothing first-class ships. Format contracts: [kernels/w4a16.md](../kernels/w4a16.md), [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md).

MoE tile: A (activations) in LDS; stream B (weights). Token-major + small BLOCK_M at decode. Under TP>1 write unreduced rows (do not fuse reduce-before-allreduce). Do not copy AITER W8A8 CK (MFMA, AGPR, CDNA).

## Basics

DeepSeek-V3: 1 shared + 256 routed, 8 activated/token, 671B/37B. Prefill wants fat expert batches (EP32). Decode: one expert per GPU at EP320, batch/expert usually ≤256, memory-bound.

AMD playbook: <1% activation density prefer no-EP (AllReduce); >3% prefer EP (AllToAll). MLA still wants DP+EP for KV reasons.

## vLLM

`--enable-expert-parallel`. `EP_SIZE = TP × DP`. TP+EP = AllReduce; DP+EP = AllToAll. ROCm A2A: `VLLM_ALL2ALL_BACKEND=allgather_reducescatter` on RCCL.

Kernels: Triton, DeepGEMM, Marlin WNA16, rocm aiter moe (CDNA). `--cpu-offload-gb` shipped (generic). Static CPU expert offload merged (#34535). LFRU GPU cache of hottest N (#37190) still open as of 2026-08-05.

## SGLang

`--moe-a2a-backend`: none / deepep / mooncake / nixl / mori (AMD, gfx942+, normal only). Hybrid with ktransformers kt-kernel shipped (CPU AMX experts). UVM offload PR #20126 open, CUDA-only.

## Offload policies

Shipped: static CPU placement (vLLM #34535, llama.cpp `--n-cpu-moe`); KT hot-expert set (uniform/frequency/front-loading). Not shipped as production: vLLM LFRU, SGLang UVM, HybriMoE MRS.

## Kernel hit-list

Grouped GEMM (prefill), skinny GEMV (decode), router/align-sort, combine, EP all-to-all (MORI on CDNA; RCCL gather/reducescatter on ROCm vLLM), fused MoE, shared-expert fusion.

## Unknowns

gfx1030 MoE kernel quality. Whether #37190 / #20126 merged later. llama.cpp compute vs H2D split. AITER fused MoE on any RDNA.

## Sources

- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [vLLM EP](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [AMD MoE playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [ktransformers layerwise](https://ktransformers.net/en/docs/optimization-techniques/layerwise-prefill)
- [llama.cpp --cpu-moe](https://github.com/ggml-org/llama.cpp/pull/15077)
- [Triton MoE excludes gfx10xx](https://github.com/vllm-project/vllm/pull/37826)
