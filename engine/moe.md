# MoE expert offloading

Date: 2026-08-17. Sourced only. DeepEP / asymmetric A2A insight: [deepep.md](deepep.md).

## Verdict on gfx1030

No sourced MoE-on-gfx1030 notes. gfx1030 is outside AITER's matrix and outside vLLM's Triton MoE ROCm whitelist (PR #37826 explicitly excluded gfx10xx). Radeon MoE work that exists is gfx1100 / gfx1201.

Honest one-box huge-MoE design: CPU-resident experts + GPU attention/hot set (ktransformers / llama.cpp `--n-cpu-moe`). Layerwise GPU prefill when the batch is large enough that PCIe+GPU math beats CPU experts.

Decode MoE is many tiny GEMMs — terrible even on RDNA3 (7900 XTX: Triton-unfused 8.5 → native HIP 48.8 tok/s).

**Live in the fork:** W4A16 / W8A16 / W8A16-FP8 (LUT→`fdot2`). **Not shipping:** W8A8/`sdot4`, mxfp4 (format contracts; PR #52391 is enablement, not a launched gfx1030 MoE). **Queued spec:** INT2 + mixed INT2/INT4 experts — [int2.md](int2.md). Native FP16 / INT8 HIP MoE: [fp16-moe.md](fp16-moe.md), [int8-moe.md](int8-moe.md). Nothing first-class ships. Format contracts: [kernels/w4a16.md](../kernels/w4a16.md), [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md), [kernels/int2.md](../kernels/int2.md).

MoE tile: A (activations) in LDS; stream B (weights). Token-major + small BLOCK_M at decode. Under TP>1 write unreduced rows (do not fuse reduce-before-allreduce). Do not copy AITER W8A8 CK (MFMA, AGPR, CDNA). `0c59068e` changes RCCL transport only; GEMM still does not own the collective ([deepep.md](deepep.md)).

Mixed INT2/INT4: sort tokens by expert, **then** group by bitwidth. Two unpackers, one `fdot2`. Do not switch unpackers inside K.

## Basics

DeepSeek-V3: 1 shared + 256 routed, 8 activated/token, 671B/37B. Prefill wants fat expert batches (EP32). Decode: one expert per GPU at EP320, batch/expert usually ≤256, memory-bound.

AMD playbook: <1% activation density prefer no-EP (AllReduce); >3% prefer EP (AllToAll). MLA still wants DP+EP for KV reasons. DeepEP is a GPU-initiated AllToAll for the high-density case, not a reason to AllToAll a 4×PCIe box by default.

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

gfx1030 MoE kernel quality. Whether #37190 / #20126 merged later. llama.cpp compute vs H2D split. AITER fused MoE on any RDNA. Named INT2 / mixed-bitwidth checkpoint. Whether 4×V620 has working `hipDeviceCanAccessPeer` after IOMMU/ACS.

## Sources

- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [vLLM EP](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [AMD MoE playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [ktransformers layerwise](https://ktransformers.net/en/docs/optimization-techniques/layerwise-prefill)
- [llama.cpp --cpu-moe](https://github.com/ggml-org/llama.cpp/pull/15077)
- [Triton MoE excludes gfx10xx](https://github.com/vllm-project/vllm/pull/37826)
- [deepep.md](deepep.md)
