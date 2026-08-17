# DeepEP insight — not a gfx1030 first ticket

Date: 2026-08-18. Engine note. Sourced from DeepSeek DeepEP (V3/R1 production; Wafer-AI / V4 Flash 0731 write-up). Silicon boundary: [silicon/deepep-v620.md](../silicon/deepep-v620.md). P2P facts: [silicon/rccl-p2p.md](../silicon/rccl-p2p.md). MoE kernels stay [fp16-moe.md](fp16-moe.md) / [int8-moe.md](int8-moe.md). PD stays [pd-disagg.md](pd-disagg.md) (**skip as a win** on this box).

## What they actually did

NCCL/RCCL AllReduce is **symmetric**. MoE dispatch is **asymmetric and data-dependent**: each token’s expert set changes every step. A CPU proxy that posts NIC work (NCCL’s classic path) is the wrong machine for that rate.

DeepEP: the GPU builds the NIC work-queue element (src addr+key, dst addr+rkey, length), writes a submission ring, rings a doorbell. IBGDA-class path, no CPU in the data plane. Intra-node NVLink peers skip the NIC and store through mapped pointers. Decode-oriented numbers they quote (77 µs / 8 GPU, 194 µs / 256) are **RDMA on H800+CX7**, not PCIe. Do not copy them. Training mode parks some SMs on comm. FP8 quant can fuse into dispatch.

Ports, not drop-ins here: AMD **MORI** (Instinct / gfx942+ in SGLang), Tencent, UCCL (EFA / other NICs). vLLM ROCm A2A today is still `VLLM_ALL2ALL_BACKEND=allgather_reducescatter` on RCCL.

## What we steal (insight only)

1. **Do not treat MoE A2A as AllReduce.** Density heuristic we already have: low activation → no-EP + AllReduce; high → EP + AllToAll. DeepEP is *how* to implement the second, not a reason to ignore the first.
2. **Collective ownership stays in the engine.** GEMM `w2` default under TP/EP is unreduced routed rows. `0c59068e` only changed RCCL transport (broken `vllm::all_reduce` dispatcher → direct PYNCCL). It does not put RCCL inside a DOT kernel.
3. **Intra-node analogue on 4×V620 is a mapped-peer scatter/combine kernel over PCIe BARs** — destination receive buffers, peer-map, routing waves write packed tokens to remote pointers. That is “skip the host bounce,” not “build a 48-byte IB WQE.” **Do not call it GPU-initiated DMA:** no opened HIP source exposes device-side peer SDMA/NIC doorbells. Measure pairwise peer-store bandwidth, remote atomic/system-scope completion, and four-GPU contention before designing a lock-free path. Unknown until [silicon/deepep-v620.md](../silicon/deepep-v620.md) + [silicon/rccl-p2p.md](../silicon/rccl-p2p.md) say otherwise.
4. **Fuse quant into the comm kernel only after a real A2A exists.** W8A8 `sdot4` still needs A8 on the compute side; do not invent a DeepEP-FP8 dispatch for a box with no RDMA fabric.

## What we do not steal

| Piece | Why |
|---|---|
| IBGDA / GPU-posted IB WQE | No sourced IB GPUDirect Async on this workstation |
| NVLink / xGMI mapped-pointer fast path | V620 is PCIe |
| MORI / rocSHMEM as a vLLM backend | gfx942+ / Instinct. Same class as AITER |
| Device-side peer SDMA doorbells | Not proven in opened HIP on gfx1030 |
| Training SM-split (20/132) | Different occupancy physics; 72 CU decode box |
| Their 887 tok/s / 77 µs | H800/CX7 workload. Do not copy onto coverage |
| Social-post 48-byte descriptor / ops/s | Secondary until a primary NVSHMEM/IBGDA source is pinned |
| PD / KV fabric | Already skipped. DeepEP is expert dispatch, not prefill-KV |

## When this becomes a card

Later, after occupancy + HIP MoE compute + a **measured** pairwise P2P matrix on TP=4 (peer-store + `hipMemcpyPeerAsync` + RCCL + host-bounce). Then: one research card “asymmetric MoE A2A on PCIe” — compare RCCL gather/reducescatter vs a mapped-peer scatter/combine of routed rows. Not DeepEP-the-library. Not IBGDA/MORI.

Fair prototype gate lives on the silicon page: decode latency vs host/RCCL, four-GPU contention, graph-safe completion, payload match to FP16/INT8 MoE, end-to-end after count/prefix/quant/combine.

No new DOT. No change to `combine_mode` ownership.

## Sources

- DeepSeek DeepEP (published library used for V3/R1)
- Wafer-AI DeepSeek V4 Flash 0731 Fast write-up (2026-08, secondary)
- SGLang `--moe-a2a-backend` `deepep` / `mori`
- [silicon/deepep-v620.md](../silicon/deepep-v620.md)
- [moe.md](moe.md), [pd-disagg.md](pd-disagg.md)
