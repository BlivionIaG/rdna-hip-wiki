# DeepEP insight on 4× V620 — PCIe P2P silicon boundary

Date: 2026-08-18. Engine context: [engine/deepep.md](../engine/deepep.md). **Research / Later.** This page separates the transferable idea from NVIDIA/Instinct-specific transport.

## Transferable insight

MoE dispatch/combine is sparse, asymmetric all-to-all, not symmetric AllReduce. The useful design principle is:

- route directly by destination expert/GPU;
- avoid padding every peer to the same payload;
- fuse packing/quantization with dispatch where it wins;
- give low-latency decode and high-throughput prefill different schedules;
- keep communication ownership outside GEMM.

That principle applies to V620. DeepEP's implementation details do not transfer unchanged.

## What DeepEP actually proves

Upstream DeepEP provides dedicated MoE dispatch/combine kernels with throughput and low-latency modes. Its legacy primary table reports H800 + CX7 400 Gb/s, 128 tokens, hidden 7168, top-8, FP8 dispatch/BF16 combine:

| EP | Dispatch latency | Combine latency |
|---:|---:|---:|
| 8 | 77 us | 114 us |
| 16 | 118 us | 195 us |
| 32 | 155 us | 273 us |
| 64 | 173 us | 314 us |
| 128 | 192 us | 369 us |
| 256 | 194 us | 360 us |

These are not V620 targets and must not be copied into our project performance claims.

The social summary's 48-byte NIC descriptor, 1.7M vs 180M operations/s and exact warp-doorbell sequence were not found in the opened primary DeepEP performance table. Treat them as secondary explanation until a primary NVSHMEM/IBGDA source is pinned.

## Transport matrix

| DeepEP/MoRI mechanism | 4× V620 status |
|---|---|
| NVLink direct peer path | absent |
| xGMI/Infinity Fabric | absent |
| IBGDA / CX7 GPU NIC queues | no NIC requirement supplied; not the intranode V620 path |
| AMD DeepEP/rocSHMEM | published for MI300-class + xGMI/RDMA, not gfx1030 |
| MoRI-EP/SHMEM | current hardware matrix lists Instinct MI300/308/325/355, not V620 |
| HIP PCIe P2P | attested working on this box; bandwidth/latency still unmeasured |
| Kernel access to mapped peer allocation | candidate experiment after `hipDeviceEnablePeerAccess`; not a proven production transport |

AMD's current MoRI path is an Instinct/RDMA result. It is evidence that GPU-centric asymmetric communication matters, not evidence that MoRI supports Radeon PRO V620.

## V620 analogue

There is no network NIC or fabric to program for intranode TP=4. The only possible direct path is PCIe BAR peer memory:

1. Allocate destination receive buffers on each GPU.
2. Enable/map every required peer pair.
3. Build per-destination token counts and offsets.
4. Launch a routing kernel whose waves write coalesced packed token vectors directly to destination peer pointers.
5. Signal completion with a separately proven synchronization mechanism.
6. Destination launches local expert GEMM from its receive buffer; combine reverses the route.

This bypasses a host copy in principle. It does **not** bypass PCIe bandwidth, root-complex topology, ACS/IOMMU, BAR mapping or synchronization costs.

HIP documents peer mapping/copies, but no opened source proves a user device kernel can enqueue arbitrary peer SDMA from inside the kernel like IBGDA rings a NIC. Direct peer stores consume GPU issue/memory resources unless a supported device-side DMA API is proven. Do not call this 'GPU-initiated DMA' yet.

## Hard unknown: completion and atomics

PCIe atomics are a ROCm platform requirement, but that alone does not prove the required system-scope atomic semantics on a mapped remote V620 allocation. Before a lock-free queue:

- test remote atomic add/CAS visibility in both directions;
- test `__threadfence_system`/HIP memory-scope ordering;
- inspect generated ISA and page-fault behavior;
- verify completion without a CPU poll or global RCCL barrier;
- include invalid/partial peer topology.

Initial prototype may use host/HIP events for correctness measurement, but it is not the final low-latency design.

## Two prototype modes

### Decode / low latency

- Very small routed-token count.
- One compact persistent or graph-warmed dispatch kernel.
- Warp owns a destination; prefix offsets are preallocated/bounded where graph capture requires static storage.
- No separate quant kernel: optionally fuse FP16→INT8/FP8 only if the receiving GEMM consumes it.
- Optimize fixed launch/synchronization latency before bandwidth.

### Prefill / throughput

- Larger token buckets and variable payloads.
- Hierarchical count → prefix → coalesced peer copy.
- Use 16-byte or larger aligned vector transactions.
- Partition waves between local packing and remote writes only after measuring CU interference with GEMM.
- Compare overlap against SDMA/RCCL/PYNCCL, not just isolated copy bandwidth.

## Required baseline before implementation

For all 12 directed GPU pairs:

1. `hipDeviceCanAccessPeer` and enabled peer map.
2. `hipMemcpyPeerAsync` latency and GB/s for 4 B through 256 MiB.
3. Kernel peer load/store latency and GB/s, one and many writers.
4. RCCL/PYNCCL all-to-all or send/recv equivalent where available.
5. Host-bounce baseline (`NCCL_P2P_DISABLE=1`).
6. Topology, BAR size, ACS/IOMMU mode, link width/speed and RCCL transport log.
7. Correctness under simultaneous four-GPU traffic.

P2P being attested removes the existence question, not these measurements.

## Fair prototype gate

Build a V620-specific asymmetric dispatcher only if:

- peer kernel writes are correct and materially lower latency than host/RCCL routing for decode;
- aggregate traffic does not collapse under four-GPU contention;
- completion is graph-safe and does not reintroduce CPU proxy polling;
- routed payload/layout exactly matches the FP16 or INT8 MoE kernel;
- end-to-end gain survives count/prefix/quant/combine costs.

GEMM never owns the collective. Expose routed rows plus explicit `combine_mode`; engine code owns P2P/RCCL/PYNCCL and TP/EP semantics.

## Project order

Later, after:

1. current `fa_rdna2` occupancy fix;
2. reproducible dispatch/baseline harness;
3. FP16/INT8 HIP MoE compute kernels;
4. measured pairwise P2P matrix.

Then: peer-store microbench → asymmetric dispatch prototype → fused quant experiment → overlap → engine integration.

## Primary sources

- DeepEP: https://github.com/deepseek-ai/DeepEP
- DeepEP legacy measurements: https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md
- AMD DeepEP port: https://github.com/ROCm/DeepEP
- MoRI and hardware matrix: https://github.com/ROCm/mori
- V620 P2P platform contract: [rccl-p2p.md](rccl-p2p.md)
