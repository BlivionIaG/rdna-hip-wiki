# Prefill-decode disaggregation

Date: 2026-08-17. Sourced only.

## Verdict on 4× V620

Skip as a win.

Every current production PD stack (vLLM Nixl/Mooncake, SGLang Mooncake/NIXL, MoRI-IO) is built around RDMA and/or NVLink/xGMI. Transfer Engine *can* speak TCP; that is not a documented way to win TTFT/ITL on RDNA. No gfx1030/gfx1100 PD recipe found.

vLLM: PD does **not** raise throughput; it decouples TTFT vs ITL. DistServe: wrong design for throughput-only offline and few/single-GPU boxes.

Sourced alternatives on this box: colocate + [chunked prefill](batching.md), OffloadingConnector, ktransformers-style CPU sparse attn (don't move KV on decode).

This is collectives / a KV fabric, not something RCCL P2P unlocks. See [silicon/rccl-p2p.md](../silicon/rccl-p2p.md).

## What moves

Payload is prefill KV (plus token ids so decode can skip re-tokenize). Classic path does not move decode-generated KV until bidirectional/multi-turn.

When: DistServe pull after prefill; Mooncake layer-wise stream to D CPU DRAM overlapped with compute; vLLM Nixl D pulls (READ) by default; MoRI WRITE = P pushes after every layer; SGLang P pushes.

Handshake is always TCP/HTTP/ZMQ control + registered-memory data plane.

## Connectors

vLLM: NixlConnector (NVIDIA primary, UCX), MooncakeConnector (V1 experimental, default protocol rdma), MoRIIOConnector (ROCm only: rdma or xgmi). OffloadingConnector is GPU→CPU, not PD.

SGLang: `--disaggregation-mode prefill|decode`; backend mooncake (default) or nixl. `page_size` and `kv_cache_dtype` must match.

Mooncake TE protocols: tcp (any), rdma, efa, nvlink, hip (ROCm intra-node IPC). HIP path: peer access iff `hipDeviceCanAccessPeer`. No consumer-RDNA PD recipe.

NIXL: no first-class PCIe-P2P plugin. UCX cuda_ipc may ride PCIe P2P if the driver allows; not a reliable default. Production data planes are RDMA (GDR), NVLink/xGMI, or host-bounce TCP.

## Layout contract

Both sides must match: architecture, dtype, KV heads, head size, layers, attn backend, cache_dtype, spec method, push vs pull. Dynamic FP8 scales are not transferred. Hetero TP: SGLang without staging ~30k small RDMA ops; staging gather→bulk RDMA→scatter.

## Unknowns

- Any production PD recipe for gfx1030/gfx1100.
- Whether HIP IPC on RDNA ever gets true P2P.
- Measured TCP/PCIe-host-bounce PD goodput on consumer GPUs.
- MoRI xGMI on anything that is not Instinct HGX-class.

## Sources

- [DistServe](https://arxiv.org/abs/2401.09670)
- [Mooncake](https://arxiv.org/abs/2407.00079)
- [vLLM disagg](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [MoRI](https://docs.vllm.ai/en/latest/features/moriio_connector_usage/)
- [SGLang PD](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)
- [NIXL](https://github.com/ai-dynamo/nixl)
