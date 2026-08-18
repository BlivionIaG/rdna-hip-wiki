# Heterogeneous MoE: 2x W7800 + 8x V620 — silicon placement contract

Date: 2026-08-18. **Research / Later.** Engine placement belongs on an engine page; this page owns ISA, bytes, and which kernels live where. DeepEP-style routing: [deepep-v620.md](deepep-v620.md). MoE compute: [../kernels/fp16-moe.md](../kernels/fp16-moe.md), [../kernels/int8-moe.md](../kernels/int8-moe.md).

## Topology (as specified)

| Domain | Hardware | ISA | Job |
|---|---|---|---|
| Fast | 2x Radeon PRO W7800 48GB | **gfx1100** RDNA3 | attention, router, embeddings, LM head, live KV |
| Expert | 8x Radeon PRO V620 32GB | **gfx1030** RDNA2 | expert GEMMs only |
| Wire | PCIe 4.0, no xGMI/NVLink | — | **activations only** between domains |

This is two HIP targets, not one fat binary. gfx1100 has WMMA 16x16x16, BF16 DOT, and VOPD. gfx1030 has none of those. Shared source must compile twice and dispatch by `hipGetDeviceProperties` / arch string.

Official W7800 48GB page: RDNA3, 48 GB GDDR6, 96 MB Infinity Cache, PCIe 4.0 x16. Official V620: RDNA2, 32 GB, 128 MB IC, 72 CU, 512 GB/s. Do not treat the two cards as interchangeable Navi.

Ten dual-slot cards almost certainly span more than one root complex or host. Record topology before claiming a single-node P2P fabric.

## What belongs on W7800

- Token embeddings, RMSNorm, QKV, RoPE, attention, output projection, router GEMM, SiLU-free residuals, LM head.
- Live KV cache for all active requests. Attention is already on this domain; moving KV off it turns every decode into a PCIe read.
- Prefix-cache / radix state if the engine has it. That is metadata + KV, not expert weights.
- gfx1100 kernels may use WMMA for fat prefill attention/GEMM and `fdot2` for skinny decode. BF16 is legal on W7800; keep activations **fp16** on the V620 hop unless both sides agree.

## What belongs on V620

- Expert weights (W4A16 / W8A16 / W8A8 / native FP16) and the existing gfx1030 DOT kernels.
- Local expert bias / scales / routing metadata copies required by those kernels.
- Optional parked/cold KV **only** after W7800 2x48 GB is proven full. Parked KV is a bandwidth tax, not a free tier.

Do not run attention or the router on V620 in this design. That would reintroduce weight or KV traffic the split exists to avoid.

## Activation-only hop

Per routed expert copy, payload is:

```text
bytes = tokens_to_that_expert * hidden * elem_size
```

`elem_size` is 2 for fp16. INT8/FP8 dispatch is allowed only if the V620 expert kernel consumes that dtype; otherwise you pay a second convert.

Order-of-magnitude (illustrative, substitute the served hidden):

| Phase | tokens x top_k | hidden 4096 fp16 | Comment |
|---|---:|---:|---|
| Decode bs=1, top_k=8 | 8 | 64 KiB | latency, not bandwidth |
| Decode bs=32, top_k=8 | 256 | 2 MiB | still launch-dominated if naive |
| Prefill 2048, top_k=8 | 16384 | 128 MiB | the PCIe problem |

Sending expert weights or KV instead of activations is the anti-pattern. A single W4A16 expert shard of hidden 4096 x inter 11000 is already several MiB; repeating that every token is worse than the activation hop.

Combine is the reverse hop: expert output rows back to the W7800 residual. Same byte formula. Count both directions.

## KV park (optional, last)

If attention stays on W7800, keep KV there.

Approximate fp16 KV bytes per layer:

```text
2 * seq * kv_heads * head_dim * 2
```

Example: seq 32k, 8 kv heads, 128 dim → 128 MiB/layer. Fetching that from V620 every decode is larger than the MoE activation hop. INT8 KV reduces bytes but still crosses PCIe and needs a gfx1100 reader.

Allowed later experiments:

1. Overflow-only park of idle prefixes on V620 or host.
2. Prefill KV built on W7800, never streamed per token from experts.
3. Disaggregated prefill/decode later; do not mix it into the first placement prototype.

## Cross-SKU communication

Same-SKU V620 P2P is attested; **W7800↔V620 P2P is unmeasured**. Mixed BAR sizes (48 GB vs 32 GB) and possibly different root ports make this a new checklist, not a reuse of the 4x V620 result.

Measure before writing a scatter kernel:

1. `hipDeviceCanAccessPeer` for every W7800–V620 pair and every V620–V620 pair.
2. `hipMemcpyPeer` plus a kernel peer-store microbench in both directions.
3. Host-bounce / RCCL / MPI comparison.
4. Four-plus-two contention, not just one pair.

Until that matrix exists, assume host-staged activation copies. Do not port IBGDA, MoRI, or DeepEP NIC doorbells. The analogue remains mapped-peer scatter/combine over PCIe BARs, with host fallback ([deepep-v620.md](deepep-v620.md)).

## Kernel inventory

### Must exist on gfx1100 (W7800)

- Attention (stock Triton/ROCm or a later gfx1100 HIP FA).
- Skinny + fat unquantized GEMM for QKV/O/router/LM head. WMMA is legal here.
- Router top-k / softmax / expert-id pack. This is not an sdot4 problem.
- Pack/unpack of routed activation rows for the hop.

### Must exist on gfx1030 (V620)

- Native FP16 and quantized expert GEMMs from the existing MoE silicon pages.
- Decode `M=1/2/4/8` and prefill grouped tiles.
- Fused w13 SiLU×up and w2 route-scale only if the route scale is applied on V620. Prefer applying route scale on W7800 after combine if that keeps the expert kernel simpler.
- No WMMA, no bf16 DOT, no AITER MFMA fragments.

### Must not be shared blindly

- One Triton autotune table for both SKUs.
- gfx1100 WMMA objects loaded on gfx1030.
- A single occupancy attribute for both attention and expert GEMM.

## Runtime fit: Llaminar vs SGLang fork

Llaminar ([Llaminar/llaminar](https://github.com/Llaminar/llaminar)) is the closer **placement model**: explicit heterogeneous domains, TP/PP/MoE EP (WiP), GGUF, prefix-cache already present, OpenMPI + NCCL/RCCL/host collectives. Current ROCm backend is **gfx906 only**. gfx906 is Vega MFMA-class, not RDNA2/3. Their CUDA+ROCm examples are pipeline-parallel dense layers or host-staged TP, not attn-on-RDNA3 / experts-on-RDNA2.

Contributing kernels to Llaminar still means writing new gfx1030 and gfx1100 HIP backends. Their C++ graph/domain split is the reusable idea.

An SGLang fork buys radix, continuous batching, and serving immediately, and still requires the same two HIP backends plus a heterogeneous EP scheduler that SGLang does not have for this topology. ktransformers is CPU+CUDA and is a source of ideas, not a ROCm backend.

Silicon recommendation: implement V620 expert kernels and W7800 attention/router first **inside the stack that already compiles them** (current vLLM HIP work). Prototype activation placement on a thin runtime or Llaminar-style domain graph only after those kernels exist. Do not block occupancy or HIP MoE compute on a runtime choice.

## Order

1. `fa_rdna2` occupancy and V620 HIP MoE compute (already first).
2. gfx1100 attention/router baseline on W7800 (stock, then HIP if needed).
3. Measured W7800↔V620 peer/host activation matrix.
4. Asymmetric activation dispatch/combine (PCIe BAR or host).
5. Optional KV overflow park.
6. Only then pick Llaminar contribution vs SGLang fork as the serving shell.

## Sources

- AMD W7800 48GB: https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7800-48gb.html
- AMD V620: https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html
- Llaminar README: https://github.com/Llaminar/llaminar
- This wiki: [valu.md](valu.md), [rccl-p2p.md](rccl-p2p.md), [deepep-v620.md](deepep-v620.md)
