# Heterogeneous MoE — silicon placement contract

Date: 2026-08-18 (W7800/V620 HIP–HIP). **Retipped 2026-08-27:** W7800 fast-tier is **dead** (cards selling). New Later form is Spark×2 + 8×V620. Engine placement: [../engine/multi-tier.md](../engine/multi-tier.md). DeepEP-style routing (V620-internal only): [deepep-v620.md](deepep-v620.md). MoE compute: [../kernels/fp16-moe.md](../kernels/fp16-moe.md), [../kernels/int8-moe.md](../kernels/int8-moe.md).

## 2026-08-27 retip — Spark (GB10) + 8× V620

| Domain | Hardware | ISA | Job |
|---|---|---|---|
| Fast | 2× NVIDIA DGX Spark | **GB10** CUDA (Blackwell TC / FP4) | attention, router, embeddings, LM head, n-gram, **live KV** |
| Expert | 8× Radeon PRO V620 32GB | **gfx1030** HIP | expert GEMMs only (`fdot2` / `sdot4`) |
| Spark↔Spark | ConnectX-7 | NCCL | first-party NV hop |
| Spark→V620 | CX-7 ↔ (V620 host NIC, if any) | **not** HIP peer | **activations only**, two machines / two collectives unless the V620 rig also has a CX NIC |
| V620↔V620 | 88096 PIX (attested P2P) | RCCL / `hipMemcpyPeer` | expert A2A **inside** the V620 box |

This is **CUDA + HIP**, not two HIP targets. extras cannot drive GB10. hippih gfx1100 WMMA fat is more Later if the W7800s go. Do not start a second DOT tree or a Llaminar gfx1030 backend for this.

**Official Spark numbers only** ([NVIDIA DGX Spark specs](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), [hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)): 128 GB LPDDR5x unified @ **273 GB/s**, ConnectX-7 **200 Gbps**. Do not invent hop GB/s or tok/s.

**Why the split still holds in silicon:** Spark has capacity for live KV/attn (128 GB) but is bandwidth-poor vs V620 GDDR6 **512 GB/s**. Experts (W4 / EXL3 / `moe_q_gemm`) stay on V620. Do not park live KV on V620.

**Wire:** extras `moe_q_gemm` eats **fp16**. Spark is FP4-native. Convert NVFP4/FP8/BF16 → half **on Spark** (or on the hop). Do not land those dtypes on V620 and do not add a new V620 DOT for the hop.

**P2P / DeepEP:** 88096 PIX `hipMemcpyPeer` stays **V620-internal**. Spark cannot peer-store into a V620 BAR. DeepEP mapped-peer scatter/combine does **not** apply across the NIC. Until a V620-host CX NIC exists and is measured, assume host/NIC-staged activations. Leave [Llaminar/llaminar](https://github.com/Llaminar/llaminar) (still gfx906 / sm86 / GGUF). Their 3090+MI50 is one box, not this.

**Order unchanged:** extras live HIP on V620 first (W4 / GDN / QSA indexer). Hetero hop is Later. FA pin stays closed.

## Later (not a workstream) — R9700 / RDNA5

Note 2026-08-27: maybe add a faster GPU later (R9700 or RDNA5). **No kernels now. No new card.**

| SKU | ISA | Official | Role if bought |
|---|---|---|---|
| AI PRO R9700 | **gfx1201** RDNA4 | 32 GB GDDR6, 640 GB/s, 64 MB IC, 64 CU, LDS 128 KiB ([AMD](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html), [ROCm gpu-specs](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)) | **Expert SKU** (WMMA + hw FP8). Same 32 GB class as one V620 — **not** a Spark replacement. |
| RDNA5 | unknown | no ISA PDF as of 2026-08-27 | Wait. Do not invent opcodes. |

- extras gfx1030 objects **will not load** (ANTIBLEED). New fatbin / TU if it ever lands. hippih would grow a fourth ISA (today: 1030 / 1100 / 900).
- vLLM `supports_fp8()` / `on_rdna4()` / AITER RDNA4 analog stay **off** extras gfx1030. Do not import that path.
- R9700 P2P/ipc is a known miss ([ROCm/rccl-tests#162](https://github.com/ROCm/rccl-tests/issues/162) `hipIpcGetMemHandle` on 4×R9700). 88096 V620 attestation does **not** transfer.
- Do not copy R9700 TOPS / tok/s onto coverage.


## Historical (2026-08-18) — 2× W7800 + 8× V620, two HIP ISAs

Kept below so old links still resolve. Do not treat W7800 as the live fast tier.


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
