# ROCmFPX — leverage, not a vLLM port

Date: 2026-08-18. Contract. Digest: [notes/rocmfpx.md](notes/rocmfpx.md). Source: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX). Occupancy still first. Do not edit `perf/rdna2_w4a16` from here.

**Verdict:** no vLLM port. llama.cpp GGUF + HIP/Vulkan. Tuned on Strix `gfx1151`; AMD also gave them an **R9700 (RDNA4 / gfx1200)**. `scripts/build-rdna2.sh` exists. We **leverage** codebook→`perm`→`sdot4`. Side project: [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md).

**Do not copy their tok/s** (Vulkan 90 / HIP 76 / MTP 116 on Strix or R9700). RDNA4 has WMMA/better FA. gfx1030 does not.

## Serving: prefix cache and concurrency

ROCmFPX does **not** remove llama.cpp cache. `llama-server` defaults:

- `--cache-prompt` **on** — reuse common prefix KV (same conversation / same system prompt).
- `-cb` / `--cont-batching` **on** — slot continuous batching (not vLLM V1).
- `-np` / `--parallel` — **slots**. Each slot holds a context. KV budget ≈ `slots × ctx × kv_bytes/token`.

What you **lose vs vLLM / SGLang**:

| Feature | llama-server (ROCmFPX) | vLLM fork |
|---|---|---|
| Prefix reuse | per-slot / previous-prompt compare | radix + paged block share across many prefixes |
| Concurrency | fixed slots (`-np`) | continuous batch + paged KV pool |
| Distinct hot prefixes | one per slot (`-np` keeps N prefixes hot) | many, as long as blocks fit |
| Chunked prefill / MBT | no | yes |
| HIP FA / MoE we wrote | no | yes |

Concurrency is **VRAM**, not the codebook. ROCmFP4 is ~12% smaller than matched Q4_K_M (their 35B-A3B: 19.05 vs 21.71 GB). That leftover goes to KV/slots, not to a serving farm.

Order-of-magnitude on **one 32 GB V620**, 35B-A3B ROCmFP4 (~19 GB weights): leftover ~10 GB after FA scratch. That is a **handful of 8k slots** or **1–2 at 32k**, not tens of concurrent users. TurboQuant `-ctk/-ctv` can stretch KV (they recommend `q8_0` K + `turbo4` V for agents) — still slot math, still not radix. Measure; do not invent a slot count.

`-np` does not make decode faster. It keeps more prefixes hot and lets `-cb` mix slots. Per-slot tok/s usually drops as slots fill.

**Consider ROCmFPX when:** single-user / few-session decode density, MTP-in-llama.cpp, GGUF-only. **Not when:** multi-tenant prefix-heavy serving (stay on vLLM, then Llaminar).

## What we steal

HIP codebook (`rocmfp4_hip_codebook.cuh`): nibble → `__builtin_amdgcn_perm` → **i8 DP4A operands** (Codebook10 `{0,±1,±2,±3,±4,±6,±8,±10}`). Scale is finite **UE4M3**. Activations on the hot path are **Q8** (`vec_dot_*_q8_1`).

```
// keep
nibble → perm codebook → i8x4 → ggml_cuda_dp4a / sdot4 vs A8
C *= ue4m3

// drop
sdot8 on raw nibbles
fdot2 E2M1 LUT
porting MMVQ/MMQ into vLLM
FlashInfer / WMMA / FP4 MMA
their R9700 / Strix tok/s
```

Closest existing contract: **W4A8** after W8A8. Different codebook, same DOT class.

## What we do not steal

| Piece | Why |
|---|---|
| Engine / scheduler | ggml slots, not vLLM V1 |
| Custom GGUF types | stock vLLM GGUF is Q4_0/K/IQ* |
| MTP | llama.cpp `draft-mtp`. We still need fat tile `q>1` |
| Vulkan | their Strix default. We are HIP/vLLM |
| TurboQuant KV | runtime cache type, not a vLLM dtype |

## vLLM (later, optional)

Only if someone names a **safetensors** checkpoint with this codebook. Then: new quant key + `perm`→`sdot4` GEMM **after W8A8**.

## Sources

- ROCmFPX README (2026-08): R9700 disclosure, Strix tables, `llama-server` example
- llama.cpp `tools/server`: `--cache-prompt`, `-cb`, `-np`
- [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md), [notes/rocmfpx.md](notes/rocmfpx.md)
