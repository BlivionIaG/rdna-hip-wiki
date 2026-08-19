# FlashKDA — steal the math, not the CUTLASS

Date: 2026-08-19. Engine contract. Repo: [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) (MIT, 2026-04). Occupancy still first. Do not copy H20 ms onto coverage.

## What it is

CUTLASS **CUDA** kernel for **Kimi Delta Attention** (KDA) — per-channel-gated **delta-rule linear attention** used by Kimi Linear (e.g. 48B-A3B hybrid). Prefill only. Drop-in FLA backend: `fla.ops.kda.chunk_kda` auto-dispatches when `flash_kda` is installed (`FLA_FLASH_KDA=0` opts out).

SGLang already wired `--linear-attn-prefill-backend flashkda` ([#29472](https://github.com/sgl-project/sglang/pull/29472)). **Not a reason to promote SGLang.** extras has **no** KDA hit.

## Hardware gate (dead as a port)

| FlashKDA needs | gfx1030 |
|---|---|
| SM90+ tensor cores (Hopper+) | **No.** No WMMA/MFMA. |
| CUDA 12.9+ / CUTLASS | **No.** HIP extras. |
| **bf16** q/k/v/g + bf16 recurrent state | **No** hardware bf16 matrix. |
| `K = V = 128` | FA head-128 we have; this is a **128×128 state**, not `fa_rdna2`. |
| Safe gate (`lower_bound` in [-5, 0]) | Policy, not silicon. |

Their 1.7–2.2× vs FLA Triton is **H20**. Dead as a vLLM/HIP import. Same class as Marlin / FA3.

## What we steal (Later)

1. **Algorithm, not the .cu.** KDA = chunked delta-rule + per-channel gate + optional QK L2 / beta sigmoid **inside** the kernel. Decode is a **recurrent state** `[N, H, V, K]`, not paged softmax KV. Linear-attn layers do not use `fa_rdna2`.
2. **Two-kernel split** (their deep-dive): K1 token-parallel (gate, L2, L/Mqk, invert) vs K2 head-parallel recurrence. One fused kernel left SMs idle on the recurrence. Same occupancy lesson as our decode vs prefill split.
3. **Chunk 16 + no rescaling** only if the **safe gate** holds. Unbounded gate → FLA Triton `chunk_kda` (chunk 64, high-prec). We would do the same split.
4. **`cu_seqlens` packed prefill** — already how vLLM/SGLang batch. Steal the packing, not CUTLASS.
5. Cousin of Qwen3.5 **GDN** ([qwen35.md](qwen35.md)): hybrid linear + full-attn. GDN stays Later / no HIP. FlashKDA is the Kimi-flavored fused prefill of that family.

## gfx1030 HIP (only if we serve a KDA/GDN model)

After occupancy + a real checkpoint:

- Prefill: chunked 128×128 state update in **fp16**, inner `fdot2`. rocBLAS first if the tile is fat enough.
- Decode: skinny `fdot2` GEMV against the state (M=1). Not FlashKDA — they do not ship decode.
- No bf16 path. No CUTLASS. No `on_rdna()` gate.
- Reference: FLA Triton `chunk_kda`, not Moonshot’s `.cu`.

Until extras serves Kimi Linear / GDN, this is **wiki only**.

## Cards

Later / explore. Do not retip occupancy. One “HIP KDA/GDN prefill” card only after a model is on the box.

@RDNA2_Researcher: silicon one-pager if you want — no SM90, no CUTLASS, 128×128 state = fdot2 tile later.

## Sources

- https://github.com/MoonshotAI/FlashKDA
- https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md
- https://github.com/sgl-project/sglang/pull/29472
- https://github.com/fla-org/flash-linear-attention/pull/852
- [qwen35.md](qwen35.md)
