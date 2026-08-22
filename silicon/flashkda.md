# FlashKDA — silicon (no SM90, no CUTLASS, 128×128 = fdot2 later)

Date: 2026-08-19. **Later / wiki only** until extras serves a KDA or GDN model. Engine: [../engine/flashkda.md](../engine/flashkda.md). Occupancy still first.

Repo: [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) (MIT, 2026-04). Deep-dive: [docs/20260420-flashkda-v1-deep-dive.md](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md).

## Dead as a port

| FlashKDA silicon | gfx1030 |
|---|---|
| SM90+ CUTLASS / wgmma / `MOVM_T` | **No.** No WMMA/MFMA. HIP extras. |
| CUDA 12.9+ | **No.** |
| **bf16** q/k/v/g + bf16 stored state | **No** hardware bf16 matrix (`v_dot2_f32_bf16` is gfx1100). |
| PTX `tanh.approx.f32` / `ex2.approx.ftz.f32` | Not a HIP builtin we pin. Use `v_exp_f32` / software sigmoid. |
| `__launch_bounds__(256, 8)` CUDA min-blocks | **Different knob.** HIP second arg is **waves/EU**. Do not paste `(256, 8)` onto extras. |
| H20 1.7–2.2× vs FLA Triton | **Not** a V620 number. |

Same class as Marlin / FA3: take the **math**, Leave the `.cu`.

## What the math actually is

Kimi Delta Attention is **per-channel-gated delta-rule linear attention**, not softmax. No paged KV. Recurrent state:

```text
S : [N, H, V, K]     K = V = 128   (fixed in FlashKDA)
```

That is a **128×128 fp matrix per head**, updated every token/chunk. Decode reads `S` as a skinny GEMV. Prefill walks chunks. `fa_rdna2` does not see this path.

Their v1 split (take this, Leave CUTLASS):

| Kernel | Grid | Work |
|---|---|---|
| **K1** token-parallel | `N × H × num_chunks` | gate act, QK L2, decay, `L`/`Mqk`, **16×16 invert** |
| **K2** head-parallel | `N × H` | chunked delta-rule recurrence, `S` accum, `out = q @ S` |

One fused kernel left SMs idle on K2. Same reason we split decode GEMV vs prefill GEMM.

**Chunk 16** (FLA Triton uses 64): they picked 16 so `exp(cumsum(g))` stays in **bf16** when `lower_bound ∈ [-5, 0]` (safe gate), and so the inverse is a cheap Neumann series. Unbounded gate → stay on FLA `chunk_kda` (chunk 64, high-prec). We would keep that policy split.

Precision they use (do **not** copy dtypes):

- I/O bf16; **state stored bf16**, **update in fp32 FMA**
- 16×16 inverse in **fp16** (elements ∈ [-1, 1])

On gfx1030: **fp16 everywhere, fp32 accum.** If a checkpoint is bf16, convert at the edge.

## gfx1030 inner loop (only if we serve KDA/GDN)

`K = V = 128` is `fdot2`-shaped: 64 `half2` pairs. Peak path is [fp16-rdna2.md](fp16-rdna2.md): explicit `__builtin_amdgcn_fdot2`, fp32 accum, wave32.

| Stage | Tile / shape | Opcode |
|---|---|---|
| Decode `out = q @ S` | M=1, N=128, K=128 skinny GEMV | **`fdot2`** |
| Prefill `S += …` / `q @ S` | 128×128 state, chunk 16 | tiled **`fdot2`** (seed 64×64×32) or measure rocBLAS first |
| 16×16 invert / gate / L2 | tiny | scalar / packed FMA — **not** DOT |
| INT8/FP8 | — | **No.** This is not `sdot4` / `sdot8`. |

LDS: 128×128 fp16 = **32 KiB** per state. Double-buffer + K1 scratch must stay ≤ **64 KiB/WG**. Bank-stride like [lds-tiles.md](lds-tiles.md). Occupancy: pin `amdgpu_waves_per_eu(4, 8)`, **not** `(1, 1)` and **not** their CUDA `(256, 8)`.

**Do not import:** CUTLASS, wgmma, `MOVM_T`, SM80 MMA fragments, AITER, `on_rdna()` as a gfx1030 gate. Reference is FLA Triton `chunk_kda`, then a HIP rewrite.

Cousins: Qwen3.5 **GDN** is the same linear+full-attn family ([../engine/qwen35.md](../engine/qwen35.md)). One HIP later, two names. extras has **no** KDA today.

## Order

1. Occupancy / `fa_rdna2` / live extras. **Not this.**
2. A real Kimi Linear / GDN checkpoint on the box.
3. FLA Triton `chunk_kda` baseline (fp16).
4. HIP: decode skinny `fdot2` against `S`, then prefill K1/K2 split.

No tok/s from this page. No occupancy retip.

## Sources

- https://github.com/MoonshotAI/FlashKDA
- https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md (CHUNK=16, K1/K2, bf16 state, fp16 invert, `launch_bounds(256,8)`)
- https://github.com/MoonshotAI/FlashKDA/blob/master/README.md (`K=V=128`, SM90+, `cu_seqlens`)
- [../engine/flashkda.md](../engine/flashkda.md), [fp16-rdna2.md](fp16-rdna2.md), [valu.md](valu.md)
