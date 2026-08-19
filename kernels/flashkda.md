# FlashKDA / KDA — kernel contract (Later)

Date: 2026-08-19. Silicon: [../silicon/flashkda.md](../silicon/flashkda.md). Engine: [../engine/flashkda.md](../engine/flashkda.md). **Not in extras.** Occupancy still first.

Dead as a CUTLASS/SM90/bf16 port. If we serve Kimi Linear or Qwen3.5 GDN:

| Kernel | Shape | Inner loop | Status |
|---|---|---|---|
| Decode `q @ S` | M=1, N=K=V=128 | `fdot2`, fp32 accum | Spec |
| Prefill K2 recurrence | 128×128 `S`, chunk 16 | tiled `fdot2` | Spec |
| Prefill K1 prep | gate / L2 / 16×16 invert | scalar/packed FMA | Spec |

No `sdot4`. No `on_rdna()`. Reference = FLA Triton `chunk_kda`, not Moonshot `.cu`.
