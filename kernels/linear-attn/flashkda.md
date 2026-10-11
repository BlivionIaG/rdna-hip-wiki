---
title: "FlashKDA / KDA — kernel contract (Later)"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---
# FlashKDA / KDA — kernel contract (Later)

Date: 2026-08-19. Silicon: [silicon/attention/flashkda.md](../../silicon/attention/flashkda.md). Engine: [engine/attention/flashkda.md](../../engine/attention/flashkda.md). **Not in extras.** Occupancy still first.[^isa]

Dead as a CUTLASS/SM90/bf16 port. If we serve Kimi Linear or Qwen3.5 GDN:[^isa]

| Kernel | Shape | Inner loop | Status |
|---|---|---|---|
| Decode `q @ S` | M=1, N=K=V=128 | `fdot2`, fp32 accum | Spec [^isa] |
| Prefill K2 recurrence | 128×128 `S`, chunk 16 | tiled `fdot2` | Spec [^isa] |
| Prefill K1 prep | gate / L2 / 16×16 invert | scalar/packed FMA | Spec [^isa] |

No `sdot4`. No `on_rdna()`. Reference = FLA Triton `chunk_kda`, not Moonshot `.cu`.[^isa]

## Sources

[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
