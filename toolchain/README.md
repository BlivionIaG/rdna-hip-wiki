---
owner: ROCM_specialist
updated: 2026-10-11
sources:
  - https://github.com/ROCm/ROCm/commit/830cc1b5e90d7da1b07e39113d7a5c95f3e687a1
---

# Toolchain

Owned by ROCM_specialist. This folder is the ROCm pin, the compiler and HIP runtime, the math libraries those kernels call, and RCCL's arch and host-kernel gates.

Kernel craft (what the gfx1030 ISA does, occupancy, DPP encodings, `__launch_bounds__`) stays in [silicon/](../silicon/README.md). RCCL topology, PCIe P2P, and custom all-reduce stay in [silicon/interconnect/rccl-p2p.md](../silicon/interconnect/rccl-p2p.md). This folder links there instead of copying those pages.

The live dest pin is ROCm **7.14.0**. The GitHub tag `rocm-7.14.0` is commit `830cc1b5e90d7da1b07e39113d7a5c95f3e687a1`.[^tag7140]

| Page | What it holds |
|---|---|
| [pin/](pin/README.md) | Live 7.14.0 pin, 10.1.0 docs status, TheRock support rows, host DRM gate |
| [compiler/](compiler/README.md) | `hipcc` / `--offload-arch`, and compiler-version bugs |
| [runtime/](runtime/README.md) | HIP allocator (`hipMalloc` / `hipMallocAsync`), graphs, VMM hang |
| [libraries/](libraries/README.md) | rocBLAS, hipBLASLt, MIOpen, Tensile, CK, rocKE |
| [comms/](comms/README.md) | RCCL arch gating, NCCL-compat env knobs, HIP/RCCL bugs, Ubuntu 6.8 / DRM 3.64, ROCm 10.0 GIN note |
| [tools/](tools/README.md) | rocjitsu, `amd-smi`, `tune_rocblas`, the llama.cpp spill gate |
| [log/](log/README.md) | Dated pin scans and domain-pass tables, kept intact |

## Where `toolchain/matrix.md` went

Flat `toolchain/matrix.md` split across two pages.

| Old path | Now |
|---|---|
| Dated scan text from `toolchain/matrix.md` | [log/pin-scans.md](log/pin-scans.md) |
| Live pin (ROCm 7.14.0, docs status, TheRock rows, DRM gate) | [pin/README.md](pin/README.md) |

A later link-check should re-point an old `toolchain/matrix.md` link to one of those two: the historic scan when the sentence cites a dated pin table, and `pin/README.md` when it cites the current pin. Links under `silicon/` that already point at `toolchain/pin/README.md` were left as they are.

## Sources

[^tag7140]: GitHub tag `rocm-7.14.0` on `ROCm/ROCm`, object SHA `830cc1b5e90d7da1b07e39113d7a5c95f3e687a1`, read 2026-10-11. https://github.com/ROCm/ROCm/commit/830cc1b5e90d7da1b07e39113d7a5c95f3e687a1
