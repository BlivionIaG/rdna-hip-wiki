# RDNA HIP wiki

Sourced notes for custom HIP kernels on **gfx1030** (RDNA 2 / V620) and the gfx1100 delta. Target work: W4A16, W8A8, mxfp4, and `fa_rdna2` on a 4× V620 box (ROCm 7.2, vLLM fork).

Do not invent IC TB/s, L2 associativity, or a P2P-works claim. Hardware numbers come from ISA 70648, LLVM, HIP, AMD product pages, and `kfd_crat.c`.

## Layout

| Folder | What lives here | Owner |
|---|---|---|
| [silicon/](silicon/README.md) | Arch, LDS, cache, RCCL/P2P, HIP craft, occupancy, codegen | RDNA2_Researcher |
| [kernels/](kernels/README.md) | Format contracts: W4A16, W8A8, mxfp4 | RDNA2_Researcher |
| [engine/](engine/README.md) | vLLM/SGLang dispatch, P/D, MoE, KV + [notes](engine/notes/README.md) | LLM_Inference_specialist |
| [fork/](fork/README.md) | Branch gates, tickets, what not to touch | VLLM_FORK_Manager |

## Hardware contract (gfx1030)

- Wave32 native, WGP default, 16 waves/SIMD, 1024 VGPR/SIMD32.
- No WMMA, no MFMA, no FP8 unit, no FP4 unit, no bf16 matrix.
- Inner ops: `fdot2` (W4A16 / W8A16 / mxfp4-after-unpack), `sdot4` (W8A8 i8×i8).
- LDS 128 KB/WGP, 64 KB/WG, 64 banks. V620: 72 CU, 128 MB IC, 4 MB L2, GDDR6 512 GB/s.
- TP=4 is RCCL over PCIe 4.0. No XGMI. Custom all-reduce is gfx94/95 only.

## Start here

1. [silicon/architecture.md](silicon/architecture.md) — CU / WGP / caches
2. [silicon/hip-craft.md](silicon/hip-craft.md) — waitcnt, occupancy, builtins
3. [kernels/w4a16.md](kernels/w4a16.md) — the live dense/MoE path
4. [silicon/fa-occupancy.md](silicon/fa-occupancy.md) — `fa_rdna2` launch_bounds / LDS
5. [engine/full-map.md](engine/full-map.md) — one-page engine verdict + technique table
6. [engine/attention-dispatch.md](engine/attention-dispatch.md) — fa_rdna2 vs Triton, head-64, short vs split-K
7. [engine/notes/kiely-inference-engineering.md](engine/notes/kiely-inference-engineering.md) — Kiely mapped to gfx1030

Fork branch `perf/rdna2_w4a16` is human-only. Tickets go on [project 4](https://github.com/users/BlivionIaG/projects/4).
