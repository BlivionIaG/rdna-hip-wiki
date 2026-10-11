---
title: "Sage QK on gfx1030 (silicon contract)"
sources:
  - https://arxiv.org/abs/2410.02367
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://gpuopen.com/learn/wmma_on_rdna3/
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/fa_rdna2.cu
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/clang/include/clang/Basic/BuiltinsAMDGPU.td#L448-L459
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292
---
# Sage QK on gfx1030 (silicon contract)

Engine dispatch and the paper live in [engine/attention/sage-attention.md](../../engine/attention/sage-attention.md). This page is what a HIP kernel must do. Check a future implementation against this list.[^sage][^isa][^gpuopen]

**Paper:** SageAttention, Zhang et al., [arXiv 2410.02367](https://arxiv.org/abs/2410.02367) — INT8 QK + FP16 PV.[^sage]

## Do

- Prefill only. Decode stays `fa_rdna2` (`fdot2`).[^f-ea78104d-fa-rdna2-cu]
- QK: pack `D` as 4×i8 per dword. `D % 4 == 0` (128 and 256 are fine; head-64 is also fine).[^sage][^isa][^gpuopen]
- Inner loop: `__builtin_amdgcn_sdot4(q, k, acc, false)` → `v_dot4c_i32_i8`. Keep the K reduction in **i32**. Scale once in the epilogue (`scale_q * scale_k`).[^isa]
- Per-block / per-tile scales like the paper (not per-tensor only). Symmetric int8.[^sage][^isa][^gpuopen]
- PV: still `__builtin_amdgcn_fdot2` on fp16. Do not INT8 the V·P matmul on this chip unless you measure it.[^sage][^isa][^gpuopen]
- Wave32, WGP, LDS ≤ 64 KB/WG, `amdgpu_waves_per_eu(4, 8)`. Sequential `int` along D is conflict-free on 64 banks.[^isa]
- After occupancy flip on FA2. Do not land Sage on `__launch_bounds__(*, 1)`.[^sage]

## Do not

- `sudot4` (gfx11+).[^clang-dot][^llvm-gfx1030]
- `sdot8` / INT4 QK (Sage2) — no INT4 tensor path we trust for QK numerics.[^sage]
- FA3 (Hopper TMA/WGMMA/FP8).[^sage][^isa][^gpuopen]
- FP8 QK (`supports_fp8()` is false; no FP8 unit).[^sage][^isa][^gpuopen]
- Putting a nibble LUT in K$ (serializes).[^sage][^isa][^gpuopen]
- Claiming a decode win. Decode is KV bandwidth.

## Rate

GPUOpen RDNA2 table: IU8 **512** ops/clock/CU vs packed FP16 **256**. QK is the only place that 2× shows up. PV stays on the FP16 pipe.[^gpuopen]

## Checklist when a kernel lands

- [ ] Opcode in the ISA dump is `v_dot4c_i32_i8` / `v_dot4_i32_i8`, not `v_dot2c_f32_f16` on QK[^isa]
- [ ] `D` walked as `int`, `K` reduction in i32[^sage][^isa][^gpuopen]
- [ ] Scales applied once per tile, not per DOT4[^sage][^isa][^gpuopen]
- [ ] Occupancy > 0 at WG=256 (`llvm-calc-occupancy -mcpu=gfx1030` + runtime query)[^sage][^isa][^gpuopen]
- [ ] Prefill tok/s vs `fa_rdna2` prefill 128/256 on V620 (measure, don’t invent)[^f-ea78104d-fa-rdna2-cu]
- [ ] Decode path unchanged

## Sources

[^sage]: Zhang et al., *SageAttention: Accurate 8-bit attention for plug-and-play inference acceleration*, arXiv:2410.02367. <https://arxiv.org/abs/2410.02367>
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^gpuopen]: GPUOpen, *How to accelerate AI applications on RDNA 3 using WMMA* — per-CU FLOPS/clock table (RX 6950 XT: FP16 256, BF16 N/A, IU8 512, IU4 1024; RX 7900 XTX: FP16/BF16 512). <https://gpuopen.com/learn/wmma_on_rdna3/>
[^f-ea78104d-fa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/fa_rdna2.cu` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/fa_rdna2.cu).
[^clang-dot]: Clang `BuiltinsAMDGPU.td` @ llvm-project `eb61ef62df` L448–L459: `fdot2` = `dot10-insts`, `fdot2_f32_bf16` = `dot12-insts`, `sdot4`/`sdot8` = `dot1-insts`, `udot4`/`udot8` = `dot7-insts`, `sudot4`/`sudot8` = `dot8-insts`. <https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/clang/include/clang/Basic/BuiltinsAMDGPU.td#L448-L459>
[^llvm-gfx1030]: LLVM `AMDGPU.td` @ `eb61ef62df` L2278–L2292, `FeatureISAVersion10_3_0` (gfx1030): `FeatureDot1/2/5/6/7/10Insts`, no `Dot8`/`Dot12`, `FeatureMaxWavesPerEU16`. <https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292>
