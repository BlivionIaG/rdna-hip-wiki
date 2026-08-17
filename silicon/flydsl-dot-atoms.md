# FlyDSL DOT atoms for gfx1030 — silicon feasibility contract

Snapshot: ROCm/FlyDSL `96af91bcab7703df1b8d5536ffbb4ff72b6dfacd` (2026-08-17), release v0.3.0. Engine order: [engine/flydsl.md](../engine/flydsl.md). **Not implemented or supported today.**

## Verdict

FlyDSL is AMD-first: Python tracing → Fly/GPU/arith/vector MLIR → Fly layout lowering → atom-to-SSA → `convert-fly-to-rocdl` → GPU-to-ROCDL → LLVM → AMDGPU HSACO. The in-tree native backend is ROCDL, and `gfx10*` is classified as RDNA/wave32.

This makes gfx1030 generic code generation plausible. It does **not** provide a gfx1030 GEMM/MoE/attention kernel: official verified platforms omit gfx1030, optimized kernels use CDNA MFMA or newer-RDNA WMMA, and no DOT atom/device test exists.

## What the pinned toolchain already has

FlyDSL pins LLVM `e2a39f504fee836e4def9581bed817ecc327b9dc`.

- MLIR ROCDL defines `ROCDL_fdot2` → `llvm.amdgcn.fdot2`.
- MLIR ROCDL defines `ROCDL_sdot4` → `llvm.amdgcn.sdot4`.
- AMDGPU defines `V_DOT2_F32_F16` behind `HasDot10Insts`.
- AMDGPU defines `V_DOT4_I32_I8` behind signed-DOT features.
- gfx1030 feature set includes `FeatureDot1Insts` and `FeatureDot10Insts`.

The missing layer is a Fly atom/front-end integration plus proof, not LLVM ISA support.

## Required atom A: FP16 DOT2

Model one packed intra-lane operation, not a matrix core:

| Field | Contract |
|---|---|
| Logical shape | `(M,N,K)=(1,1,2)` |
| Thread layout | one thread |
| A/B | `vector<2xf16>` / packed `half2` |
| C/D | scalar `f32` |
| Lowering | `ROCDL::fdot2`, clamp=false |
| Final ISA | `v_dot2_f32_f16` or carry form |

Semantics: `d = a0*b0 + a1*b1 + c`, fp32 accumulator.

## Required atom B: signed INT8 DOT4

| Field | Contract |
|---|---|
| Logical shape | `(1,1,4)` |
| Thread layout | one thread |
| A/B | logical `vector<4xi8>`, packed little-endian to `i32` |
| C/D | scalar signed `i32` |
| Lowering | `ROCDL::sdot4`, clamp=false |
| Final ISA | `v_dot4_i32_i8` or carry form |

Do not substitute `sudot4`: mixed-sign DOT is gfx11+, not gfx1030.

## FlyDSL integration shape

Implement a stateless `MmaOp` payload for each atom:

1. TableGen declaration and verifier.
2. C++ type implementation with `getThrLayout()` and ThrVal layouts.
3. Both `emitAtomCall` and `emitAtomCallSSA`; the normal pipeline runs `fly-convert-atom-call-to-ssa-form`.
4. CMake registration.
5. Python binding/wrapper and architecture dispatch.
6. MLIR lowering, ISA and device tests.

The official atom-authoring recipe permits single-thread atoms and arbitrary backend intrinsics. Once implemented, these atoms can compose through `make_mma_atom`, `make_tiled_mma` and potentially `fx.gemm` without pretending to be MFMA/WMMA.

A direct Python ODS call to re-exported `rocdl.fdot2/sdot4` may be a faster smoke test, but no in-tree use/device test proves its builder ergonomics. Use it only for Gate 1; formal tiled work should use an atom.

## Architecture matrix

| Architecture | Current fast atom | Status |
|---|---|---|
| gfx942/gfx950 | MFMA / scaled MFMA | officially verified |
| gfx1030 | none | generic ROCDL path only; no device proof |
| gfx1100–gfx1152 | wave32 WMMA 16x16x16 | source/lowering/device tests exist, not official verified list |
| gfx1200/1201 | newer WMMA ABI | gfx1201 verified |
| gfx1250 | WMMA/scaled-WMMA/TDM | verified |

`examples/03-tiledMma.py` and `04-preshuffle_gemm.py` are MFMA-centric. Current gfx11 GEMM uses WMMA. Existing MoE is chiefly CDNA MFMA or gfx1250-specific. None is a gfx1030 starting kernel.

## Proof gates, in order

0. Compile/run vector-add with `chip=gfx1030`; verify wave32 code object.
1. Lower standalone atom call and SSA form to `rocdl.fdot2/sdot4`.
2. Disassemble exact `v_dot2_f32_f16` / `v_dot4_i32_i8`; reject scalarized mul/add.
3. Device correctness: FP16 NaN/Inf/subnormal/denormal behavior; signed i8 edges, packing, accumulation and clamp=false.
4. Repeated K reduction and tiled composition through Fly layouts.
5. Add gfx1030 LDS capacity: current `SMEM_CAPACITY_MAP` omits it. Use 64 KiB/WG as the hard allocation cap and retain the 128 KiB/WGP occupancy model externally.
6. ISA/VGPR/LDS/spill/occupancy comparison against direct HIP intrinsics.
7. Warm/cold JIT and graph-safety proof before vLLM integration.
8. Explicitly reject chips without the relevant DOT feature instead of late AMDGPU failure.

## First useful kernels

Only after the atoms pass:

1. Dense FP16 GEMV/GEMM `M=1/2/4/8` using the same ownership as [fp16-moe.md](../kernels/fp16-moe.md).
2. W8A8 skinny GEMM with packed signed i8 and i32 scale epilogue from [int8-moe.md](../kernels/int8-moe.md).
3. Compare FlyDSL vs Triton vs direct HIP vs rocBLAS with identical layout and epilogue.
4. MoE routing/epilogues later. FlashAttention last; current FlyDSL “generic” FA is wave64/MFMA, not RDNA2-generic.

## Unknowns

- No gfx1030 CI/device result or FlyDSL issue/PR for RDNA2 DOT was found.
- No existing Fly DOT atom or repository use of ROCDL fdot2/sdot4 was found.
- Direct Python ODS construction is untested.
- Final ISA from Fly-produced IR is unproven until Gate 2.
- No gfx10/gfx11 MoE implementation was found.

## Primary sources

- Backend pipeline: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/python/flydsl/compiler/backends/rocm.py
- Architecture classification: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/python/flydsl/runtime/device.py
- Atom declarations: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td
- gfx11 atom implementation: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/lib/Dialect/FlyROCDL/GFX11/MmaAtom.cpp
- Atom-authoring recipe: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/.claude/skills/add-target-atom-op/SKILL.md
- Pinned LLVM: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/thirdparty/llvm-build-info.json
- ROCDL ops: https://github.com/llvm/llvm-project/blob/e2a39f504fee836e4def9581bed817ecc327b9dc/mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td
- gfx1030 features: https://github.com/llvm/llvm-project/blob/e2a39f504fee836e4def9581bed817ecc327b9dc/llvm/lib/Target/AMDGPU/AMDGPU.td
- LDS map: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/python/flydsl/utils/smem_allocator.py
- Verified platforms: https://github.com/ROCm/FlyDSL/blob/96af91bcab7703df1b8d5536ffbb4ff72b6dfacd/README.md
