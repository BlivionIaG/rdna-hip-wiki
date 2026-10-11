# FlyDSL on gfx1030 — feasibility before integration

Date: 2026-08-18. Source snapshot: `ROCm/FlyDSL` `96af91bc` (2026-08-17). **Research / not supported.** This is not a vLLM backend claim.

## Verdict

FlyDSL's compiler substrate may emit ordinary ROCDL/LLVM code for `gfx1030`, but **none of its current optimized GEMM, MoE, or FlashAttention kernels support RDNA2**. Existing fast kernels depend on CDNA MFMA or gfx11/gfx12 WMMA. The first useful experiment is a tiny DOT/GEMV proof, not porting an MFMA pipeline.

Gates, in order:

0. `gfx1030` code object generation + vector-add correctness.
1. Explicit `fdot2` / `sdot4` wrappers lower to clean ISA.
2. Small-M GEMV correctness and graph-safe warm/cache behavior.
3. Occupancy + performance against Triton, stock HIP skinny and rocBLAS.
4. Only then consider vLLM integration. MoE/attention remain later rewrites.

## What is portable

Pipeline:

```text
Python trace → Fly/layout IR → GPU/ROCDL → LLVM → HSACO
```

- `python/flydsl/compiler/backends/rocm.py`: `RocmBackend.detect_target`, `make_target`, `_pipeline_parts` pass the detected chip to ROCDL/LLVM.
- `python/flydsl/runtime/device.py`: `is_rdna_arch()` accepts `gfx10*`, `gfx11*`, `gfx120*`; RDNA uses wave32.
- No top-level compiler whitelist rejects `gfx1030`.
- PyPI wheels package the compiler/runtime; GPU code is JIT-generated. A wheel existing does **not** prove gfx1030 kernel support.

Official wheel CI validates MI325/gfx942 and MI355/gfx950. README verified platforms do not include gfx1030.

## What is not portable

| FlyDSL area | Current assumption | gfx1030 verdict |
|---|---|---|
| `kernels/gemm/preshuffle_gemm.py` | MFMA pipeline / fragment preshuffle | Rewrite |
| `kernels/moe/moe_gemm_2stage` | gfx94/gfx95 gate, MFMA, CDNA layouts | Rejected before compile |
| `kernels/gemm/rdna3_f16_gemm.py` | gfx11 WMMA | Rejects gfx10 |
| `kernels/gemm/rdna_f16_gemm.py` | gfx120 WMMA ABI | Wrong ISA |
| `kernels/gemm/rdna_fp8_preshuffle_gemm.py` | RDNA4 FP8 WMMA | Wrong ISA |
| `kernels/attention/flash_attn_generic.py` | wave64 + `MFMA(32,32,16)` | “Generic” is not RDNA2-generic |
| `expr/rocdl/universal.py::WMMA` | gfx11/gfx120/gfx1250 dispatch | raises on gfx1030 |

Do not copy MFMA/WMMA preshuffled weight ABIs. The fragment layout is tied to hardware gfx1030 does not have.

## DOT path

gfx1030 supports:

- `llvm.amdgcn.fdot2(<2 x half>, <2 x half>, float, i1)` → `v_dot2_f32_f16` / `v_dot2c_f32_f16`.
- `llvm.amdgcn.sdot4(i32, i32, i32, i1)` → `v_dot4_i32_i8` / `v_dot4c_i32_i8`.

FlyDSL has no verified public wrappers for either. It does expose the needed escape hatches:

- intrinsic calls via `_llvm.call_intrinsic(...)` (`expr/rocdl/__init__.py`, e.g. `perm_b32`),
- AMDGCN inline assembly via `llvm.inline_asm(...)` (`expr/rocdl/inline_asm.py`).

Prefer LLVM intrinsics; inline assembly is fallback. ISA evidence is mandatory.

## Gate 0 — compiler smoke

1. Install a matching FlyDSL wheel/source on ROCm 7.2.
2. Set `FLYDSL_GPU_ARCH=gfx1030`.
3. Compile/run the simplest vector-add kernel.
4. Dump code object/ISA: correct gfx1030 target, wave32, no gfx11/12 opcodes.
5. Record first-JIT time, cached launch time, and cache key.

Failure here ends the project.

## Gate 1 — packed DOT microtests

Add experimental local wrappers:

```text
fly_fdot2(half2 a, half2 b, float c) -> float
fly_sdot4(int a4, int b4, int c) -> int
```

For each: one packed operation/lane, PyTorch reference, tails, negative INT8, clamp=false. Verify exact ISA (`v_dot2*` / `v_dot4*`), no scalarized multiply chain, no wrong wave ABI.

## Gate 2 — inference-relevant skinny kernels

First kernels:

- FP16 GEMV/GEMM `M=1/2/4/8`: `fdot2`, fp32 accum.
- W8A8 GEMV/GEMM: packed i8 + `sdot4`, i32 accum + scale epilogue.

Use Fly layout algebra for vector loads, LDS and reductions. Do not call `WMMA`/`MFMA` helpers. Start from the same contracts as [kernels/fp16-moe.md](../kernels/fp16-moe.md) and [kernels/int8-moe.md](../kernels/int8-moe.md), but prove a standalone dense skinny op first.

## Gate 3 — fair comparison

Same shape/layout/output contract against:

1. rocBLAS / `torch.mm`.
2. vLLM vecMatMul/LLMM1 and wvSplitK.
3. Triton skinny baseline/tuned variants.
4. Existing custom HIP.
5. FlyDSL cold JIT and warm cached execution separately.

Record ISA, VGPR/SGPR/LDS, spills, waves/SIMD32, kernel time, launch count, compile/JIT time, and graph replay. Stop if warm performance or startup cost cannot beat the standard baseline for a useful shape range.

## vLLM integration gate

Only after Gate 3:

- one narrow custom op / `forward_hip()` path,
- fake/meta implementation and `torch.library.opcheck`,
- warm every specialization before capture,
- no compilation/allocation/host sync during replay,
- non-default streams, concurrency and TP=4 tests,
- fallback remains stock.

FlyDSL launch through DLPack/ExecutionEngine does not automatically guarantee PyTorch/HIP graph safety.

## MoE and attention

**MoE:** current FlyDSL GEMM1/GEMM2 are gfx94/gfx95 MFMA. A gfx1030 port needs new DOT kernels plus routing/epilogue semantics. It is not an adaptation of the existing fragments.

**FlashAttention:** current “generic” kernel is wave64/MFMA. A gfx1030 version requires a new wave32 non-MFMA algorithm. Do not place it before the skinny proof.

## Project order

One FlyDSL research card, blocked behind baseline harness and skinny comparisons:

- [ ] Gate 0 gfx1030 object + vec-add.
- [ ] Gate 1 `fdot2` and `sdot4` intrinsic wrappers + ISA.
- [ ] Gate 2 FP16/W8A8 skinny proof.
- [ ] Gate 3 standard-vs-HIP-vs-Triton-vs-FlyDSL table.
- [ ] Gate 4 graph-safe vLLM prototype.
- [ ] Later: MoE; attention last.

## Sources

- `ROCm/FlyDSL` @ `96af91bc`: `docs/architecture_guide.md`
- `python/flydsl/compiler/backends/rocm.py`, `runtime/device.py`
- `python/flydsl/expr/rocdl/universal.py`, `expr/rocdl/inline_asm.py`
- `kernels/gemm/{preshuffle_gemm,rdna3_f16_gemm,rdna_f16_gemm,rdna_fp8_preshuffle_gemm}.py`
- `kernels/moe/moe_gemm_2stage/{gemm1,gemm2}.py`
- `kernels/attention/{flash_attn_generic,flash_attn_utils}.py`
- FlyDSL wheel/CI: `.github/workflows/{build-whl,test-whl}.yaml`
- LLVM AMDGPU gfx1030 assembler and `llvm.amdgcn.fdot2` tests
