---
title: "Native HIP INT8 MoE on gfx1030 — dual-route silicon contract"
sources:
  - https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb
  - https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/moe_w8a16_rdna2.cu
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/clang/include/clang/Basic/BuiltinsAMDGPU.td#L448-L459
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292
  - https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/fa_rdna2.cu
---
# Native HIP INT8 MoE on gfx1030 — dual-route silicon contract

Date: 2026-08-18. **Spec / Todo.** Separate from [native FP16 MoE](fp16-moe.md). Target: gfx1030 wave32. Dispatch by routed rows **per expert**, not total tokens. Extras @ `83de31cf` has **W8A16 `fdot2` only** (`moe_w8a16_rdna2.cu`); W8A8 `sdot4` + act quant are still absent — [w8a8.md](../quant/w8a8.md).[^c-83de31cf][^f-83de31cf-moe-w8a16-rdna2-cu]

## Two routes, two meanings

| Route | A | W | Native inner op | Accum | Initial role |
|---|---|---|---|---|---|
| **W8A16** | fp16 | i8 + scale | i8→fp16 dequant, `V_DOT2_F32_F16` | fp32 | decode default [^isa] |
| **W8A8** | signed i8 + scale | signed i8 + scale | `V_DOT4_I32_I8` (`sdot4`/DP4A) | i32, then scaled fp32 | prefill/fat experts [^isa] |

W8A16 is INT8 **storage**, not INT8 compute. W8A8 is the native integer-compute path. Keep separate kernels and packed layouts; do not add a runtime branch in the K loop.[^isa][^clang-dot][^f-83de31cf-moe-w8a16-rdna2-cu]

## Shared engine/layout contract

Consume vLLM's existing sorted/aligned routed-token metadata. One-time weight packing only:[^isa][^clang-dot]

- Expert-major outer dimension, K contiguous.
- `w13` keeps adjacent gate/up ownership so one thread can fuse `silu(gate)*up`.[^isa][^clang-dot]
- `w2` fuses route scale in fp32 before fp16 conversion; combine routed rows only where TP/EP semantics allow.[^isa][^clang-dot]
- Empty experts and expert sentinel `-1` return without touching output.[^isa][^clang-dot]

Both routes have tiny-bucket specializations `M={1,2,4,8}` plus a grouped prefill kernel. Crossover is benchmarked over rows/expert `{1,2,4,8,16,32,64,128}`.[^isa][^clang-dot]

# Route A — W8A16 dequant → `fdot2`

## Format and ISA

- W physical layout: signed i8 K-contiguous plus per-channel or per-group fp16 scale (and zero only when the checkpoint requires it).[^isa][^clang-dot]
- Load packed weight bytes, sign-extend/convert pairs to `half2`, apply scale/zero in VGPR, then issue:[^isa][^clang-dot]

```cpp
acc_f32 = __builtin_amdgcn_fdot2(a_half2, w_half2, acc_f32, false);
```

ISA must contain `v_dot2_f32_f16`; never call this DP4A. Activations stay fp16, so there is no activation-quant reduction or i8 activation workspace.[^isa]

## Decode seed

| Item | Contract |
|---|---|
| WG | 128 threads / 4 wave32 [^isa][^clang-dot] |
| N tile | 512 outputs; wave owns 128, lane owns 4 [^isa][^clang-dot] |
| K stage | 256 values [^isa][^clang-dot] |
| A LDS | `M x (256+8)` fp16 = 528 B..4224 B [^isa][^clang-dot] |
| B | streamed packed i8; dequant in VGPR [^isa][^clang-dot] |
| Acc/lane | `4*M` fp32 [^isa][^clang-dot] |

Reuse each dequantized weight pair across all M rows before advancing K. Target <=64 VGPR/lane, no scratch; split M8 into two M4 passes rather than spill.[^isa][^clang-dot]

## Prefill seed

Start **64x64x32**, WG256. LDS per stage: A fp16 `64x32` = 4 KiB; packed B i8 `32x64` = 2 KiB; **12 KiB double-buffered** before padding. Dequant B to `half2` in VGPR and use fp32 `fdot2` accumulation. Sweep BM `{32,64,128}`, BN `{32,64}`, BK `{32,64}` only after the seed is correct.[^isa][^clang-dot]

# Route B — W8A8 native `sdot4`

## Quant/packing contract

- Symmetric signed i8 on both sides; pack four adjacent K bytes per dword, K divisible by 4.[^isa][^clang-dot]
- Preferred scales: dynamic per-token A (`[M,1]`) and per-channel W (`[N]`). Static per-tensor is allowed only when specified by the checkpoint.[^isa][^clang-dot]
- At M=1, scale tensor shape `(1,1)` is still **per-token**, not per-tensor.[^isa][^clang-dot]
- gfx1030 has signed `sdot4` and unsigned `udot4`, but no gfx11 mixed-sign `sudot4`; do not mix signed and unsigned payloads.[^clang-dot][^llvm-gfx1030]

```cpp
acc_i32 = __builtin_amdgcn_sdot4(a_i8x4, w_i8x4, acc_i32, false);
float y = float(acc_i32) * a_scale[row] * w_scale[col];
```

Check `K*127*127 < INT32_MAX` for every supported shape. Apply bias/SiLU/route scale in fp32 after dequantizing the accumulator.[^isa][^clang-dot]

## Activation quantization

Decode may fuse absmax + quantization into the GEMM launch only if it does not serialize all output work behind repeated reductions. Compute each routed row's scale exactly once, stage packed A8, then reuse it across N tiles. Prefill should use one graph-safe quant kernel producing packed A8 plus per-row scales, followed by grouped GEMM; compare fusion rather than assuming it wins.[^isa][^clang-dot]

## Decode seed

Same ownership as W8A16: WG128, N512, K-stage256, `4*M` i32 accumulators/lane. A LDS is `M x (256+8)` **bytes** = 264 B..2112 B; B is streamed i8. Initial policy: use only when a prequantized/static A8 input exists or measured fused quant beats W8A16.[^isa][^clang-dot]

## Prefill seed

Start **64x64x64 i8**, WG256, 4x4 outputs/thread. A and B are 4 KiB each per stage; double buffer = **16 KiB** before padding. Inner loop consumes packed dwords with explicit `sdot4`; no WMMA/MFMA/AITER `(32,16)` shuffle.[^isa][^clang-dot]

# LDS and occupancy

gfx1030 WGP LDS is 64 banks at 4 B/bank. Read `half2` or packed i8 dwords, not scalar halves/bytes. Pad row strides initially (`+8` elements is a seed) and sweep `{0,4,8}` plus XOR swizzle using measured conflicts. Separate `vmcnt` global waits from `lgkmcnt` LDS waits.[^isa]

For each M/tile report VGPR, LDS, spills, waves/SIMD32 and runtime active blocks. Do not blindly use HIP `amdgpu_waves_per_eu(4,8)`; verify the register budget first. The current `fa_rdna2` occupancy fix remains higher priority.[^f-83de31cf-fa-rdna2-cu]

# Dispatch hypothesis to test

1. Rows/expert 1–8: W8A16 default; test W8A8 only with static/prequantized A8 or fused quant.[^isa][^clang-dot]
2. Rows/expert >=16/32: W8A8 expected candidate because activation quant is amortized and `sdot4` doubles packed work versus `fdot2`.[^isa][^clang-dot]
3. Final crossover is per shape `(Mexpert,K,N)`, scale scheme, and TP rank; record a small dispatch table, not one global threshold.[^isa][^clang-dot]
4. Fallback on unsupported zero-points, layouts, tails, or overflow risk.[^isa][^clang-dot]

# Correctness/performance gates

- Parity for empty/skewed experts, tails and `top_k={1,2,8}`.[^isa][^clang-dot]
- W8A16: prove `v_dot2_f32_f16`; W8A8: prove `v_dot4_i32_i8`; no scratch.[^isa]
- Dynamic quant parity, saturation and `(1,1)` per-token-scale case.[^isa][^clang-dot]
- Fused w13 SiLU×up and w2 route-scale against PyTorch fp32 reference.[^isa][^clang-dot]
- TP=4 and graph safety; no forward allocation, packing or host sync.[^isa][^clang-dot]
- Report quant time separately, GEMM-only and routed end-to-end, plus DRAM bytes, IC/L2 hit rate, occupancy and launch count.[^isa][^clang-dot]

# Landing order

1. Keep/fix existing W8A16 MoE path and add ISA/occupancy dumps.[^isa][^clang-dot]
2. Add W8A8 activation quant + reference grouped op.[^isa][^clang-dot]
3. Decode W8A8 only after quant cost is visible.[^isa][^clang-dot]
4. 64x64x64 W8A8 prefill seed and crossover sweep.[^isa][^clang-dot]
5. One dispatcher choosing W8A16 versus W8A8 outside the K loop.[^isa][^clang-dot]

## Sources

[^c-83de31cf]: BlivionIaG/vllm commit [`83de31cf`](https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb) (2026-09-01) — "fix(rocm/layernorm): close namespace braces before global-scope public entries".
[^f-83de31cf-moe-w8a16-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/moe_w8a16_rdna2.cu` @ [`83de31cf`](https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/moe_w8a16_rdna2.cu).
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^clang-dot]: Clang `BuiltinsAMDGPU.td` @ llvm-project `eb61ef62df` L448–L459: `fdot2` = `dot10-insts`, `fdot2_f32_bf16` = `dot12-insts`, `sdot4`/`sdot8` = `dot1-insts`, `udot4`/`udot8` = `dot7-insts`, `sudot4`/`sudot8` = `dot8-insts`. <https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/clang/include/clang/Basic/BuiltinsAMDGPU.td#L448-L459>
[^llvm-gfx1030]: LLVM `AMDGPU.td` @ `eb61ef62df` L2278–L2292, `FeatureISAVersion10_3_0` (gfx1030): `FeatureDot1/2/5/6/7/10Insts`, no `Dot8`/`Dot12`, `FeatureMaxWavesPerEU16`. <https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292>
[^f-83de31cf-fa-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/fa_rdna2.cu` @ [`83de31cf`](https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/fa_rdna2.cu).
