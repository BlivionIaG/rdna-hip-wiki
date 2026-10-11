---
title: "sdot4 explore — which kernels actually use it"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://gpuopen.com/learn/wmma_on_rdna3/
  - https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb
  - https://github.com/opengfx1030/vllm-rdna/commit/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0
  - https://github.com/opengfx1030/vllm-rdna/blob/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0/csrc/rocm/w4a8_sdot4_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0/csrc/rocm/fa_rdna2.cu
---
# sdot4 explore — which kernels actually use it

Explore list, not a land-now ticket. Occupancy leftover is **FA prefill `(N, 1)` / LDS 64 KiB WG** — pin closed. Extras vs contract: [w8a8.md](w8a8.md). Engine index: [fork-delta.md](../../engine/serving-stack/fork-delta.md).[^isa]

`V_DOT4_I32_I8` / `__builtin_amdgcn_sdot4` is 4×i8 → i32. GPUOpen RDNA2: IU8 **512** ops/clk/CU vs packed FP16 **256**. That 2× only shows up when **both** operands are integer i8 **and** the kernel is compute-bound.[^isa][^gpuopen]

`V_DOT8_I32_I4` / `sdot8` is 8×i4 → i32 (IU4 **1024**). **W4A4 integer only.** See [w4a4.md](../../engine/quant-formats/w4a4.md).[^isa][^gpuopen]

## Yes — explore

| Kernel | Why | Tile / notes | Existing page |
|---|---|---|---|
| **W8A8 INT8 dense + MoE** | Both sides i8. Prefill GEMM is compute-bound. | Prefill **64×64×64 i8**, WG=256, LDS 8 KB (recipe 9). BK%4==0. i32 through K, scale in epilogue. Decode `M=1/2/4` skinny, A in LDS. | extras vs contract: [w8a8.md](w8a8.md). Study: [w8a8-mxfp4.md](w8a8-mxfp4.md). **Not in extras @ `83de31cf`.** [^c-83de31cf] |
| **Sage QK** (prefill) | Q and K both i8. QK is the FLOP side of prefill attn. | Pack D as 4×i8. PV stays `fdot2`. | [sage-qk.md](../attention/sage-qk.md) [^isa][^gpuopen] |
| **W4A8** | W is i4, A is i8. Unpack nibble → i8, then `sdot4`. | **Live / opt-in** @ `3a0786ea`: wired tile `a8_lds_k32_ag` (not 64×64×64). `sdot8` is **W4A4 only**. | [w4a8.md](w4a8.md) [^c-3a0786ea][^f-3a0786ea-w4a8-sdot4-rdna2-cu] |
| **W4A4 integer** (later) | Both sides i4. `sdot8`. Prefill first. | After W8A8. Decode stays W4A16 unless measured. Not E2M1. | [w4a4.md](../../engine/quant-formats/w4a4.md) [^isa][^gpuopen] |

## No — do not open an sdot4 variant

| Kernel | Why |
|---|---|
| W4A16 / W8A16 | A is fp16. Dequant W → `fdot2`. [^isa][^gpuopen] |
| W8A16-FP8 / W8A8-FP8 | Bytes are E4M3, not integer i8. Integer DOT on FP8 bits is wrong. Bit-trick / LUT → fp16 → `fdot2`. [^isa][^gpuopen] |
| mxfp4 / NVFP4 | E2M1 unpack → fp16 → `fdot2`. `sdot8` on E2M1 is wrong math. [^isa][^gpuopen] |
| INT8 KV in `fa_rdna2` | Storage only. Q stays fp16. Decode is KV bandwidth. Quantizing Q every step is Sage, and still will not beat gather. [^f-3a0786ea-fa-rdna2-cu] |
| HIP MLA decode | Gather + FP8 unpack is the sink. Scalar FMA → later `fdot2`, not `sdot4`. [^isa][^gpuopen] |
| Skinny GEMM | Occupancy trap. No MFMA on gfx1030. Not an INT8 kernel. [^isa][^gpuopen] |

## W4A8 sketch (superseded by land @ `3a0786ea`)

```
load 8×i4 as uint32
unpack nibbles → 8×i8 in VGPR          // sign-extend, no LUT
sdot4 against packed A i8              // i32 accum
epilogue: C *= a_scale * w_scale
```

Pack W K-contiguous, `K % 8 == 0` (two sdot4 per dword of W). Same LDS seed as W8A8 after the unpack. No `sdot8` unless someone later ships integer W4A4 and accepts the numerics — that is [w4a4.md](../../engine/quant-formats/w4a4.md), not this sketch.[^c-3a0786ea]

## Cards

Reuse the existing W8A8 INT8 and Sage cards (spec / Todo). W4A8 `sdot4` **landed** opt-in on `rdna_extras` @ `3a0786ea` — see [w4a8.md](w4a8.md). Do not reopen an explore card for dense/MoE W4A8. Add **one** explore card: W4A4 integer `sdot8` ([w4a4.md](../../engine/quant-formats/w4a4.md)). Do not clone a card per “no” row. Do not use the W4A4 card for MXFP4/NVFP4 A4.[^c-3a0786ea]

## Sources

- RDNA 2 ISA 70648: `V_DOT4_I32_I8`, `V_DOT8_I32_I4`
- GPUOpen WMMA table: IU8 512, IU4 1024, packed FP16 256
- LDS recipe 9: [silicon/lds-and-tiles/lds-tiles.md](../../silicon/lds-and-tiles/lds-tiles.md)

[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^gpuopen]: GPUOpen, *How to accelerate AI applications on RDNA 3 using WMMA* — per-CU FLOPS/clock table (RX 6950 XT: FP16 256, BF16 N/A, IU8 512, IU4 1024; RX 7900 XTX: FP16/BF16 512). <https://gpuopen.com/learn/wmma_on_rdna3/>
[^c-83de31cf]: BlivionIaG/vllm commit [`83de31cf`](https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb) (2026-09-01) — "fix(rocm/layernorm): close namespace braces before global-scope public entries".
[^c-3a0786ea]: opengfx1030/vllm-rdna commit [`3a0786ea`](https://github.com/opengfx1030/vllm-rdna/commit/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0) (2026-09-30) — "merge(rdna2): land w4a8-wiring onto rdna_extras (W4A8 opt-in; W4A16 default unchanged)".
[^f-3a0786ea-w4a8-sdot4-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/w4a8_sdot4_rdna2.cu` @ [`3a0786ea`](https://github.com/opengfx1030/vllm-rdna/blob/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0/csrc/rocm/w4a8_sdot4_rdna2.cu).
[^f-3a0786ea-fa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/fa_rdna2.cu` @ [`3a0786ea`](https://github.com/opengfx1030/vllm-rdna/blob/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0/csrc/rocm/fa_rdna2.cu).
