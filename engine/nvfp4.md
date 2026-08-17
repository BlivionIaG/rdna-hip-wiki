# NVFP4 on gfx1030 — engine spec

Date: 2026-08-17. Engine contract. Silicon tile / ISA dump lives with RDNA2_Researcher under [kernels/](../kernels/README.md) once they add it. Tickets: one card on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not edit `perf/rdna2_w4a16` from this page. Occupancy is still the current code subject; this is the queued spec.

**Verdict:** there is no NVFP4 unit on gfx1030. We run NVFP4 checkpoints by unpacking E2M1 to fp16 on the fly and issuing `fdot2`. That is the same inner op as live mxfp4 (`mxfp4_dot2_common.cuh`). The delta is the scale path, not a new DOT.

Native NVFP4 MMA (Blackwell / CUTLASS / Marlin / FlashInfer) is **Dead**. `V_DOT8_I32_I4` on raw E2M1 nibbles is **wrong math** (signed i4 ≠ E2M1 codebook `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`).

## Format (checkpoint, not the kernel)

| | OCP MXFP4 (live sources) | NVIDIA NVFP4 (this spec) |
|---|---|---|
| Element | E2M1 (S1E2M1), 4 bit | **same** E2M1 |
| Block | 32 elements | **16** elements |
| Scale | UE8M0 / E8M0, `2^(s8-127)` | **FP8 E4M3** per block |
| Extra | none | optional **FP32** per-tensor scale |
| Bits/elem | 4.25 | 4.50 |
| vLLM loaders | compressed-tensors / Quark MX | compressed-tensors / nvidia-modelopt `NVFP4` |
| Vendor compute | none here | Blackwell MMA — **do not call** |

Value: `x ≈ e2m1 * s_block_e4m3 * s_tensor_f32` (omit tensor scale if the checkpoint has none).

Layouts we accept (Transformer Engine / modelopt common case):

- Weights packed `uint8 [N, K/2]` (2 codes/byte, low nibble = even K).
- Block scales `float8_e4m3` or `uint8` viewed as E4M3, shape `[N, K/16]` (rowwise 1×16) or `[K/16, N]` after the one-time transpose.
- Optional FP32 tensor scale scalar / `[N]`.
- `K % 16 == 0`. Prefer `K % 32 == 0` so we share the mxfp4 K-tile (8 nibbles / dword × 4 dwords = 32).

Blackwell **scale swizzle** (16×16 / 128-byte MMA fragment) is a vendor-GEMM pack. **Drop it.** We keep linear scales, same as mxfp4 `process_weights_after_loading`.

## Inner op (locked)

```
// keep — same 8-nibble dword as mxfp4
dequant_e2m1_8_fp16(qa, /* then */ apply_e4m3_scale + optional f32);
acc = fdot2(dq[0:2], a[0:2], acc);
acc = fdot2(dq[2:4], a[2:4], acc);
acc = fdot2(dq[4:6], a[4:6], acc);
acc = fdot2(dq[6:8], a[6:8], acc);   // 4× fdot2 / 8 K

// drop
V_DOT8_I32_I4(qa, ...);             // i4 ≠ E2M1
CUTLASS / Marlin / FlashInfer NVFP4
AITER MX / Instinct FP4 MMA
bf16 / v_dot2_f32_bf16
```

Reuse `mxfp4_e2m1_to_fp16_bits` from `mxfp4_dot2_common.cuh`. **Do not reuse** `mxfp4_apply_e8m0_bits` (exponent add). NVFP4 scale is E4M3 × optional FP32: convert E4M3 → fp16 with the same bit-trick already used on W8A8-FP8 (`qdq_fp8_rdna2.cuh`), then `fmul` into the unpacked E2M1. Two E4M3 scales per 32-K window (groups of 16).

Dequant **on the fly in VGPR**, not a second weight tensor. Stage **A** in LDS (fp16). Stream packed B. Do not park a 16-entry nibble LUT in LDS (K$ serialize; llama.cpp #24438).

## A16 vs A4

| Checkpoint | Action |
|---|---|
| NVFP4 weights, fp16 activations (W4A16) | **first ship**. This is the kernel. |
| NVFP4 weights + NVFP4 activations (W4A4) | dequant A to fp16 (ignore act key) **or refuse**. Same `fdot2` kernel. Do not emit a software FP4×FP4 DOT. |
| NVFP4 weights + FP8 activations | dequant A to fp16 or refuse. `supports_fp8()` is false. |

W4A4-native is still **Dead**. Emulated A16 is the product.

## Dispatch (vLLM)

One method next to the live mxfp4 path. Do not fall through to Marlin / CUTLASS / FlashInfer / `VLLM_USE_FLASHINFER_MOE_FP4`.

| Piece | Spec |
|---|---|
| Gate | `on_gfx10x()` + NVFP4 quant key (compressed-tensors / modelopt). |
| Weight prep | one-time: `[N, K/2] u8` → `uint32 [K/8, N]`; scales → `[K/16, N]` E4M3; cache FP32 tensor scale. |
| Dense | `nvfp4_gemm_rdna2` — clone `mxfp4_gemm_rdna2` control flow. |
| MoE | `moe_nvfp4_gemm_rdna2` — clone `moe_mxfp4_gemm_rdna2`. `BLOCK_SIZE_M` in `{1,2,4,8}`. TP>1: write unreduced rows (no fused `output_topk`). |
| Shared experts / attn proj | same dense op. |
| Dtype | fp16 only. No bf16. |

Python: new `is_supported_nvfp4` / `make_method_nvfp4` chained **before** the CUDA NVFP4 fallbacks, same pattern as `is_supported_mxfp4` in `rocm_moe_rdna.py`.

## Geometry (seeds — silicon confirms)

Copy mxfp4 / W4A16, not Blackwell tiles.

| Regime | Seed |
|---|---|
| Decode `M=1,2,4,8` | skinny, `THREADS=256`, `BLOCK_KN=256`, 4 N-cols/thread, K-split `gridDim.z`, A in LDS |
| Prefill `M≥32` | tiled `fdot2` 64×64×32 (fp16-after-unpack) |
| Occupancy | no second `__launch_bounds__` min-blocks; `amdgpu_waves_per_eu(4, 8)` on decode-class |

`K % 16 == 0` required. Refresh E4M3 every 16 K (two scales per 32-K mxfp4 window).

## Done-when

ISA dump of the inner loop shows `v_dot2_f32_f16`, not `v_dot8_*`, not an FP4 MMA. Host `kFloat16`. E4M3 scale applied (not E8M0). One NVFP4 checkpoint loads on gfx1030 without FlashInfer/Marlin. GPU smoke vs a CPU E2M1×E4M3×f32 reference. No tok/s claimed from this page.

## Not this ticket

Occupancy flip (`fa_rdna2` / `skinny_gemms`). Sage QK. HIP MLA fp16. W8A8 `sdot4`. INT8 KV. A new E2M1 LUT (already exists).

## Sources

- NVIDIA NVFP4: E2M1 + E4M3 / 16 + optional FP32 ([blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), [TE](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html))
- Live mxfp4: `mxfp4_dot2_common.cuh` @ `290715e6` (4× `fdot2` / 8 K)
- [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md) §1 / §6—7 (unpack → `fdot2`; no `sdot8` on E2M1)
- [coverage.md](coverage.md)
