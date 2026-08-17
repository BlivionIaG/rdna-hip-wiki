# sdot4 explore — which kernels actually use it

Explore list, not a land-now ticket. Occupancy (`fa_rdna2` / skinny) is still first. Engine index: [fork-delta.md](../engine/fork-delta.md).

`V_DOT4_I32_I8` / `__builtin_amdgcn_sdot4` is 4×i8 → i32. GPUOpen RDNA2: IU8 **512** ops/clk/CU vs packed FP16 **256**. That 2× only shows up when **both** operands are integer i8 **and** the kernel is compute-bound.

`V_DOT8_I32_I4` / `sdot8` is 8×i4 → i32 (IU4 **1024**). **W4A4 integer only.** See [w4a4.md](../engine/w4a4.md).

## Yes — explore

| Kernel | Why | Tile / notes | Existing page |
|---|---|---|---|
| **W8A8 INT8 dense + MoE** | Both sides i8. Prefill GEMM is compute-bound. | Prefill **64×64×64 i8**, WG=256, LDS 8 KB (recipe 9). BK%4==0. i32 through K, scale in epilogue. Decode `M=1/2/4` skinny, A in LDS. | [w8a8-mxfp4.md](w8a8-mxfp4.md) (spec). **Not in the fork.** |
| **Sage QK** (prefill) | Q and K both i8. QK is the FLOP side of prefill attn. | Pack D as 4×i8. PV stays `fdot2`. | [sage-qk.md](sage-qk.md) |
| **W4A8** (later) | W is i4, A is i8. Unpack nibble → i8, then `sdot4`. | Same 64×64×64 after unpack. `sdot8` is **W4A4 only** — both sides i4. Do not use `sdot8` for W4A8. | this page |
| **W4A4 integer** (later) | Both sides i4. `sdot8`. Prefill first. | After W8A8. Decode stays W4A16 unless measured. Not E2M1. | [w4a4.md](../engine/w4a4.md) |

## No — do not open an sdot4 variant

| Kernel | Why |
|---|---|
| W4A16 / W8A16 | A is fp16. Dequant W → `fdot2`. |
| W8A16-FP8 / W8A8-FP8 | Bytes are E4M3, not integer i8. Integer DOT on FP8 bits is wrong. Bit-trick / LUT → fp16 → `fdot2`. |
| mxfp4 / NVFP4 | E2M1 unpack → fp16 → `fdot2`. `sdot8` on E2M1 is wrong math. |
| INT8 KV in `fa_rdna2` | Storage only. Q stays fp16. Decode is KV bandwidth. Quantizing Q every step is Sage, and still will not beat gather. |
| HIP MLA decode | Gather + FP8 unpack is the sink. Scalar FMA → later `fdot2`, not `sdot4`. |
| Skinny GEMM | Occupancy trap. No MFMA on gfx1030. Not an INT8 kernel. |

## W4A8 sketch (explore only)

```
load 8×i4 as uint32
unpack nibbles → 8×i8 in VGPR          // sign-extend, no LUT
sdot4 against packed A i8              // i32 accum
epilogue: C *= a_scale * w_scale
```

Pack W K-contiguous, `K % 8 == 0` (two sdot4 per dword of W). Same LDS seed as W8A8 after the unpack. No `sdot8` unless someone later ships integer W4A4 and accepts the numerics — that is [w4a4.md](../engine/w4a4.md), not this sketch.

## Cards

Reuse the existing W8A8 INT8 and Sage cards (spec / Todo). Add **one** explore card: W4A8 `sdot4`. Add **one** explore card: W4A4 integer `sdot8` ([w4a4.md](../engine/w4a4.md)). Do not clone a card per “no” row. Do not use the W4A4 card for MXFP4/NVFP4 A4.

## Sources

- RDNA 2 ISA 70648: `V_DOT4_I32_I8`, `V_DOT8_I32_I4`
- GPUOpen WMMA table: IU8 512, IU4 1024, packed FP16 256
- LDS recipe 9: [silicon/lds-tiles.md](../silicon/lds-tiles.md)
