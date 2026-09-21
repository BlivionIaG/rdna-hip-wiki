# EXL3 — silicon / HIP contract

Occupancy + live W4 still first. Silicon: [../silicon/exl3.md](../silicon/exl3.md).

Take the math only: unpack trellis → fp16 → `fdot2`. Same job as mxfp4, fatter decode. **No** Marlin / MMA / WMMA port.

## Live extras (2026-08-28)

In the fork. Tip `e268c7d3`. HIP ISA lock `a2c8d5cf`: own `exl3_dot2_*` files, not a fatter `mxfp4_dot2_moe` unpack. Inner loop is 16×16 window → `decode_3inst<cb>` → `half2` → `fdot2`. No GEMM `__launch_bounds__`. `e268c7d3` deleted unused `TILES_N` / `offset_m` / `c0` (gfx1030 `-Werror`); behavior unchanged.

HIP compiles `cb==0` (`3inst`) and `cb==1` (`mcg`). `cb==2` (`mul1`) does not launch. Live Python still only feeds unmarked 2/3/4-bit single-shard into the kernel; marked mul1/mcg layers fold fp16.

## Arch guard lock 2026-09-21 (dest tip `f3dd65fa` / `3d6df9ed`)

`__HIP__RDNA__` on `exl3_dot2_{dense,dequant,moe}.cu` matches the docker RDNA fatbin set: `gfx1030`/`1031`, `gfx1100`/`1101`, `gfx1150`/`1151`, `gfx1200`/`1201`. Compile-guard only — ISA and tile contract above unchanged. Not `gfx900`/`906`/`1013`. See [../silicon/exl3.md](../silicon/exl3.md).

## Pack

```
B: (k/16, n/16, 16*K) uint16     K = bpw ∈ {1..8}
suh: (k,) half                    input scales + Hadamard flips
svh: (n,) half                    output scales + Hadamard flips
cb:  0 | mcg=1 | mul1=2           one codebook, not both flags
```

`k % 16 == 0`, `n % 16 == 0`. Tile is **format-mandatory 16×16**, not a sweep. 4 bpw tile = 128 B packed → 512 B half after decode.

Writer/reader must agree on bit-extract order (`exl3_dq.cuh`: aligned 1/2/4-bit paths, `dq4`/`dq8` for the rest). Lock that when a real checkpoint is named.

## extras WIP (tip `a2c8d5cf`, 2026-08-28)

CMake + `torch_bindings` + `Exl3Config` wired (`d3fe4c98` / `a2c8d5cf`). Still not the default AWQ/GPTQ path. Occupancy hygiene missing on the GEMMs.

| File | Role |
|---|---|
| `csrc/rocm/exl3_dot2_common.cuh` | `decode_3inst<cb>`, real `exl3_window_*` (flat `dq8_flat` is unused scaffold) |
| `csrc/rocm/exl3_dot2_dense.cu` | `exl3_gemm_rdna2` — 256 thr, `BLOCK_N=1024`, `BLOCK_K=256`, `M_PER` 1/2/4/8 |
| `csrc/rocm/exl3_dot2_moe.cu` | `moe_exl3_gemm_rdna2` — 256 thr / 1024 N; `BLOCK_SIZE_M` 1/2/4/8; CAS epilogue |
| `csrc/rocm/exl3_hadamard.cu` | `exl3_hadamard_128` — `__launch_bounds__(32)`, not in the K-dot |

Inner: real 16×16 window → `decode_3inst<cb>` → `half2` → `V_DOT2_F32_F16`. Launch (pre-`8c23f0bd`): `bits` ∈ {2,3,4}, `cb` ∈ {0,1} (rejected `mul1`). At `8c23f0bd`: dense also launches `bits=6` and `cb=2`; 6bpw lm_head uses load-time dequant. Dense/MoE GEMM have **no** `__launch_bounds__` / `waves_per_eu` — do not ship on `(1,1)`. Dense LDS is A only (`M×264` half). MoE LDS is A `M×16` half + `s_tile[64][8*bits]` u32. Occupancy risk is VGPR (`w0`+`w1` 4×16 halves on dense), not LDS. `suh`/`svh` stay caller-side / Hadamard kernel.

Do not copy tok/s. Do not treat as Live until occupancy dump + named 3inst checkpoint.

## Inner loop (landed shape)

| Step | Op |
|---|---|
| Load W | `uint32` from packed tile |
| extract | K-bit / 16-bit state in VGPR |
| `decode_3inst<cb>` | mul + (LOP3-emulate or byte-sum) → `half` / `half2` |
| Hadamard | 128-wide Walsh on A (`suh`) and C (`svh`), `1/sqrt(128)` — not in the K DOT |
| A16 · W16 | `V_DOT2_F32_F16` |

Decode kernels (`cb==0/1`): integer mul + bit-force + `hadd` of the two forced halves. `cb==2`: mul + byte-sum (`sdot4` vs `0x01010101` + `0x6400`) + `hfma`. That `sdot4` is **not** the GEMM.

Tile seed after unpack: same as W4A16 (`64×64×32` fp16) for prefill; decode skinny like `q_gemm_rdna2`. Do not copy CUDA `16×16×128` MMA shapes.

## vs other unpack-then-DOT

| Format | Unpack | Then |
|---|---|---|
| W4A16 | nibble → scale | `fdot2` |
| mxfp4 | E2M1 + E8M0 | `fdot2` |
| INT2 | 16×i2 | `fdot2` |
| **EXL3** | trellis state → 3-inst codebook | `fdot2` |
| ROCmFPX | codebook10 → `perm` → i8 | `sdot4` (llama.cpp only) |

EXL3 is closest to mxfp4 (weird storage → half → DOT2). It is **not** ROCmFPX.

## Do not

- Invent a gfx1030 MMA fragment layout.
- Mix `cb` in one launch.
- `sdot4`/`sdot8` the reconstructed halves.
- Cooperative-grid Hadamard as a ship gate.
- Land on `waves_per_eu(1,1)`.
- vLLM GGUF / old `exllama.py` uint4 path — those are not this format.

## Done-when (if ever)

- ISA dump: integer decode + `v_dot2_f32_f16`. No `wmma` / `mfma` / mystery DOT.
- One named checkpoint (`mcg` xor `mul1` xor default) bit-matches `decode_3inst`.
- `suh`/`svh` 128-Hadamard applied, scale locked.
- Smoke vs fp16 / W4A16 on the same model. No tok/s from this page.

## extras tip `8c23f0bd` (2026-09-01)

HIP: 6bpw `mul1` load-time dequant + dense launch of `bits∈{2,3,4,6}`, `cb∈{0,1,2}`.

| File | Role |
|---|---|
| `csrc/rocm/exl3_dot2_dequant.cu` | `exl3_dequant_bits6_mul1` — `16×16` block, no LDS, no launch_bounds; one-shot at weight load for 6bpw lm_head |
| `exl3_dot2_common.cuh` | `bits==6` → `dq4<6>` window extract |
| `exl3_dot2_dense.cu` | launches `bits=6` and `cb=2`; sizes from tensors |

Inner GEMM still 16×16 → `decode_3inst` → `half2` → `fdot2`. Dequant is not a K-dot path (no Hadamard in kernel). Occupancy hygiene still missing on GEMMs. Produce experts as `3inst`; do not treat `mul1` launch as produce flip. FA pin closed. No tok/s.


## extras tip `aac1fcd6` (2026-09-04)

Prefill unpack-once: `exl3_decode_trellis_rdna2` — 16×16 block, 256 thr, `decode_3inst` only, no LDS / no fdot2 / no launch_bounds. bits 2/3/4; bits=6 stays mul1 dequant. Dot work after unpack is rocBLAS, not a new HIP DOT tile. Fused decode GEMM path unchanged. Produce `3inst`. No tok/s.

## Sources

- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) `codebook.cuh`, `exl3_dq.cuh` (2026-08-21)
- [kernels/mxfp4.md](mxfp4.md), [kernels/w4a16.md](w4a16.md)
