# EXL3 — silicon / HIP contract

Later. Not in the fork. Occupancy + live W4 still first. Silicon: [../silicon/exl3.md](../silicon/exl3.md).

Take the math only: unpack trellis → fp16 → `fdot2`. Same job as mxfp4, fatter decode. **No** Marlin / MMA / WMMA port.

## Pack

```
B: (k/16, n/16, 16*K) uint16     K = bpw ∈ {1..8}
suh: (k,) half                    input scales + Hadamard flips
svh: (n,) half                    output scales + Hadamard flips
cb:  0 | mcg=1 | mul1=2           one codebook, not both flags
```

`k % 16 == 0`, `n % 16 == 0`. Tile is **format-mandatory 16×16**, not a sweep. 4 bpw tile = 128 B packed → 512 B half after decode.

Writer/reader must agree on bit-extract order (`exl3_dq.cuh`: aligned 1/2/4-bit paths, `dq4`/`dq8` for the rest). Lock that when a real checkpoint is named.

## Inner loop (when someone writes it)

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

## Sources

- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) `codebook.cuh`, `exl3_dq.cuh` (2026-08-21)
- [kernels/mxfp4.md](mxfp4.md), [kernels/w4a16.md](w4a16.md)
