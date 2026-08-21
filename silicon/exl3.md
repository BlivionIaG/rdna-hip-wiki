# EXL3 / QTIP on gfx1030 — silicon

Later. Occupancy + live W4 still first. Kernel contract: [../kernels/exl3.md](../kernels/exl3.md). Engine/vLLM: specialist (no official loader; `#19896` stale-closed).

Not EXL2. Not Marlin. Not a CUDA `exl3_gemv` / `mgemm` port.

## What inference actually does

Viterbi is **quantize-time only**. The checkpoint already is the walk.

At infer, each 16×16 tile is packed indices, `K` = bpw (1–8). `B` is `(k/16, n/16, 16*K)` `uint16`. Per weight:

1. Bit-extract a 16-bit state from the packed stream (bitshift trellis — parallel, no sequential walk).
2. `decode_3inst<cb>(state)` → one `half` (procedural codebook, no table).
3. GEMM that `half` against A16.

CUDA then feeds `FragB` (`half2`×2) into `mma.m16n8k16`. **That MMA is dead on V620.** After step 2 we already have fp16 weights. Inner op is `V_DOT2_F32_F16`, same as mxfp4 / W4A16 after unpack.

## Codebooks (`cb`)

Exactly one of: default / `mcg` / `mul1`. Not both flags.

| `cb` | Flag | First ops | Then |
|---|---|---|---|
| 0 | (none) | `x *= 89226354; x += 64248484` | LOP3 `0x8fff8fff`/`0x3b603b60`/`0x6a` → two packed fp16 → add |
| 1 | `mcg` | `x *= 0xCBAC1FED` | same LOP3 + add |
| 2 | `mul1` | `x *= 0x83DCD12D` | byte-sum (`dp4a(x, 0x01010101, 0x6400)`) → `hfma` with `k_inv=0x1eee`, `k_bias=0xc931` |

gfx1030:

- No `lop3.b32`. Emulate the bit-force with `V_AND_B32` / `V_OR_B32` / `V_XOR_B32` (or `V_AND_OR_B32`). Do not invent a LUT.
- `cb==2` byte-sum is **codebook only**. `__dp4a` there is `sdot4` vs `+1` bytes, not W8A8 GEMM. Reconstructed values are irregular Gaussian-like halves — **not** an INT4/INT8 grid. No `sdot4`/`sdot8` in the K loop.

## Hadamard (`suh` / `svh`)

QuIP# incoherence, not INT2-KV Hadamard. 128-wide Walsh on A (pre, `suh` along K) and C (post, `svh` along N), scale `1/sqrt(128) ≈ 0.088388347648`. CUDA fuses it with `grid.sync()` around the MMA kernel.

HIP: cheap VALU butterfly, separate kernel or tile-edge fuse. Do **not** require `cudaLaunchCooperativeKernel` / HIP cooperative launch to ship a first GEMV.

## Why the CUDA path is the wrong shape

- `exl3_gemm` tiles are MMA fragments: `16×{16,32}×{128,256,512}`, 4–6 smem stages, 3–5 frag stages, 256–512 threads, cooperative grid.
- `exl3_gemv` is the skinny `m≤8` path — still `mma.m16n8k16`, occupancy-tuned on Ampere, 2–4 bpw only.
- 2 bpw is decode-bound on NVIDIA (same 3-inst cost, half the bytes). gfx1030 VALU decode is not cheaper. If we ever ship, start at **3–4 bpw**.

Our first kernel, if any: skinny like `q_gemm_rdna2` — dword load, extract, `decode_3inst` in VGPR, `fdot2`, no `FragB`, no `waves_per_eu(1,1)`.

## Do not

- Port Marlin / CUTLASS / WMMA / `ldmatrix`.
- Treat EXL3 as W4A16 nibble dequant (indices are trellis states, not i4 codes).
- Fire `sdot*` on the reconstructed halves.
- Land this before occupancy / live W4.
- Quote NVIDIA tok/s.

## Sources

- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) `codebook.cuh`, `exl3_dq.cuh`, `exl3_gemm_inner.cuh`, `exl3_gemv.cu`, `doc/exl3.md` (read 2026-08-21)
- QTIP: arXiv 2406.11235 (bitshift trellis, ≤4 inst/weight codes)
- QuIP#: arXiv 2402.04396 (Hadamard incoherence)
