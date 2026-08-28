# EXL3 / QTIP on gfx1030 — silicon

Occupancy + live W4 still first. Kernel contract: [../kernels/exl3.md](../kernels/exl3.md). Engine/vLLM: [../engine/exl3.md](../engine/exl3.md).

Not EXL2. Not Marlin. Not a CUDA `exl3_gemv` / `mgemm` port.

## Live extras (2026-08-28)

Tip: `rdna2_extras` @ `e268c7d3`. Last HIP ISA lock: `a2c8d5cf` (own `exl3_dot2_dense` / `exl3_dot2_moe` / `exl3_hadamard`; CMake/bindings `d3fe4c98`). Not default AWQ/GPTQ. Do not copy tok/s.

ISA (unchanged by `e268c7d3`): 16×16 `exl3_window_*` → `decode_3inst<cb>` → `half2` → `__builtin_amdgcn_fdot2`. Occupancy lever is VGPR (`w0[4][16]` + `w1[4][16]`), not LDS. Dense A is `M_PER × (BLOCK_K + LDS_PAD)` half (`BLOCK_K=256`, `LDS_PAD=8`). No GEMM `__launch_bounds__` / `waves_per_eu`. HIP templates `cb==0` and `cb==1`; `cb==2` (`mul1`) is `TORCH_CHECK` false. Hadamard stays `__launch_bounds__(32)`; `suh`/`svh` outside the GEMM.

`e268c7d3` is compile-only: deleted unused `TILES_N` / `offset_m` / `c0` from `gemm_exl3_kernel_rdna`. Live tile index is `tile_idx0` / `tile_idx1` + `n_tiles_total`. Kernel behavior unchanged. gfx1030 `-Werror -Wunused-variable` now compiles (failed on `.176`). CMake gfx1030 EXL3 list unchanged.

Dispatch (Python `38bdfec5`, not ISA): unmarked 2/3/4-bit single-shard stays on the HIP kernel. `*.mul1` / `*.mcg` markers, fused suh-shards, and bits-6 lm_head fold to fp16 (`reconstruct_had_slice` / `VLLM_EXL3_FOLDED_CACHE`). HIP GEMM is not the mul1 path.

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


## RDNA2 flexibility (infer vs convert)

16×16 + `suh`/`svh` + packed trellis are **frozen at infer**. We do not re-walk.

Convert-time (producer): integer `K` per tensor (1–8), codebook id, who gets EXL3. RDNA2 pick: **experts only**, **3.0–3.5 bpw**, **`cb=0` (`3inst`)**. `-hq` on attn/shared is wrong when leftover is official MXFP8.

Infer-time (our HIP):

| Knob | Do |
|---|---|
| Codebook | Template `cb`. `0` and `1` (`mcg`) are the same VALU class (mul + LOP3-emulate + `hadd`). `2` (`mul1`) is the expensive one (byte-sum + `hfma`). Produce `3inst`; still compile `mcg` for 0xSero K216. |
| `K` | One `K` per launch (template like CUDA `bits`). Do **not** mix bpw in one WG. |
| Tile | Skinny `BLOCK_M=1/2/4/8`, A in LDS, stream packed B, `BLOCK_KN=256` seed. Pair two states → `half2` → one `fdot2`. |
| Hadamard | Required if `suh`/`svh` exist. 128-wide Walsh is cheap vs expert GDDR. Separate kernel is fine. |
| Occupancy | Same ticket as `mxfp4_dot2_moe`. No CUDA 16×16 MMA shapes. |

2 bpw is decode-bound on 512 GB/s (same 3-inst cost, half the bytes).

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
