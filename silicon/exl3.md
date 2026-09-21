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

## extras lock 2026-08-28 (tip `a2c8d5cf`)

HIP matches the contract: real 16×16 tile (`exl3_window_pos` / `exl3_window_at`) → `decode_3inst<cb>` → `half2` → `__builtin_amdgcn_fdot2`. Not a fatter unpack on `mxfp4_dot2_moe`. `dq8_flat` in the header is leftover scaffold — dense/MoE fire the real tile path. No `sdot*` on halves.

| | Dense `exl3_gemm_rdna2` | MoE `moe_exl3_gemm_rdna2` |
|---|---|---|
| Geometry | `THREADS_X=256`, `BLOCK_N=1024`, `BLOCK_K=256` (W4 skinny) | same 256 thr / 1024 N; loops all K-tiles |
| LDS | `half s_a[M_PER][256+8]` (~0.5–4 KiB) | `s_a[M][16]` + `s_tile[64][8*bits]` u32 (~6–8 KiB at 3–4 bpw) |
| Occupancy | **no** `__launch_bounds__` / `waves_per_eu` | same |
| VGPR lever | `w0[4][16]+w1[4][16]` halves (second tile always allocated) | `w[4][16]` reused across `BLOCK_SIZE_M` rows |
| Epilogue | split-K Y: 64-bit CAS `atomic_add_pk4_f16` if `gridDim.y>1` | CAS if `output_topk>0` |

Launch: `bits` ∈ {2,3,4}, `cb` ∈ {0,1}. `mul1` compiled in `decode_3inst`, not launched. `bits==7` window still approximate (produce is 3.0). Hadamard is `__launch_bounds__(32)` `H_128` via `__shfl_xor_sync`; `suh`/`svh` stay outside the K-dot (`exl3_hadamard_128`). MoE comment `TILES_PER_BLOCK // 16` is wrong; math is `1024/16=64` and the array matches. A-stage writes `t%16` from 256 threads (16× overwrite, not a correctness bug).

CMake lists `exl3_dot2_{dense,moe}.cu` + `exl3_hadamard.cu` unconditionally (RDNA-generic). `torch_bindings` + `Exl3Config` registered; still not default AWQ/GPTQ. Last Live HIP lock remains `f263172a` W4A16 split-K. FA occupancy pin stays closed. Do not invent numbers. Do not copy tok/s.

## extras lock 2026-09-01 (tip `8c23f0bd`)

After `02357ecd` (Python glue) and `25e8788` (hybrid pages): HIP delta is EXL3 6bpw + `mul1` launch paths. Not FA/W4. FA pin stays closed. Occupancy leftover still FA prefill `(N,1)` / EXL3 GEMM VGPR (still no `__launch_bounds__` / `waves_per_eu` on dense/MoE).

| File | Delta |
|---|---|
| `csrc/rocm/exl3_dot2_dequant.cu` | **new** load-time `exl3_dequant_bits6_mul1`: tile `16×16` thr, **no LDS**, no `__launch_bounds__`; `dq4<6>` window → `decode_3inst<2>` (`mul1`) → fp16 out. Caller applies `suh`/`svh` in PyTorch. `__HIP__RDNA__` arch set — see 2026-09-21 lock. |
| `csrc/rocm/exl3_dot2_common.cuh` | `exl3_window_at` special-cases `bits==6` via `dq4<6>` (generic pair path wrong on odd indices in a dq4 batch). Generic leftover is `{3,5,7,8}`. |
| `csrc/rocm/exl3_dot2_dense.cu` | Launch now accepts `bits==6` and `cb==2` (`mul1`). `size_m/n/k` derived from tensor shapes (dynamo ABI), not Python ints. |
| `CMakeLists.txt` / `ops.h` / `torch_bindings.cpp` | Adds `exl3_dot2_dequant.cu` + `exl3_dequant_bits6_mul1` binding; GEMM signature drops size ints. |

Do **not** read this as flipping produce policy: expert produce stays `3inst` (`cb==0`); `mcg` still compile-for-K216; `mul1` here is the **6bpw lm_head** one-shot dequant (was `NotImplementedError` at `02357ecd`), not a default expert codebook. No DOT change in the K-loop (still `fdot2` after decode). No KV quant / FA / GDN HIP in this tip. Do not invent numbers. Do not copy tok/s.

## extras lock 2026-09-03 (tip `f9361950`)

`faf87f80` marks `exl3_hadamard_128`'s `output` as **mutating** in the rocm op schema (`Tensor` → `Tensor!`, `csrc/rocm/torch_bindings.cpp`, +1/-1). Impl and the `.cu` are untouched; no DOT, tile, or `__launch_bounds__` change. It was the last stale schema on the rocm bindings — every other out arg already had `Tensor!`.

This one matters here because `suh`/`svh` are a **separate kernel outside the K-dot**: the Hadamard's only product is its out tensor, so a non-mutating schema is the exact shape functionalization can treat as dead. Full rule + the capture-path half of the same lock: [graph-capture.md](graph-capture.md).

Produce policy unchanged: expert produce stays `3inst` (`cb==0`), `mcg` compile-only, `mul1` still the 6bpw lm_head one-shot. Occupancy leftover stays FA prefill `(N,1)`. Do not copy tok/s.



## extras lock 2026-09-21 (tip `f3dd65fa` / `3d6df9ed`)

HIP-only delta since tip `4425834a`: `__HIP__RDNA__` preprocess guard on `exl3_dot2_{dense,dequant,moe}.cu` widened so device-compile for the docker multi-arch list still sees the kernel macros.

Was: `gfx1030` | `gfx1100` only.
Now: `gfx1030` | `gfx1031` | `gfx1100` | `gfx1101` | `gfx1150` | `gfx1151` | `gfx1200` | `gfx1201`.

Why: docker-bake `PYTORCH_ROCM_ARCH` builds those RDNA consumer arches; non-matching device passes dropped `V2_*` / kernel bodies and failed the fatbin (unknown type name). Intentional: these kernels are wave32 + `V_DOT2` generic, not generation-gated like `q_gemm_rdna2` / some mxfp4 paths.

Unchanged: tile layout, `decode_3inst` → `half2` → `fdot2`, LDS shapes, no GEMM `__launch_bounds__` / `waves_per_eu`, produce policy (`3inst` experts; `mul1` = 6bpw lm_head dequant). Does **not** add `gfx900` / `gfx906` / `gfx1013` (no DOT fatbin onto those). Occupancy leftover still FA prefill `(N,1)` / EXL3 VGPR. Do not invent numbers. Do not copy tok/s.

Other tip commits (`f3dd65fa` PR #15 QSA live-prefill bound, PLE/MTP CPU restore, amdsmi fallback) are Python/docs/serve — not HIP/ISA.

## extras lock 2026-09-04 (tip `aac1fcd6`)

Range `f9361950` → `aac1fcd6` (+5). Silicon Take on EXL3: `62694f20` adds **unpack-once prefill** via `exl3_decode_trellis_rdna2` in `csrc/rocm/exl3_dot2_dense.cu` (+ `ops.h` / `torch_bindings.cpp`). Not a new DOT tile.

| | Shape |
|---|---|
| Kernel | `decode_trellis_kernel_rdna<bits,cb>` |
| Grid | `(K/16, N/16)` blocks; **256 thr** (one out elem / thread) |
| Tile | same 16×16 packed trellis window as fused GEMM |
| Inner | `exl3_window_pos` / `exl3_window_at` → **`decode_3inst<cb>` only** → fp16 out |
| LDS / DOT / bounds | **no LDS**, **no `fdot2`**, **no `__launch_bounds__` / `waves_per_eu`** |
| bits / cb | bits ∈ {2,3,4}; cb ∈ {0,1,2} launched. **bits=6 stays** `exl3_dequant_bits6_mul1` (not this op) |
| Binding | `exl3_decode_trellis_rdna2(Tensor trellis, Tensor! out, int bits, int cb)` — mutating out |

Why: fused dense GEMM re-decodes each tile per M-block (`M_PER=8` cap). Prefill pulls decode out once, then **rocBLAS** on the fp16 matrix (Python in `exl3.py`). Produce stays `-cb 3inst`. Occupancy leftover still FA prefill `(N,1)`. Do **not** copy any prefill speedup claim from the commit message. Do not invent numbers.

Non-silicon in the same tip (for dest context only): `38595867` sliding_window arg on `fa_rdna2` splitk wrapper (Python); `9ba5d5f4` drops lm_head debug print. GDN HIP delta is on [kernels/gdn-prefill.md](../kernels/gdn-prefill.md).

## Sources

- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) `codebook.cuh`, `exl3_dq.cuh`, `exl3_gemm_inner.cuh`, `exl3_gemv.cu`, `doc/exl3.md` (read 2026-08-21)
- QTIP: arXiv 2406.11235 (bitshift trellis, ≤4 inst/weight codes)
- QuIP#: arXiv 2402.04396 (Hadamard incoherence)
