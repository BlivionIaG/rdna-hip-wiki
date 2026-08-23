# gfx1030 FA occupancy report (BlivionIaG/vllm)

Live human branch: **`rdna2_extras`** @ **`6e20b239`** (GDN prefill HIP + multi-chunk fixes). Occupancy leftover unchanged: FA prefill still `(N, 1)`. Decode pin is off: FA 128/256 `__launch_bounds__(N)` + `amdgpu_waves_per_eu(4, 8)`; skinny dropped `(1, 1)`. Prefill FA still `__launch_bounds__(N, 1)`. Merge +1% on Qwen3.8-27B-AWQ 16k/1k TP=4 is **noise** — not a measured occupancy win. GPU occupancy query still TBD.

This dump below is a **historical snapshot** of `perf/rdna2_w4a16` (tree SHA `9ac015d0a936e9e3bdbe5dc7483e1a8b48c65370`). Decode rows in the table are stale vs `d414eac5`.

Live tree: [`csrc/rocm/fa_rdna2.cu` on `rdna2_extras`](https://raw.githubusercontent.com/BlivionIaG/vllm/rdna2_extras/csrc/rocm/fa_rdna2.cu).
Snapshot sources: [`fa_rdna2.cu`](https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/fa_rdna2.cu), [`sparse_mla_rdna2.cu`](https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/sparse_mla_rdna2.cu), [`indexer_paged_mqa_rdna2.cu`](https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/indexer_paged_mqa_rdna2.cu), [`ops.h`](https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/ops.h).

Nothing here is invented from a compiled `.s` / `-Rpass-analysis=kernel-resource-usage` dump. VGPR columns are source comments or live-state estimates. LDS bytes are the host `size_t smem` formulas (what HIP actually reserves).

Hardware model used for occupancy (as requested): **16 waves/SIMD32**, **1024 VGPR/SIMD**, **granule 16**, **LDS 64 KB/WG**. RDNA2 default HIP mode is WGP (4 SIMD32, 128 KB LDS address space) but a single WG still cannot allocate more than 64 KB (ISA §3.6.6 / comments in `fa_rdna2.cu`). Authors treat gfx1030 as “64 KB per-CU, 1 block/CU”.

---

## d414eac5 decode pin (2026-08-23)

| Kernel | Now | Still leftover |
|---|---|---|
| `fa_decode_paged_splitk_kernel` | `__launch_bounds__(128)` + `waves_per_eu(4, 8)` | VGPR / hipOccupancy not dumped |
| `fa_decode_paged_splitk_kernel_256` | `__launch_bounds__(256)` + **same** `(4, 8)` | Wiki CUDA-port for 256-thr is min **8** (`(256,8)`). `(4,8)` is looser min, max 8 = 1 WG/SIMD |
| `wvSplitKrc_` | dropped `waves_per_eu(1,1)` | no max pin |
| prefill `fa_prefill_*` 128/256 / splitk / int8 | `__launch_bounds__(N, 1)` | **still the trap** |

`(4, 8)` is a compiler occupancy *range*, not a runtime query. If decode already sat ≤128 VGPR, +1% is expected. Do not close the occupancy card. Fat-M W4 / ConfigA unchanged.

`gdn_decode_rdna2` + five `gdn_prefill_*` (`b53a7a2`): all `(2, 4)`, not `(1,1)`. kkt stays scalar FMA. wy ~58 KB. Not the FA occupancy leftover. See [../kernels/gdn-prefill.md](../kernels/gdn-prefill.md).

## 1. `__launch_bounds__` → `amdgpu_waves_per_eu`


HIP does **not** treat the second argument as CUDA `minBlocksPerMultiprocessor`.

HIP docs (`__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)`) and the `amd_hip_runtime.h` macro (ROCm/hip#2521) lower:

```c
__attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),
               amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
```

The second argument is passed **verbatim** as `amdgpu_waves_per_eu(N)` = **minimum waves per SIMD32**. Default is already 1.

| Written in source | Lowers to | Meaning on gfx1030 wave32 | If they meant CUDA min-blocks=1 they should have written |
|---|---|---|---|
| `__launch_bounds__(128, 1)` | `flat_work_group_size(1,128)` + `amdgpu_waves_per_eu(1)` | max WG 128; **min 1 wave/SIMD32** (default). VGPR cap ≈ 1024/1 = 256 (file max). | `__launch_bounds__(128, 4)` because `(1 * 128) / 32 = 4` |
| `__launch_bounds__(256, 1)` | `flat_work_group_size(1,256)` + `amdgpu_waves_per_eu(1)` | max WG 256; **min 1 wave/SIMD32**. Same default VGPR cap (256). Does **not** mean “1 block/CU”. | `__launch_bounds__(256, 8)` because `(1 * 256) / 32 = 8` |
| `__launch_bounds__(THREADS)` with `THREADS=32` | `flat_work_group_size(1,32)` only (no 2nd arg) | max WG 32; waves/EU defaults to 1 | n/a |
| no attribute | compiler assumes device-max block (1024) | worse VGPR budget than pinning the real WG size | pin the real size |

**`(256, 1)` on HIP = 1 wave per SIMD32.** That is a *minimum-occupancy* hint, and 1 is the default, so it does **not** force 1-wave occupancy and does **not** encode the “1 block/CU” the LDS comments want. It only pins `maxThreadsPerBlock=256`. The compiler is free to spend up to 256 VGPR/lane. A CUDA-style “please keep enough registers for 1 block of 256” would have been `(256, 8)` = 8 waves/EU = VGPR ≤ 128.

Conversion formula from HIP docs (porting CUDA):

```
MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK) / warpSize
```

---

## 2. Short table

LDS-limited WGs use the 64 KB/WG cap (authors + HIP `MaxSharedMemoryPerBlock`). After the 1 KB ISA granule the rounded allocation is still ≤ 64 KB for every kernel, so **no LDS occupancy-0**. Closest is prefill-256 at 59620 B (5.9 KB headroom; 60416 B after 1 KB round).

| Kernel | `__launch_bounds__` | `amdgpu_waves_per_eu` | WG | waves/WG | LDS B (KiB) | VGPR | Br / Bc / D | split-K | LDS WGs / 64KB | occ-0? |
|---|---|---|---|---|---|---|---|---|---|---|
| `fa_decode_paged_splitk_kernel` | `(128, 1)` | **1** | 128 | 4 | 33300 (32.52) | not in source | 1 / 64 / 128 | yes (grid.z, max 16) | 1 | no (LDS) |
| `fa_decode_paged_splitk_kernel_256` | **`(256, 1)`** | **1** | 256 | 8 | 33444 (32.66) | not in source | 1 / 32 / 256 | yes | 1 | **VGPR risk** if compiled VGPR is huge; LDS OK |
| `fa_decode_combine_kernel` | none | default 1 | D (128 or 256) | 4 or 8 | `(3*splits+1)*4` = 196 @ splits=16 | not in source | n/a | reduce | many | no |
| `fa_prefill_paged_varlen_kernel_128` | `(128, 1)` | **1** | 128 | 4 | 49364 (48.21) | not in source | 16 / 64 / 128 | no | 1 | no (LDS) |
| `fa_prefill_paged_varlen_kernel_128_short` | **`(256, 1)`** | **1** | 256 | 8 | 45476 (44.41) | not in source | 32 / 32 / 128 | no | 1 | **VGPR risk** (same as other 256,1) |
| `fa_prefill_paged_varlen_kernel_256` | **`(256, 1)`** | **1** | 256 | 8 | **59620 (58.22)** | not in source | 16 / 32 / 256 | no | 1 | **LDS-tight** (5.9 KB left). Not 0. |
| `fa_prefill_paged_varlen_splitk_kernel_128` | `(128, 1)` | **1** | 128 | 4 | 49364 (48.21) | not in source | 16 / 64 / 128 | yes (z = seq*splits) | 1 | no (LDS) |
| `fa_prefill_paged_varlen_splitk_kernel_256` | **`(256, 1)`** | **1** | 256 | 8 | **59620 (58.22)** | not in source | 16 / 32 / 256 | yes | 1 | **LDS-tight** |
| `fa_prefill_*_splitk_reduce_kernel_{128,256}` | none | default 1 | D | 4 or 8 | 0 | not in source | n/a | reduce | n/a | no |
| `sparse_mla_decode_kernel` | `(32)` | default 1 | 32 | 1 | 1168 (1.14) static | comment: **64+64 fp32/thread** (~128–160+) | H=64, D=512 (448+64), 4 heads/CTA | no (16 CTAs/query) | 56 | no. V1 spilled ~1028 fp32/thread; V2 is the live kernel |
| `paged_mqa_logits_decode_kernel<128,64,64,128>` | none | default 1 | 64 | 2 | 49152 (48.00) | not in source | H=64, D=128, Bc=128 | no | 1 | no (LDS) |

**Sage INT8 QK:** **not present** in this tree. No `sage*`, no `qk_int8`, no `sdot4` / `__builtin_amdgcn_sdot4` in the attention kernels. Planned only (research note). Live QK opcode is **`fdot2`** — see §4.

---

## 3. Per-kernel detail

### 3.1 Decode D=128 — `fa_decode_paged_splitk_kernel`

```c
__global__ __launch_bounds__(128, 1) void fa_decode_paged_splitk_kernel(...)
```

- Host launch: `dim3 block1(128)`, grid `(num_tokens, H_q, kv_splits)`, `kv_splits ∈ [1,16]`.
- LDS (host formula):

```
HEAD_DIM*2 + BC*HEAD_DIM*2*2 + BC*4 + (THREADS/32+1)*4
= 256 + 16384 + 16384 + 256 + 20
= 33300 B
```

  Layout: `sQ[D] half`, `sK[Bc,D] half`, `sV[Bc,D] half`, `sP[Bc] float`, `sRed[5] float`.
- Live state/thread: `m_i`, `l_i`, `o_acc` (3 fp32) + QK `fdot2` temps. No VGPR attribute.
- QK: `acc = fdot2(q2, k2, acc)` over `d += 2`. PV is scalar `sP[k] * half2float(sV[k,t])`.
- No XOR swizzle (linear `sK[i]`).
- Occupancy: 4 waves/WG. LDS → 1 WG in 64 KB (2 WG/WGP if the second CU’s 64 KB is used). Waves/SIMD ≈ 1 (CU) or 1–2 (WGP with 2 WGs). Far below 16.

### 3.2 Decode D=256 — `fa_decode_paged_splitk_kernel_256`

```c
__global__ __launch_bounds__(256, 1) void fa_decode_paged_splitk_kernel_256(...)
```

- **This is the `(256, 1)` the prompt called out.** HIP meaning: **1 wave/SIMD32 min**, not 1 block/CU. 256 threads = 8 waves. Correct CUDA-style 1-block hint would be `(256, 8)`.
- Host: `THREADS=256`, `BC_256=32`.
- LDS: `512 + 32*256*2*2 + 32*4 + 9*4 = 33444 B`.
- Same algorithm, `fdot2` over 256, no swizzle.
- Occupancy: 8 waves/WG → 2 waves/SIMD in WGP mode for 1 WG. LDS still 1 WG/64 KB. `(256, 1)` lets the compiler use up to 256 VGPR; if it actually does, occupancy stays 1 WG (2 waves/SIMD) — low, not zero. Occupancy-0 would require the compiler to miss the WG (VGPR > 256, or a CU-mode calculator that tries to put all 8 waves on one SIMD).

### 3.3 Decode combine — `fa_decode_combine_kernel`

- No `__launch_bounds__`. Launch `block2(D)` = 128 or 256.
- LDS: `(3 * kv_splits + 1) * 4` → 196 B at `kv_splits=16`.
- Softmax merge of split partials. Not occupancy-interesting.

### 3.4 Prefill D=128 — `fa_prefill_paged_varlen_kernel_128`

```c
__global__ __launch_bounds__(128, 1) void fa_prefill_paged_varlen_kernel_128(...)
```

- Host LDS:

```
BR*D*2 + BC*D*2*2 + BC*BR*4 + BR*4*3 + BR*D*4 + (T/32+1)*4
= 4096 + 32768 + 4096 + 192 + 8192 + 20
= 49364 B
```

  Kernel pointer chain is `sQ, sK, sV, sP, sM, sL, sO` = **49280 B** live. Host adds one extra `BR` floats + `sRed` (84 B). Comment in file: “≈ 48 KB (fits gfx1030 64 KB per-CU limit with 1 block/CU)”.
- XOR swizzle on sK/sV (see §5).
- QK: `fdot2`. PV: scalar `fmaf` at swizzled V.
- Occupancy: LDS → 1 WG/64 KB. 4 waves/WG.

### 3.5 Prefill D=128 short (q_len < 4096) — `fa_prefill_paged_varlen_kernel_128_short`

```c
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_kernel_128_short(...)
```

- Local constexpr: `BR=32`, `BC=32`, `THREADS=256`, `D=128`.
- LDS: `8192 + 16384 + 4096 + 384 + 16384 + 36 = 45476 B`.
- Another **`(256, 1)`** → 1 wave/SIMD32. Comment claims “~48 KB”; measured 44.41 KiB.
- Swizzle on (vectorized half2) K/V. Softmax uses 32-lane `__shfl_xor` butterfly per row.
- Occupancy: 8 waves/WG, 1 WG/64 KB.

### 3.6 Prefill D=256 — `fa_prefill_paged_varlen_kernel_256`

```c
__global__ __launch_bounds__(256, 1) void fa_prefill_paged_varlen_kernel_256(...)
```

- `BC_256=32` “to stay under gfx1030's 64KB shared memory limit”.
- LDS: `8192 + 32768 + 2048 + 192 + 16384 + 36 = 59620 B` (**58.22 KiB**, 5916 B free, 60416 B after 1 KB granule).
- **No swizzle** (linear `sK[i]`). `fdot2` QK, scalar PV.
- Occupancy-0 from LDS: **no**, but this is the tightest tile. One extra `BR*D` fp32 array would die. 2 WGs/WGP still fit in 128 KB (118 KB).

### 3.7 Prefill split-K 128 / 256

Same `__launch_bounds__` and same host LDS formulas as the non-split twins.

- Grid.z = `num_seqs * kv_splits`. Each CTA owns a KV slice; writes unnormalized `(O,M,L)`.
- **Split-K 128 does not call `fa_swz_d`** (linear stores), unlike the non-split 128 kernel. Inconsistency in the same file.
- Reduce kernels: no bounds, `block=HEAD_DIM`, `smem=0`.

### 3.8 Sparse MLA decode (DeepSeek V4)

```c
__global__ void __launch_bounds__(THREADS) sparse_mla_decode_kernel(...)  // THREADS=32
```

- Grid `(num_queries, 16)` — 16 CTAs/query, 4 heads/CTA, 1 wave32.
- Static LDS, 16-byte aligned, double-buffered:

```
s_fp8[2][448]      = 896 B   // FP8 e4m3 OCP K_nope
s_k_rope[2][64]    = 256 B   // bf16
s_scales_raw[2][8] =  16 B   // E8M0
total                1168 B
```

- VGPR (quoted): “Q slices and accumulators are register-resident **(64 + 64 floats/thread)**”. Plus `k_nope[14]`, `k_rope[2]`, `m_i[4]`, `l_i[4]` → ~152 fp32 before addresses. Granule 16 → **~160 VGPR**. Waves/SIMD from VGPR: `1024/160 = 6`. One-wave WG, so ~6 WGs/SIMD, LDS is not the limiter.
- V1 (replaced) “~1028 floats/thread … spilled to local memory” — that *was* an occupancy disaster; not the live kernel.
- `ops.h` still says “2 heads per thread”; the `.cu` header is V2 (4 heads/CTA). Believe the `.cu`.

### 3.9 Paged MQA logits (Lightning Indexer fallback)

- No `__launch_bounds__`. Instantiation `HEAD_DIM=128, N_HEADS=64, BLOCK_THREADS=64, BLOCK_K=128`.
- LDS: `sizeof(uint16_t)*64*128 + sizeof(float)*64*128 = 16384 + 32768 = 49152 B` exactly 48 KiB.
- Occupancy: 2 waves/WG, 1 WG/64 KB.

### 3.10 Not in this tree

- HEAD_DIM=72 vision prefill: **comment only** (“NO swizzle: 72 isn't a power of 2”). No `__global__` body, not launched.
- HEAD_DIM=64 FA tile: **missing** (research note: Triton hole).
- Sage INT8 QK kernel: **missing**.

---

## 4. Sage INT8 QK — opcode

**There is no Sage INT8 QK kernel** on this snapshot / on `rdna2_extras`. `ops.h` registers only FA2 fp16, sparse MLA, and paged MQA logits. Repo-wide search of the downloaded ROCm sources found no `sage`, `qk_int8`, `sdot4`, `__builtin_amdgcn_sdot4`, or `v_dot4c`.

What the live attention paths actually issue:

**FA2 QK (every `fa_*` kernel that scores):**

```c
__device__ __forceinline__ float fdot2(half2 q, half2 k, float acc) {
  return __builtin_amdgcn_fdot2(q, k, acc, false);
}
// ...
acc = fdot2(q2, k2, acc);   // V_DOT2_F32_F16, 2×fp16 FMA, fp32 accum
```

File header: “V_DOT2_F32_F16 intrinsic: 2 fp16 multiply-adds per instruction.”

**Sparse MLA QK:** FP8 e4m3 → half via `__hip_cvt_fp8_to_halfraw(..., __HIP_E4M3)`, scale by `exp2f(E8M0-127)`, then **scalar `float` mul-add**. No `sdot4`, no `fdot2`.

```c
__half_raw r = __hip_cvt_fp8_to_halfraw(s_fp8[buf][ch], __HIP_E4M3);
k_nope[j] = __half2float(r) * sc;
// ...
partial += q_nope[hh][j] * k_nope[j];
```

**Paged MQA logits QK:** FP8 → fp16 bits, then **scalar float** after `__half2float`:

```c
dot += __half2float(q_half) * __half2float(k_half);
```

**W8A16-FP8 GEMM** (not attention; `ops.h` comment only): “Per-tile FP8 (E4M3) -> fp16 dequant via 256-entry LUT, then `v_dot2_f32_f16`.” Still fdot2 after dequant, not sdot4.

Research note `10-fa-dispatch-sage-head64.md` says Sage INT8 QK via **sdot4** is a *future prefill* idea (IU8 pipe, 512 ops/clock/CU). It is not implemented. If/when it is, the intrinsic to look for is `__builtin_amdgcn_sdot4` → `V_DOT4_I32_I8`. Today QK is **fdot2 after fp16**, or **float FMA after FP8 dequant**.

---

## 5. Bank-conflict / padding

Visible in `fa_rdna2.cu`:

```c
// XOR swizzle for sK/sV shared-memory indexing to reduce LDS bank conflicts.
// Pattern: d_swizzled = d ^ ((k & 7) << 4). For HEAD_DIM=128 / RDNA2 64-bank
// LDS, this shifts access patterns so 32-thread lanes reading different rows
// (k) but the same column (d) hit distinct banks rather than colliding on one.
// Smem cost: zero (same storage layout, remapped indexing on read & write).
__device__ __forceinline__ int fa_swz_d(int d, int k) {
  return d ^ ((k & 7) << 4);
}
```

| Kernel | Swizzle? |
|---|---|
| decode 128 / decode 256 | **no** — linear `sK[i]` |
| prefill 128 (non-split) | **yes** — store/load `fa_swz_d(d, k)` |
| prefill 128 short | **yes** — half2 store at swizzled even-d |
| prefill 256 (non-split and split-K) | **no** |
| prefill 128 split-K | **no** (same tile as swizzled non-split; regression / incomplete port) |
| D=72 comment | “NO swizzle: 72 isn't a power of 2 so XOR swizzle bits would alias” |
| sparse MLA | 16-byte `__align__(16)` on the three static arrays; no bank pad. Double-buffer only. |
| paged MQA | no pad / no swizzle. `q_shared[h,d]` then `partial[h, BLOCK_K]`. |

No column-padding (`D+1` / `D+8`) anywhere. Swizzle is index remap, zero extra LDS.

ISA LDS allocations are 1 KB granules, so HIP-reserved 33300 → 33792, 49364 → 50176, 59620 → 60416. Still 1 WG in 64 KB.

---

## 6. Occupancy estimate (16 waves/SIMD, 1024 VGPR, granule 16, 64 KB/WG)

Assumptions: wave32, WGP mode (4 SIMD32), one WG’s waves spread across the WGP. LDS pool quoted both as 64 KB (author “per-CU”) and 128 KB (WGP).

| Kernel | waves/WG | LDS WGs @64KB | LDS WGs @128KB WGP | waves/SIMD if 1 WG | waves/SIMD if 2 WG/WGP | VGPR-limited waves/SIMD | Bound |
|---|---|---|---|---|---|---|---|
| decode 128 | 4 | 1 | 2 | 1 | 2 | unknown (likely ≫ 2; decode state is tiny) | **LDS** → 2/16 = 12.5% |
| decode 256 | 8 | 1 | 2 | 2 | 4 | unknown; `(256,1)` allows 256 VGPR → 4 waves max | **LDS** (or VGPR if ≥128) |
| prefill 128 / splitk 128 | 4 | 1 | 2 | 1 | 2 | unknown | **LDS** → 12.5% |
| prefill 128 short | 8 | 1 | 2 | 2 | 4 | unknown | **LDS** |
| prefill 256 / splitk 256 | 8 | 1 | 2 (118 KB) | 2 | 4 | unknown | **LDS-tight** |
| combine / reduce | 4–8 | many | many | 1–2 | — | low | not interesting |
| sparse MLA | 1 | 56 | 56 | 1 / WG | VGPR ~6 | **~6 / 16 = 37.5%** | **VGPR** |
| paged MQA | 2 | 1 | 2 | 0.5–1 | 1 | unknown | **LDS** |

**Occupancy-0 flags**

1. **LDS > 64 KB:** none. Worst is 59620 B.
2. **`(256, 1)` VGPR blow-up:** four kernels (`decode_256`, `prefill_128_short`, `prefill_256`, `prefill_splitk_256`). The hint does not *cause* occ-0; it *fails to prevent* a 256-VGPR compile. A 256-thread WG at 256 VGPR still places 1 WG in WGP mode (2 waves × 256 = 512 ≤ 1024). Occ-0 needs VGPR > 256 (compile fail / spill) or a tool that requires all 8 waves on one SIMD (`1024/align(vgpr,16) < 8`). hip-craft notes a **203 VGPR / 256-thread** tile reported occ-0 on gfx1030 in that accounting — **unverified here** (no resource dump). Treat `(256, 1)` as “unpinned occupancy”, not as a 1-block guarantee.
3. **MLA V1 (dead):** 1028 fp32/thread would be occ-0 / scratch. V2 is fine.
4. **Wanted “1 block/CU”** is achieved by LDS size, not by `__launch_bounds__(*, 1)`.

Recommended pin if they actually want the LDS-era 1–2 WG occupancy *and* a VGPR cap that matches it:

```c
// 256-thread FA tiles: 8 waves, 2 waves/SIMD at 1 WG/WGP → ask for 2..4
__launch_bounds__(256, 2)   // HIP: min 2 waves/EU, VGPR ≤ 256 (file cap)
// or, if they really meant CUDA minBlocks=1:
__launch_bounds__(256, 8)   // min 8 waves/EU, VGPR ≤ 128 — compile may fail if the kernel needs more
```

Do not keep `(256, 1)` under the belief it means 1 block/CU.

---

## 7. Sources

- https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/fa_rdna2.cu
- https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/sparse_mla_rdna2.cu
- https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/indexer_paged_mqa_rdna2.cu
- https://raw.githubusercontent.com/BlivionIaG/vllm/perf/rdna2_w4a16/csrc/rocm/ops.h
- HIP `__launch_bounds__`: https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.1/how-to/hip_cpp_language_extensions.html (second arg = `MIN_WARPS_PER_EXECUTION_UNIT`; CUDA port formula)
- Macro lowering: https://github.com/ROCm-Developer-Tools/HIP/issues/2521
- Live branch: `rdna2_extras` @ `3e05abc9` (still `(1,1)`).
- Snapshot tree listing: `GET /repos/BlivionIaG/vllm/git/trees/perf/rdna2_w4a16?recursive=1` SHA `9ac015d0…`
