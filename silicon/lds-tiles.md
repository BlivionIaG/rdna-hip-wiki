# RDNA 2 (gfx1030) LDS bank math and GEMM / attention tiles

Audience: someone writing custom HIP kernels for LLM inference/training (vLLM, SGLang). Not a consumer GPU review.

Companion silicon brief: `/workspace/rdna2-architecture-brief.md`. This note does **not** re-derive WGP/CU/cache sizes; it uses them as given and only re-states a number when the bank or tile argument needs it.

Rule: every concrete number is attributed. If a figure is not in a source that was opened, it is marked **unknown**. No invented microbenchmarks.

---

## Recipes — try these first

Nine starting points (eight FP16/FP32/WMMA, plus INT8 DP4A). All assume HIP default **wave32 + WGP mode**, `__launch_bounds__` / workgroup **256** (8 waves), FP16 inputs, **FP32 accum**, LDS **≤ 64 KB/workgroup** (RDNA 2 ISA §2.3.1). “2 WG/WGP” means each workgroup’s LDS is ≤ 32 KB so two can coexist on the 128 KB WGP (ISA §2.3.1 + §10.3).

| # | Kernel | BM / BN / BK or Br / Bc / d | LDS (single buf, no pad) | 2 WG/WGP? | Why this first |
|---|---|---|---|---|---|
| 1 | GEMM FP16→F32 | **64 × 64 × 32**, WG=256 | A 4 KB + B 4 KB = **8 KB** | yes (double-buf 16 KB) | Small C (16 F32/thread), full 16-wave occupancy still possible at 64 VGPR. Sequential `float`/`half2` along K is conflict-free on 64 banks. |
| 2 | GEMM FP16→F32 | **128 × 64 × 32**, WG=256 | A 8 KB + B 4 KB = **12 KB** | yes (double-buf 24 KB) | First step up in arithmetic intensity. C = 32 F32/thread. |
| 3 | GEMM FP16→F32 | **128 × 128 × 32**, WG=256 | A 8 KB + B 8 KB = **16 KB** | yes if single-buf; **no** if you also pad + double-buf | C = 64 F32/thread → you are already in the 128-VGPR / 8-wave regime once addresses and fragments land. Measure before going bigger. |
| 4 | GEMM FP32 | **64 × 64 × 16**, WG=256 | A 4 KB + B 4 KB = **8 KB** | yes | Same occupancy story as #1; BK=16 because each element is 4 B and you still want a 128 B global line (32×FP32). |
| 5 | Attention d=64 FP16 | **Br=64, Bc=64**, Q in VGPR, K/V in LDS | K 8 KB + V 8 KB (+ optional P 8 KB) = **16–24 KB** | yes | Matches FlashAttention-1 `Bc=⌈M/(4d)⌉` with M=32768 FP16 elems in 64 KB (`Bc=128` is the paper max; 64 is the FA-2 “typical” that still leaves room for softmax scratch). |
| 6 | Attention d=128 FP16 | **Br=64, Bc=32** (or 64 if P stays in VGPR) | K 8 KB + V 8 KB = **16 KB** at Bc=32; 32 KB at Bc=64 | yes at Bc=32 | FA-1 formula gives `Bc=64, Br=64` if Q/K/V/O all sit in 64 KB. Putting Q and O in VGPRs is the FA-2/CK pattern and is what makes Bc=64 fit. |
| 7 | Attention d=256 FP16 | **Br=32, Bc=32**, Q/O in VGPR | K 16 KB + V 16 KB = **32 KB** | borderline (one WG if you also keep P) | FA-1: `Bc=⌈32768/(4·256)⌉=32`. llama.cpp HIP tile kernel has already shown 256-thread / large-d configs blowing occupancy on gfx1030 — start at 128 threads if VGPR spikes. |
| 8 | gfx1100 only | **64 × 64 × 16** (or 128 × 64 × 16) via **WMMA 16×16×16**, WG=256 wave32 | same LDS bytes as #1/#2, but layout is **fragments**, not VALU vectors | yes | First WMMA tile: multiples of 16, A col-major / B row-major, lanes 0–15 replicated into 16–31 (GPUOpen WMMA). Do not run this on gfx1030. |
| 9 | GEMM INT8→I32 (DP4A) | **64 × 64 × 64**, WG=256 | A 4 KB + B 4 KB = **8 KB** | yes (double-buf 16 KB) | Twin of #1. `v_dot4c_i32_i8` / `__builtin_amdgcn_sdot4`. BK multiple of 4 (pack 4×i8 per dword). Same 8 KB as FP16 64×64×32 because K is 2× denser. |

How to read the table: these are **compile-and-measure** seeds, not claimed peaks. Occupancy math is LLVM `IsaInfo` (1024 VGPR/SIMD32, granule 16, 16 waves/SIMD). LDS bytes are `rows × cols × sizeof(T)` with no padding and no XOR hole — add 12.5–25% if you pad (CK bank-conflict doc).

### INT8 / DP4A (recipe 9)

RDNA 2 has no CUDA `__dp4a` name, but the silicon op is the same class:

| Name | What it is | Source |
|---|---|---|
| `v_dot4c_i32_i8` / `v_dot4_i32_i8` | 4-way signed i8 dot, i32 accum | RDNA 2 ISA feature list; LLVM GFX10 asm |
| `__builtin_amdgcn_sdot4(a, b, c, false)` | HIP/Clang intrinsic → `sdot4` / `v_dot4c_i32_i8` | llama.cpp `ggml_cuda_dp4a` (all gfx103x as of #8629) |
| `__builtin_amdgcn_udot4` | unsigned | LLVM `llvm.amdgcn.udot4` |
| `v_dot8_i32_i4` / `v_dot8_u32_u4` | 8-way i4, 1024 ops/clock/CU | RDNA 2 ISA; GPUOpen WMMA table |
| gfx1100 `__builtin_amdgcn_sudot4` | mixed-sign DOT4 | llama.cpp RDNA3 path |

GPUOpen “WMMA on RDNA 3” table: RX 6950 XT **IU8 = 512** ops/clock/CU, **IU4 = 1024**. RDNA 3 IU8 is also 512 — WMMA does not beat DOT4 on INT8 the way it beats packed F16.

Packing: each thread holds K in dwords of 4×i8. BK must be a multiple of 4. Bank formula is still dword-based, so packed i8 walks banks like `int`, not like `half`. Sequential `int` along K is conflict-free for a wave32 (32 lanes, 64 banks).


---

## 1. LDS addressing and bank index

### 1.1 Hardware (ISA, not HIP paraphrase)

RDNA 2 ISA §2.3.1 and §10.1 (Document ID 70648):

- 128 KB LDS per WGP.
- **64 banks**, each **512 entries × 4 bytes** (64 × 512 × 4 = 131072).
- Those 64 banks are **two sets of 32**, each set affiliated with one CU (one pair of SIMD32s).
- Each bank is a **512×32, 1R/1W per clock** RAM.
- **“Dwords are placed in the banks serially, but all banks can execute a store or load simultaneously.”**
- One workgroup may allocate **at most 64 KB**.
- Allocation granule: **256 dwords (1 KB), 256-dword aligned**, no wrap (ISA §3.6.6).
- Conflict-free indexed/atomic: **as little as 1 cycle (wave32) or 2 cycles (wave64)**; worst case **64 cycles** (ISA §10.4.3).
- Peak concurrent wording: “concurrently execute **32** write or read instructions, each nominally 32-bits; `read2`/`write2` can be 64-bits each.”
- Same-bank **different addresses** serialize. HIP states the complementary rule: **same-address broadcast is not a conflict** (HIP Hardware implementation, LDS “Conflict resolution”).

HIP (https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html): RDNA 2/3/4 have **64 banks × 4 B = 256 B/cycle** aggregate; “SIMDs connect to the LDS in pairs, with each pair sharing a **64-byte bidirectional port**.”

Those two peak wordings (ISA “32 × 32-bit” vs HIP “64 × 4 B”) are **not reconciled in either document**. Use them as bounds, not as a single number you can plug into a roofline without measuring.

### 1.2 Bank index — WGP mode (HIP/LLVM default)

ISA does **not** print `bank = …`. The only layout sentence is “Dwords are placed in the banks serially” across 64 dword-wide banks. The unique mapping consistent with that sentence is:

```
dword_index = byte_address >> 2          // ignore byte-in-dword; banks are 4 B
bank        = dword_index % 64           // WGP mode, 64 banks
```

Equivalently: `bank = (byte_address / 4) % 64`.

This is the GCN-family formula with the modulus changed from 32 → 64. CK’s published formula is the 32-bank (GCN/CDNA) version: `bank = (address_in_bytes / 4) mod 32` (CK Tile “Understanding AMD GPU LDS and Bank Conflicts”). On gfx1030 in WGP mode, replace 32 with 64.

Two addresses **conflict** when `bank(a) == bank(b)` and the dwords are **not** the same address. Alias distance: **64 dwords = 256 bytes**.

```
WGP 64 banks × 4 B. Dwords placed serially (ISA §2.3.1 / §10.1):

  byte     0    4    8   …  252   256   260
  dword    0    1    2   …   63    64    65
  bank     0    1    2   …   63     0     1

wave32 sequential `int`/`half2`: 32 lanes × 1 dword → banks 0–31, conflict-free.
packed i8/i4 is still a dword walk — same map as `int`, not as `char`.
stride 256 B = +64 dwords = same bank → serialize (worst 64 cycles, ISA §10.4.3).
same-address broadcast is not a conflict (HIP LDS “Conflict resolution”).
```

`ds_read_b128` cycle count on gfx1030 is still **unknown** in the ISA (§8). Prefer `ds_read2_b64`. DOT8 (`V_DOT8_I32_I4`) does not change the bank map — 8×i4 still rides one dword. No invented diagram beyond this serial walk.

### 1.3 Bank index — CU mode (`-mcumode`)

ISA §2.3.1 / §10.3:

- LDS is split into an **upper and lower half**, each serving two SIMD32s.
- A workgroup’s waves live on **one CU**; they only see the half attached to that CU.
- “This mode can provide **higher LDS memory bandwidth** than WGP mode.”
- Cross-half reads are illegal (upper cannot read lower).

The ISA still does **not** write a CU-mode modulo. Two readings are consistent with the text; they are **not** the same program:

1. **Physical 64-bank map unchanged.** Your 64 KB allocation is carved entirely out of one 32-bank half. Consecutive dwords *inside that allocation* then walk 32 banks, i.e. the programmer-visible map is `bank = dword_index % 32` relative to the allocation base.
2. **Hardware remaps the half to a 32-bank serial map** at the port. Same formula, different physical banks.

**Unknown:** which of those two the silicon does. Confirm with a 32-vs-64 stride microbench under `-mcumode` if a kernel is LDS-bandwidth bound. Until then, treat CU mode as “32 banks, `bank = (byte_address/4) % 32` relative to the workgroup’s LDS base” — that is the conservative conflict model and matches CK’s GCN formula.

Workgroup LDS cap is **still 64 KB** in both modes (ISA §2.3.1). The extra 64 KB is so **two** workgroups (or the two CU-mode halves) can each hold 64 KB.

### 1.4 Conflict phases vs instruction width

CK (CDNA, 32 banks, wave64) documents that `ds_write_b128` / `ds_read_b128` are checked in **8-lane phases**, not across the whole wave at once. That phase table is **CDNA/wave64**. It is **not** an RDNA 2 fact.

What *is* an RDNA 2 fact:

- Wave32 native; a conflict-free indexed op can retire in **1 cycle** (ISA §10.4.3).
- 32 consecutive dword lanes cannot fill 64 banks — a single wave32 of sequential `float` uses 32 of 64 banks and is conflict-free.
- A second wave on the other CU-half can use the other 32 banks in the same cycle (CU mode “both halves run in parallel”, ISA §2.3.1). In WGP mode the ISA warns far-side access “may be lower in some cases” (§10.3).

Do not import CK’s 8-lane `ds_read_b128` phase groups into a gfx1030 kernel without measuring. The **address → bank** map above is what you can trust.

### 1.5 Worked examples (WGP mode, `bank = (addr/4) % 64`)

Assume LDS base 0, `lane = threadIdx.x % 32`, one wave32. “Conflict-free” means no two lanes touch **different dwords in the same bank**.

#### `float` (4 B) — sequential

```
addr[lane] = 4 * lane
bank[lane] = lane % 64          // 0..31
```

32 distinct banks. **Conflict-free.** Uses half the banks.

#### `float2` (8 B) — sequential

```
addr[lane] = 8 * lane           // two consecutive dwords
banks     = {2*lane, 2*lane+1} % 64
```

Lane 0 → banks 0,1; lane 15 → 30,31; lane 16 → 32,33; lane 31 → 62,63. **64 banks filled once. Conflict-free** if the instruction actually issues both dwords (ISA `read2`/`write2` is the 64-bit form).

#### `float4` (16 B) — sequential

```
addr[lane] = 16 * lane
banks     = {4*lane .. 4*lane+3} % 64
```

32 lanes × 4 dwords = 128 dword-slots on 64 banks → **2-way** if the whole vector is one indexed op. ISA peak is “32 × 32-bit or read2 64-bit”, so a 128-bit op is **width-limited even without a classic stride conflict**. Prefer `ds_read2_b64` / two `b64` over hoping `b128` is one-cycle on gfx1030 — **cycle count of `ds_read_b128` on gfx1030 is unknown in the ISA**.

#### `half` (2 B) — sequential — **disaster**

```
addr[lane] = 2 * lane
dword     = lane / 2
bank      = (lane / 2) % 64
```

Lanes 0 and 1 hit **the same dword-bank, different bytes**. That is a **2-way conflict** (different addresses, same bank). Never walk LDS as scalar `half`. Pack to `half2`/`half4`.

#### `half2` (4 B) — sequential

Same map as `float`. **Conflict-free.**

#### `half4` (8 B) — sequential

Same map as `float2`. **Conflict-free** under the `read2`/`write2` caveat.

#### Stride-32 dwords (128 B) — **2-bank, 16-way**

Column walk of a 32-float-wide tile, or `lds[lane * 32]`:

```
bank[lane] = (32 * lane) % 64 = 0 if lane even, 32 if lane odd
```

16 lanes → bank 0, 16 lanes → bank 32. **16-way conflict.** ISA worst-case wording is 64 cycles; this is 16 serial beats on two banks.

#### Stride-64 dwords (256 B) — **32-way disaster**

```
bank[lane] = (64 * lane) % 64 = 0    // every lane, different dword
```

All 32 lanes serialize on bank 0. This is the “leading dimension is a multiple of 64 dwords” foot-gun.

#### Padding recipes

**+1 column (the one that always works for a power-of-two stride).**

If you walk **rows** with consecutive lanes and the logical row stride is `S` dwords:

```
bank = (S * row + col) % 64
```

- `S = 32` → `bank = (32*row + col) % 64` → column `col` is 16-way (above).
- `S = 33` → `33 ≡ 1 (mod 64)` → `bank = (row + col) % 64` → column walk is sequential. **Conflict-free.**
- `S = 64` → 32-way. `S = 65` → `65 ≡ 1 (mod 64)` → sequential again.

Cost: one extra dword per row. For a 64×32 FP16 tile stored as 64 rows × 16 dwords, padding to 17 dwords/row is +64×4 B = 256 B. CK quotes **12.5–25%** extra LDS for padding in general (CK bank-conflict page).

**+1 bank of padding on the inner dimension** is the same idea: make `leading_dwords % 64 == 1` (or any **odd** number — `S ≡ odd` avoids the 32-stride collapse; `S ≡ 1 (mod 64)` is the cleanest).

**Do not pad to a multiple of 64 dwords.** That *creates* the stride-64 disaster.

Triton AMD (commit `3c3f48b`, PR #9780) sets

```
padInterval = max(innerDimLength, 64 * 4 / elemBytes)
```

so padding lands on the **bank-wrap boundary** (64 dwords × 4 B = 256 B), not on every row. That is the compiler-side version of “pad at the 64-bank period, not per row.”

### 1.6 Allocation gotcha

ISA §3.6.6: allocations are **1 KB aligned, 1 KB granules**. A “33-column” pad that makes the tile 33×4 = 132 B wide still rounds the *workgroup* allocation up to a kilobyte. Two 31.5 KB tiles do **not** fit as two workgroups; two 32 KB tiles do.

---

## 2. Conflict-free layouts

### 2.1 GEMM A-tile / B-tile

Notation: workgroup computes `C[BM, BN] += A[BM, BK] @ B[BK, BN]`. A and B live in LDS; C lives in VGPRs.

**Row-major A (`A[m, k]`, k inner).** Threads along `threadIdx.x` load consecutive K.

- Store as `ldsA[m * (BK + PAD) + k]` with `PAD` chosen so `(BK + PAD)` in **dwords** is not a multiple of 32 or 64.
- FP16, BK=32 elements = 16 dwords. 16 is fine for a *row* walk (sequential half2). The conflict appears when the inner loop **reads a column** of A (fixed k, consecutive m) — that is a stride-`(BK in dwords)` access.
- BK=32 FP16 = 16 dwords → column stride 16, `16*lane % 64` is 4-way (lane 0,4,8,… share a bank). **Pad to 17 dwords** (34 FP16) or XOR.

**Column-major A** swaps the problem onto the K-walk. Pick the layout so the **LDS load in the inner loop** (the one that feeds VALU every cycle) is the sequential one. The global→LDS store can tolerate a cheaper conflict; the inner-loop load cannot.

**XOR / swizzle (preferred when you can afford the index math).** CK Tile (official, opened):

```
K0' = K0 XOR (M % (KPerBlock / KPack * MLdsLayer))
```

then unmerge `K0'` into `(L, K0'')` when `MLdsLayer > 1`, then merge back to `(M', K')`.

CK’s documented defaults (LDS index-swapping page):

```
MLdsLayer = (TileSize <= 32) ? 1 : (TileSize <= 64) ? 2 : 4;
KPack     = sizeof(T)==2 ? 8 : sizeof(T)==4 ? 4 : 2;   // FP16/BF16 : FP32
// require TileSize % (MLdsLayer * KPack) == 0
```

Worked CK example (ROCm blog “Avoiding LDS Bank Conflicts”, 2025-07-25): `kMPerBlock=64, kKPerBlock=32, kKPack=8`, naive layout is write-clean / **2-way read conflict**; XOR makes both clean, **zero extra LDS**. That example is a CDNA MFMA kernel (32 banks, wave64, `ds_*_b128` phases). The **transform** is architecture-agnostic; the **phase check** is not. On gfx1030, apply the same XOR, then verify with `rocprof` LDS-bank counters — do not assume the CDNA 2-way number.

Minimal HIP XOR for a `BM × BK` FP16 tile with `KPack=8` (16 B = one `float4` / `half8`):

```
// col is in units of KPack vectors, row is M
col_swz = col ^ (row & (BK/KPack - 1));
offset  = row * BK + col_swz * KPack;   // still tight packed
```

Use the **same** swizzle on the read path. Forgetting that is how XOR “makes it worse.”

**B-tile** is the same problem transposed. If B is row-major `B[k, n]`, the inner-loop load is a row of N (sequential — easy) and the store from global may be a column of K (XOR or pad).

### 2.2 Softmax / attention scratch (QKᵀ tile, P tile)

FlashAttention-2 (Dao, 2023, §3.3): split **Q across warps**, keep K and V visible to all warps, so each warp owns a slice of `S = Q Kᵀ` and a slice of `O` with **no inter-warp reduction** through LDS. That is the layout you want on gfx1030 too — LDS is then only K and V (and optionally P if you cannot hold `Br×Bc` in VGPRs).

S / P access patterns:

- **Write S** after the QK VALU loop: usually a row per thread (sequential, easy).
- **Read P as GEMM-1 A** (P @ V): you now walk **columns** of P. This is the GEMM-A column problem. XOR P the same way you XOR A, or keep P in VGPRs (FA-2 / CK: Q stays in VGPRs; P often does too for small Br×Bc).

`Br=64, Bc=64`, FP16 P = 8 KB; FP32 P = 16 KB. At 256 threads that is 16 FP16 or 16 FP32 per thread — P-in-VGPR is realistic at 64×64 and tight at 128×128 (64 FP32/thread for P alone).

Row-max / row-sum of S: **do not** bounce the row through LDS if it already lives in the warp. Use `v_permlane16` / `v_permlanex16` / DPP / `llvm.amdgcn.wave.reduce.fmax` (LLVM AMDGPUUsage). LDS is for **cross-wave** reductions only.

### 2.3 Reduction / scan in LDS

- **Same address, many lanes:** broadcast, not a conflict (HIP). A workgroup reduction tree that ends in `lds[0] +=` is fine at the last step; the *middle* steps are the ones that conflict if you use stride 32/64.
- **Tree with stride `s`:** `lds[lane] += lds[lane + s]`. For `s = 32` dwords you hit the 16-way pattern. Use `s` that is **not** a multiple of 32 dwords, or do the intra-wave part with DPP and only one LDS write per wave.
- **Scan:** exclusive scan with padded stride (`+1` column) or XOR of the scan buffer; or `ds_ordered_count` is **not** an RDNA 2 compute tool you should reach for here (GDS / graphics). Stick to DPP + one LDS slot per wave.
- **Atomics:** 64 integer atomic units on the WGP (ISA §2.3.1). Same-address atomics serialize (HIP). Fine for a workgroup counter; not a GEMM path.

---

## 3. Practical GEMM tiles on gfx1030 (no WMMA, no MFMA)

### 3.1 What the inner loop actually is

RDNA 2 has **no** `v_wmma_*` and **no** `v_mfma_*` (GPUOpen “How to accelerate AI applications on RDNA 3 using WMMA”, comparison table; LLVM `FeatureMAIInsts` / `FeatureVOPDInsts` are gfx11+ / CDNA).

ISA “Feature Changes in RDNA2 Devices” + LLVM `llvm.amdgcn.{s,u}dot2/4/8`:

| Op | Role in a GEMM inner loop |
|---|---|
| `v_fma_f32` | FP32 accum. 1 FMA/lane/clk. 128 FP32 FLOPS/clk/CU (2×SIMD32×32×2). |
| packed F16 FMA (`v_pk_fma_f16` etc.) | **2×** FP32 rate (GPUOpen: **256** FP16 FLOPS/clk/CU on RX 6950 XT). Accum is F16. Use for epilogues, not for a stable GEMM accum. |
| `V_DOT2_F32_F16` / `V_DOT2C_F32_F16` | Two F16 muls + F32 add into an F32 accum. This is the **correct** mixed-precision inner product on gfx1030. |
| `V_DOT4_*` / `V_DOT8_*` | INT8 / INT4. Inference quant kernels, not FP16 GEMM. |

GPUOpen WMMA table is the official “RDNA 2 vs 3” FLOPS/clk/CU statement. There is **no** BF16 matrix type on gfx1030 (`N/A` in that table).

### 3.2 Occupancy vs VGPR (wave32)

LLVM `getTotalNumVGPRs` = 1024/SIMD32; granule 16 (ISA §3.6.4); 16 wave slots/SIMD (LLVM `getMaxWavesPerEU`, GPUOpen Occupancy explained).

| VGPRs / lane | Waves / SIMD | Waves / WGP (4 SIMD) | 256-thread WGs / WGP (8 waves each) |
|---|---|---|---|
| 64 | 16 | 64 | 8 (LDS will cap first) |
| 128 | 8 | 32 | 4 |
| 256 | 4 | 16 | 2 |

SGPRs do **not** limit occupancy on GFX10+ (LLVM `isSGPROccupancyLimited` false; GPUOpen).

LDS cap: 128 KB/WGP, 64 KB/WG. Two 64 KB WGs → 2 WGs × 8 waves = 16 waves on the WGP = **4 waves/SIMD**, i.e. you have already given up the 16-slot file for LDS. That is why recipes #1–#2 stay ≤ 32 KB.

Barrier slots: LLVM `getMaxWorkGroupsPerCU` = 16 barriers in CU mode, 32 in WGP mode (GFX10+). A 256-thread WG is 8 waves and consumes one barrier. You will hit LDS or VGPR long before you hit 32 barriers.

### 3.3 Elements per thread and K-unroll

Workgroup 256, tile `BM × BN`:

```
c_elems = (BM * BN) / 256
```

| Tile | F32 C elems/thread | VGPRs for C alone |
|---|---|---|
| 64×64 | 16 | 16 |
| 128×64 | 32 | 32 |
| 128×128 | 64 | 64 |
| 256×128 | 128 | 128 |

Add ~8–24 VGPR for A/B fragments, addresses, and loop state (this range is **not** a measured number; it is “what a DOT2 inner loop looks like in IR”). Crossing 64 → 128 VGPR drops you from 16 to 8 waves/SIMD.

**5-cycle VALU dest latency** (RDNA architecture deck: “5 cycles of latency are exposed”). Hide it with:

1. Other waves (occupancy), and/or
2. ILP in the same wave: **≥ 5 independent FMAs/DOT2s** in flight.

A `BK=32` FP16 panel with `DOT2` (2 F16 per op) is 16 DOT2 per output element along K. That is already > 5. Unroll K by **8 or 16 DOT2** (K=16 or 32) and software-pipeline:

```
load A,B from LDS for k
s_waitcnt lgkmcnt(…)
DOT2 into C
prefetch next A,B
```

Do **not** `s_waitcnt vmcnt(0)` in this loop — that waits for vector *loads*, not LDS. LDS is `lgkmcnt`. Stores are `vscnt` (RDNA deck: separate VMCNT/VSCNT).

Transcendentals (`v_exp_f32`, `v_rcp_f32`, …) are **¼ rate** and can co-issue with non-transcendental VALU (RDNA deck). Softmax in the same kernel as GEMM will eat issue slots; keep the exp/rcp outside the K loop.

I$ is **32 KB/WGP** (kfd_crat / RDNA deck). A fully unrolled 128×128 epilogue will miss it. Unroll the inner K; do not unroll the whole kernel.

### 3.4 When packed F16 vs DOT2 vs F32

| Situation | Use | Source |
|---|---|---|
| FP16 GEMM, need a stable accum | `V_DOT2_F32_F16` (or unpack + `v_fma_f32`) | ISA DOT2; GPUOpen 256 FP16 FLOPS is *packed math*, not “F16 accum is OK” |
| FP32 GEMM | `v_fma_f32` | default VALU |
| Activation / residual / store pack | packed F16 | 2× rate, F16 dest |
| INT8/INT4 inference | `V_DOT4_*` / `V_DOT8_*` | ISA feature list |
| BF16 GEMM on gfx1030 | **no first-class path** | GPUOpen table: BF16 `N/A` on RX 6950 XT |

`V_DOT2C` is the carry/inline form (LLVM lowers `sdot2` without clamp to `v_dot2c_*` when applicable). Same math.

### 3.5 LDS footprint so two workgroups still coexist

```
lds_single = (BM * BK + BK * BN) * sizeof(T) + pad
lds_double = 2 * lds_single
```

Must be ≤ 64 KB (hard). For **two WGs/WGP**: `lds_* ≤ 32 KB` after 1 KB rounding.

| BM×BN×BK, T | single | double | 2 WG/WGP |
|---|---|---|---|
| 64×64×32 FP16 | 8 KB | 16 KB | yes |
| 128×64×32 FP16 | 12 KB | 24 KB | yes |
| 128×128×32 FP16 | 16 KB | 32 KB | yes if no pad; **no** with 25% pad + double |
| 128×128×64 FP16 | 32 KB | 64 KB | no (one WG owns the WGP’s LDS) |
| 64×64×16 FP32 | 8 KB | 16 KB | yes |

**Give up two-WG** when the extra BK (or double-buffer) buys you more than a second WG’s latency hiding. A 64 KB, 256-thread WG is 8 waves on 4 SIMD = 2 waves/SIMD. That can still hide a 5-cycle VALU dest; it will **not** hide a cold GDDR6 hit. If the kernel is L2/IC-bound, two skinny WGs that share a K-panel in L1 often win. If the kernel is VALU-bound and C is huge, one fat WG is fine.

CK GEMM on CDNA uses double-buffer + XOR as the default policy (CK “building efficient GEMM” blog). Same policy is legal on gfx1030; the limiter is 64 KB, not MFMA fragment size.

rocBLAS/hipBLASLt: gfx1030 has **no MFMA/WMMA**, so Tensile `MI…` kernels are the wrong mental model (rocBLAS #1554 is the “devices without mfma and wmma” class). Do not copy an `MT128x128x32_MI16x16x16` name off a gfx11/gfx90a log and expect it to exist here.

---

## 4. Attention tiles on 64 KB LDS

### 4.1 What must fit

FlashAttention-1 (Dao et al., NeurIPS 2022, Algorithm 1) sizes SRAM as **M elements** and sets

```
Bc = ⌈M / (4d)⌉
Br = min(⌈M / (4d)⌉, d)
```

The `4d` is Qᵢ + Kⱼ + Vⱼ + Oᵢ (author note on issue #766: also `Br·Bc ≤ M/4` because a few `Br×Bc` temporaries live in SRAM).

On gfx1030 the programmable SRAM you own is **64 KB/workgroup**, not the WGP’s 128 KB.

| dtype | M (elements in 64 KB) |
|---|---|
| FP16 / BF16 | 32768 |
| FP32 | 16384 |

Plug in:

| d | FA-1 Bc | FA-1 Br | Notes |
|---|---|---|---|
| 64 | ⌈32768/256⌉ = **128** | min(128, 64) = **64** | 4 × 128 × 64 × 2 B = 64 KB exactly if Q,K,V,O all sit in LDS |
| 128 | ⌈32768/512⌉ = **64** | min(64, 128) = **64** | 4 × 64 × 128 × 2 B = 64 KB exactly |
| 256 | ⌈32768/1024⌉ = **32** | min(32, 256) = **32** | 4 × 32 × 256 × 2 B = 64 KB exactly |

FlashAttention-2 (Dao 2023, §3 “Tuning block sizes”): “Typically we choose blocks of size **{64, 128} × {64, 128}**, depending on the head dimension and the device shared memory size.” That sentence is an A100-class SRAM budget. On 64 KB it is an **upper bound**, not a default.

FA-2 also moves Q (and O) into registers and splits Q across warps (§3.3). That is the layout that actually fits on gfx1030 at the FA-2 “typical” sizes:

```
LDS  ≈  K[Bc, d] + V[Bc, d]           // Q, O, m, ℓ in VGPR
     =  2 * Bc * d * 2 B
```

| d | Bc | K+V LDS | leftover in 64 KB | leftover in 32 KB (2 WG) |
|---|---|---|---|---|
| 64 | 64 | 16 KB | 48 KB (P, double-buf, pad) | 16 KB |
| 64 | 128 | 32 KB | 32 KB | 0 — one WG |
| 128 | 32 | 16 KB | 48 KB | 16 KB |
| 128 | 64 | 32 KB | 32 KB | 0 — one WG |
| 256 | 32 | 32 KB | 32 KB | 0 — one WG |
| 256 | 64 | 64 KB | 0 | impossible |

P tile if you must spill it: `Br × Bc × 2` (FP16) or `× 4` (FP32). `64×64` FP32 P = 16 KB — fits next to K+V at d=64, Bc=64 (16+16=32 KB, 2 WG OK). `128×128` FP32 P = 64 KB — **does not fit** with anything else.

### 4.2 What real AMD-side kernels actually pick

Opened, not inferred:

- **CK-Tile FA-v2 blog** (2025-05-21): problem `K0=N1=128`, workgroup tiles `kM0PerBlock=128, kN0PerBlock=128, kK0PerBlock=32, kN1PerBlock=128, kK1PerBlock=32`. Q stays in VGPRs (`q_reg_tensor`); K and V go LDS. This is a **CDNA MFMA** sample (the blog’s terminology table says “Wavefront (64)”). On gfx1030 you can keep the *pipeline* (Q in VGPR, K/V K-panels of 32) but **cut Br/Bc to 64** or you will lose the WG to VGPR/LDS.
- **TileLang AMD FA** (`example_amd_flash_attn_fwd.py`, opened via search): `IsRDNA()` restricts `block_M, block_N ∈ {16,32}` (fwd) or `{16,32,64}` (bwd) because **WMMA 16×16** fragment layout breaks the softmax→GEMM-2 A transpose above `16 * num_warps`. That constraint is **gfx11+ WMMA**, not gfx1030. On gfx1030 you are not bound to 16; you *are* bound to 64 KB and VALU VGPR.
- **llama.cpp HIP FA** (issue #24672 / discussion #23310): gfx1030/1031 tile kernel. Working decode config `<256,256,1,8>`: 128 threads, **21 504 B** LDS, 102 VGPR, occupancy 6 waves/SIMD. Failing prefill `<256,256,4,8>`: 256 threads, **37 888 B** LDS, 203 VGPR, compiler occupancy 4 waves/SIMD, **runtime occupancy 0** (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` → 0). `nbatch_fa` 32 (decode) / 64 (prefill). Takeaway: a 256-thread, d=256, 200-VGPR FA tile **does not launch** on gfx1030 even when LDS is 38 KB. Shrink threads or VGPR before you grow Br.
- **vLLM ROCm**: Triton attention is the default portable backend on AMD (vLLM blog 2026-03-04). AITER FA is **CDNA-only** (vLLM PR #32944). Radeon fallback is `ROCM_ATTN` (custom HIP paged-attention) or `TRITON_ATTN` (ROCm vLLM optimization page). `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` opts into the FA Triton path on RDNA3+; gfx1030 is not in that RDNA3 enablement PR. **Unknown:** a vLLM-supported first-class FA tile table for gfx1030 — do not claim one.

### 4.3 Practical Br/Bc on gfx1030 (VALU)

Start here, then change **one** axis:

| d | Start (Br, Bc) | WG | LDS plan | If it spills VGPR |
|---|---|---|---|---|
| 64 | **64, 64** | 256 (8 waves) | K+V 16 KB, P in VGPR (16 F32/thread if you keep S) | drop to WG=128, or P to LDS |
| 128 | **64, 32** | 256 | K+V 16 KB; raise Bc→64 if C/P stay in 128 VGPR | Br=32, Bc=32 |
| 256 | **32, 32** | 128 (4 waves) | K+V 32 KB; one WG | stay at 128 threads (llama.cpp lesson) |

Online softmax stats `m, ℓ` are `Br` FP32 — noise next to K/V.

Causal: FA-2 §3.1.1 — skip blocks entirely above the diagonal; mask only the diagonal block. That is independent of LDS.

---

## 5. Cache-aware K-panels

Hierarchy (already established; kfd_crat Sienna Cichlid + RDNA deck):

```
VGPR → LDS 64 KB/WG → L0 16 KB/CU (128 B line) → L1 128 KB/SA (10 CU on Navi 21)
     → L2 4 MB (128 B) → Infinity Cache 128 MB (64 B line) → GDDR6 512 GB/s
```

### 5.1 L0 (16 KB/CU)

A 128×128 FP16 tile is 32 KB. L0 will not hold a GEMM panel. Treat it as a **coalescing filter**. Happy global load: wave32 × 4 B = **128 B = one L0/L1/L2 line**. Structure-of-arrays, consecutive `threadIdx.x`.

### 5.2 L1 (128 KB / SA, 10 CU)

Ten CUs share 128 KB. A K-panel that is **reused by neighboring CUs in the same SA** should be ≤ 128 KB, ideally a fraction of that so two or three live panels fit.

| Panel | Bytes (FP16) | Fits in L1? |
|---|---|---|
| 64 × 64 | 8 KB | yes, many |
| 128 × 64 | 16 KB | yes, ~8 |
| 256 × 128 | 64 KB | two |
| 512 × 128 | 128 KB | one — the whole SA fights over it |

**Recipe:** pick `BK` (GEMM) or `Bc·d` (attention K) so that `sizeof(K-panel) × (CUs simultaneously using it) ≲ 128 KB`. If you launch a grid that walks M with one K-panel held constant, the 10 CUs of an SA can share that panel in L1. If every CU walks a different K, L1 is useless.

Workgroup→CU mapping is **non-deterministic** (HIP SPI). You cannot pin “these 10 WGs to this SA.” You *can* make the working set small enough that **whichever** 10 CUs land in an SA still hit.

### 5.3 L2 (4 MB) vs Infinity Cache (128 MB)

Line sizes differ: L2 **128 B**, IC **64 B** (kfd_crat). Align persistent buffers to 128 B (satisfies both).

| Working set | Aim at | Why |
|---|---|---|
| Active GEMM K-panel, one layer’s hot rows, decoder residual of a few MB | **L2 (4 MB)** | Coherence point; write-through L1; atomics live here (HIP). 4 MB = 2M FP16 = one 4096×1024 FP16 panel, or 512 rows of a 4096-wide FP16 matrix. |
| Whole layer weights that miss L2 but fit in tens of MB; KV window of a decode batch | **IC (128 MB)** | 128 MB = 64M FP16 ≈ 4096×8192 FP16. AMD: “global cache seen by the entire graphics core” (RDNA 2 Explained, W6000). |
| Cold 7B FP16 weights (14 GB) or a KV cache that thrashes 128 MB | **GDDR6** | IC does not help a streaming miss. Quantize or layer-prefetch. |

Navi 22/23/24 IC is 96/32/16 MB (press + kfd_crat). A “fits in 128 MB” tile is wrong on a 6600 XT. Query the L3 size; do not hard-code 128.

**IC absolute TB/s:** not in the ISA or product pages opened for the companion brief. Do not quote one.

Two concurrent kernels share IC. A GEMM that “persists” weights in IC will lose them if a copy engine or another stream thrashes 128 MB.

---

## 6. gfx1100 (RDNA 3) delta — fragments, not a WMMA tutorial

Same WGP dual-CU, same 16 wave slots/SIMD, same 64 KB/WG LDS cap (companion brief). What changes for the inner loop:

### 6.1 WMMA 16×16×16 (GPUOpen WMMA article; RDNA 3 ISA §7.9)

```
D_frag = __builtin_amdgcn_wmma_<CD>_16x16x16_<AB>_w32(A_frag, B_frag, C_frag, OPSEL)
```

| Item | Fact | Source |
|---|---|---|
| Tile | **16×16×16 only**. Bigger GEMMs are a grid of 16×16. | GPUOpen |
| Wave | `w32` or `w64` in the intrinsic name. Default HIP is wave32. | GPUOpen; LLVM |
| A,B format | FP16, BF16, IU8, IU4 | GPUOpen table |
| C,D format | FP32, FP16, BF16, I32 | GPUOpen |
| A_frag / B_frag | **16 elements / lane**, packed: 8 VGPR (FP16/BF16), 4 VGPR (IU8), 2 VGPR (IU4) | GPUOpen |
| C_frag / D_frag | wave32: **8 VGPR**; wave64: 4 VGPR. Unpacked. OPSEL selects lo/hi 16 bits when CD is 16-bit. | GPUOpen |
| Replication | wave32: lanes **0–15 must equal 16–31**. wave64: also copy into 32–47 and 48–63. | GPUOpen |
| Layout | **A column-major, B/C/D row-major.** A holds 16 columns in VGPRs; B holds 16 rows. | GPUOpen |
| FLOPS/clk/CU | FP16 **512** (vs 256 on RDNA 2); BF16 **512** (vs N/A) | GPUOpen table |
| Issue | one WMMA “coordinates 32 clocks of optimal work scheduling” (Mantor, quoted in GPUOpen) | GPUOpen |

LDS consequence: the inner-loop load is **no longer** “32 consecutive dwords into 32 VALU lanes.” It is “16 packed F16 for this lane’s column of A, duplicated into the other half-wave.” XOR/pad still exist to keep that 16-wide load conflict-free; the **swizzle period** should be the WMMA K-pack (16 F16 = 8 dwords), not a VALU `float4`.

TileLang’s RDNA restriction (`block_M,N ≤ 32`) is exactly this: WMMA D-fragment layout ≠ WMMA A-fragment layout, so a softmax tile sitting in the D registers cannot be fed into GEMM-2 as A without a shared-memory transpose, and that transpose breaks above `16 × num_warps`. Budget an LDS transpose buffer (`Br × Bc`) if you fuse FA on gfx1100; that is why recipe #8 stays at 64×64.

### 6.2 VOPD

RDNA 3 ISA §7.6; LLVM `FeatureVOPDInsts`. Selected VALU ops dual-issue in wave32 (`v_dual_*`). rocprofiler-compute: FP32 128 → **256** FLOPS/CU/cycle when VOPD fires. This does **not** replace WMMA for GEMM; it helps the F32 softmax / residual path that sits next to WMMA.

### 6.3 What not to change

Bank count is still 64 (HIP). Workgroup LDS cap is still 64 KB. Wave slots still 16. `-mcumode` / `-mwavefrontsize64` still exist (LLVM). VGPR file is 1024 on most gfx1100; some SKUs have 1536 (`Feature1536VGPRs`) — **unknown per SKU in this pass**, query LLVM `getTotalNumVGPRs` if you need it.

---

## 7. HIP / LLVM knobs

### 7.1 Wave32 vs wave64

| | wave32 (default) | wave64 (`-mwavefrontsize64`) |
|---|---|---|
| Hardware | Native. One pass / SIMD32. | Same instruction twice (lo 32, hi 32). ISA §2.1. |
| VGPR granule | 16 | 8 |
| VGPR accounting | 1024 physical | 512 “compiler total” (same SRAM, counts double) |
| Conflict-free LDS | 1 cycle | 2 cycles (ISA §10.4.3) |
| Occupancy at 256 VGPR | 4 waves/SIMD | 2 waves/SIMD |

**Default to wave32.** Wave64 doubles the VGPR cost of a wave and turns every VALU into two beats. Measure wave64 only if you have a wide reduction / interpolation-class reason, or you are matching a wave64 WMMA intrinsic on gfx1100.

Clang: `-m[no-]wavefrontsize64` (LLVM AMDGPUUsage Target Features; ROCm `rocmcc` page).

### 7.2 WGP vs CU mode

| | WGP (`-mno-cumode`, default) | CU (`-mcumode`) |
|---|---|---|
| SIMDs / WG | 4 | 2 |
| LDS visible | 128 KB address space; WG still ≤ 64 KB | 64 KB half, local to the CU |
| ISA reason to switch | more ALU + texture BW for a WG of ≥ 4 waves | “higher LDS memory bandwidth”; both halves in parallel |
| Far-side LDS | legal, “performance may be lower” | illegal |

**When `-mcumode` is worth measuring:** the kernel is **LDS-bandwidth bound**, the workgroup is ≤ 2 waves of useful work per CU (or you are happy with 2 SIMD), and you are not using the second CU’s ALUs. Attention softmax scratch that streams through LDS is the candidate. A VALU-bound 8-wave GEMM that wants 4 SIMD should stay in WGP mode.

LLVM: `cumode` feature, “When disabled native WGP wavefront execution mode is used, when enabled CU wavefront execution mode is used” (AMDGPUUsage). HIP default follows LLVM default = WGP.

### 7.3 Occupancy hints

```c
__attribute__((amdgpu_waves_per_eu(4, 8)))
__attribute__((amdgpu_flat_work_group_size(256, 256)))
```

Clang AttributeReference:

- `amdgpu_waves_per_eu(min[, max])` — optimization hint. Backend **limits VGPR/LDS/scratch** so that at least `min` and at most `max` waves fit on an EU (SIMD32). Requesting more waves hides latency and risks spill; requesting fewer can cut cache thrash. `0,0` = no limit. A warning is emitted if the backend cannot meet it.
- `amdgpu_flat_work_group_size(min, max)` — dispatch will stay in `[min, max]`. `0,0` implies default **128, 256**. Helps barrier codegen and scratch promotion.
- `amdgpu_num_vgpr` / `amdgpu_num_sgpr` — **deprecated**; use `waves_per_eu`.

LLVM IR names: `"amdgpu-waves-per-eu"="m,n"`, `"amdgpu-flat-work-group-size"="min,max"`.

EU on RDNA = one SIMD32 (GPUOpen Occupancy explained; LLVM `getMaxWavesPerEU`). So `amdgpu_waves_per_eu(4)` means “compile so 4 waves fit on each SIMD32” → 64 VGPR at wave32 (1024/4 = 256 is the *max* VGPR at 4 waves; 4 waves is the *min occupancy*, so VGPR ≤ 256). Read it as occupancy, not as a VGPR cap.

Pre-check:

```
llvm-calc-occupancy -mcpu=gfx1030 --wg-size=256 --vgprs=N --lds=K
```

(LLVM CommandGuide `llvm-calc-occupancy`; same math as the backend.)

### 7.4 Build lines

```
# gfx1030 (Navi 21). Default: wave32, WGP.
hipcc --offload-arch=gfx1030 -O3 ...

# experiments — measure both, do not ship on a guess
hipcc --offload-arch=gfx1030 -mcumode ...
hipcc --offload-arch=gfx1030 -mwavefrontsize64 ...

# gfx1100 WMMA path
hipcc --offload-arch=gfx1100 ...
# intrinsic needs ROCm ≥ 5.4 (GPUOpen WMMA)
```

`gfx10-3-generic` covers gfx1030–1036 with no ISA restrictions listed (LLVM Processors table). Prefer the specific `gfx1030` for a kernel you will only run on Navi 21.

vLLM / SGLang on Radeon: `ROCM_ATTN` or Triton; do not enable AITER FA (CDNA). `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` is the documented RDNA3+ FA Triton opt-in, not a gfx1030 guarantee.


---

## Radiance A-tile methodology (Take Later)

Source: [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) @ **`22c69cd`** (`radiance_mxfp4_fp8.hip`, `patch_unified_attention_lds.py`). Target card there is **gfx1201** (RDNA4 / R9700). Engine Take/Leave and tok/s live elsewhere — **do not** duplicate them here. This section is **LDS methodology only**, rewritten onto our tiles Later.

### Take Later (rewrite on gfx1030 / EXL3 / `q_gemm` tiles)

| Idea | What they do | Our map |
|---|---|---|
| **+8 B row pad** | `#define PAD 8`; `ASTR = BK + PAD`, `WSTR = BK + PAD`. Comment: without it, 16 lanes reading rows 64 B apart collide 8-ways on **32 × 4 B** banks (WMMA fragment half-wave). | Same pad already appears on extras EXL3 (`LDS_PAD=8`, `BLOCK_K=256`). On gfx1030 banks are **64 × 4 B** (alias 256 B) — keep +8 (or verify XOR) when staging A/W; do **not** import their 32-bank arithmetic blindly. |
| **64 KiB clamp before capture** | `patch_unified_attention_lds.py`: shrink `num_stages` then `TILE_SIZE` until `TILE × next_pow2(head) × el × stages + 256 ≤ 65536`. Hard correctness for cudagraph capture (head 256 × 2 B × 2 stages and head 512 × fp8 both hit 65792 without it). | Same formula for any Triton / HIP tile that stages K/V (or A/W) into LDS. Pointer: [fa-occupancy.md](fa-occupancy.md), [hip-craft.md](hip-craft.md) checklist #3. Prefill leftover stays `(N,1)` / 1 WG @ 64 KiB — clamp does not replace a BR/BC shrink. |
| **LUT / kMag in `__constant__` / LDS; fold block scale out of inner loop** | `kLUT[16]` + `kMag[16][2]` in `__constant__`; A-tiled round 2 copies `kMag` into `__shared__ sMag[32]` so the fold is a `ds_load` (own counter), not a dependent constant load that drains `loadcnt` every slab. MX E8M0 block exponent folded into the weight table → one per-row epilogue factor. | Maps to **EXL3 grain / `q_gemm`**, not WMMA: procedural codebook stays VGPR; any small magnitude/scale table belongs in `__constant__` or a tiny LDS copy; **do not** park a per-lane nibble LUT in K$ (llama.cpp #24438). Dest produce stays EXL3 grain v2 `bits=3` `M≤8`. |

### Leave (explicit)

- **All WMMA / FP8-WMMA objects** (`v_wmma_f32_16x16x16_fp8_fp8`, fragment layouts, gfx1201-only). gfx1030 has **no** WMMA / MFMA / FP8 unit. Dest mxfp4 is unpack E2M1+UE8M0 → `fdot2`.
- AutoRound HIP, R4D, their tok/s, gfx1201 launch geometry.
- Their “32 × 4 B banks” sentence as a gfx1030 fact — that is WMMA half-wave geometry on RDNA4, not ISA §2.3.1 on Navi 21.

### Cite

- `radiance_mxfp4_fp8.hip` @ `22c69cd` — `PAD 8`, `kLUT` / `kMag`, `sMag` LDS copy, `radiance_lds_barrier()` (LDS-scoped fence).
- `patch_unified_attention_lds.py` @ `22c69cd` — `TILE * hs * el * stages + 256 > 65536` clamp.

---

## 8. Unknowns (do not invent)

| Question | Status |
|---|---|
| Exact CU-mode bank modulo (32 vs 64, remapped or not) | ISA does not print it |
| `ds_read_b128` / `ds_write_b128` cycle count and phase groups on gfx1030 | CK phases are CDNA wave64; ISA only gives 1–64 cycles |
| Reconciliation of ISA “32 × 32-bit concurrent” vs HIP “256 B/cycle” | both opened; not the same sentence |
| gfx1030 GEMM peak % of 256 FP16 FLOPS/clk/CU | no microbench in this pass |
| vLLM first-class FA Br/Bc table for gfx1030 | not in the opened vLLM/ROCm pages |
| Infinity Cache TB/s | not in ISA / product pages |
| rocBLAS Tensile MT* list that actually ships for gfx1030 | rocBLAS historically did not treat gfx1030 as a first-class MFMA target |

---

## 9. Sources actually opened

1. AMD “RDNA 2” ISA 70648 — https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture — text extract `/workspace/rdna2-src/rdna2-isa.txt` (§2.3.1, §3.6.6, §10.1, §10.3, §10.4.3)
2. Companion brief `/workspace/rdna2-architecture-brief.md` (WGP/CU, VGPR, caches, DOT/WMMA absence)
3. HIP Hardware implementation — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (64 banks, 256 B/cycle, 64 B port, broadcast, WGP)
4. CK Tile LDS bank conflicts — https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
5. CK Tile LDS index swapping — https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/lds_index_swapping.html (`K0'`, `MLdsLayer`, `KPack`)
6. ROCm blog, LDS bank conflict / XOR — https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html (64×32, KPack 8, 2-way naive read)
7. ROCm blog, CK-Tile FlashAttention-v2 — https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html (128/128/32 tiles, Q in VGPR)
8. Triton AMD pad interval — https://github.com/triton-lang/triton/commit/3c3f48beaea14b2b58dcaf6aab5768f0ab0a5827 (`64*4/elemBytes`)
9. Triton padded layout + row permute — https://github.com/triton-lang/triton/pull/7929
10. FlashAttention (Dao et al. 2022) — https://ar5iv.labs.arxiv.org/html/2205.14135 — `Bc=⌈M/4d⌉`, `Br=min(⌈M/4d⌉,d)`
11. FlashAttention-2 (Dao 2023) — https://tridao.me/publications/flash2/flash2.pdf — `{64,128}×{64,128}`, Q-split warps, no split-K
12. GPUOpen WMMA on RDNA 3 — https://gpuopen.com/learn/wmma_on_rdna3/ (fragments, replication, 16×16×16, FLOPS table)
13. LLVM AMDGPUUsage — https://llvm.org/docs/AMDGPUUsage.html (`cumode`, `wavefrontsize64`, gfx1030 features)
14. Clang `amdgpu_waves_per_eu` / `amdgpu_flat_work_group_size` — https://clang.llvm.org/docs/AttributeReference.html
15. ROCm `rocmcc` — https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html (`-mcumode`, `-mwavefrontsize64`)
16. GPUOpen Occupancy explained — https://gpuopen.com/learn/occupancy-explained/
17. llama.cpp HIP FA occupancy on gfx1030 — https://github.com/ggml-org/llama.cpp/issues/24672 and https://github.com/ggml-org/llama.cpp/discussions/23310
18. vLLM Triton backend / ROCm FA enablement — https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive ; https://github.com/vllm-project/vllm/pull/32944 ; https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html
19. TileLang RDNA FA block sizes — https://github.com/tile-ai/tilelang/blob/17c4b384/examples/amd/example_amd_flash_attn_fwd.py
20. RDNA architecture deck — https://gpuopen.com/download/RDNA_Architecture_public.pdf (5-cycle dest, VMCNT/VSCNT, I$ 32 KB)
