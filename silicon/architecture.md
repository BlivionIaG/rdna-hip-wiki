# RDNA 2 (gfx1030) silicon briefing for HIP kernel writers

Audience: kernel engineers writing custom HIP for LLM inference/training (vLLM, SGLang), not a consumer GPU review.

Rule used throughout: every concrete number is attributed to a primary/official source with URL. If a figure is not in those sources, it is marked **unknown** and a likely home is named. Secondary write-ups (Chips and Cheese, press recaps) are used only as pointers, never as the source of a number.

Document IDs:
- AMD “RDNA 2” Instruction Set Architecture: Reference Guide, Document ID **70648**, 2020-11-30. Official landing page: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture — PDF: https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content
- AMD “RDNA 3” Instruction Set Architecture: Reference Guide, Document ID **70650**. Official landing page: https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0
- AMD RDNA Architecture presentation (GPUOpen, 2019): https://gpuopen.com/download/RDNA_Architecture_public.pdf
- LLVM User Guide for AMDGPU Backend: https://llvm.org/docs/AMDGPUUsage.html
- LLVM `AMDGPUBaseInfo.cpp` IsaInfo: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
- ROCm HIP Hardware implementation: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- GPUOpen “Occupancy explained”: https://gpuopen.com/learn/occupancy-explained/
- Linux `amdkfd` cache tables (`kfd_crat.c`): https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html and https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_crat.c

---

## ISA Nov 2020 extract (user PDF)

Same book as 70648 / [RDNA2 Shader ISA Nov 2020](https://developer.amd.com/wp-content/resources/RDNA2_Shader_ISA_November2020.pdf). Useful for extras; already our lock:

| ISA | Inference take |
|---|---|
| Feature list: DOT **added to accelerate inferencing** — `V_DOT2_F32_F16`/`DOT2C`, `V_DOT4_I32_I8`/`DOT4C`, `V_DOT8_I32_I4` | **We fire** `fdot2` now; `sdot4`/`sdot8` after W8A8. Opcode 19: `D.f32 = s0.h0*s1.h0 + s0.h1*s1.h1 + s2` |
| No WMMA / MFMA / bf16 DOT in this book | Dead on V620 |
| Wave32 native; wave64 = two wave32 issues | HIP default wave32 |
| LDS: **128 kB/WGP**, **64 banks** × 512 × 4 B, **one WG ≤ 64 kB** | wy 58 kB is tight vs 64 kB/WG |
| WGP mode: 4 SIMD32 share one LDS; CU mode: split halves, higher LDS BW, no cross-half share | Default HIP = WGP. Do not flip CU unless LDS-bound |
| VGPR & LDS allocation-unit **doubled** vs RDNA1 | wave32 VGPR granule 16 |
| GDS 64 kB GPU-wide | Leave (we don’t use it) |
| Ray / MSAA / Add-TID | Leave |

Occupancy leftover and fat-M unchanged. DOT issue rate is **not** in this PDF (GPUOpen ops/clk).

## 0. What “gfx1030” is

LLVM names the RDNA 2 family **GCN GFX10.3** and maps:

| LLVM processor | Triple arch | Features | Example products |
|---|---|---|---|
| `gfx1030` | `amdgpu10.30` | `cumode`, `wavefrontsize64` | RX 6800, RX 6800 XT, RX 6900 XT, PRO W6800, PRO V620 |
| `gfx1031` | `amdgpu10.31` | same | RX 6700 XT |
| `gfx1032`…`gfx1036` | `amdgpu10.32`… | same | smaller dGPU / APU SKUs |

Source: LLVM AMDGPUUsage “Processors” table, https://llvm.org/docs/AMDGPUUsage.html

Generic family target: `gfx10-3-generic` / `amdgpu10.3` covers gfx1030–gfx1036 with **no ISA restrictions** listed.

HIP/ROCm offload: `--offload-arch=gfx1030` (or `gfx1031`, …). Triple: `amdgcn-amd-amdhsa--gfx1030`. Default code-gen is **wave32** and **WGP mode**; `-mwavefrontsize64` and `-mcumode` flip those (LLVM AMDGPUUsage “Target Features”).

RDNA 2 discrete parts are **monolithic graphics dies**. The CDNA/RDNA 3 term **GCD (Graphics Compute Die)** is not a RDNA 2 product construct. There is one die, one L2, one Infinity Cache, one set of GDDR6 controllers. Do not model gfx1030 as a multi-GCD part.

---

## 1. Chip hierarchy

ISA terminology (RDNA 2 ISA §1.1 Table 1, https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture):

| Block | ISA definition |
|---|---|
| **Workgroup Processor (WGP)** | “The basic unit of shader computation hardware, including scalar & vector ALU’s and memory, as well as LDS and scalar caches.” |
| **Compute Unit (CU)** | “One half of a WGP. Contains 2 SIMD32’s which share one path to memory.” |
| **Wavefront** | “A collection of 32 or 64 work-items that execute in parallel on a single RDNA processor.” |
| **Workgroup** | Wavefronts that can `s_barrier` and share LDS. |
| **SALU** | One value per wavefront; all control flow. |
| **VALU** | Per-lane VGPRs; arithmetic unique per work-item. |

So the programmable compute leaf is:

```
Die (monolithic; not a GCD pair)
 └─ Shader Engines (SE)          [count is SKU-specific; not in the ISA]
     └─ Shader Arrays (SA)       [count is SKU-specific; L1/GL1 lives here]
         └─ Workgroup Processors (WGP)
             ├─ CU 0: SIMD32 + SIMD32, shared vector-memory / texture path, 16 KB L0
             ├─ CU 1: SIMD32 + SIMD32, shared vector-memory / texture path, 16 KB L0
             ├─ LDS 128 KB (see §4)
             └─ I$ + K$ (scalar instruction + scalar data) shared by the WGP
```

HIP’s three-level description matches the *names* (SE → SA → CU) but its RDNA diagram text is slightly loose on where L1 sits; the kernel and the RDNA architecture deck put **GL1 / “L1” on the shader array**, not inside the WGP. Use the kernel + RDNA deck for cache placement (below).

### 1.1 How many SE / SA / WGP / CU?

The ISA does **not** give SE/SA counts. Those are SKU-level.

**Official product / press CU and WGP counts:**

| SKU | LLVM target | CUs (official) | WGPs (CUs/2) | Infinity Cache | Memory | Source |
|---|---|---|---|---|---|---|
| RX 6900 XT | gfx1030 | 80 | 40 | 128 MB | 16 GB GDDR6, 256-bit, up to 512 GB/s | AMD product page https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6900-xt.html ; AMD 2020-10-28 press https://www.amd.com/en/newsroom/press-releases/2020-10-28-amd-unveils-next-generation-pc-gaming-with-amd-rad.html ; GPUOpen RDNA2 table https://gpuopen.com/rdna2/ |
| RX 6800 XT | gfx1030 | 72 | 36 | 128 MB | 16 GB GDDR6, 256-bit, up to 512 GB/s | AMD product page https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800-xt.html ; same press; GPUOpen |
| RX 6800 | gfx1030 | 60 | 30 | 128 MB | 16 GB GDDR6, 256-bit | AMD 2020-10-28 press; GPUOpen |
| RX 6700 XT | gfx1031 | 40 | 20 | 96 MB | 12 GB GDDR6, 192-bit | AMD 2021-03-03 press https://www.amd.com/en/newsroom/press-releases/2021-3-3-amd-unveils-amd-radeon-rx-6700-xt-graphics-card-d.html |
| RX 6600 XT | gfx1032 (Navi 23) | 32 | 16 | 32 MB | 8 GB GDDR6, 128-bit | AMD 2021-07-29 press https://www.amd.com/en/newsroom/press-releases/2021-7-29-amd-radeon-rx-6600-xt-graphics-card-sets-new-stand.html |

GPUOpen’s RDNA 2 page lists WGPs directly: RX 6800 = 30, 6800 XT = 36, 6900 XT = 40 (https://gpuopen.com/rdna2/). That is exactly CU/2, confirming WGP = 2 CU.

**SE / SA reconstruction from AMD kernel cache tables** (not a whitepaper, but AMD-written hardware facts):

`kfd_crat.c` for Sienna Cichlid (Navi 21 / gfx1030) lists “GL1 Data Cache per SA” with `.num_cu_shared = 10` and `.cache_size = 128` (KB). So **one SA owns 10 CUs** on Navi 21.

80 CU / 10 CU per SA = **8 shader arrays** on a full Navi 21.

The RDNA 1 whitepaper (same hierarchy, smaller chip) states each shader engine contains **two shader arrays** (https://www.techpowerup.com/gpu-specs/docs/amd-rdna-whitepaper.pdf, also the GPUOpen RDNA deck). Applying that packing: 8 SA / 2 = **4 shader engines** on Navi 21. That matches the commonly published Navi 21 floorplan (4 SE × 2 SA × 5 WGP × 2 CU = 80 CU).

**rocminfo / HSA caveat.** A gfx1030 `rocminfo` dump (ROCm/HIP issue #2238, https://github.com/ROCm/HIP/issues/2238) reports `Compute Unit: 80`, `SIMDs per CU: 4`, `Shader Engines: 8`, `Shader Arrs. per Eng.: 2`, `Max Waves Per CU: 64`. Those last three fields are **GCN-era HSA fields**. On RDNA they do not match the ISA’s CU = 2×SIMD32. Treat them as:

- HSA “CU” count = ISA CU count (80) — this one is right.
- HSA “SIMDs per CU: 4” and “Max Waves Per CU: 64” describe a **WGP** (4 SIMD32 × 16 waves), not an ISA CU.
- HSA “Shader Engines: 8” is counting **shader arrays**, not marketing SEs.

Use the ISA + kernel tables, not rocminfo field names, when you reason about mapping.

### 1.2 Command / dispatch side (HIP, official)

HIP Hardware implementation (https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html):

- **CP** (command processor) = CPF fetch + CPC decode. Kernel launches go to **ACEs** (asynchronous compute engines). “Multiple ACEs enable concurrent kernel execution, with each ACE capable of dispatching one kernel at a time.” Exact ACE count on Navi 21: **not stated in that page**.
- **SPI** (shader processor input / workgroup manager) receives workgroups from ACEs, places them on CUs/WGPs, initializes SGPRs, and **keeps all waves of a workgroup on the same CU (CU mode) or WGP (WGP mode)** so LDS and `s_barrier` work.
- Workgroup-to-CU mapping is **non-deterministic**, resource-based. Do not assume a fixed SE/SA walk.

---

## 2. Compute Unit / WGP internals

### 2.1 SIMDs and ALUs

| Item | Number | Source |
|---|---|---|
| SIMD32 per CU | **2** | RDNA 2 ISA §1.1: “Contains 2 SIMD32’s which share one path to memory.” |
| SIMD32 per WGP | **4** | Two CUs. ISA §10.3: WGP mode distributes waves “across all 4 SIMD32’s.” |
| ALUs / “stream processors” per SIMD32 | **32** FP32 lanes | Implied by the SIMD32 name; RDNA architecture deck: “Each SIMD32 issues 1 instruction every cycle; Vector instruction throughput is 1 every cycle (for Wave32).” https://gpuopen.com/download/RDNA_Architecture_public.pdf |
| ALUs per CU | **64** | 2 × 32. Matches AMD marketing “64 shader cores per CU” / 80 CU → 5120 SP on RX 6900 XT (AMD 2020-10-28 press). |
| ALUs per WGP | **128** | 4 × 32 |
| SALU per CU | **1** (HIP); ISA says the WGP has “scalar & vector ALU’s” | HIP Hardware implementation “RDNA architecture” / “Dual compute units”: “Each CU has two 32-wide SIMD units.” Scalar count per CU vs per WGP is **not pinned to 1 vs 2 in the ISA text**. The RDNA deck draws one SALU next to each SIMD32 pair (i.e. per CU). Treat it as **one SALU per CU, two per WGP**. |
| Scalar SGPRs addressable | **S0–S105** (106) + VCC in S106/S107 + 16 trap temps | RDNA 2 ISA §3.6.2 and Table of registers (V0–V255, S0–S105). |

RDNA 2 ISA §2.1: hardware is **natively wave32**. Wave64 VALU/VMEM is the same instruction issued twice (low 32, then high 32). Either half may be skipped if `EXEC` for that half is 0 (except VALU ops that write an SGPR/VCC, which do not skip a pass).

### 2.2 Instruction issue

From the RDNA architecture deck (RDNA 1 WGP; RDNA 2 ISA does not change the SIMD32 issue model, and GPUOpen occupancy still describes the same SIMD):

| Fact | Number | Source |
|---|---|---|
| Issue width, wave32 | **1 VALU instruction / cycle / SIMD32** | RDNA deck: “Each SIMD32 issues 1 instruction every cycle. Vector instruction throughput is 1 every cycle (for Wave32).” https://gpuopen.com/download/RDNA_Architecture_public.pdf |
| Wave64 VALU | **2 cycles** (low then high) | RDNA 2 ISA §2.1; RDNA deck “Wave64 via dual-issue” (this is *half-wave replay*, **not** RDNA 3 VOPD). |
| VALU dest latency exposed | **5 cycles** (HW dependency check) | RDNA deck: “5 cycles of latency are exposed (automatic dependency check in hardware). Dependency stalls can be filled by other waves.” |
| Transcendentals | **¼ rate**; can **co-issue** with non-transcendental VALU | RDNA deck: “rcp/rsq/sqrt/log/exp/sin/cos … Transcendental instructions are ¼ rate (like GCN). Non-transcendental instructions can execute in parallel.” ISA opcodes: `V_EXP_F32`, `V_LOG_F32`, `V_RCP_F32`, `V_RSQ_F32`, `V_SQRT_F32`, `V_SIN_F32`, `V_COS_F32` (RDNA 2 ISA ch. 12). |
| RDNA 3-style VOPD dual-issue | **absent on RDNA 2** | VOPD is documented in RDNA 3 ISA §7.6 and LLVM `hasVOPD` / `FeatureVOPDInsts` (gfx11+). |

HIP’s generic “issue arbiter can issue five instructions per cycle (VALU + VMEM + SALU/SMEM + LDS + branch)” paragraph is written against a **CDNA CU** diagram. Do not treat “5-issue” as an RDNA 2 WGP fact; the RDNA deck only guarantees **1 VALU/cycle/SIMD32** plus SALU/VMEM/LDS as separate pipes. Exact RDNA 2 issue-group width beyond “VALU every cycle, SALU independent, VMEM independent” is **not quantified in the ISA**.

### 2.3 Wavefront size

| Mode | Lanes | How it runs | Compiler control |
|---|---|---|---|
| **wave32** (native) | 32 | One pass on one SIMD32 | LLVM default (`-mno-wavefrontsize64`) |
| **wave64** | 64 | Two passes on the same SIMD32 | `-mwavefrontsize64` / `+wavefrontsize64` |

Sources: RDNA 2 ISA §2.1; LLVM AMDGPUUsage Target Features.

HIP/HSA `rocminfo` on gfx1030 reports `Wavefront Size: 32` (https://github.com/ROCm/HIP/issues/2238). That is the **runtime default**, not a hardware-only mode.

Workgroup size: LLVM `getMaxFlatWorkGroupSize()` = **1024**. HSA dump: `Workgroup Max Size: 1024`. Same number.

### 2.4 Occupancy (waves per SIMD / per CU / per WGP)

Hardware wave slots:

| Limit | RDNA 1 (gfx1010) | RDNA 2 (gfx1030) | Source |
|---|---|---|---|
| Max waves per SIMD (“per EU”) | **20** | **16** | LLVM `IsaInfo::getMaxWavesPerEU`: `return hasGFX10_3Insts(STI) ? 16 : 20;` https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp ; GPUOpen Occupancy explained: “RDNA1, each SIMD has 20 slots … RDNA 2 and RDNA 3 have 16 slots per SIMD.” https://gpuopen.com/learn/occupancy-explained/ |
| SIMDs that share a workgroup in WGP mode | 4 | 4 | LLVM `getEUsPerCU`: 4 unless `FeatureCuMode` |
| SIMDs that share a workgroup in CU mode | 2 | 2 | same |
| Max waves per WGP (WGP mode) | 80 | **64** | 16 × 4 |
| Max waves per CU (CU mode) | 40 | **32** | 16 × 2 |
| Max work-items per WGP at wave32, full slots | — | **2048** | 64 waves × 32. Matches HSA `Max Work-item Per CU: 2048` if that “CU” is a WGP. |

SGPRs do **not** limit occupancy on GFX10+: LLVM `isSGPROccupancyLimited()` is false for ISA major ≥ 10; GPUOpen: “On RDNA GPUs however, each wavefront is assigned a fixed number of SGPRs and there are always enough of those to fill the 16 slots.”

Occupancy is then `min(wave slots, VGPR budget, LDS budget, workgroup-size / barrier slots)`.

VGPR occupancy math (LLVM `getNumWavesPerEUWithNumVGPRs` + `getTotalNumVGPRs` + `getVGPRAllocGranule`):

| | wave32 | wave64 | Source |
|---|---|---|---|
| Physical VGPR file (compiler’s “total”) | **1024** | **512** (same SRAM; wave64 counts double) | LLVM `getTotalNumVGPRs`: GFX10+ without `Feature1536VGPRs` → 1024 / 512. RDNA deck: “Each SIMD32 has 1024 physical registers. Wave64 counts double.” |
| Addressable VGPRs per lane | **256** (`V0–V255`) | **256** | RDNA 2 ISA register table; LLVM `getAddressableNumArchVGPRs` = 256 (no `Feature1024AddressableVGPRs` on gfx1030) |
| Allocation granule | **16** VGPRs | **8** VGPRs | RDNA 2 ISA §3.6.4: “VGPRs are allocated in groups of 8 Dwords for wave64, and 16 Dwords for wave32.” LLVM `getVGPRAllocGranule` for `hasGFX10_3Insts`: 16 / 8. (RDNA 2 feature note: “VGPR & LDS allocation-unit size doubled” vs RDNA 1.) |
| Waves at 64 VGPR (wave32) / 32 VGPR (wave64) | **16** (max) | **16** (max) | 1024/64 = 16; 512/32 = 16 |
| Waves at 256 VGPR | **4** | **2** | 1024/256 = 4; 512/256 = 2 |

RDNA deck examples (still valid for the 1024-register file; RDNA 2 only cuts the *slot* cap from 20 → 16):

- 4× wave32 × 256 VGPR
- 2× wave64 × 256 VGPR
- 16× wave32 × 64 VGPR
- 8× wave64 × 64 VGPR  ← on RDNA 2 you can actually go to 16× wave64 × 32 VGPR

Physical size of the file, if you want bytes: 1024 VGPR × 32 bits × 32 lanes = **128 KiB per SIMD32**. That arithmetic is just the product of the official 1024 × 32-bit × SIMD32; the ISA does not print “128 KB”.

---

## 3. Execution model

### 3.1 Workgroup → WGP/CU

1. Host AQL/HIP dispatch hits CP → ACE → SPI.
2. SPI allocates **wave slots + VGPRs + SGPRs + LDS** on one WGP (WGP mode) or one CU (CU mode). If any resource is short, the workgroup waits.
3. All waves of the workgroup stay on that WGP/CU so `s_barrier` and LDS are valid (HIP Hardware implementation; RDNA 2 ISA §2.3.1 / §10.3).
4. Mapping is **not** a fixed SE-major walk. Re-launches can land differently (HIP).

**WGP mode vs CU mode** (RDNA 2 ISA §2.3.1 and §10.3):

| | CU mode (`-mcumode`) | WGP mode (LLVM/HIP default) |
|---|---|---|
| SIMDs used by one workgroup | 2 (one CU) | 4 (whole WGP) |
| LDS visible | 64 KB half attached to that CU | full 128 KB |
| Why use it | “higher LDS memory bandwidth” (ISA §10.3); both LDS halves run in parallel | more ALU + texture bandwidth for a workgroup of ≥ 4 waves |
| Workgroup LDS cap | still **64 KB per workgroup** | still **64 KB per workgroup** |

A single workgroup **cannot** allocate more than 64 KB even in WGP mode (ISA §2.3.1: “A single workgroup may allocate up to 64kB of LDS space.”). The extra 64 KB is so **two** workgroups (or CU-mode halves) can each hold 64 KB.

LLVM `getLocalMemorySize`: addressable 64 KB, **doubled to 128 KB in WGP mode** (`BytesPerCU *= 2` when GFX10+ and not `FeatureCuMode`).

Barrier slots: LLVM `getMaxWorkGroupsPerCU` uses **16 barriers in CU mode, 32 in WGP mode** on GFX10+. GPUOpen Occupancy explained: “The GPU can track up to 16 barriers in flight per pair of SIMDs.” Those two statements are consistent if “pair of SIMDs” = one CU (16) and a WGP has two pairs (32).

### 3.2 Wave scheduling

- A wave is bound to **one SIMD32** for its life (RDNA deck: “Each wave is assigned to one SIMD32”).
- Only one wave executes VALU on that SIMD in a given cycle; the SIMD switches waves with **zero context-switch cost** because VGPR/SGPR/LDS are pre-reserved (GPUOpen Occupancy explained; HIP “single-cycle context switching”).
- Purpose: hide VMEM/LDS/export latency. Occupancy is the *capacity* to hide latency, not a performance guarantee (GPUOpen: memory-bound kernels can lose from extra waves via cache thrash).
- VALU 5-cycle dest latency is hidden by other waves or by ILP in the same wave (RDNA deck).

### 3.3 VGPR / SGPR files and occupancy vs pressure

**VGPR**

- Per-lane 32-bit; pairs have no alignment requirement (ISA §3.6.4).
- Out of range (`vgpr >= vgpr_size`): src reads VGPR0, dst is ignored (ISA §3.6).
- Wave64 may also allocate **wave-shared VGPRs** in blocks of 16, sitting just above private VGPRs, shared by the two halves. Not usable for export/GDS (ISA §3.6.5). Rare in HIP compute.
- Compiler occupancy is `min(16, floor(total / align(used, granule)))`. A single hot VGPR spike sets the allocation for the whole kernel (GPUOpen / RGA).
- Spilling: compiler can evict VGPRs to scratch. RGP pipeline tab reports it. Scratch is **dword-interleaved** across lanes (LLVM AMDGPUUsage Private address space).

**SGPR**

- Fixed **106 + VCC pair** per wave (ISA §3.6.2). Alignment: even for 64-bit / SMEM base; quad for SMEM returns ≥ 4 dwords.
- VCC lives in the SGPR file and **counts against the number of SGPRs an instruction may source** (ISA §3.9 note).
- Do not spend time shrinking SGPRs for occupancy on gfx1030; they are not the limiter (LLVM + GPUOpen).

**LDS occupancy**

- Allocated in **256-dword (1 KB) blocks, 256-dword aligned**, no wrap (ISA §3.6.6).
- Workgroups of 1 wave do not consume a barrier; multi-wave workgroups do (LLVM `getMaxWorkGroupsPerCU`).
- Example: 64 KB LDS / workgroup → **1 workgroup per CU-half**, **2 per WGP**. Combined with 256-thread workgroups at wave32 (8 waves) that is 8 or 16 waves — often the real cap on attention/softmax scratchpads.

---

## 4. Memory hierarchy

### 4.1 Register files

| Resource | Size | Notes | Source |
|---|---|---|---|
| VGPR file per SIMD32 | 1024 × 32-bit × 32 lanes (= 128 KiB) | Addressable 256/lane; occupancy vs pressure as §2.4 | LLVM `getTotalNumVGPRs`; RDNA deck; ISA V0–V255 |
| SGPR file per wave | 106 + VCC + 16 trap | Fixed; not an occupancy knob | ISA §3.6.2 |
| AGPR / accum VGPRs | **none** | CDNA-only (`v_accvgpr_*`, `FeatureGFX90AInsts`) | LLVM; HIP MFMA section is CDNA |

### 4.2 LDS (Local Data Share)

All of the following is RDNA 2 ISA §2.3.1 and Chapter 10 (https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture):

| Item | Value |
|---|---|
| Capacity per WGP | **128 KB** |
| Banks | **64** banks, each **512 entries × 4 bytes** (64 × 512 × 4 = 131072) |
| Bank split | 64 banks = two sets of **32 banks**, each set affiliated with one CU (one pair of SIMD32s) |
| Bank RAM | **512×32, 1R/1W per clock** |
| Atomic units | **64** integer atomic units |
| Max allocation per workgroup | **64 KB** |
| Allocation granule | **256 dwords (1024 B)**, 256-dword aligned |
| Conflict-free indexed/atomic | “as little as **one cycle** (wave32) or **2 cycles** (wave64)” |
| Worst conflict | “as many as **64 cycles**” depending on bank conflicts |
| Conflict rule | Multiple lanes to **different addresses in the same bank** are **serialized**. Same-address broadcast is the non-conflict case (HIP states this explicitly; ISA says hardware serializes same-bank indexed/atomic). |
| Peak concurrent | “concurrently execute **32** write or read instructions, each nominally 32-bits; `read2`/`write2` can be 64-bits each.” |
| Bandwidth wording (HIP) | 64 banks × 4 B = **256 B/cycle** aggregate on RDNA 2/3/4; SIMDs connect in pairs sharing a **64-byte bidirectional port** (https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) |
| CU vs WGP | CU mode: each half is local → “higher LDS memory bandwidth.” WGP mode: waves may hit the “far” 32-bank half → “performance may be lower in some cases.” |

Bank index (not printed as a formula in the ISA, but the layout is): dwords are “placed in the banks serially.” So for 32-bit elements,

`bank = (byte_address / 4) % 64`

in WGP mode. In CU mode you only see 32 banks on your half — the same sequential mapping restricted to that half. **Confirm with a microbench** if you need the exact CU-mode alias; the ISA does not write the modulo.

GDS (rarely used from HIP): **64 KB**, **32 banks × 512 × 4 B**, **128 B/cycle**, 32 atomics, shared by all WGPs (ISA §2.3.2). LLVM: GDS is **not implemented for AMDHSA**. Ignore GDS for ROCm kernels.

### 4.3 L0 / L1 / I$ / K$

Two independent official pictures, consistent on sizes:

**A. RDNA architecture deck** (RX 5700 XT / RDNA 1 WGP; RDNA 2 keeps these first-level sizes — confirmed by the kernel table below): https://gpuopen.com/download/RDNA_Architecture_public.pdf

| Cache | Size | Line | R/W | Scope |
|---|---|---|---|---|
| Instruction cache (I$) | **32 KB per WGP** | 64 B | RO | WGP (~2 CU) |
| Scalar cache (K$) | **16 KB per WGP** | 64 B | RO | WGP |
| L0 (vector) | **2 × 16 KB per WGP** | 128 B | RO | per CU |
| L1 | **128 KB per shader array** | 128 B | RO | SA |
| L2 (RDNA 1 Navi10) | 4 MB | 128 B | R/W | GPU |

**B. AMD `kfd_crat.c` for Sienna Cichlid (Navi 21)** — AMD engineer patch “Update L1 and add L2/3 cache information”: https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html

| Entry (comment in source) | Size (KB) | Level | Line | Shared by |
|---|---|---|---|---|
| TCP L1 Cache per CU (this is the **vector L0**) | **16** | 1 | 128 B | 1 CU |
| Scalar L1 Instruction Cache per SQC (I$) | **32** | 1 | 64 B | 2 CU (one WGP) |
| Scalar L1 Data Cache per SQC (K$) | **16** | 1 | 64 B | 2 CU |
| GL1 Data Cache per SA (graphics/compute L1) | **128** | 1 | 128 B | 10 CU (one SA on Navi 21) |
| L2 Data Cache per GPU | **4096** (4 MB) | 2 | 128 B | whole GPU |
| L3 Data Cache per GPU (Infinity Cache) | **128×1024** (128 MB) | 3 | **64 B** | whole GPU |

Navy Flounder (Navi 22): L2 **3072 KB**, L3 **96 MB**. Dimgrey Cavefish (Navi 23): L2 **2048 KB**, L3 **32 MB**. Beige Goby (Navi 24) is in later tables as L2 **1 MB**, L3 **16 MB** (same file family; treat Navi 24 L3 as kernel-confirmed, product-page confirmation not opened in this pass).

HIP: vector L1/L0 is **write-through** to L2, “typical size of 16 KB per CU”, software-managed coherence between CUs. Scalar L1 and I$ are **not** hit-on-miss (duplicate pending fills count as misses). L2 **is** hit-on-miss.

**Associativity:** **unknown** in the ISA, the RDNA deck, HIP, and `kfd_crat.c`. Do not invent 16-way. If you need it, it lives in an unreleased block diagram or a microbenchmark (Chips and Cheese reports 16-way L1/L2 on RDNA 2; that is not an AMD doc).

ISA §2.4 read path: “multiple channels of L2 cache that provides data to Read-only L1 caches, and finally to L0 caches per WGP.” That is the official topology: **L0 (per CU/WGP) → L1 (per SA) → L2 (GPU) → (Infinity Cache) → GDDR6**.

### 4.4 L2

| SKU / die | L2 | Line | Source |
|---|---|---|---|
| Navi 21 (gfx1030, 6800/6800 XT/6900 XT) | **4 MB** | 128 B | `kfd_crat.c` Sienna Cichlid `.cache_size = 4096` |
| Navi 22 (gfx1031, 6700 XT) | **3 MB** | 128 B | same, Navy Flounder 3072 |
| Navi 23 (6600 XT) | **2 MB** | 128 B | Dimgrey Cavefish |
| Navi 24 (6500 XT class) | **1 MB** | 128 B | Beige Goby table (kernel) |
| RDNA 1 Navi10 (for comparison) | 4 MB | 128 B | RDNA deck |

L2 is the **coherence point**. HIP: write-through L1, atomics execute at L2, software fences + cache invalidation between CUs. Channel count / interleave on Navi 21: **not in the ISA**. HIP’s “32 channels, 256-byte interleave” sentence is in a **CDNA** paragraph — do not reuse it for Navi 21.

### 4.5 Infinity Cache (MALL / L3)

What AMD says it is:

- “An all-new additional cache level … This **global cache is seen by the entire graphics core**, capturing ‘Temporal Reuse’ … enabling data to be accessed virtually instantaneously.” AMD RDNA 2 Explained (Radeon PRO W6000), https://www.amd.com/content/dam/amd/en/documents/products/graphics/workstation/rdna2-explained-radeon-pro-W6000.pdf
- “A high-performance, **last-level data cache** … **128 MB of on-die cache** dramatically reduces latency and power consumption.” AMD 2020-10-28 RX 6000 press release.
- 6700 XT: “**96MB of last-level data cache on the GPU die** provides up to 2.5X higher bandwidth at the same power level as traditional architectures.” AMD 2021-03-03 press.
- 6600 XT: “**32 MB of last-level data cache integrated on the GPU die** reduces latency and power consumption.” AMD 2021-07-29 press.

Where it sits: **after L2, before GDDR6**, on the same monolithic die (RDNA 2). Kernel programs it as **cache_level = 3**. Internally AMD and the driver also call it **MALL** (memory-attached last level); that name is in driver/community material, not in the ISA.

Line size: **64 B** (`kfd_crat.c` L3 `.cache_line_size = 64`), unlike L0/L1/L2’s 128 B. That is a real programming detail: an Infinity Cache miss/fill is a 64 B transaction, not 128 B.

Who it helps:

- Working sets that **miss L2 (4 MB on Navi 21) but hit 32–128 MB**: tiled GEMM panels, KV-cache chunks, decoder residual streams, weight tiles reused across batches.
- It is a **bandwidth amplifier** for a relatively narrow GDDR6 bus (256-bit on Navi 21 vs the 512-bit bus AMD said they were trying to avoid — AMD architect quote is in secondary coverage; the official claim is the 2.4× bandwidth-per-watt vs “GDDR6-only RDNA” in the 2020 press).
- It does **not** help a cold streaming load larger than the IC with no reuse (token-by-token decode of a huge KV cache that thrashes 128 MB will still go to GDDR6).
- It is **shared by the whole GPU**, so two concurrent kernels fight for it.

**Internal IC bandwidth (TB/s):** **not published in the ISA or the product pages opened here.** Marketing “up to 2.4×” / “2.5×” is relative, not an absolute GB/s. Do not quote 1.8–2.0 TB/s unless you measure it or find it in an AMD Hot Chips slide (not opened in this pass).

### 4.6 HBM / GDDR and memory controllers

RDNA 2 discrete = **GDDR6**, not HBM. HBM is CDNA/Instinct.

| SKU | Bus | Pin rate | Peak BW | Source |
|---|---|---|---|---|
| RX 6900 XT / 6800 XT / 6800 | **256-bit** | up to **16 Gbps** | **up to 512 GB/s** | AMD RX 6900 XT product page |
| RX 6700 XT | **192-bit** | (press does not print Gbps; 12 GB GDDR6) | **unknown in the press table** | AMD 2021-03-03 press |
| RX 6600 XT | **128-bit** | up to **16 Gbps** | 128-bit × 16 Gbps = 256 GB/s (arithmetic; AMD page lists interface 128-bit, 16 Gbps, not the product) | AMD 2021-07-29 press + product page |

Controller count / channels: **unknown** in ISA and product pages. Topology is “memory controller ↔ L2 ↔ (IC) ↔ GDDR6.” ISA §1: the RDNA memory controller is a DMA to device + host-visible memory.

PCIe: RX 6000 press: **PCIe 4.0**. HIP: typically two SDMA engines for host copies; Navi 24 is reported in kernel comments as 1 SDMA — verify per SKU if you care about copy overlap.

---

## 5. Special units (compute-relevant)

### 5.1 Matrix / WMMA / MFMA — be precise

**RDNA 2 has no MFMA and no WMMA.**

- CDNA MFMA (`v_mfma_*`, AGPRs, matrix cores) is a CDNA feature (`FeatureMAIInsts`). HIP’s MFMA chapter is explicitly “CDNA architectures (MI100 and newer).”
- RDNA 3 WMMA (`v_wmma_*`, 16×16×16, wave-cooperative) is documented in RDNA 3 ISA §7.9 and GPUOpen “How to accelerate AI applications on RDNA 3 using WMMA” (https://gpuopen.com/learn/wmma_on_rdna3/). That article’s comparison table is the official “RDNA 2 vs 3 matrix” statement:

| Type | RX 6950 XT (RDNA 2) FLOPS/clock/CU | RX 7900 XTX (RDNA 3) FLOPS/clock/CU |
|---|---|---|
| FP16 | **256** | **512** |
| BF16 | **N/A** | **512** |
| IU8 | **512** | **512** |
| IU4 | **1024** | **1024** |

Source: https://gpuopen.com/learn/wmma_on_rdna3/

What RDNA 2 *does* have for ML (RDNA 2 ISA “Feature Changes in RDNA2 Devices”):

- `V_DOT2_F32_F16` / `V_DOT2C_F32_F16`
- `V_DOT2_I32_I16` / `V_DOT2_U32_U16`
- `V_DOT4_I32_I8` / `V_DOT4C_I32_I8` / `V_DOT4_U32_U8`
- `V_DOT8_I32_I4` / `V_DOT8_U32_U4`

These are **per-lane packed DOT** ops, not wave-cooperative matrix tiles. LLVM exposes them as `llvm.amdgcn.sdot2/udot2/sdot4/…`. Useful for INT8/INT4 / packed-F16 inner products; they are **not** a substitute for MFMA/WMMA tiling.

Packed F16 math in general is **2× FP32 rate** on the VALU (256 F16 FLOPS/clock/CU in the table above = 2 × 128 FP32 FMA). Arithmetic: each CU has 2×SIMD32 = 64 FP32 ALUs; one FMA per ALU per clock is 128 FP32 FLOPS/clock/CU. A WGP (2 CU) is therefore 256 FP32 FLOPS/clock. That matches GPUOpen’s per-CU convention.

### 5.2 Texture / image / ray

- Each CU has a **texture addresser + texture data path** sharing the vector-memory pipe (RDNA deck WGP diagram; ISA: CU’s 2 SIMD32s “share one path to memory”).
- RDNA deck (RDNA 1, inherited): loads **32 addresses/clk** and **32 dwords/clk**; stores **32 addresses / 2 clk**. Prefer `Load` over point-sample (separate low-latency load path).
- **Ray tracing:** RDNA 2 ISA feature list “Ray Tracing”; image instructions include BVH (`image_bvh_*` in ch. 8.2.10). One **Ray Accelerator per CU** on the product pages (80 RA on RX 6900 XT). Intersection rate (4 box / 1 tri per cycle) is **not in the ISA**; it is in secondary microarch write-ups. Do not use RT hardware for GEMM.
- **Export / primitive / RB:** graphics. Compute kernels do not export. Skip unless you are writing a pixel shader.

### 5.3 Other

- **DPP / permute:** `v_permlane16`, `v_permlanex16`, DPP mov — wave-internal shuffles without LDS. Preferred for reductions when the data is already in VGPRs (LLVM intrinsics).
- **SMEM:** scalar cache path for kernel args, descriptors, uniform constants. Put `__constant__` / kernarg traffic here; do not blow I$ with giant unrolled epilogues (RDNA deck: “mind the I$ size”).
- **Separate VMCNT / VSCNT** (RDNA): stores are fire-and-forget; `s_waitcnt vmcnt(0)` no longer waits for stores. `s_waitcnt vscnt(0)` before barrier/atomics (RDNA deck).

---

## 6. RDNA 2 (gfx1030) vs RDNA 3 (gfx1100) — short

| Topic | RDNA 2 gfx1030 | RDNA 3 gfx1100 | Source |
|---|---|---|---|
| CU / WGP shape | CU = 2×SIMD32; WGP = 2 CU | Same WGP dual-CU | HIP RDNA3 WGP diagram; RDNA 3 ISA |
| Wave slots / SIMD | **16** | **16** | LLVM `getMaxWavesPerEU`; GPUOpen occupancy |
| VALU issue | 1 instr/cycle/SIMD32 | **VOPD dual-issue** of selected ops (`v_dual_*`) in wave32; wave64 can start both halves in one cycle if dual-issuable | RDNA 3 ISA §7.6; LLVM `FeatureVOPDInsts`; ROCm rocprofiler-compute “Dual-issue VALU (VOPD)” https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/rdna/system-speed-of-light.html : FP32 128 → **256** FLOPS/CU/cycle when VOPD fires |
| Matrix | Packed DOT only; **no WMMA, no MFMA** | **WMMA 16×16×16** FP16/BF16/IU8/IU4, wave-cooperative | GPUOpen WMMA article; RDNA 3 ISA §7.9 |
| FP16 peak / CU / clock | **256** | **512** (WMMA + dual-issue / packed) | GPUOpen WMMA table |
| BF16 | **N/A** as a first-class matrix type | **512** FLOPS/clock/CU | same |
| VGPR file | 1024/SIMD (wave32) | 1024 on most; **1536** on some SKUs (`Feature1536VGPRs`, granule 24/12) | LLVM `getTotalNumVGPRs` |
| L0 vector | 16 KB / CU | **32 KB / CU** reported in RDNA 3 architecture coverage; **not independently confirmed in an AMD PDF opened this pass** — treat as “commonly stated, verify in RDNA 3 whitepaper / kfd_crat for Navi 31” | kernel tables for gfx11 not opened here |
| L1 / SA | 128 KB | **256 KB** (same caveat) | same |
| L2 | 4 MB (Navi 21) | **6 MB** (Navi 31 class) | Navi 21: `kfd_crat`; Navi 31 6 MB is widely published, official product pages do not always print L2 |
| Infinity Cache | 128 / 96 / 32 / 16 MB **on-die** | up to **96 MB on MCDs** (chiplet), higher latency | AMD product/press vs RDNA 3 chiplet articles |
| Die | Monolithic | Navi 31 = **GCD + MCDs** (this is where “GCD” becomes real) | RDNA 3 architecture coverage |
| Flat scratch | Absolute | **Architected** flat scratch; packed work-item IDs | LLVM Processors table |
| Occupancy math | SGPR-unlimited; VGPR granule 16/8 | same 16-wave cap; larger file on 1536-VGPR parts | LLVM |

For LLM kernels: gfx1100 is the first Radeon generation where **WMMA** is the correct GEMM inner loop. On gfx1030 you write **VALU FMA / packed F16 / DOT** plus LDS tiling. Do not expect rocWMMA MFMA paths to light up on gfx1030.

---

## 7. Official docs to keep

| Doc | What it is | URL |
|---|---|---|
| **“RDNA 2” Instruction Set Architecture: Reference Guide (70648)** | The ISA. Hierarchy, wave32/64, LDS banks, VGPR/SGPR, DOT, RT image ops. | https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture — PDF https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content — GPUOpen announcement https://gpuopen.com/news/rdna2-isa-available/ |
| **“RDNA 3” Instruction Set Architecture: Reference Guide (70650)** | VOPD §7.6, WMMA §7.9 | https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0 |
| **RDNA Architecture presentation** | WGP vs GCN CU, 1024 VGPR, I$/K$/L0/L1/L2 table, issue/latency, LDS 128 KB / 32-dword peak (RDNA 1 numbers; RDNA 2 ISA updates LDS to 64 banks) | https://gpuopen.com/download/RDNA_Architecture_public.pdf |
| **RDNA 2 Explained (Radeon PRO W6000)** | Infinity Cache as global last-level cache | https://www.amd.com/content/dam/amd/en/documents/products/graphics/workstation/rdna2-explained-radeon-pro-W6000.pdf |
| **LLVM AMDGPUUsage** | Targets `gfx1030`/`gfx1100`, triples, `cumode` / `wavefrontsize64`, address spaces, ABI | https://llvm.org/docs/AMDGPUUsage.html |
| **LLVM IsaInfo** | Occupancy, VGPR totals, LDS 64/128 KB, waves/EU = 16 | https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp |
| **LLVM gfx1030 asm** | Instruction syntax | https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.0.1/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html |
| **HIP Hardware implementation** | SPI, ACE, LDS banks (64 on RDNA 2), WGP/Wave32 | https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html |
| **GPUOpen Occupancy explained** | 16 slots, SGPR not limiting, how to read RGP | https://gpuopen.com/learn/occupancy-explained/ |
| **GPUOpen WMMA on RDNA 3** | Official “no WMMA on RDNA 2” FLOPS table | https://gpuopen.com/learn/wmma_on_rdna3/ |
| **`llvm-calc-occupancy`** | Same math as the backend | https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html |
| **amdkfd `kfd_crat.c`** | L0/I$/K$/GL1/L2/L3 sizes per Navi 2x | https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html |

**HIP/ROCm target names**

| Use | String |
|---|---|
| Offload | `--offload-arch=gfx1030` (RDNA 2 Navi 21), `gfx1031` (Navi 22), `gfx1100` (RDNA 3 Navi 31) |
| Triple | `amdgcn-amd-amdhsa` or `amdgpu-amd-amdhsa` |
| Generic | `gfx10-3-generic`, `gfx11-generic` |
| Features | `-mcumode` / `-mno-cumode`; `-mwavefrontsize64` / `-mno-wavefrontsize64` |
| Occupancy hint | `__attribute__((amdgpu_waves_per_eu(min,max)))` → LLVM `amdgpu-waves-per-eu` |

---

## 8. What this means for a kernel writer

### LDS banks

- You have **64 banks × 4 B** on the WGP (ISA). A wave32 of consecutive `float` at a 4-byte stride is `lane % 64` — 32 lanes cannot fill 64 banks, so a single wave32 of 4-byte sequential access is **conflict-free** and uses half the banks.
- `float4` / 16-byte elements: two or four banks per lane. Still fine if the *bank set* does not collide. The ISA’s conflict-free peak is “32 × 32-bit or read2/write2 64-bit.”
- **Stride 32 dwords** (128 B) on 64 banks: `bank = (addr/4) % 64`, so stride 32 dwords → 2-way conflicts; stride 64 dwords → 32-way disaster. Pad shared tiles so the leading dimension is **not** a multiple of 64 dwords if you walk that dimension with consecutive lanes.
- Default HIP is **WGP mode**: a workgroup’s waves sit on all 4 SIMD32s and may touch both 32-bank halves. Cross-half LDS is legal but the ISA warns it can be slower. If a kernel is LDS-bandwidth bound and the workgroup fits on 2 SIMD32s, try **`-mcumode`** (or the HIP/LLVM equivalent) and measure.
- 64 KB / workgroup hard cap. Two 64 KB workgroups per WGP is the LDS ceiling. Attention scratch that wants 96 KB in one block **will not fit**; split the tile or spill to L0/L2/IC.

### Occupancy

- Budget **64 VGPR/lane (wave32)** or **32 VGPR/lane (wave64)** for 16 waves/SIMD. That is tight for an LLM GEMM+softmax fused kernel. 128 VGPR → 8 waves; 256 VGPR → 4 (wave32) or 2 (wave64).
- Prefer **wave32** unless you have a measured reason (wave64 can help interpolation-class shaders and some wide reductions; it doubles VGPR cost per wave and issues VALU in two beats).
- Occupancy is necessary for latency hiding, **not sufficient**. GPUOpen’s warning applies directly to decode: extra waves on a memory-bound KV load will thrash L0/L1/IC.
- SGPRs are free for occupancy. Use them for descriptors, strides, and uniform pointers.
- Check spill with RGP / `llvm-objdump` / RGA. A spill on gfx1030 is a scratch hit through L0/L2/IC — fatal in an inner loop.

### Cache-aware tiling

```
registers  →  LDS (128 KB/WGP, 64 KB/WG)  →  L0 16 KB/CU  →  L1 128 KB/SA  →  L2 4 MB  →  IC 128 MB  →  GDDR6 512 GB/s
```

- **L0 16 KB** is tiny. A 128×128 FP16 tile is 32 KB already. Do not expect L0 to hold a GEMM panel; it is a streaming filter. Coalesce to **128 B** lines (32×FP32 or 64×FP16 per line). ISA/RDNA deck: 128 B L0/L1/L2 lines.
- **L1 128 KB/SA** is shared by **10 CUs** on Navi 21. Ten CUs fighting one 128 KB L1 means L1 is a *hot* shared resource, not your private cache. Tile so neighboring CUs in an SA reuse the same K-panel.
- **L2 4 MB** is the first GPU-wide reusable store. A 4096-d model’s FP16 weight row is 8 KB; 4 MB holds 512 such rows. Good for a working set of active layers, not the whole model.
- **Infinity Cache 128 MB (Navi 21)** is the interesting one for LLM:
  - 128 MB holds **64M FP16** elements ≈ a **4096×8192** FP16 matrix, or a **batch × seq × d** KV slice of that size.
  - Prefetch / persist weights or the hot KV window so L2+IC absorb reuse; stream the rest.
  - IC line is **64 B** — slightly different packing than L2’s 128 B. Align persistent buffers to at least 64 B; 128 B still satisfies both.
  - On Navi 22/23/24 the IC is 96/32/16 MB. A kernel tuned to “the whole layer fits in 128 MB” will fall out of cache on a 6600 XT. Query `hipDeviceGetAttribute` / rocminfo L3 and tile accordingly.
- **GDDR6 512 GB/s** (Navi 21) is the floor when IC misses. A 7B FP16 model (14 GB) does not fit; weight-streaming decode is IC-miss bound unless you quantize or layer-prefetch.
- Coalescing: wave32 × 4 B = 128 B = one L0/L1/L2 line. That is the native happy path. Wave64 × 4 B = 256 B = two lines. Structure-of-arrays, consecutive `threadIdx.x` on consecutive elements.

### GEMM / attention on gfx1030 (no WMMA)

- Inner kernel: `v_fma_f32` / packed `f16` / `V_DOT2_F32_F16` / `V_DOT4_*`. There is **no** `v_wmma_*` and **no** `v_mfma_*`.
- Tile in LDS with 64-bank-aware padding; keep K-unroll high enough to cover 5-cycle VALU dest latency plus VMEM.
- Use SMEM for the kernel descriptor and problem sizes; keep the I$ working set small (RDNA deck: ILP by unrolling is good, code bloat is not).
- `s_barrier` is WGP/CU scoped. Do not invent grid sync in LDS.
- Separate `vmcnt` and `vscnt`. Do not `s_waitcnt vmcnt(0)` when you only need loads; you will pin stores for no reason (RDNA deck).
- Launch enough waves to fill **40 WGP × 4 SIMD × 16 slots = 2560 wave32** on a 6900 XT only if the working set still hits IC. For a 256-thread workgroup that is 320 workgroups. Smaller grids leave SIMDs idle (GPUOpen occupancy).

### Defaults to set in the build

```
hipcc --offload-arch=gfx1030                 # Navi 21
# hipcc --offload-arch=gfx1031               # Navi 22
# optional experiments, measure both:
#   -mwavefrontsize64
#   -mcumode
```

Use `llvm-calc-occupancy -mcpu=gfx1030 --wg-size=256 --vgprs=N --lds=K` before arguing about occupancy.

---

## 9. Unknowns (do not invent)

| Question | Status | Where it might live |
|---|---|---|
| L0/L1/L2/IC associativity | Unknown in ISA, HIP, kfd_crat, RDNA deck | Unreleased block diagram; microbenchmark |
| L2 channel count / interleave on Navi 21 | Unknown (HIP’s 32×256 B is CDNA text) | `amdgpu` TCC docs, Hot Chips |
| Infinity Cache absolute TB/s | Not in ISA or product pages opened | Hot Chips / ISSCC; measure |
| Exact ACE count, SPI width | HIP says “multiple ACEs”, no number | Hardware whitepaper |
| SALU count officially 1 vs 2 per WGP | Deck draws one per CU | ISA wording is “scalar & vector ALU’s” on the WGP |
| Navi 21 SE count in an AMD PDF | Inferred 4 SE × 2 SA from kernel `num_cu_shared=10` + RDNA 1 packing | Navi 21 floorplan / Hot Chips |
| RDNA 2 Hot Chips / ISSCC paper | Not opened this pass | Search “AMD Navi 21 Hot Chips 32” |
| Navi 24 L3 16 MB on an AMD.com product page | Kernel yes; AMD.com page not successfully fetched this pass | RX 6500 XT product page |
| RDNA 3 L0=32 KB / L1=256 KB / L2=6 MB in an AMD PDF | Widely reported; not pulled from an AMD PDF here | RDNA 3 whitepaper / kfd_crat Navi 31 |

---

## 10. Primary URLs actually opened

1. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture — ISA landing page (70648)
2. https://docs.amd.com/api/khub/documents/Et~wpu9g~Ffl7d9q0QZ~Og/content — ISA PDF (downloaded, `pdftotext`)
3. https://gpuopen.com/news/rdna2-isa-available/ — ISA announcement
4. https://gpuopen.com/rdna2/ — SKU / WGP / IC table
5. https://gpuopen.com/download/RDNA_Architecture_public.pdf — RDNA WGP, VGPR, cache table, issue
6. https://gpuopen.com/learn/occupancy-explained/ — 16 slots, SGPR, limiters
7. https://gpuopen.com/learn/wmma_on_rdna3/ — WMMA; RDNA 2 vs 3 FLOPS table
8. https://llvm.org/docs/AMDGPUUsage.html — processors, features, address spaces
9. https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp — occupancy / VGPR / LDS / waves
10. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
11. https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html — SPI, LDS banks, RDNA WGP
12. https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.0.1/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX1030.html
13. https://www.amd.com/content/dam/amd/en/documents/products/graphics/workstation/rdna2-explained-radeon-pro-W6000.pdf — Infinity Cache description
14. https://www.amd.com/en/newsroom/press-releases/2020-10-28-amd-unveils-next-generation-pc-gaming-with-amd-rad.html — 80/72/60 CU, 128 MB IC, 256-bit, 512 GB/s
15. https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6900-xt.html
16. https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6800-xt.html
17. https://www.amd.com/en/newsroom/press-releases/2021-3-3-amd-unveils-amd-radeon-rx-6700-xt-graphics-card-d.html — 40 CU, 96 MB IC, 192-bit
18. https://www.amd.com/en/newsroom/press-releases/2021-7-29-amd-radeon-rx-6600-xt-graphics-card-sets-new-stand.html — 32 CU, 32 MB IC, 128-bit
19. https://lists.freedesktop.org/archives/amd-gfx/2021-March/061392.html — kfd_crat L0/I$/K$/GL1/L2/L3
20. https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_crat.c
21. https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0 — RDNA 3 ISA landing
22. https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/rdna/system-speed-of-light.html — VOPD peaks
23. https://github.com/ROCm/HIP/issues/2238 — gfx1030 rocminfo dump (HSA fields; use with the caveat in §1.1)
24. https://github.com/llvm/llvm-project/commit/03663e4130d700c6c8ea28b357fcac4d31b617f7 — gfx1030 occupancy 16

ISA PDF text extract used for quotations: `/workspace/rdna2-src/rdna2-isa.txt` (from the official 70648 PDF).
