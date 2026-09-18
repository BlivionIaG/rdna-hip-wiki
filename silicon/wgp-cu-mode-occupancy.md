# WGP vs CU mode occupancy (gfx1030)

Lock: **launch mode (`-mno-cumode` WGP vs `-mcumode` CU) is not a PIX limiter row — it rescales the LDS pool, barrier budget, and SIMDs-per-WG that feed those rows.** On gfx1030 HIP/LLVM default is **WGP** (`FeatureCuMode` off → `-mno-cumode`). Flip to CU and one workgroup sits on **2** SIMD32s with a **64 KB** LDS half and **16** barriers; WGP keeps **4** SIMD32s, **128 KB** shared pool, **32** barriers. Per-WG LDS addressable cap stays **64 KB** either way. Completes the occupancy set as the **launch-mode axis** — sibling of [wave-size-occupancy.md](wave-size-occupancy.md); VGPR / LDS / WG / barrier pages still own their terms.

Does **not** change extras HIP or tickets. No tok/s. Do not restate the FA pin.

Companions: [lds-occupancy.md](lds-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [wave-size-occupancy.md](wave-size-occupancy.md), [wg-size-occupancy.md](wg-size-occupancy.md), [occupancy-composite.md](occupancy-composite.md), [occupancy-dump.md](occupancy-dump.md), [lds-bank-occupancy.md](lds-bank-occupancy.md), [hip-craft.md](hip-craft.md) §1 / §1.2, [architecture.md](architecture.md) §3.1 / §4.2.

## Take / Leave

| | |
|---|---|
| **Take** | Ship HIP extras as **WGP** (`hipcc --offload-arch=gfx1030` without `-mcumode`). LLVM default is `-mno-cumode` → native WGP ([AMDGPUUsage Target Features](https://llvm.org/docs/AMDGPUUsage.html), opened this pass). |
| **Take** | Fold with the mode that matches the binary: WGP → `LocalMemorySize=131072`, barriers **32**, SIMDs/WG **4**; CU → `65536`, barriers **16**, SIMDs/WG **2** (LLVM `getLocalMemorySize` / `getMaxWorkGroupsPerCU`, tip opened this pass). |
| **Take** | Per-WG LDS hard cap is **64 KB** in both modes (ISA 70648 §2.3.1: “A single workgroup may allocate up to 64kB of LDS space.”). The second 64 KB is for a **second** WG (WGP) or the other CU half (CU), not a bigger single tile. |
| **Take (craft)** | Measure `-mcumode` only when the kernel is **LDS-bandwidth** bound, the WG is happy on 2 SIMD32s, and you are not counting on the second CU’s ALUs ([hip-craft.md](hip-craft.md) §1.2). ISA §10.3: CU can give “higher LDS memory bandwidth”; WGP gives “more ALU and texture memory bandwidth to a single workgroup (of at least 4 waves).” |
| **Leave** | Do **not** flip `-mcumode` to “get more LDS WGs” or “more barriers” — CU **halves** both the shared LDS pool and the barrier budget ([lds-occupancy.md](lds-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md)). |
| **Leave** | Do not ship CU-mode objects on a guess. Do not retip extras / open UNC for launch-mode occupancy. |
| **Leave** | Do not treat far-side LDS as free in WGP. ISA §10.3: waves may hit “near” or “far” side “equally, but performance may be lower in some cases.” CU mode makes the far half **illegal** for that WG (ISA §2.3.1). |

## 1. Defaults and flags

| Knob | Value on gfx1030 | Source (this pass) |
|---|---|---|
| Clang / LLVM feature `cumode` | **off** by default → WGP | [AMDGPUUsage Target Features](https://llvm.org/docs/AMDGPUUsage.html): “When disabled native WGP wavefront execution mode is used, when enabled CU wavefront execution mode is used.” Separate `-m[no-]cumode`; default if unspecified is **off**. |
| Enable CU | `-mcumode` / `-mattr=+cumode` | same |
| Keep WGP (explicit) | `-mno-cumode` / `-mattr=-cumode` | same |
| Processor row | `gfx1030` lists `cumode` as **supported**, not as a default-on feature | AMDGPUUsage Processors table (V620 listed under gfx1030) |
| HIP craft default | wave32 + **WGP** | [hip-craft.md](hip-craft.md) checklist #1; do not add `-mcumode` until measured |

`COMPUTE_PGM_RSRC1.WGP_MODE` is set from the same feature bit (LLVM `S_00B848_WGP_MODE(FeatureCuMode ? 0 : 1)` in `AMDGPUBaseInfo.cpp`, tip opened this pass).

## 2. Resource table (what the mode changes)

ISA 70648 Document ID **70648** §2.3.1 / §10.3 (local text `/workspace/rdna2-src/rdna2-isa.txt` opened this pass) + LLVM occupancy math:

| | CU (`-mcumode`) | WGP (HIP/LLVM default) |
|---|---|---|
| SIMDs used by one WG | **2** (one ISA CU) | **4** (whole WGP) |
| LDS visible / pool for concurrent WGs | **64 KB** half attached to that CU | full **128 KB** address space |
| Workgroup LDS addressable cap | still **64 KB** | still **64 KB** |
| Barrier slots (multi-wave WG) | **16** | **32** |
| Wave slots on the placement unit | **32** (`16×2`) | **64** (`16×4`) |
| ISA reason | “higher LDS memory bandwidth”; both halves run in parallel; upper waves cannot read lower half | more ALU + texture BW for a WG of ≥ 4 waves |
| Far-side LDS | **illegal** (half split) | legal; “performance may be lower in some cases” |

LLVM encoding (`AMDGPUBaseInfo.cpp`, tip opened this pass):

```text
isFullSIMDMode = !FeatureCuMode          // gfx1030; gfx1250 always full
LocalMemorySize = Physical(131072) / (isFullSIMDMode ? 1 : 2)
                // WGP: 131072 ; CU: 65536
Addressable/WG  = min(65536, LocalMemorySize)   // 65536 either way on gfx10
MaxBarriers     = (GFX10+ && !FeatureCuMode) ? 32 : 16
MaxWaves_unit   = 16 * getNumWorkGroupSIMDs(isFullSIMDMode)
                // WGP: 64 ; CU: 32   (SIMDs 4 vs 2 — ISA §10.3; wiki lock)
```

GPUOpen Occupancy explained (opened this pass): “up to **16** barriers in flight **per pair of SIMDs**.” Pair of SIMDs = one ISA CU → WGP (two pairs) = **32**. Matches LLVM.

VGPR file per SIMD32 (**1024** wave32) and max waves/SIMD (**16**) do **not** change with mode — only how many SIMDs one WG may span.

## 3. Occupancy impact (fold into the min?)

Launch mode **does** change terms inside `min(VGPR, LDS, WG/barrier)` — it is not “bandwidth topology only.”

| Term | WGP | CU | PIX row? |
|---|---|---|---|
| `MaxWGsLDS` | `floor(131072 / align(LDS))` | `floor(65536 / align(LDS))` | scales **LDS** |
| Barriers / multi-wave WGs | `min(64/N, 32)` | `min(32/N, 16)` | scales **Barriers** |
| Waves from WG packing → waves/EU | spread over **4** SIMDs | spread over **2** SIMDs | scales conversion |
| VGPR waves/EU | unchanged formula | unchanged | **VGPR** unchanged |
| Thread Group Size semantics | unchanged | unchanged | same SPI rule |

Worked sketch, FlatWG=**256** (`N=8` wave32), no VGPR pressure:

| LDS / WG (aligned) | MaxWGsLDS WGP | MaxWGsLDS CU | Barrier WGs WGP | Barrier WGs CU | Binding WGs WGP → waves/EU* | Binding WGs CU → waves/EU* |
|---|---|---|---|---|---|---|
| 4 KiB | 32 | 16 | 8 | 4 | **8** → 8/EU | **4** → 8/EU† |
| 32 KiB | **4** | 2 | 8 | 4 | **4** → 8/EU | **2** → 8/EU† |
| 48–64 KiB | **2** | **1** | 8 | 4 | **2** → 4/EU | **1** → 4/EU† |

\*waves/EU ≈ `(BindingWGs × N) / NumWorkGroupSIMDs` then clamp to 16.  
†CU places the same waves on half the SIMDs, so waves/EU can look similar while **WGs per WGP** and ALU parallelism drop. Fat LDS tiles that were 2 WGs/WGP become **1 WG per CU-half** under `-mcumode`.

So: CU mode can **look** occupancy-neutral on a per-SIMD read while cutting concurrent WGs and ALU width. Prefer shrinking LDS / VGPR over flipping mode when theory is short ([occupancy-composite.md](occupancy-composite.md)).

CLI (theory; match the attr to the object):

```text
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32             --wg-size=256 --vgprs=128 --lds=32k
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32,+cumode    --wg-size=256 --vgprs=128 --lds=32k
```

## 4. Practical Take/Leave for V620 extras

1. Keep `--offload-arch=gfx1030` **without** `-mcumode`. Default WGP + wave32.
2. Size with `llvm-calc-occupancy … -mattr=+wavefrontsize32` (no `+cumode`) and gate with `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
3. Prefer lowering LDS / VGPR / WG size when measured occupancy is short — same ladder as [occupancy-composite.md](occupancy-composite.md).
4. **When (if ever) to measure `-mcumode`:** LDS-bandwidth-bound kernel, WG fits on 2 SIMD32s, second CU’s ALUs idle (softmax / small-scratch attention candidate per [hip-craft.md](hip-craft.md) §1.2). Compare A/B with identical VGPR/LDS/WG; do **not** ship the CU object unless the microbench wins.
5. VALU-bound 8-wave GEMM / DOT tiles that want 4 SIMD stay in WGP.
6. Do not retip extras or open UNC for this axis.

## 5. Sources

1. RDNA 2 ISA 70648 §2.3.1 (LDS 128 KB/WGP, 64 KB/WG, CU vs WGP split) / §10.3 (SIMDs 2 vs 4, higher LDS BW vs far-side caveat) — Document ID 70648; local text opened this pass (`/workspace/rdna2-src/rdna2-isa.txt`); AMD listing https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
2. LLVM AMDGPUUsage Target Features — `cumode` / `-m[no-]cumode`, default off → WGP — https://llvm.org/docs/AMDGPUUsage.html (opened this pass)
3. LLVM `IsaInfo::getLocalMemorySize` / `getAddressableLocalMemorySize` / `getMaxWorkGroupsPerCU` / `isFullSIMDMode` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp (tip opened this pass)
4. GPUOpen Occupancy explained (updated 2024-06-26) — 16 barriers / SIMD pair; PIX limiters — https://gpuopen.com/learn/occupancy-explained/ (opened this pass)
5. HIP hardware implementation — RDNA WGP / LDS bank picture — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (opened this pass)
6. Sibling wiki: [hip-craft.md](hip-craft.md) §1, [lds-occupancy.md](lds-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [wave-size-occupancy.md](wave-size-occupancy.md), [architecture.md](architecture.md) §3.1 / §4.2

Idle pass 2026-09-18.
