---
title: "In-kernel wave placement: HW_ID1 / __smid() as a residency probe (gfx1030)"
sources:
  - https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/llvm/lib/Target/AMDGPU/SIDefines.h
  - https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/llvm/lib/Target/AMDGPU/Utils/AMDGPUAsmUtils.cpp
  - https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
  - https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/clang/lib/Basic/Targets/AMDGPU.cpp
  - https://gitlab.freedesktop.org/mesa/mesa/-/blob/4912245c6fa768e4a9c90686e724195605d24b2f/src/amd/registers/gfx103.json
  - https://github.com/ROCm/rocm-systems/blob/566f98b36d9ab04f7289bd5bd1da82ec6126b3eb/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h
  - https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
---

# In-kernel wave placement: `HW_ID1` / `__smid()` as a residency probe (gfx1030)

Lock: **on gfx1030 a wave can read where it is running (SE, SA, WGP, SIMD, wave slot) with one `s_getreg_b32 hwreg(HW_REG_HW_ID1)`. That is the cheapest way to measure *achieved* residency per SIMD against the theoretical occupancy fold. HIP's `__smid()` is not a substitute: on gfx10 it drops the wave slot and one SIMD bit, and in the default WGP mode it still appends a "CU" bit because of a `defined()` test on a macro that is always defined.**[^hwid1fields][^smidgfx10][^cumodemacro][^probe]

It is not a PIX / `llvm-calc-occupancy` MaxWaves row. It is a measurement tool for the gap between theory ([occupancy-composite.md](occupancy-composite.md)) and what the SPI actually launched ([spi-ace-occupancy.md](spi-ace-occupancy.md)).

Does **not** change extras HIP or any UNC ticket. It is a debug and review recipe for new kernels. No tok/s. Do not restate the FA pin.

Date: **2026-11-17** Europe/Paris.

Companions: [occupancy-dump.md](occupancy-dump.md) (static VGPR/SGPR/LDS dump), [spi-ace-occupancy.md](spi-ace-occupancy.md) (measured ≤ theory), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md) (WGP vs `-mcumode`), [kcache-occupancy.md](kcache-occupancy.md).

## 1. The registers (gfx103)

LLVM numbers `HW_REG_HW_ID1` = 23 and `HW_REG_HW_ID2` = 24 on GFX10+. The old GFX9 `HW_REG_HW_ID` (4) is not valid on GFX10+; on GFX10 the assembler accepts the name `HW_REG_HW_ID` as an alias of ID 23.[^sidefines][^asmutils]

Field layout from Mesa's gfx103 register database:[^hwid1fields]

| Register | Field | Bits |
|---|---|---|
| `SQ_WAVE_HW_ID1` | `WAVE_ID` | 4:0 |
| | `SIMD_ID` | 9:8 |
| | `WGP_ID` | 13:10 |
| | `SA_ID` | 16 |
| | `SE_ID` | 19:18 |
| `SQ_WAVE_HW_ID2` | `QUEUE_ID` | 3:0 |
| | `PIPE_ID` | 5:4 |
| | `ME_ID` | 9:8 |
| | `STATE_ID` | 14:12 |
| | `WG_ID` | 20:16 |
| | `VM_ID` | 27:24 |

The builtin immediate is `(size-1) << 11 | offset << 6 | id`. HIP's header defines this as `GETREG_IMMED(SZ, OFF, REG)`.[^getregimm] So the full `HW_ID1` word is `__builtin_amdgcn_s_getreg((31 << 11) | 23)`.

## 2. What `__smid()` returns on gfx1030

`__smid()` in `amd_device_functions.h` (rocm-systems `develop` @ `566f98b3`) has this gfx10/gfx11 branch:[^smidgfx10][^smiddefs]

- it reads SE (offset 18, 2 bits on gfx10), SA (16, 1 bit) and WGP (10, 4 bits) with three separate `s_getreg`;
- under `#if defined(__AMDGCN_CUMODE__)` it reads a fourth "CU_ID" from offset 8, 1 bit;
- it packs them as `se | sa | wgp | cu`, high to low.[^assemble]

Clang defines `__AMDGCN_CUMODE__` **always**, with value 0 or 1: `Builder.defineMacro("__AMDGCN_CUMODE__", Twine(CUMode))`.[^cumodemacro] So `defined()` is true in WGP mode too. In the probe, the macro is 0 without `-mcumode` and 1 with it, and `__smid()` emits the same four `s_getreg_b32` in both modes.[^probe]

What follows for gfx1030 code:

1. **Bit 0 of `__smid()` is HW_ID1 bit 8.** Mesa names bits 9:8 `SIMD_ID`, not a CU id.[^hwid1fields] The header calls the same bit `CU_ID`. This page does not claim which SIMD numbering maps to which CU half of a WGP (unsourced). Do not bucket `__smid()` values as "CUs" in WGP mode. A WGP-mode workgroup can use the SIMDs of both CUs ([wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md)).
2. **`__smid()` has no wave slot and only one SIMD bit.** It cannot count waves per SIMD. Use the raw `HW_ID1` word.
3. **Values are sparse.** `WGP_ID` is a 4-bit field. Read the distinct values a device actually returns; do not assume a dense 0..N-1 range (unsourced for V620 harvesting).
4. The header warns that results can change over time.[^smidgfx10]

## 3. Codegen cost (probe)

Compiler Explorer, `--offload-arch=gfx1030 --cuda-device-only -O3`, `__launch_bounds__(256)`, ROCm 6.4.0 / 7.14.0 / 10.0.0, with and without `-mcumode`. Same result on every compiler:[^probe]

| Kernel | Emitted | VGPR / SGPR (`next_free`) |
|---|---|---|
| raw `HW_ID1` word, one store per wave | 1× `s_getreg_b32 s0, hwreg(HW_REG_HW_ID1)` | 2 / 7 |
| `__smid()` | 4× `s_getreg_b32` (`18,2`, `16,1`, `10,4`, `8,1`) | 2 / 8 |
| the same raw read twice | 2× `s_getreg_b32` (not merged) | 2 / 8 |
| raw `HW_ID2` word | 1× `s_getreg_b32 s0, hwreg(HW_REG_HW_ID2)` | 2 / 7 |
| `wall_clock64()` + `clock64()` + raw `HW_ID1` | `s_memrealtime`, `s_memtime`, `s_getreg_b32`, one `s_waitcnt lgkmcnt(0)` | — |

The reads are scalar (SGPR), so they cost no VGPRs beyond the store. They are never CSE'd or hoisted together: `int_amdgcn_s_getreg` is `IntrNoMem` + `IntrHasSideEffects` "so it can be used to read cycle counters".[^intrinsic] Read once at wave start and keep the value. `__smid()` costs four reads where one raw read plus shifts would do.

`wall_clock64()` lowers to `__builtin_readsteadycounter` (`s_memrealtime`) and `clock64()` to `__builtin_readcyclecounter` (`s_memtime`) on gfx1030. Both are SMEM, so they need `lgkmcnt`.[^clocks][^probe]

## 4. Residency recipe (Take, Later)

Not run on silicon here. This is a method, not a measurement:

1. Per wave, at entry: `hw = s_getreg(HW_ID1)` and `t0 = wall_clock64()`. At exit: `t1 = wall_clock64()`. Lane 0 writes `{hw, t0, t1}` to a buffer indexed by global wave id.
2. Host: key each record by `(SE, SA, WGP, SIMD)` and sweep the `[t0, t1]` intervals to get the peak number of concurrent waves per SIMD. The highest `WAVE_ID` seen per SIMD is a cross-check (field semantics from the Mesa name only).
3. Compare that peak with the static fold from [occupancy-dump.md](occupancy-dump.md) / [occupancy-composite.md](occupancy-composite.md). Peak < theory with idle SIMDs means launch rate or not enough work ([spi-ace-occupancy.md](spi-ace-occupancy.md)). Peak = theory means the limiter is inside the wave.
4. Keep the probe out of produce builds. It adds a store and a `lgkmcnt` wait per wave, and its results are only valid for the build being measured.

**Leave:** `__smid()`-based "SM id" histograms ported from CUDA code. On gfx1030 WGP mode they mix a SIMD bit into the id (§2).

## Sources

[^hwid1fields]: Mesa `src/amd/registers/gfx103.json` @ `4912245c6fa768e4a9c90686e724195605d24b2f`, register types `SQ_WAVE_HW_ID1` (`WAVE_ID` 0–4, `SIMD_ID` 8–9, `WGP_ID` 10–13, `SA_ID` 16, `SE_ID` 18–19) and `SQ_WAVE_HW_ID2`. https://gitlab.freedesktop.org/mesa/mesa/-/blob/4912245c6fa768e4a9c90686e724195605d24b2f/src/amd/registers/gfx103.json#L14611-L14630
[^sidefines]: LLVM `SIDefines.h` @ `766b1121998d6a2e273a1a114908ce1f4376ef02`, `ID_HW_ID = 4`, `ID_HW_ID1 = 23`, `ID_HW_ID2 = 24`. https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/llvm/lib/Target/AMDGPU/SIDefines.h#L753-L769
[^asmutils]: LLVM `AMDGPUAsmUtils.cpp` @ `766b1121998d6a2e273a1a114908ce1f4376ef02`, hwreg name table: `HW_REG_HW_ID` → ID 4 `isNotGFX10Plus`; `HW_REG_HW_ID1` / `HW_REG_HW_ID2` `isGFX10Plus`; `HW_REG_HW_ID` → ID 23 `isGFX10`. https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/llvm/lib/Target/AMDGPU/Utils/AMDGPUAsmUtils.cpp#L186-L227
[^intrinsic]: LLVM `IntrinsicsAMDGPU.td` @ `766b1121998d6a2e273a1a114908ce1f4376ef02`, `int_amdgcn_s_getreg` `[IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]` with the cycle-counter comment. https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L2094-L2099
[^cumodemacro]: Clang `clang/lib/Basic/Targets/AMDGPU.cpp` @ `766b1121998d6a2e273a1a114908ce1f4376ef02`, `Builder.defineMacro("__AMDGCN_CUMODE__", Twine(CUMode))` (always defined, value 0/1). https://github.com/llvm/llvm-project/blob/766b1121998d6a2e273a1a114908ce1f4376ef02/clang/lib/Basic/Targets/AMDGPU.cpp#L294
[^smiddefs]: rocm-systems `projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h` @ `566f98b36d9ab04f7289bd5bd1da82ec6126b3eb`, GFX10/11/12 `HW_ID 23`, `HW_ID_WGP_ID_OFFSET 10`/`SIZE 4`, `#if (defined(__AMDGCN_CUMODE__))` → `HW_ID_CU_ID_OFFSET 8`/`SIZE 1`, `HW_ID_SA_ID_OFFSET 16`/`SIZE 1`, gfx10 `HW_ID_SE_ID_SIZE 2`, `OFFSET 18`. https://github.com/ROCm/rocm-systems/blob/566f98b36d9ab04f7289bd5bd1da82ec6126b3eb/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h#L765-L795
[^getregimm]: Same file @ `566f98b36d9ab04f7289bd5bd1da82ec6126b3eb`, `GETREG_IMMED(SZ, OFF, REG) (((SZ) << 11) | ((OFF) << 6) | (REG))`. https://github.com/ROCm/rocm-systems/blob/566f98b36d9ab04f7289bd5bd1da82ec6126b3eb/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h#L814-L821
[^assemble]: Same file @ `566f98b36d9ab04f7289bd5bd1da82ec6126b3eb`, `assemble_smid` (packs xcc, se, sa, wgp, cu high to low). https://github.com/ROCm/rocm-systems/blob/566f98b36d9ab04f7289bd5bd1da82ec6126b3eb/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h#L823-L834
[^smidgfx10]: Same file @ `566f98b36d9ab04f7289bd5bd1da82ec6126b3eb`, `__smid` comment ("results vary over time") and the `__GFX10__ || __GFX11__` branch with the `__AMDGCN_CUMODE__` CU read. https://github.com/ROCm/rocm-systems/blob/566f98b36d9ab04f7289bd5bd1da82ec6126b3eb/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h#L836-L889
[^clocks]: Same file @ `566f98b36d9ab04f7289bd5bd1da82ec6126b3eb`, `__clock64` → `__builtin_readcyclecounter`, `wall_clock64` → `__builtin_readsteadycounter`. https://github.com/ROCm/rocm-systems/blob/566f98b36d9ab04f7289bd5bd1da82ec6126b3eb/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h#L634-L647
[^probe]: Compiler Explorer compile API probe run 2026-11-17 (Paris) against `hipclang-rocm-60400`, `hipclang-rocm-71400`, `hipclang-rocm-100000`; args `--offload-arch=gfx1030 --cuda-device-only -O3` with and without `-mcumode`; kernels as in §3 plus a macro probe (`__AMDGCN_CUMODE__` = 0 default, 1 with `-mcumode`, on 7.14.0); `s_getreg_b32` / `s_memtime` / `s_memrealtime` lines and `.amdhsa_next_free_vgpr` / `_sgpr` read from emitted assembly. Compiler Explorer builds are not pinned to a SHA, and the headers it ships may predate rocm-systems `566f98b3`; re-run to confirm. https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
