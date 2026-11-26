---
title: "IEEE_MODE / DX10_CLAMP mode bits vs effective occupancy (gfx1030)"
sources:
  - https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/docs/AMDGPUUsage.md
  - https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/lib/Target/AMDGPU/AMDGPUTargetTransformInfo.cpp
  - https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/lib/Target/AMDGPU/SIModeRegisterDefaults.h
  - https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/lib/Target/AMDGPU/SIModeRegisterDefaults.cpp
  - https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/clang/include/clang/Options/Options.td
  - https://github.com/opengfx1030/vllm-rdna/blob/69e947313c49d2052d6cf9f8199b2eaa2441def9/CMakeLists.txt
  - https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
---

# IEEE_MODE / DX10_CLAMP mode bits vs effective occupancy (gfx1030)

Lock: **`COMPUTE_PGM_RSRC1.ENABLE_IEEE_MODE` (bit 23) and `ENABLE_DX10_CLAMP` (bit 21) are NaN-handling mode bits, not a PIX / `llvm-calc-occupancy` MaxWaves input.** The new fact on this page is a side effect of the switch that clears IEEE mode: **`-mno-amdgpu-ieee` stops the device-library helpers from inlining.** `threadIdx` / `blockIdx` / `blockDim` and some math (`rsqrtf`) turn into real calls to `__ockl_*` / `__ocml_*`. The kernel then gets the call ABI: `s_swappc_b64`, a stack, and `next_free_vgpr` / `next_free_sgpr` raised to 32 / 33 in the probe. Each of those is a register-budget hit, so the switch costs occupancy rather than saving instructions.[^probe][^tti][^mrd]

Does **not** change extras HIP or UNC tickets. The dest tip's `CMakeLists.txt` and `cmake/utils.cmake` do not pass `-mno-amdgpu-ieee`, `-fno-honor-nans` or `-ffast-math` (grep at `69e94731`; files that set per-kernel flags elsewhere were not audited).[^dest] No tok/s. Do not restate the FA pin.

Date: **2026-11-12** Europe/Paris.

Companions: [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md) (the max-reduction canonicalize note), `fwd-progress-occupancy.md` (open PR #63; other RSRC1 bits), [vgpr-occupancy.md](vgpr-occupancy.md), [scratch-occupancy.md](scratch-occupancy.md), [occupancy-dump.md](occupancy-dump.md).

## 1. What the bits mean (AMDGPUUsage, GFX9–GFX11)

| Bit | Field | Meaning on gfx1030 |
|---|---|---|
| 21 | `ENABLE_DX10_CLAMP` | The vector ALU uses DX10-style NaN handling: when set, a clamp turns NaN into zero; otherwise NaN passes through.[^usage-rsrc1] |
| 23 | `ENABLE_IEEE_MODE` | FP opcodes that gather exception flags quiet and propagate signaling-NaN inputs per IEEE 754-2008, and `min_dx10` / `max_dx10` become IEEE 754-2008 compliant.[^usage-rsrc1] |

Both are reserved on GFX11.7 and GFX12, so this whole page is a GFX9–GFX11 concern (gfx1030 and gfx1100 included).[^usage-rsrc1] The `.amdhsa_dx10_clamp` and `.amdhsa_ieee_mode` directives both default to 1.[^usage-directives] The IR attributes `"amdgpu-ieee"` / `"amdgpu-dx10-clamp"` say what the function expects on entry and override the calling-convention default.[^usage-attrs] Clang's `-mamdgpu-ieee` is on by default; its help text says turning it off "changes the ABI".[^clang-opt]

## 2. Why the switch blocks inlining

`GCNTTIImpl::areInlineCompatible` builds a `SIModeRegisterDefaults` for caller and callee and refuses to inline unless `isInlineCompatible` holds. That check is `DX10Clamp == CalleeMode.DX10Clamp && IEEE == CalleeMode.IEEE`, an exact match. `AlwaysInline` / `InlineHint` are consulted only **after** the mode check, so they do not help.[^tti][^mrd] The mode is read from the `"amdgpu-ieee"` / `"amdgpu-dx10-clamp"` function attributes when present.[^mrd-cpp]

So a kernel compiled with `-mno-amdgpu-ieee` has `IEEE = false`, while the prebuilt device libraries it links (`ockl` / `ocml` bitcode) keep the default `IEEE = true`, and the inliner leaves those calls in place. That pairing (prebuilt libs keep the default) is inferred from the probe below, not from a ROCm document.

## 3. Probe (2026-11-12)

Compiler Explorer API, `--offload-arch=gfx1030 --cuda-device-only -O3`, three small kernels: an `fmaxf` row max using `blockIdx.x*blockDim.x+threadIdx.x`, the same row max indexed with `__builtin_amdgcn_workgroup_id_x()` / `__builtin_amdgcn_workitem_id_x()`, and `__expf(x)+rsqrtf(x)` with the builtin indexing.[^probe]

| Compiler | Flags | `rowmax` (HIP index) | `rowmax_raw` (builtins) | `ex` (`rsqrtf`) |
|---|---|---|---|---|
| ROCm 6.4.0 | default | no calls, VGPR 8, SGPR 7, scratch 0 | no calls, 8 / 7 / 0 | no calls, 5 / 7 / 0 |
| ROCm 6.4.0 | `-mno-amdgpu-ieee -fno-honor-nans` | **calls `__ockl_get_group_id`, `__ockl_get_local_size`, `__ockl_get_local_id`**, VGPR 32, SGPR 33, scratch 8 | no calls, 8 / 7 / 0 | **calls `__ocml_rsqrt_f32`**, 6 / 33 / 8 |
| ROCm 7.0.1 | same | same as 6.4.0 | same | same |
| ROCm 7.14.0 (dest pin) | default | no calls, 8 / 9 / 0 | no calls, 8 / 8 / 0 | no calls, 5 / 7 / 0 |
| ROCm 7.14.0 | `-mno-amdgpu-ieee -fno-honor-nans` | **3 `__ockl` calls**, 32 / 33 / 8 | no calls, 8 / 8 / 0 | **`__ocml_rsqrt_f32` call**, 6 / 33 / 8 |
| ROCm 10.0.0 | `-mno-amdgpu-ieee -fno-honor-nans` | **3 `__ockl` calls**, 32 / 33 / 0 | no calls, 8 / 8 / 0 | **`__ocml_rsqrt_f32` call**, 5 / 33 / 0 |

VGPR / SGPR are `.amdhsa_next_free_vgpr` / `.amdhsa_next_free_sgpr`; scratch is `.amdhsa_private_segment_fixed_size`. In every no-IEEE row the descriptor has `.amdhsa_ieee_mode 0`, `.amdhsa_dx10_clamp 1`.

Other results from the same probe on ROCm 7.14.0 and 10.0.0:

- `-fgpu-flush-denormals-to-zero` sets `.amdhsa_float_denorm_mode_32 0` (default 3), keeps `ieee_mode 1`, and still inlines everything. Denorm mode is not part of the inline check.[^probe][^mrd]
- f32 `fmaxf` over loaded values compiled to `v_max3_f32` with **no** `v_max_f32 v,v,v` canonicalize under default IEEE mode, so IEEE mode cost nothing in this f32 reduction.
- The packed-f16 max (`__builtin_elementwise_max` on `_Float16x2`) emitted one canonicalize `v_pk_max_f16 v2, v2, v2` under default flags. `-ffinite-math-only` and `-ffast-math` removed it **while keeping IEEE mode on and inlining intact**, which matches the canonicalize note in [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md).

In this toy kernel, 32 VGPRs do not cut waves per SIMD (see [vgpr-occupancy.md](vgpr-occupancy.md) for the granule table). In a real kernel the cost adds on top of the kernel's own register use: the call ABI's clobbered registers and saves, the call itself, and a stack frame where scratch was 0. See [scratch-occupancy.md](scratch-occupancy.md) for why scratch is latency poison.

## 4. Take / Leave

| | |
|---|---|
| **Take (Now, craft)** | To drop a NaN-quieting canonicalize from a max/min reduction, use `-ffinite-math-only` (or per-function fast-math / `nnan`), **not** `-mno-amdgpu-ieee`. The probe shows the former removes the f16 canonicalize with no ABI change. |
| **Take (Now, dump)** | When an object shows unexpected `s_swappc_b64`, `__ockl_*` / `__ocml_*` relocations, `next_free_vgpr` ≥ 32 or non-zero scratch on a simple kernel, check `.amdhsa_ieee_mode` / `.amdhsa_dx10_clamp` in the [occupancy-dump.md](occupancy-dump.md) read-out, and grep the build flags for `-mno-amdgpu-ieee`. |
| **Take (Later, A/B only)** | If a kernel truly needs IEEE mode off, it must avoid every device-library call: index with `__builtin_amdgcn_workitem_id_*` / `__builtin_amdgcn_workgroup_id_*` and use builtins instead of `ocml` math. Verify "no calls" in the disassembly before measuring. |
| **Leave** | Do not add `-mno-amdgpu-ieee` to extras or third-party HIP builds to "save" canonicalizes. Neither bit is an occupancy-calculator input; keep `min(VGPR, LDS, WG/barrier)` in [occupancy-composite.md](occupancy-composite.md) unchanged. |
| **Leave** | No ticket, no retip. The FIXME comments in LLVM say DX10 clamp inlining "should be OK" in principle; nothing upstream changes this today.[^tti][^mrd] |

## Sources

[^usage-rsrc1]: LLVM `AMDGPUUsage.md` @ `95b733c2fb85d76d6ec74ebb8b381ef004e9f18f`, `compute_pgm_rsrc1` GFX6–GFX12 table, bits 21 (`ENABLE_DX10_CLAMP`) and 23 (`ENABLE_IEEE_MODE`), reserved on GFX11.7 / GFX12. https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/docs/AMDGPUUsage.md#L7555-L7600
[^usage-directives]: LLVM `AMDGPUUsage.md` @ `95b733c2fb85d76d6ec74ebb8b381ef004e9f18f`, `.amdhsa_dx10_clamp` / `.amdhsa_ieee_mode` default 1, GFX6–GFX11 (except GFX11.7). https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/docs/AMDGPUUsage.md#L17883-L17890
[^usage-attrs]: LLVM `AMDGPUUsage.md` @ `95b733c2fb85d76d6ec74ebb8b381ef004e9f18f`, function attributes `"amdgpu-ieee"` and `"amdgpu-dx10-clamp"`. https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/docs/AMDGPUUsage.md#L3344-L3349
[^clang-opt]: Clang `Options.td` @ `95b733c2fb85d76d6ec74ebb8b381ef004e9f18f`, `amdgpu_ieee` BoolMOption, `DefaultTrue`, help text "This option changes the ABI. (AMDGPU only)". https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/clang/include/clang/Options/Options.td#L6124-L6130
[^tti]: LLVM `AMDGPUTargetTransformInfo.cpp` @ `95b733c2fb85d76d6ec74ebb8b381ef004e9f18f`, `GCNTTIImpl::areInlineCompatible`: mode check before the `AlwaysInline` / `InlineHint` early return; FIXME on dx10_clamp. https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/lib/Target/AMDGPU/AMDGPUTargetTransformInfo.cpp#L1875-L1895
[^mrd]: LLVM `SIModeRegisterDefaults.h` @ `95b733c2fb85d76d6ec74ebb8b381ef004e9f18f`, `isInlineCompatible` returns `DX10Clamp == CalleeMode.DX10Clamp && IEEE == CalleeMode.IEEE`. https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/lib/Target/AMDGPU/SIModeRegisterDefaults.h#L88-L92
[^mrd-cpp]: LLVM `SIModeRegisterDefaults.cpp` @ `95b733c2fb85d76d6ec74ebb8b381ef004e9f18f`, reads `"amdgpu-ieee"` / `"amdgpu-dx10-clamp"` when the subtarget has `FeatureDX10ClampAndIEEEMode`. https://github.com/llvm/llvm-project/blob/95b733c2fb85d76d6ec74ebb8b381ef004e9f18f/llvm/lib/Target/AMDGPU/SIModeRegisterDefaults.cpp#L18-L26
[^dest]: `opengfx1030/vllm-rdna` @ `69e947313c49d2052d6cf9f8199b2eaa2441def9` (`rdna_extras` tip, 2026-11-12): `CMakeLists.txt`, `cmake/utils.cmake` and `setup.py` contain none of `amdgpu-ieee`, `honor-nans`, `fast-math`, `finite-math`. https://github.com/opengfx1030/vllm-rdna/blob/69e947313c49d2052d6cf9f8199b2eaa2441def9/CMakeLists.txt
[^probe]: Compiler Explorer API probe run 2026-11-12 Europe/Paris against `hipclang-rocm-60400`, `hipclang-rocm-70001`, `hipclang-rocm-71400`, `hipclang-rocm-100000`; flags `--offload-arch=gfx1030 --cuda-device-only -O3` plus the variant flags in the tables; values read from the emitted `.amdhsa_*` directives and the kernel bodies. https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
