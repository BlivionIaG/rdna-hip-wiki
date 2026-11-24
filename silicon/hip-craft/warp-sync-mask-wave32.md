---
title: "HIP *_sync warp intrinsics: 64-bit mask on wave32 gfx1030"
updated: 2026-11-24
sources:
  - https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h
  - https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h
  - https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/hip_assert.h
  - https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/hip/docs/how-to/hip_cpp_language_extensions.rst
  - https://github.com/ROCm/TheRock/tree/f51dc6c91e0d3214f22853fd5cb3f96dbc7d2c4b
  - https://github.com/ROCm/rocm-systems/blob/32ac5965428d7db430fa260bedb8bff8a936bffb/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h
  - https://github.com/ROCm/clr/blob/dd7f9576620569fc8847cef767efcc1f4bab9d1a/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h
  - https://github.com/ROCm/clr/blob/0f2d6024245abde73eaff463cdc1f10f193395b1/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/aea49b7ec322484cffada5fe263868fbbc88e1b2
  - https://github.com/BlivionIaG/rdna-hip-wiki/tree/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24
  - https://github.com/opengfx1030/vllm-rdna/commit/0fa8915b254b9223da3542c9a70ded9b2d9254a4
  - https://github.com/opengfx1030/vllm-rdna/commit/7823f209757812123a68f2aa62d9438f9e1739e8
  - https://github.com/opengfx1030/vllm-rdna/commit/f22a15b64bf4b3525dba8a7fca0379c3513cf5ca
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/setup.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/core/cuda_compat.h
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/topk.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/exl3_hadamard.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/gdn_prefill_prep_rdna2.cu
---

# HIP `*_sync` warp intrinsics: 64-bit mask on wave32 gfx1030

Up: [hip-craft/](README.md) · See also: [hip-craft.md](hip-craft.md) §5 (intrinsics), [dpp-crosslane-occupancy.md](../occupancy/dpp-crosslane-occupancy.md), [wave-size-occupancy.md](../occupancy/wave-size-occupancy.md), [toolchain/compiler/](../../toolchain/compiler/README.md) (`__shfl_xor` `v_cndmask`, rocm-systems#11618).

Trigger: stew675/llama-cpp-rdna-boosts r339 (`aea49b7e`) says a contributor patch did not build because five shuffle calls passed a 32-bit `0xFFFFFFFF` mask, which HIP's `amd_warp_sync_functions.h` rejects with a `static_assert`. The commit reports reproducing that on ROCm 10.1.0, 10.2.0 and 7.14.1, and folds the code onto ggml's own `ggml_cuda_shfl()` / `ggml_cuda_shfl_xor()` helpers instead.[^stew-r339]

Header pin used below: `ROCm/rocm-systems` `ca887ee8`, which is the `rocm-systems` submodule commit in TheRock tag `therock-7.14.1` (target `f51dc6c9`).[^tr7141] The wiki's live dest is ROCm 7.14.0 ([toolchain/pin/](../../toolchain/pin/README.md)). The GitHub API returned 404 for `ROCm/TheRock` ref `tags/therock-7.14.0` on 2026-11-24 (unsourced beyond that API call), so the 7.14.0 headers were not read at a SHA. The Compiler Explorer probe below used ROCm 7.14.0 headers, and its `static_assert` diagnostic points at line 298 of `amd_warp_sync_functions.h`. That is the same line as the `__shfl_xor_sync` `static_assert` at `ca887ee8`.[^probe-bad][^ws-shfl]

## Take / Leave

| | Item | Timing |
|---|---|---|
| **Take** | Every `*_sync` mask on HIP is a 64-bit integer type (`0xffffffffffffffffULL`, `uint64_t`). A 32-bit mask is a **compile error**, not a runtime problem, on wave32 and wave64 alike. | Now (already the dest convention) |
| **Take** | On wave32, `0xffffffffull` and `0xffffffffffffffffULL` behave the same, because the header drops the upper 32 bits first. | Now |
| **Take** | In a build without `NDEBUG`, a `*_sync` call whose mask does not equal the active lanes compiles to an `exec` compare plus `s_trap 2`. A full-mask shuffle reached from a divergent branch traps in such a build and runs silently with `-DNDEBUG`. | Now (debug-build hygiene) |
| **Leave** | Treating `__shfl_*_sync` as a different shuffle. With `NDEBUG` its ISA matched plain `__shfl_xor` in the probe. | — |
| **Later** | Upstream clang trunk lowers constant-offset `__shfl_xor` to `v_permlanex16` + DPP. The ROCm 7.14.0 clang still emits `ds_bpermute` + `v_cndmask`. Re-check when the dest compiler moves. | Later |

## 1. Mask type: 64-bit `static_assert`

- Every masked builtin is a template on `MaskT` that opens with `static_assert(is_integral<MaskT> && sizeof(MaskT) == 8, "The mask must be a 64-bit integer. Implicitly promoting a smaller integer is almost always an error.")`. That covers `__syncwarp(mask)`, `__ballot_sync`, `__all_sync`, `__any_sync`, `__match_*_sync`, `__shfl{,_up,_down,_xor}_sync` and `__reduce_*_sync`.[^ws-assert]
- The check is on `sizeof`, so `0xFFFFFFFF`, `0xffffffffu` and `uint32_t(-1)` all fail whatever the wave size. On gfx1030 both the ROCm 7.14.0 clang and clang trunk stopped with `static assertion failed due to requirement 'sizeof(unsigned int) == 8'`.[^probe-bad]
- HIP's language-extension docs say: "All mask values returned or accepted by these built-ins are 64-bit unsigned integer values, even when compiled for a device with 32 threads per warp. On such devices the higher bits are unused." They also say the implementation "includes a static assert to check that the program source uses the correct type for the mask."[^doc-mask]
- Our own record: dest commit `f22a15b6` already fixed a 64-bit `shfl` mask in `layernorm` ([kernels/aot-glue/layernorm.md](../../kernels/aot-glue/layernorm.md)).[^c-f22a15b6] The closed Glm5Next PR #51 hit the same thing. Its first commit `0fa8915b` used `__shfl_xor_sync(0xffffffffu, …)` in `glm5_kda_decode_rdna2.cu`, and commit `7823f209` in the same PR removed those lines.[^pr51-a][^pr51-b]

## 2. `HIP_ENABLE_WARP_SYNC_BUILTINS` → `HIP_DISABLE_WARP_SYNC_BUILTINS`

- In ROCm 6.2 (`clr` `dd7f9576`) the masked builtins were an opt-in preview behind `#ifdef HIP_ENABLE_WARP_SYNC_BUILTINS`, and the comment says the same applies to `__activemask()`.[^clr620] The ROCm 6.4 header (`clr` `0f2d6024`) still gates on `HIP_ENABLE_WARP_SYNC_BUILTINS`.[^clr640]
- At the 7.14.1 pin the gate is inverted. The builtins are on unless `HIP_DISABLE_WARP_SYNC_BUILTINS` is defined, and `__activemask()` is behind the same macro.[^ws-gate][^wf-activemask] The docs say the `__sync` variants are "available in ROCm 7.0 (and enabled by default, unlike in previous versions)".[^doc-mask]
- So on the dest toolchain `HIP_ENABLE_WARP_SYNC_BUILTINS` has no effect: the string does not appear in the pinned `amd_warp_sync_functions.h` (searched 2026-11-24).[^ws-gate]
- `HIP_ENABLE_EXTRA_WARP_SYNC_TYPES` is a separate opt-in. It adds 64-bit integer and floating-point `__reduce_*_sync` overloads that CUDA does not have.[^ws-extra]

## 3. What the mask actually does on wave32

- `__hip_adjust_mask_for_wave32(MASK)` runs `if (static_cast<int>(warpSize) == 32) MASK &= 0xFFFFFFFF;` before anything else.[^ws-adjust] That is why `0xffffffffull` (a 64-bit type holding a 32-bit value, as in dest `exl3_hadamard.cu`) is a full mask on gfx1030.[^dest-exl3]
- For the shuffles, the mask is **only checked**. `__shfl_xor_sync` returns `__shfl_xor(var, laneMask, width)` after the check and does not use the mask to move data.[^ws-shfl] For `__ballot_sync` and `__match_any_sync`, the result is also ANDed with the mask.[^ws-ballot]
- The check (`__hip_check_mask`) is a waterfall loop. Each lane reads the first lane's mask, and lanes whose mask matches assert `MASK == __ballot(true)`, so the masks must split the active lanes into groups exactly.[^ws-check] `__hip_assert` expands to nothing under `NDEBUG`. Otherwise it is `if (!(COND)) __builtin_trap();`.[^hip-assert]
- In ISA (ROCm 7.14.0 clang, gfx1030, no `NDEBUG`), the full-mask reduction starts with `s_cmp_eq_u32 exec_lo, -1` and branches to `s_trap 2`. With `-DNDEBUG`, the `__shfl_xor_sync` kernel is instruction-for-instruction the same as the plain `__shfl_xor` kernel.[^probe-amd-debug][^probe-amd-ndebug]
- Wave64 (not a HIP config on gfx10+, see [wave-size-occupancy.md](../occupancy/wave-size-occupancy.md)): the upper-half drop happens only when `warpSize == 32`. A wave64 build that passes a value-32-bit mask while all 64 lanes are active would fail the `MASK == __ballot(true)` check in a non-`NDEBUG` build. That follows from reading the header; it was not compiled here.[^ws-adjust][^ws-check]

## 4. `__ballot` / `__activemask` width and `warpSize`

- `__ballot` returns `unsigned long long` from `__builtin_amdgcn_ballot_w64`, and `__activemask()` is `__ballot(true)`.[^wf-ballot][^wf-activemask] The docs say both "have a 64-bit return value (unlike the 32-bit value returned by the CUDA built-ins)".[^doc-ballot]
- On gfx1030 the upper word is zero. The probe stores `vcc_lo` (ballot) and `exec_lo` (activemask) as the low dwords and a literal `v_mov_b32 v1, 0` as both high dwords.[^probe-amd-vote]
- `warpSize` is a `const` struct object whose `operator int()` returns `__builtin_amdgcn_wavefrontsize()`. It is not a `constexpr` variable.[^wf-warpsize] The docs say it is not a host compile-time constant and must be queried with `hipDeviceAttributeWarpSize`. Since ROCm 7.0 it is "early folded by the compiler", so it works for loop bounds and unrolling. For a device-side compile-time constant the docs suggest an `__GFX8__`/`__GFX9__` macro switch.[^doc-warpsize] In the probe, device `warpSize` folds to `v_mov_b32 v1, 32` on gfx1030.[^probe-amd-ws]
- `__lane_id()` on wave32 is `__builtin_amdgcn_mbcnt_lo(-1, 0)`.[^wf-laneid]

## 5. ISA lowering (shuffle) and occupancy

| Compiler (Compiler Explorer, ROCm 7.14.0 headers, `-O3 --offload-arch=gfx1030 -DNDEBUG`) | 5-step `__shfl_xor{,_sync}` float butterfly | VGPR / SGPR (`.vgpr_count` / `.sgpr_count`) |
|---|---|---|
| ROCm 7.14.0 clang (`clang-rocm-71400`) | 5 × `ds_bpermute_b32`, each with a `v_xor` / `v_cmp_lt` / `v_cndmask` lane bound and `s_waitcnt lgkmcnt(0)`[^probe-amd-ndebug] | 6 / 8, the same for `_sync` and plain[^probe-amd-ndebug] |
| clang trunk (`hipclang-trunk-rocm-71400`) | `v_permlanex16_b32` + 4 × `v_add_f32_dpp` (`row_ror:8`, `row_xmask:4`, two `quad_perm`), no LDS[^probe-trunk-ndebug] | 3 / 6, the same for `_sync` and plain[^probe-trunk-ndebug] |

- At the 7.14.1 pin, `__shfl_xor` computes `index = self ^ lane_mask`, clamps it with `index >= ((self + width) & ~(width - 1)) ? self : index`, and calls `__builtin_amdgcn_ds_bpermute(index << 2, var)`.[^wf-shflxor] The `v_cndmask` above is that clamp, the same leftover described in [toolchain/compiler/](../../toolchain/compiler/README.md).
- On `rocm-systems` `develop` (`32ac5965`), `__shfl` / `__shfl_xor` call `__builtin_amdgcn_wave_shuffle` when the compiler has it and fall back to `ds_bpermute`.[^wf-dev] The trunk DPP result above used the 7.14.0 headers, which only have the `ds_bpermute` path.[^probe-trunk-pre] So the DPP rewrite comes from the trunk backend, but the LLVM change that does it was not identified (unsourced).
- Occupancy: in these probes the masked `_sync` form adds no VGPR or SGPR over the plain form, with or without `NDEBUG`. Without `NDEBUG` it adds scalar compares and branches and an `s_trap` block.[^probe-amd-debug][^probe-amd-ndebug] No wave-occupancy change was measured, and none is claimed.

## 6. Dest tip check (`rdna_extras` `ba9b43bc`, 2026-11-24)

- `csrc/rocm/` `*_sync` calls use `0xffffffffffffffffULL` (for example `gdn_prefill_prep_rdna2.cu`) or `0xffffffffull` (`exl3_hadamard.cu`). Both are 64-bit types.[^dest-gdnprep][^dest-exl3]
- Upstream TUs that HIP compiles pick a 64-bit mask under `USE_ROCM`. Examples: `FINAL_MASK 0xffffffffffffffffULL` in `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`, `using WarpMask = uint64_t` in the MUSE-Spark header, and `cuda_compat.h` mapping `VLLM_SHFL_XOR_SYNC` to plain `__shfl_xor` on ROCm.[^dest-dsv4][^dest-muse-mask][^dest-compat]
- The upstream files that still pass `0xffffffff` / `0xffffffffu` are CUDA-only. They sit in the `if(VLLM_GPU_LANG STREQUAL "CUDA")` block (CMakeLists L510–L1246), or behind `#ifndef USE_ROCM` (`persistent_topk.cuh` via `topk.cu`).[^dest-cmake][^dest-topk]
- Awareness, not a defect claim: the MUSE-Spark header sets `kWarpSize = 64` under `USE_ROCM` and derives `group_shift = (threadIdx.x & (kWarpSize - 1)) & ~(kLanes - 1)`.[^dest-muse-mask][^dest-muse-shift] On a wave32 device, lanes with `threadIdx.x & 63 >= 32` would get a `group_mask` whose set bits are all above bit 31. The wave32 adjust turns that into 0, which trips `"mask must be non-zero"` only in a non-`NDEBUG` build. Data movement is unaffected because the shuffle ignores the mask.[^ws-adjust][^ws-check][^ws-shfl] Whether this kernel is ever dispatched on gfx1030 was not checked (unsourced).
- Whether HIP TUs in the dest build get `-DNDEBUG` was not verified. `setup.py` defaults to `CMAKE_BUILD_TYPE=RelWithDebInfo`, or `Debug` with `--debug`, and the flags that build type passes to HIP sources were not traced (unsourced).[^dest-setup]

## Sources

[^stew-r339]: stew675/llama-cpp-rdna-boosts commit `aea49b7ec322484cffada5fe263868fbbc88e1b2` (r339, 2026-11-22 02:42 Europe/Paris), commit message, paragraph 2. https://github.com/stew675/llama-cpp-rdna-boosts/commit/aea49b7ec322484cffada5fe263868fbbc88e1b2
[^tr7141]: TheRock tree at `f51dc6c91e0d3214f22853fd5cb3f96dbc7d2c4b` (target of tag `therock-7.14.1`): submodule `rocm-systems` → `ca887ee80abfb82671fe1d6d8da708a713438e05` (GitHub git-trees API, read 2026-11-24). https://github.com/ROCm/TheRock/tree/f51dc6c91e0d3214f22853fd5cb3f96dbc7d2c4b
[^ws-gate]: `amd_warp_sync_functions.h` L9–L14 (`HIP_DISABLE_WARP_SYNC_BUILTINS`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L9-L14
[^ws-extra]: same file, L41–L42 (`#ifdef HIP_ENABLE_EXTRA_WARP_SYNC_TYPES`, "this macro enable types that are not in CUDA") @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L41-L42
[^ws-adjust]: same file, L95–L99 (`__hip_adjust_mask_for_wave32`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L95-L99
[^ws-check]: same file, L100–L141 (mask semantics comment, `__hip_check_mask`, `__hip_assert(MASK && "mask must be non-zero")`, `MASK == __ballot(true)`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L100-L141
[^ws-assert]: same file, L167–L173 (`__syncwarp(MaskT)` `static_assert`). The same `sizeof(MaskT) == 8` assert appears 11 times in the file @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L167-L173
[^ws-ballot]: same file, L177–L185 (`__ballot_sync` → `__ballot(predicate) & mask`) and L234 (`__match_any(value) & mask`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L177-L185
[^ws-shfl]: same file, L296–L304 (`__shfl_xor_sync`: `static_assert` at L298, adjust, check, `return __shfl_xor(var, laneMask, width)`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L296-L304
[^hip-assert]: `hip_assert.h` L70–L77 (`__hip_assert` empty under `NDEBUG`, else `__builtin_trap()`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/hip_assert.h#L70-L77
[^wf-warpsize]: `amd_warp_functions.h` L96–L101 (`warpSize` const struct, `operator int()` → `__builtin_amdgcn_wavefrontsize()`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h#L96-L101
[^wf-ballot]: same file, L108–L110 (`unsigned long long __ballot` → `__builtin_amdgcn_ballot_w64`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h#L108-L110
[^wf-activemask]: same file, L114–L119 (`__activemask()` = `__ballot(true)` under `!HIP_DISABLE_WARP_SYNC_BUILTINS`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h#L114-L119
[^wf-laneid]: same file, L121–L122 (`__lane_id()` wave32 → `mbcnt_lo(-1, 0)`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h#L121-L122
[^wf-shflxor]: same file, L475–L480 (`__shfl_xor(int)` index, clamp, `ds_bpermute`) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h#L475-L480
[^wf-dev]: `amd_warp_functions.h` L153–L164 (`__shfl(int)`: `#if __has_builtin(__builtin_amdgcn_wave_shuffle)` else `ds_bpermute`) @ `rocm-systems` `develop` `32ac5965428d7db430fa260bedb8bff8a936bffb` (read 2026-11-24). https://github.com/ROCm/rocm-systems/blob/32ac5965428d7db430fa260bedb8bff8a936bffb/projects/clr/hipamd/include/hip/amd_detail/amd_warp_functions.h#L153-L164
[^clr620]: ROCm/clr `rocm-6.2.0` (`dd7f9576620569fc8847cef767efcc1f4bab9d1a`) `amd_warp_sync_functions.h` L25–L32 (opt-in `HIP_ENABLE_WARP_SYNC_BUILTINS`). https://github.com/ROCm/clr/blob/dd7f9576620569fc8847cef767efcc1f4bab9d1a/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L25-L32
[^clr640]: ROCm/clr `rocm-6.4.0` (`0f2d6024245abde73eaff463cdc1f10f193395b1`) `amd_warp_sync_functions.h` L28–L30 (still `#ifdef HIP_ENABLE_WARP_SYNC_BUILTINS`). https://github.com/ROCm/clr/blob/0f2d6024245abde73eaff463cdc1f10f193395b1/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L28-L30
[^doc-mask]: HIP docs source `hip_cpp_language_extensions.rst` L780–L794 (64-bit masks even on 32-lane devices, default-on since ROCm 7.0, `HIP_DISABLE_WARP_SYNC_BUILTINS`, static assert) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/hip/docs/how-to/hip_cpp_language_extensions.rst#L780-L794
[^doc-ballot]: same file, L829–L831 (`__ballot` / `__activemask` 64-bit return) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/hip/docs/how-to/hip_cpp_language_extensions.rst#L829-L831
[^doc-warpsize]: same file, L337–L363 (host query, "AMD devices return 64 for gfx9 and 32 for gfx10 and above", early-folded since ROCm 7.0) and L467–L474 (`__GFX8__`/`__GFX9__` macro) @ `ca887ee8`. https://github.com/ROCm/rocm-systems/blob/ca887ee80abfb82671fe1d6d8da708a713438e05/projects/hip/docs/how-to/hip_cpp_language_extensions.rst#L337-L363
[^probe-bad]: Probe `bad.hip` (`__shfl_xor_sync(0xffffffffu, …)`). The error is at `amd_warp_sync_functions.h:298:62` for ROCm 7.14.0 clang (`bad.amdclang.stderr.txt` L7) and clang trunk (`bad.stderr.txt` L7) @ wiki `dbe85b7f`. https://github.com/BlivionIaG/rdna-hip-wiki/blob/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24/bad.amdclang.stderr.txt#L7
[^probe-amd-ndebug]: Probe `k.amdclang.ndebug.s` L1–L45 (`red_sync`: 5× `ds_bpermute_b32` + `v_cndmask`) and L47–L91 (`red_nosync`, identical body), `.vgpr_count` 6 at L146 / L176, `.sgpr_count` 8 at L141 / L171 @ wiki `dbe85b7f`. https://github.com/BlivionIaG/rdna-hip-wiki/blob/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24/k.amdclang.ndebug.s#L1-L91
[^probe-amd-debug]: Probe `k.amdclang.debug.s` L1–L3 (`s_cmp_eq_u32 exec_lo, -1`, `s_cbranch_scc0 .LBB0_6`), L57 (`s_trap 2`), `.vgpr_count` 6 at L158 / L188 @ wiki `dbe85b7f`. https://github.com/BlivionIaG/rdna-hip-wiki/blob/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24/k.amdclang.debug.s#L1-L57
[^probe-amd-vote]: Probe `k.amdclang.ndebug.s` L93–L106 (`vote`: `v_mov_b32 v2, exec_lo`, `v_cmp_ne_u32 vcc_lo`, `v_mov_b32 v1, 0`, `global_store_dwordx4`) @ wiki `dbe85b7f`. https://github.com/BlivionIaG/rdna-hip-wiki/blob/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24/k.amdclang.ndebug.s#L93-L106
[^probe-amd-ws]: Probe `k.amdclang.ndebug.s` L108–L114 (`ws`: `v_mov_b32_e32 v1, 32`) @ wiki `dbe85b7f`. https://github.com/BlivionIaG/rdna-hip-wiki/blob/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24/k.amdclang.ndebug.s#L108-L114
[^probe-trunk-ndebug]: Probe `k.ndebug.s` L1–L15 (`red_sync`: `v_permlanex16_b32` + 4× `v_add_f32_dpp`) and L16–L30 (`red_nosync`, identical), `.vgpr_count` 3 at L81 / L110, `.sgpr_count` 6 at L76 / L105 @ wiki `dbe85b7f`. https://github.com/BlivionIaG/rdna-hip-wiki/blob/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24/k.ndebug.s#L1-L30
[^probe-trunk-pre]: Probe README (compiler ids, header path `/opt/compiler-explorer/libs/rocm/7.14.0`). A `-E` run of the same compiler on 2026-11-24 showed `__shfl*` bodies calling `__builtin_amdgcn_ds_bpermute(index << 2, var)` with no `wave_shuffle` branch. The preprocessed output was not committed, so that detail rests on this run log only. https://github.com/BlivionIaG/rdna-hip-wiki/blob/dbe85b7f0e313da24dc7f27dd3a40ac2c448b7a4/silicon/hip-craft/probes/warp-sync-mask-2026-11-24/README.md
[^c-f22a15b6]: opengfx1030/vllm-rdna commit `f22a15b64bf4b3525dba8a7fca0379c3513cf5ca` — "fix(rocm/layernorm): 64-bit shfl mask + move public entries to global scope". https://github.com/opengfx1030/vllm-rdna/commit/f22a15b64bf4b3525dba8a7fca0379c3513cf5ca
[^pr51-a]: opengfx1030/vllm-rdna PR #51 (closed, not merged) commit `0fa8915b254b9223da3542c9a70ded9b2d9254a4`, `csrc/rocm/rdna/glm5next/glm5_kda_decode_rdna2.cu` (`__shfl_xor_sync(0xffffffffu, …)`). https://github.com/opengfx1030/vllm-rdna/commit/0fa8915b254b9223da3542c9a70ded9b2d9254a4
[^pr51-b]: PR #51 commit `7823f209757812123a68f2aa62d9438f9e1739e8` ("glm5next: keep RDNA leftovers, SwiGLU, and KDA honest") removes those lines. The PR head `a11188cf` has no 32-bit `*_sync` mask in `csrc/`. https://github.com/opengfx1030/vllm-rdna/commit/7823f209757812123a68f2aa62d9438f9e1739e8
[^dest-gdnprep]: `csrc/rocm/gdn_prefill_prep_rdna2.cu` L240–L247 @ `ba9b43bcd479acc4bddcc98a375a31152279a872`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/gdn_prefill_prep_rdna2.cu#L240-L247
[^dest-exl3]: `csrc/rocm/exl3_hadamard.cu` L44–L45, L68 (`__shfl_xor_sync(0xffffffffull, …)`) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/rocm/exl3_hadamard.cu#L44-L68
[^dest-dsv4]: `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` L63–L67 (`FINAL_MASK` 64-bit under `USE_ROCM`) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu#L63-L67
[^dest-muse-mask]: `csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cuh` L50–L54 (`FINAL_MASK`) and L76–L82 (`WarpMask = uint64_t`, `kWarpSize = 64` under `USE_ROCM`) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cuh#L50-L82
[^dest-muse-shift]: same file, L117–L126 (`groupReduceSum` → `__shfl_xor_sync(group_mask, …, 32)`) and L287–L292 (`kGroupBits`, `group_shift`, `group_mask`) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/fused_muse_spark_qknorm_rope_kv_insert_kernel.cuh#L287-L292
[^dest-compat]: `csrc/core/cuda_compat.h` L46–L60 (`VLLM_SHFL_*_SYNC` → `__shfl_xor` / `__shfl_down` on ROCm) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/core/cuda_compat.h#L46-L60
[^dest-cmake]: `CMakeLists.txt` L405 (CUDA-or-HIP block), L446–L485 (`VLLM_STABLE_EXT_SRC`, includes the DSv4 and MUSE-Spark TUs), L510–L1246 (`if(VLLM_GPU_LANG STREQUAL "CUDA")` … `# if CUDA endif` + `endif()`) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/CMakeLists.txt#L510-L1246
[^dest-topk]: `csrc/topk.cu` L10–L11 (`#ifndef USE_ROCM` → `#include "persistent_topk.cuh"`) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/csrc/topk.cu#L10-L11
[^dest-setup]: `setup.py` L265–L271 (`default_cfg = "Debug" if self.debug else "RelWithDebInfo"`) @ `ba9b43bc`. https://github.com/opengfx1030/vllm-rdna/blob/ba9b43bcd479acc4bddcc98a375a31152279a872/setup.py#L265-L271
