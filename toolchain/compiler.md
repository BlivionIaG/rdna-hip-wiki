# Compiler / hipcc

Date: **2026-09-10**. Audience: someone compiling HIP for **gfx1030** (live), with later **gfx1100** and **gfx900** TUs.

Companions: [silicon/hip-craft.md](../silicon/hip-craft.md).

## Live rule

- Compiler in every existing build line is **`hipcc --offload-arch=gfx1030`**.
- Default is **wave32 + WGP**. Do not add `-mwavefrontsize64` or `-mcumode` until measured. HIP: warpSize 64 is not supported on gfx10+.
- Live dest stays **ROCm 7.14.0** (`rocm-7.14.0` @ `830cc1b5e90d`). 7.14.1 is quality-only (no GitHub `rocm-7.14.1` tag). Do not bump to 10.0 / 10.1 nightlies.
- Live extras stay **`-O3`**. [TheRock#7751](https://github.com/ROCm/TheRock/issues/7751) (gfx1034 `-O0` i32 `udiv`/`urem`) is still open.




## 2026-09-10 — TheRock SMP ww33-2.9 (nightly pin, not dest)

[TheRock#8108](https://github.com/ROCm/TheRock/pull/8108) **merged** 2026-09-10 14:36 (`f3f46df9`). Compiler pin **SMP ww33-2.8 / amd-llvm `d6f6cb691863` → ww33-2.9 / `d19dd10a11f4`**. hipify `0e051929` and spirv `4fd57e73` unchanged. Single CP: `[AMDGPU] Use first operand of zext to test first bit zero` ([llvm#217195](https://github.com/ROCm/llvm-project/pull/217195) / commit `d19dd10a11f4`) — general AMDGPU codegen, not gfx1250-only and not an RDNA ISA lever. TheRock HEAD `f3f46df9`. Nightly tip still **`10.1.0a20260910` L+W** (core + device-gfx1030/1100/900); no `10.1.0a20260911` yet, so published wheels do not yet carry ww33-2.9. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Draft [#8125](https://github.com/ROCm/TheRock/pull/8125) (COT+ASAN) and open [#8124](https://github.com/ROCm/TheRock/pull/8124) (systems bump) not merged.

## 2026-09-10 — nightly tip / #8077 host PER_TARGET revert (not dest)

Nightly tip **`10.1.0a20260909` → `10.1.0a20260910`** Linux+Windows for `rocm-sdk-core` and `device-gfx1030` / `gfx1100` / `gfx900`. Compiler pin **unchanged** (SMP ww33-2.8 / amd-llvm `d6f6cb691863`). TheRock HEAD `67fbce01`.

[TheRock#8077](https://github.com/ROCm/TheRock/pull/8077) **merged** 2026-09-09 22:44Z: limited revert of [#7082](https://github.com/ROCm/TheRock/pull/7082) host `LLVM_ENABLE_PER_TARGET_RUNTIME_DIR` — host flag forced **OFF**, restores `lib/llvm/lib` layout (drops triple RPATH) because the ON layout broke binaries/libs built against **10.0** (ROCM-30441). Device-runtime `RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_PER_TARGET_RUNTIME_DIR` stays **ON**. Follow-on [#8098](https://github.com/ROCm/TheRock/pull/8098) (compat symlinks) still **open**. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP.

## 2026-09-09 — nightly tip / SMP ww33-2.8 (not dest)

TheRock compiler pin **SMP ww33-2.8** / amd-llvm `d6f6cb691863` (commit `b0153d7a`, 2026-09-08). hipify `0e051929` and spirv `4fd57e73` unchanged. Nightly tip **`10.1.0a20260909` Linux+Windows** for `rocm-sdk-core` and `device-gfx1030` / `gfx1100` / `gfx900` on `https://nightly.repo.amd.com/rocm/whl-next/`. First published date that can carry ww33-2.8. TheRock HEAD `8943c014` (#8072 systems `09199b3`, #7986 libraries `bf1f3c1`). **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP.


## 2026-09-03 — TheRock SMP ww33-2.6 (nightly pin, not dest)

[TheRock#7859](https://github.com/ROCm/TheRock/pull/7859) merged 2026-09-03 16:11 (`186b09250f1a`). Compiler pin **SMP ww33-2.5 / amd-llvm `9148b61ffec4` → ww33-2.6-1 / `e9e55b898a1b`**. hipify `0e051929` and spirv `4fd57e73` unchanged. PR says the CP is **gfx1250-strict**. Five llvm commits: gfx1250 `s_monitor_sleep`, gfx12 test regen, SOP1_Real NFC, `S_BARRIER_SIGNAL_ISFIRST` barrier-id validate, comgr hotswap `hotswap-barrier-isfirst.s` xfail. **No RDNA ISA delta.** Do not bump live dest off **7.14.0**. Nightly tip `10.1.0a20260904` **Linux** for gfx1030/1100/900 (+core); Windows 0904 still absent (last L+W was 0903). 0904 Linux is the first published wheel date carrying SMP ww33-2.6-1 / `e9e55b898a1b`.


## 2026-09-03 — HIP wave64 request (not dest)

[TheRock#7909](https://github.com/ROCm/TheRock/issues/7909) (open, filed 2026-09-03 ~15:48 by llama.cpp maintainer `pwilkin`): wants an official HIP path for architecture-specific **wave64** kernels. Cites gfx1151 (Strix Halo) decode lag vs Vulkan. Notes HIP wave-size constant was removed and **`-mwavefrontsize64` is deprecated**. Unresolved. Not a dest lever for gfx1030 (wave32 native). Do not flip extras to wave64.

Adjacent: [ROCm/llvm-project#4267](https://github.com/ROCm/llvm-project/pull/4267) (COMGR hotswap source wave size) was **closed unmerged** 2026-09-03. Not a pin change.

Watch both. Do not treat either as a gfx1030 compiler bump.

## 2026-09-03 — wave64 follow-ups (still not a dest lever)

Two more data points on the same thread, both open:

- [rocm-systems#9647](https://github.com/ROCm/rocm-systems/pull/9647) (`jmmartinez`, updated 2026-09-03) **removes a CLR hack that forced wavefront size 64** when the running executable's name matched Geekbench 5 (`geekbench_x86_64.exe`). Details worth knowing: the hack was **Windows-only, never on Linux**, and it missed the GUI's own `geekbench_avx2.exe`, so AMD's internal numbers were not reproducible by hand (ROCM-24860). Takeaway for us: a *runtime-side* wave-size override has existed in CLR, keyed on process name — so on Windows, wave size is not purely a compile-time property. On Linux/gfx1030 there is no such lever; our wave32 dest is unaffected.
- [ROCm/llvm-project#4213](https://github.com/ROCm/llvm-project/issues/4213) is the Comgr **hotswap** tracker, and it grew a concrete checklist: int VOP1/2/3 arithmetic and VOPD dual-issue ([#4234](https://github.com/ROCm/llvm-project/pull/4234)), int VOP bit ops ([#4251](https://github.com/ROCm/llvm-project/pull/4251)), SOPC compare, SOP2 scalar shifts, wider SMEM (`s_load_b96`), SOPP control flow ([#4079](https://github.com/ROCm/llvm-project/pull/4079)), EXEC-mask handling ([#4081](https://github.com/ROCm/llvm-project/pull/4081)), and `waitcnt` ([#4077](https://github.com/ROCm/llvm-project/pull/4077)). [#4267](https://github.com/ROCm/llvm-project/pull/4267) (source wave size; closed unmerged) was part of the same series. Note **VOPD is gfx11+**, so that piece is a gfx1100 concern, not gfx1030 (GFX10.3 has no dual-issue VOPD).

Neither changes the pin. Keep watching #4213 as the place where hotswap coverage per ISA family is tracked.
