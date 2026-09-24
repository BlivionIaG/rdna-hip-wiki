# Compiler / hipcc

Date: **2026-09-24**. Audience: someone compiling HIP for **gfx1030** (live), with later **gfx1100** and **gfx900** TUs.

Companions: [silicon/hip-craft.md](../silicon/hip-craft.md).

## 2026-09-24 — systems/libraries pin move (not dest)

TheRock [#8449](https://github.com/ROCm/TheRock/pull/8449) **merged** systems **`0816fc8` → `9799b78`**; [#8444](https://github.com/ROCm/TheRock/pull/8444) **merged** libraries **`2e8f62c` → `5911365`**. TheRock HEAD **`9e795452` → `0c46fb3364`**. Compiler **unchanged**: ww-37-SMP1.1 / amd-llvm `4f43f4746ede`; nightly **unchanged**: `10.2.0a20260924` L+W (core + libraries + device-gfx1030/1100/900). Systems delta is HIP/CLR pool/mem/FP4–FP6 host convert etc — **not a gfx1030/1100/900 ISA lever**; libraries delta is CK/TensileLite/hipBLASLt — **zero RDNA keyword lever**. The nullstream + `hipEventSynchronize` deadlock fix was reverted in that systems range; **watch only**. Official Core SDK remains **10.0.0**. **No RDNA dest bump** — live dest stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Drop merged #8449 and #8444 from open watches; keep watching systems [#8442](https://github.com/ROCm/TheRock/pull/8442)/[#8440](https://github.com/ROCm/TheRock/pull/8440)/[#8418](https://github.com/ROCm/TheRock/pull/8418), draft COT [#8445](https://github.com/ROCm/TheRock/pull/8445), [#7909](https://github.com/ROCm/TheRock/issues/7909)/[#7976](https://github.com/ROCm/TheRock/issues/7976), vLLM#52391, and SGLang#37398.

## 2026-09-24 — nightly tip 0924 + #7751 closed (not dest)

Nightly tip **`10.2.0a20260923` → `10.2.0a20260924` L+W** (core + libraries + device-gfx1030/1100/900) on `https://nightly.repo.amd.com/rocm/whl-next/`. TheRock HEAD **`e0e238c0` → `9e795452`** overnight (#8435 CI gfx `ci:` labels, #8460 manylinux Python 3.15, #8464 compiler-runtime large runner, #8451 Windows mesa-fork xcopy) — **no submodule pin move**. Compiler still **ww-37-SMP1.1** / amd-llvm `4f43f4746ede`; systems still `0816fc8`; libraries still `2e8f62c`; hipify `501cd6c1`; spirv `2c14c774`. [TheRock#7751](https://github.com/ROCm/TheRock/issues/7751) **closed** 2026-09-23 21:19 Europe/Paris (schung-amd): gfx1034 `-O0` i32 `udiv`/`urem` float-reciprocal miscompile claimed fixed in current nightlies via [llvm#201186](https://github.com/llvm/llvm-project/pull/201186) / [llvm#202753](https://github.com/llvm/llvm-project/pull/202753); also repro'd on gfx1100. **Not a live-dest lever** — extras stay `-O3` on **7.14.0**. AMD pin on TheRock main still pre-dates those llvm merges in the submodule SHA; treat nightlies as the claimed fix surface, reopen if `-O0` still wrong. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Open watches: systems [#8442](https://github.com/ROCm/TheRock/pull/8442)/[#8440](https://github.com/ROCm/TheRock/pull/8440)/[#8418](https://github.com/ROCm/TheRock/pull/8418); draft COT [#8445](https://github.com/ROCm/TheRock/pull/8445); [#7909](https://github.com/ROCm/TheRock/issues/7909)/[#7976](https://github.com/ROCm/TheRock/issues/7976); vLLM#52391; SGLang#37398. Drop closed #7751.

## 2026-09-23 — HEAD e0e238c0 + sqtt-marker packaging (not dest)

TheRock [#7972](https://github.com/ROCm/TheRock/pull/7972) **merged** 2026-09-23 17:06 Europe/Paris (`f8ed5f01`). Linux amd-llvm builds now ship `sqtt-marker` as an LLVM external project (`LLVM_EXTERNAL_PROJECTS` + `libsqtt-marker`; loaded via `-fpass-plugin`; not dlopened standalone; Windows skipped). **Profiler pass packaging only — amd-llvm SHA still `4f43f4746ede` / ww-37-SMP1.1; no gfx1030/1100/900 ISA lever.** Same-hour PyTorch CI [#8426](https://github.com/ROCm/TheRock/pull/8426)/[#8113](https://github.com/ROCm/TheRock/pull/8113) ignored. HEAD **`46441674` → `e0e238c0`**. Nightly tip still **`10.2.0a20260923` L+W**; systems `0816fc8`; libraries `2e8f62c`. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Open watches unchanged (#8442/#8440/#8418/#8445/#7751/#7909/#7976).

## 2026-09-23 — compiler pin 4f43f4746ede (not dest)

TheRock [#8430](https://github.com/ROCm/TheRock/pull/8430) **merged** 2026-09-23 16:02 Europe/Paris (`3efd90cc`). amd-llvm **`528402e03c23` → `4f43f4746ede`** on the same **ww-37-SMP1.1** base (+2 cherry-picks: gfx1250-strict offload-arch report to match rocminfo; `HSA_DISABLE_GFX12_STRICT` env). **gfx1250 A0 only — not an RDNA ISA lever for gfx1030/1100/900.** hipify `501cd6c1` / spirv `2c14c774` unchanged. Same-hour [#8420](https://github.com/ROCm/TheRock/pull/8420) libdrm MI350P VF (CDNA) — ignore for RDNA. HEAD **`b617466b` → `46441674`**. libraries still `2e8f62c`; systems still `0816fc8`; nightly tip still **`10.2.0a20260923` L+W** (no `a20260924`). **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Open: systems [#8442](https://github.com/ROCm/TheRock/pull/8442)/[#8440](https://github.com/ROCm/TheRock/pull/8440)/[#8418](https://github.com/ROCm/TheRock/pull/8418); draft COT [#8445](https://github.com/ROCm/TheRock/pull/8445) (replaces watch on merged #8430); [#7751](https://github.com/ROCm/TheRock/issues/7751)/[#7909](https://github.com/ROCm/TheRock/issues/7909)/[#7976](https://github.com/ROCm/TheRock/issues/7976).

## 2026-09-23 — libraries pin 2e8f62c (not dest)

TheRock [#8416](https://github.com/ROCm/TheRock/pull/8416) **merged** 2026-09-23 13:28 Europe/Paris (`b617466b`). libraries **`9bb8b54` → `2e8f62c`** (+10; rocSPARSE lane/overflow fixes, TensileLite **gfx1250** marks/SIA4, rocke WaveScope PMC, MIOpen standalone binary pointwise, CK CI revert). **No gfx1030/1100/900 ISA or dest lever.** Compiler pin **unchanged** (ww-37-SMP1.1 / amd-llvm `528402e03c23`; hipify `501cd6c1`; spirv `2c14c774`). systems still `0816fc8`; nightly tip still **`10.2.0a20260923` L+W** (no `a20260924`). HEAD **`70648e2e` → `b617466b`**. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Open: systems [#8442](https://github.com/ROCm/TheRock/pull/8442)/[#8440](https://github.com/ROCm/TheRock/pull/8440)/[#8418](https://github.com/ROCm/TheRock/pull/8418); draft [#8412](https://github.com/ROCm/TheRock/pull/8412) COT; [#8430](https://github.com/ROCm/TheRock/pull/8430) ww37 offload-arch strict (gfx1250). Drop closed [#8416] from watches.

## 2026-09-23 — nightly tip 0923 (not dest)

Nightly tip **`10.2.0a20260922` → `10.2.0a20260923` L+W** (core + libraries + device-gfx1030/1100/900) on `https://nightly.repo.amd.com/rocm/whl-next/`. Compiler pin **unchanged** (ww-37-SMP1.1 / amd-llvm `528402e03c23`; hipify `501cd6c1`; spirv `2c14c774`). systems still `0816fc8`; libraries still `9bb8b54`; mesa-fork still `38c6ef8`. TheRock HEAD **`e49f29a2` → `70648e2e`** overnight (CI/packaging/docs/#8377 target-ownership metadata/#8408 rocgdb/#8071 hipcc_migration doc only — no submodule pin). **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Open: [#8418](https://github.com/ROCm/TheRock/pull/8418) systems, [#8416](https://github.com/ROCm/TheRock/pull/8416) libraries, draft [#8412](https://github.com/ROCm/TheRock/pull/8412) COT.

## 2026-09-22 — libraries pin 9bb8b54 (not dest)

TheRock [#8406](https://github.com/ROCm/TheRock/pull/8406) **merged** 2026-09-22 15:13 Europe/Paris (`e9ae0596`). libraries **`994fed0` → `9bb8b54`** (+34; mostly gfx950/gfx1250/CK/docs). RDNA-adjacent only: MIOpen drop of CK perf configs invalidated by large-tensor gate on **gfx1100** (rocm-libraries#12117); hipBLASLt Navi library-logic dict refactor (#12361). Also [#8370](https://github.com/ROCm/TheRock/pull/8370) mesa-fork sysdeps `667584b`→`38c6ef8` (not HIP/runtime). Compiler pin **unchanged** (ww-37-SMP1.1 / amd-llvm `528402e03c23`); systems still `0816fc8`; nightly tip still **`10.2.0a20260922` L+W**. HEAD **`e9ae0596`**. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Open: draft [#8412](https://github.com/ROCm/TheRock/pull/8412) COT compiler, [#8409](https://github.com/ROCm/TheRock/pull/8409)/[#8404](https://github.com/ROCm/TheRock/pull/8404) systems. Drop closed [#8410](https://github.com/ROCm/TheRock/pull/8410) (superseded by #8406).

## 2026-09-22 — nightly tip 0922 + systems/libraries pins (not dest)

Nightly tip **`10.2.0a20260921` → `10.2.0a20260922` L+W** (core + device-gfx1030/1100/900). Compiler pin **unchanged** (ww-37-SMP1.1 / amd-llvm `528402e03c23`). TheRock [#8386](https://github.com/ROCm/TheRock/pull/8386) systems → `0816fc8`; [#8392](https://github.com/ROCm/TheRock/pull/8392) libraries → `994fed0`. HEAD **`40d2bb4`**. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Follow-ons at that tip: [#8404](https://github.com/ROCm/TheRock/pull/8404) systems still open; [#8406](https://github.com/ROCm/TheRock/pull/8406) libraries later merged (see tip above).

## 2026-09-21 — TheRock ww-37-SMP1.1 (nightly pin, not dest)

[TheRock#8369](https://github.com/ROCm/TheRock/pull/8369) **merged** 2026-09-21 15:38 Europe/Paris (`c18fccc2`). Compiler pin **SMP ww37.1.0 / amd-llvm `6bd80f15ed27` → ww-37-SMP1.1 / `528402e03c23`**. Single cherry-pick: LLVM OpenMP hwloc `INSTALL_INTERFACE` fix (`bf5bef4` / [llvm#218923](https://github.com/llvm/llvm-project/pull/218923)) — **build/host OpenMP packaging**, not an RDNA ISA lever for gfx1030/1100/900. hipify/spirv unchanged in this bump. TheRock HEAD advanced past libraries [#8385](https://github.com/ROCm/TheRock/pull/8385) (`90d12f3a`) through build-graph cleanups to **`3a02cd2c`** (also `#7002`/`#8345`/`#8323`). Nightly tip still **`10.2.0a20260921` L+W** (no newer dated wheels). systems still `9f9214b`; libraries still `3082a49`. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP.

## 2026-09-21 — TheRock SMP ww37.1.0 (nightly pin, not dest)

[TheRock#8306](https://github.com/ROCm/TheRock/pull/8306) **merged** 2026-09-18 19:19 Europe/Paris (`2dbede93`) — landed just after the 19:11 scan that still had it open. Compiler pin **SMP ww36-2.1 / amd-llvm `16df93c778f8` → ww37.1.0 / `6bd80f15ed27`**. hipify `06ebcc28` → `501cd6c1`; spirv `0dcc5cc5` → `2c14c774`. CP window is mostly **gfx1250-strict** instruction disables + **Comgr hotswap** expansion (VOPD, SOP1 bitops, SOPP fences, wider SMEM) + general AMDGPU (DPP combine, WWM RegisterClassInfo refresh, PromoteAlloca, waitcnt `BUFFER_INV`, TargetParser VGPR APIs). **No RDNA ISA dest lever** for gfx1030/1100/900. FDOT2 fold fix for non-constant lane indices is in the window but is not a live gfx1030 flip (and gfx900 still has no fdot2).

Same-evening compiler layout: [#8307](https://github.com/ROCm/TheRock/pull/8307) re-enables `LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON` (compat-symlink path after the earlier host-OFF revert); [#8358](https://github.com/ROCm/TheRock/pull/8358) points compiler-rt device cache at `AMDGPU.cmake`.

Nightly tip **`10.2.0a20260918` → `10.2.0a20260921` L+W** (core + device-gfx1030/1100/900; 0920 is also published; no 0919). Published nightly tip `90fea14c`; systems `9f9214b`; libraries `80414d8` at the scan's prior tip. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Drop `#8306` from watches.

Post-13:01 CEST movement: [TheRock#8385](https://github.com/ROCm/TheRock/pull/8385) merged 2026-09-21 13:12 CEST, bumping rocm-libraries `80414d8` → `3082a49` (hipBLASLt bad-tile removal for gfx1250 and Origami TemporalHints; neither is a gfx1030/1100/900 compiler/runtime pin). Open [#8392](https://github.com/ROCm/TheRock/pull/8392) proposes `3082a49` → `994fed0`, not merged at scan time.

RCCL (systems, not compiler): [rocm-systems#11752](https://github.com/ROCm/rocm-systems/pull/11752) restore Navi opts post-NCCL sync; [rocm-systems#11803](https://github.com/ROCm/rocm-systems/pull/11803) gfx110x AlltoAll channels=1 temp fix (variance). Track under comms; do not change dest AR knobs from this alone.

## 2026-09-17 — TheRock SMP ww36-2.1 (nightly pin, not dest)

[TheRock#8266](https://github.com/ROCm/TheRock/pull/8266) **merged** 2026-09-17 14:19 Europe/Paris (`fd55c6ac`). Compiler pin **SMP ww36-2.0 / amd-llvm `bc1e171b6a53` → ww36-2.1 / `16df93c778f8`**. hipify `06ebcc28` and spirv `0dcc5cc5` **unchanged**. Single CP: revert `[clang][DebugInfo] Emit static local variables in their lexical block scope` ([ROCm/llvm-project#4465](https://github.com/ROCm/llvm-project/pull/4465) / `16df93c778f8`) — ASAN-DEBUG build fix (LCOMPILER-2794), **not** an RDNA ISA lever. TheRock HEAD was `fd55c6ac` at merge; later same day systems bump [#8258](https://github.com/ROCm/TheRock/pull/8258) → HEAD `bbd401f0`, systems `d378a17` (compiler pin unchanged). Libraries still `320d658`. Nightly tip still **`10.2.0a20260917` L+W** (core + device-gfx1030/1100/900); no `a20260918`, so published wheels do **not** yet carry ww36-2.1. **No RDNA dest bump** — live stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP. Still open: systems `#8288`, libraries `#8289`, COT+ASAN `#8248`. Drop `#8266` from watches.

## Live rule

- Compiler in every existing build line is **`hipcc --offload-arch=gfx1030`**.
- Default is **wave32 + WGP**. Do not add `-mwavefrontsize64` or `-mcumode` until measured. HIP: warpSize 64 is not supported on gfx10+.
- Live dest stays **ROCm 7.14.0** (`rocm-7.14.0` @ `830cc1b5e90d`). 7.14.1 is quality-only (no GitHub `rocm-7.14.1` tag). Do not bump to 10.0 / 10.1 nightlies.
- Live extras stay **`-O3`**. [TheRock#7751](https://github.com/ROCm/TheRock/issues/7751) (gfx1034/gfx1100 `-O0` i32 `udiv`/`urem`) **closed** 2026-09-23 — claimed fixed in nightlies via llvm#201186/#202753; keep `-O3` on live 7.14.0 regardless.




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
