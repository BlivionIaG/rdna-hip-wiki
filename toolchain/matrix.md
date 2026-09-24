# Official vs unofficial matrix

Date: **2026-09-24**.

| Piece | Official AMD | This project | Do |
|---|---|---|---|
| HIP runtime + compiler | 10.0.0 latest. **7.14.1** quality | **Live 7.14.0** | Stay on 7.14.0 |
| Wave size | `-mwavefrontsize64` deprecated. [TheRock#7909](https://github.com/ROCm/TheRock/issues/7909) asks for official wave64 (gfx1151) | gfx1030 dest is **wave32** | Do not flip extras to wave64 |
| TheRock `gfx103X-dgpu` | gfx1030 Release Ready; libs excluded. Nightly tip **`10.2.0a20260924` L+W** (core + libraries + device-gfx1030/1100/900); compiler **ww-37-SMP1.1** / amd-llvm `4f43f4746ede` via [#8430](https://github.com/ROCm/TheRock/pull/8430) (base [#8369](https://github.com/ROCm/TheRock/pull/8369)); systems `9799b78` ([#8449](https://github.com/ROCm/TheRock/pull/8449)); libraries `5911365` ([#8444](https://github.com/ROCm/TheRock/pull/8444)); HEAD `0c46fb3364` | Compiler target is real | Build for `gfx1030`; no CK/hipBLASLt. Live dest still 7.14.0 |
| gfx900 | Official dead | TheRock `device-gfx900` nightly (Path A) | Later host; `mad_mix` |


## 2026-09-24 — systems/libraries pin move (not dest)

TheRock [#8449](https://github.com/ROCm/TheRock/pull/8449) **merged** systems **`0816fc8` → `9799b78`**; [#8444](https://github.com/ROCm/TheRock/pull/8444) **merged** libraries **`2e8f62c` → `5911365`**. TheRock HEAD **`9e795452` → `0c46fb3364`**. Compiler **unchanged**: ww-37-SMP1.1 / amd-llvm `4f43f4746ede`. Nightly **unchanged**: `10.2.0a20260924` L+W (core + libraries + device-gfx1030/1100/900). Systems delta is HIP/CLR pool/mem/FP4–FP6 host convert etc — **not a gfx1030/1100/900 ISA lever**; libraries delta is CK/TensileLite/hipBLASLt — **zero RDNA keyword lever**. The nullstream + `hipEventSynchronize` deadlock fix was reverted in that systems range; **watch only**. Official Core SDK remains **10.0.0**. **No RDNA dest bump** — live dest stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP.

## 2026-09-24 — nightly tip 0924 + #7751 closed (not dest)

Nightly tip **`10.2.0a20260923` → `10.2.0a20260924` L+W** for core + libraries + device-gfx1030/1100/900. TheRock HEAD **`e0e238c0` → `9e795452`** (CI/docker/Windows mesa only). Compiler still **ww-37-SMP1.1** / `4f43f4746ede`; systems `0816fc8`; libraries `2e8f62c`. [TheRock#7751](https://github.com/ROCm/TheRock/issues/7751) **closed** (nightly-claimed `-O0` udiv/urem fix; also gfx1100). **No gfx1030/1100/900 ISA / dest lever.** Live dest still **7.14.0**. Support table unchanged (gfx1030/1100 Release Ready; gfx900 Build Passing).

## 2026-09-23 — HEAD e0e238c0 + sqtt-marker packaging (not dest)

TheRock [#7972](https://github.com/ROCm/TheRock/pull/7972) ships Linux `sqtt-marker` with amd-llvm (**SHA unchanged** `4f43f4746ede`). HEAD **`46441674` → `e0e238c0`**. Nightly tip still **`10.2.0a20260923` L+W**. systems `0816fc8`; libraries `2e8f62c`. **No gfx1030/1100/900 ISA / dest lever.** Live dest still **7.14.0**. Support table unchanged.

## 2026-09-23 — compiler pin 4f43f4746ede (not dest)

TheRock [#8430](https://github.com/ROCm/TheRock/pull/8430) amd-llvm **`528402e03c23` → `4f43f4746ede`** (ww-37-SMP1.1 + gfx1250-strict offload-arch CPs only). HEAD **`b617466b` → `46441674`**. Nightly tip still **`10.2.0a20260923` L+W**. libraries still `2e8f62c`; systems still `0816fc8`. **No gfx1030/1100/900 ISA / dest lever.** Live dest still **7.14.0**. Support table unchanged (gfx1030/1100 Release Ready; gfx900 Build Passing). Draft COT follow-on [#8445](https://github.com/ROCm/TheRock/pull/8445).

## 2026-09-23 — libraries pin 2e8f62c (not dest)

TheRock [#8416](https://github.com/ROCm/TheRock/pull/8416) libraries **`9bb8b54` → `2e8f62c`**; HEAD **`70648e2e` → `b617466b`**. Nightly tip still **`10.2.0a20260923` L+W**. Compiler still **ww-37-SMP1.1** / `528402e03c23`; systems still `0816fc8`. Library delta is rocSPARSE / TensileLite gfx1250 / rocke WaveScope / MIOpen binary pointwise — **no gfx1030 ISA / dest lever**. Live dest still **7.14.0**. Support table unchanged (gfx1030/1100 Release Ready; gfx900 Build Passing). Follow-on [#8444](https://github.com/ROCm/TheRock/pull/8444) later merged to `5911365`.

## 2026-09-23 — nightly tip 0923 (not dest)

Nightly tip **`10.2.0a20260922` → `10.2.0a20260923` L+W** for core + libraries + device-gfx1030/1100/900. Compiler still **ww-37-SMP1.1** / `528402e03c23`; systems `0816fc8`; libraries `9bb8b54`. TheRock HEAD **`70648e2e`** (CI/packaging only since prior tip). Live dest still **7.14.0**. Support table unchanged (gfx1030/1100 Release Ready; gfx900 Build Passing).

## 2026-09-22 — libraries pin 9bb8b54 (not dest)

TheRock [#8406](https://github.com/ROCm/TheRock/pull/8406) libraries **`994fed0` → `9bb8b54`**; HEAD **`40d2bb4` → `e9ae0596`**. Nightly tip still **`10.2.0a20260922` L+W**. Compiler still **ww-37-SMP1.1** / `528402e03c23`; systems still `0816fc8`. RDNA-adjacent library notes only (MIOpen gfx1100 CK config drop; hipBLASLt Navi dict refactor) — **no gfx1030 ISA / dest lever**. Live dest still **7.14.0**. Support table unchanged. mesa-fork [#8370](https://github.com/ROCm/TheRock/pull/8370) ignored for HIP dest.

## 2026-09-22 — nightly tip 0922 + libraries/systems pins (not dest)

Nightly tip **`10.2.0a20260921` → `10.2.0a20260922` L+W** for core + device-gfx1030/1100/900 on `nightly.repo.amd.com`. TheRock [#8386](https://github.com/ROCm/TheRock/pull/8386) systems `9f9214b` → `0816fc8`; [#8392](https://github.com/ROCm/TheRock/pull/8392) libraries `3082a49` → `994fed0` (gfx950/gfx1250/gfx803-focused; no gfx1030 ISA lever). Compiler still **ww-37-SMP1.1** / `528402e03c23`. HEAD **`40d2bb4`**. Live dest still **7.14.0**. Support table unchanged.

## 2026-09-21 — ww-37-SMP1.1 + nightly tip (not dest)

Nightly tip still **`10.2.0a20260921` L+W** for core + device-gfx1030/1100/900 (no newer dated wheels). Compiler **ww-37-SMP1.1** / amd-llvm `528402e03c23` via [TheRock#8369](https://github.com/ROCm/TheRock/pull/8369) (OpenMP hwloc INSTALL_INTERFACE cherry-pick; not an RDNA ISA lever). Prior same-day: [#8306](https://github.com/ROCm/TheRock/pull/8306) ww37.1.0 / `6bd80f15ed27`; [#8385](https://github.com/ROCm/TheRock/pull/8385) libraries → `3082a49`. TheRock HEAD **`3a02cd2c`**. systems `9f9214b`. Open [#8392](https://github.com/ROCm/TheRock/pull/8392) (`994fed0`) still unmerged. Live dest still **7.14.0**. Support table unchanged (gfx1030/1100 Release Ready; gfx900 Build Passing).

## 2026-09-03 — wave64 watch

[#7909](https://github.com/ROCm/TheRock/issues/7909) open. [llvm-project#4267](https://github.com/ROCm/llvm-project/pull/4267) COMGR source-wave-size **closed unmerged** (2026-09-03). Neither is a dest pin.
