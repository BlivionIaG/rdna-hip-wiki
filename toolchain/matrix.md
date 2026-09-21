# Official vs unofficial matrix

Date: **2026-09-21**.

| Piece | Official AMD | This project | Do |
|---|---|---|---|
| HIP runtime + compiler | 10.0.0 latest. **7.14.1** quality | **Live 7.14.0** | Stay on 7.14.0 |
| Wave size | `-mwavefrontsize64` deprecated. [TheRock#7909](https://github.com/ROCm/TheRock/issues/7909) asks for official wave64 (gfx1151) | gfx1030 dest is **wave32** | Do not flip extras to wave64 |
| TheRock `gfx103X-dgpu` | gfx1030 Release Ready; libs excluded. Nightly tip **`10.2.0a20260921` L+W** (core + device-gfx1030/1100/900); compiler **SMP ww37.1.0** / amd-llvm `6bd80f15ed27`; systems `9f9214b`; libraries `3082a49` after [#8385](https://github.com/ROCm/TheRock/pull/8385); published nightly tip `90fea14c` | Compiler target is real | Build for `gfx1030`; no CK/hipBLASLt. Live dest still 7.14.0 |
| gfx900 | Official dead | TheRock `device-gfx900` nightly (Path A) | Later host; `mad_mix` |


## 2026-09-21 — SMP ww37.1.0 + nightly tip (not dest)

Nightly tip **`10.2.0a20260921` L+W** (was `a20260918`) for core + device-gfx1030/1100/900. Compiler **SMP ww37.1.0** / amd-llvm `6bd80f15ed27` via [TheRock#8306](https://github.com/ROCm/TheRock/pull/8306). systems `9f9214b`, libraries `80414d8` at prior tip; [TheRock#8385](https://github.com/ROCm/TheRock/pull/8385) merged after the 13:01 CEST baseline to `3082a49`. The 0920 and 0921 wheels are published; open [#8392](https://github.com/ROCm/TheRock/pull/8392) proposes `994fed0` and is not merged. RCCL: Navi opts restored ([rocm-systems#11752](https://github.com/ROCm/rocm-systems/pull/11752)); gfx110x AlltoAll 1-channel temp ([rocm-systems#11803](https://github.com/ROCm/rocm-systems/pull/11803)). Live dest still **7.14.0**. Support table unchanged (gfx1030/1100 Release Ready; gfx900 Build Passing).

## 2026-09-03 — wave64 watch

[#7909](https://github.com/ROCm/TheRock/issues/7909) open. [llvm-project#4267](https://github.com/ROCm/llvm-project/pull/4267) COMGR source-wave-size **closed unmerged** (2026-09-03). Neither is a dest pin.
