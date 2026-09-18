# Official vs unofficial matrix

Date: **2026-09-10**.

| Piece | Official AMD | This project | Do |
|---|---|---|---|
| HIP runtime + compiler | 10.0.0 latest. **7.14.1** quality | **Live 7.14.0** | Stay on 7.14.0 |
| Wave size | `-mwavefrontsize64` deprecated. [TheRock#7909](https://github.com/ROCm/TheRock/issues/7909) asks for official wave64 (gfx1151) | gfx1030 dest is **wave32** | Do not flip extras to wave64 |
| TheRock `gfx103X-dgpu` | gfx1030 Release Ready; libs excluded. Nightly tip still `10.2.0a20260917` L+W (core + device-gfx1030/1100/900; no 0918 yet); compiler **SMP ww36-2.1** / amd-llvm `16df93c778f8`; systems `d378a17` via [#8258](https://github.com/ROCm/TheRock/pull/8258) (TheRock HEAD `bbd401f0`) | Compiler target is real | Build for `gfx1030`; no CK/hipBLASLt. Live dest still 7.14.0 |
| gfx900 | Official dead | TheRock `device-gfx900` nightly (Path A) | Later host; `mad_mix` |

## 2026-09-03 — wave64 watch

[#7909](https://github.com/ROCm/TheRock/issues/7909) open. [llvm-project#4267](https://github.com/ROCm/llvm-project/pull/4267) COMGR source-wave-size **closed unmerged** (2026-09-03). Neither is a dest pin.
