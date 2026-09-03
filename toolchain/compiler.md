# Compiler / hipcc

Date: **2026-09-03**. Audience: someone compiling HIP for **gfx1030** (live), with later **gfx1100** and **gfx900** TUs.

Companions: [silicon/hip-craft.md](../silicon/hip-craft.md).

## Live rule

- Compiler in every existing build line is **`hipcc --offload-arch=gfx1030`**.
- Default is **wave32 + WGP**. Do not add `-mwavefrontsize64` or `-mcumode` until measured. HIP: warpSize 64 is not supported on gfx10+.
- Live dest stays **ROCm 7.14.0** (`rocm-7.14.0` @ `830cc1b5e90d`). 7.14.1 is quality-only (no GitHub `rocm-7.14.1` tag). Do not bump to 10.0 / 10.1 nightlies.
- Live extras stay **`-O3`**. [TheRock#7751](https://github.com/ROCm/TheRock/issues/7751) (gfx1034 `-O0` i32 `udiv`/`urem`) is still open.

## 2026-09-03 — HIP wave64 request (not dest)

[TheRock#7909](https://github.com/ROCm/TheRock/issues/7909) (open, filed 2026-09-03 ~15:48 Paris by llama.cpp maintainer `pwilkin`): wants an official HIP path for architecture-specific **wave64** kernels. Cites gfx1151 (Strix Halo) decode lag vs Vulkan. Notes HIP wave-size constant was removed and **`-mwavefrontsize64` is deprecated**. Unresolved. Not a dest lever for gfx1030 (wave32 native). Do not flip extras to wave64.

Adjacent, still open: [ROCm/llvm-project#4267](https://github.com/ROCm/llvm-project/pull/4267) — COMGR hotswap asks the projection for the **source** wave size. Not a pin change.

Watch both. Do not treat either as a gfx1030 compiler bump.
