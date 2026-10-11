# ROCm host — official matrix vs this box

Date: 2026-08-19. Engine contract. Do **not** treat AMD’s OS footnote as “this box cannot run.” Occupancy still first.

## Official (AMD docs — do not erase)

ROCm **7.2.0** and **7.14.0** system-requirements: V620 / gfx1030 is ✅ with footnote **Ubuntu 24.04.x and 22.04.5 only**. RHEL is in the *general* ROCm table; the V620 footnote **excludes** it (`RHEL 10.1 / 9.7 … except AMD Radeon PRO V620`).

Sources: [7.2 system-requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/system-requirements.html) footnote [8]; [latest](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) footnote [8] / [9].

That is **QA scope**, not physics. Fedora is not in the matrix. “Not on RHEL” in [silicon/rccl-p2p.md](../../silicon/rccl-p2p.md) meant the footnote, not a measured fail.

## Attested (2026-08-19)

| Stack | Status |
|---|---|
| ROCm **7.2.0** on **Fedora 43** | Works |
| **RHEL 10** | Works |
| ROCm **7.14.0** (current) | Works |

No tok/s claimed. This is host/runtime, not a kernel win.

## Engine rule

- **Live target:** ROCm **7.14.0** on this host (Fedora 43 and/or RHEL 10 as used).
- **Official matrix:** keep quoting Ubuntu-only for V620 so we do not file AMD bugs as if Fedora/RHEL were supported SKUs.
- Docker (`vllm-rdna-docker`) may still be Ubuntu — that is image policy, not a host ban.
- *(historical)* The `rdna2_extras` (BlivionIaG/vllm) PYNCCL bypass (`3e05abc9`) was for Torch 2.12 + a 7.14-class venv. The current all-reduce lock is in [fork/serve-locks.md](../../fork/serve-locks.md); this page does not set it.
- **Host gate:** the host kernel's amdgpu must report DRM **≥ 3.64** (`dmesg | grep "Initialized amdgpu"`). ROCm 7.14-era userspace on older in-kernel amdgpu gives silent RCCL `all_gather` / `reduce_scatter` corruption (Ubuntu 24.04 stock 6.8) and a HIP VMM hang below 3.64; `HSA_DISABLE_CACHE=1` is only a stopgap. Record the DRM version next to the ROCm version in bring-up and bug reports. Detail: [toolchain/hip-runtime.md](../../toolchain/hip-runtime.md) ("Gotcha — the host kernel's amdgpu DRM version")[^drm-6565][^drm-12145][^drm-9820].
- Do not mix V340L onto this ROCm 7 host ([plx.md](../multi-gpu/plx.md)).

retip the “Not on RHEL” cell in `silicon/rccl-p2p.md` to **official Ubuntu-only / attested Fedora 43 + RHEL 10 + 7.14**.

## ROCm 10.x — do not bump

Pin and status live in [toolchain/matrix.md](../../toolchain/matrix.md); this section only keeps the engine reading.

- **ROCm 10.1.0 docs and compatibility matrix are live (since 2026-10-06)**[^rocm1010]. Per toolchain/matrix.md, Core SDK 10.1.0 still lists **gfx1030** (V620, W6800) and gfx1100/1101/1102; the framework support lists exclude gfx1030. Breaking for host scripts: `rocm-smi` is no longer the default tool, use `amd-smi`.
- ROCm 10.0 (2026-08-27): [Newsroom](https://newsroom.amd.com/news/rocm-10-software-ai-native-developer-experiences/) is PR. [Blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html) is TheRock-on-7.14 plus ROCm.AI (CLI / amd/skills / Hyperloom). Skills/Hyperloom/vLLM adapters are Instinct (MI300/325/350/355) and EPYC. Official 3.3×/2.4× is MI355x — do not copy.
- hipBLASLt local tuner is still not a gfx1030 dest (TheRock #1062[^therock1062]). FlyDSL/CK/GEAK/TileLang stay Leave. Roofline in rocprofiler-compute is **GFX11xx**, not gfx1030.
- Silicon: [silicon/rccl-p2p.md](../../silicon/rccl-p2p.md) (GIN ≠ 88096 PIX).
- **gfx1030 not dropped (checked 2026-08-28):** TheRock `main` `cmake/therock_amdgpu_targets.cmake` still adds `gfx1030` (`gfx103X-dgpu`). Excludes are libraries (hipBLASLt #1062, CK, rocWMMA, hipTensor, hipSPARSELt, rocprofiler-compute) — not the target. RCCL stays. The ROCm 10.0.0 matrix listed V620 / W6800 (Ubuntu-only V620 footnote).

## Sources

- Note 2026-08-19
- AMD ROCm 7.2 / 7.14 install system-requirements (V620 footnote)

Added in the 2026-10-11 sourcing pass:

- AMD Radeon PRO V620 product page — <https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html>
- ROCm 10.1.0 release notes — <https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html>
- [toolchain/matrix.md](../../toolchain/matrix.md) (2026-10-06 entry), [toolchain/hip-runtime.md](../../toolchain/hip-runtime.md), [toolchain/comms.md](../../toolchain/comms.md)

[^drm-6565]: ROCm/legacy-rocm-build#6565, silent RCCL corruption on 2× gfx1100 (open) — <https://github.com/ROCm/legacy-rocm-build/issues/6565>
[^drm-12145]: ROCm/rocm-systems#12145, first-enumerated GPU RCCL all_gather half-wrong on gfx1030 V620 (open) — <https://github.com/ROCm/rocm-systems/issues/12145>
[^drm-9820]: ROCm/rocm-systems#9820, ROCr VM timeline wait hangs on AMDGPU DRM before 3.64 (open) — <https://github.com/ROCm/rocm-systems/issues/9820>
[^rocm1010]: ROCm 10.1.0 release notes — <https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html>
[^therock1062]: ROCm/TheRock#1062, hipBLASLt gfx archs excluded — <https://github.com/ROCm/TheRock/issues/1062>
