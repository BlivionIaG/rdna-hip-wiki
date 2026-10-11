---
owner: ROCM_specialist
updated: 2026-10-11
sources:
  - https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/README.md
  - https://github.com/ROCm/rocm-systems/tree/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/configs
  - https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/shared/machine-readable-isa/isa/amdgpu_isa_rdna2.xml
  - https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html
  - https://github.com/Niko1221/Strata/pull/1167
  - https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/hip-quality-check.yml
  - https://github.com/ggml-org/llama.cpp/pull/29910
---

# Tools

Date: **2026-10-11**. Profilers and small gates that touch the compiler or the libraries. Engine and fork tooling stays in [engine/](../../engine/) and [fork/](../../fork/). The domain-pass tables are [../log/domain-passes.md](../log/domain-passes.md), kept intact.

## rocjitsu

rocjitsu lives in the ROCm systems tree at `emulation/rocjitsu` (README at develop `65575fbbe6f73b9d9d96bfbcb523c856709e664c`). Three modes:[^rj]

- **Simulation.** Full ISA emulation via `LD_PRELOAD`. No physical GPU and no kernel module.
- **DBT.** Run a code object built for one architecture on another. The README calls CDNA4→CDNA3 dynamic binary translation experimental. The `configs/` directory at that commit contains `guest_gfx950_on_gfx1201.json`, `guest_gfx950_on_gfx942.json`, and `guest_gfx950_on_simulated_gfx942.json`.
- **DBI.** Runtime instrumentation of kernels for profiling, tracing, and analysis.

Simulation support in that README: gfx1030 is **Experimental**, gfx110x is **Beta**, gfx908 and gfx90a are Experimental. gfx900 is not a row.[^rj]

`configs/` at that commit has `gfx1100_w7900.json` and no file whose name contains `gfx1030`.[^rjcfg] Experimental for gfx1030, in this tree, is the ISA-family row without a topology JSON next to the gfx1100 one.

The race-detector plugin notes from the previous flat page were not re-opened this pass (unsourced). They stay in the [domain-pass log](../log/domain-passes.md).

## ISA XML

`shared/machine-readable-isa/isa/` at the same develop commit contains `amdgpu_isa_rdna2.xml` (11,230,558 bytes, blob `ec3012b7f175f2402aa84086f7b0bdcf07e2786e`) plus rdna1, rdna3, rdna3_5, rdna4, and cdna1–cdna5.[^isaxml] That is a checked-in encoding file. The handbook-versus-XML audit order described on the old tools page was not re-opened (unsourced); see the log.

## `amd-smi` and `rocm-smi`

ROCm 10.1.0 release notes (2026-10-05) say `rocm-smi-lib` is no longer built or installed in the standard TheRock build. `amd-smi` is the replacement.[^rn1010] The live dest is still 7.14.0, so this is a 10.1+ fact. Topology commands that still say `rocm-smi` live in [silicon/rccl-p2p.md](../../silicon/rccl-p2p.md); switch them to `amd-smi` when a host is actually on 10.1.

## `tune_rocblas` and the spill gate

Strata [#1167](https://github.com/Niko1221/Strata/pull/1167) adds `tools/hip/tune_rocblas`, a per-shape rocBLAS solution index for gfx1030. The measured gains are on [libraries/](../libraries/README.md).[^st1167]

llama.cpp's spill job (`.github/workflows/hip-quality-check.yml` on master, read 2026-10-11) runs `scripts/hip/gcn-cdna-vgpr-check.py` against a build with `-DGPU_TARGETS=gfx908`.[^hipci] The spill counts that motivated pointing the same parser at gfx1030 and gfx1100 are [compiler/](../compiler/README.md) and llama.cpp [#29910](https://github.com/ggml-org/llama.cpp/pull/29910).[^llama29910]

## Sources

[^rj]: rocjitsu README at rocm-systems `65575fbbe6f73b9d9d96bfbcb523c856709e664c`. https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/README.md
[^rjcfg]: `emulation/rocjitsu/configs/` at that commit. Names read 2026-10-11: `gfx1100_w7900.json`, `gfx1151.json`, `gfx1201_r9700.json`, CDNA and guest-gfx950 JSON files, and no gfx1030 file. https://github.com/ROCm/rocm-systems/tree/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/configs
[^isaxml]: `amdgpu_isa_rdna2.xml` contents API at that commit, `size` 11230558. https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/shared/machine-readable-isa/isa/amdgpu_isa_rdna2.xml
[^rn1010]: ROCm 10.1.0 release notes, "ROCm SMI is removed from the standard build". https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html
[^st1167]: Strata#1167, open. https://github.com/Niko1221/Strata/pull/1167
[^hipci]: llama.cpp `hip-quality-check.yml` on master. https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/hip-quality-check.yml
[^llama29910]: llama.cpp#29910, open. https://github.com/ggml-org/llama.cpp/pull/29910
