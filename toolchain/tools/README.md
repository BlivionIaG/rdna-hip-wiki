---
owner: ROCM_specialist
updated: 2026-11-18
sources:
  - https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/README.md
  - https://github.com/ROCm/rocm-systems/tree/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/configs
  - https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/shared/machine-readable-isa/isa/amdgpu_isa_rdna2.xml
  - https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html
  - https://github.com/Niko1221/Strata/pull/1167
  - https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/hip-quality-check.yml
  - https://github.com/ggml-org/llama.cpp/pull/29910
  - https://github.com/ROCm/hip-python/issues/111
  - https://github.com/vivienfanghuagood/hk-dsl
  - https://github.com/vivienfanghuagood/hk-dsl/releases/tag/v0.9.10
  - https://pypi.org/project/hk-dsl/0.9.10/
  - https://github.com/ROCm/TheRock/pull/9752
  - https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu/configs
  - https://github.com/vivienfanghuagood/hk-dsl/tree/main/python/hk/target
  - https://github.com/ROCm/TheRock/pull/9822
  - https://github.com/ROCm/rocm-systems/pull/13957
  - https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/README.md
  - https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/loom/docs/src/targets/index.md
  - https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/BUILDING.md
  - https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/libhrx/src/binding/hip/amdhip64.map
  - https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/libhrx/src/binding/common/init.c
  - https://github.com/ROCm/hrx-system/tree/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/docs/reference/amd/gpu
---

# Tools

Date: **2026-10-13**. Profilers and small gates that touch the compiler or the libraries. Engine and fork tooling stays in [engine/](../../engine/) and [fork/](../../fork/). The domain-pass tables are [../log/domain-passes.md](../log/domain-passes.md), kept intact.

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

ROCm 10.1.0 release notes (2026-10-05) say `rocm-smi-lib` is no longer built or installed in the standard TheRock build. `amd-smi` is the replacement.[^rn1010] The live dest is still 7.14.0, so this is a 10.1+ fact. Topology commands that still say `rocm-smi` live in [silicon/interconnect/rccl-p2p.md](../../silicon/interconnect/rccl-p2p.md); switch them to `amd-smi` when a host is actually on 10.1.

## `tune_rocblas` and the spill gate

Strata [#1167](https://github.com/Niko1221/Strata/pull/1167) adds `tools/hip/tune_rocblas`, a per-shape rocBLAS solution index for gfx1030. The measured gains are on [libraries/](../libraries/README.md).[^st1167]

llama.cpp's spill job (`.github/workflows/hip-quality-check.yml` on master, read 2026-10-11) runs `scripts/hip/gcn-cdna-vgpr-check.py` against a build with `-DGPU_TARGETS=gfx908`.[^hipci] The spill counts that motivated pointing the same parser at gfx1030 and gfx1100 are [compiler/](../compiler/README.md) and llama.cpp [#29910](https://github.com/ggml-org/llama.cpp/pull/29910).[^llama29910]

## numba-hip atomics

[hip-python#111](https://github.com/ROCm/hip-python/issues/111) (open): the generated numba-hip bindings treat the pointer argument of HIP atomics as output-only, so `numba.hip.atomic.add` and friends get wrong signatures and kernels that use them do not compile. The reporter is on ROCm 10.0 with an RX 6700 XT and says `atomicAdd` in HIP C++ works. The issue includes a custom `@intrinsic` workaround. An AMD maintainer (`domcharrier`, 2026-10-13) confirmed the analysis for the 10.0 bindings and said the fix will come in the next patch release and/or the 10.1.0 bindings.[^hp111] Python-side kernel prototyping on RDNA through numba-hip should avoid its atomics until then.

## hk-dsl (HipKittens Python DSL)

[vivienfanghuagood/hk-dsl](https://github.com/vivienfanghuagood/hk-dsl) is a Python DSL over HipKittens: AST rewrite, MLIR dialect, and translation to HipKittens C++ with build-time spill / wait / MFMA-hazard checks (`hk.check`).[^hkrepo] Target modules on main (read 2026-11-06): `gfx1100.py`, `gfx1201.py`, `gfx942.py`, `gfx950.py` — **no gfx1030 / gfx900**.[^hktarget]

**Watch (tools)** for gfx1100 craft and for the `hk.check` spill gate pattern. **Leave** dest-produce and engine plugins (vLLM / SGLang route injection is Inference / fork lane). PyPI **0.9.10** (2026-11-06) adds opt-in `HK_VLLM_ROCM_DECODE=1` decode-only ROCM_ATTN path on gfx1100 and AWQ prefill on gfx1201; dialect stay `hk-dialect-8`.[^hk910]

## rocjitsu packaging (tip)

[TheRock#9752](https://github.com/ROCm/TheRock/pull/9752) (merged 2026-11-05) packages `libexec/rocjitsu/rocjitsu-memory` into the rocjitsu artifact so tip `rocm[devel]` nightlies can start the helper. Dest stays 7.14.0 — tip Watch only.[^tr9752] gfx1030 still has **no** topology JSON under `emulation/rocjitsu/configs/` on systems develop (configs still `gfx1100_w7900.json` / `gfx1151.json` / `gfx1201_r9700.json` and CDNA/guest rows).[^rjcfg1106]


## HRX / Loom (ROCm/hrx-system)

[ROCm/hrx-system](https://github.com/ROCm/hrx-system) (Apache-2.0) holds Loom (compiler, runtime and `loom-serve` inference server), `libhrx` (C API with an optional HIP compatibility binding) and `libamdf` (native AMD device access through amdgpu/KFD, with no ROCm SDK or ROCr). The README calls it a tech preview and "not an official component of the ROCm stack".[^hrxreadme]

- **gfx1030 / gfx900: Leave.** The target guide says "GCN, CDNA 1, and RDNA 1/2 do not currently have native Loom compilation paths", so only a Vulkan device can use them through the SPIR-V path.[^hrxtargets] `BUILDING.md` lists `gfx10-3-generic` and `gfx9-generic` as runtime selectors that the Loom compiler does not support.[^hrxbuild] There is no native `v_dot2` / `sdot4` codegen and nothing that plugs into a ROCm 7.x / RCCL stack.
- **gfx1100: Later.** RDNA 3 (`gfx1100`–`gfx1103`) is listed as a primary target with native HSACO execution, rated "Broad" correctness and "Tuned" performance.[^hrxtargets] A `loom-serve` smoke test would be a side experiment, separate from any vLLM stack.
- **HIP binding.** `libhrx/src/binding/hip/amdhip64.map` is a `libamdhip64` symbol version script "generated from TheRock nightly libamdhip64.so exports".[^hrxhip] The binding's device-info code maps `gfx1030` to compute capability 10.3,[^hrxinit] but loading hipcc-built gfx1030 code objects through it is untested here (unsourced as working).
- **Reference docs.** `docs/reference/amd/gpu/` documents PM4, SDMA, queues and counters with footnotes to driver and kernel sources.[^hrxdocs] Cache-policy and silicon claims from it belong on the [silicon/](../../silicon/) pages, not here.

## Sources

TheRock main now pins rocm-systems `16ad3db` ([TheRock#9822](https://github.com/ROCm/TheRock/pull/9822), merged 2026-11-09), which brings in rocjitsu's RDNA comparison, conversion, register/lane-routing and graphics-arithmetic handlers routed through its `rj-arith` library ([#13955](https://github.com/ROCm/rocm-systems/pull/13955), [#13957](https://github.com/ROCm/rocm-systems/pull/13957), [#13958](https://github.com/ROCm/rocm-systems/pull/13958), [#13959](https://github.com/ROCm/rocm-systems/pull/13959)). The PRs say "RDNA" without naming a gfx103x target, and the configs list still has no gfx1030 JSON, so treat this as tip-only emulator progress, not a gfx1030 path.[^tr9822]

[^rj]: rocjitsu README at rocm-systems `65575fbbe6f73b9d9d96bfbcb523c856709e664c`. https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/README.md
[^rjcfg]: `emulation/rocjitsu/configs/` at that commit. Names read 2026-10-11: `gfx1100_w7900.json`, `gfx1151.json`, `gfx1201_r9700.json`, CDNA and guest-gfx950 JSON files, and no gfx1030 file. https://github.com/ROCm/rocm-systems/tree/65575fbbe6f73b9d9d96bfbcb523c856709e664c/emulation/rocjitsu/configs
[^isaxml]: `amdgpu_isa_rdna2.xml` contents API at that commit, `size` 11230558. https://github.com/ROCm/rocm-systems/blob/65575fbbe6f73b9d9d96bfbcb523c856709e664c/shared/machine-readable-isa/isa/amdgpu_isa_rdna2.xml
[^rn1010]: ROCm 10.1.0 release notes, "ROCm SMI is removed from the standard build". https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html
[^st1167]: Strata#1167, open. https://github.com/Niko1221/Strata/pull/1167
[^hipci]: llama.cpp `hip-quality-check.yml` on master. https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/hip-quality-check.yml
[^llama29910]: llama.cpp#29910, open. https://github.com/ggml-org/llama.cpp/pull/29910
[^hp111]: hip-python#111, open, issue body and `domcharrier` comment 2026-10-13. https://github.com/ROCm/hip-python/issues/111
[^tr9822]: TheRock#9822 (rocm-systems 37d5e5d to 16ad3db), merged 2026-11-09, and the four rocm-systems PRs linked above, read 2026-11-09. https://github.com/ROCm/TheRock/pull/9822
[^hrxreadme]: hrx-system README at `39d683333c2b`, read 2026-11-18. https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/README.md
[^hrxtargets]: Loom target guide at `39d683333c2b` (RDNA and CDNA tables, line 70). https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/loom/docs/src/targets/index.md
[^hrxbuild]: `BUILDING.md` AMDGPU selector section at `39d683333c2b` (around line 940). https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/BUILDING.md
[^hrxhip]: `amdhip64.map` header at `39d683333c2b`. https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/libhrx/src/binding/hip/amdhip64.map
[^hrxinit]: `libhrx/src/binding/common/init.c` device-info comment at `39d683333c2b` (line 78). https://github.com/ROCm/hrx-system/blob/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/libhrx/src/binding/common/init.c
[^hrxdocs]: `docs/reference/amd/gpu/` at `39d683333c2b`. https://github.com/ROCm/hrx-system/tree/39d683333c2b5d393ddd51aef1eb3c1d01e9a07e/docs/reference/amd/gpu
