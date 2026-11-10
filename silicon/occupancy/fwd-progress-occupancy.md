---
title: "Kernel descriptor FWD_PROGRESS / MEM_ORDERED vs effective occupancy (gfx1030)"
sources:
  - https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/docs/AMDGPUUsage.md
  - https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/lib/Target/AMDGPU/AMDGPUAsmPrinter.cpp
  - https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/lib/Target/AMDGPU/SIProgramInfo.cpp
  - https://github.com/llvm/llvm-project/commit/c1fabd681fed3e024879f64f2e122e667778b9be
  - https://github.com/llvm/llvm-project/commit/1c49ce676caa161250624714c3698b87dc2f8628
  - https://github.com/llvm/llvm-project/commit/cd68939326ed073ad6d50e034310cc295eddd72e
  - https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
---

# Kernel descriptor FWD_PROGRESS / MEM_ORDERED vs effective occupancy (gfx1030)

Lock: **`COMPUTE_PGM_RSRC1.FWD_PROGRESS` (bit 31) and `MEM_ORDERED` (bit 30) are per-kernel scheduling / counter-reporting bits, not a PIX / `llvm-calc-occupancy` MaxWaves limiter.** They do not add or free VGPR∩LDS slots. `FWD_PROGRESS` picks the SIMD wave-pick policy among waves already resident; `MEM_ORDERED` picks whether `vmcnt` reports in order. The new fact on this page: **`FWD_PROGRESS` flipped from 0 to 1 between ROCm 6.4 and ROCm 7.0.1 clang for gfx1030 HIP**, so a gfx1030 object's wave-pick policy depends on which ROCm built it.[^usage-rsrc1][^probe]

Does **not** change extras HIP or UNC tickets: the dest compile pin is ROCm 7.14, which already emits `FWD_PROGRESS = 1` (probe below). No tok/s. Do not restate the FA pin.

Date: **2026-11-10** Europe/Paris.

Companions: [wave-priority-occupancy.md](wave-priority-occupancy.md) (the arbiter's priority/age pick that `FWD_PROGRESS` modifies), [spin-hwq-occupancy.md](spin-hwq-occupancy.md) (device-side spins), [waitcnt-occupancy.md](waitcnt-occupancy.md) (`vmcnt` parks that `MEM_ORDERED` shapes), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md) (bit 29 `WGP_MODE`, same register), [occupancy-dump.md](occupancy-dump.md).

## 1. What the bits mean (AMDGPUUsage, GFX10–GFX12)

| Bit | Field | 0 | 1 |
|---|---|---|---|
| 29 | `WGP_MODE` | CU wavefront execution mode | WGP wavefront execution mode |
| 30 | `MEM_ORDERED` | `vmcnt` reports load / atomic-with-return completion **out of order** with sample instructions; `vscnt` in order | `vmcnt` reports load, atomic-with-return and sample completion **in order**; `vscnt` in order |
| 31 | `FWD_PROGRESS` | execute SIMD wavefronts using **oldest first** policy | execute SIMD wavefronts **to ensure wavefronts will make some forward progress** |

All three are reserved (must be 0) on GFX6–GFX9 and are copied by CP into `COMPUTE_PGM_RSRC1`.[^usage-rsrc1]

The memory-model section states that GFX10+ wavefronts run in native mode with in-order reporting of loads and sample instructions, which is the `MEM_ORDERED = 1` case.[^usage-memmodel] The assembler defaults for both `.amdhsa_memory_ordered` and `.amdhsa_forward_progress` are listed as **1** for GFX10–GFX12.[^usage-directives]

AMDGPUUsage gives only the one-line policy description for `FWD_PROGRESS`. It does not say how the forward-progress policy picks waves, whether it costs latency, or how it interacts with `S_SETPRIO` priority (see [wave-priority-occupancy.md](wave-priority-occupancy.md)). Those details are (unsourced) here; do not invent them.

## 2. What the compiler emits

Upstream LLVM main (`AMDGPUAsmPrinter.cpp`): for ISA major ≥ 10, `MemOrdered = 1` and `FwdProgress = !F.hasFnAttribute("amdgpu-no-fwd-progress")`; `SIProgramInfo.cpp` ORs `S_00B848_FWD_PROGRESS` and `S_00B848_MEM_ORDERED` into RSRC1.[^asmprinter][^programinfo]

History of `FWD_PROGRESS`:

| Commit | Date (Paris) | Change |
|---|---|---|
| `c1fabd681fed3e024879f64f2e122e667778b9be` (#128367) | 2025-03-18 | Enables `FWD_PROGRESS` unconditionally for GFX10+ on `amdhsa`. Message: "beneficial for algorithms that rely on more complex concurrent coordination and it is generally performance neutral otherwise."[^c1fabd68] |
| `1c49ce676caa161250624714c3698b87dc2f8628` (#139895) | 2025-07-21 | Extends it to PAL (graphics); message reports no significant gains or losses on graphics workloads.[^1c49ce67] |
| `cd68939326ed073ad6d50e034310cc295eddd72e` (#181675) | 2026-02-26 | Adds the IR function attribute `amdgpu-no-fwd-progress` to turn it off per shader.[^cd689393] |

`MEM_ORDERED = 1` was already set for GFX10+ before these commits (the #139895 diff shows `enable_mem_ordered = 1` as unchanged context next to the `enable_fwd_progress 0 → 1` change).[^1c49ce67]

## 3. Probe: which ROCm sets it for gfx1030 (2026-11-10)

Compiler Explorer API, `--offload-arch=<arch> --cuda-device-only -O3`, one trivial HIP kernel, directives kept, reading `.amdhsa_*` from the emitted assembly:[^probe]

| Compiler Explorer id | gfx1030 `forward_progress` | gfx1030 `memory_ordered` | gfx1030 `workgroup_processor_mode` | gfx1100 `forward_progress` |
|---|---|---|---|---|
| `hipclang-rocm-60303` (ROCm 6.3.3) | **0** | 1 | 1 | 0 |
| `hipclang-rocm-60400` (ROCm 6.4.0) | **0** | 1 | 1 | 0 |
| `hipclang-rocm-70001` (ROCm 7.0.1) | **1** | 1 | 1 | 1 |
| `hipclang-rocm-71400` (ROCm 7.14.0, dest pin) | **1** | 1 | 1 | 1 |
| `hipclang-rocm-100000` (ROCm 10.0.0) | **1** | 1 | 1 | 1 |

`.amdhsa_shared_vgpr_count` was 0 everywhere (wave32 HIP).

HIP source has no direct spelling for `amdgpu-no-fwd-progress`; it is an IR string attribute (not probed from clang in this pass).

## 4. Take / Leave

| | |
|---|---|
| **Take (Now, craft)** | When comparing a gfx1030 object across ROCm builds (old wheels, third-party `.hsaco`, a 6.x baseline vs dest 7.14), check `FWD_PROGRESS` in the kernel descriptor. Objects from ROCm ≤ 6.4 run oldest-first; 7.0.1+ run with the forward-progress policy. Treat a spin / ticket / lookback kernel that only hangs or skews on the old build as a possible scheduling-policy difference before blaming the kernel. |
| **Take (Now, dump)** | Add `.amdhsa_forward_progress` and `.amdhsa_memory_ordered` to the [occupancy-dump.md](occupancy-dump.md) read-out when disassembling `-S` output. LLVM's `kd-gfx10.s` objdump test (touched by #128367) covers decoding these fields from a `.kd` symbol.[^c1fabd68] |
| **Leave** | No ticket, no retip, no extras change: extras builds at ROCm 7.14, which already sets the bit. Do not hand-write `.amdhsa_forward_progress 0` or chase `amdgpu-no-fwd-progress` without a measured A/B. Upstream calls the bit "generally performance neutral". |
| **Leave** | Neither bit is an occupancy-calculator input. Keep `min(VGPR, LDS, WG/barrier)` in [occupancy-composite.md](occupancy-composite.md) unchanged. |

## Sources

[^usage-rsrc1]: LLVM `AMDGPUUsage.md` @ `3ad61ff817ac1409d66993cace413fbd8498bfc1`, `compute_pgm_rsrc1` GFX6–GFX12 table, bits 29–31 (`WGP_MODE`, `MEM_ORDERED`, `FWD_PROGRESS`). https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/docs/AMDGPUUsage.md#L7649-L7685
[^usage-memmodel]: LLVM `AMDGPUUsage.md` @ `3ad61ff817ac1409d66993cace413fbd8498bfc1`, GFX10+ memory model ("native mode with in-order reporting of loads and sample instructions"). https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/docs/AMDGPUUsage.md#L12109-L12113
[^usage-directives]: LLVM `AMDGPUUsage.md` @ `3ad61ff817ac1409d66993cace413fbd8498bfc1`, `.amdhsa_memory_ordered` / `.amdhsa_forward_progress` default 1, GFX10–GFX12; `enable_mem_ordered` defaults to 1 for GFX10 onwards. https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/docs/AMDGPUUsage.md#L17915-L17922
[^asmprinter]: LLVM `AMDGPUAsmPrinter.cpp` @ `3ad61ff817ac1409d66993cace413fbd8498bfc1` (`MemOrdered = 1`, `FwdProgress = !F.hasFnAttribute("amdgpu-no-fwd-progress")` for ISA major ≥ 10). https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/lib/Target/AMDGPU/AMDGPUAsmPrinter.cpp#L1504-L1507
[^programinfo]: LLVM `SIProgramInfo.cpp` @ `3ad61ff817ac1409d66993cace413fbd8498bfc1` (`getComputePGMRSrc1Reg` ORs `S_00B848_WGP_MODE`, `S_00B848_MEM_ORDERED`, `S_00B848_FWD_PROGRESS`). https://github.com/llvm/llvm-project/blob/3ad61ff817ac1409d66993cace413fbd8498bfc1/llvm/lib/Target/AMDGPU/SIProgramInfo.cpp#L82-L90
[^c1fabd68]: LLVM commit `c1fabd681fed3e024879f64f2e122e667778b9be`, "[llvm][AMDGPU] Enable FWD_PROGRESS bit for GFX10+ (#128367)", 2025-03-17T23:17:46Z. https://github.com/llvm/llvm-project/commit/c1fabd681fed3e024879f64f2e122e667778b9be
[^1c49ce67]: LLVM commit `1c49ce676caa161250624714c3698b87dc2f8628`, "[AMDGPU] Enable FWD_PROGRESS bit for GFX10+ on PAL (#139895)", 2025-07-21T16:29:06Z. https://github.com/llvm/llvm-project/commit/1c49ce676caa161250624714c3698b87dc2f8628
[^cd689393]: LLVM commit `cd68939326ed073ad6d50e034310cc295eddd72e`, "[AMDGPU] Add attribute for FWD_PROGRESS (#181675)", 2026-02-26T11:35:36Z. https://github.com/llvm/llvm-project/commit/cd68939326ed073ad6d50e034310cc295eddd72e
[^probe]: Compiler Explorer compile API probe run 2026-11-10 (Paris) against `hipclang-rocm-60303`, `-60400`, `-70001`, `-71400`, `-100000`; args `--offload-arch=gfx1030|gfx1100 --cuda-device-only -O3`; values read from emitted `.amdhsa_*` directives. Compiler Explorer builds are not pinned to a SHA; re-run to confirm. https://godbolt.org/api/compiler/hipclang-rocm-71400/compile
