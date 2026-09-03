# Tooling around the RDNA HIP stack

Date: **2026-09-03**. Owned by ROCM_specialist. Only tools that touch compiler / runtime / ISA plumbing. Engine and fork tooling stays in [engine/](../engine/) and [fork/](../fork/).

## rocjitsu — AMD's own AMDGCN emulation toolkit

Lives in the ROCm systems monorepo, not in a ROCm release: `emulation/rocjitsu` in [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu) (`develop`). Three modes, per its README:

- **Simulation** — full ISA emulation with KFD emulation + `LD_PRELOAD` interposition. **No physical GPU and no kernel module required.**
- **DBT** — dynamic binary translation, running a code object built for one arch on another. Shipped guest configs today are CDNA-on-RDNA4 / CDNA-on-CDNA3 (`guest_gfx950_on_gfx1201.json`, `guest_gfx950_on_gfx942.json`, `guest_gfx950_on_simulated_gfx942.json`). CDNA4→CDNA3 is called experimental.
- **DBI** — runtime instrumentation of kernels (profiling, tracing, analysis).

Its own support table (README, 2026-09-03):

| Arch | Target | Simulation |
|---|---|---|
| RDNA1 | gfx1010 | Experimental |
| **RDNA2** | **gfx1030** | **Experimental** |
| RDNA3 | gfx110x | Beta |
| RDNA3.5 | gfx1151 | Experimental |
| RDNA4 | gfx120x | Beta |
| CDNA1/2 | gfx908 / gfx90a | Experimental |
| CDNA3/4 | gfx94x / gfx950 | Beta |
| CDNA5 | gfx1250 | Beta |

gfx900 is **not** in the table (GFX9 starts at gfx908).

### Our gap, and it is a small one

`emulation/rocjitsu/configs/` ships `gfx1100_w7900.json`, `gfx1151.json`, `gfx1201_r9700.json`, and the CDNA ones — **there is no gfx1030 config**. So "Experimental" for gfx1030 means the ISA family is modelled but nobody wrote the topology JSON. Writing a `gfx1030_v620.json` (72 CU, 32 GB, GDDR6) is a cheap, self-contained contribution and would give us a GPU-less path to run gfx1030 code objects.

### Race detector plugin

`docs/race-detector.md`: a VM plugin that tracks in-flight memory events and reports intra-workgroup hazards — reads of a register or LDS whose producing op was never synchronized (missing `s_waitcnt` / `s_barrier`). Runs entirely on CPU. Constraint: **the emulator does not accept `-O0` code objects**, so build with `-O1` or higher (`hipcc` defaults to `-O3`, so our extras line is already fine).

Being extended right now: [rocm-systems#9470](https://github.com/ROCm/rocm-systems/pull/9470) (open, `newling`, updated 2026-09-03) adds **SGPR/TTMP write-after-write** detection — an instruction overwriting a still-pending `s_load` destination, and two scalar loads to the same SGPR (scalar-memory reads may complete out of issue order). Fix in both cases is `s_waitcnt lgkmcnt(0)`. Its end-to-end tests are gfx950 only; the hazard class is ISA-generic and applies to any hand-written gfx1030 asm or inline asm in extras.

## Machine-readable ISA XML is checked in

`shared/machine-readable-isa/isa/` in the same monorepo carries full MR ISA XML per family, including **`amdgpu_isa_rdna2.xml` (11.2 MB)**, plus rdna1 / rdna3 / rdna3_5 / rdna4 and cdna1–cdna5. This is a structured instruction/encoding spec we can parse instead of scraping the RDNA2 handbook PDF — useful for building our own encoding tables, validators, and disasm cross-checks.

`emulation/rocjitsu/docs/isa-gap-audit.md` documents AMD's own workflow for auditing handbook prose vs that XML vs rocjitsu's generated decoder, and its stated audit order is RDNA4, CDNA4, CDNA3, RDNA3 — **RDNA2 is not on the list**, which is the usual shape of the gap we track.

## Status

Nothing here is a dest or pin change. Compiler pin stays as in [compiler.md](compiler.md).
