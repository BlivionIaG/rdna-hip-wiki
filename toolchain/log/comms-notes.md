---
owner: ROCM_specialist
kind: scan-log
sources:
  - "https://github.com/ROCm/TheRock/pull/8567"
  - "https://github.com/ROCm/rocm-systems/pull/12025"
  - "https://github.com/ROCm/TheRock/pull/8583"
  - "https://github.com/ROCm/TheRock/pull/8533"
  - "https://github.com/ROCm/rocm-systems/pull/10670"
  - "https://github.com/ROCm/rocm-systems/pull/11752"
  - "https://github.com/ROCm/rocm-systems/pull/11803"
  - "https://github.com/ROCm/rocm-systems/pull/11035"
  - "https://github.com/ROCm/rocm-systems/pull/12679"
  - "https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl/CHANGELOG.md"
---

> Historical scan record, kept intact from the flat `toolchain/` pages on branch `toolchain/gotchas-2026-10-09`. Inline links are the sources recorded on the scan date. This restructure did not re-open every SHA or open/merged status below. Those point-in-time claims are **(unsourced)** as of 2026-10-11, except where a topic page repeats the claim with a footnote.

# RCCL on RDNA — what the source actually gates

Date: **2026-09-03**. Owned by ROCM_specialist (library/runtime plumbing). Measured P2P and topology work stays in [silicon/interconnect/rccl-p2p.md](../../silicon/interconnect/rccl-p2p.md).


## 2026-10-16 — RCCL Tree+SHM crash with extra channels fixed (systems tip, not dest)

- [rocm-systems#12679](https://github.com/ROCm/rocm-systems/pull/12679) merged 2026-10-16 04:48 UTC (06:48 Europe/Paris).[^r12679] Title: "TREE+SHM with more than default channels crash on Navi".[^r12679]
- Symptom: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` when the Tree algorithm runs over shared memory with more channels than the Tree graph search found, e.g. `NCCL_ALGO=Tree` + `NCCL_P2P_DISABLE=1`.[^r12679-cl] The PR body reproduces it on gfx120x with forced Tree over SHM and >2 channels; the CHANGELOG line names gfx110x and gfx120x.[^r12679] [^r12679-cl]
- Root cause: a long-standing design bug in the `ncclTopoPreset` → `AllGather3D` → `ncclTopoPostset()` sequence in `initTransportsRank()`; channels added after Preset had no intra-node tree links. Exposed by the gfx120x/gfx110x channel-count change in [#11035](https://github.com/ROCm/rocm-systems/pull/11035) (see 2026-09-03 below). Fix: `ncclTopoPostset` now sets tree and CollNet chain links on every channel, as it already did for rings.[^r12679] [^r12679-cl] [^r11035]
- gfx1030: **not mentioned** in the PR (tests use gfx1201). This note does not claim gfx1030 is affected or safe; #11035's channel bump auto-enables only on gfx110x/120x.[^r12679] [^r11035]
- Practical: not in a ROCm release; the live dest pin is ROCm 7.14.0 ([toolchain/pin/README.md](../pin/README.md)). If multi-GPU collectives crash on the gfx1100 boxes (W7800 / 7900 XTX), check this first. Watch status only, not a dest pin change.

## 2026-10-07 — RCCL corruption traced to host amdgpu, not RCCL

Silent `all_gather` / `reduce_scatter` corruption on gfx1030 and gfx1100 (first-enumerated GPU, SHM) is an amdgpu regression in the Ubuntu 24.04 stock 6.8 kernel; HWE 7.0 (amdgpu DRM 3.64) or `amdgpu-dkms` fixes it, `HSA_DISABLE_CACHE=1` works around it. Details and the related DRM-3.64 VMM hang: [hip-runtime.md](../runtime/README.md). Not a dest change.

## 2026-09-29 — RCCL symmetric restore after sync (systems tip, not dest)

Landed in TheRock systems pin `0bf70ef` via [#8567](https://github.com/ROCm/TheRock/pull/8567) / [rocm-systems#12025](https://github.com/ROCm/rocm-systems/pull/12025): restore RCCL cost-model constants (61 blocks vs NCCL 10), `nWarps` 256, drop per-task bootstrap allgather (~85 µs), set `maxChannels`. Measured on 8×gfx950; **not a gfx1030 dest lever** and does not change live PYNCCL / pin-7.14 policy (AR knobs stay server-side; `rdna_ar` remains opt-in/0). Watch follow-on TheRock [#8583](https://github.com/ROCm/TheRock/pull/8583) bkc RCCL cherry-picks →`4fbbb315`.

## 2026-09-28 — RCCL LL 128-bit vector path (systems tip, not dest)

Landed in TheRock systems pin `0ad5f73` via [#8533](https://github.com/ROCm/TheRock/pull/8533) / [rocm-systems#10670](https://github.com/ROCm/rocm-systems/pull/10670): explicit 128-bit vector loads/stores on the LL path. Reported test surface includes gfx1100 / gfx1201 / gfx950; GFX11 inline-asm b128 path preserved. **Not a gfx1030 dest lever** and does not change live PYNCCL / pin-7.14 policy (AR knobs stay server-side; `rdna_ar` remains opt-in/0).

## 2026-09-21 — RCCL Navi restores (systems tip, not dest)

Landed in TheRock systems pin `9f9214b` window:
- [rocm-systems#11752](https://github.com/ROCm/rocm-systems/pull/11752) restores Navi optimizations lost after an NCCL sync.
- [rocm-systems#11803](https://github.com/ROCm/rocm-systems/pull/11803) sets gfx110x AlltoAll to **1 channel** to cut >32MB variance (ROCM-30174).

Live dest still PYNCCL / pin 7.14; do not flip `VLLM_RDNA_AR` / `FORCE_CUSTOM_ALL_REDUCE` from these alone.

## 2026-09-03 — "Navi optimizations" ([rocm-systems#11035](https://github.com/ROCm/rocm-systems/pull/11035))

Open PR by `PJAvinash` (created 2026-09-01, updated 2026-09-03, JIRA AICOMRCCL-1906) against `projects/rccl`. Author's own claims, unbenchmarked by us:

- **Navi defaults to only 2 channels.** Raising the channel count helps saturate the PCIe paths — "4% throughput improvements are possible" on P2P/IPC.
- With `NCCL_P2P_DISABLE=1` (SHM path), generating **load-balanced Hamiltonian rings** gives ">10% throughput improvements".
- Default P2P/IPC behaviour is otherwise unchanged except LL tuning on gfx110x.

### What is gated to what (read from the diff, not the prose)

`src/graph/connect.cc` and `src/graph/tuning.cc` compute:

```c
isGfx_110x_120x     = IsArchMatch(gcn,"gfx110") || IsArchMatch(gcn,"gfx120");
intraGraphGen       = rcclParamIntraGraphGen() || (p2pDisabled && isGfx_110x_120x);
interGraphGen       = rcclParamInterGraphGen();
disableRingDiversity= isGfx_110x_120x && !p2pDisabled;
```

New knobs, both **default 0** and **arch-agnostic** (`RCCL_PARAM(IntraGraphGen,"INTRA_GRAPH_GEN",0)`, `RCCL_PARAM(InterGraphGen,"INTER_GRAPH_GEN",0)`):

| Knob | Effect |
|---|---|
| `RCCL_INTRA_GRAPH_GEN=1` | forces the new Walecki+greedy edge-disjoint Hamiltonian ring generator (`generateRings()`) for intra-node rings on **any** arch |
| `RCCL_INTER_GRAPH_GEN=1` | adds `findRingCutIndices()` inter-node cutting; also forces the gfx1151 graph-gen path |
| `RCCL_INIT_CHANNELS=<n>` | channel count used by that path; otherwise `nChannels` defaults per arch (gfx1151 → 6; gfx110x/120x with P2P disabled → 56 = 14 K8 edge-disjoint cycles × 4) |

**So the gfx1030 lever exists.** The auto-enable is gfx11/gfx12-only, but the env knob is not arch-checked: on 4× V620 we can try `RCCL_INTRA_GRAPH_GEN=1` with `RCCL_INIT_CHANNELS=<n>` on the SHM path once this lands in a build we use. Unmeasured — treat as an experiment, not a default.

### Tuning table: gfx1030 stays on the generic row

`rcclGetTuningIndexForArch()` before → after in this PR:

| Arch | Before | After |
|---|---|---|
| **gfx1030** | **0** | **0 (unchanged)** |
| gfx1100 / gfx1101 | 0 | **10** (new Navi3 row) |
| gfx1102 | 0 | 0 |
| gfx1151 | 9 | 9 |
| gfx1200 / gfx1201 | 7 | 7 |
| gfx942 / gfx950 | 5 / 6 | 5 / 6 |

This answers the open unknown in [silicon/interconnect/rccl-p2p.md](../../silicon/interconnect/rccl-p2p.md) ("a published gfx1030 RCCL Ring/Tree tuning table"): there is a per-arch tuning index in RCCL source, and **gfx1030 sits on index 0, the generic row** — the same row as gfx906/gfx908/gfx90a. AMD is adding Navi3 tuning and leaving Navi2x on the default table. Filling a gfx1030 row is our own tuning work, and it needs `NCCL_DEBUG_SUBSYS=TUNING` measurements first.

Also in this PR: the `NCCL_TUNING` `INFO` spam is now rank-0 only, and `ncclTopoPreset()` gained an `nChannels` sanity check (`[1,MAXCHANNELS]`, plus a warn when a graph's channel count is below `comm->nChannels`).

Not a dest or pin change. RCCL in the live 7.14.0 pin does **not** have any of this.

## Sources

[^r12679]: https://github.com/ROCm/rocm-systems/pull/12679
[^r12679-cl]: https://github.com/ROCm/rocm-systems/pull/12679/files (projects/rccl/CHANGELOG.md hunk); https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl/CHANGELOG.md
[^r11035]: https://github.com/ROCm/rocm-systems/pull/11035
