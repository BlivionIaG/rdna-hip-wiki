# RCCL on RDNA — what the source actually gates

Date: **2026-09-03**. Owned by ROCM_specialist (library/runtime plumbing). Measured P2P and topology work stays in [silicon/rccl-p2p.md](../silicon/rccl-p2p.md).


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

This answers the open unknown in [silicon/rccl-p2p.md](../silicon/rccl-p2p.md) ("a published gfx1030 RCCL Ring/Tree tuning table"): there is a per-arch tuning index in RCCL source, and **gfx1030 sits on index 0, the generic row** — the same row as gfx906/gfx908/gfx90a. AMD is adding Navi3 tuning and leaving Navi2x on the default table. Filling a gfx1030 row is our own tuning work, and it needs `NCCL_DEBUG_SUBSYS=TUNING` measurements first.

Also in this PR: the `NCCL_TUNING` `INFO` spam is now rank-0 only, and `ncclTopoPreset()` gained an `nChannels` sanity check (`[1,MAXCHANNELS]`, plus a warn when a graph's channel count is below `comm->nChannels`).

Not a dest or pin change. RCCL in the live 7.14.0 pin does **not** have any of this.
