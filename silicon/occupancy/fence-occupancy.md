---
title: "Fences and acquire/release (s_waitcnt drain, buffer_gl0_inv / buffer_gl1_inv) vs effective occupancy (gfx1030)"
sources:
  - https://github.com/llvm/llvm-project/blob/fff43f8f733e80aeee2c48145275df14718313e6/llvm/docs/AMDGPUUsage.md
  - https://github.com/ROCm/clr/blob/fc23bef497bb6ecc1c67c4101dff5d8b4623aa00/hipamd/include/hip/amd_detail/amd_device_functions.h
  - https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/csrc/rocm/hc_prefill_rdna2.cu
  - https://github.com/pytorch/pytorch/pull/198208
---

# Fences and acquire/release vs effective occupancy (gfx1030)

Lock: **A fence is not a PIX / `llvm-calc-occupancy` MaxWaves row.** It allocates nothing. What it does on GFX10 is drain the wave's memory counters and, for acquire at agent scope, invalidate two cache levels. The wave keeps its VGPR∩LDS slot the whole time, so the cost shows up as parked time inside a held slot (sibling of [waitcnt-occupancy.md](waitcnt-occupancy.md)) and as lost cache contents (sibling of [l0-gl1-occupancy.md](l0-gl1-occupancy.md)).

No tok/s. No ms. Do not restate the FA pin.

Date: **2026-11-02** Europe/Paris.

Companions: [atomic-occupancy.md](atomic-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md), [cache-policy.md](../arch-and-memory/cache-policy.md) (HIP fence scopes), [toolchain/compiler/README.md](../../toolchain/compiler/README.md) (pytorch#198208 one-lane acquire bug).

## Take / Leave

| | |
|---|---|
| **Take** | Read every HIP fence through the LLVM GFX10-GFX11 code-sequence table. `__threadfence()` is a seq_cst **agent** fence[^hip-fence], which is lowered like acq_rel: `s_waitcnt lgkmcnt(0) & vmcnt(0) & vscnt(0)`, then `buffer_gl1_inv; buffer_gl0_inv`.[^llvm-seqcst][^llvm-acqrel-agent] |
| **Take** | A **release-only** agent fence is just the counter drain (`s_waitcnt lgkmcnt(0) & vmcnt(0) & vscnt(0)`), with no invalidates.[^llvm-rel-agent] The invalidates belong only to the acquire side. |
| **Take (verified, no change)** | Extras `rdna_extras` @ `b30757c2` `hc_prefill_rdna2.cu` split-K down GEMM: every thread runs `__threadfence()` before the block barrier and ticket `atomicAdd`, and every thread of the last block runs a second `__threadfence()` before reading the other slices' partials.[^hc-ticket] Since all waves run the acquire, each CU's L0 gets invalidated. That is the condition pytorch#198208 broke when Triton ran the acquire on one lane in WGP mode.[^pt198208] The ordering is correct on gfx1030 per the table; nothing to change. |
| **Take Later (fork call, A/B only)** | In that kernel the first fence only needs release semantics. Per the table, `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` would keep the drain and drop `buffer_gl1_inv; buffer_gl0_inv` in every block, including the non-last blocks that exit right after.[^llvm-rel-agent][^llvm-acqrel-agent] The benefit is unmeasured and probably small (one pair of invalidates per block); it is not a correctness issue. Routed as a fork note, not a ticket. |
| **Leave** | Do **not** "cheapen" an inter-block ticket to `__threadfence_block()`. Workgroup scope emits at most `buffer_gl0_inv` (and nothing in CU mode), never `buffer_gl1_inv`; GL1 is per shader array, so blocks on another SA can read stale lines.[^llvm-gfx10-desc][^llvm-acqrel-wg] |
| **Leave** | `-mcumode` does not make agent-scope acquire cheaper. CU mode only drops the workgroup-scope `glc`/`buffer_gl0_inv` steps; agent scope still needs both invalidates.[^llvm-acqrel-wg][^llvm-acqrel-agent] |
| **Leave** | Fences are not a cache-residency or occupancy tuning knob. MALL/Infinity Cache is coherent with GPU memory and needs no fence action.[^llvm-gfx10-desc] |

## 1. Why GFX10 needs two invalidates

The LLVM description of the GFX10-GFX11 memory model gives the cache layout that sets the sequences:[^llvm-gfx10-desc]

- One vector L0 per CU. In WGP mode a work-group's waves can run on both CUs of a WGP, so `buffer_gl0_inv` is needed even for coherence inside a work-group, and also across work-groups.
- One L1 (GL1) per shader array, shared by its WGPs. `buffer_gl1_inv` is needed for coherence between work-groups, which may sit on different SAs.
- L0 and L1 have per-quadrant / per-channel request queues, so operations from different waves can reorder. `s_waitcnt vmcnt(0) & vscnt(0)` is the ordering point for acquire, release and seq_cst.
- On GFX10.3 (gfx1030) the MALL is fully coherent with GPU memory.

## 2. Code sequences that matter for split-K / ticket kernels

From the GFX10-GFX11 code-sequence table at the pinned LLVM SHA:

| HIP / LLVM | gfx1030 machine code | Source |
|---|---|---|
| `fence release` agent | `s_waitcnt lgkmcnt(0) & vmcnt(0) & vscnt(0)` | [^llvm-rel-agent] |
| `fence acq_rel` workgroup | `s_waitcnt lgkmcnt(0) & vmcnt(0) & vscnt(0)`; `buffer_gl0_inv` (omitted in CU mode) | [^llvm-acqrel-wg] |
| `fence acq_rel` agent / system | `s_waitcnt lgkmcnt(0) & vmcnt(0) & vscnt(0)`; `buffer_gl1_inv; buffer_gl0_inv` | [^llvm-acqrel-agent] |
| `fence seq_cst` (any scope) | same as acq_rel | [^llvm-seqcst] |
| `__threadfence()` / `__threadfence_block()` | `__builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent" / "workgroup")` | [^hip-fence] |

Not compiler-probed this pass: the box's clang 22 used for [hazard-occupancy.md](hazard-occupancy.md) is no longer installed. Re-run a `--offload-arch=gfx1030 -S` probe before quoting exact instruction placement.

## 3. Occupancy reading

- The drain parks the wave until all its outstanding loads, stores and LDS ops complete; at a block's end, just before exit, that wait is mostly already paid.
- The invalidates throw away L0 (per CU) and GL1 (per SA) lines. Other resident work-groups on that CU/SA lose those lines too and re-miss (inference from the shared-cache layout in §1; the LLVM doc does not describe the cross-wave cost).
- None of this changes waves/SIMD. Count it under effective occupancy, next to waitcnt parks and atomic CAS loops.

## Sources

[^llvm-gfx10-desc]: LLVM AMDGPUUsage.md, "Memory Model GFX10-GFX11" description, L11889-L11963 at llvm-project `fff43f8f733e80aeee2c48145275df14718313e6`. https://github.com/llvm/llvm-project/blob/fff43f8f733e80aeee2c48145275df14718313e6/llvm/docs/AMDGPUUsage.md#L11889-L11963 (checked 2026-11-02)
[^llvm-rel-agent]: Same file, GFX10-GFX11 code-sequence table, `fence release` agent/system row, L12561-L12576. https://github.com/llvm/llvm-project/blob/fff43f8f733e80aeee2c48145275df14718313e6/llvm/docs/AMDGPUUsage.md#L12561-L12576 (checked 2026-11-02)
[^llvm-acqrel-wg]: Same file, `fence acq_rel` workgroup row (step 3 "If CU wavefront execution mode, omit"), L12726-L12752; the `load atomic acquire` workgroup row also omits `glc=1` and `buffer_gl0_inv` in CU mode. https://github.com/llvm/llvm-project/blob/fff43f8f733e80aeee2c48145275df14718313e6/llvm/docs/AMDGPUUsage.md#L12726-L12752 (checked 2026-11-02)
[^llvm-acqrel-agent]: Same file, `fence acq_rel` agent/system row, L12753-L12777. https://github.com/llvm/llvm-project/blob/fff43f8f733e80aeee2c48145275df14718313e6/llvm/docs/AMDGPUUsage.md#L12753-L12777 (checked 2026-11-02)
[^llvm-seqcst]: Same file, `fence seq_cst` row: "Same as corresponding fence acq_rel", L12874-L12882. https://github.com/llvm/llvm-project/blob/fff43f8f733e80aeee2c48145275df14718313e6/llvm/docs/AMDGPUUsage.md#L12874-L12882 (checked 2026-11-02)
[^hip-fence]: ROCm/clr `amd_device_functions.h` at `fc23bef497bb6ecc1c67c4101dff5d8b4623aa00`: `__threadfence()` = `__builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent")`, `__threadfence_block()` = workgroup. https://github.com/ROCm/clr/blob/fc23bef497bb6ecc1c67c4101dff5d8b4623aa00/hipamd/include/hip/amd_detail/amd_device_functions.h (checked 2026-11-02)
[^hc-ticket]: opengfx1030/vllm-rdna `rdna_extras` `csrc/rocm/hc_prefill_rdna2.cu` at `b30757c257c25c737d671179025bbc5c2ec1ee50`, L395-L433 (release fence, barrier, ticket `atomicAdd`, acquire fence, nontemporal slice sum, re-arm). https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/csrc/rocm/hc_prefill_rdna2.cu#L395-L433 (checked 2026-11-02)
[^pt198208]: pytorch/pytorch#198208 (closed 2026-10-14), RDNA WGP-mode one-lane acquire `buffer_gl0_inv` race; summarized in [toolchain/compiler/README.md](../../toolchain/compiler/README.md). https://github.com/pytorch/pytorch/pull/198208 (checked 2026-10-14 by that page)
