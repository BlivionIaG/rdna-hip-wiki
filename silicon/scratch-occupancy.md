# Scratch / private segment vs occupancy (gfx1030)

Lock: **`.private_segment_fixed_size` is not a waves/SIMD limiter in LLVM or GPUOpen theoretical occupancy.** It is a **runtime GDDR scratch pool** (private AS 5). Spills always hurt latency; they cut concurrent waves only when ROCr fails to map the requested pool and lowers `waves_per_cu`. Dump/reject recipe stays on [occupancy-dump.md](occupancy-dump.md) §6; VGPR/LDS math on [hip-craft.md](hip-craft.md) §6.

Does **not** change extras HIP or UNC cards. No tok/s.

## Take / Leave

| | |
|---|---|
| **Take** | Treat `.private_segment_fixed_size > 0` (and COV5+ `.uses_dynamic_stack`) as an **inner-loop reject** for latency — same gate as spill counts. |
| **Take** | Theoretical waves/EU = `min(slots, VGPR, LDS, WG/barrier)`. **Do not put scratch bytes in that min.** `llvm-calc-occupancy` has no `--scratch`. |
| **Take** | If a spilled kernel somehow ships, watch ROCr scratch reclaim / `HSA_SCRATCH_SINGLE_LIMIT` — runtime can **reduce dispatch occupancy** when the pool will not map. |
| **Leave** | Do not invent a “scratch bytes → waves/SIMD” table like the VGPR granule ladder. LLVM explicitly does not compute it. |
| **Leave** | Do not confuse Clang `amdgpu_waves_per_eu` “limit private segment” with SPI treating scratch like VGPRs. That attribute is a **codegen budget hint**, not a hardware file size. |

## 1. What `.private_segment_fixed_size` is

| Field | Meaning |
|---|---|
| `.private_segment_fixed_size` | Fixed private/scratch bytes per work-item (`ProgramInfo.ScratchSize`). LLVM private **AS 5**. |
| `.vgpr_spill_count` / `.sgpr_spill_count` | Allocator spill *ops* — usually the reason private size is non-zero. |
| `.uses_dynamic_stack` (COV5+) | Dynamic call stack; still needs a private-segment budget (`LIBOMPTARGET_STACK_SIZE` path in AMDGPUUsage). |

Hardware (LLVM AMDGPUUsage, Private): if the dispatch uses scratch, the device allocates from a **runtime backing pool** per wavefront. Lanes use **dword (4 B) interleave**:

```
wavefront-scratch-base
  + (private-address / 4) * wavefront-size * 4
  + wavefront-lane-id * 4
  + (private-address % 4)
```

Traffic is L0 → L2 → Infinity Cache → GDDR6 — **not** a per-SIMD register file. gfx1030 ABI = **Absolute flat scratch** (`amdgpu10.30` / `gfx10-3-generic`).

Per-wave hardware ceiling (LLVM `GCNSubtarget::getMaxWaveScratchSize`, pre-GFX11 / gfx1030):

```
COMPUTE_TMPRING_SIZE.WAVESIZE = 13-bit field in units of 256 dword
max = (256 * 4) * ((1 << 13) - 1) = 8_387_584 B  (~8 MiB / wave)
```

That is a **tmpring encoding limit**, not an occupancy ladder.

## 2. Compiler occupancy math — scratch omitted

| Tool / API | Inputs | Scratch? |
|---|---|---|
| `llvm-calc-occupancy` | `--wg-size`, `--vgprs`, `--sgprs`, `--lds` | **No flag** |
| `GCNSubtarget::computeOccupancy(F, LDS, SGPRs, VGPRs)` | LDS + SGPRs + VGPRs (+ WG-size range) | **Explicitly not modeled** |
| GPUOpen “Occupancy explained” limiters | VGPR, LDS, threadgroup size, barriers | **Scratch absent** |
| hip-craft / architecture wiki formula | `min(slots, VGPR, LDS, WG/barrier)` | **Scratch absent** |

LLVM `GCNSubtarget.h` on `computeOccupancy` (opened this pass):

> Note that occupancy can be affected by the scratch allocation as well, but we do not have enough information to compute it.

So: compiler **knows** scratch *can* matter at run time, and **refuses** to fold it into waves/EU. Do not fill that gap with guesswork.

Clang `__attribute__((amdgpu_waves_per_eu(min[,max])))` may constrain “size of available group and private memory segments” so the requested wave range still fits — that is the backend **shrinking budgets / refusing spills** to meet the hint, not SPI subtracting scratch from the 16 wave slots.

## 3. Runtime — when scratch *does* cut waves

ROCr allocates scratch from a **device memory pool** attached to the queue/agent (`AcquireQueueScratch`). If the requested size will not map, the runtime **lowers targeted occupancy** and retries:

ROCm/rocm-systems `bd63e50` (`amd_gpu_agent.cpp`):

> If the required scratch allocation is too large, ROCr will attempt to reduce it by lowering the dispatch's targeted occupancy.

Debug string in that path: `Failed to map requested scratch (%ld) - reducing queue occupancy.` The loop steps `waves_per_cu` down by `waves_per_group` (with SE/XCC alignment on newer GC).

Knob (ROCR env, docs-7.1.1):

| Env | Default (docs) | Role |
|---|---|---|
| `HSA_SCRATCH_SINGLE_LIMIT` | `146800640` (~140 MiB) | Threshold for allocate / reclaim behavior per dispatch |
| `HSA_NO_SCRATCH_RECLAIM` | `0` | `1` = permanently pin scratch to the queue even above the threshold |
| `HSA_SCRATCH_SINGLE_LIMIT_ASYNC` / `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM` | async GPUs | Async reclaim variants (not the gfx1030 occupancy formula) |

HIP also exposes `hipExtLimitScratchCurrent` (ROCm 7.0+ note in the same page) to change the default allocation size programmatically.

**Net for gfx1030 HIP kernels:**

1. Compile-time / RGP “theoretical” occupancy = VGPR∩LDS∩WG∩barrier. Scratch ≠ that list.
2. Non-zero private size ⇒ latency poison in the hot path (occupancy-dump §6 reject).
3. If you ever launch with large private size, measured waves can still drop because **ROCr starved the scratch pool** and cut `waves_per_cu` — not because SPI ran out of a scratch register file.

## 4. Relation to existing wiki pages

| Page | What it already said | What this lock adds |
|---|---|---|
| [occupancy-dump.md](occupancy-dump.md) §6 | Reject private>0; dword interleave; Absolute flat scratch; latency fatal | Clarifies: **not** a waves/SIMD term in `llvm-calc-occupancy` |
| [hip-craft.md](hip-craft.md) §6.1 | Occupancy = min(slots, VGPR, LDS, WG/barrier) | Confirms scratch stays **out** of that min; runtime path is separate |
| [architecture.md](architecture.md) | SPI reserves wave slots + VGPR + SGPR + LDS | Scratch backing is the **runtime pool**, not that SPI file set |
| [fa-occupancy.md](fa-occupancy.md) | FA resource pins | Unchanged |

## Sources (opened this pass)

1. LLVM AMDGPUUsage — Private AS 5, dword interleave, runtime backing pool, Absolute flat scratch on `gfx1030` — https://llvm.org/docs/AMDGPUUsage.html
2. LLVM `GCNSubtarget.h` — `computeOccupancy` comment (“scratch … do not have enough information”); `getMaxWaveScratchSize` pre-GFX11 = `(256*4)*((1<<13)-1)` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNSubtarget.h
3. LLVM `llvm-calc-occupancy` — options: wg-size / vgprs / sgprs / lds only — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
4. LLVM PR #123748 / commit `6206f54` — same scratch caveat on `computeOccupancy` — https://github.com/llvm/llvm-project/pull/123748
5. Clang AttributeReference `amdgpu_waves_per_eu` — may limit group **and private** segment size as a codegen hint — https://clang.llvm.org/docs/AttributeReference.html#amdgpu-waves-per-eu
6. GPUOpen Occupancy explained — limiters VGPR / LDS / threadgroup / barriers; no scratch — https://gpuopen.com/learn/occupancy-explained/
7. ROCm/rocm-systems `bd63e50` — ROCr reduces `waves_per_cu` when scratch map fails — https://github.com/ROCm/rocm-systems/commit/bd63e5045c363404c1e9cfd0250705d7d28bce65
8. ROCR env vars — `HSA_SCRATCH_SINGLE_LIMIT` default `146800640` — https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-7.1.1/api-reference/environment_variables.html
