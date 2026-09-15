# Idle 2026-09-15 — LLVM ABI occupancy (object linking) + gfx10 FA LDS pick

Occupancy-first. Sources via `gh api` / PR patch only (no clones). No invented tok/s. Does **not** change extras HIP or UNC.

Companions already covering the dump / hint side: [silicon/occupancy-dump.md](../../silicon/occupancy-dump.md) (`NT_AMDGPU_METADATA`), [silicon/occupancy-composite.md](../../silicon/occupancy-composite.md), [silicon/hip-craft.md](../../silicon/hip-craft.md) §1.3, [silicon/wg-size-occupancy.md](../../silicon/wg-size-occupancy.md). This note adds the **object-linking ABI floor** that those pages do not yet lock.

## Verdict

**No dest bump. No new UNC. No extras HIP change.** Live compile path stays whole-program / non–object-linking until upstream ships and ROCm hipcc opts in. Awareness only.

## 1. Take Later — LLVM `[AMDGPU] Introduce ABI occupancy for object linking`

Source: [llvm/llvm-project#199475](https://github.com/llvm/llvm-project/pull/199475) (**open** as of 2026-09-08; author `shiltian`). Docs land under new `AMDGPUUsage` § **ABI Occupancy (Object Linking)**. Flag in tests: `-amdgpu-enable-object-linking`.

### What it is

Under object linking, TUs no longer share whole-program register visibility. Callers/callees must agree on a **register-budget floor** called *ABI occupancy* (waves/EU). The backend records that floor in each function’s **`.amdgpu.info`** as **`.amdgpu_occupancy`** / note type **`INFO_OCCUPANCY` (11)**. HSA kernel metadata (`.vgpr_count`, KD `NumVGPRsForWavesPerEU`, resource-derived `Occupancy`) stays local — the ABI field is a **contract**, not a rewrite of the dump recipe on [occupancy-dump.md](../../silicon/occupancy-dump.md).

### Default floor (formula)

```text
ABI_waves_per_EU = ceil( ceil(1024 / wavefront_size) / workgroup_SIMDs )
```

PR examples: gfx900 wave64 / 4 SIMDs → **4** (VGPR budget 64); gfx1200 wave32 / 4 SIMDs → **8** (budget 192 on that file).

**gfx1030 HIP default (wave32 + WGP, 4 workgroup SIMDs)** → `ceil(ceil(1024/32)/4) = 8` → VGPR budget **128** (`1024/8`). Matches the PR’s gfx10 object-linking reject case: 193 VGPR with `flat_work_group_size=1,1024` errors as *exceeds limit (128)*.

| Flat max WG (wave32, 4 SIMDs) | ABI waves/EU (`.amdgpu_occupancy`) | VGPR budget (1024-file) |
|---|---|---|
| 1024 (IR kernel default) | **8** | **128** |
| 512 | **4** | **256** |
| 256 | **2** | **256** (file cap) |

### Attribute / module-flag precedence (new vs today’s craft)

| Knob | Role under object linking |
|---|---|
| `amdgpu-flat-work-group-size` / Clang `amdgpu_flat_work_group_size` | **ABI-significant.** Implied occupancy **replaces** the default floor for that function (can raise or lower budget). Module override does **not** apply when this attribute is present. |
| Module flag `amdgpu_abi_waves_per_eu` (driver should set IR flag; backend also mentions `-amdgpu-abi-waves-per-eu`) | Overrides default floor for functions **without** explicit flat-WG attribute. |
| `amdgpu-waves-per-eu` / `__launch_bounds__` 2nd arg | Still a **hint**. Cannot lower the ABI floor (hint below ABI is dropped). Hint above ABI is accepted and **tightens** the budget further. |

Tests (`object-linking-abi-occupancy*.ll`) cover gfx900 + **gfx10.1** wave32/64 + gfx11/12. Without the attribute, kernels **and** device functions both get `.amdgpu_occupancy 4` on gfx900; module-level `amdgpu.max_num_*` symbols are suppressed under object linking.

### Craft implication for gfx1030 (when/if enabled)

Today’s wiki rule already says: pin `amdgpu_flat_work_group_size` to the **real** launch flat size; IR omit → `1,1024` ([wg-size-occupancy.md](../../silicon/wg-size-occupancy.md)). Under object linking that pin is no longer only a compile hint — it **is** the ABI VGPR contract. A 256-thread kernel left at IR default `1,1024` would compile against an **8-wave / 128-VGPR** ABI floor even if it never launches 1024. Pinning `256,256` lowers the ABI floor to **2** and restores the full 256-VGPR file budget for separately compiled device helpers.

**Do not** flip hipcc to `-amdgpu-enable-object-linking` on dest until the PR merges and TheRock/ROCm llvm carries it. Watch only.

## 2. Take Later — `dmonkman/RDNAttention` LDS layout ladder (gfx1030-tested)

Repo: [dmonkman/RDNAttention](https://github.com/dmonkman/RDNAttention) (`dev`, HIP, pushed 2026-09-10). Forward FA for RDNA2/GCN5; **gfx1030 (RX 6800 XT) is the only confirmed run target** in their docs — not a V620 dest object.

Portable occupancy patterns (from `src/rdna/fa2_forward_f16.hip` + `arch.hpp` via API):

| Pattern | Detail |
|---|---|
| **LDS budget constant** | `kLdsBudget = 65536` — matches the per-WG programming envelope already used on V620. |
| **Compile-time layout pick** | `pickLayout`: try plain → plain+swizzle → alias → alias+swizzle; `static_assert` if none fit (`shrink Br or Bc`). Prefer intrusion order over runtime fallback. |
| **+1 column pad vs swizzle** | Non-swizzle strides use `Bc+1` / `Br+1` (classic bank-conflict pad); swizzle path keeps exact `Bc`/`Br` when power-of-two. |
| **`__launch_bounds__(256)`** | First arg only (real 16×16 block); no second-arg waves hint — pairs with wiki “set MAX_THREADS to real block size”. |
| **`fdot2` / `sdot4` builtins** | `__builtin_amdgcn_fdot2` / `sdot4` on gfx1030; portable FMA/`mad_mix` only for **dot-less** gfx1010/gfx900 — aligns with dest fdot2 path, not a new atom. |

## 3. Leave

| | |
|---|---|
| **Leave** | Landing ABI occupancy / `-amdgpu-enable-object-linking` into extras or hippihx **now** (PR open; not in live 7.14 llvm). |
| **Leave** | Treating `.amdgpu_occupancy` / `INFO_OCCUPANCY` as a replacement for `.vgpr_count` dumps — different channel (`.amdgpu.info` vs `NT_AMDGPU_METADATA`). |
| **Leave** | `zihaomu/SageAttention-AMD` product objects (gfx1201 / R9700 registry, WMMA/BF16 campaigns) — wrong ISA class for V620. |
| **Leave** | RDNAttention as a dest FA swap / claimed consumer tok/s / I-cache anecdote as a silicon lock. Take the LDS pick ladder only. |
| **Leave** | Radiance / RDNA4 WMMA-FP8 stacks (already Leave on prior idle). |

## Sources

1. llvm/llvm-project PR [#199475](https://github.com/llvm/llvm-project/pull/199475) — `AMDGPUUsage.rst` ABI Occupancy section; tests `object-linking-abi-occupancy*.ll` (incl. `amdgpu10.10` wave32).
2. dmonkman/RDNAttention `dev` — `src/rdna/fa2_forward_f16.hip` (`pickLayout`, `__launch_bounds__(256)`), `src/rdna/arch.hpp` (`fdot2`/`sdot4`), `docs/hardware.md` / `FEATURES.md`.
3. Wiki priors: occupancy-dump / composite / hip-craft §1.3 / wg-size-occupancy (flat-WG pin).
