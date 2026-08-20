# SGLang fork — parallel serving path

Date: 2026-08-20. Engine contract. Room asked for **our own SGLang path** (stability + radix/overlap), not a drop of extras. Occupancy still first. Do not invent tok/s.

Silicon/dispatch: same gates as extras — `on_gfx10x()`, not `on_rdna()` (v0.27-class = gfx11/12). AITER / MFMA / FlashKDA CUTLASS stay dead ([flashkda.md](flashkda.md)).

## Role (corrected)

Not “radix ideas only.” **A rebase-on-release SGLang overlay**, same process as `rdna2_extras`: track an upstream tag, add gfx1030 HIP + dispatch, never PR upstream from this room.

What we want from SGLang that extras does not own:

| Piece | Why it is worth a fork |
|---|---|
| Radix / prefix tree | Finer reuse than vLLM block-hash APC ([cache-aware.md](cache-aware.md)) |
| Overlap schedule | CPU preps N+1 while GPU runs N — ITL, not a new DOT |
| Decode-first + optional mixed-chunk | Same IC rule: don’t blow the 128 MB with a fat leftover prefill |
| Serving shell | Deterministic continuous batching; this is the “stability” claim |

What we do **not** want: their Instinct/AITER kernels, `on_rdna()` gates, SM90 FlashKDA, a second occupancy trap.

## Product path (corrected 2026-08-20)

| Order | Engine | Why |
|---|---|---|
| **1. Now** | vLLM **`rdna2_extras`** | Live HIP. Occupancy + `load_row` + CMake gap. |
| **2. Parallel** | **SGLang rdna2 overlay** | Own serving path. **Import extras HIP after occupancy** — do not rewrite `fdot2`. |
| **3. Next hetero** | **Llaminar** | Heterogeneous domains. ROCm gfx906 only today. |
| **4. In-house** | **hippih** | Three ISA. Steal extras kernels + SGLang serving + Llaminar placement. |

Do **not** start the SGLang overlay before extras occupancy is measured. Do not pivot off extras. Do not create a second kernel tree.

## Steal / port order (after occupancy)

1. Repo + rebase-on-release (human branch, same rule as extras).
2. Dispatch: `on_gfx10x()` whitelist; refuse AITER/MFMA/WMMA on V620.
3. Import extras: `fa_rdna2` (occupancy-fixed), skinny GEMV `fdot2`, W4A16 dequant. Same `__launch_bounds__` / `waves_per_eu` — no `(1,1)`.
4. Leave radix + overlap **on**. Measure MBT on V620; mixed-chunk is opt-in.
5. FlashKDA: math/two-kernel only if a KDA model shows up — not a CUTLASS port.

gfx1100 may use local WMMA on the W7800 attn tier. gfx900/V340L is hippih, own host.

## Not first

Occupancy on extras. MLA `load_row`. CMake gap. No SGLang card until that lands. No tok/s copied from their A100/H20 posts.

## Cards

Later, after occupancy. One overlay card (rebase + gfx10x dispatch + import extras HIP). @VLLM_FORK_Manager tracks extras; this fork is a sibling, not a PR to `sgl-project/sglang`.

## Sources

- [batching.md](batching.md), [cache-aware.md](cache-aware.md), [rdna2-extras.md](rdna2-extras.md), [hippih.md](hippih.md)
- Room 2026-08-20: own SGLang fork / own path
