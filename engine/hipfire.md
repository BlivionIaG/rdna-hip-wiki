# warpfront/hipfire

Date: 2026-08-21. Engine review of [warpfront/hipfire](https://github.com/warpfront/hipfire) (Apache-2.0 as of v0.3.0; v0.2.1 was dual MIT/Apache). **Standalone Rust + HIP engine**, not a vLLM plugin and not a fork of extras. Occupancy still first. Do **not** copy their tok/s onto [coverage.md](coverage.md).

Silicon: [../silicon/hipfire.md](../silicon/hipfire.md).

Headline benches are **gfx1100 / gfx1151 / gfx1201 WMMA**. gfx1030 is listed as *portable HIP + Redline dispatch*, not the tuned family. Their `tests/speed-baselines/gfx1030.txt` exists (2026-04-28, MQ4 + HIPGRAPH + asym3 KV; small models on 16 GB 6900 XT, **27B on a V620 Pro 32 GB and not CI-enforced**). Measurement peer after occupancy — not our extras W4 / 4× / 7.14 / 88096 number.

## Already ours

| Their piece | extras / wiki |
|---|---|
| Capability atoms, not chip-string bleed (`has_dot2_f32_f16`, `has_hfq3_sdot4`) | `on_gfx10x()` not `on_rdna()`. [fp16-rdna2.md](fp16-rdna2.md). |
| `has_wmma = rdna3 \| rdna4` — gfx1030 stays off | Already locked. No WMMA/MFMA/FP8 on V620. |
| Wave32 + packed DOT | Live `fdot2` / planned `sdot4`. |
| DFlash / MTP | Fat tile first. [dflash.md](dflash.md), [mtp.md](mtp.md). Their DFlash is gfx11-tuned. |
| ANTIBLEED (do not apply gfx11 WMMA sources to gfx10) | Same as extras: no AITER, no `on_rdna()` gate. |

## Steal (intent, after occupancy)

Do **not** vendor hipfire into extras or start hippih as a clone. Re-implement the *intent*. Apache-2.0 is safer than leapdragon GPL, still not a wholesale import.

| Item | Why |
|---|---|
| **sdot4 MMQ prefill on gfx10** | `has_hfq3_mmq` / `has_hfq4_mmq` default-on for sdot4 archs (`gfx1011/12` + `gfx1030/31/32`). Generic Q8 `has_mmq` is **false** on gfx1030 (gfx906 + RDNA3 only). Closest cousin of our W8A8 / W4A8 `sdot4` prefill tile — steal **tile / occupancy / cutoff**, not HFQ/MQ byte layout. |
| **RDNA2 MMQ cutoff = 256** | Their comment: untested lower on gfx10. gfx11 sits at 128. Sweep ours; do not paste. |
| **`gemv_rows_default = 1` on RDNA2** | Skinny decode default. Matches extras `M∈{1,2,4,8}` intent. |
| **Redline** | ROCr retained-replay of a proven HIP graph; fail-closed to ordinary HIP. Launch-overhead substrate. **hippih Later only** — not extras, not before occupancy. |
| **V620 measurement peer** | Re-run *their* MQ4 fixture on our 7.14 / 4× box after occupancy. Compare stacks, do not write their baseline into coverage. |

## Dead / do not pivot

- **WMMA paths** (gfx11/12). Dead as a gfx1030 port. Same as FlashKDA CUTLASS / AITER.
- **MQ / HFQ formats into vLLM.** Own quant family (MQ4R, HFQ3/4, Lloyd). Same class as ROCmFPX — **no vLLM port**. [rocmfpx.md](rocmfpx.md).
- **Rust runtime as extras replacement.** Product path stays extras → SGLang overlay → Llaminar → hippih. [hippih.md](hippih.md).
- **Redline / PM4 before occupancy.** Fail-closed is correct; still Later.
- **7900 / Strix / R9700 / their gfx1030 tok/s** on our coverage page.
- **leapdragon-style plugin vendor.** Different class (whole engine). Still no silent overlay.

Their MQ3-on-gfx10 saga (`docs/plans/mq3_gfx10.md`) is a loader-gate post-mortem, not a kernel win: soup was AWQ sidecar gated on `MQ4G256` only. Lesson: **trace the wrapper**, do not invent a gfx10 GEMV bug. Scope-B “write four gfx10 WMMA replacements” was correctly estimated as weeks and then dropped.

## vs leapdragon / hippih

| | leapdragon | hipfire | hippih |
|---|---|---|---|
| Class | vLLM **recipe + GPL plugins** | Standalone **Rust+HIP engine** | Our **empty** three-ISA stub |
| gfx1030 | Tuned Triton/Exllama on 2× ×16 | Portable DOT / sdot4 MMQ | Future DOT backend |
| Steal now | Grid / LDS / `BLOCK_KN` after occupancy | Capability table + sdot4-MMQ *intent* | Nothing — do not start |

## Cards

No new first ticket. Occupancy on extras still first.

After occupancy, fold sdot4-MMQ tile ideas onto the existing W8A8 / W4A8 cards — do not file a “port hipfire” card. Redline rides the hippih-contract Later card ([hippih.md](hippih.md)).

## Sources

- https://github.com/warpfront/hipfire (README, `crates/rdna-compute/src/arch_caps.rs`, `docs/plans/mq3_gfx10.md`, `docs/BENCHMARKS.md`, `tests/speed-baselines/gfx1030.txt`)
- https://hipfire.dev/
- Room 2026-08-21
