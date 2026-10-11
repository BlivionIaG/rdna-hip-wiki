# warpfront/hipfire

Date: 2026-08-21. Engine review of [warpfront/hipfire](https://github.com/warpfront/hipfire) (Apache-2.0 as of v0.3.0; v0.2.1 was dual MIT/Apache). **Standalone Rust + HIP engine**, not a vLLM plugin and not a fork of extras. Occupancy still first. Do **not** copy their tok/s onto [coverage.md](coverage.md).

Silicon: [../silicon/hipfire.md](../silicon/hipfire.md).

Headline benches are **gfx1100 / gfx1151 / gfx1201 WMMA**. gfx1030 is listed as *portable HIP + Redline dispatch*, not the tuned family. Their `tests/speed-baselines/gfx1030.txt` exists (2026-04-28, MQ4 + HIPGRAPH + asym3 KV; small models on 16 GB 6900 XT, **27B on a V620 Pro 32 GB and not CI-enforced**). Measurement peer after occupancy — not our extras W4 / 4× / 7.14 / 88096 number.

## Build features in hipfire? **No**

Ask 2026-08-21: *besides extras, improve hipfire and keep hippih for visual tools — or keep the full hippih way?*

**Keep the full hippih way.** hipfire stays steal + measurement peer. Do not make it the feature tree.

| Why not build in hipfire | Lock |
|---|---|
| gfx1030 is **portable**, not tuned | Their energy is WMMA gfx11/12. V620 work is second-class by design. |
| Vega column is **gfx906**, not V340L **gfx900** | gfx906 objects will not load on gfx900. 8× V340L has **no home** in hipfire. hippih is the three-ISA engine. |
| MQ / HFQ / Lloyd | Features there do not transfer to extras W4. Same as no vLLM ROCmFPX port. |
| Serving model | Ollama-style single binary. Our serving path after occupancy is **SGLang overlay**, not hipfire daemon. |
| Control | warpfront roadmap + their gates. We already wanted an own serving path for the same reason. |

Optional upstream gfx1030 portable patches to *them* is neighborly. That is not “build our features in hipfire.”

**hippih tools/microbench now ≠ hippih is a viz tool forever.** Tools are the *now* slice (occupancy still first). Destination stays three-ISA engine. [hippih.md](hippih.md).

## Serving: MoE / CB / prefix (contribute?)

Note 2026-08-21. These are **their** daemon features. Contributing here does not buy extras / SGLang serving.

| Surface | What they have | Not |
|---|---|---|
| **MoE** | Per-family carriers: Qwen3.5 A3B (`arch_id` 6), DS4 (9), MiniMax-M2 (10), LFM2.5-MoE (11), Cohere2 (12). Dispatch `moe` / `moe_buckets`. | vLLM paged expert offload / DeepEP. Grouped **WMMA** MoE is gfx11/12. |
| **Multi-GPU MoE** | **EP** (`--tp`) = DS4 + MiniMax only. **PP** = Qwen3.5 HFQ layer bands (dense + A3B) — capacity, sequential, not TP serving. `tp>1 && pp>1` errors. | extras TP=4 / 88096 PIX. PP decode is sequential bands. |
| **Continuous batch** | Host `ContinuousBatchScheduler`: fixed lanes, sampling **cohort**, opt-in (`serve_continuous_batch`, size>1). Eligible: Qwen 5/6 + dense LFM 11; single-GPU; no PP/EP, tools, images, stop, spec, PFlash, history, thinking. | Default serve is **one generation holds the lock** + admission queue. Not vLLM V1 CB + paged mix + chunked prefill. |
| **Prefix** | Prefix-capable arches (ds4 / qwen3.5 / qwen3.5_moe) **skip per-request reset** so multi-turn **LCP** hits on the same daemon. | SGLang radix / vLLM APC across distinct prefixes. |
| **CASK / TriAttention** | Opt-in eviction sidecar (`cask=false` default). `pp=1` only. | Prefix sharing. Experimental. |

Steal later (hippih / SGLang overlay): lane/cohort idea if useful. Do not contribute CB/prefix/MoE into hipfire as our serving path.

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
| **`gemv_rows_default = 1` on RDNA2** | Skinny decode default. Matches extras `M\in{1,2,4,8}` intent. |
| **Redline** | ROCr retained-replay of a proven HIP graph; fail-closed to ordinary HIP. Launch-overhead substrate. **hippih Later only** — not extras, not before occupancy. |
| **V620 measurement peer** | Re-run *their* MQ4 fixture on our 7.14 / 4× box after occupancy. Compare stacks, do not write their baseline into coverage. |

## Dead / do not pivot

- **WMMA paths** (gfx11/12). Dead as a gfx1030 port. Same as FlashKDA CUTLASS / AITER.
- **MQ / HFQ formats into vLLM.** Own quant family (MQ4R, HFQ3/4, Lloyd). Same class as ROCmFPX — **no vLLM port**. [rocmfpx.md](rocmfpx.md).
- **Rust runtime as extras replacement.** Product path stays extras → SGLang overlay → Llaminar → hippih. [hippih.md](hippih.md).
- **Build-our-features-in-hipfire.** See above. Steal only.
- **Redline / PM4 before occupancy.** Fail-closed is correct; still Later.
- **7900 / Strix / R9700 / their gfx1030 tok/s** on our coverage page.
- **leapdragon-style plugin vendor.** Different class (whole engine). Still no silent overlay.

Their MQ3-on-gfx10 saga (`docs/plans/mq3_gfx10.md`) is a loader-gate post-mortem, not a kernel win: soup was AWQ sidecar gated on `MQ4G256` only. Lesson: **trace the wrapper**, do not invent a gfx10 GEMV bug. Scope-B “write four gfx10 WMMA replacements” was correctly estimated as weeks and then dropped.

## vs leapdragon / hippih

| | leapdragon | hipfire | hippih |
|---|---|---|---|
| Class | vLLM **recipe + GPL plugins** | Standalone **Rust+HIP engine** | Our **empty** three-ISA stub |
| gfx1030 | Tuned Triton/Exllama on 2× ×16 | Portable DOT / sdot4 MMQ | Future DOT backend |
| Steal now | Grid / LDS / `BLOCK_KN` after occupancy | Capability table + sdot4-MMQ *intent* | Tools/microbench only — do not start the engine |

## Cards

No new first ticket. Occupancy on extras still first.

After occupancy, fold sdot4-MMQ tile ideas onto the existing W8A8 / W4A8 cards — do not file a “port hipfire” or “fork hipfire” card. Redline rides the hippih-contract Later card ([hippih.md](hippih.md)).

## Sources

- https://github.com/warpfront/hipfire (README, `crates/rdna-compute/src/arch_caps.rs`, `docs/plans/mq3_gfx10.md`, `docs/BENCHMARKS.md`, `tests/speed-baselines/gfx1030.txt`, `docs/ARCHITECTURE.md`, `docs/SERVE.md`, `docs/multi-gpu.md`, `crates/hipfire-engine/src/scheduler.rs`)
- https://hipfire.dev/
- Note 2026-08-21: steal not fork; keep full hippih way; MoE/CB/prefix are their daemon, not extras serving
