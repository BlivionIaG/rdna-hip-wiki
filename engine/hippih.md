# hippih — in-house HIP engine

Date: 2026-08-20. Repo: [BlivionIaG/hippih](https://github.com/BlivionIaG/hippih). README today: *HIP maxxing inference engine for localLLM masters*. Tree is README + LICENSE only — **contract first**, no tok/s.

Silicon: [silicon/hippih.md](../silicon/hippih.md). **Three compile targets, no shared fatbins.**

## Role (corrected)

Not a discard. **hippih is the custom HIP engine** for the three ISAs we own:

| ISA | Card | Inner op |
|---|---|---|
| **gfx1030** | V620 | packed DOT only — `fdot2` / `sdot4` / `sdot8`. Wave32. No WMMA/MFMA/FP8. |
| **gfx1100** | W7800 | RDNA3. WMMA/bf16 **local** ok (fat prefill). Do not require WMMA on V620. |
| **gfx900** | V340L | Vega10. `v_mad_mix_f32` + `v_pk_fma_f16`. **No DOT.** gfx906 / gfx1030 objects will not load. [silicon/v340l.md](../silicon/v340l.md) |

`on_rdna()` (v0.27.1 = gfx11/12) must **not** gate gfx1030 or gfx900 paths. No shared occupancy attribute across the three TUs.

## Product path (corrected 2026-08-20)

| Order | Engine | Why |
|---|---|---|
| **1. Now** | vLLM **`rdna2_extras`** | Live HIP + serving. Occupancy first. |
| **2. Parallel** | **SGLang rdna2 overlay** | Own serving path (radix + overlap). Import extras HIP. [sglang-fork.md](sglang-fork.md) |
| **3. Next hetero** | **Llaminar** | Steal hetero domains + add CB. ROCm is gfx906 only today. |
| **4. In-house** | **hippih** | Our engine. Three ISA backends. Steal extras + SGLang serving + Llaminar placement. |

Do **not** start hippih or the SGLang overlay before extras occupancy. Do not port ROCmFPX GGUF types first. Do not pivot off extras. Do not rewrite DOT for SGLang.

## Steal when we write it

From extras (after occupancy + `load_row`): skinny decode GEMV `fdot2` (`M∈{1,2,4,8}`), prefill rocBLAS first (HIP 64×64×32 only if it wins), `fa_rdna2` occupancy `waves_per_eu(4, 8)` not `(1,1)`, W4A16 dequant→`fdot2`, later W8A8 `sdot4`.

From vLLM: paged KV + continuous batching.

From SGLang: radix prefix tree + overlap schedule. Not AITER.

From Llaminar: heterogeneous domains. Bus rule stays activations-only; **live KV on gfx1100**, not V620. [multi-tier.md](multi-tier.md).

gfx900 backend is packed mix/FMA — Later, third ISA, new TU.

## Not first

Occupancy on extras. MLA `load_row` OOB. CMake gap (`moe_w8a16_fp8_rdna2.cu`). Measured W7800↔V620 P2P (`v620_toolbox/pcie_p2p`).

## Cards

Later. Occupancy still first. One hippih-contract card after extras occupancy lands. SGLang overlay is a sibling Later card — [sglang-fork.md](sglang-fork.md).

## Sources

- hippih README (stub)
- [silicon/hippih.md](../silicon/hippih.md)
- [rdna2-extras.md](rdna2-extras.md), [sglang-fork.md](sglang-fork.md), [fp16-rdna2.md](fp16-rdna2.md), [alt-engines.md](alt-engines.md)
- Room 2026-08-20: own SGLang path
