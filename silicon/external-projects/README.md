# External projects (Take/Leave)

Third-party kernels, forks and recipes read against gfx1030: what we Take, what we Leave.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [curvedinf-int8-vllm.md](curvedinf-int8-vllm.md) | curvedinf INT8 fork: Take PTH-KV + GDN fp32; Leave CK/UA/XGMI; `sdot4` Later |
| [sglang-fork.md](sglang-fork.md) | SGLang overlay imports extras HIP; no second DOT / AITER tree |
| [hippih.md](hippih.md) | hippih stub: three ISAs (`fdot2` / WMMA / `mad_mix`); extras stays first |
| [hipfire.md](hipfire.md) | hipfire: capability dispatch + sdot4 MMQ on gfx10; WMMA is gfx11/12 only |
| [leapdragon.md](leapdragon.md) | leapdragon recipe: grid fill, LDS@256, BLOCK_KN sweep; not our occupancy card |
| [flydsl-dot-atoms.md](flydsl-dot-atoms.md) | FlyDSL gfx1030 feasibility; formal `fdot2`/`sdot4` atoms and proof gates |
| [petit-kernel.md](petit-kernel.md) | causalflow petit FP4: Take shuffle/LDS/denorm caveats; Leave MFMA/CDNA |
| [wafer-gpu-perf.md](wafer-gpu-perf.md) | wafer-ai list: ISA is 70648 + valu, not their MI350/CDNA4 |
| [hipkittens-patterns.md](hipkittens-patterns.md) | HipKittens patterns on wave32 `fdot2`: Take register double-buffer + cheap addressing (Live in EXL3 v3); Later sched hints / 4-wave / prefill LDS DMA; Leave ping-pong, wave-spec, AGPR, XCD |

## Sources

This is an index page; it makes no technical claims of its own. Primary sources are cited with footnotes and a `## Sources` section on each linked page.
