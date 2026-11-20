# Attention silicon

`fa_rdna2` launch bounds and LDS/VGPR budget, GQA subgroup dispatch, FlashKDA on gfx1030.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [fa-occupancy.md](fa-occupancy.md) | Live `fa_rdna2` LDS/VGPR/`__launch_bounds__` |
| [split-kv-grid-occupancy.md](split-kv-grid-occupancy.md) | Split-KV grid vs WGP fill: HIP `multiProcessorCount` = 36 WGPs (CLR), stew675 r322 per-device tile rule, dest `kv_splits=16` / GQA gate (no change) |
| [fa-gqa.md](fa-gqa.md) | FA GQA subgroup default; HEAD_DIM 128/256 prefill (@ `bb40498c`); true variant dropped |
| [flashkda.md](flashkda.md) | FlashKDA: no SM90/CUTLASS/bf16; 128×128 state = `fdot2` later |

## Sources

This is an index page; it makes no technical claims of its own. Primary sources are cited with footnotes and a `## Sources` section on each linked page.
