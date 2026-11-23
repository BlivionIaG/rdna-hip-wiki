# Attention silicon

`fa_rdna2` launch bounds and LDS/VGPR budget, GQA subgroup dispatch, FlashKDA on gfx1030.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [fa-occupancy.md](fa-occupancy.md) | Live `fa_rdna2` LDS/VGPR/`__launch_bounds__` |
| [fa-gqa.md](fa-gqa.md) | FA GQA subgroup default; HEAD_DIM 128/256 prefill (@ `bb40498c`); true variant dropped |
| [flashkda.md](flashkda.md) | FlashKDA: no SM90/CUTLASS/bf16; 128×128 state = `fdot2` later |
| [fa-prefill-tile-caps.md](fa-prefill-tile-caps.md) | stew675 r334 deleted RDNA2 FA `cols_per_block` caps: crash workaround for HIP occupancy 0, cost was KV reuse not waves; current CLR occupancy math; dest GQA prefill = 16-row tile, VGPR-bound audit Later |

## Sources

This is an index page; it makes no technical claims of its own. Primary sources are cited with footnotes and a `## Sources` section on each linked page.
