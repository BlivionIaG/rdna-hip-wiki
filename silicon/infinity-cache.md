# Infinity Cache — SKU table (RDNA2 intro)

Date: 2026-08-20. Policy / persist bits stay in [cache-policy.md](cache-policy.md). Occupancy still first.

**Yes: Infinity Cache was introduced with RDNA 2** (Navi 21 / GFX10.3, 2020). RDNA 1, GCN, Vega 10 (V340L) do **not** have it. It is an **on-die last-level cache (L3 / MALL)**, after L2, before GDDR6. It is **not** Infinity Fabric / XGMI and is **not** a GPU-to-GPU path.

Do **not** invent TB/s. AMD’s “2.4× / 2.5×” is relative, not a V620 number. Line size is **64 B** (`kfd_crat` L3); L0/L1/L2 are 128 B.

## Our cards

| SKU | ISA | IC | Notes |
|---|---|---|---|
| **Radeon PRO V620** | gfx1030 Navi 21 | **128 MB** | 72 CU harvest, 4 MB L2, GDDR6 512 GB/s. HIP cannot pin or bypass IC. |
| **Radeon PRO W6800** | gfx1030 Navi 21 | **128 MB** | Same die family; official gfx1030 PRO alongside V620. |
| **Radeon PRO W7800 48 GB** | gfx1100 RDNA3 | **96 MB** | On **MCDs** (chiplet), higher latency than Navi 21 on-die. WMMA is legal here; IC is still not a fabric. |
| V340L | gfx900 Vega 10 | **none** | Own host. |

Decode GEMV / FA gathers **can** hit the 128 MB if the hot set fits (W4A16 TP=4 shard does; see [cache-policy.md](cache-policy.md) §4). Cold streams larger than IC still pay GDDR6.

## Family (so harvests are not surprises)

| Die | Example | IC |
|---|---|---|
| Navi 21 (gfx1030) | 6900 XT / 6800 XT / 6800 / V620 / W6800 | **128 MB** on-die |
| Navi 22 (gfx1031) | 6700 XT | **96 MB** |
| Navi 23 (gfx1032) | 6600 XT | **32 MB** |
| Navi 24 | 6500-class | **16 MB** |
| Navi 31 (gfx1100) | 7900 XTX / W7800 | **96 MB** on MCDs |
| RDNA 1 / Vega / GCN | — | **no IC** |

A kernel tuned to “the layer fits in 128 MB” falls out on 96/32/16 MB parts. Query L3; do not hard-code V620’s 128.

## Sources

- AMD 2020-10-28 RX 6000 press (IC as last-level on-die cache): https://www.amd.com/en/newsroom/press-releases/2020-10-28-amd-unveils-next-generation-pc-gaming-with-amd-rad.html
- RDNA 2 Explained (W6000): https://www.amd.com/content/dam/amd/en/documents/products/graphics/workstation/rdna2-explained-radeon-pro-W6000.pdf
- V620 product page: 128 MB IC, 72 CU, 512 GB/s — https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html
- W7800 48 GB: 96 MB IC — https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7800-48gb.html
- `kfd_crat.c` Sienna Cichlid L3 128×1024 KB, line 64 B
- [cache-policy.md](cache-policy.md), [architecture.md](architecture.md) §4.5, [rccl-p2p.md](rccl-p2p.md) (IC ≠ XGMI)
