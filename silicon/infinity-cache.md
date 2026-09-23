# Infinity Cache — SKU table (RDNA2 intro)

Date: 2026-08-20. Policy / persist bits stay in [cache-policy.md](cache-policy.md). Engine batching: [../engine/cache-aware.md](../engine/cache-aware.md). Occupancy still first. L2 mid-level thrash vs occupancy: [l2-occupancy.md](l2-occupancy.md).

**Yes: Infinity Cache was introduced with RDNA 2** (Navi 21 / GFX10.3, 2020). RDNA 1, GCN, Vega 10 (V340L) do **not** have it. It is an **on-die last-level cache (L3 / MALL)**, after L2, before GDDR6. It is **not** Infinity Fabric / XGMI and is **not** a GPU-to-GPU path.

Do **not** invent TB/s. AMD’s “2.4× / 2.5×” is relative, not a V620 number. Line size is **64 B** (`kfd_crat` L3); L0/L1/L2 are 128 B.

## Can inference use it?

**Yes, as reuse — not as a switch.** HIP on gfx1030 has **no persist bit, no IC prefetch, no IC bypass.** Default global loads already go through MALL. Speedup happens when the **hot set ≤ 128 MB** and you reread it before something else evicts it. Miss = GDDR6 512 GB/s.

| Do | Don’t |
|---|---|
| Quantize / TP-shard until the **layer shard** fits (W4A16 TP=4 on 7B/27B does) | Nontemporal a shard you are trying to keep |
| Default loads on that shard + live KV pages this step touches | Over-occupy (`waves_per_eu(1,1)` is the other bug; too *many* waves also thrash IC) |
| One quiet GPU; two HIP streams fight the same 128 MB | Pretend 27B FP16 or unsharded 7B W8A8 “lives in IC” |
| Align persistent buffers to **128 B** (covers IC 64 B + L2 128 B) | Invent an IC TB/s roofline or a `persist` intrinsic |

Worked fit (V620 128 MiB, from [cache-policy.md](cache-policy.md) §4 — capacity, not a measured hit rate):

| Workload | Layer bytes vs IC | Inference note |
|---|---|---|
| 7B W4A16 / mxfp4, **TP=4** | ~29–31 MiB / GPU | **Keep.** Leftover ~97 MiB can hold thousands of GQA KV tokens. |
| 7B W8A8, TP=4 | ~56 MiB | **Keep.** |
| 7B FP16, TP=4 | ~111 MiB | **Keep**, leftover ~17 MiB — short KV only. |
| 7B W4, **no TP** | ~115–125 MiB | Tight. Quiet GPU or it evicts. |
| 27B W4 / mxfp4, TP=4 | ~70–76 MiB | **Keep.** Leftover ~50–58 MiB ≠ long 27B KV. |
| 27B W8A8 / FP16, TP=4 | over / 2× | **Stream** weights (`nontemporal`); IC for `x` + working KV pages. |

Prefill of a fat activation GEMM that already misses 128 MB does **not** get an IC win. Decode GEMV / FA gathers / a resident W4 shard **can**. Occupancy first: a `(1,1)` trap wastes the CU before IC matters; max waves on a miss stream just multiplies GDDR6 traffic.

## No hardware cache-aware scheduler

gfx1030 has **no** IC QoS, coloring, persist queue, or “schedule this dispatch into warm lines.” MALL is one GPU-wide **LRU**. Two HIP streams, SDMA, or a fat prefill chunk on the same V620 evict the decode shard. TP=4 does **not** share IC across cards — each V620 has its own 128 MB.

What looks like “cache-aware scheduling” is therefore **engine policy**, not silicon: decode-first so the W4 shard is reread, APC for the *other* cache (paged KV prefix), one partial prefill so a miss-sized GEMM does not blow the 128 MB. Stock V1 already does that. extras has no custom scheduler. Details: [../engine/cache-aware.md](../engine/cache-aware.md).

Leave SGLang radix/router as an IC feature. Do not write a kernel to pin lines.

## Our cards

| SKU | ISA | IC | Notes |
|---|---|---|---|
| **Radeon PRO V620** | gfx1030 Navi 21 | **128 MB** | 72 CU harvest, 4 MB L2, GDDR6 512 GB/s. HIP cannot pin or bypass IC. |
| **Radeon PRO W6800** | gfx1030 Navi 21 | **128 MB** | Same die family; official gfx1030 PRO alongside V620. |
| **Radeon PRO W7800 48 GB** | gfx1100 RDNA3 | **96 MB** | On **MCDs** (chiplet), higher latency than Navi 21 on-die. WMMA is legal here; IC is still not a fabric. |
| V340L | gfx900 Vega 10 | **none** | Own host. |

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
- [cache-policy.md](cache-policy.md), [architecture.md](architecture.md) §4.5, [l2-occupancy.md](l2-occupancy.md) (L2 mid ≠ IC), [rccl-p2p.md](rccl-p2p.md) (IC ≠ XGMI)
- [../engine/cache-aware.md](../engine/cache-aware.md)

## Cite: namu RDNA §2.2 (2026-08-24)

[en.namu.wiki/w/RDNA §2.2](https://en.namu.wiki/w/RDNA#s-2.2) is a **Navi 21 recap**, not an inference ISA page. Matches our lock:

- IC = on-die L3-like SRAM, 16 × 8 MB slices, **128 MB** on Navi 21, ~**20% die**. **Fit + reread.** No persist / no HIP switch.
- AMD **2.17× BW / 0.9× power** vs 256-bit GDDR6 alone is qualitative — not a V620 TB/s, not coverage.
- L2 stayed **2–4 MB** (V620 4 MB) while SE/WGP doubled — a layer does **not** live in L2. L2↔IC **16 × 64 B/clk = 1024 B/clk**.
- WGP = two CUs share LDS / cache. wy 58 KB is tight vs 64 KB/**WG**, not auto one-WG/WGP.

**Leave:** FHD/QHD/4K “effective GB/s” (game hit-rate × 8192-bit IC marketing bus). Not V620 decode, not extras coverage. No packed DOT / wave32 / WMMA in that section. Occupancy leftover unchanged.
