# Architecture and memory

CU/WGP/SIMD32 layout, VALU and packed DOT, the cache hierarchy (L0/GL1/L2/Infinity Cache/GDDR6) and what HIP can set on it.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [architecture.md](architecture.md) | WGP/CU, wave32/64, VGPR, L0/L1/L2/IC, no WMMA |
| [valu.md](valu.md) | gfx1030 VALU: enc / size / issue / which DOT we fire. gfx1100 extras |
| [fp16-rdna2.md](fp16-rdna2.md) | Fastest FP16: explicit `fdot2` + occupancy; skinny decode, measure BLAS prefill |
| [cache-policy.md](cache-policy.md) | What HIP can set; IC fit; no persist/bypass |
| [infinity-cache.md](infinity-cache.md) | IC is RDNA2 intro; V620 128 MB / W7800 96 MB; not XGMI |
