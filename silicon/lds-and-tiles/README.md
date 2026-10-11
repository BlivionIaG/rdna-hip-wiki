# LDS and tiles

LDS pool, 64-bank mapping, padding vs XOR swizzle, and the GEMM/attention tile seeds that follow from them.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [lds-tiles.md](lds-tiles.md) | 64-bank formula, GEMM/attention tile seeds, DP4A recipe |
| [lds-occupancy.md](lds-occupancy.md) | LDS pool 128 KB WGP / 64 KB WG; LLVM 512 B vs ISA 1 KB granule; MaxWGsLDS ladder |
| [lds-bank-occupancy.md](lds-bank-occupancy.md) | Bank conflicts = LDS *latency*, not MaxWGsLDS; pad vs XOR at occupancy cliff |
