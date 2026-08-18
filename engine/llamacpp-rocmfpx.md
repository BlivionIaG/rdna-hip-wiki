# llama.cpp side project — ROCmFPX on V620

Date: 2026-08-18. **Not the vLLM fork.** Occupancy on `perf/rdna2_w4a16` still first. Contract: [rocmfpx.md](rocmfpx.md). Source: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX).

**Why a side project:** `scripts/build-rdna2.sh` already targets gfx1030. Measure codebook→integer-dot on a V620 without touching the vLLM tree. Their published numbers are Strix / **R9700 RDNA4** — do not copy.

## Serving (if we run `llama-server`)

Prefix cache is **on** (`--cache-prompt`). Continuous batching is **slot** `-cb`, not vLLM radix. Concurrency = `-np` slots × `-c` KV.

On one 32 GB V620, 35B-A3B ROCmFP4 (~19 GB): leftover ~10 GB → **handful of 8k slots**, not a farm. Sweep `-np {1,2,4,8}` × `-c {4k,8k,16k}` and record VRAM + per-slot tok/s. Numbers stay on this page.

## Scope

| Do | Don't |
|---|---|
| Clone ROCmFPX `main`, `scripts/build-rdna2.sh` | Merge into `BlivionIaG/vllm` |
| ISA dump MMVQ/MMQ on gfx1030 | Guess `sdot4` from the README |
| One small GGUF smoke + optional `-np` sweep | Copy Strix/R9700 tok/s onto coverage |
| Note `amdgcn_perm` + UE4M3 vs our LUTs | Port Vulkan, MTP, TurboQuant into vLLM |

## Done-when

1. `build-rdna2/` runs on a V620.
2. ISA of hot MMVQ shows `v_dot4c` vs scalar FMA.
3. One greedy decode smoke. Optional: `-np` VRAM/slot table.
4. Keep / drop vs W4A8 `sdot4`.

## Card

**Later / side-project.** Does not block occupancy. Title: `llama.cpp ROCmFPX on V620`.

## Sources

- `scripts/build-rdna2.sh`, [rocmfpx.md](rocmfpx.md)
- llama.cpp server: `--cache-prompt`, `-cb`, `-np`
