# Modal GPU Glossary + FA4 reverse-engineer — gfx1030 skim

Date: 2026-09-05. Sources: [Modal GPU Glossary](https://modal.com/gpu-glossary), [We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4) (2025-09-26). Engine note only — paraphrase. Do not invent tok/s. Silicon contracts stay under [../../silicon/](../../silicon/).

Skim rule: **gfx1030-relevant HIP/ISA ideas only** (wave / LDS / occupancy / softmax patterns). CUDA-only FA4 / sm_90 / sm_100 claims are **Leave**.

## Take (portable)

| Idea | Glossary / FA4 | gfx1030 map |
|---|---|---|
| **Shared-memory bank conflicts** | [bank-conflict](https://modal.com/gpu-glossary/perf/bank-conflict): same bank, different addresses → serialize; same address → broadcast OK. Glossary states **32 banks × 4 B** (NVIDIA). | Same *rule*, different modulus: RDNA2 WGP is **64 banks × 4 B**, alias **256 B** ([../../silicon/lds-tiles.md](../../silicon/lds-tiles.md)). Pad / XOR / swizzle still the fix; do not use their 128 B alias. |
| **Occupancy vs latency hiding** | [occupancy](https://modal.com/gpu-glossary/perf/occupancy): theoretical vs achieved; smem/VGPR/block limits; past the latency-hiding point more occupancy can hurt (less resource/thread, lower arithmetic intensity). H100 Tensor-Core GEMMs often run at single-digit %. | Matches our FA story: LDS 33–60 KB → 1 WG / 64 KiB binds occupancy; launch-bounds churn does not ([../../silicon/fa-occupancy.md](../../silicon/fa-occupancy.md)). Skinny `fdot2` still wants waves for the 5-cycle VALU dest; fat tiles may prefer ILP over more WGs. |
| **Latency hiding via warp switching** | [latency-hiding](https://modal.com/gpu-glossary/perf/latency-hiding): stall on mem → scheduler issues another eligible warp. | Wave32 on SIMD32; same idea. Pin `amdgpu_waves_per_eu` so the compiler leaves room for ≥4 waves when the kernel is latency-bound ([../../silicon/hip-craft.md](../../silicon/hip-craft.md)). |
| **Online softmax / delayed rescale** | FA4: online max/sum; **rescale only when the running max moves enough to threaten numerics** (Hot Chips: ~10× fewer corrections). Softmax can use CUDA-core approx `exp2` vs scarce SFUs. | Portable algorithm for `fa_rdna2` Later. gfx1030 transcendentals are ¼-rate VALU (RDNA deck) — same “don’t serialize the whole WG on exp” pressure. Do **not** import FA4’s Tensor Memory / SFU mix as ISA. |
| **Producer/consumer tiling + barriers** | FA4 “life of a tile”: load → MMA → softmax → correction → epilogue with explicit barriers. | Pattern only. Our barrier is `__syncthreads` / LDS-scoped fence; no TMA, no Tensor Memory. |

## Leave

| Claim | Why |
|---|---|
| **TMA** (Tensor Memory Accelerator) async global→shared copies | Hopper/Blackwell. No gfx1030 analogue in HIP craft. |
| **`tcgen05.mma` / wgmma / warpgroup MMA / Tensor Memory (`tS`/`tP`/`tO`)** | sm_100 FA4 pipeline. gfx1030 has **no** matrix cores. |
| **Warp specialization counts** (1 Load + 1 MMA + 8 Softmax + 4 Correction + Epilogue) as a port target | Tied to Blackwell async + Tensor Memory. |
| **Persistent tile scheduler / 2CTA MMA** | CUDA grid semantics + sm_100. |
| **Glossary “32 banks” as RDNA2 fact** | NVIDIA SM shared memory. Ours is 64 ([../../silicon/lds-tiles.md](../../silicon/lds-tiles.md) §1). |
| **FA4 ~20% vs cudnn / any tok/s** | NVIDIA-only; do not cite for V620. |

## Pointers

- LDS pad / 64 KiB clamp (Radiance methodology): [../../silicon/lds-tiles.md](../../silicon/lds-tiles.md) § Radiance A-tile
- FA occupancy leftover: [../../silicon/fa-occupancy.md](../../silicon/fa-occupancy.md)
- Waitcnt / waves_per_eu: [../../silicon/hip-craft.md](../../silicon/hip-craft.md)
