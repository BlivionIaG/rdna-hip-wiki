# FA GQA (gfx1030)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip including `ecfec4e4` / `13a3daf4` through **`5c3c0c6f`**. No tok/s.

## Prefill

- Templated multi-q-head CTA sharing K/V smem.
- **Production default = subgroup** (`HEADS_PER_CTA=2`, BR=8).
- `true` GQA variant dropped from dest (`13a3daf4`); env `VLLM_FA_RDNA2_GQA_MODE` = `subgroup|off` (subgroup default). Do not resurrect true-GQA without new soak.

## Decode

- `fa_decode_paged_splitk_gqa_kernel_256` with `GQA_MAX_G=8`, `__launch_bounds__(256)`, LDS K/V tiles + fdot2 QK.

## Occupancy

Prefill kernels still carry `__launch_bounds__(128|256, 1)` — **occupancy still first**. GQA is a throughput shape, not the occupancy fix. Same leftover as [fa-occupancy.md](fa-occupancy.md).
