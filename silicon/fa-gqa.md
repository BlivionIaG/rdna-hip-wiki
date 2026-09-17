# FA GQA (gfx1030)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip **`50120e13`** (was through `5c3c0c6f`; silicon delta `d1b200b1`). No tok/s.

## Prefill

- Templated multi-q-head CTA sharing K/V smem.
- **Production default = subgroup** (`HEADS_PER_CTA=2`, BR=8).
- `true` GQA variant dropped from dest (`13a3daf4`); env `VLLM_FA_RDNA2_GQA_MODE` = `subgroup|off` (subgroup default). Do not resurrect true-GQA without new soak.

### extras lock 2026-09-17 (tip `50120e13`, HIP `d1b200b1`)

`fa_prefill_paged_varlen_gqa_kernel_256` in `csrc/rocm/fa_rdna2.cu`:

| Surface | Delta |
|---|---|
| O accumulator | Left LDS (`sO`) for per-thread registers `o_acc[RDS]`. Thread `t` owns row `t/RP` and `RDS` consecutive dims (`RP = THREADS/ROWS`, `RDS = 256/RP`). Online-softmax rescale is register-only (no full sO sweep). |
| P·V | Hoists `sP[row*BC+k]` out of the dim loop; V strip via `uint4` (16 B) loads into FMA on `o_acc`. |
| S = Q·Kᵀ | Same `acc0`/`acc1` `fdot2` partition; loads widened to `uint4` / 8-half. fp16 accumulation order unchanged. |
| Launcher smem | Drops dead `sO` term; `sM`/`sL`/`sD` counted as `* 3` (was under-counted by one row-set). |

Still fdot2 / half2. No `__launch_bounds__` change on this kernel. LDS pressure falls by the old `HEADS*BR*256` float O tile — occupancy-relevant for this GQA prefill shape, but the broader FA prefill `(N, 1)` leftover on other kernels is **not** closed. Do not invent numbers. Do not copy tok/s.

## Decode

- `fa_decode_paged_splitk_gqa_kernel_256` with `GQA_MAX_G=8`, `__launch_bounds__(256)`, LDS K/V tiles + fdot2 QK.

## Occupancy

Prefill kernels still carry `__launch_bounds__(128|256, 1)` — **occupancy still first**. Register-O is an LDS shrink on this GQA kernel only, not the occupancy-card close. Same leftover as [fa-occupancy.md](fa-occupancy.md).
