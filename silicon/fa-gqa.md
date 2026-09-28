# FA GQA (gfx1030)

## extras lock 2026-09-28 — tip `bfd5286d` (PR #28 / `690cbf02`)

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. Occupancy first. No tok/s. No fabricated occupancy %.

`fa_prefill_paged_varlen_gqa_kernel_256` + shared helpers in `csrc/rocm/fa_rdna2.cu`:

| Surface | Lock |
|---|---|
| Prefill KV walk | `fa_clip_kv_walk` clips causal/sliding-window K/V tile range per CTA; tile boundaries unchanged → bit-identical vs walk-and-mask. |
| Prefill sliding window | `fa_masked`: keep keys with `q - k < w` (was `k >= q - w`); matches decode + FA/Triton. |
| GQA prefill D=256 softmax | Online softmax in registers via BC-lane `__shfl_xor` (BC=16); one barrier fewer per tile; **no** `sM`/`sL`/`sD` LDS. Launcher smem = `sQ+sK+sV+sP` only; O stays `o_acc[RDS]` registers (prior register-O lock). |
| Decode | Whole-tile splits keep `uint4` V load on every split; keys left of sliding window skipped (`kv_lo`); combine zeros `seq_len==0` (graph padding) instead of NaN. |
| GQA decode kernel | `fa_decode_paged_splitk_gqa_kernel_256` opt-in `VLLM_FA_RDNA2_GQA_DECODE=1` (default **off**). |
| Host / persist | Ops take model softmax `scale` + caller-provided `out` (RDNA_ATTN in-place); drop staging memset/`copy_`; `rdna2_persist_empty` for fully-overwritten decode/split-K scratch (GEMM/AR `rdna2_persist_zeros` unchanged). |
| Unchanged ISA | Still `__launch_bounds__(128|256)` (prefill often `(N, 1)`); `__builtin_amdgcn_fdot2`; XOR LDS swizzle where previously present. Default ATTN remains triton; FA path opt-in `VLLM_USE_RDNA2_FA`. |

CMake gfx1030 append list: **unchanged** this tip window. `97fbae58` (hippihx V1) and `f22fee02` (HIP MoE MTP) are Python-only — no `csrc/rocm` object contract.

---

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
