# GDN chunked prefill — extras HIP (`aac1fcd6`)

`b53a7a2` landed the 5 kernels. Two later fixes make multi-chunk **correct**. `e054854f` is spill/LDS on **delta_h only**. `77d6fdf8` is o-kernel BV. `5d2cf49f` is delta_h unroll (not a tok/s close). `aac1fcd6` is the multi-chunk **correctness** close for delta_h indexing:

| SHA | Bug | Silicon |
|---|---|---|
| `f563820e` | delta_h only advanced `h`; k/w/u/vnew stayed on chunk 0 | Serial recurrence re-applied chunk-0 `(I − k0ᵀw0)` → state explode. Pointers + `g` must step `BT` |
| `6e20b239` | o-kernel LDS zero-fill wrote only first 2 cols, **no barrier** before copy; varlen used global `i_t` | Sparse clobber under occupancy at NT≥2. Now full-tile zero + `__syncthreads()`. Local `i_t` for q/k/v/o/g; `h` stays global |
| `e054854f` | delta_h `GDN_BV` 32→16, `GDN_THREADS` 256→128. Same lane layout as decode (`v = lane >> 3`, `ks = lane & 7`). Still `fdot2`, still `(2, 4)` | ELF (commit): `.vgpr_spill_count` 426→94, `.sgpr_spill_count` 78→0, `.private_segment_fixed` 1628→372 B, **`.group_segment_fixed` 65536→0**. Prior wiki “0 for h (VGPR)” was source intent; the 256-thr binary still reserved **64 KB** LDS. 128-thr actually 0. Grid tiles 4→8. Not FA occupancy. Do not copy tok/s |
| `77d6fdf8` | o-kernel `GDN_BV` 32→64. Still 256 thr, `(2, 4)`, `fdot2`. Grid tiles V/BV 4→2 | LDS **45312→57600 B** (~44→~56 KB): `h` 8→16 KB, `v_T` 4→8 KB. Still <64 KB/WG (~8 KB headroom). Same 1-WG class as wy. Source comments still say BV=32 / 8+4 KB — stale. Not FA occupancy. Do not copy kernel/e2e % |
| `5d2cf49f` | delta_h `#pragma unroll 8` on the 32-iter t-pair loop (full unroll hoists w/k into VGPR) | Source: still 128 thr / BV 16 / `(2, 4)` / LDS 0 / `fdot2`. Full unroll spilled (94 VGPR / 372 B scratch). Partial 8: 182 VGPR, 0 scratch. 182→192 granule; `(2,4)` VGPR cap 512 — fits. Not FA occupancy. Do not copy −14% |
| `aac1fcd6` | delta_h loop used chunk-local `t` on unbaselined `p_k`/`p_w`/`p_u`/`p_vnew` (chunk 0 only; later chunks garbage / unwritten `v_new`) | Rebase those four row pointers by `chunk_start` each chunk. Still 128 thr / BV 16 / `(2, 4)` / LDS 0 / `fdot2`. No tile change. Cousin of `f563820e` (that one advanced `h`/`g`; this one fixes the in-chunk row base). Not FA occupancy. Do not copy tok/s |


Five kernels. Replaces Triton `chunk_gated_delta_rule` + `fused_post_conv_prep` on gfx10x. Decode already HIP (`69d2efe`). Occupancy leftover is still **FA prefill `(N, 1)`** — do not close that card. Fat-M / ConfigA unchanged. Do **not** put 16k/c=8 tok/s on coverage.

All five: `__launch_bounds__(N)` + `amdgpu_waves_per_eu(2, 4)` — not `(1,1)`.

| Kernel | Threads | Inner | LDS | Note |
|---|---|---|---|---|
| `gdn_prefill_prep_rdna2` | 128 | scan / half cvt | 1 float carry | fused post-conv + chunk cumsum |
| `gdn_prefill_kkt_rdna2` | 256 | **scalar fp32 FMA** | 16 KB `k[64,128]` half | `_CAST_DOT_TO_K_DTYPE=False` on gfx1030. **No `fdot2`** |
| `gdn_prefill_solve_wy_rdna2` | 256 | Phase1 fp32 FMA Schur; Phase2 **`fdot2`** | **~58 KB** (`A`/`Ai` 16+16, `Aih` 8, rhsT 16.5, pad 2) | 64 KB tight. `expf` not `__expf` |
| `gdn_prefill_delta_h_rdna2` | **128** (`e054854f`; was 256) | **`fdot2`** (rtne half pairs) | **0** (ELF). Was 64 KB group_segment at 256-thr | serial inter-chunk. Same `(2, 4)` |
| `gdn_prefill_o_rdna2` | 256 (`77d6fdf8` BV=64) | **`fdot2`** qk / qh / Av | **~56 KB** (was ~44). q+k reuse slot for `b_A` | causal `>=`. Tiles 4→2 |

Chunk `BT=64`, `K=V=128`. hipOccupancy still TBD. JIT hang at 16k/c=8 is the Triton surface they removed — not an FA occupancy flip.

## extras lock 2026-09-04 (tip `aac1fcd6`)

HIP: `gdn_prefill_delta_h_rdna2.cu` chunk-local pointer rebase (table row above). ISA shape unchanged vs `5d2cf49f`.

Engine note (not ISA): `5ce7d86e` defaulted the HIP prefill chain **off** (`VLLM_GDN_HIP_PREFILL=1` opt-in) after multi-chunk corruption; `aac1fcd6` re-enables the chain by default (`VLLM_GDN_HIP_PREFILL=0` rollback). Decode HIP stayed on through both. Until that fix, hybrid chunked-prefill for GDN state layers was Triton/generic for prefill. Regression: `tests/kernels/test_gdn_prefill_rdna2.py`. Occupancy leftover still FA prefill `(N,1)`. Do not invent numbers. Do not copy tok/s.

