# `rdna2_extras` @ `3e05abc9` — HIP / silicon review

Date: 2026-08-19. Tip of `BlivionIaG/vllm` `rdna2_extras` = v0.27.1 + `perf/rdna2_w4a16` merge `9ff87936` + PYNCCL bypass `3e05abc9`. Engine process: rebase-on-release is correct; keep this as the human branch.

## Survived (compile list is gfx1030-gated)

`CMakeLists.txt` `VLLM_GPU_ARCHES MATCHES gfx1030` still builds:

`q_gemm_rdna2` + prefill, `fa_rdna2`, `q_gemm_w8a16_fp8`, `gemm_w8a8_fp8_dense`, `moe_q_gemm_rdna2`, `moe_w8a16_rdna2`, `mxfp4_dot2_{dense,moe}`, indexer, `sparse_mla_rdna2`.

`q_gemm_rdna2_common.cuh` still issues explicit `__builtin_amdgcn_fdot2`. `fa_rdna2` still uses the same builtin for QK. Good.

## Did not move (still first)

| Bug | Still at tip |
|---|---|
| `fa_rdna2` occupancy | `__launch_bounds__(128, 1)` / `(256, 1)` — HIP second arg is **min waves/EU** |
| skinny occupancy | `wvSplitKrc_` still `amdgpu_waves_per_eu(1, 1)` |
| MLA prefill `load_row` | same blob as `66bb24d7` (`c3db1df8`): `chunk = i*THREADS+tid` into 64 float4s |

These are not “lost in rebase.” They were never fixed on `perf/rdna2_w4a16`.

## Merge gap

`csrc/rocm/moe_w8a16_fp8_rdna2.cu` is **on the branch but not in the gfx1030 CMake list**. It will not ship in `_rocm_C` until appended.

## Dispatch landmine on v0.27.1

```python
_ON_RDNA = _ON_GFX1X and not _ON_CDNA   # gfx11/12 only
on_gfx10x()                             # gfx1030
on_rdna()                               # NOT gfx1030
is_navi()                               # "gfx1" in arch — true on gfx1030 by accident
```

Use `on_gfx10x()` for V620. Do not write new Python as `if on_rdna()`.

`use_rocm_custom_paged_attention` now allows `_ON_GFX10X` for head 128 / block 16. Confirm the object that fires is `fa_rdna2`, not a gfx11 paged kernel. Profiler-proof it on the first extras bench.

## Overlay replay list (next tag)

1. Occupancy flip (`fa_rdna2` + skinny `waves_per_eu`).
2. MLA `load_row` OOB, then prefill `fdot2`.
3. Add `moe_w8a16_fp8_rdna2.cu` to CMake or delete the orphan.
4. Re-check every `on_rdna()` / `on_gfx1x()` call site against `on_gfx10x()`.
5. ISA dump: `v_dot2*` in W4A16/FA; no WMMA in gfx1030 objects.

Occupancy still first.
