# Skinny GEMM — silicon / HIP contract

Live: `skinny_gemms.cu`. **Incomplete.** Same occupancy ticket as `fa_rdna2`.

## gfx1030 path

`use_mfma` is `#if __HIP__MI3XX__` only. On gfx1030 it is **false** — scalar / packed FMA, not MFMA. The `mfma_f32_4x4x4bf16_1k` / `16x16x32` / FP8 MFMA sites are CDNA/GFX12 and must not run here.

`wvSplitKrc_` is `__attribute__((amdgpu_waves_per_eu(1, 1)))` — same trap as `fa_rdna2` `__launch_bounds__(*, 1)`. Other `wvSplitK_hf_*` kernels are `__launch_bounds__(WvPrGrp* THRDS)` with no min-waves (safer).

Fix: drop `(1, 1)`, use `amdgpu_waves_per_eu(4, 8)` after the FA occupancy flip. Do not “enable MFMA” on gfx1030.
