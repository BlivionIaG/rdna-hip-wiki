# Skinny GEMM — silicon / HIP contract

Live: `skinny_gemms.cu`. **Incomplete.** Same occupancy ticket as `fa_rdna2`.

## gfx1030 path

`use_mfma` is `#if __HIP__MI3XX__` only. On gfx1030 it is **false** — scalar / packed FMA, not MFMA. The `mfma_f32_4x4x4bf16_1k` / `16x16x32` / FP8 MFMA sites are CDNA/GFX12 and must not run here.

`wvSplitKrc_` is `__attribute__((amdgpu_waves_per_eu(1, 1)))` — same trap as `fa_rdna2` `__launch_bounds__(*, 1)`. Other `wvSplitK_hf_*` kernels are `__launch_bounds__(WvPrGrp* THRDS)` with no min-waves (safer).

Fix: drop `(1, 1)`, use `amdgpu_waves_per_eu(4, 8)` after the FA occupancy flip. Do not “enable MFMA” on gfx1030.

## MoE skinny int4 (recipe 0008) @ `5c3c0c6f`

- `skinny_gemms_int4.cu` gained wave-per-row W4A16 MoE skinny GEMV (LDS A tile, `__builtin_amdgcn_fdot2` / gfx10 path; LDS_SIZE 64 KiB).
- `VLLM_ROCM_MOE_SKINNY` **default True** at tip `5c3c0c6f` (restored in `59237b3d`). Tip history: briefly flipped default-off (`1c1dbee8`, bad test), then restored — kernel was correct; unit-scale synthetic weights overflowed FP16.
- **Leave** RDNAHybridW4A16 + bf16 on gfx10 via skinny (Triton abort) — op rejects that combo cleanly; fp16 path stays.
- Does not replace shuffled RDNA2 fused HIP MoE (`moe_gptq_gemm_rdna2`).
- Occupancy note: existing `wvSplitKrc_` `amdgpu_waves_per_eu(1,1)` trap still open; same ticket as FA.
