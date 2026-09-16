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

## extras lock 2026-09-16 (tip `3092d635`, silicon commit `c350fa21`)

PR #11 / donor PR #5 (George Muravei-Alkhavoi / GeorgeMA-Strong). Dest: `opengfx1030/vllm-rdna` `rdna_extras`.

| Surface | Delta |
|---|---|
| `csrc/rocm/skinny_gemms.cu` `wvSplitK` | Drop shared static `Rdna2PersistBuf` + `rdna2_persist_zeros`. Output is per-call `torch::empty({N,M}, …)`. Shared persist storage aliased live projection outs across later GEMV / graph nodes. |
| `vllm/.../layers/utils.py` | gfx1030 FP16/BF16 decode: qualified `wvSplitK` for **n = 1..5**. `gemv_f16_rdna2` for other gfx10x, for **n = 6..8** on gfx1030, and when `VLLM_RDNA_DENSE_GEMV=1`. New `on_gfx1030()` gate. |
| `vllm/envs.py` | `VLLM_RDNA_DENSE_GEMV` default off (force GEMV for A/B). |
| `rdna_all_reduce.py` | Integer device-index normalize for opt-in `rdna_ar` (still `VLLM_RDNA_AR=0` default; not production AR). |

No `__launch_bounds__` / `waves_per_eu` change on `wvSplitKrc_`. No DOT builtin, LDS-tile, KV-quant, or CMake gfx1030 list change. Occupancy leftover still FA + skinny `(1,1)`. Do not invent numbers.
