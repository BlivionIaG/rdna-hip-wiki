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

## extras lock 2026-09-16 (tip `3092d635`, silicon commit `c350fa21`) — SUPERSEDED for dispatch

PR #11 / donor PR #5 (George Muravei-Alkhavoi / GeorgeMA-Strong). Dest: `opengfx1030/vllm-rdna` `rdna_extras`.

| Surface | Delta |
|---|---|
| `csrc/rocm/skinny_gemms.cu` `wvSplitK` | Drop shared static `Rdna2PersistBuf` + `rdna2_persist_zeros`. Output is per-call `torch::empty({N,M}, …)`. Shared persist storage aliased live projection outs across later GEMV / graph nodes. |
| `vllm/.../layers/utils.py` | gfx1030 FP16/BF16 decode: qualified `wvSplitK` for **n = 1..5**. `gemv_f16_rdna2` for other gfx10x, for **n = 6..8** on gfx1030, and when `VLLM_RDNA_DENSE_GEMV=1`. New `on_gfx1030()` gate. |
| `vllm/envs.py` | `VLLM_RDNA_DENSE_GEMV` default off (force GEMV for A/B). |
| `rdna_all_reduce.py` | Integer device-index normalize for opt-in `rdna_ar` (still `VLLM_RDNA_AR=0` default; not production AR). |

No `__launch_bounds__` / `waves_per_eu` change on `wvSplitKrc_`. No DOT builtin, LDS-tile, KV-quant, or CMake gfx1030 list change. Occupancy leftover still FA + skinny `(1,1)`. Do not invent numbers.

## extras lock 2026-09-18 (tip `3b59ee16`, silicon commit `5c4ab989`) — LIVE decode dispatch

**Revert** of the PR #5 / PR #11 gfx1030 `wvSplitK` decode port. Verified on `.176` (4× V620): patched path device-asserts in `wvSplitK_hf_big_` (`skinny_gemms.hip:1168`, `assert(false)`) and dies during cudagraph capture; revert restores PASS 3/3.

| Surface | Delta |
|---|---|
| `vllm/.../layers/utils.py` | gfx1030 decode: **`gemv_f16_rdna2` for all FP16 M≤8**. No gfx1030 `wvSplitK` / LLMM1 path. Comment: wvSplitK/LLMM1 are gfx9/gfx11+ only and RDNA build is numerically wrong on gfx1030. |
| `vllm/platforms/rocm.py` | Remove `on_gfx1030()` (only used by the reverted port). `on_gfx10x()` stays. |
| tests | Drop `test_gfx1030_decode_dispatch`. |

PersistBuf ownership fix inside `wvSplitK` host (`torch::empty` per call) remains in tree for non-gfx1030 skinny users; **gfx1030 no longer dispatches into it for decode**. No DOT / LDS-tile / CMake gfx1030 list / `__launch_bounds__` change. Occupancy leftover still FA + skinny `(1,1)`. Do not invent numbers.

## extras lock 2026-09-24 (tip `e1315629`, PR #17) — resident MoE decode + MoE dequant fix

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. HIP/ISA only from the V620 baseline port merge.

| Surface | Delta |
|---|---|
| `csrc/rocm/moe_resident_decode.cu` (**new**) | Native resident-layout W4A16 MoE skinny GEMV. Packed `int32` weights `[E,K/8,N]` after RDNA2 shuffle; scales fp16 `[E,K/gs,N]`. Inner loop `__builtin_amdgcn_fdot2`. Nibble dequant magic `0x64006400` / bias `0x64086408` with per-pair scales in `moe_resident_dequant_pair`. Symmetric uint4b8 only. |
| same | WG = **128** (4× wave32). `w13` grid `(ceil(inter/32), topk, M)`; `w2` grid `(ceil(hidden/32), 1, M)`. Four waves split K and reduce FP32 partials in LDS: `gates[4][32]` + `ups[4][32]` (SiLU gate×up), `routed_sums[4][32]` (down). Host gate **M = 1..4**; `K%8==0`; `intermediate`/`hidden` `%32==0`. Template `GroupSize=128` specialized else runtime. |
| `CMakeLists.txt` | Adds `moe_resident_decode.cu` to `VLLM_SKINNY_SRC` (`_rocm_C_skinny` object with `skinny_gemms*.cu`). Not a gfx1030 arch-list change. |
| `ops.h` / `torch_bindings.cpp` | Register `moe_resident_int4_decode`. Sibling `moe_skinny_int4_decode` (sequential Triton pack) unchanged. |
| `moe_q_gemm_rdna2.cu` | **Take** scale-after-signed-INT4: local `refresh_moe_group` builds ZP with scale=`1.0`, then `__hmul2(dq, group_scales)` before `dot22_8_f`. Folding scale into the 1024/64 encoding rounded quantized zero to nonzero fp16. Same GPTQv1 `zero_offset=1` and `fdot2` path. |

**Dispatch (opt-in, not dest default):** `VLLM_RDNA_MOE_RESIDENT=0`, `VLLM_RDNA_MOE_RESIDENT_SKINNY=0`. When both armed + SILU + contiguous fp16 + shape gates, M≤4 fires resident skinny; else tiled `_rdna2_fused_moe` on the same resident pack.

No `__launch_bounds__` / `waves_per_eu` on the new kernels. No EXL3 DOT, FA, KV-quant, or CMake gfx1030 list change. Occupancy leftover still FA + skinny `(1,1)`. Leave tok/s and flipping resident defaults on. Do not invent numbers.
