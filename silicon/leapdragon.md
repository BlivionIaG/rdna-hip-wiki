# leapdragon recipe — silicon review

Date: 2026-08-21. Engine: [../engine/leapdragon.md](../engine/leapdragon.md). Repo: [leapdragon/vllm-rdna2-recipe](https://github.com/leapdragon/vllm-rdna2-recipe) (GPL-3.0-or-later **recipe**, not a fork). Occupancy still first. Do **not** copy their tok/s.

Their box ≠ ours: 2× V620 in **×16** slots (TP=2), ROCm 7.2.3 in-container, `Qwen3.8-27B-GPTQ-4bit` (`head_dim=256`, GQA-4, hybrid GDN). We are 8× V620 / two 88096s / attested **7.14** / extras HIP. Steal **intent**, not plugins, not numbers.

## Their measured silicon (sourced, not our bench)

From [00-HARDWARE.md](https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/00-HARDWARE.md):

| Claim | Their measure | Our lock |
|---|---|---|
| DRAM | **506 GB/s** (232 W cap) | Spec 512 GB/s. Use as “near-peak is real,” not a 4×/88096 number. |
| DOT family | `v_dot2_f32_f16` / `v_dot4_i32_i8` / `v_dot8_i32_i4`; Triton `tl.dot` reaches them | Same ISA. We still want **HIP** `__builtin_amdgcn_fdot2`, not Triton as the path ([fp16-rdna2.md](fp16-rdna2.md)). |
| WMMA/MFMA | none | Already locked. |
| LDS | **64 KiB / WG** | ISA hard cap. `head_dim=256` overflows stock Triton tiles. |
| IC | 128 MB; one layer’s 43k KV ~45 MB **fits and lies** | Same as [infinity-cache.md](infinity-cache.md): rotate ≥16 buffers; IC is this-step, not 16 layers. |
| P2P | **push 14.30 GB/s / pull 5.70 GB/s**; ×16 vs ×8 is 14 vs 7 | First **sourced** V620 GB/s. **Not** our 4×/88096 PIX number. Collectives must be **push** (peer store). |
| Finegrain | `hipDeviceMallocFinegrained` = **silent no-op**; need `hipDeviceMallocUncached` + `hipHostMallocCoherent` handshake | DeepEP analogue / custom AR ([deepep-v620.md](deepep-v620.md)). |

Plain FMA vs `tl.dot`: they measured **4.9×** slower. Matches [valu.md](valu.md) — do not write scalar FMA “because no WMMA.”

## Steal after occupancy (silicon intent)

**Do not vendor GPL plugins.** Re-implement.

### 1. Softmax segments = fill 36 WGP

Decode grid is `seqs × kv_heads × segments`. GQA-4 on one V620 is **tiny**: they quote 32 WGs vs **36 WGP** — idle chip. Scale segments toward `MIN_LAUNCH_GRID_SIZE_2D`, cap 64. Their plateau 16→64.

This is **grid fill**, not the extras `__launch_bounds__(N,1)` / `waves_per_eu(1,1)` trap. Triton unified-attn first; then prove `fa_rdna2` already fills or add a segment/KV-split there. Occupancy flip still first — a full grid of `(1,1)` waves is still one wave/EU.

### 2. LDS tile 16 at `head_dim ≥ 256`

64 KiB WG cap. Qwen3.8-27B `head_dim=256` is why their 0002 exists. Clamp Triton `TILE_*` to 16, `launch_num_stages≤2` ([qwen35.md](../engine/qwen35.md)). `fa_rdna2` needs its **own** 256-head LDS budget — do not assume the Triton clamp covers HIP.

### 3. `BLOCK_KN` 256, after a sweep

Exllama GPTQ: they swept this model’s shapes. 256 → 8 wave32 / block, shallower K-split (less atomic C traffic): **400 vs 365 GB/s** at 128. 64 worse (306); 512 starves `o_proj`. “Skinny wants smaller blocks” was **backwards**.

Sweep extras skinny / W4. Do **not** paste 256. Same occupancy rule: `waves_per_eu(4,8)`.

### 4. INT8 KV flash-decode → `fa_rdna2`

`fd_rdna2` is Triton `tl.dot` over 520 B/entry (`int8_per_token_head`). Contract stays [../kernels/kv-int8.md](../kernels/kv-int8.md): fused dequant + `fdot2` **inside occupancy-fixed `fa_rdna2`**. Plugin is A/B only.

### 5. Push AR is Later

`ar_rdna2` TP=2 push matches the 14.3 vs 5.7 asymmetry. extras PYNCCL first. Import **push + uncached**, not the monkey-patch. Not a 4-GPU PIX number.

## Not a dead end for us

Their “custom W4 GEMV is dead” assumes Exllama already at **~91% of 506 GB/s**. extras `fa_rdna2` / `skinny_gemms.cu` still sit on `(1,1)`. That card is **not** this dead-end.

Other dead ends they paid for (do not reopen): `dwordx4` KV (loads ~83% peak), 544 B KV align, `GPU_MAX_HW_QUEUES=8`, hoist attn range-mask.

## Sources

- https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/00-HARDWARE.md
- https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/01-PATCHES.md
- [../engine/leapdragon.md](../engine/leapdragon.md), [fa-occupancy.md](fa-occupancy.md), [infinity-cache.md](infinity-cache.md), [fp16-rdna2.md](fp16-rdna2.md)
