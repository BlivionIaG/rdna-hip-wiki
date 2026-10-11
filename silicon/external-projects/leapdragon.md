---
title: "leapdragon recipe — silicon review"
sources:
  - https://github.com/leapdragon/vllm-rdna2-recipe
  - https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/00-HARDWARE.md
  - https://github.com/opengfx1030/vllm-rdna/commit/e9ec6f631666f2e0c7826c3121d46163d84569e7
  - https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/01-PATCHES.md
  - https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/builds/shared/plugins/fd_rdna2/fd_rdna2/fd_kernel2.py
---

# leapdragon recipe — silicon review

Date: 2026-08-21. Engine: [../engine/leapdragon.md](../../engine/run-recipes/leapdragon.md). Repo: [leapdragon/vllm-rdna2-recipe](https://github.com/leapdragon/vllm-rdna2-recipe) (GPL-3.0-or-later **recipe**, not a fork). Occupancy still first. Do **not** copy their tok/s.[^github-leapdragon-vllm-rdna2-recipe]

Their box ≠ ours: 2× V620 in **×16** slots (TP=2), ROCm 7.2.3 in-container, `Qwen3.8-27B-GPTQ-4bit` (`head_dim=256`, **24Q/4KV = GQA-6**, hybrid GDN). We are 8× V620 / two 88096s / attested **7.14** / extras HIP. Take **intent**, Leave plugins and numbers.

## Their measured silicon (sourced, not our bench)

From [00-HARDWARE.md](https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/00-HARDWARE.md):[^github-main-00-hardware-md]

| Claim | Their measure | Our lock |
|---|---|---|
| DRAM | **506 GB/s** (232 W cap) | Spec 512 GB/s. Use as “near-peak is real,” not a 4×/88096 number. |
| DOT family | `v_dot2_f32_f16` / `v_dot4_i32_i8` / `v_dot8_i32_i4`; Triton `tl.dot` reaches them | Same ISA. We still want **HIP** `__builtin_amdgcn_fdot2`, not Triton as the path ([fp16-rdna2.md](../arch-and-memory/fp16-rdna2.md)). |
| WMMA/MFMA | none | Already locked. |
| LDS | **64 KiB / WG** | ISA hard cap. `head_dim=256` overflows stock Triton tiles. [^rocm-gpu-arch-specs][^llvm-baseinfo] |
| IC | 128 MB; one layer’s 43k KV ~45 MB **fits and lies** | Same as [infinity-cache.md](../arch-and-memory/infinity-cache.md): rotate ≥16 buffers; IC is this-step, not 16 layers. [^rocm-gpu-arch-specs] |
| P2P | **push 14.30 GB/s / pull 5.70 GB/s**; ×16 vs ×8 is 14 vs 7 | First **sourced** V620 GB/s. **Not** our 4×/88096 PIX number. Collectives must be **push** (peer store). |
| Finegrain | `hipDeviceMallocFinegrained` = **silent no-op**; need `hipDeviceMallocUncached` + `hipHostMallocCoherent` handshake | DeepEP analogue / custom AR ([deepep-v620.md](../interconnect/deepep-v620.md)). |

Plain FMA vs `tl.dot`: they measured **4.9×** slower. Matches [valu.md](../arch-and-memory/valu.md) — do not write scalar FMA “because no WMMA.”

## Take after occupancy (silicon intent)

**Do not vendor GPL plugins.** Re-implement.

### 1. Softmax segments = fill 36 WGP

Decode grid is `seqs × kv_heads × segments`. **4 KV heads** on one V620 is **tiny**: they quote 32 WGs vs **36 WGP** — idle chip. (GQA-6 is 24Q/4KV; do not call that GQA-4.) Scale segments toward `MIN_LAUNCH_GRID_SIZE_2D`, cap 64. Their plateau 16→64.

This is **grid fill**, not the extras `__launch_bounds__(N,1)` / `waves_per_eu(1,1)` trap. Triton unified-attn first; then prove `fa_rdna2` already fills or add a segment/KV-split there. Occupancy flip still first — a full grid of `(1,1)` waves is still one wave/EU.

### 2. LDS tile 16 at `head_dim ≥ 256`

64 KiB WG cap. Qwen3.8-27B `head_dim=256` is why their 0002 exists. Clamp Triton `TILE_*` to 16, `launch_num_stages≤2` ([qwen35.md](../../engine/run-recipes/qwen35.md)). `fa_rdna2` needs its **own** 256-head LDS budget — do not assume the Triton clamp covers HIP.

### 3. `BLOCK_KN` 256, after a sweep

Exllama GPTQ: they swept this model’s shapes. 256 → 8 wave32 / block, shallower K-split (less atomic C traffic): **400 vs 365 GB/s** at 128. 64 worse (306); 512 starves `o_proj`. “Skinny wants smaller blocks” was **backwards**.

Sweep extras skinny / W4. Do **not** paste 256. Same occupancy rule: `waves_per_eu(4,8)`.

### 4. `fa_rdna2` vs `fd_rdna2` — INT8 gather, not their slice

`fa_rdna2` is extras HIP FA2: decode split-K + prefill + short extend, D=128/256, **fp16 KV**, explicit `__builtin_amdgcn_fdot2`. Split-K grid.z already exists (`kv_splits ≤ 16`). Occupancy still `(1,1)`.

`fd_rdna2` is a Triton plugin: **decode / batch=1 / D=256 / int8-per-token-head / GQA=6** only. Load KV as packed `i32`, sext four bytes to f16, four `tl.dot` on `(TILE,64)` so D stays permuted `d=4w+b` until the final store. That dodge exists because their v0 unpack-then-reshape `(TILE,64,4)→(TILE,256)` forced Triton to write 256-wide operands through LDS (`ds_write_b16`). HIP does **not** need that — we already have `fdot2` on `half2`.

ISA they actually fire: i8 → f16 → `tl.dot` = **`fdot2`**, not `sdot4`. Matches [../kernels/kv-int8.md](../../kernels/kv-int8.md).

Plugin hardcodes `GQA=6` / `PAD=8` (`q.shape[1] % 6 == 0`). **Hits their 27B (24Q/4KV).** Misses GQA-4 and 7B 28/4. Take **fused INT8 gather** into occupancy-fixed `fa_rdna2` (VGPR cvt + scale, existing tiles). Do not vendor the plugin, do not copy the 4-way Q permute, do not land gather on `(1,1)`.

520 B/entry = 64 i32 K + scale + 64 i32 V + scale. Bandwidth win is the i8 load, not a new DOT.

### 5. Push AR is Later

`ar_rdna2` TP=2 push matches the 14.3 vs 5.7 asymmetry. extras PYNCCL first. Import **push + uncached**, not the monkey-patch. Not a 4-GPU PIX number.

## 0006 (new on recipe) — hybrid W4 / gfx10 skinny

Stock 0.27.1, not extras. `RDNAHybridW4A16` + `skinny_gemms_int4.cu` `__GFX10__` + scalar bf16 stub + **dequant→rocBLAS for M≥256**.

| Piece | Take / Leave |
|---|---|
| gfx10 guard on skinny int4 (wave32, 64 KiB, `fdot2`, DPP `row_shr`) | **Take the check** — occupancy / skinny family. DPP encodings still **unverified** on gfx1030; do not flip `wvSplitK` on. [^github-main-01-patches-md] |
| Asymmetric `uint4` ZP so Exllama can run | **Already Live** on extras: `e9ec6f63` `use_v2_format` / `zero_offset=0`. Do not import hybrid. [^commit-e9ec6f63] |
| dequant→rocBLAS fat-M | **Leave.** Their fused Triton GEMM is the 3× loser; our leftover is HIP `gptq_gemm_rdna2_prefill` + ConfigA, not unpack+BLAS. |
| `num_stages=1` on leftover Triton (0002 + 0006) | **Take.** stages>1 doubles K/V LDS at head 256 and halves occupancy. HIP FA is explicit — do not copy Triton tiles. |
| `PYTORCH_TUNABLEOP_TUNING=0` | Engine. Prefill M is prompt-length; tuning stalls + flips greedy. |
| `fd_rdna2` now `q≤4` for MTP verify | **Leave** the plugin. If INT8 gather ever lands in `fa_rdna2`, that card must cover `max_seqlen_q≤4` or MTP stays a regression. |

Wider Triton `BLOCK_M` 32–128 at D=256 is a measured dead end (fp32 acc `BLOCK_M×256` VGPR). Custom Triton **prefill** attn is also dead (0.70× / 0.58× vs stock). Occupancy still FA first.

## Not a dead end for us

Their “custom W4 GEMV is dead” assumes Exllama already at **~91% of 506 GB/s**. extras `fa_rdna2` / `skinny_gemms.cu` still sit on `(1,1)` *(as of 2026-08-21; at `1dc3bac4` FA paged decode is `waves_per_eu(4, 8)` and FA prefill is still `__launch_bounds__(N, 1)`, and skinny `wvSplitKrc_` is still `waves_per_eu(1, 1)`)*.[^ld-tip-occ] That card is **not** this dead-end.

Other dead ends they paid for (do not reopen): `dwordx4` KV (loads ~83% peak), 544 B KV align, `GPU_MAX_HW_QUEUES=8`, hoist attn range-mask, unpack-then-reshape INT8 to `(TILE,256)` LDS.

## Sources

- https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/00-HARDWARE.md
- https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/01-PATCHES.md
- https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/builds/shared/plugins/fd_rdna2/fd_rdna2/fd_kernel2.py (intent only; GPL — do not vendor)
- [../engine/leapdragon.md](../../engine/run-recipes/leapdragon.md), [../kernels/kv-int8.md](../../kernels/kv-int8.md), [fa-occupancy.md](../attention/fa-occupancy.md), [infinity-cache.md](../arch-and-memory/infinity-cache.md), [fp16-rdna2.md](../arch-and-memory/fp16-rdna2.md)

Footnotes:

[^github-leapdragon-vllm-rdna2-recipe]: GitHub - leapdragon/vllm-rdna2-recipe: Recipe for running vLLM productively on Radeon PRO V620 (Navi 21 / gfx1030) on Linux. Multiple Qwen models supported.. https://github.com/leapdragon/vllm-rdna2-recipe (checked 2026-10-11)
[^github-main-00-hardware-md]: vllm-rdna2-recipe/00-HARDWARE.md at main · leapdragon/vllm-rdna2-recipe. https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/00-HARDWARE.md (checked 2026-10-11)
[^commit-e9ec6f63]: opengfx1030/vllm-rdna commit `e9ec6f631666`. https://github.com/opengfx1030/vllm-rdna/commit/e9ec6f631666f2e0c7826c3121d46163d84569e7 (exists, verified via GitHub API 2026-10-11)
[^github-main-01-patches-md]: vllm-rdna2-recipe/01-PATCHES.md at main · leapdragon/vllm-rdna2-recipe. https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/01-PATCHES.md (checked 2026-10-11)
[^github-fd-rdna2-fd-kernel2-py]: vllm-rdna2-recipe/builds/shared/plugins/fd_rdna2/fd_rdna2/fd_kernel2.py at main · leapdragon/vllm-rdna2-recipe. https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/builds/shared/plugins/fd_rdna2/fd_rdna2/fd_kernel2.py (checked 2026-10-11)
[^llvm-baseinfo]: LLVM AMDGPUBaseInfo.cpp (IsaInfo), main snapshot eb61ef62df4f. https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp (checked 2026-10-11)
[^rocm-gpu-arch-specs]: ROCm docs 7.2.4, "GPU architecture specifications" (Radeon PRO V620 row: gfx1030, 32 GiB, 72 CU, wave 32/64, LDS 128 KiB, Infinity Cache 128 MiB, L2 4 MiB, GL1 128 KiB, L0 vector 16 KiB, L0 scalar 16 KiB, L0 instruction 32 KiB, VGPR file 512 KiB). https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html (checked 2026-10-11)
[^ld-tip-occ]: `csrc/rocm/fa_rdna2.cu` L283-L284 (decode `__launch_bounds__(128)` + `amdgpu_waves_per_eu(4, 8)`) and L1054 (prefill `(N, 1)`); `csrc/rocm/skinny_gemms.cu` L1345 (`amdgpu_waves_per_eu(1, 1)`), dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L283-L284 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/fa_rdna2.cu#L1054 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/skinny_gemms.cu#L1345
