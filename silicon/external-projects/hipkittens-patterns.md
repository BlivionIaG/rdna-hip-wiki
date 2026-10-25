---
title: "HipKittens scheduling and memory patterns mapped onto gfx1030 (wave32 fdot2)"
updated: 2026-10-25
sources:
  - https://arxiv.org/abs/2511.08083
  - https://arxiv.org/html/2511.08083
  - https://github.com/HazyResearch/HipKittens/tree/8cde3dd146bd1c4be78023b582c1f4375c48489f
  - https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/README.md
  - https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna5/ops/warp/sched/sched.cuh
  - https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna5/common/util.cuh
  - https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna5/ops/warp/register/tile/mma.cuh
  - https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna4/common/util.cuh
  - https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna4/ops/warp/memory/tile/global_to_shared.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/exl3_dot2_dense.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/exl3_dot2_common.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/q_gemm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/skinny_gemms_int4.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/2d461709e2bf864f85b23ede1f5171f122f13407
  - https://github.com/BlivionIaG/rdna-hip-wiki/pull/29
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/3b58c3917cae806659cc0ab7be4dee32f3031b16/silicon/occupancy/lds-dma-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/lds-and-tiles/lds-bank-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/occupancy/wave-priority-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/occupancy/trans-occupancy.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/hip-craft/hip-craft.md
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/arch-and-memory/architecture.md
---

# HipKittens scheduling and memory patterns mapped onto gfx1030 (wave32 fdot2)

Date: **2026-10-25** Europe/Paris. Assigned by VLLM_FORK_Manager in the GFX1030 Inference room as a wiki note only, with no card. The HipKittens (HK) code is a **Leave** as a dependency. This page takes **ideas only** and ranks them against EXL3 decode first ([UNC-26](../../kernels/quant/exl3.md), produce `-cb 3inst` → `fdot2`) and the W4A16 decode GEMV second ([w4a16.md](../../kernels/quant/w4a16.md), [skinny-gemm.md](../../kernels/gemm-moe/skinny-gemm.md)). No tok/s, no timing numbers, no dest or UNC change.

Inputs:
- One bounded read of the HK paper, arXiv 2511.08083 (HTML).[^hk-paper]
- The HK repo at `8cde3dd146bd1c4be78023b582c1f4375c48489f`.[^hk-tree]
- dest `opengfx1030/vllm-rdna` `rdna_extras` at `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. Origin has since moved to `2d461709` (PR #138 merged 2026-10-25 13:12 Europe/Paris), and `git diff --stat 8cc4e2d5 2d461709` over the four `.cu`/`.cuh` files cited here is empty, so every line cited below also holds at the new tip.[^dest-tip]

## Why HK's patterns don't map 1:1 (silicon)

- **HK targets:** CDNA3/CDNA4 MFMA on wave64 (`WARP_THREADS{64}`), plus a `cdna5` backend for gfx1250 that is wave32 but uses `__builtin_amdgcn_wmma_f32_16x16x32_*`.[^hk-readme][^hk-cdna4-util][^hk-cdna5-util][^hk-cdna5-mma] gfx1030 has neither MFMA nor WMMA. Its matrix work is `fdot2` (`V_DOT2_F32_F16`), which is an ordinary VALU instruction.[^hip-craft]
- **No second pipe:** HK's schedules overlap a matrix pipe with memory. On gfx1030 there is no separate matrix pipe to overlap with. What's left to overlap is **VALU (`fdot2` plus dequant/decode math) against VMEM latency**. The levers for that are occupancy (other resident waves), in-wave lookahead (independent loads issued early), and the issue-order knobs `s_setprio` and `sched_group_barrier`.[^hip-craft][^wave-prio][^trans-occ]
- **Decode is a GEMV:** each weight byte is read once and there is no B-operand reuse to stage. Both dest decode kernels stage only **A** (activations) in LDS and stream **B** (weights) from global memory straight into VGPRs.[^dest-exl3-v3-stage][^dest-q4-lds]
- **Geometry:** WGP mode has 4 SIMD32 sharing one LDS. LDS is 64 banks × 4 B, `bank = (byte_addr / 4) % 64`.[^arch][^lds-bank] Navi21 is a monolithic die with one L2 and one Infinity Cache.[^arch]

## Ranking (EXL3 decode first, W4A16 GEMV second)

| # | HK pattern | EXL3 decode (UNC-26) | W4A16 GEMV | Verdict |
|---|---|---|---|---|
| 1 | Register-file double buffering (HK's MI325X GEMM variant) | **Take, already Live** (v3 one-k-tile lookahead) | **Later** (only intra-iteration batching today) | Take |
| 2 | Cheap address generation (constant offsets, folded indices) | **Take, already Live** (`readfirstlane` + `constexpr` window positions) | **Later**, low value | Take |
| 3 | Compiler schedule hints (`sched_barrier` / `sched_group_barrier`, `s_setprio`) | **Later** (measured A/B only) | **Later** (measured A/B only) | Later |
| 4 | 4-wave interleave (one wave per SIMD, fine-grained mix) | **Later**, as an idea | **Later**, as an idea | Later |
| 5 | Bank-conflict-free tile layouts / swizzles | **Leave** for decode (A tile already conflict-free) | **Leave** for decode (pad already conflict-free) | Take the method, Leave their tables |
| 6 | Buffer→LDS async loads (LDS DMA) | **Leave** for decode | **Leave** for decode | Later (prefill only, per wiki PR #29) |
| 7 | 8-wave ping-pong (two waves per SIMD, compute/memory role swap) | **Leave** | **Leave** | Leave for decode; prefill Later |
| 8 | Wave specialization (producer/consumer) | **Leave** | **Leave** | Leave (HK itself rejects it) |
| 9 | Pinned registers / AGPR operands | **Leave** | **Leave** | Leave (no AGPR on gfx1030) |
| 10 | Chiplet / XCD grid ordering | **Leave** | **Leave** | Leave (monolithic Navi21) |

### 1. Register-file double buffering: Take (Live in EXL3, Later for W4A16)

- **What HK does:** the HK paper (App. E.1) describes how, on MI325X with roughly 64 KB of LDS, HK cannot double-buffer in shared memory. Instead it "double buffer[s] using the register file": it loads from HBM into a register buffer while the waves compute on the previous register tiles, then stores the buffer down with `ds_write`.[^hk-paper-e1] That LDS budget is the same class as gfx1030's 64 KiB, and it's the HK variant that transfers best.
- **EXL3 decode (Live):** `gemm_exl3_v3_kernel_rdna` already does exactly this, without the LDS write-back. "Tile words of k-tile kt+1 are loaded while kt is decoded (one tile of lookahead hides the global-load latency at low occupancy)": the `uint4 next[NW/4]` buffer is loaded before the 3inst decode and the `fdot2` loop of the current tile.[^dest-exl3-v3-body] Take this as the reference shape. Nothing in HK improves on it for decode.
- **W4A16 decode (Later):** `gemm_q4_kernel_rdna2` issues four `int4` weight loads up front *inside* each K step, then dequantizes them and runs `fdot2`. The comment says this relies on the backend to overlap their latency with the dequant work of earlier iterations.[^dest-q4-loop] It has no explicit cross-iteration lookahead like EXL3 v3. A one-step `next[]` buffer is a candidate, but it costs VGPRs. It only helps if the kernel stays on the same VGPR allocation rung ([vgpr-occupancy.md](../occupancy/vgpr-occupancy.md)). Whether it pays is **(unsourced)**, so measure before adopting.
- **Skinny `wvSplitK_int4`:** it unrolls `UNRL` weight loads into `bigB[YTILE][UNRL]` before the dot loop. That is the same idea expressed as unroll depth.[^dest-skinny-int4]

### 2. Cheap address generation: Take (Live in EXL3)

- **What HK does:** the paper notes that buffer loads "accept constant offsets that also help mitigate address calculation overheads" (App. D, Loads).[^hk-paper-d]
- **EXL3 decode (Live):** the v3 kernel picks its column quarter wave-uniformly with `__builtin_amdgcn_readfirstlane(t >> 5)` and dispatches to a template body. This keeps "every window position and tile-word index ... a compile-time constant (registers, no scratch)". `exl3_wpos_c` is `constexpr` so the window math folds once the loops unroll.[^dest-exl3-v3-addr] This is the gfx1030 form of HK's cheap address generation, and it's already in place.
- **W4A16:** the `b_ptr += 4 * size_n` stride walk is already simple.[^dest-q4-loop] Expect little from further work here **(unsourced)**.

### 3. Compiler schedule hints: Later (measured A/B)

- **What HK does:** HK uses `sched_barrier` to fence instruction clusters and `sched_group_barrier(mask, size, sync_id)` to pin interleaves such as VMEM×4 → MFMA×4 → DS×8. In ping-pong it brackets compute clusters with `s_setprio(1)` / `s_setprio(0)`.[^hk-paper-d4] The repo wraps these in `kittens::sched` (`set_priority`, `boost_priority` → `s_setprio_inc_wg`, `compiler_fence` → `sched_barrier(0)`). Its "expert scheduling" mode and `lock_simd` are gfx1250 `s_setreg` encodings.[^hk-sched]
- **On gfx1030:**
  - `S_SETPRIO` exists. The wiki already locks it as an arbitration knob, a measured A/B and never default craft (raise around a burst, restore 0).[^wave-prio]
  - `s_setprio_inc_wg` (GFX1250/GFX13) is not available.[^wave-prio]
  - The `sched_group_barrier` VALU, VMEM, DS and TRANS mask bits apply. The MFMA/WMMA bit is Dead.[^trans-occ]
  - HK's expert-mode and `lock_simd` `s_setreg` operands are gfx1250-specific. **Leave** them.[^hk-sched]
- **EXL3 and W4A16 mapping:** the interleave to try is VMEM (next tile words) → VALU (3inst decode, or nibble dequant) → VALU (`fdot2`). Treat it as a codegen A/B on the pinned ROCm 7.14.0 compiler, checking the ISA with `--save-temps`. Whether it beats the default scheduler on these kernels is **(unsourced)**.

### 4. 4-wave interleave: Later (idea only)

- **What HK does:** one wave per SIMD, with each wave issuing both compute and memory in a "carefully staggered sequence" over small tiles. HK picks it for imbalanced workloads (paper §3.3.2). It costs much more hot-loop code (paper Tab. 3).[^hk-paper-332]
- **On gfx1030:** this is the closest HK pattern to how a decode GEMV already works on gfx1030. Each wave mixes VMEM and VALU, and latency is hidden by lookahead within the wave plus other resident waves. The EXL3 v3 block is 128 threads, that is 4 waves.[^dest-exl3-v3-stage] Whether the hardware places those 4 waves one per SIMD32 of a WGP is **(unsourced)**. HK's version also assumes a CDNA CU with 4 SIMDs and a separate MFMA pipe.[^hk-paper-332]
- **Take** the principle that a wave should own both memory and math, which is what we already do. **Leave** the hand-staggered instruction listings. Revisit only together with item 3, as a scheduling A/B.

### 5. Bank-conflict-free tile layouts / swizzles: Take the method, Leave their tables

- **What HK does:** HK shows that one swizzle can't serve every AMD layout. It derives phase and bank behaviour per `ds_*` instruction with a solver, because "the phases are undocumented in the CDNA ISA" (paper §3.2.2, App. D.1–D.2).[^hk-paper-322]
- **Leave the tables:** HK's phase table is for **wave64 CDNA** lanes 0–63, and some instructions see only 32 banks. That conflicts with gfx1030's 64-bank WGP map. The wiki already Leaves CK's CDNA phase groups for the same reason.[^lds-bank]
- **Take the method:** derive or measure per instruction, and prefer XOR over padding near an LDS cliff ([lds-bank-occupancy.md](../lds-and-tiles/lds-bank-occupancy.md), [lds-tiles.md](../lds-and-tiles/lds-tiles.md)).[^lds-bank]
- **EXL3 decode:** v3 reads A rows from LDS as two 128-bit loads per row, and every lane of a wave reads the same address. The kernel calls them "broadcasts", and same-address reads are not conflicts.[^dest-exl3-v3-body][^lds-bank] There's nothing to swizzle, so this is a Leave for decode.
- **W4A16 decode:** `block_a[M_COUNT][256 + 8]` pads each row to 528 B. The source comment works the bank math with `% 32`. Under the wiki's 64-bank formula, `(m*132) % 64 = 4m` is still distinct for every `M_COUNT ≤ 8`, so the conclusion holds and only the comment's bank count is off.[^dest-q4-lds][^lds-bank] Leave for decode.

### 6. Buffer→LDS async loads: Leave for decode, Later for prefill (see wiki PR #29)

- **What HK does:** HK's CDNA4 `global_to_shared` issues `llvm_amdgcn_raw_buffer_load_lds` per lane. It applies the tile swizzle **to the global address** while the LDS destination advances linearly per instruction. The paper says the same: on AMD, "swizzling shared memory ... is instead accomplished by swizzling on the HBM addresses" (§3.2.2, Global).[^hk-g2s][^hk-paper-322]
- **On gfx1030:** not re-derived here. See wiki [PR #29](https://github.com/BlivionIaG/rdna-hip-wiki/pull/29) ([lds-dma-occupancy.md](https://github.com/BlivionIaG/rdna-hip-wiki/blob/3b58c3917cae806659cc0ab7be4dee32f3031b16/silicon/occupancy/lds-dma-occupancy.md), still open).[^pr29] That page records:
  - **Width:** gfx1030 has `lds`-bit loads only at dword width or narrower. There is no `dwordx4`.
  - **`M0`:** each load needs its own `M0` write.
  - **gfx1100:** it lacks the feature, so the code needs a gfx10 guard.
  - **VGPRs:** the gain is VGPR relief, not LDS bytes.
  - **Must:** re-probe on the ROCm 7.14.0 compiler before use.
- **EXL3 and W4A16 decode (Leave):** decode weights are read once and consumed straight from VGPRs, with no LDS staging of B.[^dest-exl3-v3-body][^dest-q4-loop] The A stage is a prologue before `__syncthreads()`, so there's no in-flight tile for LDS DMA to relieve.
- **Later (prefill):** HK's "swizzle the global address, keep LDS linear" trick fits PR #29's "lane-contiguous dword row per instruction" constraint. That would make it the way to build a swizzled LDS tile from dword `lds` loads in a future prefill or attention kernel. This fit is an inference from the two sources and has not been probed on gfx1030 **(unsourced)**.

### 7. 8-wave ping-pong: Leave for decode, prefill Later

- **What HK does:** each block has 8 waves, two resident per SIMD. A conditional barrier staggers the two groups. One wave issues only MFMA while its partner issues only memory, then they swap. HK says this "excels when compute and memory durations are roughly balanced" (paper §3.3.2, App. E.1 listing with `s_barrier` and `s_setprio`).[^hk-paper-332][^hk-paper-e1]
- **Why it doesn't fit gfx1030 decode:**
  - **No compute pipe to overlap:** the "compute" half would be `fdot2` VALU. There is no MFMA pipe to keep busy while the partner issues memory.[^hip-craft]
  - **Not balanced:** decode GEMV is memory-bound, not balanced. The paper sends imbalanced workloads to the 4-wave pattern, not ping-pong.[^hk-paper-332]
  - **Barrier cost:** the role swap costs `s_barrier` slots and block-wide sync. Our decode loops deliberately avoid barriers. The EXL3 dense kernel header says "No barrier inside the K loop".[^dest-exl3-hdr]
  - **Already covered:** the hardware already interleaves VALU and VMEM from other resident waves on the same SIMD32. Our W4A16 and EXL3 v1/v2 kernels run 8 waves per block (`THREADS_X 256`) for that reason.[^dest-q4-hdr][^dest-exl3-hdr]
- **Later (prefill):** a tiled-DOT2 prefill or `fa_rdna2`-class kernel with real LDS reuse could try a stagger. That would be a separate design read, not part of this note.

### 8–10. Leave

- **Wave specialization:** HK rejects producer/consumer on AMD because statically allocated registers on producer waves shrink the output tile (paper §3.3.1).[^hk-paper-331] The same static VGPR split applies on gfx1030 ([vgpr-occupancy.md](../occupancy/vgpr-occupancy.md)). **Leave.**
- **Pinned registers / AGPR operands:** HK pins tiles so that AGPRs can feed MFMA (paper §3.2.1, App. D.3).[^hk-paper-321] gfx1030 has no AGPR and no MFMA.[^hip-craft] **Leave.**
- **Chiplet / XCD grid ordering:** HK's Algorithm 1 remaps block IDs so that chunks land on one XCD's private L2 while also keeping LLC reuse, because the scheduler round-robins blocks across 8 XCDs (paper §3.4).[^hk-paper-34] Navi21 is one die with one GPU-wide L2.[^arch] **Leave.** Decode GEMV also has no output-tile reuse for a windowed traversal to exploit. A plain L2 window for future prefill GEMMs is a separate question **(unsourced)**.

## What this does not change

- No card, UNC, dest tip or `rdna_extras` change. HipKittens stays **Leave** as a dependency, and the code-level AITER and TransformerEngine backends are CDNA-only.
- EXL3 v3 lookahead and the `readfirstlane` address folding are already the gfx1030 answer for items 1–2. Any W4A16 lookahead or `sched_group_barrier` A/B is a future kernel task for whoever owns it, routed through VLLM_FORK_Manager.

## Sources

[^hk-paper]: Hu et al., "HipKittens: Fast and Furious AMD Kernels", arXiv 2511.08083. https://arxiv.org/abs/2511.08083 ; HTML https://arxiv.org/html/2511.08083 (read 2026-10-25). Validated on CDNA3 MI325X and CDNA4 MI355X (§1, Evaluation).
[^hk-paper-321]: arXiv 2511.08083 §3.2.1 "Developer-controlled register scheduling" and App. D.3 "Pinned register tiles".
[^hk-paper-322]: arXiv 2511.08083 §3.2.2 "Tiles for heterogeneous matrix core shapes" (Optimized tile memory: Shared, Global) and App. D.1–D.2 (single swizzle not possible; phase/bank solver, Tab. 5).
[^hk-paper-331]: arXiv 2511.08083 §3.3.1 "Wave specialization underperforms on AMD".
[^hk-paper-332]: arXiv 2511.08083 §3.3.2 "Performant scheduling patterns for AMD AI kernels" (8-wave ping-pong; 4-wave interleave; Tab. 3).
[^hk-paper-34]: arXiv 2511.08083 §3.4 "Optimizing the access patterns of non-programmable GPU memory" (cost model, Algorithm 1 XCD swizzle).
[^hk-paper-d]: arXiv 2511.08083 App. D, "Loads" paragraph (`buffer_load_dword` to LDS; constant offsets; `vmcnt`/`lgkmcnt`).
[^hk-paper-d4]: arXiv 2511.08083 App. D.4 "Compiler hints" (`sched.barrier`, `sched.group.barrier` masks, `s_setprio` around compute clusters).
[^hk-paper-e1]: arXiv 2511.08083 App. E.1 "Matrix Multiply" (8-wave hot loop with `s_barrier`/`s_setprio`; MI325X variant double-buffers in the register file and writes back with `ds_write`).
[^hk-tree]: HazyResearch/HipKittens at `8cde3dd146bd1c4be78023b582c1f4375c48489f`. https://github.com/HazyResearch/HipKittens/tree/8cde3dd146bd1c4be78023b582c1f4375c48489f
[^hk-readme]: HipKittens `README.md` L22 "We support CDNA3 and CDNA 4"; L94/L98 CDNA4 gfx950 and CDNA5 gfx1250 sections @ `8cde3dd146bd1c4be78023b582c1f4375c48489f`. https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/README.md#L22
[^hk-cdna4-util]: `include/cdna4/common/util.cuh` L32 `WARP_THREADS{64}`; L84 `chiplet_transform_chunked` @ `8cde3dd146bd1c4be78023b582c1f4375c48489f`. https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna4/common/util.cuh#L32
[^hk-cdna5-util]: `include/cdna5/common/util.cuh` L34 `WARP_THREADS{32}`; L116–125 and L305–375 gfx1250 LDS segments and A/B ring placement @ `8cde3dd146bd1c4be78023b582c1f4375c48489f`. https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna5/common/util.cuh#L305-L375
[^hk-cdna5-mma]: `include/cdna5/ops/warp/register/tile/mma.cuh` L31/L70 `__builtin_amdgcn_wmma_f32_16x16x32_bf16` @ `8cde3dd146bd1c4be78023b582c1f4375c48489f`. https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna5/ops/warp/register/tile/mma.cuh#L70
[^hk-sched]: `include/cdna5/ops/warp/sched/sched.cuh` L37–66 (gfx1250 `s_setreg` expert-mode operands), L121–123 `lock_simd`, L171–175 `s_setprio`, L193–197 `s_setprio_inc_wg`, L211–213 `sched_barrier(0)` @ `8cde3dd146bd1c4be78023b582c1f4375c48489f`. https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna5/ops/warp/sched/sched.cuh
[^hk-g2s]: `include/cdna4/ops/warp/memory/tile/global_to_shared.cuh` L19–22 (`bytes_per_thread`, per-warp bytes), L52–70 (swizzled global byte offset, linear `lds_addr`, `llvm_amdgcn_raw_buffer_load_lds`) @ `8cde3dd146bd1c4be78023b582c1f4375c48489f`. https://github.com/HazyResearch/HipKittens/blob/8cde3dd146bd1c4be78023b582c1f4375c48489f/include/cdna4/ops/warp/memory/tile/global_to_shared.cuh#L52-L70
[^dest-tip]: opengfx1030/vllm-rdna `rdna_extras` read at `8cc4e2d504f29bb4e92532a1730ffee1e1422099`; origin tip on 2026-10-25 is [`2d461709e2bf864f85b23ede1f5171f122f13407`](https://github.com/opengfx1030/vllm-rdna/commit/2d461709e2bf864f85b23ede1f5171f122f13407) (merge of PR #138). `git diff --stat 8cc4e2d5 2d461709 -- csrc/rocm/{exl3_dot2_dense.cu,exl3_dot2_common.cuh,q_gemm_rdna2.cu,skinny_gemms_int4.cu}` is empty.
[^dest-exl3-hdr]: `csrc/rocm/exl3_dot2_dense.cu` L17–22 (geometry follows `q_gemm_rdna2.cu`: `THREADS_X=256`, 8 waves wave32, A staged once in LDS, "No barrier inside the K loop") @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/exl3_dot2_dense.cu#L17-L22
[^dest-exl3-v3-addr]: `csrc/rocm/exl3_dot2_dense.cu` L300–331 (v3 design note: wave-uniform `readfirstlane` quarter, compile-time window/tile-word indices; `constexpr exl3_wpos_c`) and L501 `__builtin_amdgcn_readfirstlane(t >> 5)` @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/exl3_dot2_dense.cu#L300-L331
[^dest-exl3-v3-body]: `csrc/rocm/exl3_dot2_dense.cu` L408–458 (`gemm_exl3_v3_body`: one-k-tile `next[]` lookahead, 3inst decode via `exl3_rowpair`, A read as two `uint4` per row from LDS, `__builtin_amdgcn_fdot2`) @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/exl3_dot2_dense.cu#L408-L458 ; decode helpers in `exl3_dot2_common.cuh` L82–113 (`decode_3inst`) https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/exl3_dot2_common.cuh#L82-L113
[^dest-exl3-v3-stage]: `csrc/rocm/exl3_dot2_dense.cu` L61–63 (`V3_THREADS 128`), L479–498 (`__launch_bounds__(V3_THREADS)`, A rows → LDS, `__syncthreads()`), L651–664 (v3 gated to `gfx103*`), L683–692 (v3 for decode/MTP-verify `M ≤ 64`) @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/exl3_dot2_dense.cu#L479-L498
[^dest-q4-hdr]: `csrc/rocm/q_gemm_rdna2.cu` L20–26 (`THREADS_X=256` = 8 waves, `BLOCK_KN_SIZE=256`, split-K via `gridDim.z`, `fdot2`) and L61–62 @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/q_gemm_rdna2.cu#L20-L26
[^dest-q4-lds]: `csrc/rocm/q_gemm_rdna2.cu` L88–95 (`LDS_PAD = 8`, `block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD]`, bank comment using `% 32`) and L97–133 (A staging, `__syncthreads()`) @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/q_gemm_rdna2.cu#L88-L95
[^dest-q4-loop]: `csrc/rocm/q_gemm_rdna2.cu` L166–190 (four `int4` weight loads issued up front per K step, "the dependent dequant + FMA work below hides their latency"; `b_ptr += 4 * size_n`) @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/q_gemm_rdna2.cu#L166-L190
[^dest-skinny-int4]: `csrc/rocm/skinny_gemms_int4.cu` L15–23 (gfx1x wave32, 64 KiB LDS), L143–176 (`wvSplitK_int4_hf_sml_`: A → LDS then `__syncthreads()`), L192–231 (`bigB[YTILE][UNRL]` unrolled weight loads), L109–114 (`REDUCE_SUM_DPP_WAVE32`) @ `8cc4e2d504f29bb4e92532a1730ffee1e1422099`. https://github.com/opengfx1030/vllm-rdna/blob/8cc4e2d504f29bb4e92532a1730ffee1e1422099/csrc/rocm/skinny_gemms_int4.cu#L143-L231
[^pr29]: rdna-hip-wiki PR #29 (open), branch `idle/lds-dma-occupancy-2026-10-14`, head `3b58c3917cae806659cc0ab7be4dee32f3031b16`, `silicon/occupancy/lds-dma-occupancy.md` Take/Leave table. https://github.com/BlivionIaG/rdna-hip-wiki/blob/3b58c3917cae806659cc0ab7be4dee32f3031b16/silicon/occupancy/lds-dma-occupancy.md
[^lds-bank]: Wiki [lds-bank-occupancy.md](../lds-and-tiles/lds-bank-occupancy.md) @ wiki `224f07322d29a52b26758a6bcb6bd8c22708250d` (64 banks × 4 B in WGP mode, same-address broadcast not a conflict, XOR vs pad, Leave CDNA phase groups; ISA 70648 §2.3.1/§10.4.3). https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/lds-and-tiles/lds-bank-occupancy.md
[^wave-prio]: Wiki [wave-priority-occupancy.md](../occupancy/wave-priority-occupancy.md) @ wiki `224f07322d29a52b26758a6bcb6bd8c22708250d` (`S_SETPRIO` = USER_PRIO arbitration, measured A/B only; `s_setprio_inc_wg` GFX1250/GFX13 only; Leave CDNA around-MFMA recipes). https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/occupancy/wave-priority-occupancy.md
[^trans-occ]: Wiki [trans-occupancy.md](../occupancy/trans-occupancy.md) @ wiki `224f07322d29a52b26758a6bcb6bd8c22708250d` (`sched_barrier` / `sched_group_barrier` TRANS bit real on gfx1030; MFMA/WMMA bit Dead). https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/occupancy/trans-occupancy.md
[^hip-craft]: Wiki [hip-craft.md](../hip-craft/hip-craft.md) @ wiki `224f07322d29a52b26758a6bcb6bd8c22708250d` §5.1 (packed DOT inner loop), §7.4 (no WMMA/MFMA), §7.5 (no CDNA 5-issue; 1 VALU/cycle/SIMD32; VMEM separate pipe). https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/hip-craft/hip-craft.md
[^arch]: Wiki [architecture.md](../arch-and-memory/architecture.md) @ wiki `224f07322d29a52b26758a6bcb6bd8c22708250d` (WGP mode 4 SIMD32 share one LDS; RDNA 2 discrete parts are monolithic, one L2, one Infinity Cache). https://github.com/BlivionIaG/rdna-hip-wiki/blob/224f07322d29a52b26758a6bcb6bd8c22708250d/silicon/arch-and-memory/architecture.md
