# Fork vs upstream — gfx1030 features

Date: 2026-08-19. Tip **`rdna2_extras`** @ **`3e05abc9`** (read-only). Overlay of vLLM **v0.27.1** + gfx1030 work (`9ff87936` merge). Historical source: `perf/rdna2_w4a16`. Review: [rdna2-extras.md](rdna2-extras.md). Index of **features this fork added** vs stock vLLM. Completeness bar: **W4A16 dense**. On the branch = **In Progress**. Spec-only = **Todo**. Tickets: one card per row on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not edit that branch from this page.

Session dump: [notes/session-2026-08-17.md](notes/session-2026-08-17.md).

**Most complete** = W4A16 dense (still In Progress — leftover tile/occupancy).
**In Progress** = sources are on the branch. Not done.
**Todo / spec** = wiki + card, no fork commit yet.

Silicon index: [kernels/README.md](../kernels/README.md). W8A16 / W8A16-FP8 / W8A8-FP8 are all `fdot2` (W8A8-FP8 is **not** `sdot4`). No `q_gemm_w8a16_rdna2.cu` at tip.

Upstream on gfx1030: Triton / `torch.nn.functional.linear` / rocBLAS. AITER, CUTLASS, Marlin, FlashInfer, hipBLASLt-as-required-HW, `supports_fp8()`, `supports_mx()` — all off or CUDA/CDNA. Stock skinny (`wvSplitK` / `LLMM1`) is **not** compiled in: `on_gfx9() or on_gfx1x()` only. [kernels/triton-skinny-gemm.md](../kernels/triton-skinny-gemm.md).

**Dispatch watch (v0.27.1):** `on_rdna()` is gfx11/12 only. V620 is `on_gfx10x()`. New upstream `on_rdna()` gates skip gfx1030.

## GEMM

| Feature | Upstream | Fork | Status | Missing |
|---|---|---|---|---|
| **W4A16 dense** `q_gemm_rdna2` | `gptq_gemm_rdna3` is gfx1100. CUDA Marlin. | HIP dequant→`fdot2`. Default dispatch. | **Most complete** / In Progress | Prefill tile sweep, occupancy re-check after the FA flip. |
| W4A16 MoE `moe_q_gemm_rdna2` | Triton MoE excludes gfx10xx (#37826). | HIP fused MoE. | In Progress | TP>1 `output_topk`, decode `BLOCK_M`, GPU matrix. |
| W8A16 dense + MoE | None on gfx1030. | LUT→`fdot2`. No dedicated `q_gemm_w8a16_rdna2.cu` at tip. | In Progress | Same as W4A16 MoE. Not `sdot4`. |
| W8A16-FP8 dense + MoE | CUTLASS FP8 = CDNA. | LUT→`fdot2`. One-time transpose fix in `9ac015d0`. | In Progress | GPU verify of all shapes. Storage looks FP8; compute is `fdot2`. |
| W8A8-FP8 dense | CUTLASS / AITER FP8 MMA. | `750ca545` bit-trick→`fdot2`. | In Progress | GPU correctness pending (commit says so). Not `sdot4`. |
| mxfp4 dense + MoE | CUTLASS MX / Marlin = CDNA/CUDA. | `290715e6` unpack→`fdot2`. | In Progress | GPU smoke pending. |
| Skinny GEMM `skinny_gemms.cu` | Stock gfx1030 = `F.linear` / BLAS. `wvSplitK`/`LLMM1` need gfx9/gfx1x. | In-tree. MFMA path compiled out. | In Progress | `waves_per_eu(1,1)` — **same occupancy card** as FA. LLMM1 gfx1030 gate is a note on this card ([qwen35.md](qwen35.md)). |
| Native HIP FP16 MoE | `F.linear` / Triton grouped. | **Not on the branch.** | Todo | [fp16-moe.md](fp16-moe.md) — decode skinny + prefill grouped, `fdot2`. After occupancy. |
| Native HIP INT8 MoE | AITER CK CDNA. | **Not on the branch.** | Todo | [int8-moe.md](int8-moe.md) — W8A16 `fdot2` + W8A8 `sdot4`. |
| W8A8 INT8 `sdot4` | AITER CK CDNA. | **Not on the branch.** | Todo | [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md) |
| NVFP4 | FlashInfer / CUTLASS Blackwell. | **Not on the branch.** | Todo | [nvfp4.md](nvfp4.md) |
| INT2 / W2A16 | None. | **Not on the branch.** | Todo | [int2.md](int2.md) + [kernels/int2.md](../kernels/int2.md) |
| Mixed INT2/INT4 MoE | None. | **Not on the branch.** | Todo | same [int2.md](int2.md) — two unpackers, one DOT |
| W4A4 integer `sdot8` | FlashInfer MXFP4 W4A4 = E2M1 / SM100. | **Not on the branch.** | Todo / explore | [w4a4.md](w4a4.md) — i4×i4 only, after W8A8 |

## Attention / DSv4

| Feature | Upstream | Fork | Status | Missing |
|---|---|---|---|---|
| `fa_rdna2` / `RDNA_ATTN` | ROCM_ATTN is gfx11+ `attention.cu`. Triton FA. | Standalone backend, D=128/256. | In Progress | Occupancy (`launch_bounds` / `waves_per_eu`). Head-64 hole. Default path still often Triton. |
| `VLLM_USE_RDNA2_FA` gate | gfx11-only. | Opens gfx10x + Triton autotune. | In Progress | Gate ≠ FA kernel firing. |
| Triton fp16 sparse MLA | CDNA bf16 AITER MLA. | `add17dd7` default DSv4 path. | In Progress | Compile tax. Mix/spec off. No fat tile. |
| HIP sparse MLA decode | AITER / gfx950 HIP. | `9a344444` fp16 + H-generic. Env `VLLM_USE_RDNA2_MLA=1`. | In Progress | Inner loop scalar FMA, not `fdot2`. Env not default. |
| HIP sparse MLA prefill | Triton ragged prefill. | `66bb24d7` `sparse_mla_prefill_rdna2`. Same env. | In Progress | **`load_row` OOB** is the correctness gate. Then `fdot2` (both sides half). Plain fp16 KV. **Not** fat tile `q>1`. |
| Lightning Indexer HIP | AITER paged-MQA (CDNA). | `paged_mqa_logits_decode_rdna2` + int64 slot + `8496f4ca` radix top-k. | In Progress | H/D specialization. |
| DSv4 gfx10x routing | `deepseek_v4/amd/rocm.py` AITER/bf16. | `on_gfx10x()` → rdna2 module, fp16 workspace, inv-RoPE Triton. | In Progress | MTP off (`VLLM_DISABLE_DSPARK_MTP`). Mix off. |
| MHC fp16 | tilelang / bf16. | gfx10x fp16 gate. | In Progress | Not the load-bearing path. |
| Sage INT8 QK | None. | **Not on the branch.** | Todo | [sage-attention.md](sage-attention.md) |
| INT8 KV | Triton `int8_per_token*`. FP8 KV on MI. | **Not on the branch.** | Todo | [kv-int8.md](kv-int8.md) |
| MTP | method `mtp`. | **Off** (gate). | Todo | [mtp.md](mtp.md) — fat tile first |
| DFlash | method `dflash`. | **Not on the branch.** | Todo | [dflash.md](dflash.md) — same gate |
| DSpark | method `dspark`. | **Off** (gate). | Todo | [dspark.md](dspark.md) — same gate |

## Sourced, not in this fork (ikantkode overlay)

File-mounts on `blivioniag/vllm-rdna:v0.26.0`. Digest: [notes/ikantkode-qwen35.md](notes/ikantkode-qwen35.md). Contract: [qwen35.md](qwen35.md). Do not copy their tok/s here.

| Feature | Their tree | Our take | Status |
|---|---|---|---|
| LLMM1 gfx1030 + wvSplitK off | `utils.py` arch-gate | Confirm vs `skinny_gemms.cu`. wvSplitK stays off. | Todo — **same skinny/occupancy family** |
| Qwen3.5 / Gemma RMSNorm `(1+w)` | Triton fuse | After occupancy. HIP optional. | Todo |
| Qwen3.5 AWQ-vd recipe | `requant/quant.py` | Checkpoint post-pass, not a DOT. | Todo |
| Qwen3.5 GDN linear-attn | none (stock FLA) | Later. No HIP. | Todo / later |

## Engine glue (not kernels)

| Feature | Upstream | Fork | Status |
|---|---|---|---|
| `RDNA_ATTN` enum / `--attention-backend` | missing | added | In Progress (backend itself In Progress) |
| causal_conv1d padding (`total_entries`) | NULL_BLOCK_ID=0 collision | on extras (bounds-check + upstream PDL) | In Progress |
| W8A16 `weight_scale` alias | Marlin renamed to `_inv` | both names | In Progress (glue for W8A16-FP8) |
| gfx1030/gfx1100 all-reduce bypass | `vllm::all_reduce` CUDA dispatcher (broken under Torch 2.12) | `3e05abc9` → `_all_reduce_out_place` / PYNCCL (`use_custom_op_collectives` False, ROCm-wide) | In Progress (glue). Transport only. Does not put RCCL inside MoE GEMM. |

## Research / later (not fork features)

| Item | Status | Page |
|---|---|---|
| Stock Triton / ROCM_ATTN / skinny map | Snapshot `main` @ `49fb2ee` | [triton-rocm.md](triton-rocm.md), [baseline-order.md](baseline-order.md) |
| FlyDSL gfx1030 | Gate 0 — compiler maybe, shipped kernels no | [flydsl.md](flydsl.md) |
| DeepEP / PCIe A2A | Later — steal asymmetry, not MORI | [deepep.md](deepep.md) |
| ROCmFPX llama.cpp side project | Later — no vLLM port | [rocmfpx.md](rocmfpx.md), [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md) |

## Card list (for VLLM_FORK_Manager)

Reuse a card if it already exists. In-tree → In Progress. Spec-only → Todo. Retip In Progress cards to `rdna2_extras` @ `3e05abc9`.

1. W4A16 dense — bar; leftover tile/occupancy only
2. W4A16 MoE
3. W8A16 dense + MoE
4. W8A16-FP8
5. W8A8-FP8 dense
6. mxfp4 dense + MoE
7. Skinny GEMM — **same occupancy card** as `fa_rdna2` (LLMM1 gfx1030 note; stock is BLAS)
8. `fa_rdna2` / occupancy — current subject (**not fixed by rebase**)
9. Triton + HIP MLA decode + HIP MLA prefill — **same MLA card** (`66bb24d7`; **OOB first**, then `fdot2`; fat tile still Later)
10. Lightning Indexer HIP — includes `8496f4ca` radix top-k
11. DSv4 gfx10x routing (inv-RoPE / workspace / MTP gate)
12. NVFP4 — already on the board (Todo)
13. INT8 KV — already on the board (Todo)
14. Sage QK — already on the board (Todo)
15. INT2 / W2A16 — already on the board (Todo) — [int2.md](int2.md)
16. Mixed INT2/INT4 MoE — already on the board (Todo) — same page
17. MTP — already on the board (Todo) — [mtp.md](mtp.md)
18. DFlash — already on the board (Todo) — [dflash.md](dflash.md)
19. DSpark — already on the board (Todo) — [dspark.md](dspark.md)
20. Qwen3.5 / Gemma RMSNorm `(1+w)` — Todo — [qwen35.md](qwen35.md)
21. Qwen3.5 AWQ-vd recipe — Todo — same page
22. Qwen3.5 GDN linear-attn — Later — same page
23. W4A4 integer `sdot8` — Explore — [w4a4.md](w4a4.md) (not MXFP4/NVFP4 A4)
24. Native HIP FP16 MoE — Todo — [fp16-moe.md](fp16-moe.md)
25. Native HIP INT8 MoE (W8A16 + W8A8) — Todo — [int8-moe.md](int8-moe.md)
26. `3e05abc9` PYNCCL all-reduce bypass — In Progress (glue)
27. Baseline epic 0–5 (harness → stock map → skinny → Triton FA → HIP A/B → FlyDSL) — [baseline-order.md](baseline-order.md)
28. llama.cpp ROCmFPX on V620 — Later / side-project — [llamacpp-rocmfpx.md](llamacpp-rocmfpx.md)
29. DeepEP / PCIe A2A — Later, not a first ticket — [deepep.md](deepep.md)

Occupancy still first. Rebase did not fix occupancy, MLA `load_row` OOB, or the stock skinny gate. No tok/s invented here.
