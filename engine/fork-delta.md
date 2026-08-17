# Fork vs upstream — gfx1030 features

Date: 2026-08-17. Tip `perf/rdna2_w4a16` @ `9a344444` (read-only). Index of **features this fork added** vs stock vLLM. Completeness bar: **W4A16 dense**. On the branch = **In Progress**. Spec-only = **Todo**. Tickets: one card per row on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not edit that branch from this page.

**Most complete** = W4A16 dense (still In Progress — leftover tile/occupancy).
**In Progress** = sources are on the branch. Not done.
**Todo / spec** = wiki + card, no fork commit yet.

Silicon index: [kernels/README.md](../kernels/README.md). W8A16 / W8A16-FP8 / W8A8-FP8 are all `fdot2` (W8A8-FP8 is **not** `sdot4`). No `q_gemm_w8a16_rdna2.cu` at tip.

Upstream on gfx1030: Triton / `torch.nn.functional.linear` / rocBLAS. AITER, CUTLASS, Marlin, FlashInfer, hipBLASLt-as-required-HW, `supports_fp8()`, `supports_mx()` — all off or CUDA/CDNA.

## GEMM

| Feature | Upstream | Fork | Status | Missing |
|---|---|---|---|---|
| **W4A16 dense** `q_gemm_rdna2` | `gptq_gemm_rdna3` is gfx1100. CUDA Marlin. | HIP dequant→`fdot2`. Default dispatch. | **Most complete** / In Progress | Prefill tile sweep, occupancy re-check after the FA flip. |
| W4A16 MoE `moe_q_gemm_rdna2` | Triton MoE excludes gfx10xx (#37826). | HIP fused MoE. | In Progress | TP>1 `output_topk`, decode `BLOCK_M`, GPU matrix. |
| W8A16 dense + MoE | None on gfx1030. | LUT→`fdot2`. No dedicated `q_gemm_w8a16_rdna2.cu` at tip. | In Progress | Same as W4A16 MoE. Not `sdot4`. |
| W8A16-FP8 dense + MoE | CUTLASS FP8 = CDNA. | LUT→`fdot2`. One-time transpose fix in `9ac015d0`. | In Progress | GPU verify of all shapes. Storage looks FP8; compute is `fdot2`. |
| W8A8-FP8 dense | CUTLASS / AITER FP8 MMA. | `750ca545` bit-trick→`fdot2`. | In Progress | GPU correctness pending (commit says so). Not `sdot4`. |
| mxfp4 dense + MoE | CUTLASS MX / Marlin = CDNA/CUDA. | `290715e6` unpack→`fdot2`. | In Progress | GPU smoke pending. |
| Skinny GEMM `skinny_gemms.cu` | rocBLAS / Triton. | In-tree. MFMA path compiled out. | In Progress | `waves_per_eu(1,1)` — **same occupancy card** as FA. LLMM1 gfx1030 gate is a note on this card ([qwen35.md](qwen35.md)). |
| W8A8 INT8 `sdot4` | AITER CK CDNA. | **Not on the branch.** | Todo | [kernels/w8a8-mxfp4.md](../kernels/w8a8-mxfp4.md) |
| NVFP4 | FlashInfer / CUTLASS Blackwell. | **Not on the branch.** | Todo | [nvfp4.md](nvfp4.md) |
| INT2 / W2A16 | None. | **Not on the branch.** | Todo | [int2.md](int2.md) + [kernels/int2.md](../kernels/int2.md) |
| Mixed INT2/INT4 MoE | None. | **Not on the branch.** | Todo | same [int2.md](int2.md) — two unpackers, one DOT |

## Attention / DSv4

| Feature | Upstream | Fork | Status | Missing |
|---|---|---|---|---|
| `fa_rdna2` / `RDNA_ATTN` | ROCM_ATTN is gfx11+ `attention.cu`. Triton FA. | Standalone backend, D=128/256. | In Progress | Occupancy (`launch_bounds` / `waves_per_eu`). Head-64 hole. Default path still often Triton. |
| `VLLM_USE_RDNA2_FA` gate | gfx11-only. | Opens gfx10x + Triton autotune. | In Progress | Gate ≠ FA kernel firing. |
| Triton fp16 sparse MLA | CDNA bf16 AITER MLA. | `add17dd7` default DSv4 path. | In Progress | Compile tax. Mix/spec off. No fat tile. |
| HIP sparse MLA decode | AITER / gfx950 HIP. | `9a344444` fp16 + H-generic. Env `VLLM_USE_RDNA2_MLA=1`. | In Progress | Prefill still Triton. Inner loop scalar FMA, not `fdot2`. Env not default. |
| Lightning Indexer HIP | AITER paged-MQA (CDNA). | `paged_mqa_logits_decode_rdna2` + int64 slot fix. | In Progress | H/D specialization, topk is `torch.topk`. |
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
| causal_conv1d padding (`total_entries`) | NULL_BLOCK_ID=0 collision | on the branch | In Progress |
| W8A16 `weight_scale` alias | Marlin renamed to `_inv` | both names | In Progress (glue for W8A16-FP8) |

## Card list (for VLLM_FORK_Manager)

Reuse a card if it already exists. In-tree → In Progress. Spec-only → Todo.

1. W4A16 dense — bar; leftover tile/occupancy only
2. W4A16 MoE
3. W8A16 dense + MoE
4. W8A16-FP8
5. W8A8-FP8 dense
6. mxfp4 dense + MoE
7. Skinny GEMM — **same occupancy card** as `fa_rdna2` (LLMM1 gfx1030 note)
8. `fa_rdna2` / occupancy — current subject
9. Triton fp16 MLA + HIP MLA decode — **same MLA card** (`fdot2` later)
10. Lightning Indexer HIP
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

Occupancy still first. No tok/s invented here.
