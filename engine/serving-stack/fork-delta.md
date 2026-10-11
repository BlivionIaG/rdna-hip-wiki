# Fork vs upstream — gfx1030 features

> **Dest today (2026-10-11):** [`opengfx1030/vllm-rdna` `rdna_extras`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras) = vLLM **v0.31.0** port ([PR #47](https://github.com/opengfx1030/vllm-rdna/pull/47)), tip `1dc3bac4` (contains `8abf4c97`). Tips and branch names quoted below (`rdna2_extras` on BlivionIaG/vllm, `perf/rdna2_w4a16`) are **historical snapshots**. Serve locks (ROCm pin, all-reduce path, prefix caching) live in [fork/serve-locks.md](../../fork/serve-locks.md) (fork PR #10) and [toolchain/matrix.md](../../toolchain/matrix.md); this page only says what the engine does.

Date: 2026-08-19. Tip **`rdna2_extras`** @ **`3e05abc9`** (read-only). Overlay of vLLM **v0.27.1** + gfx1030 work (`9ff87936` merge). Historical source: `perf/rdna2_w4a16`. Review: [rdna2-extras.md](rdna2-extras.md). Index of **features this fork added** vs stock vLLM. Completeness bar: **W4A16 dense**. On the branch = **In Progress**. Spec-only = **Todo**. Tickets: one card per row on [project 4](https://github.com/users/BlivionIaG/projects/4). Do not edit that branch from this page.

Session dump: [notes/session-2026-08-17.md](../notes/session-2026-08-17.md).

**Most complete** = W4A16 dense (still In Progress — leftover tile/occupancy).
**In Progress** = sources are on the branch. Not done.
**Todo / spec** = wiki + card, no fork commit yet.

Silicon index: [kernels/README.md](../../kernels/README.md). W8A16 / W8A16-FP8 / W8A8-FP8 are all `fdot2` (W8A8-FP8 is **not** `sdot4`). No `q_gemm_w8a16_rdna2.cu` at tip.

Upstream on gfx1030: Triton / `torch.nn.functional.linear` / rocBLAS. AITER, CUTLASS, Marlin, FlashInfer, hipBLASLt-as-required-HW, `supports_fp8()`, `supports_mx()` — all off or CUDA/CDNA. Stock skinny (`wvSplitK` / `LLMM1`) is **not** compiled in: `on_gfx9() or on_gfx1x()` only. [kernels/triton-skinny-gemm.md](../../kernels/triton-skinny-gemm.md).

**Dispatch watch (v0.27.1):** `on_rdna()` is gfx11/12 only. V620 is `on_gfx10x()`. New upstream `on_rdna()` gates skip gfx1030. At `1dc3bac4` (v0.31.0) upstream `on_rdna()` is still gfx11/12; the dest adds `on_rdna_family()` (gfx10x or RDNA3/4) and `on_rdna2()` (gfx10x) in `vllm/platforms/rdna.py`, and fork behaviour in shared code is gated on them. Example: the EXL3 fp16 cast used `on_rdna()` and never fired on gfx1030; it now uses `on_rdna_family()`[^d-rdnapy][^d-exl3cast]

## State at `1dc3bac4` (2026-10-11, v0.31.0 port)

The tables below this section are the **2026-08-19 snapshot** and stay as history. This table re-checks each row against dest `rdna_extras` @ `1dc3bac4`.

| Row (snapshot) | Snapshot status | At `1dc3bac4` | Source |
|---|---|---|---|
| W4A16 dense `q_gemm_rdna2` | Most complete / In Progress | Built for gfx1030 together with `q_gemm_rdna2_prefill.cu`. | [^d-cmake] |
| W4A16 MoE `moe_q_gemm_rdna2` | In Progress | Built. Oracle priority `RDNA2_W4A16` → RDNA3 → upstream; the CT WNA16 RDNA2 method now transposes upstream's N-first buffers (`e5ecbfd20a`). | [^d-cmake][^d-port] |
| W8A16 dense + MoE | In Progress | `moe_w8a16_rdna2.cu` built; its compressed-tensors method had a `NameError` on the 0.28 line, fixed in the port. Still no `q_gemm_w8a16_rdna2.cu`. | [^d-cmake][^d-port] |
| W8A16-FP8 dense + MoE | In Progress | Dense `q_gemm_w8a16_fp8_rdna2.cu` built. **MoE `moe_w8a16_fp8_rdna2.cu` is excluded from the gfx1030 build**; FP8-weight MoE experts fall back. | [^d-cmake] |
| W8A8-FP8 dense | In Progress | `gemm_w8a8_fp8_dense_rdna2.cu` built. GPU validation state not re-checked *(unverified)*. | [^d-cmake] |
| mxfp4 dense + MoE | In Progress, GPU smoke pending | Kernels never ran on the 0.28 line (every call rejected). Three dequant bugs fixed in `523eb846e0`; `test_mxfp4_rdna2` 50/50; DSv4 serve still pending. | [^d-port-mxfp4] |
| Skinny GEMM | In Progress, `(1,1)` | `wvSplitKrc_` still `amdgpu_waves_per_eu(1, 1)`. | [^d-skinny] |
| W8A8 INT8 `sdot4` / W4A8 | Todo | W4A8 `sdot4` dense prefill landed **opt-in** (`VLLM_RDNA2_W4A8_SDOT4=1`, M ≥ 33). `moe_w4a8_rdna2.cu` built for gfx1030 only; MoE W4A8 serving not validated. W8A8 INT8×INT8 GEMM still not on the branch. | [^d-port-w4a8][^d-cmake] |
| `fa_rdna2` / `RDNA_ATTN` | In Progress, occupancy | Decode occupancy landed; prefill still `(N, 1)`. Head sizes still `(128, 256)` (head-64 hole stays). | [^d-heads] · [attention-dispatch.md](../attention/attention-dispatch.md) |
| `VLLM_USE_RDNA2_FA` gate | "Opens gfx10x + Triton autotune" | Env default off; `== "1"` pins the RDNA_ATTN backend in `rocm.py`. The launcher's `ATTN=fa` exports it (and `VLLM_FA_RDNA2_GQA_DECODE=1`; dest PR #53 would stop the latter). | [^d-faopt] |
| HIP sparse MLA decode / prefill, Lightning Indexer | In Progress | Kernels moved to `csrc/rocm/rdna/dsv4/` (`sparse_mla_decode.cu` also holds `sparse_mla_prefill_rdna2`; `indexer_mqa_logits.cu` holds `paged_mqa_logits_decode_rdna2`). Env `VLLM_USE_RDNA2_MLA` still present. RDNA2 decode now takes upstream's extra kwargs and returns 0 on the HIP path. `load_row` OOB status *(unverified)*. | [^d-dsv4k][^d-port] |
| DSv4 gfx10x routing, MTP gate | In Progress, MTP off via `VLLM_DISABLE_DSPARK_MTP` | Flag no longer exists. DSv4-Flash recipe: `MTP=0`, `ATTN=none`. | [^d-dsv4env] |
| INT8 KV | Not on the branch | Fused `IS_INT8` decode template and INT8 cache writer exist; decode op not registered and RDNA_ATTN accepts only `auto`/`float16` KV → **not served**. | [^d-int8] |
| MTP | Off (gate) | **Served** on Flash-Next (`flashnext-mtp2`, `MTP=2`), verify rows on FA-RDNA2 split decode (PR #29). | [^d-mtp2] |
| DFlash / DSpark | Not on branch / Off | Not served by any recipe. | [^d-recipes] |
| Qwen3.5 / Gemma RMSNorm `(1+w)` | Todo | RDNA Gemma RMSNorm opaque ops and a HIP `RMSNormGated` path are in tree (RDNA-gated). | [^d-norm][^d-port-gating] |
| Qwen3.5 GDN linear-attn | Later, no HIP | HIP GDN decode, `causal_conv1d_*_rdna2` and opaque `qwen_gdn_full_forward` built; HIP prefill chain opt-in (`VLLM_GDN_HIP_PREFILL=1`, recipes set 0; its unit tests fail), so prefill serves on Triton FLA. | [^d-gdnfiles][^d-port-gdn] |
| all-reduce bypass (PYNCCL) | In Progress (glue) | TP collectives are opaque custom ops on RDNA only; `rdna_ar` one-shot is first in the chain (gate min(`VLLM_RDNA_AR_MAX_KB`, `VLLM_RDNA_AR_ONESHOT_KB`), default 64 KiB), RCCL above. Whether it is on: [fork/serve-locks.md](../../fork/serve-locks.md). | [^d-port-gating][^d-ar] |
| CUDA graphs / capture | (FULL glue, coverage.md) | Recipes run `FULL_AND_PIECEWISE`; replay-copy wrapper, collective split and capture-freeze flags removed 2026-10-10. | [^d-port-replay][^d-recipes] |
| Scheduler | stock + `VLLM_ROCM_NO_MIXED_BATCH` | Mixed decode+prefill is the only mode (PR #38); adaptive prefill opt-in `VLLM_RDNA_DYNAMIC_PREFILL=1` feeds upstream's `long_prefill_token_threshold`. | [^d-pr38][^d-sched][^d-port] |
| Launchers | `scripts/serve_gfx1030_*.sh` | `tools/rdna/serve_rdna.sh RECIPE=<name>` + `tools/rdna/recipes/*.env`. | [^d-recipes] |
| `on_rdna()` handling | `on_gfx10x()` only | Dest gates `on_rdna_family()` / `on_rdna2()`; EXL3 fp16 cast now fires on gfx1030. | [^d-rdnapy][^d-exl3cast] |

## Snapshot 2026-08-19 (historical) — GEMM

| Feature | Upstream | Fork | Status | Missing |
|---|---|---|---|---|
| **W4A16 dense** `q_gemm_rdna2` | `gptq_gemm_rdna3` is gfx1100. CUDA Marlin. | HIP dequant→`fdot2`. Default dispatch. | **Most complete** / In Progress | Prefill tile sweep, occupancy re-check after the FA flip. |
| W4A16 MoE `moe_q_gemm_rdna2` | Triton MoE excludes gfx10xx (#37826). | HIP fused MoE. | In Progress | TP>1 `output_topk`, decode `BLOCK_M`, GPU matrix. |
| W8A16 dense + MoE | None on gfx1030. | LUT→`fdot2`. No dedicated `q_gemm_w8a16_rdna2.cu` at tip. | In Progress | Same as W4A16 MoE. Not `sdot4`. |
| W8A16-FP8 dense + MoE | CUTLASS FP8 = CDNA. | LUT→`fdot2`. One-time transpose fix in `9ac015d0`. | In Progress | GPU verify of all shapes. Storage looks FP8; compute is `fdot2`. |
| W8A8-FP8 dense | CUTLASS / AITER FP8 MMA. | `750ca545` bit-trick→`fdot2`. | In Progress | GPU correctness pending (commit says so). Not `sdot4`. |
| mxfp4 dense + MoE | CUTLASS MX / Marlin = CDNA/CUDA. | `290715e6` unpack→`fdot2`. | In Progress | GPU smoke pending. |
| Skinny GEMM `skinny_gemms.cu` | Stock gfx1030 = `F.linear` / BLAS. `wvSplitK`/`LLMM1` need gfx9/gfx1x. | In-tree. MFMA path compiled out. | In Progress | `waves_per_eu(1,1)` — **same occupancy card** as FA. LLMM1 gfx1030 gate is a note on this card ([qwen35.md](../run-recipes/qwen35.md)). |
| Native HIP FP16 MoE | `F.linear` / Triton grouped. | **Not on the branch.** | Todo | [fp16-moe.md](../moe/fp16-moe.md) — decode skinny + prefill grouped, `fdot2`. After occupancy. |
| Native HIP INT8 MoE | AITER CK CDNA. | **Not on the branch.** | Todo | [int8-moe.md](../moe/int8-moe.md) — W8A16 `fdot2` + W8A8 `sdot4`. |
| W8A8 INT8 `sdot4` | AITER CK CDNA. | **Not on the branch.** | Todo | [kernels/w8a8-mxfp4.md](../../kernels/w8a8-mxfp4.md) |
| NVFP4 | FlashInfer / CUTLASS Blackwell. | **Not on the branch.** | Todo | [nvfp4.md](../quant-formats/nvfp4.md) |
| INT2 / W2A16 | None. | **Not on the branch.** | Todo | [int2.md](../quant-formats/int2.md) + [kernels/int2.md](../../kernels/int2.md) |
| Mixed INT2/INT4 MoE | None. | **Not on the branch.** | Todo | same [int2.md](../quant-formats/int2.md) — two unpackers, one DOT |
| W4A4 integer `sdot8` | FlashInfer MXFP4 W4A4 = E2M1 / SM100. | **Not on the branch.** | Todo / explore | [w4a4.md](../quant-formats/w4a4.md) — i4×i4 only, after W8A8 |

## Snapshot 2026-08-19 (historical) — Attention / DSv4

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
| Sage INT8 QK | None. | **Not on the branch.** | Todo | [sage-attention.md](../attention/sage-attention.md) |
| INT8 KV | Triton `int8_per_token*`. FP8 KV on MI. | **Not on the branch.** | Todo | [kv-int8.md](../kv-cache/kv-int8.md) |
| MTP | method `mtp`. | **Off** (gate). | Todo | [mtp.md](../spec-decoding/mtp.md) — fat tile first |
| DFlash | method `dflash`. | **Not on the branch.** | Todo | [dflash.md](../spec-decoding/dflash.md) — same gate |
| DSpark | method `dspark`. | **Off** (gate). | Todo | [dspark.md](../spec-decoding/dspark.md) — same gate |

## Sourced, not in this fork (ikantkode overlay)

File-mounts on `blivioniag/vllm-rdna:v0.26.0`. Digest: [notes/ikantkode-qwen35.md](../notes/ikantkode-qwen35.md). Contract: [qwen35.md](../run-recipes/qwen35.md). Do not copy their tok/s here.

| Feature | Their tree | Our take | Status |
|---|---|---|---|
| LLMM1 gfx1030 + wvSplitK off | `utils.py` arch-gate | Confirm vs `skinny_gemms.cu`. wvSplitK stays off. | Todo — **same skinny/occupancy family** |
| Qwen3.5 / Gemma RMSNorm `(1+w)` | Triton fuse | After occupancy. HIP optional. | Todo |
| Qwen3.5 AWQ-vd recipe | `requant/quant.py` | Checkpoint post-pass, not a DOT. | Todo |
| Qwen3.5 GDN linear-attn | none (stock FLA) | Later. No HIP. | Todo / later |

## Snapshot 2026-08-19 (historical) — Engine glue (not kernels)

| Feature | Upstream | Fork | Status |
|---|---|---|---|
| `RDNA_ATTN` enum / `--attention-backend` | missing | added | In Progress (backend itself In Progress) |
| causal_conv1d padding (`total_entries`) | NULL_BLOCK_ID=0 collision | on extras (bounds-check + upstream PDL) | In Progress |
| W8A16 `weight_scale` alias | Marlin renamed to `_inv` | both names | In Progress (glue for W8A16-FP8) |
| gfx1030/gfx1100 all-reduce bypass | `vllm::all_reduce` CUDA dispatcher (broken under Torch 2.12) | `3e05abc9` → `_all_reduce_out_place` / PYNCCL (`use_custom_op_collectives` False, ROCm-wide) | In Progress (glue) *(historical @ `3e05abc9`; current all-reduce path: [fork/serve-locks.md](../../fork/serve-locks.md))*. Transport only. Does not put RCCL inside MoE GEMM. |

## Research / later (not fork features)

| Item | Status | Page |
|---|---|---|
| Stock Triton / ROCM_ATTN / skinny map | Snapshot `main` @ `49fb2ee` | [triton-rocm.md](../stock-baseline/triton-rocm.md), [baseline-order.md](../stock-baseline/baseline-order.md) |
| FlyDSL gfx1030 | Gate 0 — compiler maybe, shipped kernels no | [flydsl.md](../alt-engines/flydsl.md) |
| DeepEP / PCIe A2A | Later — steal asymmetry, not MORI | [deepep.md](../multi-gpu/deepep.md) |
| ROCmFPX llama.cpp side project | Later — no vLLM port | [rocmfpx.md](../quant-formats/rocmfpx.md), [llamacpp-rocmfpx.md](../alt-engines/llamacpp-rocmfpx.md) |

## Card list (for VLLM_FORK_Manager)

Reuse a card if it already exists. In-tree → In Progress. Spec-only → Todo. *(historical)* Retip In Progress cards to `rdna2_extras` @ `3e05abc9`. Tip bumps now belong to [fork/](../../fork/README.md).

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
15. INT2 / W2A16 — already on the board (Todo) — [int2.md](../quant-formats/int2.md)
16. Mixed INT2/INT4 MoE — already on the board (Todo) — same page
17. MTP — already on the board (Todo) — [mtp.md](../spec-decoding/mtp.md)
18. DFlash — already on the board (Todo) — [dflash.md](../spec-decoding/dflash.md)
19. DSpark — already on the board (Todo) — [dspark.md](../spec-decoding/dspark.md)
20. Qwen3.5 / Gemma RMSNorm `(1+w)` — Todo — [qwen35.md](../run-recipes/qwen35.md)
21. Qwen3.5 AWQ-vd recipe — Todo — same page
22. Qwen3.5 GDN linear-attn — Later — same page
23. W4A4 integer `sdot8` — Explore — [w4a4.md](../quant-formats/w4a4.md) (not MXFP4/NVFP4 A4)
24. Native HIP FP16 MoE — Todo — [fp16-moe.md](../moe/fp16-moe.md)
25. Native HIP INT8 MoE (W8A16 + W8A8) — Todo — [int8-moe.md](../moe/int8-moe.md)
26. `3e05abc9` PYNCCL all-reduce bypass — In Progress (glue) *(historical card; current path in [fork/serve-locks.md](../../fork/serve-locks.md))*
27. Baseline epic 0–5 (harness → stock map → skinny → Triton FA → HIP A/B → FlyDSL) — [baseline-order.md](../stock-baseline/baseline-order.md)
28. llama.cpp ROCmFPX on V620 — Later / side-project — [llamacpp-rocmfpx.md](../alt-engines/llamacpp-rocmfpx.md)
29. DeepEP / PCIe A2A — Later, not a first ticket — [deepep.md](../multi-gpu/deepep.md)

Occupancy still first. Rebase did not fix occupancy, MLA `load_row` OOB, or the stock skinny gate. No tok/s invented here.

## Sources

Linked inline on this page (each URL checked 2026-10-11):

- <https://github.com/users/BlivionIaG/projects/4>

Added in the 2026-10-11 sourcing pass:

- dest `docs/rdna2/v0.31.0-port-notes.md` — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/v0.31.0-port-notes.md>
- dest `vllm/platforms/rdna.py` (`on_rdna_family`, `on_rdna2`) — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/vllm/platforms/rdna.py>
- vLLM `vllm/platforms/rocm.py` (arch gates, `supports_fp8`, custom paged-attn, custom AR) — <https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py>
- vLLM PR #37826 — Triton MoE ROCm whitelist excludes gfx10xx — <https://github.com/vllm-project/vllm/pull/37826>
- vLLM PR #34709 — wvSplitK skinny GEMM for gfx1x (gfx1030 not included) — <https://github.com/vllm-project/vllm/pull/34709>

[^d-rdnapy]: dest `vllm/platforms/rdna.py` (`on_rdna_family`, `on_rdna2`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/platforms/rdna.py>
[^d-exl3cast]: dest `vllm/model_executor/models/qwen3_5.py` `_exl3_needs_fp16_cast()` uses `on_rdna_family()` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/model_executor/models/qwen3_5.py#L108-L114>
[^d-cmake]: dest `CMakeLists.txt` gfx1030 source list (incl. the `moe_w8a16_fp8_rdna2.cu` exclusion) and RDNA-generic EXL3 list at `1dc3bac4` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1517-L1572>
[^d-port]: dest v0.31.0 port notes at `1dc3bac4` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md>
[^d-port-mxfp4]: dest v0.31.0 port notes, "MXFP4 kernels — fixed `523eb846e0`" — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L509-L523>
[^d-skinny]: dest `csrc/rocm/skinny_gemms.cu` `wvSplitKrc_` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/skinny_gemms.cu#L1344-L1346>
[^d-port-w4a8]: dest v0.31.0 port notes, "W4A8 prefill (opt-in)" — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L279-L297>
[^d-heads]: dest `rdna_attn.py` `_SUPPORTED_HEAD_SIZES = (128, 256)` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/v1/attention/backends/rdna_attn.py#L62>
[^d-faopt]: dest `vllm/envs.py` `VLLM_USE_RDNA2_FA` (default `"False"`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/envs.py#L1442-L1444>; `vllm/platforms/rocm.py` RDNA_ATTN pin — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/platforms/rocm.py#L1134-L1147>; launcher `rdna_select_attention` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/tools/rdna/rdna_launcher_common.sh#L106-L118>
[^d-dsv4k]: dest DSv4 RDNA kernels under `csrc/rocm/rdna/dsv4/` (`sparse_mla_decode.cu` also defines `sparse_mla_prefill_rdna2`; `indexer_mqa_logits.cu` defines `paged_mqa_logits_decode_rdna2`) — <https://github.com/opengfx1030/vllm-rdna/tree/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna/dsv4>
[^d-dsv4env]: dest `tools/rdna/recipes/dsv4-flash.env` (`MTP=0`, `ATTN=none`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/tools/rdna/recipes/dsv4-flash.env>; `VLLM_DISABLE_DSPARK_MTP` has no match in `vllm/`, `csrc/` or `tools/rdna/` at `1dc3bac4`
[^d-int8]: dest INT8 KV writer binding ("Not wired up yet") — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/torch_bindings.cpp#L430-L440>; RDNA_ATTN KV dtypes — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/v1/attention/backends/rdna_attn.py#L582-L583>
[^d-mtp2]: dest `tools/rdna/recipes/flashnext-mtp2.env` (`MTP=2`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/tools/rdna/recipes/flashnext-mtp2.env>; dest PR #29 FA-RDNA2 decode for spec-decode verify (merged) — <https://github.com/opengfx1030/vllm-rdna/pull/29>
[^d-recipes]: dest recipes at `1dc3bac4` — <https://github.com/opengfx1030/vllm-rdna/tree/1dc3bac4cda3201a1b4fb345e044578e84a272cf/tools/rdna/recipes>
[^d-norm]: dest `vllm/model_executor/layers/layernorm.py` (`GemmaRMSNorm`, HIP `RMSNormGated.forward_hip`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/model_executor/layers/layernorm.py#L165>, <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/model_executor/layers/layernorm.py#L392-L401>
[^d-port-gating]: dest v0.31.0 port notes, "RDNA gating" — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L83-L152>
[^d-gdnfiles]: dest GDN HIP kernels (`gdn_decode_rdna2.cu`, `gdn_prefill_*_rdna2.cu`, `causal_conv1d_rdna2.cu`) — <https://github.com/opengfx1030/vllm-rdna/tree/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm>; recipes set `VLLM_GDN_HIP_PREFILL=0` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/tools/rdna/recipes/flashnext.env>
[^d-port-gdn]: dest v0.31.0 port notes, GDN HIP prefill opt-in and failing unit tests — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L540-L557>
[^d-ar]: dest `vllm/distributed/device_communicators/rdna_all_reduce.py` `resolve_gate_kb()` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/distributed/device_communicators/rdna_all_reduce.py#L52-L80>; `csrc/rocm/rdna_allreduce.cuh` `RDNA_AR_ONESHOT_MAX` (compile-time default) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cuh#L66-L70>; runtime override in `rdna_allreduce.cu` — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cu#L88-L91>
[^d-port-replay]: dest v0.31.0 port notes, "Replay-copy wrapper removed (2026-10-10)" — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/docs/rdna2/v0.31.0-port-notes.md#L772-L829>
[^d-pr38]: dest PR #38 "restore upstream mixed decode+prefill batching" (merged 2026-10-06, commit `e3bf3d338f`; removes `VLLM_ROCM_NO_MIXED_BATCH`) — <https://github.com/opengfx1030/vllm-rdna/pull/38>
[^d-sched]: dest `vllm/v1/core/sched/scheduler.py` at `1dc3bac4` (no `_no_mixed` / `VLLM_ROCM_NO_MIXED_BATCH`) — <https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/v1/core/sched/scheduler.py>
