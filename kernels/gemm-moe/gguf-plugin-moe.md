---
title: "GGUF plugin MoE kernels (vllm-gguf-plugin fork) on gfx1030 — silicon lock"
updated: 2026-11-17
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/tools/rdna/recipes/flashnext-gguf-mtp0.env
  - https://github.com/opengfx1030/vllm-rdna/compare/34e2ebc52ade9dd314aa77f65fdcc792f133cb8c...b56907c4339779d2076f0886532833689aec93ec
  - https://github.com/opengfx1030/vllm-rdna/commit/b56907c4339779d2076f0886532833689aec93ec
  - https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/docker/Dockerfile.rdna
  - https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/vllm/model_executor/model_loader/gguf_plugin_compat.py
  - https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/docs/rdna2/gguf-flashnext-2026-11-16.md
  - https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/tools/rdna/recipes/flashnext-gguf-mtp0.env
  - https://github.com/opengfx1030/vllm-gguf-plugin/commit/b1c5e038d7cca09929e408ea76331485f1b94ea9
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/moe.cuh
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/mmq.cuh
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/ggml-common.h
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/__init__.py
---
# GGUF plugin MoE kernels (vllm-gguf-plugin fork) on gfx1030

Read at dest `rdna_extras` tip `b56907c4` (`b56907c4339779d2076f0886532833689aec93ec`, merge of PR #189, read 2026-11-17).[^tip] The range `34e2ebc5..b56907c4` (+22) adds the `qwen4exp_gguf` load format (PR #169 with #174, #177, #189).[^cmp] It changes **nothing under `csrc/rocm`** and no extras `__launch_bounds__`, DOT, LDS tile or CMake gfx list.[^cmp] The new silicon surface is out of tree: the GGUF GEMMs run through `vllm-gguf-plugin`, and the image now builds the opengfx1030 fork of it for gfx1030 only.[^docker] No tok/s here.

## What the image builds

| Item | Value @ `b56907c4` |
|---|---|
| Plugin | `opengfx1030/vllm-gguf-plugin` at `GGUF_PLUGIN_REF=b1c5e038` [^docker][^plug] |
| Arch | `GGUF_PLUGIN_ARCH=gfx1030` → `PYTORCH_ROCM_ARCH`. Reason given: the kernels are tested nowhere else and each arch adds a full HIP compile [^docker] |
| Build gate | asserts `MOE_MMQ_MAX_EXPERTS` in the plugin `__init__.py` and `_C_gguf.abi3.so` > 1,000,000 B [^docker] |
| Fork marker | `MOE_MMQ_MAX_EXPERTS = 1 << 20` [^init] |

## Kernel shape (plugin fork, ROCm branch)

- `WARP_SIZE_GGUF` is 32.[^common]
- Q5_1 MMQ tile on ROCm: `MMQ_X_Q5_1 64`, `MMQ_Y_Q5_1 128`, `NWARPS_Q5_1 8` (the non-ROCm branch is 4 / 32 / 4).[^mmq]
- Each `moe_q*` kernel carries `__launch_bounds__(WARP_SIZE_GGUF * NWARPS_<type>, 2)` under `USE_ROCM` only; the block is `(WARP_SIZE_GGUF, nwarps, 1)`.[^moe-lb]
- Y tiles live in static LDS: `tile_y_qs[mmq_x * WARP_SIZE_GGUF]` (int) and `tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1]` (half2).[^moe-lds]
- Dispatch in the plugin: ≤ 64 tokens → `ggml_moe_a8_vec` (per slot, no expert limit); above that → `moe_align_block_size` + `ggml_moe_a8` (MMQ, `moe_q`).[^doc] The seam hard-codes the same threshold as `_PLUGIN_MMQ_MIN_TOKENS = 65`.[^seam-thr]

## Three kernel defects and how the dest handles them

| # | Defect | Seen on gfx1030 | Fix @ `b56907c4` |
|---|---|---|---|
| 1 | `topk_ids == -1` (vLLM "no expert") read as `expert * stride` unchecked | GPU memory fault at CUDA-graph warmup (dummy batch routes every slot to -1) | seam: weight × (id ≥ 0), id clamped to 0 → exact zero contribution [^seam-hard] |
| 2 | stock `moe_q` returns early for `exp_idx > 255` | experts 256+ contribute nothing (512-expert model) | fork: only `exp_idx < 0` returns, and blocks past `num_tokens_post_padded` exit before reading `expert_ids` [^moe-idx]. Without the fork marker the seam slices batches into 64-token chunks on the vector kernel; `VLLM_QWEN4EXP_GGUF_MMQ_MAX_EXPERTS` overrides the bound [^seam-thr] |
| 3 | MMQ tile loader reads a few blocks past the last row when K is not a multiple of its per-warp span (K 160 in the TP4 Q5_1 / Q8_0 down projections) | Memory Fault in `moe_q5_1` on every rank at boot when the tensor ends on a 2 MiB page boundary | seam: 4096 B zero slack after `w13_weight` / `w2_weight`, only when MMQ is allowed [^seam-pad] |

**Lesson for our own kernels:** tile loaders on gfx1030 that over-read the tail of a weight buffer can fault when the allocation ends exactly on a page; either guard the tail or pad the allocation. Sentinel expert ids (-1) must be masked before the address is formed.[^seam-pad][^seam-hard]

## Recipe env (opt-in)

`tools/rdna/recipes/flashnext-gguf-mtp0.env` sets `--load-format qwen4exp_gguf`, `ATTN=fa`, `VLLM_GDN_HIP_PREFILL=1`, `VLLM_RDNA_FUSED_SE=1`, `VLLM_RDNA_DENSE_INT8=0`, `VLLM_RDNA_DENSE_GEMV=0` *(historical @ `b56907c4`: the variable was removed by the v0.34 merge `2bf77978`, setting it now raises at boot, and the recipe no longer sets it at `00f106c5`; do not copy)*,[^gemv-removed] `VLLM_RDNA_AR=1` with `AR_MAX_KB`/`AR_ONESHOT_KB` 64, `GPU_MAX_HW_QUEUES=4`, and both `VLLM_CAUSAL_CONV1D_RDNA2_*` off.[^recipe] No C++ default changed.[^cmp]

## Not covered

MTP (the GGUF has no MTP head), vision, TP2/TP8/EP, gfx1100 and CDNA were not run.[^doc] Occupancy of the plugin kernels on gfx1030 (VGPR/LDS per block from the built `.so`) was not measured here (unsourced; open).

## Sources

[^tip]: Dest tip [`b56907c4`](https://github.com/opengfx1030/vllm-rdna/commit/b56907c4339779d2076f0886532833689aec93ec), "Merge pull request #189: Image: build the GGUF plugin fork; flashnext-gguf-mtp0 recipe; README and serving docs" (2026-11-16 23:33 Paris).
[^cmp]: [Compare `34e2ebc5...b56907c4`](https://github.com/opengfx1030/vllm-rdna/compare/34e2ebc52ade9dd314aa77f65fdcc792f133cb8c...b56907c4339779d2076f0886532833689aec93ec): 22 commits; files touched are README, docker, docs, tests, a recipe, and Python under `vllm/model_executor/model_loader/` and `vllm/models/qwen4_exp/`.
[^docker]: [docker/Dockerfile.rdna:107–121](https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/docker/Dockerfile.rdna#L107-L121) @ b56907c4339779d2076f0886532833689aec93ec.
[^plug]: opengfx1030/vllm-gguf-plugin commit [`b1c5e038`](https://github.com/opengfx1030/vllm-gguf-plugin/commit/b1c5e038d7cca09929e408ea76331485f1b94ea9) "Expose MOE_MMQ_MAX_EXPERTS so consumers can tell the MMQ MoE kernel has no 255-expert bound".
[^init]: [vllm_gguf_plugin/__init__.py:11](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/__init__.py#L11) @ b1c5e038d7cca09929e408ea76331485f1b94ea9.
[^common]: [csrc/gguf/ggml-common.h:4](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/ggml-common.h#L4) @ b1c5e038d7cca09929e408ea76331485f1b94ea9.
[^mmq]: [csrc/gguf/mmq.cuh:259–267](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/mmq.cuh#L259-L267) @ b1c5e038d7cca09929e408ea76331485f1b94ea9.
[^moe-lb]: [csrc/gguf/moe.cuh:145–150, 181](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/moe.cuh#L145-L181) @ b1c5e038d7cca09929e408ea76331485f1b94ea9 (same pattern for Q4_1 … Q3_K at L209, 270, 331, 392, 453, 514).
[^moe-lds]: [csrc/gguf/moe.cuh:50–51](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/moe.cuh#L50-L51) @ b1c5e038d7cca09929e408ea76331485f1b94ea9.
[^moe-idx]: [csrc/gguf/moe.cuh:32–37](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/moe.cuh#L32-L37) @ b1c5e038d7cca09929e408ea76331485f1b94ea9; stock bound quoted in the dest doc table row 2.
[^seam-thr]: [gguf_plugin_compat.py:72–91](https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/vllm/model_executor/model_loader/gguf_plugin_compat.py#L72-L91) @ b56907c4339779d2076f0886532833689aec93ec.
[^seam-pad]: [gguf_plugin_compat.py:94–112, 156–162](https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/vllm/model_executor/model_loader/gguf_plugin_compat.py#L94-L162) @ b56907c4339779d2076f0886532833689aec93ec.
[^seam-hard]: [gguf_plugin_compat.py:115–154](https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/vllm/model_executor/model_loader/gguf_plugin_compat.py#L115-L154) @ b56907c4339779d2076f0886532833689aec93ec.
[^doc]: [docs/rdna2/gguf-flashnext-2026-11-16.md](https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/docs/rdna2/gguf-flashnext-2026-11-16.md) @ b56907c4339779d2076f0886532833689aec93ec (lines 10, 46–52, 84).
[^recipe]: [tools/rdna/recipes/flashnext-gguf-mtp0.env](https://github.com/opengfx1030/vllm-rdna/blob/b56907c4339779d2076f0886532833689aec93ec/tools/rdna/recipes/flashnext-gguf-mtp0.env) @ b56907c4339779d2076f0886532833689aec93ec.
[^gemv-removed]: `vllm/envs.py` @ `00f106c5` L729-L738 lists `VLLM_RDNA_DENSE_GEMV` as removed; `tools/rdna/recipes/flashnext-gguf-mtp0.env` @ `00f106c5` no longer sets it. <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/vllm/envs.py#L729-L738> ; <https://github.com/opengfx1030/vllm-rdna/blob/00f106c5b035bd6d562267bfee9e4ac020b7055d/tools/rdna/recipes/flashnext-gguf-mtp0.env>
