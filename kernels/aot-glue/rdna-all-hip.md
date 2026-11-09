---
title: "VLLM_RDNA_ALL_HIP — v1 worker/sampler HIP kernel set (dest `30d6e82d`)"
sources:
  - https://github.com/opengfx1030/vllm-rdna/compare/a01f26374e4ce6e8b8072c6bd0ef81ebd2387e0a...30d6e82dcd6e649d0421c2c20fb6813454efbdb1
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/csrc/rocm/v1_topk_topp_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/csrc/rocm/hc_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/b8efd1bf033502abe578034f321261686c92db44
  - https://github.com/opengfx1030/vllm-rdna/commit/d6267ee3b09ef7d1ccbdfb9bc0d39e66809e4bd3
  - https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/csrc/rocm/v1_worker_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/csrc/rocm/v1_input_batch_rdna2.cu
---
# `VLLM_RDNA_ALL_HIP`: the RDNA2 HIP kernel set (dest `30d6e82d`)

Back to [AOT glue](README.md).

At dest `rdna_extras` @ `30d6e82d` (checked 2026-10-22), more of the Triton leftovers are AOT HIP on gfx1030.

| Item | State |
|---|---|
| Umbrella gate | `VLLM_RDNA_ALL_HIP` defaults **on** (`"1"`). It covers the v1 worker/sampler kernels, top-k/top-p masking, QSA metadata, MTP conv1d plus GDN spec-verify, unquantized MoE and the vision pos-embed. Setting it to 0 falls back to Triton.[^ah-env] |
| Sibling gates | `VLLM_RDNA_HC_PREFILL_HIP` and `VLLM_RDNA_QSA_HIP` default on; `VLLM_RDNA_PLE_CONV_HIP` stays off.[^ah-env] |
| Build list | Ten new `csrc/rocm` sources join the HIP build: `moe_unquantized_rdna2`, `v1_worker`, `v1_input_batch`, `v1_spec_decode`, `v1_mamba`, `v1_rope_struct`, `v1_sample`, `v1_topk_topp`, `qsa_metadata` and `vit_pos_embed` (all `_rdna2.cu`).[^ah-cmake] |
| Occupancy notes | `v1_topk_topp_rdna2` uses `__launch_bounds__(kThreads, 1)`; the rewritten HC one-pass kernels use `__launch_bounds__(1024, 1)`.[^ah-lb] |
| ISA | The QSA paths in `qsa_rdna2.cu` and `mono_qsa_decode_rdna2.cu` use `__builtin_amdgcn_fdot2` (packed fp16 dot), not WMMA.[^ah-lb] |

## Update at dest tip `c4a05415` (checked 2026-11-02)

The v0.33 merge left two worker paths falling back to Triton or reading out of bounds; both are fixed in HIP at this tip.

| Item | State at `c4a05415` |
|---|---|
| Slot-mapping bound | `compute_slot_mappings_kernel` writes `PAD_SLOT_ID` when the block index falls outside the request's block-table row (`block_index < block_table_stride`). Recurrent and circular cache rows can be shorter than the sequence, and without the bound a long prefill read the next request's row.[^ah-c4a0-b8ef][^ah-c4a0-worker] |
| SWA masking on HIP | `gather_block_tables_kernel` now nulls entries wholly below the sliding window of the request's first scheduled token, matching `_gather_block_tables_kernel`. Before this commit the HIP path fell back to Triton whenever window masking was needed, so the MTP/spec-decode path now stays on HIP.[^ah-c4a0-b8ef][^ah-c4a0-worker] |
| Draft scatter on HIP | `post_update_kernel` copies `draft_tokens[req_id]` into `draft_tokens_out[req_state_idx]` for every live row; masked rows (`idx_mapping < 0`) still return before any write. Before, any step with drafts fell back to the Triton kernel.[^ah-c4a0-d626][^ah-c4a0-ib] |
| Launch shape | Unchanged: one block per request, no LDS tile, no DOT, no new `__launch_bounds__`.[^ah-c4a0-worker][^ah-c4a0-ib] |

GDN prefill changes in the same range are on [linear-attn/gdn-prefill.md](../linear-attn/gdn-prefill.md).

## Sources

[^ah-env]: `vllm/envs.py` L337-L348 (type stubs and comments) and L2202-L2207 (defaults `"1"`, PLE conv `"0"`), dest at 30d6e82dcd6e649d0421c2c20fb6813454efbdb1. https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/vllm/envs.py#L337-L348 ; https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/vllm/envs.py#L2202-L2207
[^ah-cmake]: `CMakeLists.txt` L1544-L1558 (HIP source list), dest at 30d6e82dcd6e649d0421c2c20fb6813454efbdb1. https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/CMakeLists.txt#L1544-L1558
[^ah-lb]: Compare a01f2637...30d6e82d: `__launch_bounds__(kThreads, 1)` added in `csrc/rocm/v1_topk_topp_rdna2.cu`; `__launch_bounds__(1024, 1)` on `grouped_gemma_rmsnorm_onepass_kernel` and `hc_combine_norm_onepass_kernel` in `csrc/rocm/hc_rdna2.cu`; `__builtin_amdgcn_fdot2` in `csrc/rocm/qsa_rdna2.cu` and `csrc/rocm/mono_qsa_decode_rdna2.cu`. https://github.com/opengfx1030/vllm-rdna/compare/a01f26374e4ce6e8b8072c6bd0ef81ebd2387e0a...30d6e82dcd6e649d0421c2c20fb6813454efbdb1
[^ah-c4a0-b8ef]: opengfx1030/vllm-rdna commit [`b8efd1bf`](https://github.com/opengfx1030/vllm-rdna/commit/b8efd1bf033502abe578034f321261686c92db44) "fix(rdna2): bound slot mappings, mask SWA gathers in HIP" (message: regressions from the v0.33 merge; 70 worker tests pass on V620); an ancestor of `c4a05415cbda030c784bdc91c7d32e55c287a828`.
[^ah-c4a0-worker]: `csrc/rocm/v1_worker_rdna2.cu` L38 and L65-L77 (window nulling), L158-L171 (`valid` bound, `PAD_SLOT_ID`) @ c4a05415cbda030c784bdc91c7d32e55c287a828. https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/csrc/rocm/v1_worker_rdna2.cu#L38-L171
[^ah-c4a0-d626]: opengfx1030/vllm-rdna commit [`d6267ee3`](https://github.com/opengfx1030/vllm-rdna/commit/d6267ee3b09ef7d1ccbdfb9bc0d39e66809e4bd3) "perf(rdna2): scatter drafts in the HIP post_update kernel"; an ancestor of `c4a05415cbda030c784bdc91c7d32e55c287a828`.
[^ah-c4a0-ib]: `csrc/rocm/v1_input_batch_rdna2.cu` L214-L219 (draft scatter) @ c4a05415cbda030c784bdc91c7d32e55c287a828. https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/csrc/rocm/v1_input_batch_rdna2.cu#L214-L219
