---
title: "Lightning Indexer HIP — silicon / HIP contract"
sources:
  - https://github.com/BlivionIaG/vllm/blob/9a344444645f34b3fb7306fde78d68df2c716078/csrc/rocm/indexer_paged_mqa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu
---
# Lightning Indexer HIP — silicon / HIP contract

## Current state at dest `rdna_extras` @ `1dc3bac4` (checked 2026-10-11)

| Item | State at `1dc3bac4` |
|---|---|
| File | `indexer_paged_mqa_rdna2.cu` is **not** in dest. The indexer kernels are `csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu`, built in the gfx1030 CMake list.[^idx-tip-file][^idx-tip-cmake] |
| Ops | `paged_mqa_logits_decode_rdna2` (decode, paged) and `mqa_logits_prefill_rdna2` (ragged prefill, fused and tiled).[^idx-tip-bind] |
| Inner product | `__builtin_amdgcn_fdot2` in both the decode and the prefill kernel, not the scalar `half2float` FMA described below.[^idx-tip-file] |
| Selection | Used when the op is registered (`rdna_ops.has_paged_mqa_logits()` / `has_mqa_logits_prefill()`); no env gate found.[^idx-tip-sel] |
| Top-k | Not re-traced at this tip *(unsourced)*. |

Everything below describes the BlivionIaG archive file at `9a344444` and is historical.


Historical (BlivionIaG `9a344444`): `indexer_paged_mqa_rdna2.cu`. **Incomplete.** H/D specialization open; topk is still `torch.topk`.[^f-9a344444-indexer-paged-mqa-rdna2-cu]

Decode MQA logits over the paged indexer cache. Inner product in the dump is scalar `__half2float(q)*__half2float(k)` — same “unpack then FMA” shape as MLA, not `fdot2` / `sdot4`. No MFMA.[^f-9a344444-indexer-paged-mqa-rdna2-cu]

Not the DSv4 sparse MLA kernel. Not FA. Occupancy still first.[^f-9a344444-indexer-paged-mqa-rdna2-cu]

## Sources

[^f-9a344444-indexer-paged-mqa-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/indexer_paged_mqa_rdna2.cu` @ [`9a344444`](https://github.com/BlivionIaG/vllm/blob/9a344444645f34b3fb7306fde78d68df2c716078/csrc/rocm/indexer_paged_mqa_rdna2.cu).
[^idx-tip-file]: `csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu` (`paged_mqa_logits_decode_kernel` L65 with `fdot2` at L179; `mqa_logits_prefill_kernel` L307 with `fdot2` at L415), dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna/dsv4/indexer_mqa_logits.cu
[^idx-tip-cmake]: gfx1030 append list in `CMakeLists.txt` (DeepSeek-V4 RDNA kernels), dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/CMakeLists.txt#L1544-L1549
[^idx-tip-bind]: `paged_mqa_logits_decode_rdna2` / `mqa_logits_prefill_rdna2` in `csrc/rocm/torch_bindings.cpp`, dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/torch_bindings.cpp#L349-L362
[^idx-tip-sel]: `has_paged_mqa_logits` / `has_mqa_logits_prefill` in `vllm/models/deepseek_v4/amd/rdna/ops.py` and their use in `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`, dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/models/deepseek_v4/amd/rdna/ops.py#L28-L41 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py#L780-L800
