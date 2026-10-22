---
title: "VLLM_RDNA_ALL_HIP — v1 worker/sampler HIP kernel set (dest `30d6e82d`)"
sources:
  - https://github.com/opengfx1030/vllm-rdna/compare/a01f26374e4ce6e8b8072c6bd0ef81ebd2387e0a...30d6e82dcd6e649d0421c2c20fb6813454efbdb1
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/csrc/rocm/v1_topk_topp_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/csrc/rocm/hc_rdna2.cu
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

GDN prefill changes in the same range are on [linear-attn/gdn-prefill.md](../linear-attn/gdn-prefill.md).

## Sources

[^ah-env]: `vllm/envs.py` L337-L348 (type stubs and comments) and L2202-L2207 (defaults `"1"`, PLE conv `"0"`), dest at 30d6e82dcd6e649d0421c2c20fb6813454efbdb1. https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/vllm/envs.py#L337-L348 ; https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/vllm/envs.py#L2202-L2207
[^ah-cmake]: `CMakeLists.txt` L1544-L1558 (HIP source list), dest at 30d6e82dcd6e649d0421c2c20fb6813454efbdb1. https://github.com/opengfx1030/vllm-rdna/blob/30d6e82dcd6e649d0421c2c20fb6813454efbdb1/CMakeLists.txt#L1544-L1558
[^ah-lb]: Compare a01f2637...30d6e82d: `__launch_bounds__(kThreads, 1)` added in `csrc/rocm/v1_topk_topp_rdna2.cu`; `__launch_bounds__(1024, 1)` on `grouped_gemma_rmsnorm_onepass_kernel` and `hc_combine_norm_onepass_kernel` in `csrc/rocm/hc_rdna2.cu`; `__builtin_amdgcn_fdot2` in `csrc/rocm/qsa_rdna2.cu` and `csrc/rocm/mono_qsa_decode_rdna2.cu`. https://github.com/opengfx1030/vllm-rdna/compare/a01f26374e4ce6e8b8072c6bd0ef81ebd2387e0a...30d6e82dcd6e649d0421c2c20fb6813454efbdb1
