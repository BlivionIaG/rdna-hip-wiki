---
title: "Lightning Indexer HIP — silicon / HIP contract"
sources:
  - https://github.com/BlivionIaG/vllm/blob/9a344444645f34b3fb7306fde78d68df2c716078/csrc/rocm/indexer_paged_mqa_rdna2.cu
---
# Lightning Indexer HIP — silicon / HIP contract

Live: `indexer_paged_mqa_rdna2.cu`. **Incomplete.** H/D specialization open; topk is still `torch.topk`.[^f-9a344444-indexer-paged-mqa-rdna2-cu]

Decode MQA logits over the paged indexer cache. Inner product in the dump is scalar `__half2float(q)*__half2float(k)` — same “unpack then FMA” shape as MLA, not `fdot2` / `sdot4`. No MFMA.[^f-9a344444-indexer-paged-mqa-rdna2-cu]

Not the DSv4 sparse MLA kernel. Not FA. Occupancy still first.[^f-9a344444-indexer-paged-mqa-rdna2-cu]

## Sources

[^f-9a344444-indexer-paged-mqa-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/indexer_paged_mqa_rdna2.cu` @ [`9a344444`](https://github.com/BlivionIaG/vllm/blob/9a344444645f34b3fb7306fde78d68df2c716078/csrc/rocm/indexer_paged_mqa_rdna2.cu).
