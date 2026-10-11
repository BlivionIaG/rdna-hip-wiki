# Lightning Indexer HIP — silicon / HIP contract

Live: `indexer_paged_mqa_rdna2.cu`. **Incomplete.** H/D specialization open; topk is still `torch.topk`.

Decode MQA logits over the paged indexer cache. Inner product in the dump is scalar `__half2float(q)*__half2float(k)` — same “unpack then FMA” shape as MLA, not `fdot2` / `sdot4`. No MFMA.

Not the DSv4 sparse MLA kernel. Not FA. Occupancy still first.
