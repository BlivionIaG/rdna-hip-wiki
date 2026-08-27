# curvedinf/int8-vllm — silicon Take / Leave

Date: 2026-08-27. [curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm) (Apache-2.0), sibling [curvedinf/int8-aiter](https://github.com/curvedinf/int8-aiter). Engine: leave the fork; no new card. Do not copy tok/s.

Tuned stack is **4× MI100 gfx908** (CDNA1, XGMI): AITER CK `a8w8` GEMMs, AITER unified attention, INT8 per-token-head KV, custom XGMI all-reduce, DFlash2 INT8 draft. Target is Qwen3.8-**27B** GPTQ W8A8 GS128, not Flash-Next.

Their own README already says RDNA2 INT8 is `v_dot4_i32_i8` / DP4A and **CK GEMMs need a replacement**.

## Take

- Policy we already own: `int8_per_token_head` KV **fused** in `fa_rdna2` (i8 load → cvt → `fdot2` QK). Not their AITER UA kernel. Live extras INT8 decode stub (`4cc1fe59`) is still not the contract.
- GDN recurrent state **fp32**. They measured fp16 ceiling and int8+int8-KV corruption. Matches extras GDN HIP (`float32` state tile).
- KLD-gated float exceptions, not “INT8 everything.” Selector / draft ctx-KV staying wide is their produce note, not a V620 DOT.

## Leave

- AITER CK `module_gemm_a8w8` / 299-row gfx908 CSV. **MFMA-class objects will not load on gfx1030.** Replacement is hand HIP `sdot4` on extras, Later (after live W4). ANTIBLEED.
- AITER unified attention (`TILE_SIZE=32` decode, adaptive flash-decoding split-K in `triton_unified_attention.py`). Not an `fa_rdna2` occupancy or tile lever.
- INT8-native **activation pipeline** (keep A as i8 between ops). extras W4 is A=fp16 + dequant-into-`fdot2`. Later W8A8 `sdot4` quantizes A **at the GEMM**, not end-to-end. No `sdot4` on KV (Q is fp16).
- Custom XGMI all-reduce. V620 hop is 88096 PIX / RCCL, not XGMI.
- Their GS128 GPTQ INT8 produce / DFlash2 INT8 draft. DFlash stays Later, fat tile `q>1` first. Produce dest stays W4 / EXL3 `3inst`.
- gfx908 Triton autotunes, `PYTORCH_ROCM_ARCH=gfx908`, `HSA_OVERRIDE`.

## Sources

- README 2026-08-27 (`pushed_at` 2026-08-27T17:15:41Z)
- [../kernels/kv-int8.md](../kernels/kv-int8.md), [../engine/kv-int8.md](../engine/kv-int8.md)
- [../kernels/sdot4-explore.md](../kernels/sdot4-explore.md) if present; W8A8 Later
