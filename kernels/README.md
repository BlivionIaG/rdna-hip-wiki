# Kernels

Format contracts for the HIP kernels we write. Not engine dispatch. Feature index vs upstream: [engine/serving-stack/fork-delta.md](../engine/serving-stack/fork-delta.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [fp16-moe.md](gemm-moe/fp16-moe.md) | native fp16 `fdot2`, fp32 accum | Spec: tiny-M GEMM + grouped prefill |
| [int8-moe.md](gemm-moe/int8-moe.md) | W8A16 `fdot2` / W8A8 `sdot4` | Spec: dual route by expert rows/shape/scales |
| [triton-skinny-gemm.md](gemm-moe/triton-skinny-gemm.md) | stock dispatch baseline | Stock gfx1030 = BLAS; LLMM1/wvSplitK gate excludes gfx10 |
| [w4a16.md](quant/w4a16.md) | nibble dequant → `fdot2` | **Most complete** (dense). MoE GPTQ in-tree + resident skinny opt-in @ `e1315629`; MoE AWQ Not done. Modes: [../engine/overview/coverage.md](../engine/overview/coverage.md) (W4A16 row; no separate engine W4A16 page exists). K_STEP split repair @ `3a0786ea` |
| [w4a8.md](quant/w4a8.md) | int4 W × int8 A → `sdot4` | **Live / opt-in** @ `3a0786ea`; MoE epilogue CAS default / fp32 scratch opt-in @ `30b0bd4e`. Default = W4A16 |
| [w8a16.md](quant/w8a16.md) | i8 → fp16 → `fdot2` | Incomplete. MoE in-tree; dense `.cu` absent at tip |
| [w8a16-fp8.md](quant/w8a16-fp8.md) | E4M3 LUT/bit-trick → `fdot2` | Incomplete |
| [w8a8-fp8.md](quant/w8a8-fp8.md) | both sides E4M3 → fp16 → `fdot2` | Incomplete. **Not** `sdot4` |
| [w8a8.md](quant/w8a8.md) | extras vs contract: live FP8-`fdot2` / W8A16-`fdot2`; INT8 `sdot4` **absent** | **Spec / not on branch** @ `83de31cf` |
| [mxfp4.md](quant/mxfp4.md) | E2M1 + E8M0 → `fdot2` | Incomplete |
| [skinny-gemm.md](gemm-moe/skinny-gemm.md) | no MFMA; gemv_f16 decode @ `3b59ee16`; MoE int4 `fdot2` @ `5c3c0c6f`; resident MoE @ `e1315629` | gfx1030 dense decode = gemv; resident MoE opt-in M=1..4; `(1,1)` leftover open |
| [mla-sparse.md](attention/mla-sparse.md) | scalar fp32 FMA (later `fdot2`) | Incomplete. Env-gated decode |
| [lightning-indexer.md](attention/lightning-indexer.md) | scalar half FMA | Incomplete |
| [int2.md](quant/int2.md) | i2 unpack → `fdot2` (later `sdot4`) | Spec. Mixed INT2/INT4 MoE = two unpackers, one DOT |
| [w4a4.md](quant/w4a4.md) | i4×i4 `sdot8` | Explore. After W8A8. Not E2M1 |
| [sdot4-explore.md](quant/sdot4-explore.md) | when `sdot4` is legal | Explore: W8A8 INT8, Sage QK; W4A8 → [w4a8.md](quant/w4a8.md) Live opt-in |
| [w8a8-mxfp4.md](quant/w8a8-mxfp4.md) | W8A8 INT8 = `sdot4` | **Spec / not added** |
| [sage-qk.md](attention/sage-qk.md) | QK `sdot4`, PV `fdot2` | Spec |
| [nvfp4.md](quant/nvfp4.md) | E2M1 + E4M3 mul → `fdot2` | Spec |
| [kv-int8.md](attention/kv-int8.md) | i8 load + cvt → `fdot2` | Spec |
| [flashkda.md](linear-attn/flashkda.md) | 128×128 `S` → `fdot2` | Later. Not extras; no CUTLASS/SM90 |

Do not conflate W8A16 / W8A16-FP8 / W8A8-FP8 (`fdot2`) with spec W8A8 INT8 (`sdot4`).
| [ikantkode-gfx1030.md](overlays/ikantkode-gfx1030.md) | sourced overlay (Triton, not HIP) | Take LLMM1 gate + RMSNorm; Leave GEMV |
| [triton-jit-aot.md](aot-glue/triton-jit-aot.md) | HIP AOT vs leftover Triton JIT | Take: conv1d / ApplyRotaryEmb / SwigluStep. Leave: FA/EXL3/W4/GDN. RMSNorm AOT ≠ PIECEWISE |
| [rocmfpx.md](quant/rocmfpx.md) | codebook10 `perm` → i8 → `sdot4` | Later. llama.cpp GGUF, not a vLLM port |
| [causal-conv1d.md](linear-attn/causal-conv1d.md) | scalar fp32 FMA (depthwise FIR) | Live HIP @ `7779514b`. UPDATE wire order Leave |
| [gdn-decode.md](linear-attn/gdn-decode.md) | fp32 FMA + shfl K-reduce | Decode HIP Live (`69d2efe`) |
| [gdn-prefill.md](linear-attn/gdn-prefill.md) | kkt scalar FMA; wy/delta_h/o `fdot2` | Prefill HIP Live (`77d6fdf8`). o BV 64 / ~56 KB. All `(2,4)` |
| [layernorm.md](aot-glue/layernorm.md) | scalar fp32 FMA + shfl + tiny LDS | Live AOT HIP (`83de31cf` + gated `71a54552`). Cudagraph-safe. Not FA leftover |
| [qwen4exp-flash-next-hip.md](models/qwen4exp-flash-next-hip.md) | T46 fdot2 glue; T47–T49 HC/QSA/PLE scalar | HC compute-correct @ `8960a3bc`; gates still default off (MoE PIECEWISE capture) |
| [exl3.md](quant/exl3.md) | trellis state → 3-inst → `fdot2` | Later. mul1 + K=1..8 + `exl3_project_rdna2` @ `30b0bd4e`. Occupancy + W4 first |
