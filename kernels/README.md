# Kernels

Format contracts for the HIP kernels we write. Not engine dispatch. Feature index vs upstream: [engine/fork-delta.md](../engine/fork-delta.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [w4a16.md](w4a16.md) | nibble dequant → `fdot2` | **Most complete** (dense). MoE incomplete |
| [w8a16.md](w8a16.md) | i8 → fp16 → `fdot2` | Incomplete. MoE in-tree; dense `.cu` absent at tip |
| [w8a16-fp8.md](w8a16-fp8.md) | E4M3 LUT/bit-trick → `fdot2` | Incomplete |
| [w8a8-fp8.md](w8a8-fp8.md) | both sides E4M3 → fp16 → `fdot2` | Incomplete. **Not** `sdot4` |
| [mxfp4.md](mxfp4.md) | E2M1 + E8M0 → `fdot2` | Incomplete |
| [skinny-gemm.md](skinny-gemm.md) | no MFMA on gfx1030 | Incomplete. `waves_per_eu(1,1)` — same FA ticket |
| [mla-sparse.md](mla-sparse.md) | scalar fp32 FMA (later `fdot2`) | Incomplete. Env-gated decode |
| [lightning-indexer.md](lightning-indexer.md) | scalar half FMA | Incomplete |
| [int2.md](int2.md) | i2 unpack → `fdot2` (later `sdot4`) | Spec. Mixed INT2/INT4 MoE = two unpackers, one DOT |
| [w4a4.md](w4a4.md) | i4×i4 `sdot8` | Explore. After W8A8. Not E2M1 |
| [sdot4-explore.md](sdot4-explore.md) | when `sdot4` is legal | Explore: W8A8 INT8, Sage QK, W4A8 |
| [w8a8-mxfp4.md](w8a8-mxfp4.md) | W8A8 INT8 = `sdot4` | **Spec / not added** |
| [sage-qk.md](sage-qk.md) | QK `sdot4`, PV `fdot2` | Spec |
| [nvfp4.md](nvfp4.md) | E2M1 + E4M3 mul → `fdot2` | Spec |
| [kv-int8.md](kv-int8.md) | i8 load + cvt → `fdot2` | Spec |

Do not conflate W8A16 / W8A16-FP8 / W8A8-FP8 (`fdot2`) with spec W8A8 INT8 (`sdot4`).
| [ikantkode-gfx1030.md](ikantkode-gfx1030.md) | sourced overlay (Triton, not HIP) | Steal LLMM1 gate + RMSNorm; do not port GEMV |
| [rocmfpx.md](rocmfpx.md) | codebook10 `perm` → i8 → `sdot4` | Later. llama.cpp GGUF, not a vLLM port |
