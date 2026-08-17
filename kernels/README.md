# Kernels

Format contracts for the HIP kernels we write. Not engine dispatch.

| Page | Inner loop | Notes |
|---|---|---|
| [w4a16.md](w4a16.md) | `fdot2` after ExLlama nibble dequant | Live on the fork (dense + MoE) |
| [w8a8-mxfp4.md](w8a8-mxfp4.md) | W8A8 = `sdot4` i32-through-K; mxfp4 = unpack then `fdot2` | W8A16-FP8 on the fork is LUT→`fdot2`, not this W8A8 path |

Do not conflate W8A16 / W8A16-FP8 (`fdot2`) with W8A8 (`sdot4`).
| [sage-qk.md](sage-qk.md) | QK = `sdot4`, PV = `fdot2` | Prefill only. Paper + dispatch: [engine/sage-attention.md](../engine/sage-attention.md) |
| [nvfp4.md](nvfp4.md) | E2M1 unpack + E4M3 mul + `fdot2` | Same DOT as mxfp4. Scale is **not** E8M0. Engine: [engine/nvfp4.md](../engine/nvfp4.md) |
| [kv-int8.md](kv-int8.md) | i8 load + cvt + `fdot2` | Fused into `fa_rdna2`. Not Sage. Engine: [engine/kv-int8.md](../engine/kv-int8.md) |
