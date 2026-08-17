# mxfp4 dense + MoE — silicon / HIP contract

Live: `mxfp4_dot2_common.cuh`, `mxfp4_dot2_dense.cu`, `mxfp4_dot2_moe.cu` @ `290715e6`. **Incomplete** — GPU smoke pending.

No FP4 unit. E2M1 nibble → fp16 (inline bit-trick, no LUT), × E8M0 scale (exponent add), then `V_DOT2_F32_F16` (`dot22_8_f` = 4× `fdot2` / 8 K).

NVFP4 is the same DOT with **E4M3** scales (mul, not exp add): [nvfp4.md](nvfp4.md). Spec W8A8 `sdot4` is unrelated: [w8a8-mxfp4.md](w8a8-mxfp4.md).
