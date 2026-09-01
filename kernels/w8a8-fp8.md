# W8A8-FP8 dense — silicon / HIP contract

Live: `gemm_w8a8_fp8_dense_rdna2.cu` @ `750ca545`. **Incomplete** — commit says GPU correctness pending.

Both A and B are E4M3 bytes. **Still `fdot2`.** Bit-trick both sides to fp16 (`fp8_e4m3_to_fp16_bits` in `qdq_fp8_rdna2.cuh`), then the W8A16-FP8 dequant helper. No LUT. No `sdot4`.

This is **not** the spec W8A8 INT8 path. Extras vs contract: [w8a8.md](w8a8.md). Study: [w8a8-mxfp4.md](w8a8-mxfp4.md) (`sdot4`, i32 through K).

## Why not sdot4

sdot4 wants **integer** i8×i8. These bytes are E4M3. Integer DOT on FP8 bit patterns is wrong. Unpack → fp16 → `fdot2` is the only legal gfx1030 path until a real INT8 W8A8 kernel exists.

## Done-when

- ISA dump: two E4M3 loads, bit-trick cvt, `v_dot2_f32_f16`. No `v_dot4c_i32_i8`.
- GPU correctness on the shapes the launcher actually hits.
