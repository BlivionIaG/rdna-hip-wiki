---
title: "rdna_hint_flex (HadamardINT2-FP16 v3.2) on gfx1030 — kernel contract"
updated: 2026-10-29
sources:
  - https://github.com/opengfx1030/vllm-rdna/compare/8015ea485b7f6c0298d8f98ce6e824ce1c444c45...b74028935393d1e21f73f80f65f506c9ea4639af
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/torch_bindings.cpp
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/ops.h
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/rdna_hint_flex.py
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/rdna_hint_flex_moe.py
  - https://github.com/opengfx1030/vllm-rdna/commit/c7bf3c962fe3ea8e4b4396ab3229336afdc57c8c
  - https://github.com/opengfx1030/vllm-rdna/commit/f2ac9370201fd6c095e684287bf357fbaed408dd
  - https://github.com/opengfx1030/vllm-rdna/commit/64b22aeb11c0b32571e4cd0da9bbaceb72e3c48a
  - https://github.com/opengfx1030/vllm-rdna/commit/326af898e270a8359915cf6ed054a57baa2d0a39
  - https://github.com/opengfx1030/vllm-rdna/commit/4dbfd4c8e56dd176a4ea3f97dcdd3f51b534ee22
  - https://github.com/opengfx1030/vllm-rdna/commit/10cf5bbe5293ffe301877bbbd1d8061fca9e5662
  - https://github.com/opengfx1030/vllm-rdna/commit/491b7d50707718f338fd0b5dd07f2f81d343c3cf
---
# rdna_hint_flex (HadamardINT2-FP16 v3.2) on gfx1030 — kernel contract

New at dest `rdna_extras` tip `b7402893` (`b74028935393d1e21f73f80f65f506c9ea4639af`, read 2026-10-29). The format first landed as "HI2" (`c7bf3c96`) and was renamed `rdna_hint_flex` in `f2ac9370`; the source file is `csrc/rocm/rdna_hint_flex_rdna2.cu`.[^cmp][^hdr] It is built only in the gfx1030 HIP list (`VLLM_GPU_ARCHES MATCHES "gfx1030"`).[^cmake] The kernel itself needs fp16 activations ("RDNA2 has no bf16 dot") with K a positive multiple of 64.[^checkact] No tok/s here.

## Storage format (one tensor W[R, K], ragged per row)

| Buffer | Type / shape | Meaning |
|---|---|---|
| `codes` | int32 [sum_r K·nbits_r/32] | codes LSB-first, word aligned: 16 INT2, 8 INT4 or 4 INT8 per word |
| `code_off` / `param_off` | int32 [R+1] | first word / first block of each row |
| `blk_params` | uint8 [2·sum_r K/B_r] | (`lo_u8`, `step_u8`) per block |
| `row_consts` | fp16 [R, 3] | `lo_min`, `lo_scale`, `step_scale` |
| `row_nbits` / `row_block` | uint8 [R] | 2, 4 or 8 bits; block 16, 32 or 64 |
| `had` | uint8 [K/16] | order-16 Walsh–Hadamard flag per 16-column group |
| `col_scale` | fp16 [K] | optional smoothing vector (numel 0 = none) |

So the bit width and block size are chosen **per output row**, not per tensor.[^hdr] Decode is `lo = fp16(fma_f32(lo_u8, lo_scale, lo_min))`, `step = fp16(step_u8 · step_scale)`, `w = fp16(lo + code · step)` with one rounding (`v_fma_f16`). The stored basis is H(W·s); the activation is rotated at run time, x' = H(x / s) on flagged groups, so y = x' · W_stored^T.[^hdr]

Code → fp16 uses the `0x6400` magic: `half(0x6400 | c) = 1024 + c`, minus 1024, valid for codes < 1024.[^magic]

## Kernels (occupancy first)

All kernels below use `kThreads = 256` (8 waves × wave32) where they carry `__launch_bounds__`; none sets `amdgpu_waves_per_eu`.[^const][^lbgrep] LDS numbers are computed from the `__shared__` declarations (before alignment padding); VGPR counts are not in the source and are not given here.

| Kernel | Launch | `__launch_bounds__` | Static LDS | Inner DOT | Notes |
|---|---|---|---|---|---|
| `act_transform_kernel` | 256 threads, one thread per (row, 16-col group) | none | 0 | none (fp32 butterflies) | x' = H16(x/s), one fp16 rounding[^xform][^launchx] |
| `gemv_kernel<MT, SPLIT>` (F16 decode) | 256 threads; RPB = 8/SPLIT rows per WG; grid (R/RPB, M/MT) | 256 | 0 for SPLIT = 1; `float red[8][MT]` (32·MT B) for SPLIT > 1 | `__builtin_amdgcn_fdot2` (v_dot2_f32_f16) | one wave per row (or SPLIT waves per row), lanes stride words, two loads in flight per lane, wave shuffle reduce[^gemv][^worddot] |
| `dequant_kernel` | 256 threads, grid.x = R | 256 | 0 | none (`__hfma2`) | one 16-col group per thread, two 16-B stores[^dequant] |
| `act_quant_kernel` (I8) | 256 threads, one thread per (token, 16-col group) | none | 0 | none | sx_g = fp16(absmax/127), xq ∈ [-127, 127], sumq_g[^i8doc] |
| `gemv_i8_kernel<MT>` (I8 decode) | 256 threads, 8 rows per WG | 256 | 0 | 4 × `__builtin_amdgcn_sdot4` per 16-col group | INT8 codes enter as code − 128, S gets 128·sumq back exactly[^i8doc][^i8dot][^gemvi8] |
| `moe_act_transform_kernel` / `moe_gemv_kernel<REDUCE>` (F16 routed) | 256 threads, 8 rows per WG, grid.y = pair or token | gemv: 256 | 0 | `fdot2` | experts stacked row-wise, per-expert `col_scale` and `had`; REDUCE sums top-k into the token row[^moe][^moegemv] |
| `moe_act_quant_kernel` / `moe_gemv_i8_kernel<REDUCE>` (I8 routed) | as above | gemv: 256 | 0 | `sdot4` | same pair layout as F16 routed[^moegemvi8] |
| `gemm2_kernel<GM, GN, MOE>` (fused decode-in-GEMM prefill) | 256 threads (16 × 16), GM × GN tile, BK = 64 | 256 | `xs[GM·72]` + `ws[GN·72]` halves (pad 8) + `RowHdrS hdr[GN]` (28 B each): 40448 B for 128 × 128, 31232 B for the MoE 64 × 128 | `fdot2` on 16-byte LDS reads | next K step's words are prefetched into registers while the current step is computed from LDS; decode happens on the LDS write[^gemm2doc][^gemm2lds][^gemm2dot] |
| `act_quant64_kernel` (I8 prefill) | 128 threads, one thread per (token, 64-col group) | none | 0 | none | sx per 64 columns (float), sq16 per 16 columns[^q64][^launchq64] |
| `i8gemm_kernel<GM, GN>` (I8 prefill) | 256 threads (16 × 16), BK = 64 | 256 | `xs[GM·80]` + `ws[GN·80]` int8 (16-B pad) + `wu`, `wl` [GN][4] + `sxs[GM]` + `sqs[GM][4]` + `hdr[GN]`: 30720 B for 128 × 128, 24320 B for 64 × 128 | 4 × `sdot4` per 16-col chunk | integer S1 per chunk, float lo and scale terms[^i8gemm][^i8gemmdot] |

## Dispatch gates (as stated in source)

- **Composite linear op** `rdna_hint_flex_linear_rdna2(…, decode_max_m, mode)`: mode 0 (F16) runs the F16 GEMV for M ≤ `decode_max_m`, else dequant + rocBLAS GEMM; mode 1 (I8) runs the I8 path for M ≤ `decode_max_m`, larger M as mode 0; mode 2 runs the I8 path for every M. The choice is made at execution time from M.[^linear][^hdr]
- **Python defaults:** `COMPUTE_MODES = {"f16": 0, "i8": 1, "i8_all": 2}`, default `"f16"`, overridable by `VLLM_RDNA_HINT_FLEX_COMPUTE_MODE`; `DECODE_MAX_M = 48`. The config only engages on ROCm with `on_gfx10x()` and the op present.[^pymode][^pyenv]
- **GEMV split:** `RDNA_HINT_FLEX_GEMV_SPLIT` (1, 2, 4, 8) forces waves per row; otherwise 1 for R ≥ 8192, 2 for R ≥ 4096, else 4. MT is 1, 2, 4 or 8 from M.[^split][^mt]
- **Fused prefill GEMM is opt-in:** `RDNA_HINT_FLEX_PREFILL` > 0 selects `gemm2_kernel<128, 128>`; the default (0) is dequant + `at::mm_out` (rocBLAS).[^prefill][^linear]
- **I8 prefill is op-only:** `act_quant64` / `i8gemm` (tile 1 → 64 × 128, else 128 × 128) are registered ops, but `rdna_hint_flex_linear_rdna2` does not call them; its I8 branch uses the per-16-column `act_quant` + `gemv_i8`.[^i8tile][^linear] The commit that added them calls them opt-in.[^c4dbf]
- **Routed experts:** `rdna_hint_flex_moe_rdna2` uses I8 when mode 2, or mode 1 with T ≤ `decode_max_m`; otherwise the F16 transform + GEMV pair.[^moehost] In Python, batches with more than `GROUPED_MIN_TOKENS` tokens in F16 mode take the grouped path (`moe_align_block_size` block 64 → `gemm2_kernel<64, 128, true>`); the default is 16 and `RDNA_HINT_FLEX_MOE_GROUPED_MIN` overrides it.[^grouped][^moeblock]

## Take / Leave

- **Take:** per-row mixed 2/4/8-bit codes with per-block (lo, step) bytes feed both packed DOTs: `fdot2` after an exact fp16 decode, or `sdot4` directly on spread codes with a `sumq` correction. This is a third pattern next to W4A16 (`fdot2`) and W4A8 (`sdot4`): [w4a16.md](w4a16.md), [w4a8.md](w4a8.md).[^worddot][^i8dot]
- **Leave (for now):** the fused `gemm2` prefill and the I8 prefill GEMM are not the default prefill path.[^prefill][^linear]
- **Open:** none of the kernels pins `amdgpu_waves_per_eu`, and the source gives no VGPR or scratch numbers, so occupancy for `gemm2_kernel<128,128>` (~40 KiB static LDS, one WG per CU by LDS on a 64 KiB CU budget) is (unsourced) until someone reads the code object.

## Sources

[^cmp]: Compare [`8015ea48...b7402893`](https://github.com/opengfx1030/vllm-rdna/compare/8015ea485b7f6c0298d8f98ce6e824ce1c444c45...b74028935393d1e21f73f80f65f506c9ea4639af) (59 commits); commits [`c7bf3c96`](https://github.com/opengfx1030/vllm-rdna/commit/c7bf3c962fe3ea8e4b4396ab3229336afdc57c8c) "feat(rdna2): HI2 (HadamardINT2-FP16 v3.2) weight kernels for gfx1030" and [`f2ac9370`](https://github.com/opengfx1030/vllm-rdna/commit/f2ac9370201fd6c095e684287bf357fbaed408dd) "refactor(quant): rename the format to rdna_hint_flex".
[^hdr]: [rdna_hint_flex_rdna2.cu:4–40](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L4-L40) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^cmake]: [CMakeLists.txt:1520, 1532](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/CMakeLists.txt#L1520-L1532) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^checkact]: [rdna_hint_flex_rdna2.cu:1464–1471](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1464-L1471) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^magic]: [rdna_hint_flex_rdna2.cu:85–102](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L85-L102) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^const]: [rdna_hint_flex_rdna2.cu:61–64](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L61-L64) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^lbgrep]: `__launch_bounds__(kThreads)` at [rdna_hint_flex_rdna2.cu:230](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L230), [346](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L346), [527](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L527), [680](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L680), [740](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L740), [1024](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1024), [1141](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1141); no `amdgpu_waves_per_eu` in the file @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^xform]: [rdna_hint_flex_rdna2.cu:108–152](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L108-L152) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^launchx]: [rdna_hint_flex_rdna2.cu:1485–1493](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1485-L1493) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gemv]: [rdna_hint_flex_rdna2.cu:226–308](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L226-L308) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^worddot]: [rdna_hint_flex_rdna2.cu:157–214](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L157-L214) (`__builtin_amdgcn_fdot2` at L187) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^dequant]: [rdna_hint_flex_rdna2.cu:310–383](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L310-L383) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^i8doc]: [rdna_hint_flex_rdna2.cu:385–396](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L385-L396) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^i8dot]: [rdna_hint_flex_rdna2.cu:455–524](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L455-L524) (`__builtin_amdgcn_sdot4` at L513–L516) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gemvi8]: [rdna_hint_flex_rdna2.cu:526–577](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L526-L577) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moe]: [rdna_hint_flex_rdna2.cu:579–641](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L579-L641) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moegemv]: [rdna_hint_flex_rdna2.cu:677–720](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L677-L720) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^moegemvi8]: [rdna_hint_flex_rdna2.cu:1023–1063](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1023-L1063) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gemm2doc]: [rdna_hint_flex_rdna2.cu:722–738](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L722-L738) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gemm2lds]: [rdna_hint_flex_rdna2.cu:66–71, 739–762](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L739-L762) (`RowHdrS` = 3 float + 4 int32; `kGemmPad = 8`, `kGemm2BK = 64`) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^gemm2dot]: [rdna_hint_flex_rdna2.cu:919](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L919) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^q64]: [rdna_hint_flex_rdna2.cu:1065–1134](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1065-L1134) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^launchq64]: [rdna_hint_flex_rdna2.cu:1772–1779](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1772-L1779) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^i8gemm]: [rdna_hint_flex_rdna2.cu:1136–1164](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1136-L1164) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^i8gemmdot]: [rdna_hint_flex_rdna2.cu:1319–1322](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1319-L1322) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^linear]: [rdna_hint_flex_rdna2.cu:2120–2169](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L2120-L2169) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^pymode]: [rdna_hint_flex.py:55–80](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/rdna_hint_flex.py#L55-L80) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^pyenv]: [rdna_hint_flex.py:124](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/rdna_hint_flex.py#L124) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^split]: [rdna_hint_flex_rdna2.cu:1514–1524](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1514-L1524) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^mt]: [rdna_hint_flex_rdna2.cu:1545–1559](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1545-L1559) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^prefill]: [rdna_hint_flex_rdna2.cu:1696–1712](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1696-L1712) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^i8tile]: [rdna_hint_flex_rdna2.cu:1800–1808](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1800-L1808) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^c4dbf]: Commit [`4dbfd4c8`](https://github.com/opengfx1030/vllm-rdna/commit/4dbfd4c8e56dd176a4ea3f97dcdd3f51b534ee22) "feat(rdna2): opt-in I8 prefill ops with per-64-column activation scales"; followed by [`14a4de27`](https://github.com/opengfx1030/vllm-rdna/commit/14a4de270867664136fbacf9de1b956435a8203f) "docs(rdna2): park I8 prefill with the activation-group measurements".
[^moehost]: [rdna_hint_flex_rdna2.cu:1887–1960](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1887-L1960) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^grouped]: [rdna_hint_flex_moe.py:48–52, 270–272](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/vllm/model_executor/layers/quantization/rdna_hint_flex_moe.py#L48-L52) @ b74028935393d1e21f73f80f65f506c9ea4639af; env override added in [`491b7d50`](https://github.com/opengfx1030/vllm-rdna/commit/491b7d50707718f338fd0b5dd07f2f81d343c3cf).
[^moeblock]: [rdna_hint_flex_rdna2.cu:1810–1832](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_hint_flex_rdna2.cu#L1810-L1832) @ b74028935393d1e21f73f80f65f506c9ea4639af.
