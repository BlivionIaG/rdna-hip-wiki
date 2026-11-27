---
title: "Q5_K dense VDR-4 mmvq dot product (llama-cpp-rdna-boosts r368) — gfx1030 read"
sources:
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/d829586b6d404d2549e980eadeedf004f476f079
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/d829586b6d404d2549e980eadeedf004f476f079/release.json
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/d829586b6d404d2549e980eadeedf004f476f079/patches/README.md#L788-L791
  - https://github.com/stew675/llama-cpp-rdna-boosts/tree/d829586b6d404d2549e980eadeedf004f476f079/archive/work/issue-374-kquant
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vendors/hip.h#L241-L256
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L1436-L1519
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L1327-L1376
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L649-L718
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L8-L14
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/common.cuh#L915-L920
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq-vecdot.cuh#L47-L51
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq.cu#L60-L121
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq.cu#L628-L680
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq.cu#L3736-L3740
  - https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq-rdna-params.cuh#L130-L200
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/mmvq.cuh#L1-L41
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/mmvq.cuh#L115-L122
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/vecdotq.cuh#L381
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/vecdotq.cuh#L1279-L1325
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/ggml-common.h#L1-L9
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/ggml-common.h#L1125-L1133
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/cuda_compat.h#L26-L38
  - https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/quantization/linear.py#L35-L47
  - https://github.com/opengfx1030/vllm-rdna/blob/71299d56b2526dc4df8c74d261d0c57c39865dd2/docs/rdna2/gguf-flashnext-2026-11-16.md#L25-L36
  - https://github.com/opengfx1030/vllm-rdna/tree/71299d56b2526dc4df8c74d261d0c57c39865dd2/csrc
  - https://github.com/BlivionIaG/rdna-hip-wiki/tree/e143080da80095221b76da995170eda3d6842261/kernels/quant/probes/q5k-dense-vdr4
  - https://github.com/BlivionIaG/rdna-hip-wiki/blob/21b7ac68e161e2bab13a088d8263831cc9aecffd/silicon/occupancy/vgpr-occupancy.md
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---
# Q5_K dense VDR-4 mmvq dot product (r368) — gfx1030 read

Date: 2026-11-27 (idle HIP/ISA pass). This is an occupancy and ISA page, not a perf page. It has no tok/s of our own. The only timing quoted is upstream's, measured on **gfx1100**. This page does **not** change extras HIP and does not open a ticket. Back to the [quant index](README.md). Neighbours: [sdot4-explore.md](sdot4-explore.md) (when `sdot4` is legal), [rocmfpx.md](rocmfpx.md) (the other llama.cpp GGUF `sdot4` read), [silicon/occupancy/vgpr-occupancy.md](../../silicon/occupancy/vgpr-occupancy.md) (the ladder used below).

## What upstream landed

stew675/llama-cpp-rdna-boosts r368 (`d829586b`, issue #374) adds `vec_dot_q5_K_q8_1_vdr4_dense`. The commit describes it as the Q4_K dense kernel plus the `qh` fifth bit. Merging the MoE VDR=4 kernel's two half-chains per sub-block is described as "a pure integer reordering (same scale, same d8, exact)". The kernel is in block 07, the 2-rows-per-block verify rule for Q5_K and IQ4_XS is in block 08, and the host VDR and dispatch case is in block 10.[^up-commit][^up-readme] The repo ships patches. Release `v18-919794500-r368` names tip `af7a57ba` and tree `bc7898a2`.[^up-release] That tip exists as stew675/llama.cpp `af7a57ba`, and its tree is `bc7898a2`. Every source line below is cited there. On the box, applying blocks 0000–0300 to ggml-org/llama.cpp `91979450` gave the same tree hash.

Upstream's own headline (gfx1100, 27B UD-Q4_K_M, m = 4096, k = 14336): Q5_K went from 33.01 to 22.23 µs at n = 1. The commit says single-stream 27B decode "is at the DRAM roof" (tg128 +0.85 %).[^up-commit] These are upstream's numbers, not ours.

## (1) Reachable on gfx1030? No: it is gated to RDNA3_0, and RDNA4 behind a flag

| Gate | Line(s) at `af7a57ba` | gfx1030 result |
|---|---|---|
| Arch macros | `RDNA3_0` needs `RDNA3`, which needs `__GFX11__`. `__gfx1030__`…`__gfx1037__` define `RDNA2` only.[^lt-hip] | `RDNA3_0` is **undefined** |
| Device function | `#if defined(RDNA3_0) \|\| (defined(RDNA4) && defined(GGML_HIP_RDNA4_Q5K_DENSE_VDR4))` → `#define GGML_CUDA_Q5K_DENSE_VDR4`. The function body is inside `#if defined(GGML_CUDA_Q5K_DENSE_VDR4)`.[^lt-dense] | **Not compiled** for gfx1030 |
| Dot selection | `get_vec_dot_q_cuda`: dense Q5_K → `vec_dot_q5_K_q8_1_vdr4_dense` only under that macro. Otherwise `vec_dot_q5_K_q8_1` is used. MoE takes `vec_dot_q5_K_q8_1_vdr4` either way.[^lt-vecdot-sel] | Dense Q5_K = stock VDR 2 kernel |
| Host VDR | `get_vdr_mmvq_dense`: `q5k_dense_vdr4 = table_id == MMVQ_PARAMETERS_RDNA3_0` (`\|\| RDNA4` only with the flag). Otherwise `get_vdr_mmvq(type)`.[^lt-tables] | VDR 2 (`VDR_Q5_K_Q8_1_MMVQ 2`)[^lt-vmmq] |
| Table id | Device and host both map `RDNA2` → `MMVQ_PARAMETERS_NAVI21`.[^lt-tables] | NAVI21 table |
| Verify-band rows (block 08) | `calc_rows_per_block_weight`: the Q5_K / IQ4_XS → `GGML_MMVQ_RDNA3_Q5K_WEIGHT_RPB` / `…IQ4XS…` (2) rules test `table_id == MMVQ_PARAMETERS_RDNA3_0`. Every other table falls through to `calc_rows_per_block`.[^lt-rpb] | New rule **not taken**. NAVI21 already gives **2 rows** at n = 2..4 and **4 rows** at n = 5..8 for every type, with 4 / 2 / 1 waves at n = 1 / 2..4 / 5..8.[^lt-navi21] |
| Verify-band fusion arch | `verify_band_arch = GGML_CUDA_CC_IS_RDNA4(cc) \|\| GGML_CUDA_CC_IS_RDNA3_0(cc)`[^lt-fuse] | Not RDNA2 |

So at r368 both halves of the change are off on gfx1030: the dense kernel and the Q5_K / IQ4_XS row rule. The row half is also moot there, because the Navi21 table's row count does not depend on type.

## (2) Inner-dot ISA on gfx1030

- `ggml_cuda_dp4a` on HIP with `RDNA2` defined is `__builtin_amdgcn_sdot4(a, b, c, false)`. RDNA3/RDNA4 use `__builtin_amdgcn_sudot4` instead.[^lt-dp4a]
- Both Q5_K kernels build each 8-bit operand as `low nibble | (qh bit << 4)`, then issue `ggml_cuda_dp4a` against q8_1. They run a second "ones" chain (`0x01010101`) for the min term. The scale multiply is `ggml_cuda_mul_small` = `__mul24` on HIP.[^lt-dense][^lt-vmmq][^lt-mul24]
- **Measured (probe compile, below):** on gfx1030 the compiler emits `v_dot4c_i32_i8_e32`, the VOP2 accumulate form (opcode 13 in the RDNA2 ISA guide), for every `sdot4`. It emits **zero** VOP3P `v_dot4_i32_i8`. The scale multiply is `v_mul_i32_i24`.[^probe][^isa] The inner dot is therefore `sdot4`, i8×i8 → i32. The 5-bit values (0..31) fit in signed i8.
- Our GGUF path has the same mapping. The plugin's `__dp4a` is `__builtin_amdgcn_sdot4` whenever the builtin exists.[^pl-dp4a]

## (3) Occupancy: dense VDR-4 vs stock Q5_K VDR-2 on gfx1030

**What was measured:** a probe compile on the box, not a hardware counter and not the upstream kernel. Toolchain: Debian `hipcc` 5.7.1 (HIP 5.7.31921, Debian clang 17.0.6), `--offload-arch=gfx1030 -O3`, `-Rpass-analysis=kernel-resource-usage`. That is **not** the dest ROCm 7.14 compiler, so VGPR counts can move by a rung on 7.14 (unsourced expectation).

`q5k_src.cuh` holds verbatim copies of both Q5_K dot products from tree `bc7898a2`, with the RDNA2 `dp4a` branch as a shim. `q5k_probe.cu` wraps them in an mmvq-shaped loop: `kbx`/`kqs` striding as upstream, a rows × ncols accumulator array, and an LDS cross-wave reduce when nwarps > 1. `-DGGML_CUDA_Q5K_DENSE_VDR4` forces the dense function to compile on gfx1030 for the probe only. Sources, script and raw remarks are committed at `e143080d`.[^probe]

Ladder: wave32, 1024 VGPR/SIMD, granule 16, cap 16 waves → `min(16, floor(1024 / alignTo(VGPR, 16)))`.[^wiki-vgpr] Compiler occupancy matched the ladder in every row. There was no scratch in any row.

| Shape (nwarps × rows, ncols_dst) | Why this shape | Stock VDR2 VGPR → waves | Dense VDR4 VGPR → waves |
|---|---|---|---|
| 1 × 1, n = 1 | plugin `mul_mat_vec_q` launch (one wave32 per row)[^pl-mmvq] | 39 → 16 | 50 → 16 |
| 4 × 1, n = 1 | NAVI21 n = 1 (4 waves)[^lt-navi21] | 40 → 16 | 52 → 16 |
| 2 × 2, n = 2 | NAVI21 n = 2..4 | 60 → 16 | 60 → 16 |
| 2 × 2, n = 4 | NAVI21 n = 2..4 | 70 → 12 | 76 → 12 |
| 1 × 4, n = 8 | NAVI21 n = 5..8 | 105 → **9** | 125 → **8** |

One-row reference shapes were also compiled: 2 × 1 at n = 2 / 4 and 1 × 1 at n = 8. All stayed at 16 waves both ways (49–64 VGPR). The dense kernel adds about 3–12 VGPR at most shapes. The one exception is the widest Navi21 verify shape, where it adds **20 VGPR and drops one rung (9 → 8 waves)**. It never spills. The dot itself uses no LDS. LDS in the probe is only the reduce buffer and is identical for both kernels.[^probe]

Instruction mix, from static counts in the 1 × 1 probe kernel (one loop body covers 16 K-values per thread for stock and 32 for dense):[^probe]

| per 32 K-values per (row, col) | stock VDR2 (×2 bodies) | dense VDR4 |
|---|---|---|
| `v_dot4c_i32_i8` | 16 | 16 |
| `v_mul_i32_i24` (scale mult) | 8 | 2 |
| `v_cvt_f32_i32` | 8 | 4 |

The `sdot4` count per element does not change. The win is fewer integer-to-float conversions and scale multiplies per element, plus half as many `kqs` iterations. That matches upstream's own explanation: "only how many times the per-(row, column) float work and the 16-element unpacking happen".[^lt-dense] For a single column (`n = 1`), that work is a small share of a DRAM-bound GEMV. Upstream reports Q4_K/Q6_K controls inside 0.5 % and calls tg128 +0.85 %.[^up-commit] Whether gfx1030 is issue-bound enough at n = 2..8 for this to matter was **not measured**.

## (4) Verdict for gfx1030

| Item | Verdict | Why |
|---|---|---|
| Upstream kernel / build flag / gfx1100 µs | **Leave** | Gated RDNA3_0 (+RDNA4 opt-in), not reachable on gfx1030. Their numbers are gfx1100.[^lt-dense][^up-commit] |
| Q5_K / IQ4_XS 2-rows verify rule (block 08) | **Leave / Dead for gfx1030** | RDNA3_0-only. The NAVI21 table already does type-agnostic 2 / 4 rows.[^lt-rpb][^lt-navi21] |
| Idea: one integer chain per sub-block (merge two 16-value halves before scale/convert), VDR 4 for dense Q5_K | **Take, Later** | It is exact integer reordering upstream[^lt-dense], it is still `sdot4` on gfx1030, and it costs 0–1 occupancy rung (probe). It belongs in our GGUF GEMV, which is **not in `rdna_extras` HIP**. `rdna_extras` has no GGUF kernels under `csrc/`, and GGUF GEMMs run through `opengfx1030/vllm-gguf-plugin` (`rdna_extras` branch).[^ex-csrc][^ex-gguf] |
| Where it would land | plugin, not extras | The plugin's Q5_K GEMV is the llama.cpp b2899-era `mul_mat_vec_q`: VDR 2, one wave32 per row, one column per block. It is used only when `x.shape[0] <= 2` (rows > 5120) or `<= 6` (else) for k-quants.[^pl-mmvq][^pl-vdr][^pl-common][^pl-warp][^pl-linear] With one column per block, the multi-column amortisation that drove the gfx1100 win does not exist there. Gain is unmeasured, so this stays Later until someone profiles a Q5_K-heavy GGUF on a V620. |

**Does any finding change extras HIP behaviour or an existing UNC card? No.** The upstream change is unreachable on gfx1030. `rdna_extras` has no k-quant HIP GEMV to change. The only possible vehicle is the out-of-tree GGUF plugin fork, and nothing here is measured enough to justify a card.

## Sources

[^up-commit]: stew675/llama-cpp-rdna-boosts commit [`d829586b`](https://github.com/stew675/llama-cpp-rdna-boosts/commit/d829586b6d404d2549e980eadeedf004f476f079) (2026-11-27 03:30 Paris), "r368: issue #374 -- the dense Q5_K VDR-4 mmvq dot product (blocks 07/08/10), and RDNA3_0's Q6_K prefill band". Message quotes: kernel description, block split, gfx1100 µs, "single-stream 27B decode is at the DRAM roof". Record dir: [`archive/work/issue-374-kquant/`](https://github.com/stew675/llama-cpp-rdna-boosts/tree/d829586b6d404d2549e980eadeedf004f476f079/archive/work/issue-374-kquant).
[^up-readme]: [`patches/README.md` L788–L791 @ `d829586b`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/d829586b6d404d2549e980eadeedf004f476f079/patches/README.md#L788-L791): block 07 "+ r368: the dense Q5_K VDR-4 dot product", block 08 "+ r368: Q5_K and IQ4_XS take 2 rows per block at the RDNA3 verify widths", block 10 "+ r368: the host VDR + dispatch case".
[^up-release]: [`release.json` @ `d829586b`](https://github.com/stew675/llama-cpp-rdna-boosts/blob/d829586b6d404d2549e980eadeedf004f476f079/release.json): base `919794500`, tip `af7a57ba8700f499fa6ca01d47470208deae8718`, tree `bc7898a2cfa2f5268160579a2933ed42f6168d76`.
[^lt-hip]: [`vendors/hip.h` L241–L256 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vendors/hip.h#L241-L256): `RDNA3_0` iff `RDNA3 && !RDNA3_5` (`RDNA3` from `__GFX11__`); `RDNA2` for `__gfx1030__`…`__gfx1037__`.
[^lt-dense]: [`vecdotq.cuh` L1436–L1519 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L1436-L1519): `VDR_Q5_K_Q8_1_MMVQ_RDNA3_0 4`, the arch gate (L1442–L1444), and `vec_dot_q5_K_q8_1_vdr4_dense` with its "pure reordering" comment and `ggml_cuda_dp4a` chains (L1501–L1509).
[^lt-vmmq]: [`vecdotq.cuh` L649–L718 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L649-L718): `VDR_Q5_K_Q8_1_MMVQ 2`, `…_MOE 4`, `vec_dot_q5_K_q8_1_impl_vmmq` / `_vmmq4`. Stock entry `vec_dot_q5_K_q8_1`: [L1327–L1376](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L1327-L1376).
[^lt-mul24]: [`vecdotq.cuh` L8–L14 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/vecdotq.cuh#L8-L14): `ggml_cuda_mul_small` → `__mul24` on HIP.
[^lt-dp4a]: [`common.cuh` L915–L920 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/common.cuh#L915-L920): `CDNA || RDNA2 || gfx906` → `__builtin_amdgcn_sdot4`; `RDNA3 || RDNA4_M || RDNA4` → `__builtin_amdgcn_sudot4`.
[^lt-vecdot-sel]: [`mmvq-vecdot.cuh` L47–L51 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq-vecdot.cuh#L47-L51).
[^lt-tables]: [`mmvq.cu` L60–L121 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq.cu#L60-L121): device `get_device_table_id()` (`RDNA2` → `MMVQ_PARAMETERS_NAVI21`), `get_vdr_mmvq_dense()` (L89–L107), host `get_device_table_id(cc)` (L109–L121).
[^lt-navi21]: [`mmvq.cu` L628–L680 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq.cu#L628-L680): NAVI21 `calc_nwarps` (4 / 2 / 1 waves) and `calc_rows_per_block` (n = 2..4 → 2 rows, 5..8 → 4 rows), with the comment citing V620 tuning from "llama-navi21-furnace".
[^lt-rpb]: [`mmvq-rdna-params.cuh` L130–L200 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq-rdna-params.cuh#L130-L200): `GGML_MMVQ_RDNA3_Q5K_WEIGHT_RPB 2`, `…IQ4XS… 2`, `calc_rows_per_block_weight` RDNA3_0 / RDNA4 cases, fallthrough to `calc_rows_per_block`.
[^lt-fuse]: [`mmvq.cu` L3736–L3740 @ `af7a57ba`](https://github.com/stew675/llama.cpp/blob/af7a57ba8700f499fa6ca01d47470208deae8718/ggml/src/ggml-cuda/mmvq.cu#L3736-L3740).
[^probe]: Probe committed in this wiki at [`e143080d`](https://github.com/BlivionIaG/rdna-hip-wiki/tree/e143080da80095221b76da995170eda3d6842261/kernels/quant/probes/q5k-dense-vdr4): `q5k_probe.cu`, `q5k_src.cuh` (verbatim excerpts), `build.sh` (toolchain + flags), `results.md` (full table incl. the one-row shapes), `rpass-kernel-resource-usage.txt` (raw remarks). The ISA counts come from the `-save-temps` `.s`, per kernel, via `table.py`.
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* — VOP2 opcode 13 `V_DOT4C_I32_I8`; VOP3P `V_DOT4_I32_I8`. <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^wiki-vgpr]: [silicon/occupancy/vgpr-occupancy.md @ `21b7ac68`](https://github.com/BlivionIaG/rdna-hip-wiki/blob/21b7ac68e161e2bab13a088d8263831cc9aecffd/silicon/occupancy/vgpr-occupancy.md): 1024-VGPR file per SIMD32, granule 16, `min(16, floor(1024 / alignTo(vgpr, 16)))`.
[^pl-mmvq]: opengfx1030/vllm-gguf-plugin [`csrc/gguf/mmvq.cuh` L1–L41 @ `b1c5e038`](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/mmvq.cuh#L1-L41) ("copied and adapted from … llama.cpp/blob/b2899"; one `tmp` per thread, `blockIdx.y` = column), Q5_K launcher [L115–L122](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/mmvq.cuh#L115-L122) (`block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1)`, `VDR_Q5_K_Q8_1_MMVQ`).
[^pl-vdr]: plugin [`vecdotq.cuh` L381](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/vecdotq.cuh#L381) `VDR_Q5_K_Q8_1_MMVQ 2`; `vec_dot_q5_K_q8_1` [L1279–L1325](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/vecdotq.cuh#L1279-L1325).
[^pl-common]: plugin [`ggml-common.h` L1–L9](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/ggml-common.h#L1-L9) (`GGML_CUDA_MMV_Y 1`).
[^pl-dp4a]: plugin [`ggml-common.h` L1125–L1133](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/gguf/ggml-common.h#L1125-L1133): `__dp4a` → `__builtin_amdgcn_sdot4` when available.
[^pl-warp]: plugin [`cuda_compat.h` L26–L38](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/csrc/cuda_compat.h#L26-L38): device `WARP_SIZE` = 64 only on `__GFX9__`, else 32.
[^pl-linear]: plugin [`quantization/linear.py` L35–L47](https://github.com/opengfx1030/vllm-gguf-plugin/blob/b1c5e038d7cca09929e408ea76331485f1b94ea9/vllm_gguf_plugin/quantization/linear.py#L35-L47): `mmvq_safe = 2 if weight.shape[0] > 5120 else 6` for non-imatrix types, then `ggml_mul_mat_vec_a8`.
[^ex-gguf]: opengfx1030/vllm-rdna [`docs/rdna2/gguf-flashnext-2026-11-16.md` L25–L36 @ `71299d56`](https://github.com/opengfx1030/vllm-rdna/blob/71299d56b2526dc4df8c74d261d0c57c39865dd2/docs/rdna2/gguf-flashnext-2026-11-16.md#L25-L36): "The loader runs the GGUF GEMMs through `vllm-gguf-plugin` … Use `opengfx1030/vllm-gguf-plugin`, branch `rdna_extras`".
[^ex-csrc]: opengfx1030/vllm-rdna [`csrc/` @ `71299d56`](https://github.com/opengfx1030/vllm-rdna/tree/71299d56b2526dc4df8c74d261d0c57c39865dd2/csrc) (`rdna_extras` tip, 2026-11-27 10:48 Paris): there is no `csrc/quantization/gguf`, and a recursive tree search for `gguf|q5_k|mmvq` finds only loader / tests / docs paths.
