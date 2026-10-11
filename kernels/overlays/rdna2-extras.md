---
title: "`rdna2_extras` @ `3e05abc9` — HIP / silicon review"
sources:
  - https://github.com/BlivionIaG/vllm/commit/9ff87936d6113c8f374ddc62c1d5dcb95197bc77
  - https://github.com/BlivionIaG/vllm/commit/3e05abc9bdb92100b0fec7a91e856c147dc6849c
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/CMakeLists.txt
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/moe_q_gemm_rdna2.cu
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/q_gemm_rdna2.cu
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/fa_rdna2.cu
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/q_gemm_rdna2_common.cuh
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/skinny_gemms.cu
  - https://github.com/BlivionIaG/vllm/commit/66bb24d7bf7ea27a2a71dbd7816308b38d039db8
  - https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/moe_w8a16_fp8_rdna2.cu
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---
# `rdna2_extras` @ `3e05abc9` — HIP / silicon review

Date: 2026-08-19. Tip of `BlivionIaG/vllm` `rdna2_extras` = v0.27.1 + `perf/rdna2_w4a16` merge `9ff87936` + PYNCCL bypass `3e05abc9`. Engine process: rebase-on-release is correct; keep this as the human branch.[^c-9ff87936][^c-3e05abc9]

## Survived (compile list is gfx1030-gated)

`CMakeLists.txt` `VLLM_GPU_ARCHES MATCHES gfx1030` still builds:[^f-3e05abc9-CMakeLists-txt]

`q_gemm_rdna2` + prefill, `fa_rdna2`, `q_gemm_w8a16_fp8`, `gemm_w8a8_fp8_dense`, `moe_q_gemm_rdna2`, `moe_w8a16_rdna2`, `mxfp4_dot2_{dense,moe}`, indexer, `sparse_mla_rdna2`.[^f-3e05abc9-moe-q-gemm-rdna2-cu][^f-3e05abc9-q-gemm-rdna2-cu][^f-3e05abc9-fa-rdna2-cu]

`q_gemm_rdna2_common.cuh` still issues explicit `__builtin_amdgcn_fdot2`. `fa_rdna2` still uses the same builtin for QK. Good.[^f-3e05abc9-q-gemm-rdna2-common-cuh][^f-3e05abc9-fa-rdna2-cu]

## Did not move (still first)

| Bug | Still at tip |
|---|---|
| `fa_rdna2` occupancy | `__launch_bounds__(128, 1)` / `(256, 1)` — HIP second arg is **min waves/EU** [^f-3e05abc9-fa-rdna2-cu] |
| skinny occupancy | `wvSplitKrc_` still `amdgpu_waves_per_eu(1, 1)` [^f-3e05abc9-skinny-gemms-cu] |
| MLA prefill `load_row` | same blob as `66bb24d7` (`c3db1df8`): `chunk = i*THREADS+tid` into 64 float4s [^c-66bb24d7] |

These are not “lost in rebase.” They were never fixed on `perf/rdna2_w4a16`.[^c-3e05abc9][^f-3e05abc9-CMakeLists-txt]

## Merge gap

`csrc/rocm/moe_w8a16_fp8_rdna2.cu` is **on the branch but not in the gfx1030 CMake list**. It will not ship in `_rocm_C` until appended.[^f-3e05abc9-moe-w8a16-fp8-rdna2-cu]

## Dispatch landmine on v0.27.1

```python
_ON_RDNA = _ON_GFX1X and not _ON_CDNA   # gfx11/12 only
on_gfx10x()                             # gfx1030
on_rdna()                               # NOT gfx1030
is_navi()                               # "gfx1" in arch — true on gfx1030 by accident
```

Use `on_gfx10x()` for V620. Do not write new Python as `if on_rdna()`.[^c-3e05abc9][^f-3e05abc9-CMakeLists-txt]

`use_rocm_custom_paged_attention` now allows `_ON_GFX10X` for head 128 / block 16. Confirm the object that fires is `fa_rdna2`, not a gfx11 paged kernel. Profiler-proof it on the first extras bench.[^f-3e05abc9-fa-rdna2-cu]

## Overlay replay list (next tag)

1. Occupancy flip (`fa_rdna2` + skinny `waves_per_eu`).[^f-3e05abc9-fa-rdna2-cu]
2. MLA `load_row` OOB, then prefill `fdot2`.[^c-3e05abc9][^f-3e05abc9-CMakeLists-txt]
3. Add `moe_w8a16_fp8_rdna2.cu` to CMake or delete the orphan.[^f-3e05abc9-moe-w8a16-fp8-rdna2-cu]
4. Re-check every `on_rdna()` / `on_gfx1x()` call site against `on_gfx10x()`.[^c-3e05abc9][^f-3e05abc9-CMakeLists-txt]
5. ISA dump: `v_dot2*` in W4A16/FA; no WMMA in gfx1030 objects.[^isa]

Occupancy still first.

## Sources

[^c-9ff87936]: BlivionIaG/vllm commit [`9ff87936`](https://github.com/BlivionIaG/vllm/commit/9ff87936d6113c8f374ddc62c1d5dcb95197bc77) (2026-08-17) — "merge: integrate perf/rdna2_w4a16 RDNA2 kernel stack into v0.27.1".
[^c-3e05abc9]: BlivionIaG/vllm commit [`3e05abc9`](https://github.com/BlivionIaG/vllm/commit/3e05abc9bdb92100b0fec7a91e856c147dc6849c) (2026-08-18) — "fix(rocm): disable vllm::all_reduce dispatcher on gfx1030/gfx1100".
[^f-3e05abc9-CMakeLists-txt]: BlivionIaG/vllm `CMakeLists.txt` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/CMakeLists.txt).
[^f-3e05abc9-moe-q-gemm-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/moe_q_gemm_rdna2.cu` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/moe_q_gemm_rdna2.cu).
[^f-3e05abc9-q-gemm-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/q_gemm_rdna2.cu` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/q_gemm_rdna2.cu).
[^f-3e05abc9-fa-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/fa_rdna2.cu` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/fa_rdna2.cu).
[^f-3e05abc9-q-gemm-rdna2-common-cuh]: BlivionIaG/vllm `csrc/rocm/q_gemm_rdna2_common.cuh` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/q_gemm_rdna2_common.cuh).
[^f-3e05abc9-skinny-gemms-cu]: BlivionIaG/vllm `csrc/rocm/skinny_gemms.cu` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/skinny_gemms.cu).
[^c-66bb24d7]: BlivionIaG/vllm commit [`66bb24d7`](https://github.com/BlivionIaG/vllm/commit/66bb24d7bf7ea27a2a71dbd7816308b38d039db8) (2026-08-17) — "feat(rocm): HIP sparse MLA prefill kernel for gfx1030".
[^f-3e05abc9-moe-w8a16-fp8-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/moe_w8a16_fp8_rdna2.cu` @ [`3e05abc9`](https://github.com/BlivionIaG/vllm/blob/3e05abc9bdb92100b0fec7a91e856c147dc6849c/csrc/rocm/moe_w8a16_fp8_rdna2.cu).
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
