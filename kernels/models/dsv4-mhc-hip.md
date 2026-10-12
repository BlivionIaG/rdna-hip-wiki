---
title: "DeepSeek-V4 mHC pre / post — RDNA HIP (`rdna/dsv4/mhc.cu`)"
updated: 2026-10-12
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/ec349e9079f6d19865362010bebcd36bfbbe0f8d
  - https://github.com/opengfx1030/vllm-rdna/commit/0f596e6e0e9b2683ef7a4ec3eb6ef93eac9e1fd9
  - https://github.com/opengfx1030/vllm-rdna/commit/64d59b8942d3b982f1d91f0f3e625dfda4418106
  - https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/vllm/models/deepseek_v4/amd/rdna/ops.py
  - https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/vllm/model_executor/layers/mhc.py
  - https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/torch_bindings.cpp
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html
---

# DeepSeek-V4 mHC pre / post — RDNA HIP

Model-specific HIP glue for DeepSeek-V4's manifold hyper-connection (mHC) on RDNA: `csrc/rocm/rdna/dsv4/mhc.cu`, added in `ec349e90` and reworked in `0f596e6e`, the last commit to touch the file before dest tip `64d59b89`.[^c-ec349e90][^c-0f596e6e][^c-64d59b89] It is not attention. DSv4 attention kernels are on [../attention/mla-sparse.md](../attention/mla-sparse.md) and [../attention/lightning-indexer.md](../attention/lightning-indexer.md). Back to the [models index](README.md).

Occupancy first; no tok/s on this page. The commit messages carry timings; they are not copied here.[^c-ec349e90][^c-0f596e6e]

## What it replaces

Without TileLang or AITER on RDNA, `MHCPreOp` / `MHCPostOp` ran the torch reference (`mhc_pre_torch` / `mhc_post_torch`). The source comment counts ~130 tiny kernels per `mhc_pre` call, 20 × 4 of them for the Sinkhorn loop, and 86 calls per decode step.[^f-mhc-head][^c-ec349e90]

Math, as written in the header comment (`HC` = 4 streams, fp32 throughout, same op order as the torch reference):[^f-mhc-head]

- `mixes = x @ fn^T`, fp32 `[T, 2·HC + HC·HC]` = `[T, 24]`
- `rms = rsqrt(sum(x²) / (HC·H) + rms_eps)`
- `pre = sigmoid(mixes[0:HC]·s0 + base) + pre_eps`; `post = sigmoid(mixes[HC:2HC]·s1 + base) · post_mult`
- `comb = softmax_j(mixes[2HC:]·s2 + base) + sinkhorn_eps`, column-normalised, then `(repeat − 1)` × (row, column) normalisation
- `layer_input[h] = Σ_c pre[c] · residual[c, h]`; post: `out[j, h] = Σ_i comb[i, j] · residual[i, h] + post[j] · x[h]`

## Kernels at `64d59b89`

| Kernel | Launch | LDS | Inner math |
|---|---|---|---|
| `mhc_pre_partial_kernel<T>` (decode only) | grid (`kPreSplits` = 32, T), 256 threads, `__launch_bounds__(256)` | `s_w[256/32][25]` fp32 = 800 B | each block sums `x²` and `x · fn[k]` (k = 0..23) over its chunk of the token's `HC·H` values, `__shfl_down` wave reduce, then one row of `partials[T, 32, 25]`[^f-mhc-partial][^f-mhc-consts] |
| `mhc_pre_kernel<T>` | grid T (one block per token), 256 threads, `__launch_bounds__(256)` | `s_red[256]` + `s_mix[24]` + `s_pre[4]` fp32 = 1136 B | reduce partials (decode) or `sum(x²)` (prefill), sigmoids, Sinkhorn, stream collapse[^f-mhc-pre][^f-mhc-consts] |
| `mhc_post_kernel<T>` | grid (`ceil(H/256)`, T), 256 threads, **no** `__launch_bounds__` | none | elementwise: 4 residual loads, 4×4 `comb` mix + `post · x`, fp32 accumulate[^f-mhc-post][^f-mhc-host-post] |

- **No DOT.** All three kernels are scalar fp32 (`expf`, `rsqrtf`, FMA). fp16 / bf16 inputs go to fp32 through `__half2float` / `__bfloat162float`. No `fdot2`, `sdot4` or WMMA.[^f-mhc-pre][^f-mhc-post][^f-mhc-dtype]
- **LDS is tiny.** Both pre kernels use under 1.2 KB. The ISA allocates LDS in 1 KiB granules, so these kernels take 1–2 KiB per workgroup and LDS is not their occupancy limiter.[^f-mhc-partial][^f-mhc-pre][^isa]
- **`mhc_post_kernel` has no `__launch_bounds__`.** HIP's default `MAX_THREADS_PER_BLOCK` is the device maximum (1024), so the compiler budgets registers for a 1024-thread block although the host always launches 256 ([hip-craft §1.3](../../silicon/hip-craft/hip-craft.md)). Its VGPR count is unmeasured. *(Watch, not a measured problem.)*[^f-mhc-post][^f-mhc-host-post][^hip-lb]
- **Graph-safe by construction.** Every launch uses `at::cuda::getCurrentCUDAStream()`. Outputs (`post`, `comb`, `layer_input`, `out`) and the decode `partials` buffer come from `torch::empty` / `torch::empty_like` per call, with no persistent buffer. The header comment calls this "CUDA-graph safe".[^f-mhc-head][^f-mhc-host-pre][^f-mhc-host-post]

## Decode pre GEMM split (`0f596e6e`)

*(Historical at `ec349e90`)* For decode, one block per token computed `x @ fn^T` + `sum(x²)` itself, streaming the ~1.5 MB fp32 `fn` through a single CU, and the Sinkhorn ran on one lane (~640 serial divides, per the commit message).[^c-0f596e6e][^c-ec349e90]

At `0f596e6e` (unchanged through `64d59b89`):[^c-0f596e6e][^f-mhc-consts]

1. **Split-K over 32 blocks per token.** `mhc_pre_partial_kernel` writes `[T, 32, 25]` partials, slot 0 for `sum(x²)` and slots 1..24 for `x · fn[k]`. `chunk = ceil(4H / 32)`. The buffer is a per-call `torch::empty` on the current stream.[^f-mhc-partial][^f-mhc-host-pre]
2. **Reduce in `mhc_pre_kernel`.** Threads `0..24` each sum one slot over the 32 splits into `s_red[0]` / `s_mix[]`.[^f-mhc-pre]
3. **Parallel Sinkhorn on 16 lanes.** Lanes `tid < 16` of the first wave each hold one `(i, j)` of the 4×4 `comb`. Row sums are `__shfl_xor` over lane bits 0–1 (offsets 1, 2) and column sums over bits 2–3 (offsets 4, 8). Every shuffle partner is inside lanes 0..15, so the inactive lanes 16..31 are never read. The source comment says these are the same per-element ops as the torch reference.[^f-mhc-sinkhorn]

The prefill path (mixes passed in) keeps the single kernel: a block-strided `sum(x²)` into `s_red[256]`, then **thread 0 serially adds the 256 entries**. The commit says prefill "keeps the single kernel"; the serial 256-add is visible in the source and has not been measured here.[^f-mhc-pre][^c-0f596e6e]

## Dispatch at `64d59b89`

- **Build.** `csrc/rocm/rdna/dsv4/mhc.cu` is on the CMake list next to the other `rdna/dsv4` kernels.[^f-cmake]
- **Ops.** `dsv4_mhc_pre_rdna(residual, mixes, hc_scale, hc_base, rms_eps, pre_eps, sinkhorn_eps, post_mult, sinkhorn_repeat) -> Tensor[]` and `dsv4_mhc_post_rdna(x, residual, post, comb) -> Tensor`. They return fresh tensors, with no `Tensor!` out arguments.[^f-bind]
- **Selection.** `MHCPreOp.forward_hip` tries AITER, then TileLang, then the RDNA HIP path, then `forward_native`. `MHCPostOp.forward_hip` uses the same order. `_use_rdna_mhc` needs `residual.shape[-2] == 4`, fp16 or bf16, an RDNA-family device (`on_rdna_family()`) and both ops built (`has_mhc()`).[^f-py-gate][^f-py-pre][^f-py-post][^f-ops]
- **Fused GEMM threshold.** `_MHC_FUSED_GEMM_MAX_TOKENS = 16`. Up to 16 tokens the wrapper passes `fn` (`[24, 4H]`) and the kernel does the GEMM through the split-K partials. Above 16 it computes `mixes = res.float() @ fn.t()` in torch, which the source comment calls "one rocBLAS GEMM", and passes `[T, 24]`.[^f-ops]
- **Only pre and post.** At `64d59b89` only `MHCPreOp` and `MHCPostOp` call `_use_rdna_mhc`. `MHCPreDelayedOp`, `HCHeadOp` and `MHCFusedPostPreOp` have no RDNA branch. Whether a DSv4 RDNA serve reaches those ops is (unsourced).[^f-py-gate][^f-py-pre][^f-py-post]

## Leave

- Copying the commit-message timings or PPL numbers as a contract.[^c-ec349e90][^c-0f596e6e]
- Treating mHC as a DOT kernel. It has no packed-DOT inner loop.[^f-mhc-pre][^f-mhc-post]

## Sources

- opengfx1030/vllm-rdna commits `ec349e90`, `0f596e6e`; tip `64d59b89` (read 2026-10-12)
- `csrc/rocm/rdna/dsv4/mhc.cu`, `vllm/models/deepseek_v4/amd/rdna/ops.py`, `vllm/model_executor/layers/mhc.py`, `CMakeLists.txt`, `csrc/rocm/torch_bindings.cpp` @ `64d59b89`
- AMD RDNA 2 ISA (doc 70648), LDS allocation granule
- HIP C++ language extensions, `__launch_bounds__` defaults; summarised in [silicon/hip-craft/hip-craft.md](../../silicon/hip-craft/hip-craft.md) §1.3–§1.4

[^c-ec349e90]: opengfx1030/vllm-rdna commit [`ec349e90`](https://github.com/opengfx1030/vllm-rdna/commit/ec349e9079f6d19865362010bebcd36bfbbe0f8d) (2026-10-09) — "perf(rdna): fused HIP mHC pre/post for DeepSeek-V4 on RDNA". Adds `csrc/rocm/rdna/dsv4/mhc.cu`, the op bindings and the `mhc.py` selection.
[^c-0f596e6e]: opengfx1030/vllm-rdna commit [`0f596e6e`](https://github.com/opengfx1030/vllm-rdna/commit/0f596e6e0e9b2683ef7a4ec3eb6ef93eac9e1fd9) (2026-10-09) — "perf(rdna): split the decode mHC pre GEMM across blocks, parallel Sinkhorn". The message describes the single-CU `fn` stream, the one-lane Sinkhorn (~640 serial divides), the 32-block split into `[T, 32, 25]` partials (torch::empty, current stream), the 16-lane xor-shuffle Sinkhorn, and prefill keeping the single kernel.
[^c-64d59b89]: opengfx1030/vllm-rdna commit [`64d59b89`](https://github.com/opengfx1030/vllm-rdna/commit/64d59b8942d3b982f1d91f0f3e625dfda4418106) — dest `rdna_extras` tip read 2026-10-12 (merge of PR #84). `git log 64d59b8942d3b982f1d91f0f3e625dfda4418106 -- csrc/rocm/rdna/dsv4/mhc.cu` lists `0f596e6e`, `c19e8c0c` (style), `ec349e90`.
[^f-mhc-head]: `csrc/rocm/rdna/dsv4/mhc.cu` L4-L25, header comment (math, op count, graph safety), dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L4-L25
[^f-mhc-consts]: `csrc/rocm/rdna/dsv4/mhc.cu` L39-L44, `kHc` 4, `kMix` 24, `kPreThreads` 256, `kPreSplits` 32, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L39-L44
[^f-mhc-dtype]: `csrc/rocm/rdna/dsv4/mhc.cu` L46-L69, `to_f32` / `from_f32` / `sigmoidf`, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L46-L69
[^f-mhc-partial]: `csrc/rocm/rdna/dsv4/mhc.cu` L71-L115, `mhc_pre_partial_kernel`, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L71-L115
[^f-mhc-pre]: `csrc/rocm/rdna/dsv4/mhc.cu` L117-L216, `mhc_pre_kernel` (LDS L134-L136, decode reduce L138-L151, prefill serial sum L152-L166), dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L117-L216
[^f-mhc-sinkhorn]: `csrc/rocm/rdna/dsv4/mhc.cu` L169-L205, 16-lane Sinkhorn with `__shfl_xor` 1/2 (row) and 4/8 (column), dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L169-L205
[^f-mhc-post]: `csrc/rocm/rdna/dsv4/mhc.cu` L218-L239, `mhc_post_kernel` (no `__launch_bounds__`), dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L218-L239
[^f-mhc-host-pre]: `csrc/rocm/rdna/dsv4/mhc.cu` L245-L316, `dsv4_mhc_pre_rdna` host (torch::empty outputs and partials, current stream, launches), dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L245-L316
[^f-mhc-host-post]: `csrc/rocm/rdna/dsv4/mhc.cu` L318-L361, `dsv4_mhc_post_rdna` host (`kThreads` 256, grid `(ceil(H/256), T)`, `torch::empty_like`), dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/rdna/dsv4/mhc.cu#L318-L361
[^f-ops]: `vllm/models/deepseek_v4/amd/rdna/ops.py` L261-L327, `_MHC_FUSED_GEMM_MAX_TOKENS`, `has_mhc`, `mhc_pre`, `mhc_post`, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/vllm/models/deepseek_v4/amd/rdna/ops.py#L261-L327
[^f-py-gate]: `vllm/model_executor/layers/mhc.py` L34-L51, `_rdna_mhc_available` / `_use_rdna_mhc`, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/vllm/model_executor/layers/mhc.py#L34-L51
[^f-py-pre]: `vllm/model_executor/layers/mhc.py` L163-L250, `MHCPreOp.forward_hip` (AITER → TileLang → RDNA → native), dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/vllm/model_executor/layers/mhc.py#L163-L250
[^f-py-post]: `vllm/model_executor/layers/mhc.py` L586-L608, `MHCPostOp.forward_hip`, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/vllm/model_executor/layers/mhc.py#L586-L608
[^f-cmake]: `CMakeLists.txt` L1546-L1550, DeepSeek-V4 RDNA kernels list incl. `csrc/rocm/rdna/dsv4/mhc.cu`, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/CMakeLists.txt#L1546-L1550
[^f-bind]: `csrc/rocm/torch_bindings.cpp` L418-L428, `dsv4_mhc_pre_rdna` / `dsv4_mhc_post_rdna` schemas, dest at 64d59b89. https://github.com/opengfx1030/vllm-rdna/blob/64d59b8942d3b982f1d91f0f3e625dfda4418106/csrc/rocm/torch_bindings.cpp#L418-L428
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648) §3.6.6: LDS is allocated in 256-dword (1 KiB) granules; quoted in [silicon/hip-craft/hip-craft.md](../../silicon/hip-craft/hip-craft.md) §1.4. <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^hip-lb]: HIP C++ language extensions, `__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)`: `MAX_THREADS_PER_BLOCK` defaults to the device maximum (1024); summarised with LLVM `AMDGPUBaseInfo.cpp` @ `eb61ef62df` in [silicon/hip-craft/hip-craft.md](../../silicon/hip-craft/hip-craft.md) §1.3. <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html> ; <https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp>
