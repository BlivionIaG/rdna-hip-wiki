---
title: "Runtime branch on a tile variant costs VGPRs even when off (gfx1030 occupancy)"
updated: 2026-11-26
sources:
  - https://github.com/stew675/llama-cpp-rdna-boosts/commit/761fcb43279c5be9569af51047a1e0896b9fe577
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/pr362/README.md
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/pr362/pr-submitted.patch
  - https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/CAMPAIGNS.md
  - https://github.com/ggml-org/llama.cpp/blob/a866b4ade63921826c530307a368fbca3a334a0b/ggml/src/ggml-cuda/fattn-common.cuh
  - https://github.com/ggml-org/llama.cpp/blob/a866b4ade63921826c530307a368fbca3a334a0b/ggml/src/ggml-cuda/vendors/hip.h
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/TargetParser/AMDGPUTargetParser.cpp
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td
  - https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/GCNProcessors.td
  - https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/moe_lowp_dot2_rdna2.cu
  - https://github.com/ROCm/llvm-project/commit/46fcb339fb61119b337f973c7ca9e710a319fdd0
---
# Runtime branch on a tile variant costs VGPRs even when off (gfx1030 occupancy)

Source: stew675/llama-cpp-rdna-boosts `761fcb43` (2026-11-25), the review record that rejected contributor PR #362, an RDNA3 padded GQA-6 tile fold.[^c-761fcb43] The PR was rejected because the fold was a **runtime** branch. Its off arm was not inert on gfx1201: VGPR went 140 → 94 on one tile instantiation.[^r-b1] This page covers what that means for **gfx1030 occupancy** and how we write tile variants. It has no throughput numbers.

Up: [hip-craft/](README.md). VGPR ladder: [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md). Constexpr arch traits: [hip-craft.md](hip-craft.md) §5.5.

## Verdicts

| Verdict | Claim |
|---|---|
| **Take (rule, Now for new HIP)** | A tile or feature variant that changes per-lane live state must be a **template / `if constexpr` parameter**. Pick the instantiation on the host. Never use a runtime `bool`/`int` inside the kernel for this. Then every existing instantiation compiles to the pre-change code by construction. This is stew675's rework item R1.[^r-r1] |
| **Take (fact)** | `.vgpr_count` is one number for the whole kernel. A runtime arm that never executes still adds to it, and the occupancy rung follows from it. On gfx1030 the rung is `min(16, 1024 / alignTo(VGPR, 16))`.[^llvm-wave-formula][^llvm-granule][^llvm-total][^llvm-maxwaves] |
| **Take (fact)** | Occupancy can feed launch geometry and numerics. llama.cpp seeds `parallel_blocks` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, which is `hipOccupancyMaxActiveBlocksPerMultiprocessor` on HIP.[^lc-occ][^lc-hip] stew675 measured `P` 18 vs 6 between the arms and notes that `P` *is* the split-KV summation order.[^r-mech] So a VGPR change can move the logits, not only the speed. |
| **Take (gfx1030 toy, CE run)** | ROCm 7.14.0 clang, `--offload-arch=gfx1030`, toy below, with `amdgpu_flat_work_group_size(1,64)`: the runtime-`fold` kernel gets **242 VGPR → occupancy 4**. The compile-time off instantiation gets **99 → 9** and the on instantiation **133 → 7**.[^ce] The runtime kernel is worse than either arm. |
| **Leave** | Their fold, their gfx1100/gfx1201 numbers and their throughput. gfx1100 and gfx1201 have a 1536-VGPR wave32 file with granule 24, so the same raw VGPR count lands on different rungs than on gfx1030 (1024 / 16).[^llvm-granule][^llvm-total][^td-1536][^procs] |
| **No extras change** | The one variant switch already cited in this wiki, Inkling-2 NVFP4 `VLLM_INKLING2_NVFP4_VARIANT`, already picks a **template instantiation on the host** (`<T, BM, 128, 256>` vs default) inside `if constexpr (BLOCK_SIZE_M <= 4)`. That is the R1 shape.[^e5-variant] No other extras kernels were audited in this pass. |

## 1. What stew675 recorded (`761fcb43`)

- The fold is a runtime branch because `gqa_ratio` is not known at compile time. So it is compiled into **every** `flash_attn_tile` instantiation.[^r-b1] In the submitted patch, `fold_gqa = kv_is_native && ncols2 > gqa_ratio` and `warp_valid` / `ncols2_eff` are runtime values used all through the tile loop.[^p-runtime]
- gfx1201, fold off vs an unpatched build: `llvm-readobj` reads **VGPR 140 → 94** on `flash_attn_tile<256,256,8,2,...>`, and "others move too". The occupancy-derived split changes, mean KLD is 0.000391, same-top-p is 98.97 %, and greedy text flips. The fold could not be reached in these runs.[^r-b1]
- gfx1100: the same patch's off arm **is** inert (mean KLD 0.000000, identical text).[^r-b1] Whether a runtime arm costs VGPRs depends on the arch. stew675 calls it "an arch-dependent codegen accident" that "has to be removed **by construction**, not measured away".[^r-b1]
- With the fold on, the mechanism they printed: fold off = `ncols2 2, … max_blocks_per_sm 18, P 18`, fold on = `ncols2 8, … max_blocks_per_sm 6, P 6`. Block b owns KV iterations b, b+P, … and the combine sums partials in block order.[^r-mech]
- R1 rework: make `fold` a template parameter that defaults to `false`, add **one** `fold = true` instantiation, and plumb `warp_valid` / `ncols2_eff` through `if constexpr`. Acceptance is an off path that matches bit for bit on gfx1201 as well as gfx1100.[^r-r1] The campaign index lists the PR as OPEN, assessed and rejected as submitted.[^campaigns]

## 2. gfx1030 arithmetic (sourced inputs, arithmetic labelled)

Inputs. All are from LLVM at `eb61ef62`:

- gfx1030 uses `FeatureISAVersion10_3_0`, which includes `FeatureGFX10_3Insts` and `FeatureMaxWavesPerEU16`.[^procs][^llvm-maxwaves]
- `getVGPRAllocGranule` returns 16 for wave32 on GFX10.3 without `Feature1536VGPRs`.[^llvm-granule]
- `getTotalNumVGPRs` returns 1024 for wave32 on GFX10.[^llvm-total]
- `getNumWavesPerEUWithNumVGPRs` = `min(max(Total / alignTo(N, granule), 1), MaxWaves)`, or MaxWaves when `N < granule`.[^llvm-wave-formula]

Full wave32 ladder for gfx1030. **Arithmetic** from the formula above:

| Raw VGPR | Waves/SIMD32 |
|---|---|
| 1–64 | 16 |
| 65–80 | 12 |
| 81–96 | 10 |
| 97–112 | 9 |
| 113–128 | 8 |
| 129–144 | 7 |
| 145–160 | 6 |
| 161–192 | 5 |
| 193–256 | 4 |

The CE run in §3 reports the 9 and 7 rungs directly in the compiler's `; Occupancy:` comment (99 → 9, 133 → 7).[^ce] Before this pass, the ladder in [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md) §2 skipped the 12/10/9/7/5 rungs. It is corrected in the same branch.

**Arithmetic only: stew675's gfx1201 counts put through the gfx1030 formula.** This is not a claim that the kernel would compile to the same counts on gfx1030. 140 → align 144 → **7** waves. 94 → align 96 → **10** waves. A runtime arm that adds about 46 VGPRs crosses three rungs on gfx1030. The same counts under the gfx1100/gfx1201 file (1536 entries, granule 24[^llvm-granule][^llvm-total]) give 144 → 10 and 96 → 16 (also arithmetic). Rung boundaries differ per arch. That is one reason a runtime arm can be inert on one target and not on another. stew675 does not give a root cause, so this is (unsourced) as an explanation of their gfx1100 result.

## 3. Toy demonstration on gfx1030 (Compiler Explorer)

Compiler: CE `clang-rocm-71400` = "x86-64 clang rocm-7.14.0", clang 23.0.0git @ ROCm/llvm-project [`46fcb339`](https://github.com/ROCm/llvm-project/commit/46fcb339fb61119b337f973c7ca9e710a319fdd0). Flags: `-x hip --offload-arch=gfx1030 --cuda-device-only -nogpulib -nogpuinc -O3 -S`. Run 2026-11-26 Europe/Paris through the CE REST API. No shortlink was made; rerun from the source below.[^ce]

```cpp
#define G __attribute__((global, amdgpu_flat_work_group_size(1, 64)))  // run B; run A drops the attribute
#define D __attribute__((device))
template <int N> D inline void tile(const float* __restrict__ a, float* __restrict__ o,
                                    int n, bool fold, int ratio) {
  int t = __builtin_amdgcn_workitem_id_x() + __builtin_amdgcn_workgroup_id_x() * 64;
  float acc[N];
#pragma unroll
  for (int i = 0; i < N; ++i) acc[i] = 0.f;
  if (fold) {                       // "variant" arm: 2N live accumulators
    float acc2[2 * N];
#pragma unroll
    for (int i = 0; i < 2 * N; ++i) acc2[i] = 0.f;
    for (int k = 0; k < n; ++k)
#pragma unroll
      for (int i = 0; i < 2 * N; ++i) acc2[i] += a[(k * 2 * N + i) * 4096 + t] * (float)((i % ratio) + 1);
#pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = acc2[i] + acc2[i + N];
  } else {
    for (int k = 0; k < n; ++k)
#pragma unroll
      for (int i = 0; i < N; ++i) acc[i] += a[(k * N + i) * 4096 + t];
  }
#pragma unroll
  for (int i = 0; i < N; ++i) o[i * 4096 + t] = acc[i];
}
template <int N> G void k_rt(const float* a, float* o, int n, bool fold, int ratio) { tile<N>(a, o, n, fold, ratio); }
template <int N, bool FOLD> G void k_ct(const float* a, float* o, int n, int ratio) {
  if constexpr (FOLD) tile<N>(a, o, n, true, ratio); else tile<N>(a, o, n, false, ratio);
}
template G void k_rt<32>(const float*, float*, int, bool, int);
template G void k_ct<32, false>(const float*, float*, int, int);
template G void k_ct<32, true>(const float*, float*, int, int);
```

Values read from the emitted `.s` (`NT_AMDGPU_METADATA` and `; Occupancy:`). Every kernel reports `.amdhsa_wavefront_size32 1` and `.workgroup_processor_mode: 1`:[^ce]

| Kernel | Run A (default max WG 1024): `.vgpr_count` / spills / private seg / Occupancy | Run B (`flat_work_group_size(1,64)`): same columns |
|---|---|---|
| `k_rt<32>` (runtime `fold`) | 128 / 0 / 244 B / 8 | **242** / 0 / 8 B / **4** |
| `k_ct<32,false>` (off instantiation) | 99 / 0 / 0 / 9 | **99** / 0 / 0 / **9** |
| `k_ct<32,true>` (on instantiation) | 128 / 6 / 28 B / 8 | **133** / 0 / 0 / **7** |

Reading:

- **Off-arm cost.** A caller that passes `fold=false` to `k_rt` still pays for the fold arm: 242 vs 99 VGPR in run B, and 4 vs 9 waves/SIMD.
- **Run A.** With the default 1024-thread bound, the VGPR budget is capped at 128. Arithmetic: 1024 threads = 32 waves over 4 SIMD32s = 8 waves/SIMD, and 1024 / 8 = 128 (see the [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md) §3 trap). The runtime kernel then shows its cost as a 244-byte private segment instead of VGPRs. The compile-time off instantiation stays clean at 99 / 0 B.
- **The runtime kernel is worse than the max of the two arms** (242 > 133). Why the compiler did this was not traced (unsourced). Treat it as a toy result, not a rule.

This is a contrived kernel, not an FA tile, and nothing was run on silicon. It shows that the effect stew675 measured on gfx1201 can be reproduced on gfx1030 codegen. It says nothing about how often this happens in real kernels.

## 4. Craft rules for gfx1030 HIP

1. **Variants that change live state** must be template parameters: tile width, GQA fold, fused epilogue, extra accumulators, K-split mode. The host `switch` / `if` picks among explicit instantiations. Precedent in extras: Inkling-2 NVFP4 variant dispatch at `ea8aa241`.[^e5-variant]
2. **Off-by-default env gates must not touch device code.** An env var (`getenv`) should choose a *kernel*, not set a kernel argument that guards a heavy arm. Otherwise "gate off" does not restore the original object. That is stew675's R6 claim withdrawal for `GGML_HIP_FA_TILE_GQA6_FOLD=0`.[^r-r1]
3. **Acceptance for a new variant:** diff the `.vgpr_count` / `.private_segment_fixed_size` / `; Occupancy:` of every *existing* instantiation against the pre-change object. They must be identical. Recipe: [occupancy-dump.md](../occupancy/occupancy-dump.md). This mirrors stew675's resubmission check 1.[^r-r1]
4. **Uniform runtime scalars are probably fine** when they only change addressing or loop bounds and do not add a block of live values: strides, `n`, a single flag that selects one load. This is an (unsourced) heuristic. Neither stew675 nor the toy tests it. The toy only shows the cost when an arm holds *extra* live state. Measure; do not assume either way.
5. If a launcher derives split counts from `hipOccupancyMaxActiveBlocksPerMultiprocessor` (the llama.cpp pattern[^lc-occ]), a VGPR change in a shared instantiation also changes the summation order. Treat any variant PR that moves VGPRs of existing instantiations as a **numerics** change, not just a performance change.[^r-mech]

## Sources

- stew675/llama-cpp-rdna-boosts `761fcb43` (commit, `wip/pr362/README.md`, `wip/pr362/pr-submitted.patch`, `wip/CAMPAIGNS.md`): https://github.com/stew675/llama-cpp-rdna-boosts/commit/761fcb43279c5be9569af51047a1e0896b9fe577
- ggml-org/llama.cpp `a866b4ad` `fattn-common.cuh` (occupancy → `parallel_blocks`), `vendors/hip.h` (CUDA → HIP occupancy mapping): https://github.com/ggml-org/llama.cpp/tree/a866b4ade63921826c530307a368fbca3a334a0b/ggml/src/ggml-cuda
- LLVM `eb61ef62` `AMDGPUBaseInfo.cpp`, `AMDGPUTargetParser.cpp`, `AMDGPU.td`, `GCNProcessors.td`: https://github.com/llvm/llvm-project/tree/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm
- opengfx1030/vllm-rdna `ea8aa241` `csrc/rocm/moe_lowp_dot2_rdna2.cu` (host-side variant → template instantiation): https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/moe_lowp_dot2_rdna2.cu
- Compiler Explorer `clang-rocm-71400` (ROCm/llvm-project `46fcb339`): https://github.com/ROCm/llvm-project/commit/46fcb339fb61119b337f973c7ca9e710a319fdd0
- Wiki priors: [vgpr-occupancy.md](../occupancy/vgpr-occupancy.md), [occupancy-dump.md](../occupancy/occupancy-dump.md), [hip-craft.md](hip-craft.md) §5.5, [inkling2-mini-hip.md](../../kernels/models/inkling2-mini-hip.md)

Idle pass 2026-11-26.

[^c-761fcb43]: stew675/llama-cpp-rdna-boosts commit [`761fcb43`](https://github.com/stew675/llama-cpp-rdna-boosts/commit/761fcb43279c5be9569af51047a1e0896b9fe577) (2026-11-25 10:30 -0600 = 17:30 Europe/Paris), "wip/pr362: contributor PR #362 (RDNA3 padded GQA-6 tile fold) -- assessment record and resubmission checklist"; on `main`.
[^r-b1]: `wip/pr362/README.md` @ `761fcb43` L24-L48 (B1: runtime branch compiled into every instantiation; gfx1201 VGPR 140 → 94, KLD/top-p/text table; gfx1100 inert; "removed by construction"). https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/pr362/README.md#L24-L48
[^r-mech]: `wip/pr362/README.md` @ `761fcb43` L76-L79 (fold off `max_blocks_per_sm 18, P 18` vs on `6, P 6`; P is the summation order). https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/pr362/README.md#L76-L79
[^r-r1]: `wip/pr362/README.md` @ `761fcb43` L101-L126 (R1 template parameter + one instantiation + `if constexpr`; R2 opt-in; R6 `=0` does not restore the original path) and L165-L166 (resubmission check 1: diff VGPR/occupancy of pre-existing objects). https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/pr362/README.md#L101-L166
[^p-runtime]: `wip/pr362/pr-submitted.patch` @ `761fcb43` L129-L181 (`gqa_ratio = ne02 / ne12`; `fold_gqa`, `ncols2_eff`, `warp_valid` computed at run time). https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/pr362/pr-submitted.patch#L129-L181
[^campaigns]: `wip/CAMPAIGNS.md` @ `761fcb43` L99 (pr362 row: OPEN, rejected as submitted). https://github.com/stew675/llama-cpp-rdna-boosts/blob/761fcb43279c5be9569af51047a1e0896b9fe577/wip/CAMPAIGNS.md#L99
[^lc-occ]: ggml-org/llama.cpp `fattn-common.cuh` @ `a866b4ad` L1461-L1464 (`cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, …)`; `parallel_blocks = max_blocks_per_sm`). https://github.com/ggml-org/llama.cpp/blob/a866b4ade63921826c530307a368fbca3a334a0b/ggml/src/ggml-cuda/fattn-common.cuh#L1461-L1464
[^lc-hip]: ggml-org/llama.cpp `vendors/hip.h` @ `a866b4ad` L152 (`#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor`). https://github.com/ggml-org/llama.cpp/blob/a866b4ade63921826c530307a368fbca3a334a0b/ggml/src/ggml-cuda/vendors/hip.h#L152
[^llvm-granule]: LLVM `AMDGPUBaseInfo.cpp` @ `eb61ef62` L1286-L1306 (`getVGPRAllocGranule`: `Feature1536VGPRs` → 24 wave32; `hasGFX10_3Insts` → 16 wave32). https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1286-L1306
[^llvm-wave-formula]: LLVM `AMDGPUBaseInfo.cpp` @ `eb61ef62` L1346-L1363 (`getNumWavesPerEUWithNumVGPRs`). https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp#L1346-L1363
[^llvm-total]: LLVM `AMDGPUTargetParser.cpp` @ `eb61ef62` L441-L450 (`getTotalNumVGPRs`: 1536 wave32 with `FEAT_1536_PHYSICAL_VGPRS`, else 1024 wave32 on GFX10+). https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/TargetParser/AMDGPUTargetParser.cpp#L441-L450
[^llvm-maxwaves]: LLVM `AMDGPU.td` @ `eb61ef62` L2278-L2292 (`FeatureISAVersion10_3_0`: `FeatureGFX10_3Insts`, no `Feature1536VGPRs`, `FeatureMaxWavesPerEU16`). https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2278-L2292
[^td-1536]: LLVM `AMDGPU.td` @ `eb61ef62` L2354-L2357 (`FeatureISAVersion11_0_0` adds `Feature1536VGPRs`) and L2414-L2462 (`FeatureISAVersion12` lists `Feature1536VGPRs`). https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2354-L2357 , https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/AMDGPU.td#L2414-L2462
[^procs]: LLVM `GCNProcessors.td` @ `eb61ef62` L192-L194 (gfx1030 → `FeatureISAVersion10_3_0`), L234-L236 (gfx1100 → `FeatureISAVersion11_0_0`), L301-L303 (gfx1201 → `FeatureISAVersion12`). https://github.com/llvm/llvm-project/blob/eb61ef62df4fd5151c4c379302f636ac430bc32d/llvm/lib/Target/AMDGPU/GCNProcessors.td#L192-L303
[^e5-variant]: opengfx1030/vllm-rdna `csrc/rocm/moe_lowp_dot2_rdna2.cu` @ `ea8aa241` L331-L406 (`nvfp4_tile_variant()` reads `VLLM_INKLING2_NVFP4_VARIANT` on the host; `launch_moe_nvfp4` picks `launch_moe_nvfp4_cfg<T, BM, 128, 256>` vs `<T, BM, THREADS_X, BLOCK_KN_SIZE>` under `if constexpr (BLOCK_SIZE_M <= 4)`). https://github.com/opengfx1030/vllm-rdna/blob/ea8aa2418e6373de1537fa56e4ae0911ef6162a6/csrc/rocm/moe_lowp_dot2_rdna2.cu#L331-L406
[^ce]: Compiler Explorer REST run, 2026-11-26 ~14:57 Europe/Paris (CET), compiler id `clang-rocm-71400` ("x86-64 clang rocm-7.14.0", `clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0)`), flags `-x hip --offload-arch=gfx1030 --cuda-device-only -nogpulib -nogpuinc -O3 -S`, source as in §3. Values read from the emitted `.s`: `.vgpr_count`, `.vgpr_spill_count`, `.private_segment_fixed_size`, `.max_flat_workgroup_size`, `; Occupancy:`. https://github.com/ROCm/llvm-project/commit/46fcb339fb61119b337f973c7ca9e710a319fdd0
