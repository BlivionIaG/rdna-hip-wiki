---
title: "LeapQuant INT8 GDN recurrent state — dest `rdna_extras` @ `3e274335`"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/3e2743351c97bd3f4367a23248b0e88495a249dd
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_layout.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/mono_gdn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/v1_mamba_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/ops.h
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/CMakeLists.txt
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/vllm/model_executor/layers/mamba/ops/leapquant_gdn.py
  - https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/vllm/platforms/rdna.py
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
---
# LeapQuant INT8 GDN recurrent state — dest `rdna_extras` @ `3e274335`

Silicon lock for vLLM-rdna PR #168 (merge `3e2743351c97bd3f4367a23248b0e88495a249dd`, previous watched tip `a63a6b0010fd4e761fb14393e32d3491f4a1460b`). Opt-in LeapQuant INT8 storage for the Gated DeltaNet recurrent state on gfx1030. No tok/s. VGPR / SGPR / occupancy were not dumped: this box has no hipcc / ROCm 7.14 toolchain.[^c-3e274335]

The default FP32 GDN path in [gdn-decode.md](gdn-decode.md) is unchanged. This page covers only the new storage kernels and the MonoKernel / align-copy hooks that read the same blob.

## Take / Leave

| | |
|---|---|
| **Take (Later, craft)** | INT8 residual `Z` with per-value-row scale `B` and per-key-column smoothing `C`, plus `r=4` FP16 compensators and a ring of FP16 records, as a gfx1030-legal GDN state layout. Dequant is `Z[v,k] * B[v] * C[k]`; the running state is then compensator `U~ K~^T` plus replay of `cursor` records. All of that math is scalar FP32 FMA after the convert. There is no `sdot4` / `fdot2` / `dot8`, no WMMA, and no bf16 dot in these files.[^f-cu-hdr][^f-load][^f-decode-isa] |
| **Take (Later, craft)** | Keep the existing GDN decode pin on the decode kernel: `__launch_bounds__(256)` + `amdgpu_waves_per_eu(2, 4)`, 32 V-rows × 8 K-slices, register-resident `h[16]`, no LDS. The fold / normalize / store kernels use `__launch_bounds__(256)` only (no `waves_per_eu`) and a 20996-byte `LqFoldSmem` for `RPT=4`.[^f-decode-attr][^f-norm-attr][^f-smem] |
| **Leave** | Flipping `VLLM_RDNA_GDN_LEAPQUANT` on, serving it as a dest default, or treating it as a gfx1100 path. The switch is default off, is not in `vllm/envs.py`, and raises if the process is not RDNA2 HIP. `gdn_leapquant_rdna2.cu` is on the gfx1030 CMake list only.[^f-env][^f-cmake] |
| **Leave** | A persistent / cooperative grid. Normalize is one ordinary launch (`grid.x = N * HV`, `grid.y = layers`). The MonoKernel LQ path is the existing `gdn_fused_k` launch with a `LQ=true` template; its `atomicAdd` counters are the same per-head tile-arrival flags as the FP32 path, not a grid-wide wait.[^f-norm-grid][^f-mono-lb][^f-mono-atom] |

No dest or UNC change follows from this silicon pass.

## Gate and layout

`leapquant_gdn_enabled()` returns true only when `VLLM_RDNA_GDN_LEAPQUANT` is exactly `"1"` (default `"0"`) and `on_rdna2_hip()` is true. Otherwise it is off; a non-RDNA2-HIP process that sets the flag raises instead of falling back to FP32 state. Serving is limited to `head_k_dim == head_v_dim == 128`. Window 32, `r = 4`, `power_iters = 2`. Ring capacity is `window + max_query_len - 1` with `max_query_len = num_spec + 1`.[^f-env][^f-layout-py]

One uint8 blob per (cache block, value head), 16-byte section alignment. Sections in order:[^f-layout-cuh][^f-layout-py]

| Section | Type | Shape |
|---|---|---|
| `z` | INT8 | `[V, K]` |
| `bscale` | FP32 | `[V]` |
| `csmooth` | FP32 | `[K]` |
| `comp_k` / `comp_u` | FP16 | `[r, K]` / `[r, V]` |
| `rec_k` / `rec_u` / `rec_g` | FP16 / FP16 / FP32 | `[cap, K]` / `[cap, V]` / `[cap]` |
| `header` | INT32 | `[cursor, pending, 0, 0]` |

Dequant of the residual is `Z[v,k] * B[v] * C[k]`. The FP32 boundary is that residual plus `sum_c U~[v,c] * K~[c,k]`. Replay step `j` is `S <- exp(g_j) * S + u_j k_j^T`. A header outside `0 <= cursor`, `0 <= pending`, `cursor + pending <= cap` is treated as empty so the ring is not indexed out of range. An all-zero blob is the zero state.[^f-cu-hdr][^f-hdr-ok][^f-py-doc]

For `V = K = 128`, `r = 4`, and `cap = 34` (window 32 and a 3-token verify step) the layout formula gives 37024 bytes per (block, head). The matching FP32 state tile `[V, K]` is 65536 bytes. Those sizes follow the section list and the 16-byte align helper; they are not measured on a device.[^f-layout-cuh][^f-layout-py]

INT8 store: per-key smoothing is `max(sqrt(mean_v |R[v,k]|), 1e-6)`, then each value-row scale is `max_k |R[v,k] / C[k]| / 127`, and the residual is rounded to INT8 with a 16-byte `uint4` packed store. Compensators and records are saturated into FP16 (`±65504`).[^f-fold-quant]

## Kernels (gfx1030)

All four LeapQuant kernels live in `csrc/rocm/gdn_leapquant_rdna2.cu`, listed only under the gfx1030 `VLLM_ROCM_EXT_SRC` block. Host wrappers are `gdn_lq_normalize_rdna2`, `gdn_lq_decode_rdna2`, `gdn_lq_materialize_rdna2`, and `gdn_lq_store_rdna2`. Fold / load / normalize / store / materialize instantiate only `RPT = V/32 = 4`.[^f-cmake][^f-ops][^f-rpt]

| Kernel | Launch | Occupancy attr | LDS | Role |
|---|---|---|---|---|
| `gdn_lq_normalize_kernel` | `dim3(N * HV, num_layers)`, 256 thr | `__launch_bounds__(256)` only | `LqFoldSmem<4>` 20996 B when a window is folded | Commit accepted pending records; if `cursor >= window`, replay and fold a new boundary[^f-norm-attr][^f-norm-grid] |
| `gdn_lq_store_kernel` | `N * HV`, 256 thr | `__launch_bounds__(256)` only | same fold smem | FP32 `[N, HV, V, K]` → blob, `cursor = pending = 0`[^f-store] |
| `gdn_lq_materialize_kernel` | `N * HV`, 256 thr | `__launch_bounds__(256)` only | none | blob → FP32 `[N, HV, V, K]`[^f-mat] |
| `gdn_lq_decode_kernel` | `(V/32, N * HV)`, 256 thr | `__launch_bounds__(256)` + `amdgpu_waves_per_eu(2, 4)` | none | Decode / MTP verify: rebuild `h[16]`, run `T` tokens, append up to `cap - cursor` records, set `pending`[^f-decode-attr] |

`gdn_lq_decode_rdna2` is fp16-only (q/k/v/a/b/out). `A_log` / `dt_bias` may be fp32 or fp16. The inner step matches the existing GDN decode: L2-normalize q/k, softplus gate, scalar FP32 FMA on the 16-wide K tile, `__shfl_xor` reduce over the 8 K-slices (masks 1, 2, 4). INT8 payload is loaded as `uint4` (16 bytes) and converted to float in registers.[^f-decode-host][^f-decode-isa]

No `__gfx11__` / `__gfx950__` / WMMA / MFMA device path in these files. A search of `gdn_leapquant_rdna2.cu`, `gdn_leapquant_layout.cuh`, and the LeapQuant branches of `mono_gdn_rdna2.cu` finds no `sdot`, `fdot`, `dot2`, `dot4`, `dot8`, `wmma`, `mfma`, or bf16 packed-dot builtin.[^f-decode-isa][^f-mono-lq]

### Fold LDS

`LqFoldSmem<RPT>` (used with `RPT=4`) is:[^f-smem]

- `float red[8][128 * 4]` — 16384 B, per-wave partials
- `float kr[128 * 4]` — 2048 B
- `float vr[4 * 32 * 4]` — 2048 B
- `float csm[128]` — 512 B
- `int flag` — 4 B

Total 20996 B, under the 64 kB per-workgroup LDS cap on RDNA2.[^isa]

`lq_reduce_over_v` writes `sm.red[wave][(k0 + j) * NCOL + c]` from the eight lanes of each wave with `lane < 8`, and `k0 = (tid & 7) * 16`. For `NCOL = 4` the dword-bank index of that store is `(4j + c) % 32` for every wave, so all 64 writers of one `(j, c)` land in the same of the 32 dword LDS banks. That is an access-pattern observation from the source and the ISA bank rule, not a measured stall.[^f-reduce][^isa]

The fold itself is a fixed-order reduction plus modified Gram-Schmidt and `power_iters` subspace iterations, so a given input is bitwise reproducible run to run per the source comment. No atomics in the LeapQuant `.cu` kernels.[^f-fold-comment]

## MonoKernel and align-copy hooks

`mono_gdn_lq_rdna2` is a new host entry that launches the same `gdn_fused_k` kernel with template flag `LQ=true`. Occupancy is unchanged: `__launch_bounds__(256)`, no `waves_per_eu`, dynamic LDS `tmax * (2 * 128 + 32) * 4` bytes for the conv outputs, plus 32 floats and one int of static smem. The LQ tile rebuilds its 32 rows from the home blob (`srow[0]`) after `gdn_lq_normalize_rdna2` has committed accepted records, then appends one record per token instead of writing per-token FP32 state. Tile 0 writes `pending = T`. There is still no `this_grid` / cooperative / persistent wait; `atomicAdd` on `cnt` / `cntqk` is the existing last-tile arrival handshake inside one launch, and `atomicCAS` is only the opt-in diagnostic flag.[^f-mono-lq][^f-mono-lb][^f-mono-atom][^f-mono-lds]

`v1_mamba_rdna2.cu` treats `conv_width < 0` as a LeapQuant blob. One CTA copies the home blob, then commits `min(token_bias + 1, pending)` records in the copy's header (`cursor += that`, `pending = 0`). The header patch is ordered with `__syncthreads` after the copy, inside that CTA only.[^f-mamba]

## Sources

[^c-3e274335]: opengfx1030/vllm-rdna merge [`3e2743351c97bd3f4367a23248b0e88495a249dd`](https://github.com/opengfx1030/vllm-rdna/commit/3e2743351c97bd3f4367a23248b0e88495a249dd) (2026-11-14, 23:12:20 UTC+1) — "Merge PR #168: LeapQuant GDN recurrent state, adaptive scheduler and flashnext-leapquant-mtp2 recipe". Range from [`a63a6b0010fd4e761fb14393e32d3491f4a1460b`](https://github.com/opengfx1030/vllm-rdna/commit/a63a6b0010fd4e761fb14393e32d3491f4a1460b).
[^f-cu-hdr]: `csrc/rocm/gdn_leapquant_rdna2.cu` L1-L24 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L1-L24
[^f-layout-cuh]: `csrc/rocm/gdn_leapquant_layout.cuh` L12-L67 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_layout.cuh#L12-L67
[^f-hdr-ok]: `csrc/rocm/gdn_leapquant_layout.cuh` `lq_header_ok` L59-L67 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_layout.cuh#L59-L67
[^f-py-doc]: `vllm/model_executor/layers/mamba/ops/leapquant_gdn.py` L8-L36 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/vllm/model_executor/layers/mamba/ops/leapquant_gdn.py#L8-L36
[^f-env]: `leapquant_gdn_enabled()` L60-L74; layout constants L47-L52 and L77-L94 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. The flag is read from `os.environ` here, not from `vllm/envs.py` (no `LEAPQUANT` string in that file at this tip). `on_rdna2_hip()` is `vllm/platforms/rdna.py` L46-L54. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/vllm/model_executor/layers/mamba/ops/leapquant_gdn.py#L60-L74
[^f-layout-py]: `LeapQuantLayout._sections` / `capacity` L113-L141 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/vllm/model_executor/layers/mamba/ops/leapquant_gdn.py#L113-L141
[^f-cmake]: `CMakeLists.txt` gfx1030 `VLLM_ROCM_EXT_SRC` includes `csrc/rocm/gdn_leapquant_rdna2.cu` at L1671; the gfx1100 block L1712-L1719 does not. @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/CMakeLists.txt#L1653-L1719
[^f-ops]: `csrc/rocm/ops.h` L577-L596 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/ops.h#L577-L596
[^f-rpt]: `LQ_DISPATCH_RPT` L664-L673 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L664-L673
[^f-smem]: `LqFoldSmem` L85-L92 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. Byte counts are `sizeof(float)` times the array extents at `RPT=4`, plus 4 B for `flag`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L85-L92
[^f-reduce]: `lq_reduce_over_v` L96-L121, in particular the `lane < 8` store at L110, @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L96-L121
[^f-fold-comment]: Fold comment L78-L82 and `lq_fold` L204-L317 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L78-L82
[^f-fold-quant]: INT8 pack and scales in `lq_fold` L259-L316 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L259-L316
[^f-load]: `lq_load_full` L319-L365 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L319-L365
[^f-norm-attr]: `gdn_lq_normalize_kernel` L368-L412 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L368-L412
[^f-norm-grid]: `gdn_lq_normalize_rdna2` host launch L708-L721 (`dim3 grid((unsigned)(N * HV), (unsigned)num_layers)`) @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L708-L721
[^f-store]: `gdn_lq_store_kernel` L414-L446 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L414-L446
[^f-mat]: `gdn_lq_materialize_kernel` L448-L483 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L448-L483
[^f-decode-attr]: `gdn_lq_decode_kernel` L491-L503 (`__launch_bounds__(LQ_THREADS)` + `amdgpu_waves_per_eu(2, 4)`, `LQ_THREADS = 256`) @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L491-L503
[^f-decode-isa]: Decode body L532-L629 (INT8 `uint4` load, FP32 FMA, `lq_ksum` shfl 1/2/4; no DOT / WMMA builtin in this file) @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L532-L629
[^f-decode-host]: `gdn_lq_decode_rdna2` L777-L858 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/gdn_leapquant_rdna2.cu#L777-L858
[^f-mono-lq]: `mono_gdn_rdna2.cu` LeapQuant header L32-L38, `LQ` path L237-L239 and L286-L436, host `mono_gdn_lq_rdna2` L669-L686 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/mono_gdn_rdna2.cu#L32-L38
[^f-mono-lb]: `gdn_fused_k` `__launch_bounds__(kThreads)` L203-L204, `kThreads = 256` L53, @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/mono_gdn_rdna2.cu#L203-L204
[^f-mono-atom]: `atomicCAS` diagnostic L105; `atomicAdd` `cntqk` / `cnt` L271-L276 and L440-L443 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/mono_gdn_rdna2.cu#L271-L276
[^f-mono-lds]: Dynamic LDS size L625 (`tmax * (2 * kK + kBV) * sizeof(float)`) and static `s_red` / `s_own` L248-L252 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/mono_gdn_rdna2.cu#L247-L252
[^f-mamba]: `v1_mamba_rdna2.cu` LeapQuant copy (`conv_width < 0`) L129-L132 and L222-L258 @ `3e2743351c97bd3f4367a23248b0e88495a249dd`. https://github.com/opengfx1030/vllm-rdna/blob/3e2743351c97bd3f4367a23248b0e88495a249dd/csrc/rocm/v1_mamba_rdna2.cu#L222-L258
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
