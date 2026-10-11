---
title: "Qwen3.8-Flash-Next / Qwen4Exp HIP — tip `8960a3bc`"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/50120e13b46895fc5d074e88fc493f1eef69da9f
  - https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/rdna_fused_glue.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/hc_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/qsa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/ple_short_conv_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/mrope_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/rdna_fused_glue.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/rdna_fused_glue.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/hc_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/d0d577f1907c45df4243c210a34d254fba626ef4
  - https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/csrc/rocm/hc_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/moe_q_gemm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/vllm/_custom_ops.py
  - https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/csrc/rocm/torch_bindings.cpp
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/hc_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/commit/60ee765f3848730fc08dc929575af85fbde8515d
  - https://github.com/opengfx1030/vllm-rdna/commit/121af1b4786feb60748afe2fd73c2a49d459dc8b
  - https://github.com/opengfx1030/vllm-rdna/blob/60ee765f3848730fc08dc929575af85fbde8515d/csrc/rocm/hc_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/60ee765f3848730fc08dc929575af85fbde8515d/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py
  - https://github.com/opengfx1030/vllm-rdna/commit/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af
  - https://github.com/opengfx1030/vllm-rdna/commit/e3bf3d338fb5cf51322a1d659808bf6c9f649cf3
  - https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/qsa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/d19b87ea6c43dbb254d089a64affaf5e465de6fa
  - https://github.com/opengfx1030/vllm-rdna/commit/c5fd67ef63b1d25f14c2acdf469bd25bd66aed2a
  - https://github.com/opengfx1030/vllm-rdna/commit/a6ab43cf9661935e05f2a601b8fc1336d6c42491
  - https://github.com/opengfx1030/vllm-rdna/blob/d19b87ea6c43dbb254d089a64affaf5e465de6fa/csrc/rocm/qsa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/csrc/rocm/qsa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/ab5ccf3d9f3445679169f2a9e6152bd777d20413
  - https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/csrc/rocm/skinny_gemms.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/vllm/v1/core/sched/dynamic_prefill.py
  - https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/ple_short_conv_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/ple_short_conv_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/mrope_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/layernorm.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/CMakeLists.txt
---
# Qwen3.8-Flash-Next / Qwen4Exp HIP — tip `8960a3bc`

Dest: `opengfx1030/vllm-rdna` `rdna_extras` @ **`50120e13`** (2026-09-17; HC capture-safe `_contig` on top of `8960a3bc`). Flash-Next / Qwen4Exp gfx1030 HIP opt-in paths. No tok/s. Occupancy still first.[^c-50120e13][^c-8960a3bc]

CMake gfx1030 `EXT_SRC` includes: `rdna_fused_glue.cu`, `hc_rdna2.cu`, `qsa_rdna2.cu`, `ple_short_conv_rdna2.cu`, `mrope_rdna2.cu` (plus prior list). Unchanged by this tip.[^f-8960a3bc-rdna-fused-glue-cu][^f-8960a3bc-hc-rdna2-cu][^f-8960a3bc-qsa-rdna2-cu][^f-8960a3bc-ple-short-conv-rdna2-cu][^f-8960a3bc-mrope-rdna2-cu]

## Take (silicon)

### T46 `rdna_fused_glue.cu` — fused decode glue (M ≤ 8)

- Explicit `__builtin_amdgcn_fdot2` on fp16 and int8→half2 weight rows (wave32 lanes stride K).[^f-c6d99ca3-rdna-fused-glue-cu][^f-50120e13-rdna-fused-glue-cu]
- `__launch_bounds__(WAVES * 32)`; tiny `__shared__ float s_gate[kMaxM]` on one kernel.[^f-c6d99ca3-rdna-fused-glue-cu][^f-50120e13-rdna-fused-glue-cu]
- Opt-in: `VLLM_RDNA_FUSED_HC=1` + `on_gfx10x()`. Default off.[^f-c6d99ca3-envs-py]
- No WMMA. Real DOT path for Flash-Next decode glue.[^f-c6d99ca3-rdna-fused-glue-cu][^f-50120e13-rdna-fused-glue-cu]

### T47 `hc_rdna2.cu` — HyperConnection prefill glue

- Five elementwise/affine kernels (grouped Gemma RMSNorm, silu scaler, gate mix, combine, combine+norm).[^f-c6d99ca3-hc-rdna2-cu][^f-8960a3bc-hc-rdna2-cu]
- vec8 fp16 (uint4) loads; scalar/half2 FMA only — **no fdot2**.[^f-c6d99ca3-hc-rdna2-cu][^f-8960a3bc-hc-rdna2-cu]
- Opt-in: `VLLM_RDNA_HC_PREFILL_HIP=1` + `on_gfx10x()`. Default off (Triton).[^f-c6d99ca3-envs-py]

#### extras lock 2026-09-16 (tip `8960a3bc`, was `d0d577f1`)

HC HIP chain compute-correct for Flash-Next production shape. Six defects cleared in `csrc/rocm/hc_rdna2.cu` + `vllm/models/qwen4_exp/amd/ops/hc_rdna2.py`. Prior tip `d0d577f1` only added the missing `_custom_ops` wrappers (Python); this tip is the first real HIP fix.[^c-d0d577f1][^f-d0d577f1-hc-rdna2-cu][^f-d0d577f1-hc-rdna2-py]

| Surface | Delta |
|---|---|
| `grouped_gemma_rmsnorm` | Loops bound by runtime `GROUP_DIM`, not `BLOCK` template. Flash-Next needs `GROUP_DIM=2560` (hidden 2560, `hc_count` 4). Host no longer `TORCH_CHECK` rejects `GROUP_DIM > 512`; launches `<512>` as unroll hint only. [^f-8960a3bc-hc-rdna2-cu] |
| `hc_silu` | Same pattern: loop over runtime `DIM`; `DIM > 2048` works. [^f-8960a3bc-hc-rdna2-cu] |
| `hc_combine_norm` | Drop `float local[BLOCK]` sized to group width. Walk `HC_DIM` in `BLOCK`-wide chunks; pass 2 re-reads rounded values from `out` (avoids spill at width 2560). [^f-8960a3bc-hc-rdna2-cu] |
| Weight index (3 sites) | `w_base` is already group-offset. Unshared path must use `w_base[b]`, not `w_base[(W_SHARED ? b : off)]` (was `w[2*base+b]` OOB; fault landed downstream in MoE). Fixed in `combine_norm` + `grouped_gemma_rmsnorm` ×2. [^f-8960a3bc-hc-rdna2-cu] |
| `hc_rdna2.py` | `_contig()` on all five call sites (HIP needs contiguous fp16; callers pass strided `split()` views). Outs: `new_empty` → `new_zeros` (gfx1030 page-commit; EXL3 / GDN / hc_combine freeze class). [^f-8960a3bc-hc-rdna2-py][^f-8960a3bc-hc-rdna2-cu] |

Author parity vs Triton at `N=4, DIM=10240, hc_count=4, HC_DIM=2560, W_SHARED=0`: grouped_gemma_rmsnorm / combine_norm / silu / gate_mix / combine all at fp16 noise. That is the commit author's claim — not a measurement of ours. Do not invent numbers.[^f-8960a3bc-hc-rdna2-cu]

**Leave gate default-off.** With `VLLM_RDNA_HC_PREFILL_HIP=1`, server still faults during PIECEWISE capture in the MoE path (`moe_gemm_q4_kernel_rdna2` or `moe_align_block_size_kernel`, order varies). HC ops correct in isolation; not proven capture-safe in context. See [silicon/hip-craft/graph-capture.md](../../silicon/hip-craft/graph-capture.md).[^f-8960a3bc-moe-q-gemm-rdna2-cu][^f-8960a3bc-envs-py]

Wrappers from `d0d577f1` remain: `hc_{grouped_gemma_rmsnorm,silu,gate_mix,combine,combine_norm}_rdna2` in `_custom_ops.py` matching `torch_bindings.cpp`. Without them the path was inert (`AttributeError`).[^c-d0d577f1][^f-d0d577f1--custom-ops-py][^f-d0d577f1-torch-bindings-cpp][^f-d0d577f1-hc-rdna2-cu]

#### extras lock 2026-09-17 (tip `50120e13`, was `8960a3bc`)

`hc_rdna2.py` `_contig()`: per-call `.contiguous()` allocated a temp whose address was recorded under cudagraph capture and freed before replay (stale reads). Now caches one `torch.empty` buffer per `(shape, dtype, device)` and `copy_` into it. Gate `VLLM_RDNA_HC_PREFILL_HIP` still default-off (path inert in validated serve). Caveat: same-shape live values in one step can clobber; prefer per-call-site persistent buffers later (EXL3 CG-PATH style). No `csrc/rocm/hc_rdna2.cu` / launch_bounds / DOT / CMake change.[^f-50120e13-hc-rdna2-py][^f-50120e13-hc-rdna2-cu][^f-50120e13-envs-py]

#### extras lock 2026-10-06 (tip `c6d99ca3`, was `121af1b4`)

First HIP-side HC fix since `8960a3bc`. Root cause of `VLLM_RDNA_HC_PREFILL_HIP=1` emitting garbage on gfx1030 (author: server reported success while output was a run of `!`). Two independent defects; no math, tile, DOT or LDS change.[^c-8960a3bc][^f-8960a3bc-envs-py]

| Surface | Delta |
|---|---|
| `csrc/rocm/hc_rdna2.cu` launch sites | Every `grouped_gemma_rmsnorm` / `hc_silu` / `hc_gate_mix` / `hc_combine` / `hc_combine_norm` launch was bare `<<<grid, block>>>` = legacy null stream on ROCm, so it did not order against torch's current stream and raced tensors the forward was still writing. Now `<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>` + `C10_CUDA_KERNEL_LAUNCH_CHECK()` in all five wrappers. Same class as the QSA stream fix (`60ee765f`, locked at `121af1b4` above) and T44b / wvSplitK. Block sizes unchanged (128; `hc_silu<2048>` 256). [^c-60ee765f][^c-121af1b4][^f-60ee765f-hc-rdna2-cu][^f-60ee765f-skinny-gemms-cu] |
| `hc_rdna2.py` `_contig()` | Cache key is now `(shape, dtype, device, data_ptr)`. Keying on shape alone aliased two distinct same-shaped strided views onto one buffer, so a kernel reading two inputs saw the second tensor twice. This closes the 2026-09-17 caveat above. Each source still gets a stable buffer address across cudagraph capture and replay. [^f-c6d99ca3-hc-rdna2-py] |

Take (Now): no bare `<<<>>>` launch in any extras HIP op; stream + launch check is the contract. Contiguity scratch caches must be keyed per source pointer, not per shape.[^c-c6d99ca3][^c-121af1b4]

Watch / Later (hunch, not measured): none of the five HC kernels carries `__launch_bounds__` (launches are 128 / 256 threads), same note as QSA. Check VGPR in the ISA before any default-on flip.[^c-c6d99ca3][^c-121af1b4]

Leave: author tok/s and the HIP-vs-Triton gap. Gate stays **default-off**: author reports HC HIP now correct but much slower than the Triton baseline. The same tip's scheduler change (`e3bf3d33`, upstream mixed decode+prefill batching restored, `VLLM_ROCM_NO_MIXED_BATCH` removed from `serve_rdna.sh` / `full.env`) is engine-side Python, not silicon — see `engine/`.[^c-e3bf3d33]


### T48 `qsa_rdna2.cu` — QSA decode glue

- `qsa_store_cache_rows` / `qsa_compress_groups` scalar; MQA host wrapper reuses existing `paged_mqa_logits_decode_rdna2` when layout matches.[^f-c6d99ca3-qsa-rdna2-cu]
- Opt-in: `VLLM_RDNA_QSA_HIP=1` + `on_gfx10x()`. Default off.[^f-c6d99ca3-envs-py]

#### extras lock 2026-10-05 (tip `121af1b4`, was `ab5ccf3d`)

QSA HIP path landed on dest (`d19b87ea` v0.30 port, `c5fd67ef` stride fix, merge `a6ab43cf`). Still opt-in, **default off** (author reports it slower than the Triton QSA reference; Triton stays source of truth). Correctness and launch-hygiene lock only — no `__launch_bounds__`, DOT, LDS-tile or CMake gfx1030 change (`qsa_rdna2.cu` was already in `EXT_SRC`).[^c-d19b87ea][^c-c5fd67ef][^c-a6ab43cf][^f-d19b87ea-qsa-rdna2-cu]

| Surface | Delta |
|---|---|
| `qsa_store_cache_rows_kernel` (QSA KV store) | Vectorised `uint4` copy now gated on **both** `stride_rows_dim == elem_bytes` and `stride_cache_dim == elem_bytes`; otherwise stride-aware scalar loop. Root bug: server passes the transposed `canonical_qsa_rope_positions` view (last-dim stride = tokens), and the dense-16 B read walked off the row → page-not-present fault under chunked prefill + MTP + prefix caching. Invalid slot now early-returns the whole CTA. [^f-121af1b4-qsa-rdna2-cu] |
| dtypes | Slots, logical/raw positions, compressed slots, rope ring and `first_positions` are **int64** end to end (were int32). Store kernel handles 2/4/8-byte elems (fp16 rows and int64 rope rows share one kernel). Strides carried as int64 bytes. [^c-121af1b4][^c-ab5ccf3d] |
| Cache rank | Cache/compressor-state/rope are 4-D `[blocks, PAGE, 1, W]`; host reads `stride(3)` for the width dim (was `stride(2)`). `raw_keys`/`pooled` keep the head=1 dim (`stride(2)`), no `squeeze().contiguous()` copy in the dispatcher. [^c-121af1b4][^c-ab5ccf3d] |
| Launch hygiene | Both kernels launch on `getCurrentCUDAStream()` (were on the legacy default stream) + `C10_CUDA_KERNEL_LAUNCH_CHECK()`. Same class as the T44b / wvSplitK stream fixes: a HIP op off the torch stream is a capture/order bug, not a perf knob. [^f-121af1b4-skinny-gemms-cu] |
| `qsa_compress_groups` | Row count clamped to the shortest metadata buffer (warns once per call via host `printf`). Fallback "neither raw nor ring" first-position now writes 0 (matches Triton) instead of `first_position`. [^f-121af1b4-qsa-rdna2-cu] |
| Op schema | Scalar ints (`page_size`, `width`, `compress_ratio`, `compressor_state_size`, `head_dim`, `max_model_len`) now passed as CPU int64 Tensors and read with `.item()` host-side. CPU tensor ⇒ no D2H sync, but a per-call host alloc. [^c-121af1b4][^c-ab5ccf3d] |
| Debug | `VLLM_QSA_RDNA2_DEBUG` adds host prints and an in-kernel `printf` branch in `qsa_compress_groups_kernel` (runtime `DEBUG` int). Python debug path calls `torch.cuda.synchronize()` — never on in serve or under capture. [^f-121af1b4-qsa-rdna2-cu] |

Take (Now): stride-dense gate before any `uint4` vector copy on a view we do not own; int64 slot/position contract for QSA; current-stream launch + launch check on every new HIP op.[^c-121af1b4][^c-ab5ccf3d]

Watch / Later (hunches, not measured): neither QSA kernel has `__launch_bounds__` (launches are 256 and 128 threads), so the compiler budgets registers for 1024-thread blocks; and the device `printf` compiled into `qsa_compress_groups_kernel` costs code size / registers even when `DEBUG=0`. Check VGPR/scratch in the ISA before any default-on flip. Not an occupancy-first item — FA prefill leftover still leads.[^f-121af1b4-qsa-rdna2-cu]

Leave: author tok/s and the HIP-vs-Triton gap figures. The same tip's adaptive prefill scheduler (`vllm/v1/core/sched/dynamic_prefill.py`, opt-in `VLLM_RDNA_DYNAMIC_PREFILL=1`) is engine-side Python, not silicon — see `engine/`.[^f-121af1b4-dynamic-prefill-py]

### T49 `ple_short_conv_rdna2.cu` — dilated PLE short-conv

- Depthwise dilated conv1d decode+prefill; per-channel threads; `history_buf[64]` in regs; scalar FMA + silu.[^f-c6d99ca3-ple-short-conv-rdna2-cu][^f-50120e13-ple-short-conv-rdna2-cu]
- Opt-in: `VLLM_RDNA_PLE_CONV_HIP=1` + `on_gfx10x()`. Default off (F.conv1d).[^f-121af1b4-envs-py]

### `mrope_forward_rdna2` build fixes

- Already room-locked Take (Qwen M-RoPE bind). Tip adds CUDAGuard include + host wrapper at global scope so `torch_bindings` resolves. Still one CTA/token, scalar, no fdot2.[^f-50120e13-mrope-rdna2-cu]

## Leave

- Do not treat fused_glue / HC / QSA / PLE defaults as on — `VLLM_RDNA_FUSED_HC`, `VLLM_RDNA_HC_PREFILL_HIP`, `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_PLE_CONV_HIP` are all env-gated opt-in (default off) until soak / capture-safe.[^f-c6d99ca3-envs-py]
- Do not flip `VLLM_RDNA_HC_PREFILL_HIP=1` on serve until MoE PIECEWISE capture fault is resolved (HC alone is not enough).[^f-c6d99ca3-envs-py]
- No transplant of HC/PLE onto GLM KDA / norm-after-gate (same split as gated_rms_norm / causal_conv).[^f-c6d99ca3-layernorm-cu]
- No WMMA/FP8 objects; gfx1030 stays fdot2 / EXL3.[^c-50120e13][^f-50120e13-CMakeLists-txt]
- No `__launch_bounds__` / DOT / LDS-tile / KV-quant / CMake gfx1030 list change in `8960a3bc` or `50120e13` (HC lock is Python dispatcher only).[^c-8960a3bc][^c-50120e13]
- Do not flip `VLLM_RDNA_HC_PREFILL_HIP=1` on serve on perf grounds even after `c6d99ca3` fixed its stream race + `_contig` aliasing (correct, still slower than Triton). No `__launch_bounds__` / DOT / LDS-tile / KV-quant / CMake change in `c6d99ca3`.[^c-c6d99ca3][^f-c6d99ca3-envs-py]
- Do not flip `VLLM_RDNA_QSA_HIP=1` on serve: `121af1b4` fixes the store fault but the path is still default-off (Triton QSA is the reference). No `__launch_bounds__` / DOT / LDS-tile / CMake change in `121af1b4`.[^c-121af1b4][^f-121af1b4-envs-py]

## Occupancy

FA prefill leftover still first (`__launch_bounds__(*, 1)`). ticket-26 EXL3 `-cb 3inst` unchanged.[^c-50120e13][^f-50120e13-CMakeLists-txt]

## Sources

[^c-50120e13]: opengfx1030/vllm-rdna commit [`50120e13`](https://github.com/opengfx1030/vllm-rdna/commit/50120e13b46895fc5d074e88fc493f1eef69da9f) (2026-09-17) — "fix(rocm,hc): make the HC HIP dispatcher's _contig() capture-safe".
[^c-8960a3bc]: opengfx1030/vllm-rdna commit [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/commit/8960a3bcbb175e8e74efd2f6c6b37b597972debc) (2026-09-16) — "fix(rocm): make the Qwen4Exp HC HIP chain compute correctly".
[^f-8960a3bc-rdna-fused-glue-cu]: opengfx1030/vllm-rdna `csrc/rocm/rdna_fused_glue.cu` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/rdna_fused_glue.cu).
[^f-8960a3bc-hc-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/hc_rdna2.cu` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/hc_rdna2.cu).
[^f-8960a3bc-qsa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/qsa_rdna2.cu` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/qsa_rdna2.cu).
[^f-8960a3bc-ple-short-conv-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/ple_short_conv_rdna2.cu` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/ple_short_conv_rdna2.cu).
[^f-8960a3bc-mrope-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/mrope_rdna2.cu` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/mrope_rdna2.cu).
[^f-c6d99ca3-rdna-fused-glue-cu]: opengfx1030/vllm-rdna `csrc/rocm/rdna_fused_glue.cu` @ [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/rdna_fused_glue.cu).
[^f-50120e13-rdna-fused-glue-cu]: opengfx1030/vllm-rdna `csrc/rocm/rdna_fused_glue.cu` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/rdna_fused_glue.cu).
[^f-c6d99ca3-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/vllm/envs.py).
[^f-c6d99ca3-hc-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/hc_rdna2.cu` @ [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/hc_rdna2.cu).
[^c-d0d577f1]: opengfx1030/vllm-rdna commit [`d0d577f1`](https://github.com/opengfx1030/vllm-rdna/commit/d0d577f1907c45df4243c210a34d254fba626ef4) (2026-09-16) — "fix(rocm): add the 10 missing Qwen4Exp _custom_ops wrappers".
[^f-d0d577f1-hc-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/hc_rdna2.cu` @ [`d0d577f1`](https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/csrc/rocm/hc_rdna2.cu).
[^f-d0d577f1-hc-rdna2-py]: opengfx1030/vllm-rdna `vllm/models/qwen4_exp/amd/ops/hc_rdna2.py` @ [`d0d577f1`](https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py).
[^f-8960a3bc-hc-rdna2-py]: opengfx1030/vllm-rdna `vllm/models/qwen4_exp/amd/ops/hc_rdna2.py` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py).
[^f-8960a3bc-moe-q-gemm-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/moe_q_gemm_rdna2.cu` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/csrc/rocm/moe_q_gemm_rdna2.cu).
[^f-8960a3bc-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`8960a3bc`](https://github.com/opengfx1030/vllm-rdna/blob/8960a3bcbb175e8e74efd2f6c6b37b597972debc/vllm/envs.py).
[^f-d0d577f1--custom-ops-py]: opengfx1030/vllm-rdna `vllm/_custom_ops.py` @ [`d0d577f1`](https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/vllm/_custom_ops.py).
[^f-d0d577f1-torch-bindings-cpp]: opengfx1030/vllm-rdna `csrc/rocm/torch_bindings.cpp` @ [`d0d577f1`](https://github.com/opengfx1030/vllm-rdna/blob/d0d577f1907c45df4243c210a34d254fba626ef4/csrc/rocm/torch_bindings.cpp).
[^f-50120e13-hc-rdna2-py]: opengfx1030/vllm-rdna `vllm/models/qwen4_exp/amd/ops/hc_rdna2.py` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py).
[^f-50120e13-hc-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/hc_rdna2.cu` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/hc_rdna2.cu).
[^f-50120e13-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/vllm/envs.py).
[^c-60ee765f]: opengfx1030/vllm-rdna commit [`60ee765f`](https://github.com/opengfx1030/vllm-rdna/commit/60ee765f3848730fc08dc929575af85fbde8515d) (2026-09-21) — "fix(rdna2): launch QSA kernels on the current stream — prefix-cache prefill faults".
[^c-121af1b4]: opengfx1030/vllm-rdna commit [`121af1b4`](https://github.com/opengfx1030/vllm-rdna/commit/121af1b4786feb60748afe2fd73c2a49d459dc8b) (2026-10-04) — "Merge remote-tracking branch 'pr/36-head' into rdna_extras".
[^f-60ee765f-hc-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/hc_rdna2.cu` @ [`60ee765f`](https://github.com/opengfx1030/vllm-rdna/blob/60ee765f3848730fc08dc929575af85fbde8515d/csrc/rocm/hc_rdna2.cu).
[^f-60ee765f-skinny-gemms-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms.cu` @ [`60ee765f`](https://github.com/opengfx1030/vllm-rdna/blob/60ee765f3848730fc08dc929575af85fbde8515d/csrc/rocm/skinny_gemms.cu).
[^f-c6d99ca3-hc-rdna2-py]: opengfx1030/vllm-rdna `vllm/models/qwen4_exp/amd/ops/hc_rdna2.py` @ [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/vllm/models/qwen4_exp/amd/ops/hc_rdna2.py).
[^c-c6d99ca3]: opengfx1030/vllm-rdna commit [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/commit/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af) (2026-10-06) — "fix(rdna2): HC HIP stream correctness + MoE prefill tile".
[^c-e3bf3d33]: opengfx1030/vllm-rdna commit [`e3bf3d33`](https://github.com/opengfx1030/vllm-rdna/commit/e3bf3d338fb5cf51322a1d659808bf6c9f649cf3) (2026-10-05) — "feat(sched): restore upstream mixed decode+prefill batching".
[^f-c6d99ca3-qsa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/qsa_rdna2.cu` @ [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/qsa_rdna2.cu).
[^c-d19b87ea]: opengfx1030/vllm-rdna commit [`d19b87ea`](https://github.com/opengfx1030/vllm-rdna/commit/d19b87ea6c43dbb254d089a64affaf5e465de6fa) (2026-10-02) — "port: QSA RDNA2 HIP decode path (v0.30 fix files onto rdna_extras)".
[^c-c5fd67ef]: opengfx1030/vllm-rdna commit [`c5fd67ef`](https://github.com/opengfx1030/vllm-rdna/commit/c5fd67ef63b1d25f14c2acdf469bd25bd66aed2a) (2026-10-02) — "fix(rdna2): respect row/cache width strides in qsa_store_cache_rows_kernel".
[^c-a6ab43cf]: opengfx1030/vllm-rdna commit [`a6ab43cf`](https://github.com/opengfx1030/vllm-rdna/commit/a6ab43cf9661935e05f2a601b8fc1336d6c42491) (2026-10-02) — "merge(rdna2): land opt-in QSA RDNA2 HIP path (v0.30 port + stride fix; default off)".
[^f-d19b87ea-qsa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/qsa_rdna2.cu` @ [`d19b87ea`](https://github.com/opengfx1030/vllm-rdna/blob/d19b87ea6c43dbb254d089a64affaf5e465de6fa/csrc/rocm/qsa_rdna2.cu).
[^f-121af1b4-qsa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/qsa_rdna2.cu` @ [`121af1b4`](https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/csrc/rocm/qsa_rdna2.cu).
[^c-ab5ccf3d]: opengfx1030/vllm-rdna commit [`ab5ccf3d`](https://github.com/opengfx1030/vllm-rdna/commit/ab5ccf3d9f3445679169f2a9e6152bd777d20413) (2026-10-02) — "docs(rdna): sweep not-FA/not-HIP gaps — verdicts + PR#33 recovery".
[^f-121af1b4-skinny-gemms-cu]: opengfx1030/vllm-rdna `csrc/rocm/skinny_gemms.cu` @ [`121af1b4`](https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/csrc/rocm/skinny_gemms.cu).
[^f-121af1b4-dynamic-prefill-py]: opengfx1030/vllm-rdna `vllm/v1/core/sched/dynamic_prefill.py` @ [`121af1b4`](https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/vllm/v1/core/sched/dynamic_prefill.py).
[^f-c6d99ca3-ple-short-conv-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/ple_short_conv_rdna2.cu` @ [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/ple_short_conv_rdna2.cu).
[^f-50120e13-ple-short-conv-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/ple_short_conv_rdna2.cu` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/ple_short_conv_rdna2.cu).
[^f-121af1b4-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`121af1b4`](https://github.com/opengfx1030/vllm-rdna/blob/121af1b4786feb60748afe2fd73c2a49d459dc8b/vllm/envs.py).
[^f-50120e13-mrope-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/mrope_rdna2.cu` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/csrc/rocm/mrope_rdna2.cu).
[^f-c6d99ca3-layernorm-cu]: opengfx1030/vllm-rdna `csrc/rocm/layernorm.cu` @ [`c6d99ca3`](https://github.com/opengfx1030/vllm-rdna/blob/c6d99ca30580f0ce6a97a168fa53bf9c6ee6b2af/csrc/rocm/layernorm.cu).
[^f-50120e13-CMakeLists-txt]: opengfx1030/vllm-rdna `CMakeLists.txt` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/CMakeLists.txt).
