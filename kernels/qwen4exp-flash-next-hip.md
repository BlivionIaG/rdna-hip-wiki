# Qwen3.8-Flash-Next / Qwen4Exp HIP — tip `8960a3bc`

Dest: `opengfx1030/vllm-rdna` `rdna_extras` @ **`50120e13`** (2026-09-17; HC capture-safe `_contig` on top of `8960a3bc`). Flash-Next / Qwen4Exp gfx1030 HIP opt-in paths. No tok/s. Occupancy still first.

CMake gfx1030 `EXT_SRC` includes: `rdna_fused_glue.cu`, `hc_rdna2.cu`, `qsa_rdna2.cu`, `ple_short_conv_rdna2.cu`, `mrope_rdna2.cu` (plus prior list). Unchanged by this tip.

## Take (silicon)

### T46 `rdna_fused_glue.cu` — fused decode glue (M ≤ 8)

- Explicit `__builtin_amdgcn_fdot2` on fp16 and int8→half2 weight rows (wave32 lanes stride K).
- `__launch_bounds__(WAVES * 32)`; tiny `__shared__ float s_gate[kMaxM]` on one kernel.
- Opt-in: `VLLM_RDNA_FUSED_HC=1` + `on_gfx10x()`. Default off.
- No WMMA. Real DOT path for Flash-Next decode glue.

### T47 `hc_rdna2.cu` — HyperConnection prefill glue

- Five elementwise/affine kernels (grouped Gemma RMSNorm, silu scaler, gate mix, combine, combine+norm).
- vec8 fp16 (uint4) loads; scalar/half2 FMA only — **no fdot2**.
- Opt-in: `VLLM_RDNA_HC_PREFILL_HIP=1` + `on_gfx10x()`. Default off (Triton).

#### extras lock 2026-09-16 (tip `8960a3bc`, was `d0d577f1`)

HC HIP chain compute-correct for Flash-Next production shape. Six defects cleared in `csrc/rocm/hc_rdna2.cu` + `vllm/models/qwen4_exp/amd/ops/hc_rdna2.py`. Prior tip `d0d577f1` only added the missing `_custom_ops` wrappers (Python); this tip is the first real HIP fix.

| Surface | Delta |
|---|---|
| `grouped_gemma_rmsnorm` | Loops bound by runtime `GROUP_DIM`, not `BLOCK` template. Flash-Next needs `GROUP_DIM=2560` (hidden 2560, `hc_count` 4). Host no longer `TORCH_CHECK` rejects `GROUP_DIM > 512`; launches `<512>` as unroll hint only. |
| `hc_silu` | Same pattern: loop over runtime `DIM`; `DIM > 2048` works. |
| `hc_combine_norm` | Drop `float local[BLOCK]` sized to group width. Walk `HC_DIM` in `BLOCK`-wide chunks; pass 2 re-reads rounded values from `out` (avoids spill at width 2560). |
| Weight index (3 sites) | `w_base` is already group-offset. Unshared path must use `w_base[b]`, not `w_base[(W_SHARED ? b : off)]` (was `w[2*base+b]` OOB; fault landed downstream in MoE). Fixed in `combine_norm` + `grouped_gemma_rmsnorm` ×2. |
| `hc_rdna2.py` | `_contig()` on all five call sites (HIP needs contiguous fp16; callers pass strided `split()` views). Outs: `new_empty` → `new_zeros` (gfx1030 page-commit; EXL3 / GDN / hc_combine freeze class). |

Author parity vs Triton at `N=4, DIM=10240, hc_count=4, HC_DIM=2560, W_SHARED=0`: grouped_gemma_rmsnorm / combine_norm / silu / gate_mix / combine all at fp16 noise. That is the commit author's claim — not a measurement of ours. Do not invent numbers.

**Leave gate default-off.** With `VLLM_RDNA_HC_PREFILL_HIP=1`, server still faults during PIECEWISE capture in the MoE path (`moe_gemm_q4_kernel_rdna2` or `moe_align_block_size_kernel`, order varies). HC ops correct in isolation; not proven capture-safe in context. See [../silicon/graph-capture.md](../silicon/graph-capture.md).

Wrappers from `d0d577f1` remain: `hc_{grouped_gemma_rmsnorm,silu,gate_mix,combine,combine_norm}_rdna2` in `_custom_ops.py` matching `torch_bindings.cpp`. Without them the path was inert (`AttributeError`).

#### extras lock 2026-09-17 (tip `50120e13`, was `8960a3bc`)

`hc_rdna2.py` `_contig()`: per-call `.contiguous()` allocated a temp whose address was recorded under cudagraph capture and freed before replay (stale reads). Now caches one `torch.empty` buffer per `(shape, dtype, device)` and `copy_` into it. Gate `VLLM_RDNA_HC_PREFILL_HIP` still default-off (path inert in validated serve). Caveat: same-shape live values in one step can clobber; prefer per-call-site persistent buffers later (EXL3 CG-PATH style). No `csrc/rocm/hc_rdna2.cu` / launch_bounds / DOT / CMake change.

#### extras lock 2026-10-06 (tip `c6d99ca3`, was `121af1b4`)

First HIP-side HC fix since `8960a3bc`. Root cause of `VLLM_RDNA_HC_PREFILL_HIP=1` emitting garbage on gfx1030 (author: server reported success while output was a run of `!`). Two independent defects; no math, tile, DOT or LDS change.

| Surface | Delta |
|---|---|
| `csrc/rocm/hc_rdna2.cu` launch sites | Every `grouped_gemma_rmsnorm` / `hc_silu` / `hc_gate_mix` / `hc_combine` / `hc_combine_norm` launch was bare `<<<grid, block>>>` = legacy null stream on ROCm, so it did not order against torch's current stream and raced tensors the forward was still writing. Now `<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>` + `C10_CUDA_KERNEL_LAUNCH_CHECK()` in all five wrappers. Same class as the QSA stream fix (`60ee765f`, locked at `121af1b4` above) and T44b / wvSplitK. Block sizes unchanged (128; `hc_silu<2048>` 256). |
| `hc_rdna2.py` `_contig()` | Cache key is now `(shape, dtype, device, data_ptr)`. Keying on shape alone aliased two distinct same-shaped strided views onto one buffer, so a kernel reading two inputs saw the second tensor twice. This closes the 2026-09-17 caveat above. Each source still gets a stable buffer address across cudagraph capture and replay. |

Take (Now): no bare `<<<>>>` launch in any extras HIP op; stream + launch check is the contract. Contiguity scratch caches must be keyed per source pointer, not per shape.

Watch / Later (hunch, not measured): none of the five HC kernels carries `__launch_bounds__` (launches are 128 / 256 threads), same note as QSA. Check VGPR in the ISA before any default-on flip.

Leave: author tok/s and the HIP-vs-Triton gap. Gate stays **default-off**: author reports HC HIP now correct but much slower than the Triton baseline. The same tip's scheduler change (`e3bf3d33`, upstream mixed decode+prefill batching restored, `VLLM_ROCM_NO_MIXED_BATCH` removed from `serve_rdna.sh` / `full.env`) is engine-side Python, not silicon — see `engine/`.


### T48 `qsa_rdna2.cu` — QSA decode glue

- `qsa_store_cache_rows` / `qsa_compress_groups` scalar; MQA host wrapper reuses existing `paged_mqa_logits_decode_rdna2` when layout matches.
- Opt-in: `VLLM_RDNA_QSA_HIP=1` + `on_gfx10x()`. Default off.

#### extras lock 2026-10-05 (tip `121af1b4`, was `ab5ccf3d`)

QSA HIP path landed on dest (`d19b87ea` v0.30 port, `c5fd67ef` stride fix, merge `a6ab43cf`). Still opt-in, **default off** (author reports it slower than the Triton QSA reference; Triton stays source of truth). Correctness and launch-hygiene lock only — no `__launch_bounds__`, DOT, LDS-tile or CMake gfx1030 change (`qsa_rdna2.cu` was already in `EXT_SRC`).

| Surface | Delta |
|---|---|
| `qsa_store_cache_rows_kernel` (QSA KV store) | Vectorised `uint4` copy now gated on **both** `stride_rows_dim == elem_bytes` and `stride_cache_dim == elem_bytes`; otherwise stride-aware scalar loop. Root bug: server passes the transposed `canonical_qsa_rope_positions` view (last-dim stride = tokens), and the dense-16 B read walked off the row → page-not-present fault under chunked prefill + MTP + prefix caching. Invalid slot now early-returns the whole CTA. |
| dtypes | Slots, logical/raw positions, compressed slots, rope ring and `first_positions` are **int64** end to end (were int32). Store kernel handles 2/4/8-byte elems (fp16 rows and int64 rope rows share one kernel). Strides carried as int64 bytes. |
| Cache rank | Cache/compressor-state/rope are 4-D `[blocks, PAGE, 1, W]`; host reads `stride(3)` for the width dim (was `stride(2)`). `raw_keys`/`pooled` keep the head=1 dim (`stride(2)`), no `squeeze().contiguous()` copy in the dispatcher. |
| Launch hygiene | Both kernels launch on `getCurrentCUDAStream()` (were on the legacy default stream) + `C10_CUDA_KERNEL_LAUNCH_CHECK()`. Same class as the T44b / wvSplitK stream fixes: a HIP op off the torch stream is a capture/order bug, not a perf knob. |
| `qsa_compress_groups` | Row count clamped to the shortest metadata buffer (warns once per call via host `printf`). Fallback "neither raw nor ring" first-position now writes 0 (matches Triton) instead of `first_position`. |
| Op schema | Scalar ints (`page_size`, `width`, `compress_ratio`, `compressor_state_size`, `head_dim`, `max_model_len`) now passed as CPU int64 Tensors and read with `.item()` host-side. CPU tensor ⇒ no D2H sync, but a per-call host alloc. |
| Debug | `VLLM_QSA_RDNA2_DEBUG` adds host prints and an in-kernel `printf` branch in `qsa_compress_groups_kernel` (runtime `DEBUG` int). Python debug path calls `torch.cuda.synchronize()` — never on in serve or under capture. |

Take (Now): stride-dense gate before any `uint4` vector copy on a view we do not own; int64 slot/position contract for QSA; current-stream launch + launch check on every new HIP op.

Watch / Later (hunches, not measured): neither QSA kernel has `__launch_bounds__` (launches are 256 and 128 threads), so the compiler budgets registers for 1024-thread blocks; and the device `printf` compiled into `qsa_compress_groups_kernel` costs code size / registers even when `DEBUG=0`. Check VGPR/scratch in the ISA before any default-on flip. Not an occupancy-first item — FA prefill leftover still leads.

Leave: author tok/s and the HIP-vs-Triton gap figures. The same tip's adaptive prefill scheduler (`vllm/v1/core/sched/dynamic_prefill.py`, opt-in `VLLM_RDNA_DYNAMIC_PREFILL=1`) is engine-side Python, not silicon — see `engine/`.

### T49 `ple_short_conv_rdna2.cu` — dilated PLE short-conv

- Depthwise dilated conv1d decode+prefill; per-channel threads; `history_buf[64]` in regs; scalar FMA + silu.
- Opt-in: `VLLM_RDNA_PLE_CONV_HIP=1` + `on_gfx10x()`. Default off (F.conv1d).

### `mrope_forward_rdna2` build fixes

- Already room-locked Take (Qwen M-RoPE bind). Tip adds CUDAGuard include + host wrapper at global scope so `torch_bindings` resolves. Still one CTA/token, scalar, no fdot2.

## Leave

- Do not treat fused_glue / HC / QSA / PLE defaults as on — `VLLM_RDNA_FUSED_HC`, `VLLM_RDNA_HC_PREFILL_HIP`, `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_PLE_CONV_HIP` are all env-gated opt-in (default off) until soak / capture-safe.
- Do not flip `VLLM_RDNA_HC_PREFILL_HIP=1` on serve until MoE PIECEWISE capture fault is resolved (HC alone is not enough).
- No transplant of HC/PLE onto GLM KDA / norm-after-gate (same split as gated_rms_norm / causal_conv).
- No WMMA/FP8 objects; gfx1030 stays fdot2 / EXL3.
- No `__launch_bounds__` / DOT / LDS-tile / KV-quant / CMake gfx1030 list change in `8960a3bc` or `50120e13` (HC lock is Python dispatcher only).
- Do not flip `VLLM_RDNA_HC_PREFILL_HIP=1` on serve on perf grounds even after `c6d99ca3` fixed its stream race + `_contig` aliasing (correct, still slower than Triton). No `__launch_bounds__` / DOT / LDS-tile / KV-quant / CMake change in `c6d99ca3`.
- Do not flip `VLLM_RDNA_QSA_HIP=1` on serve: `121af1b4` fixes the store fault but the path is still default-off (Triton QSA is the reference). No `__launch_bounds__` / DOT / LDS-tile / CMake change in `121af1b4`.

## Occupancy

FA prefill leftover still first (`__launch_bounds__(*, 1)`). ticket-26 EXL3 `-cb 3inst` unchanged.
