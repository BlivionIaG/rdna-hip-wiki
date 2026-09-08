# Graph capture vs the RDNA2 page-commit guard — silicon

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. Two separate rules, one lock:

1. A caller-provided out tensor must be declared **`Tensor!`** in the `TORCH_LIBRARY` schema.
2. The RDNA2 uncommitted-page workaround must **never** read device→host on the captured / per-step decode path.

Not an occupancy change. Occupancy leftover stays FA prefill `(N,1)`. No DOT, LDS-tile, KV-quant, or CMake gfx1030 change here.

## Why the page guard exists at all

gfx1030 hands back **uncommitted pages**: a `torch::empty` buffer that is never fully written reads as garbage, not zeros. Existing marks of the same family:

| Site | Workaround |
|---|---|
| W4A16 split-K partials | `torch::empty` instead of `hipMallocAsync` mempool reuse — [fa-occupancy.md](fa-occupancy.md) |
| EXL3 6bpw lm_head dequant out | `torch.zeros`, not `empty` — [exl3.md](exl3.md) |
| GDN `ssm_state` slot pages | NaN / all-zero probe then `zero_()` — [../kernels/gdn-decode.md](../kernels/gdn-decode.md) |

The **buffer** fix (allocate zeroed) is capture-safe. The **probe** fix is not.

## Rule 2: no D2H on the captured path

`.item()`, a blocking `hipMemcpy` D2H, and `hipStreamSynchronize` are illegal inside stream capture. Under cudagraph capture they abort with `hipErrorStreamCaptureUnsupported`, and the failure lands at **engine startup** (capture time), not as a decode-time numerics bug — so it reads like a load failure.

Gate the probe, do not delete it:

```python
if not torch.cuda.is_current_stream_capturing():
    ...  # .item() probe, then zero_()
```

Same class as the UNC-27 soak catch on `rdna_ar_timed_out()`: its blocking D2H must stay out of graph capture **and** off the per-step path. A guard that is correct but synchronous is still a per-step tax once capture is off; prefer committing the pages at warmup/allocation over probing every decode.

## Rule 1: `Tensor!` on every HIP out tensor

An op that writes into a caller buffer but declares plain `Tensor` lies to functionalization: the write can be treated as dead or the tensor aliased, and the symptom is silent wrong output, not a crash. On the rocm bindings every op already carried `Tensor!` on its out arg (`wvSplitKQ` `out_c`; all `gdn_*` `out` / `initial_state` / `q` / `k_out` / `v` / `g_cumsum` / `beta` / `A` / `A_inv` / `w` / `u` / `h` / `v_new` / `o`; `c` on `moe_gptq` / `moe_w8a16` / `moe_mxfp4` / `mxfp4` / `moe_exl3` / `exl3_gemm_rdna2` / `moe_gptq_gemm_rdna3`; `paged_attention` `out`; `rms_norm` `out`; `fused_add_rms_norm` `input` + `residual`) — `exl3_hadamard_128` was the one stale schema.

That matters because `suh` / `svh` sit **outside** the K-dot as their own kernel ([exl3.md](exl3.md)): the Hadamard's only product is its out tensor, so a non-mutating schema is exactly the shape functionalization can drop.

## extras lock 2026-09-03 (tip `f9361950`, was `ea78104d`)

Two commits, both 2026-09-03 12:11 Paris. HIP delta is one schema character; the rest is Python.

| Commit | File | Delta |
|---|---|---|
| `faf87f80` | `csrc/rocm/torch_bindings.cpp` | `exl3_hadamard_128(Tensor input, Tensor output, …)` → **`Tensor! output`**. +1/−1. Impl and `.cu` untouched. |
| `f9361950` | `vllm/…/mamba/gdn/qwen_gdn_linear_attn.py` | `_forward_core_decode_non_spec`: the `ssm_state` NaN / all-zero probe is now wrapped in `if not torch.cuda.is_current_stream_capturing():`. +8/−4. |

Per the commit message, capture-time warmup writes already commit the state pages and GDN prefill overwrites slot content before real decode, so skipping the probe under capture is safe. That is the author's rationale, not a measurement of ours — the numerics claim is unverified on our box.

Nothing else moved: no `__launch_bounds__` / `waves_per_eu`, no DOT builtin, no LDS tile, no KV quant path, no CMake gfx1030 list. Do not copy tok/s. Do not invent numbers.

## extras lock 2026-09-08 (tip `7779514b`, was `feb7b457`)

Same page-commit family, wider surface. Not occupancy. No DOT / LDS-tile / KV-quant / `__launch_bounds__` change on FA itself.

| Surface | Delta |
|---|---|
| `csrc/rocm/fa_rdna2.cu` | Host `O` / `O_partial` / `M_partial` (and decode/prefill fp16/fp8/int8 wrappers that still used `empty`) → **`torch::zeros`**. Capture bakes addresses; uncommitted pages fault on replay. |
| GDN prefill / prep Python | `A` / `A_inv` / `w` / `u` / `h` / `v_new` / `final_state` / prep outs → **`torch.zeros`** (same rationale). |
| New `causal_conv1d_rdna2.cu` | HIP AOT for GDN conv1d — registers (update) / tiny LDS (fwd). See [../kernels/causal-conv1d.md](../kernels/causal-conv1d.md). |
| FA Python dispatch | Prefer `torch.ops._rocm_C.fa_rdna2_*` over `load_inline` when the `.so` op exists (capture-safe). |
| GDN / `rdna_attn` | `@eager_break_during_capture` on GDN `forward` and `do_kv_cache_update` (capture break, not a tile change). |

Occupancy leftover still FA prefill. Do not invent numbers.
