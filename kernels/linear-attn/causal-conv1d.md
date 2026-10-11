---
title: "Causal conv1d HIP — extras (`7779514b`)"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/7779514b449c358c14cb58c739cc33cd7efca3b4
  - https://github.com/opengfx1030/vllm-rdna/commit/feb7b457e3a0e471d3231e58226603ebde3a053d
  - https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/causal_conv1d_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
---
# Causal conv1d HIP — extras (`7779514b`)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip **`7779514b`** (was `feb7b457`). New file `csrc/rocm/causal_conv1d_rdna2.cu` (+434), CMake gfx1030 list, `_rocm_C` bindings. Replaces Triton `_causal_conv1d_fwd_kernel` / `_update_kernel` for the Qwen GDN path (same JIT / cudagraph class as RMSNorm — [triton-jit-aot.md](../aot-glue/triton-jit-aot.md)).[^c-7779514b][^c-feb7b457][^f-7779514b-causal-conv1d-rdna2-cu]

Not the FA occupancy leftover. No `fdot2` / DOT — depthwise FIR is scalar fp32 FMA. Do not invent tok/s.[^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu]

## Tile (sourced)

| Path | File / op | WG | Scratch | Notes |
|---|---|---|---|---|
| Decode update | `causal_conv1d_update_rdna2` | 32 thr = 1 warp / dim_block of 32 channels; grid `(batch, dim/32)` | Registers only (0 LDS) | `dim % 32 == 0`; `width == state_len + 1`; fp16 in/out, fp32 accum; optional SiLU [^f-7779514b-causal-conv1d-rdna2-cu] |
| Varlen prefill | `causal_conv1d_fwd_rdna2` | Same 32-thr warp grid | `__shared__` = `32 * state_len * sizeof(half)` | `2 <= width <= 5` (`CONV1D_FWD_MAX_WIDTH`); channel-last `stride(dim)==1`; initial-state load + writeback [^f-7779514b-causal-conv1d-rdna2-cu] |

No `__launch_bounds__` / `waves_per_eu` on either kernel. Bias gated on `bias.defined() && bias.numel() > 0` (empty tensor ≠ nullptr).[^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu]

## Python gates (tip wiring)

Env defaults are **`"1"`** (dispatch ON) despite stale “GATED OFF” comments:[^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu][^f-7779514b-causal-conv1d-py]

| Env | Default | Path |
|---|---|---|
| `VLLM_CAUSAL_CONV1D_RDNA2_FWD` | `"1"` | Prefill: HIP **before** Triton, early `return` — correct [^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu][^f-7779514b-causal-conv1d-py] |
| `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE` | `"1"` | Decode: HIP sits **after** the `_causal_conv1d_update_kernel[grid]` launch — Triton already launched; HIP may overwrite `out` / `conv_state` [^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu][^f-7779514b-causal-conv1d-py] |

**Leave / fix later:** move the UPDATE HIP block above the Triton launch (mirror FWD), or the decode path double-fires. Author notes production faults at GDN `dim=5120` / `block_size=784` page alignment — do not claim soak-clean.[^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu][^f-7779514b-causal-conv1d-py]

## Capture class

Same family as [silicon/graph-capture.md](../../silicon/graph-capture.md): Triton JIT scratch pointers go stale under cudagraph replay; HIP keeps scratch in registers (update) or LDS (fwd). FA / GDN host scratch in this tip also moves `torch.empty` → `torch.zeros` for page commit.[^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu]

## Occupancy

Still first subject = FA prefill `(N,1)` / 1 WG @ 64 KiB. This kernel is tiny LDS or none — not a new first subject. No new ticket opened from this tip alone (UPDATE wire order is a dest bug, not a new card unless the room wants one).[^c-7779514b][^f-7779514b-causal-conv1d-rdna2-cu]

## Sources

[^c-7779514b]: opengfx1030/vllm-rdna commit [`7779514b`](https://github.com/opengfx1030/vllm-rdna/commit/7779514b449c358c14cb58c739cc33cd7efca3b4) (2026-09-08) — "fix(rocm,fa-rdna2): torch::zeros for O_partial/M_partial on RDNA2 cudagraph replay".
[^c-feb7b457]: opengfx1030/vllm-rdna commit [`feb7b457`](https://github.com/opengfx1030/vllm-rdna/commit/feb7b457e3a0e471d3231e58226603ebde3a053d) (2026-09-06) — "feat: add exllama-clone AWQ prefill kernel for high-M (gfx1030)".
[^f-7779514b-causal-conv1d-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/causal_conv1d_rdna2.cu` @ [`7779514b`](https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/causal_conv1d_rdna2.cu).
[^f-7779514b-causal-conv1d-py]: opengfx1030/vllm-rdna `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` @ [`7779514b`](https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/vllm/model_executor/layers/mamba/ops/causal_conv1d.py).
