# Causal conv1d HIP — extras (`7779514b`)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip **`7779514b`** (was `feb7b457`). New file `csrc/rocm/causal_conv1d_rdna2.cu` (+434), CMake gfx1030 list, `_rocm_C` bindings. Replaces Triton `_causal_conv1d_fwd_kernel` / `_update_kernel` for the Qwen GDN path (same JIT / cudagraph class as RMSNorm — [triton-jit-aot.md](triton-jit-aot.md)).

Not the FA occupancy leftover. No `fdot2` / DOT — depthwise FIR is scalar fp32 FMA. Do not invent tok/s.

## Tile (sourced)

| Path | File / op | WG | Scratch | Notes |
|---|---|---|---|---|
| Decode update | `causal_conv1d_update_rdna2` | 32 thr = 1 warp / dim_block of 32 channels; grid `(batch, dim/32)` | Registers only (0 LDS) | `dim % 32 == 0`; `width == state_len + 1`; fp16 in/out, fp32 accum; optional SiLU |
| Varlen prefill | `causal_conv1d_fwd_rdna2` | Same 32-thr warp grid | `__shared__` = `32 * state_len * sizeof(half)` | `2 <= width <= 5` (`CONV1D_FWD_MAX_WIDTH`); channel-last `stride(dim)==1`; initial-state load + writeback |

No `__launch_bounds__` / `waves_per_eu` on either kernel. Bias gated on `bias.defined() && bias.numel() > 0` (empty tensor ≠ nullptr).

## Python gates (tip wiring)

Env defaults are **`"1"`** (dispatch ON) despite stale “GATED OFF” comments:

| Env | Default | Path |
|---|---|---|
| `VLLM_CAUSAL_CONV1D_RDNA2_FWD` | `"1"` | Prefill: HIP **before** Triton, early `return` — correct |
| `VLLM_CAUSAL_CONV1D_RDNA2_UPDATE` | `"1"` | Decode: HIP sits **after** `_causal_conv1d_update_kernel[grid](...)` — Triton already launched; HIP may overwrite `out` / `conv_state` |

**Leave / fix later:** move the UPDATE HIP block above the Triton launch (mirror FWD), or the decode path double-fires. Author notes production faults at GDN `dim=5120` / `block_size=784` page alignment — do not claim soak-clean.

## Capture class

Same family as [../silicon/graph-capture.md](../silicon/graph-capture.md): Triton JIT scratch pointers go stale under cudagraph replay; HIP keeps scratch in registers (update) or LDS (fwd). FA / GDN host scratch in this tip also moves `torch.empty` → `torch.zeros` for page commit.

## Occupancy

Still first subject = FA prefill `(N,1)` / 1 WG @ 64 KiB. This kernel is tiny LDS or none — not a new first subject. No new UNC opened from this tip alone (UPDATE wire order is a dest bug, not a new card unless the room wants one).
