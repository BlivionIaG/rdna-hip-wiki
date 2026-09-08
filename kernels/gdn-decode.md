# GDN packed decode — extras HIP (`69d2efe` / tip `47d92b6`)

Tip **`b53a7a2`**. Qwen3.5/3.6 GatedDeltaNet **packed single-token** decode. Replaces Triton `fused_recurrent_gated_delta_rule_packed_decode` on the non-spec path. Prefill is now HIP (`b53a7a2`) — [gdn-prefill.md](gdn-prefill.md). Occupancy card stays the FA leftover — this is not a new first subject.

Do **not** put the microbench 9× on coverage. Fat-M / ConfigA unchanged.

## Tile (sourced)

| Knob | Value |
|---|---|
| File | `csrc/rocm/gdn_decode_rdna2.cu` |
| WG | 256 thr = 32 V-rows × 8 K-slices (`K=128`) |
| Occupancy attr | `__launch_bounds__(256)` + `amdgpu_waves_per_eu(2, 4)` — **not** `(1,1)` |
| LDS | **0** (register-resident fp32 `h[16]` per thread) |
| Inner | scalar fp32 FMA on state; q/k from half. **Not** `fdot2` (state is fp32 recurrent) |
| Reduce | `__shfl_xor` 1/2/4 inside the 8 k-slice lanes |
| Grid | `(ceil(V/32), B*HV)` |
| Gate | gfx10x, `K=128`, fp16 qkv, fp32 state, `use_qk_l2norm=True` |

256-thr = 8 waves/WG. On WGP (4 SIMD) that is **2 waves/SIMD**, so `(2,4)` = 1–2 such WGs/WGP. Looser than FA decode `(4,8)`. hipOccupancy still TBD.

## Leave

- Spec-decode packed path (still Triton unless extras flipped it)
- Copying 7.6 µs / 9.3× @ B=1 (launch tax vs Triton’s flat ~71 µs; @ B=32 they measured **0.93×**)

## Capture guard (dest `f9361950`, 2026-09-03)

The `ssm_state` RDNA2 uncommitted-page probe (`torch.isnan(...).any().item()` / all-zero → `zero_()`) is a **device→host sync** and is illegal under stream capture — it aborted cudagraph capture at engine startup with `hipErrorStreamCaptureUnsupported`. Dest now wraps it in `if not torch.cuda.is_current_stream_capturing():`. Python only; the `.cu` tile above is unchanged. Rule: [../silicon/graph-capture.md](../silicon/graph-capture.md).

## NULL_BLOCK_ID=0 sentinel (dest `7779514b`, 2026-09-08)

`gdn_decode_rdna2.cu`: guard was `state_idx < 0 || state_idx >= num_blocks_g`. Production sentinel is **`NULL_BLOCK_ID=0`** (`vllm.v1.attention.backends.utils`). Tip now uses **`state_idx <= 0`** — zero output, leave state untouched. Without it, slot 0 (real weights) was treated as a valid tile and corrupted the GDN recurrence on the first scheduled request. Tile / LDS / `(2,4)` occupancy attr unchanged.
