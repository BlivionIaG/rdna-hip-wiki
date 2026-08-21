# INT8 KV cache on gfx1030 — silicon / HIP contract

Engine dispatch and quant mode: [engine/kv-int8.md](../engine/kv-int8.md). This page is the ISA and the `fa_rdna2` load path.

**Win is bandwidth, not FLOPS.** Decode gather is GDDR6-bound. INT8 KV is ½ the bytes vs fp16. An unfused “dequant the whole cache to fp16, then attend” kernel throws that away.

Q stays **fp16**. QK/PV stay **`fdot2`**. This is not Sage (`sdot4` on INT8 Q). This is not FP8 KV.

## Native ops

| Step | Instruction | HIP |
|---|---|---|
| Load packed K/V | `global_load` / `buffer_load` as `int` / `char4` | dword loads, `D % 4 == 0` (128/256) |
| i8 → f32 | `v_cvt_f32_i32` after sext | or `v_cvt_f16_i16` then promote |
| × scale | `v_mul_f32` | one fp32 scale per (token, head), K and V separate |
| QK / PV | `V_DOT2_F32_F16` | `__builtin_amdgcn_fdot2` — same as today’s `fa_rdna2` |

Do **not**: `sdot4` on KV (Q is fp16), FP8 cvt, `v_dot2_f32_bf16`, a K$ scale LUT.

## Writer — `reshape_and_cache_int8_rdna2`

Per new token / head, in HIP (not Triton):

```
absmax = max(|x|) over D
scale  = max(absmax / 127, 1e-6)          // fp32 → k_scale_cache[block, slot, head]
q      = rint(clamp(x / scale, -127, 127)) // store int8 in the paged slot
```

Symmetric signed i8. No AZP. Same block table as fp16. Scales are auxiliary buffers.

Pack store as `int` (4×i8) so the reader can `ds`/`global` dword. Sequential `int` along D, wave32, 64 banks: conflict-free.

## Reader — fused into `fa_rdna2`

Same Br/Bc/D tiles as fp16 `fa_rdna2`. Only the **load + cvt** changes.

```
// per K/V element in the current tile
i8  = load packed
f   = cvt_f32_i8(i8) * scale[token, head]   // VGPR, on the fly
// then existing fdot2 against Q (fp16) or P
```

- Decode: do **not** materialize a full-seq fp16 KV workspace.
- Prefill: may stage a **tile** of dequanted K in LDS (the working set), not the sequence.
- Scale load is one fp32 per head per token — tiny vs KV. No LDS scale table.

Occupancy flip **blocks this**. Do not land INT8 gather on `__launch_bounds__(*, 1)` / `waves_per_eu(1,1)`.

## Tiles (unchanged)

| Kernel | WG | Notes |
|---|---|---|
| decode D=128 | 128 | Br=1, split-K. Extra: 1× fp32 scale / head / token |
| decode D=256 | 256 | same |
| prefill 128/256 | 128/256 | dequant into the existing K LDS tile |
| Occupancy | — | `amdgpu_waves_per_eu(4, 8)` after the FA ticket |

Head-64 stays Triton until that FA2 tile exists.

## Why not sdot4 here

Sage is **prefill Q and K both INT8**. KV-cache INT8 is **storage**. Q at decode is one row of fp16. Quantizing that Q every step to win `sdot4` is a different kernel (Sage) and does not help decode bandwidth. Keep QK on `fdot2`.

## Not their `fd_rdna2` slice

leapdragon `fd_rdna2` (GPL Triton plugin) is **not** the HIP path. It keeps packed i8 as `(TILE,64)` and fires four `tl.dot` so Triton never writes a 256-wide LDS unpack. That is a compiler dodge. HIP already has `fdot2` on `half2` — load packed, cvt+scale in **VGPR**, existing tiles. See [../silicon/leapdragon.md](../silicon/leapdragon.md) §4.

Do **not**: vendor the plugin; copy the 4-way Q permute / `PAD=8` / `GQA=6` hardcode (GQA-4 Qwen misses it); unpack-then-reshape to `(TILE,256)` LDS (their v0, dead).

## Done-when (ISA dump)

- [ ] KV load is i8 / packed `int`, not fp16 and not `fp8_e4m3`
- [ ] cvt + `v_mul_f32` by `k_scale` / `v_scale`, then `v_dot2_f32_f16`
- [ ] No `v_dot4c_i32_i8` on the KV path
- [ ] Writer and reader agree on slot + scale layout
- [ ] `waves_per_eu(4, 8)` on the decode reader
- [ ] Smoke vs fp16 KV (not bit-exact). No tok/s from this page

## Sources

- Engine: [engine/kv-int8.md](../engine/kv-int8.md), [engine/kv-quant-offload.md](../engine/kv-quant-offload.md), [engine/leapdragon.md](../engine/leapdragon.md)
- Live FA: `csrc/rocm/fa_rdna2.cu` @ `add17dd7` — [silicon/fa-occupancy.md](../silicon/fa-occupancy.md)
- vLLM INT8 KV: `KVQuantMode`, PRs [#36893](https://github.com/vllm-project/vllm/pull/36893), [#41954](https://github.com/vllm-project/vllm/pull/41954)
- RDNA 2 ISA 70648: `V_DOT2_F32_F16`; i8 cvt; no FP8 unit
