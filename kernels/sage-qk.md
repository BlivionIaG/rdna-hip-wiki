# Sage QK on gfx1030 (silicon contract)

Engine dispatch and the paper live in [engine/sage-attention.md](../engine/sage-attention.md). This page is what a HIP kernel must do. Check a future implementation against this list.

**Paper:** SageAttention, Zhang et al., [arXiv 2410.02367](https://arxiv.org/abs/2410.02367) — INT8 QK + FP16 PV.

## Do

- Prefill only. Decode stays `fa_rdna2` (`fdot2`).
- QK: pack `D` as 4×i8 per dword. `D % 4 == 0` (128 and 256 are fine; head-64 is also fine).
- Inner loop: `__builtin_amdgcn_sdot4(q, k, acc, false)` → `v_dot4c_i32_i8`. Keep the K reduction in **i32**. Scale once in the epilogue (`scale_q * scale_k`).
- Per-block / per-tile scales like the paper (not per-tensor only). Symmetric int8.
- PV: still `__builtin_amdgcn_fdot2` on fp16. Do not INT8 the V·P matmul on this chip unless you measure it.
- Wave32, WGP, LDS ≤ 64 KB/WG, `amdgpu_waves_per_eu(4, 8)`. Sequential `int` along D is conflict-free on 64 banks.
- After occupancy flip on FA2. Do not land Sage on `__launch_bounds__(*, 1)`.

## Do not

- `sudot4` (gfx11+).
- `sdot8` / INT4 QK (Sage2) — no INT4 tensor path we trust for QK numerics.
- FA3 (Hopper TMA/WGMMA/FP8).
- FP8 QK (`supports_fp8()` is false; no FP8 unit).
- Putting a nibble LUT in K$ (serializes).
- Claiming a decode win. Decode is KV bandwidth.

## Rate

GPUOpen RDNA2 table: IU8 **512** ops/clock/CU vs packed FP16 **256**. QK is the only place that 2× shows up. PV stays on the FP16 pipe.

## Checklist when a kernel lands

- [ ] Opcode in the ISA dump is `v_dot4c_i32_i8` / `v_dot4_i32_i8`, not `v_dot2c_f32_f16` on QK
- [ ] `D` walked as `int`, `K` reduction in i32
- [ ] Scales applied once per tile, not per DOT4
- [ ] Occupancy > 0 at WG=256 (`llvm-calc-occupancy -mcpu=gfx1030` + runtime query)
- [ ] Prefill tok/s vs `fa_rdna2` prefill 128/256 on V620 (measure, don’t invent)
- [ ] Decode path unchanged
