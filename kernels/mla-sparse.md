# HIP sparse MLA decode — silicon / HIP contract

Live: `sparse_mla_rdna2.cu` @ `9a344444`. Engine: [engine/fork-delta.md](../engine/fork-delta.md). **Incomplete.** Env `VLLM_USE_RDNA2_MLA=1`. Prefill still Triton.

## ISA (confirmed)

| Piece | What it is |
|---|---|
| q / out | templated `half` on gfx1030. No fp16→bf16 cast. |
| K_nope | E4M3 + E8M0 / 64-ch. `__hip_cvt_fp8_to_halfraw` then `* exp2f(sc-127)` |
| K_rope | **bf16 in the slot** (cache contract). `__bfloat162float` |
| QK / PV | **scalar fp32 FMA**, not `fdot2` |
| Occupancy | `__launch_bounds__(32)` only — no `(1,1)` trap |
| Grid | `(B, H/4)`, H multiple of 4 (K128=16, K256=64) |

Unpack already lands in `half`; 14+2 dims/thread are even. Later upgrade: stop promoting, `fdot2` the pairs. Not a new ticket. Occupancy still first for `fa_rdna2`.
