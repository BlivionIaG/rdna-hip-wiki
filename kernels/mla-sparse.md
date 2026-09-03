# HIP sparse MLA — silicon / HIP contract

## extras lock 2026-09-03 — `rdna_extras` @ `ea78104d`

HIP **prefill** landed in `csrc/rocm/sparse_mla_rdna2.cu` (was Triton-only on earlier tips).

| Piece | Value |
|---|---|
| Kernel | `sparse_mla_prefill_kernel` — `__launch_bounds__(32)` (1 wave32), grid `(T, H/HEADS_PER_CTA)` |
| KV | Plain `half` / `__hip_bfloat16` rows `[skv, COMB_DIM]` — **no** FP8 slots / E8M0 (fp8_ds_mla is decode-cache only) |
| LDS | Double-buffered `__shared__ scalar_t s_kv[2][COMB_DIM]` |
| QK / PV | Still **scalar fp32 FMA** (same as decode). Not `fdot2` yet |
| Bindings | `sparse_mla_prefill_rdna2` + decode already registered in `torch_bindings.cpp` |

Decode contract unchanged. Prefill is no longer “still Triton” on this dest tip. Occupancy leftover remains FA prefill, not MLA’s 32-thread WG.

---

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
