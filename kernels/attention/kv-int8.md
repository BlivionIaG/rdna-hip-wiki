---
title: "INT8 KV cache on gfx1030 — silicon / HIP contract"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/8f2583d2c6e49664c59c490d9843f6265f79b90a
  - https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/fa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/vllm/v1/attention/backends/rdna_attn.py
  - https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/vllm/envs.py
  - https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb
  - https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/layernorm.cu
  - https://github.com/BlivionIaG/vllm/commit/3baecdb516cb969a9ae5a24eadf345dde7b54724
  - https://github.com/BlivionIaG/vllm/blob/4cc1fe59710171585d63ba76522891305ff22423/csrc/rocm/fa_rdna2.cu
  - https://github.com/BlivionIaG/vllm/commit/4cc1fe59710171585d63ba76522891305ff22423
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/vllm-project/vllm/pull/36893
  - https://github.com/vllm-project/vllm/pull/41954
---
# INT8 KV cache on gfx1030 — silicon / HIP contract

## extras lock 2026-09-03 — `opengfx1030/vllm-rdna` `rdna_extras` @ `ea78104d`

Dest tip moved `8f2583d2` → `ea78104d` (23 commits). HIP silicon that landed for KV-INT8 / FA:[^c-8f2583d2][^c-ea78104d]

| Piece | What shipped | Silicon note |
|---|---|---|
| Decode `fa_decode_paged_splitk_kernel{,_256}` | Template `<KV_T, IS_FP8, IS_INT8>`. INT8 path loads i8, `cvt * k_scale_per_tok[n,h]` / `v_scale_per_tok` into the **same** fp16 LDS tile, then existing `fdot2` QK. | Occupancy attr stays `__launch_bounds__(128\|256)` + `amdgpu_waves_per_eu(4, 8)`. **Not** the old scalar `__hmul` pth stub. [^f-ea78104d-fa-rdna2-cu] |
| Prefill INT8 | New `fa_prefill_paged_varlen_splitk_kernel_int8_{128,256}`. Tile dequant into LDS (~32.7 KB vs ~48 KB fp16); still `__launch_bounds__(N, 1)`. | Prefill occupancy leftover **unchanged**. Do not retip FA pin closed. [^f-ea78104d-fa-rdna2-cu] |
| Writer | `reshape_and_cache_int8_rdna2` — `__launch_bounds__(HEAD_DIM, 4)`. Cache layout `[2, blocks, H_kv, D+4, block_size]` with scale as 4 LE bytes at offset D per slot. | One CTA/(token,head). Absmax → `scale = max(absmax/127, 1e-6)`. [^f-ea78104d-fa-rdna2-cu] |
| Wiring | `torch_bindings` + `vllm/v1/attention/backends/rdna_attn.py` (`VLLM_USE_RDNA2_FA=1`). Platform gate: head_size ∈ {128,256}, `block_size >= 1`. | Python dispatch only; ISA is `fa_rdna2.cu`. [^f-ea78104d-rdna-attn-py][^f-ea78104d-fa-rdna2-cu][^f-ea78104d-envs-py] |

Leave this hour: EXL3 Python mul1-fold / debug fprintf, GDN ssm zeroing, `VLLM_FORCE_CUSTOM_ALL_REDUCE` (AR fabric, already Leave / ticket-27). Do not copy tok/s. Occupancy still first (FA prefill `(N,1)` / LDS 1 WG per 64 KB).[^f-ea78104d-envs-py]

CMake gfx1030 list: unchanged for FA/INT8 objects. Only add was `layernorm.cu` on the EXL3-unconditional list (AOT RMSNorm already locked @ `83de31cf` on the old overlay).[^c-83de31cf][^f-83de31cf-layernorm-cu]

---

Engine dispatch and quant mode: [engine/kv-cache/kv-int8.md](../../engine/kv-cache/kv-int8.md). This page is the ISA and the `fa_rdna2` load path. curvedinf/int8-vllm (gfx908 AITER): [silicon/external-projects/curvedinf-int8-vllm.md](../../silicon/external-projects/curvedinf-int8-vllm.md) — Take the PTH mode, Leave their UA kernel.[^f-ea78104d-fa-rdna2-cu]

**Win is bandwidth, not FLOPS.** Decode gather is GDDR6-bound. INT8 KV is ½ the bytes vs fp16. An unfused “dequant the whole cache to fp16, then attend” kernel throws that away.[^c-ea78104d]

Q stays **fp16**. QK/PV stay **`fdot2`**. This is not Sage (`sdot4` on INT8 Q).[^c-ea78104d]

## Live on `rdna2_extras` @ `4cc1fe59` (2026-08-21)

Cherry-port of `3baecdb516`. **Not the contract.** Occupancy still first.[^c-3baecdb5]

| Path | What shipped | Silicon |
|---|---|---|
| INT8 decode `fa_decode_paged_splitk_kernel_int8_pth` | Scalar `__hmul` QK/PV, `q_reg[D]` + `o_acc[D]` in VGPR, **smem 0**, launch 64/128 threads that **all run the full D loop** (`t` unused except empty/write). `blockIdx.z` unused — `kv_splits=8` repeats the whole sweep. Scale is `k_scale[token_idx, h_kv]` (query token), not per cached slot. | No `fdot2`. Comment “no native int8 dot” is **wrong** (`sdot4` exists; we still do not want it on KV). Spill / 64× redundant. [^f-4cc1fe59-fa-rdna2-cu] |
| INT8 prefill | Python: whole cache `.to(fp32)*scale.to(fp16)` then fp16 prefill | Unfused. Throws the bandwidth win. [^c-4cc1fe59][^f-4cc1fe59-fa-rdna2-cu] |
| FP8 decode/prefill | Template on existing tiles: software `e4m3→half`, LDS stays fp16, **`fdot2`**, still `__launch_bounds__(128/256, 1)` | Closest to the contract. No FP8 unit — software unpack is fine. Occupancy trap **inherited**. [^c-4cc1fe59][^f-4cc1fe59-fa-rdna2-cu] |

Do **not** retip occupancy closed. Next INT8 work is: delete the pth stub, fuse i8 load into the occupancy-fixed fp16 tiles (same as FP8 template), per-(block,slot,head) scale. Prefill: tile dequant in LDS, not a Python materialize.[^c-4cc1fe59][^f-4cc1fe59-fa-rdna2-cu]

## Native ops

| Step | Instruction | HIP [^f-ea78104d-fa-rdna2-cu][^isa] |
|---|---|---|
| Load packed K/V | `global_load` / `buffer_load` as `int` / `char4` | dword loads, `D % 4 == 0` (128/256) [^f-ea78104d-fa-rdna2-cu][^isa] |
| i8 → f32 | `v_cvt_f32_i32` after sext | or `v_cvt_f16_i16` then promote [^f-ea78104d-fa-rdna2-cu][^isa] |
| × scale | `v_mul_f32` | one fp32 scale per **cached** (token, head), K and V separate [^f-ea78104d-fa-rdna2-cu][^isa] |
| QK / PV | `V_DOT2_F32_F16` | `__builtin_amdgcn_fdot2` — same as today’s `fa_rdna2` [^f-ea78104d-fa-rdna2-cu][^isa] |

Do **not**: `sdot4` on KV (Q is fp16), a K$ scale LUT, scalar `__hmul` over D, Python dequant of the whole cache.[^f-ea78104d-fa-rdna2-cu][^isa]

## Writer — `reshape_and_cache_int8_rdna2`

Per new token / head, in HIP (not Triton):[^f-ea78104d-fa-rdna2-cu][^f-ea78104d-fa-rdna2-cu][^isa]

```
absmax = max(|x|) over D
scale  = max(absmax / 127, 1e-6)          // fp32 → k_scale_cache[block, slot, head]
q      = rint(clamp(x / scale, -127, 127)) // store int8 in the paged slot
```

Symmetric signed i8. No AZP. Same block table as fp16. Scales are auxiliary buffers.[^f-ea78104d-fa-rdna2-cu][^f-ea78104d-fa-rdna2-cu][^isa]

Pack store as `int` (4×i8) so the reader can `ds`/`global` dword. Sequential `int` along D, wave32, 64 banks: conflict-free.[^isa]

## Reader — fused into `fa_rdna2`

Same Br/Bc/D tiles as fp16 `fa_rdna2`. Only the **load + cvt** changes. The live FP8 template is the shape to copy (not the INT8 pth stub).[^f-ea78104d-fa-rdna2-cu]

```
// per K/V element in the current tile
i8  = load packed
f   = cvt_f32_i8(i8) * scale[kv_token, head]   // VGPR, on the fly
// then existing fdot2 against Q (fp16) or P
```

- Decode: do **not** materialize a full-seq fp16 KV workspace.[^f-ea78104d-fa-rdna2-cu][^f-ea78104d-fa-rdna2-cu][^isa]
- Prefill: may stage a **tile** of dequanted K in LDS (the working set), not the sequence.[^f-ea78104d-fa-rdna2-cu][^f-ea78104d-fa-rdna2-cu][^isa]
- Scale load is one fp32 per head per **cached** token — tiny vs KV. No LDS scale table.[^f-ea78104d-fa-rdna2-cu][^f-ea78104d-fa-rdna2-cu][^isa]

Occupancy flip **blocks this**. Do not land INT8 gather on `__launch_bounds__(*, 1)` / `waves_per_eu(1,1)`.[^f-ea78104d-fa-rdna2-cu][^f-ea78104d-fa-rdna2-cu][^isa]

## Tiles (unchanged)

| Kernel | WG | Notes |
|---|---|---|
| decode D=128 | 128 | Br=1, split-K. Extra: 1× fp32 scale / head / token [^f-ea78104d-fa-rdna2-cu][^isa] |
| decode D=256 | 256 | same [^f-ea78104d-fa-rdna2-cu][^isa] |
| prefill 128/256 | 128/256 | dequant into the existing K LDS tile [^f-ea78104d-fa-rdna2-cu][^isa] |
| Occupancy | — | `amdgpu_waves_per_eu(4, 8)` after the FA ticket [^f-ea78104d-fa-rdna2-cu][^isa] |

Head-64 stays Triton until that FA2 tile exists.[^f-ea78104d-fa-rdna2-cu][^isa]

## Why not sdot4 here

Sage is **prefill Q and K both INT8**. KV-cache INT8 is **storage**. Q at decode is one row of fp16. Quantizing that Q every step to win `sdot4` is a different kernel (Sage) and does not help decode bandwidth. Keep QK on `fdot2`. The extras comment “RDNA2 has no native int8 matrix dot” is false (`v_dot4_i32_i8`); it is still the wrong op here.[^isa]

## Not their `fd_rdna2` slice

leapdragon `fd_rdna2` (GPL Triton plugin) is **not** the HIP path. It keeps packed i8 as `(TILE,64)` and fires four `tl.dot` so Triton never writes a 256-wide LDS unpack. That is a compiler dodge. HIP already has `fdot2` on `half2` — load packed, cvt+scale in **VGPR**, existing tiles. See [silicon/external-projects/leapdragon.md](../../silicon/external-projects/leapdragon.md) §4.[^f-ea78104d-fa-rdna2-cu]

Do **not**: vendor the plugin; copy the 4-way Q permute / `PAD=8` / `GQA=6` hardcode (**hits 27B 24/4; misses GQA-4 and 7B 28/4**); unpack-then-reshape to `(TILE,256)` LDS (their v0, dead).[^f-ea78104d-fa-rdna2-cu]

## Done-when (ISA dump)

- [ ] KV load is i8 / packed `int`, not fp16 and not a Python materialize[^f-ea78104d-fa-rdna2-cu][^isa]
- [ ] cvt + `v_mul_f32` by per-**slot** `k_scale` / `v_scale`, then `v_dot2_f32_f16`[^isa]
- [ ] No `v_dot4c_i32_i8` on the KV path[^isa]
- [ ] No scalar `__hmul` over D; no `q_reg[D]` / `o_acc[D]` per thread[^f-ea78104d-fa-rdna2-cu][^isa]
- [ ] Writer and reader agree on slot + scale layout
- [ ] `waves_per_eu(4, 8)` on the decode reader[^f-ea78104d-fa-rdna2-cu][^isa]
- [ ] Smoke vs fp16 KV (not bit-exact). No tok/s from this page[^f-ea78104d-fa-rdna2-cu][^isa]

## Sources

- Engine: [engine/kv-cache/kv-int8.md](../../engine/kv-cache/kv-int8.md), [engine/kv-cache/kv-quant-offload.md](../../engine/kv-cache/kv-quant-offload.md), [engine/run-recipes/leapdragon.md](../../engine/run-recipes/leapdragon.md)
- Live FA: `csrc/rocm/fa_rdna2.cu` on **`rdna2_extras`** @ `4cc1fe59` — [silicon/attention/fa-occupancy.md](../../silicon/attention/fa-occupancy.md)
- vLLM INT8 KV: `KVQuantMode`, PRs [#36893](https://github.com/vllm-project/vllm/pull/36893), [#41954](https://github.com/vllm-project/vllm/pull/41954)
- RDNA 2 ISA 70648: `V_DOT2_F32_F16`; i8 cvt; `V_DOT4_I32_I8` exists but not for this path; no FP8 unit

[^c-8f2583d2]: opengfx1030/vllm-rdna commit [`8f2583d2`](https://github.com/opengfx1030/vllm-rdna/commit/8f2583d2c6e49664c59c490d9843f6265f79b90a) (2026-09-02) — "Revert "rdna: disable dynamo on Exl3LinearMethod.apply"".
[^c-ea78104d]: opengfx1030/vllm-rdna commit [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/commit/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a) (2026-09-03) — "fix(rocm): platform gates + MoE W8A16 parity (completes fb1687bde)".
[^f-ea78104d-fa-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/fa_rdna2.cu` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/csrc/rocm/fa_rdna2.cu).
[^f-ea78104d-rdna-attn-py]: opengfx1030/vllm-rdna `vllm/v1/attention/backends/rdna_attn.py` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/vllm/v1/attention/backends/rdna_attn.py).
[^f-ea78104d-envs-py]: opengfx1030/vllm-rdna `vllm/envs.py` @ [`ea78104d`](https://github.com/opengfx1030/vllm-rdna/blob/ea78104d5fc8cb5d8ee7a5d7e2d35d89ee90229a/vllm/envs.py).
[^c-83de31cf]: BlivionIaG/vllm commit [`83de31cf`](https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb) (2026-09-01) — "fix(rocm/layernorm): close namespace braces before global-scope public entries".
[^f-83de31cf-layernorm-cu]: BlivionIaG/vllm `csrc/rocm/layernorm.cu` @ [`83de31cf`](https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/layernorm.cu).
[^c-3baecdb5]: BlivionIaG/vllm commit [`3baecdb5`](https://github.com/BlivionIaG/vllm/commit/3baecdb516cb969a9ae5a24eadf345dde7b54724) (2026-08-21) — "feat(rocm): fp8 + int8 KV cache decode and prefill for fa_rdna2".
[^f-4cc1fe59-fa-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/fa_rdna2.cu` @ [`4cc1fe59`](https://github.com/BlivionIaG/vllm/blob/4cc1fe59710171585d63ba76522891305ff22423/csrc/rocm/fa_rdna2.cu).
[^c-4cc1fe59]: BlivionIaG/vllm commit [`4cc1fe59`](https://github.com/BlivionIaG/vllm/commit/4cc1fe59710171585d63ba76522891305ff22423) (2026-08-21) — "feat(rocm): port fp8 + int8 KV cache decode and prefill from perf/rdna2_w4a16".
[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
