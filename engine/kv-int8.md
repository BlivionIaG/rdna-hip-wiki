# INT8 KV cache on gfx1030 — engine spec

Date: 2026-08-21. Engine contract. Silicon: [kernels/kv-int8.md](../kernels/kv-int8.md). Policy page (FP8 dead as a *unit*, offload physics): [kv-quant-offload.md](kv-quant-offload.md). One card on [project 4](https://github.com/users/BlivionIaG/projects/4). Human branch is **`rdna2_extras`**. Occupancy still first.

**Verdict:** cut KV bytes 2× vs fp16 by storing INT8 and **dequantizing inside `fa_rdna2`**. An unfused “dequant to an fp16 cache, then attend” kernel erases the memory win. Native FP8 KV stays **Dead** (`supports_fp8()` false). Sage INT8 QK is prefill *Q*, not this.

## Live on extras (`4cc1fe59`) — not the contract

Cherry-port of `3baecdb516` from `perf/rdna2_w4a16`. Files: `fa_rdna2.cu`, `rdna_attn.py`, `fa_rdna2_backend.py`.

| Path they landed | Engine verdict |
|---|---|
| INT8 decode `fa_decode_paged_splitk_kernel_int8_pth` | **Not the contract.** Silicon: scalar `__hmul`, `q_reg[D]` in VGPR, 64 threads same D loop, `kv_splits` unused, scale is the query token. |
| INT8 prefill | **Forbidden unfuse.** Python-dequants the **whole** cache → fp16 prefill kernel. Erases the GDDR6 win. |
| FP8 decode/prefill | Software e4m3→half on **existing** tiles + `fdot2`. Real fuse. Still `(1,1)`. Not a native FP8 unit — do not treat `supports_fp8()` as true. |

Do **not** land more INT8 gather on `(1,1)`. Occupancy flip still blocks this card. Keep the existing INT8 KV ticket — no new card.

## Why INT8, not FP8-as-unit

| Path | Status |
|---|---|
| FP8 E4M3 KV (vLLM default on MI / Hopper) | **Dead as a unit.** Software cvt only. |
| FP8 software fuse in `fa_rdna2` (`4cc1fe59`) | **Landed, occupancy-blocked.** e4m3→half + existing `fdot2` tiles. |
| INT8 + fused dequant | **Must / queued.** Native `i8` load + cvt + `fdot2`. `4cc1fe59` is not this. |
| INT4 KV | Later, after INT8 is shipping. |
| DSv4 `fp8_ds_mla` uint8 cache | Different layout (576 B token). Not this ticket. |

Kiely ladder: KV is more sensitive than weights, less than softmax. Leave softmax / attn scores wide. The win is **more resident tokens + less GDDR6 on decode gather**, not FLOPS.

## Quant mode (locked)

Ship **one** mode: `int8_per_token_head`.

vLLM already has the contract (`KVQuantMode`, `AttentionSpec.auxiliary_buffer_specs`, `bind_auxiliary_buffers`). Triton `int8_per_token` (absmax over all heads, one scale/token) and `int8_per_tensor` exist. On gfx1100 they **closed** per-tensor HIP and kept `int8_per_token_head` (same 50% VRAM, better long-decode). We copy that decision, not the gfx1100 kernel.

| Mode | Scale | Take |
|---|---|---|
| `int8_per_token_head` | fp32 scale per (token, head), K and V separate | **ship** |
| `int8_per_token` | one scale / token across heads | skip (worse accuracy, not fewer bytes) |
| `int8_per_tensor` | scalar `k_scale` / `v_scale` (fold K into `sm_scale`) | skip (gfx1100 closed it) |

Symmetric signed i8, `QUANT_MAX = 127`. `scale = max(absmax / 127, 1e-6)`. No AZP.

## Two kernels, one layout

Writer and reader must agree. Coverage already lists `reshape_and_cache` match as Must — this ticket **is** that match for INT8.

### Write — `reshape_and_cache_int8_rdna2`

HIP, not Triton (compile tax). Per new token / head:

1. absmax over `head_dim`
2. `scale = max(absmax / 127, 1e-6)` → `k_scale_cache[block, slot, head]` (fp32)
3. `q = rint(clamp(x / scale, -127, 127))` → store `int8` in the paged slot

Same paged block table as fp16. Payload is 1 byte/elem instead of 2. Scales are auxiliary buffers, counted in block-memory math.

Do **not** reuse the FP8 reshape path or view the cache as `fp8_e4m3`.

### Read — fused into `fa_rdna2`

Decode (`fa_rdna2_decode_paged`) and prefill (`fa_rdna2_prefill_*`):

```
load i8 K/V from the paged slot
cvt to fp32 (or fp16) * scale[token, head]     // in VGPR, on the fly
existing fdot2 QK / PV
```

No second kernel. No materialize-fp16-KV workspace for the whole cache. Prefill may stage a **tile** of dequanted K in LDS (the working set), not the full sequence.

Head-64 stays Triton until that FA2 tile exists — Triton INT8 path can cover the hole, HIP does D=128/256 first.

## Dispatch

| Gate | Spec |
|---|---|
| `--kv-cache-dtype int8_per_token_head` | honor on `on_gfx10x()` |
| `VLLM_USE_RDNA2_FA=1` | required so `fa_rdna2` is the reader |
| Occupancy flip | **blocks this.** Do not land INT8 gather on a decode kernel that is still `waves_per_eu(1,1)`. |
| Native FP8 KV flags | refuse as a unit. Software FP8 fuse in `fa_rdna2` is occupancy-blocked, not a promote. |

MLA / DSv4 sparse cache is out of scope. Mix/spec still off until a fat tile.

## Geometry (seeds — silicon confirms)

Same tiles as fp16 `fa_rdna2`. INT8 only changes the **load + cvt**, not Br/Bc.

| Regime | Seed |
|---|---|
| Decode D=128/256 | existing paged split-K, Br=1. Extra: 1× fp32 scale / head / token |
| Prefill | existing Br=16/32. Dequant into the K LDS tile |
| Occupancy | same ticket as today: drop min-blocks, `amdgpu_waves_per_eu(4, 8)` |

Scale loads are tiny vs KV bytes. Do not add an LDS scale LUT.

## Done-when

- ISA dump: `i8` load + integer-to-float cvt + `v_dot2_f32_f16`. No FP8 cvt-as-unit, no `sdot4` on KV (KV is storage; QK still `fdot2` until Sage).
- Writer and `fa_rdna2` agree on slot layout + `k_scale_cache` / `v_scale_cache`.
- One model serves with `--kv-cache-dtype int8_per_token_head` on gfx1030 without Triton reshape **and** without Python whole-cache dequant.
- Smoke vs fp16 KV (not bit-exact; bound the error). No tok/s from this page.
- KV bytes ≈ ½ of fp16 for the same token count (plus scale overhead).

`4cc1fe59` does **not** meet this.

## Not this ticket

Occupancy flip. Sage QK (`sdot4` on prefill Q). HIP MLA fp16. NVFP4. W8A8 `sdot4`. INT4 KV. Host/SSD KV offload. DSv4 indexer / `fp8_ds_mla`.

## Sources

- vLLM INT8 KV contract: `KVQuantMode`, PRs [#36893](https://github.com/vllm-project/vllm/pull/36893) (per-token Triton), [#41954](https://github.com/vllm-project/vllm/pull/41954) (gfx1100 closed per-tensor, kept per-token-head)
- extras `4cc1fe59` (verbatim port of `3baecdb516`)
- [kv-quant-offload.md](kv-quant-offload.md), [coverage.md](coverage.md), [attention-dispatch.md](attention-dispatch.md)
- Room 2026-08-21: silicon lock on `4cc1fe59`
