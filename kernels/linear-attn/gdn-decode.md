---
title: "GDN packed decode — extras HIP (`69d2efe` / tip `47d92b6`)"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/13fbe2f9160027e71ab6c915ecbcb95d42373a2e
  - https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/csrc/rocm/gdn_decode_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
  - https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/docs/rdna2/tuning.md
  - https://github.com/BlivionIaG/vllm/commit/b53a7a2c2bdf0739daa459ffff169c53a9a0f49e
  - https://github.com/BlivionIaG/vllm/commit/69d2efea371e52e4dca4c233e198ef627e181f0b
  - https://github.com/BlivionIaG/vllm/commit/47d92b6cfd9dddbc364cd7341c486cb534ac73db
  - https://github.com/BlivionIaG/vllm/blob/69d2efea371e52e4dca4c233e198ef627e181f0b/csrc/rocm/gdn_decode_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/gdn_decode_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/f93619500c3e55daee5f60ae7f6d4279cdeb3f0d
  - https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
---
# GDN packed decode — extras HIP (`69d2efe` / tip `47d92b6`)

## Spec-verify strided inputs at dest `rdna_extras` @ `13fbe2f9` (checked 2026-11-02)

Occupancy unchanged: `gdn_spec_verify_rdna2_kernel` keeps `__launch_bounds__(GDN_THREADS)` + `amdgpu_waves_per_eu(2, 4)`, `GDN_THREADS = 256`, `GDN_BV = 32`.[^gdnd-13fb-lb]

Contract change: the host wrapper no longer requires contiguous q/k/v/a/b. It now checks only that each token's `[heads, dim]` block is dense (`stride(3) == 1`, `stride(2) == size(3)`), that a/b are unit-stride on the head axis, and that `out` is contiguous. The kernel already took a per-tensor token stride, so q/k/v and a/b can be views into the packed conv output and the ba projection.[^gdnd-13fb-chk] *Historical:* the `requires contiguous q/k/v/a/b/out` check held up to `b30757c2`.[^gdnd-b307-chk]

Gate: new env `VLLM_GDN_STRIDED_INPUTS` (default `1`). It applies only when the whole GDN layer is the opaque eager op (`VLLM_RDNA_GDN_OPAQUE` on), on RDNA2 HIP, with `VLLM_GDN_DECODE_RDNA2` on; `0` restores the packing copies.[^gdnd-13fb-env] The fork docs say mixed prefill+decode batches and non-dense layouts still take the copies, and that results are bitwise identical to the copy path per its tests (fork claim, not re-run here).[^gdnd-13fb-doc] The gated RMSNorm half of this change is in [layernorm.md](../aot-glue/layernorm.md).

## Current state at dest `rdna_extras` @ `1dc3bac4` (checked 2026-10-11)

`gdn_decode_rdna2.cu` is still present: 256 threads, `GDN_BV = 32`, `__launch_bounds__(256)` + `amdgpu_waves_per_eu(2, 4)`. HIP decode is **default on**: `VLLM_GDN_DECODE_RDNA2` is anything but `"0"`, on gfx10x/RDNA2 with fp16 and `head_k_dim == 128`.[^gdnd-tip]


Tip **`b53a7a2`**. Qwen3.5/3.6 GatedDeltaNet **packed single-token** decode. Replaces Triton `fused_recurrent_gated_delta_rule_packed_decode` on the non-spec path. Prefill is now HIP (`b53a7a2`) — [gdn-prefill.md](gdn-prefill.md). Occupancy card stays the FA leftover — this is not a new first subject.[^c-b53a7a2]

Do **not** put the microbench 9× on coverage. Fat-M / ConfigA unchanged.[^c-69d2efe][^c-47d92b6][^f-69d2efe-gdn-decode-rdna2-cu]

## Tile (sourced)

| Knob | Value |
|---|---|
| File | `csrc/rocm/gdn_decode_rdna2.cu` [^f-7779514b-gdn-decode-rdna2-cu] |
| WG | 256 thr = 32 V-rows × 8 K-slices (`K=128`) [^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu] |
| Occupancy attr | `__launch_bounds__(256)` + `amdgpu_waves_per_eu(2, 4)` — **not** `(1,1)` [^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu] |
| LDS | **0** (register-resident fp32 `h[16]` per thread) [^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu] |
| Inner | scalar fp32 FMA on state; q/k from half. **Not** `fdot2` (state is fp32 recurrent) [^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu] |
| Reduce | `__shfl_xor` 1/2/4 inside the 8 k-slice lanes [^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu] |
| Grid | `(ceil(V/32), B*HV)` [^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu] |
| Gate | gfx10x, `K=128`, fp16 qkv, fp32 state, `use_qk_l2norm=True` [^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu] |

256-thr = 8 waves/WG. On WGP (4 SIMD) that is **2 waves/SIMD**, so `(2,4)` = 1–2 such WGs/WGP. Looser than FA decode `(4,8)`. hipOccupancy still TBD.[^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu]

## Leave

- Spec-decode packed path (still Triton unless extras flipped it)
- Copying 7.6 µs / 9.3× @ B=1 (launch tax vs Triton’s flat ~71 µs; @ B=32 they measured **0.93×**)[^c-69d2efe][^f-69d2efe-gdn-decode-rdna2-cu]

## Capture guard (dest `f9361950`, 2026-09-03)

The `ssm_state` RDNA2 uncommitted-page probe (`torch.isnan(...).any().item()` / all-zero → `zero_()`) is a **device→host sync** and is illegal under stream capture — it aborted cudagraph capture at engine startup with `hipErrorStreamCaptureUnsupported`. Dest now wraps it in `if not torch.cuda.is_current_stream_capturing():`. Python only; the `.cu` tile above is unchanged. Rule: [silicon/hip-craft/graph-capture.md](../../silicon/hip-craft/graph-capture.md).[^c-f9361950]

## NULL_BLOCK_ID=0 sentinel (dest `7779514b`, 2026-09-08)

`gdn_decode_rdna2.cu`: guard was `state_idx < 0 || state_idx >= num_blocks_g`. Production sentinel is **`NULL_BLOCK_ID=0`** (`vllm.v1.attention.backends.utils`). Tip now uses **`state_idx <= 0`** — zero output, leave state untouched. Without it, slot 0 (real weights) was treated as a valid tile and corrupted the GDN recurrence on the first scheduled request. Tile / LDS / `(2,4)` occupancy attr unchanged.[^f-7779514b-gdn-decode-rdna2-cu]

## extras lock 2026-09-17 (tip `50120e13`, engine `388a61b6`)

Python only (`qwen_gdn_linear_attn.py`): removed the one-shot `conv_state.zero_()` / `ssm_state.zero_()` “page-commit sanitizers” that ran on the first `gdn_decode_rdna2` of a process. Those wiped the paged GDN state **after** the first request’s prefill had written recurrent state, so fresh-server long prompts decoded from zeros. KV backing is already zero-initialised by `allocate_kv_cache` — wipes unnecessary and harmful. HIP decode tile / LDS / `(2,4)` unchanged. Supersedes the capture-guard narrative around one-shot `zero_()` as a correctness path; keep host-sync probes illegal under capture ([silicon/hip-craft/graph-capture.md](../../silicon/hip-craft/graph-capture.md)).[^f-50120e13-qwen-gdn-linear-attn-py]

## Sources

[^c-b53a7a2]: BlivionIaG/vllm commit [`b53a7a2`](https://github.com/BlivionIaG/vllm/commit/b53a7a2c2bdf0739daa459ffff169c53a9a0f49e) (2026-08-23) — "feat: hand-port GDN chunked prefill to HIP RDNA2 (5 kernels)".
[^c-69d2efe]: BlivionIaG/vllm commit [`69d2efe`](https://github.com/BlivionIaG/vllm/commit/69d2efea371e52e4dca4c233e198ef627e181f0b) (2026-08-23) — "feat: hand-port GDN packed decode to HIP RDNA2".
[^c-47d92b6]: BlivionIaG/vllm commit [`47d92b6`](https://github.com/BlivionIaG/vllm/commit/47d92b6cfd9dddbc364cd7341c486cb534ac73db) (2026-08-23) — "test: add GDN decode RDNA2 kernel microbench".
[^f-69d2efe-gdn-decode-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/gdn_decode_rdna2.cu` @ [`69d2efe`](https://github.com/BlivionIaG/vllm/blob/69d2efea371e52e4dca4c233e198ef627e181f0b/csrc/rocm/gdn_decode_rdna2.cu).
[^f-7779514b-gdn-decode-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/gdn_decode_rdna2.cu` @ [`7779514b`](https://github.com/opengfx1030/vllm-rdna/blob/7779514b449c358c14cb58c739cc33cd7efca3b4/csrc/rocm/gdn_decode_rdna2.cu).
[^c-f9361950]: opengfx1030/vllm-rdna commit [`f9361950`](https://github.com/opengfx1030/vllm-rdna/commit/f93619500c3e55daee5f60ae7f6d4279cdeb3f0d) (2026-09-03) — "fix(gdn): skip host-sync NaN guard during cudagraph capture".
[^f-50120e13-qwen-gdn-linear-attn-py]: opengfx1030/vllm-rdna `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` @ [`50120e13`](https://github.com/opengfx1030/vllm-rdna/blob/50120e13b46895fc5d074e88fc493f1eef69da9f/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py).
[^gdnd-tip]: `csrc/rocm/gdn_decode_rdna2.cu` L23-L54 (`GDN_BV 32`, `GDN_THREADS 256`, `__launch_bounds__(GDN_THREADS)` + `amdgpu_waves_per_eu(2, 4)`); dispatch gated on `VLLM_GDN_DECODE_RDNA2` != "0" (default on) in `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` L1540-L1548 and L2138-L2144, dest at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/gdn_decode_rdna2.cu#L23-L54 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1540-L1548 ; https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L2138-L2144
[^gdnd-13fb-lb]: `csrc/rocm/gdn_decode_rdna2.cu` L23-L24 and L323-L324, dest at 13fbe2f9. https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/csrc/rocm/gdn_decode_rdna2.cu#L323-L324
[^gdnd-13fb-chk]: `gdn_spec_verify_rdna2` stride checks, `csrc/rocm/gdn_decode_rdna2.cu` L488-L493, dest at 13fbe2f9 (commit [`13fbe2f9`](https://github.com/opengfx1030/vllm-rdna/commit/13fbe2f9160027e71ab6c915ecbcb95d42373a2e), 2026-11-02, "rdna: GDN spec-verify glue - q/k/v, a/b and z read in place, direct output"). https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/csrc/rocm/gdn_decode_rdna2.cu#L488-L493
[^gdnd-b307-chk]: Contiguity check in `gdn_spec_verify_rdna2` at b30757c2. https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/csrc/rocm/gdn_decode_rdna2.cu
[^gdnd-13fb-env]: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` L827-L838, dest at 13fbe2f9. https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L827-L838
[^gdnd-13fb-doc]: `docs/rdna2/tuning.md` (`VLLM_GDN_STRIDED_INPUTS` row) and `docs/rdna2/gdn-decode-investigation.md` section "2026-11-02: the glue around the kernel", dest at 13fbe2f9. https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/docs/rdna2/tuning.md ; https://github.com/opengfx1030/vllm-rdna/blob/13fbe2f9160027e71ab6c915ecbcb95d42373a2e/docs/rdna2/gdn-decode-investigation.md
