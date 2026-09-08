# Triton JIT graph-breaks → HIP AOT (after RMSNorm)

Date: 2026-09-01. Tip **`rdna2_extras` @ `83de31cf`** (read-only). Pattern match for `csrc/rocm/layernorm.cu` (`8e35767f` + brace/mask fixes `f22a15b6` / `83de31cf`). FA pin stays closed. **RMSNorm HIP AOT is not a PIECEWISE fix** — EXL3 dynamo/`allow_in_graph` still left PIECEWISE NaN/duct; this kernel is Order 1 of a **three-culprit JIT plan**, not graph-mode closure. Do not invent tok/s.

User wants HIP over Triton JIT. Scan is the ROCm **decode/prefill graph**, not every `@triton.jit` in the tree.

## What just landed (Leave — already AOT)

`feat(rocm): HIP AOT RMSNorm` (`8e35767f`). Upstream `torch.ops._C.rms_norm` on this box was hitting a **per-shape Triton** `layer_norm_fwd_kernel`: warmup misses a scheduler shape → JIT during inference → captured cudagraph points at a dead binary → NaN logits. Eager is fine (JIT sticks).

Replacement: `csrc/rocm/layernorm.cu` in `_rocm_C`. One CTA/row, `BLOCK_DIM` ∈ {128,256,512,1024} picked by `N`, fp16 in/out, fp32 accum, warp shuffles + tiny LDS cross-warp. Wired in `vllm/_custom_ops.py` and `RMSNorm.forward_rocm`. AOT so the binary is stable across capture/replay.

Locks on that kernel:

- Llama-style `y = x * rstd * w`, **fp16 weight**. Not Gemma `(1+w)` ([qwen35.md](../engine/qwen35.md) still Todo).
- `__launch_bounds__` does not shrink LDS. This kernel is not the FA occupancy leftover.
- Not a reason to reopen FA pin, W8A8 `sdot4`, or EXL3 occupancy.

## Take — same class as RMSNorm (small AOT HIP)

Launch-bound, per-token / per-row, Triton JIT specializes on shape, warmup cannot cover every scheduler emit. Copy the RMSNorm recipe: template a few `BLOCK_DIM`s, `_rocm_C` binding, bypass Triton.

| Candidate | Path | Why Take | Notes |
|---|---|---|---|
| **`causal_conv1d`** | HIP Live @ tip `7779514b` — [causal-conv1d.md](causal-conv1d.md). Triton still fallback. | Prefill HIP before Triton (OK). **UPDATE HIP after Triton** (wire bug — double-fire). Env default ON. | Was Take; now Live with UPDATE-order Leave. Not FA leftover. |
| **`ApplyRotaryEmb.forward_hip`** | `vllm/model_executor/layers/rotary_embedding/common.py` → `flash_attn.ops.triton.rotary.apply_rotary` | Explicit Triton rotary. GridY/Z HIP 65535 fallback to native already exists (#43684 class). | **Not** `RotaryEmbedding.forward_hip`: that calls `ops.rotary_embedding` → `torch.ops._C.rotary_embedding` (`csrc/libtorch_stable/pos_encoding_kernels.cu`, AOT after hipify) unless AITER Triton (gated off). Take only the `ApplyRotaryEmb` / vision-pack path if it is in the captured graph. |
| **`SwigluStepAndMul`** | `vllm/model_executor/layers/activation.py` `_swiglustep_and_mul_kernel` | `forward_cuda` **is** Triton, not `_C`. | Only if a served checkpoint uses this op. Ordinary `SiluAndMul` is `_C.silu_and_mul` (`activation_kernels.cu`) — AOT after hipify; **Leave** unless profiler shows inductor native/Triton instead. |

Order after RMSNorm (commit `8e35767f` text: “3 main Triton JIT culprits”): **conv1d** (`_causal_conv1d_fwd_kernel` / `_update_kernel`, both `@triton.jit`) and **the RoPE path that actually fires** (`ApplyRotaryEmb.forward_hip`) are the remaining two to prove with a profiler, then HIP. Do not guess SwigluStep vs RoPE as #2 vs #3 without a dump.

## Leave — already HIP, or not this class

| Surface | Path | Why Leave |
|---|---|---|
| **FA / `TRITON_ATTN` / prefix_prefill / unified_attention** | `vllm/v1/attention/ops/triton_*.py`, `vllm/v1/attention/backends/triton_attn.py` | `fa_rdna2` is the HIP. Leftover is **LDS 64 KiB WG / prefill `(N,1)`**, not JIT. Pin closed. |
| **EXL3** | `csrc/rocm/exl3_dot2_*.cu` | Live `3inst`→`fdot2`. Produce 3inst; 6bpw `mul1` is lm_head dequant only. Dynamo wrappers are not Triton. |
| **W4 dense/MoE** | `q_gemm_rdna2.cu`, `moe_q_gemm_rdna2.cu` | HIP `fdot2`. Do not port `awq_triton.py` / ikantkode GEMV (`tl.sum`). |
| **`causal_conv1d` HIP** | `csrc/rocm/causal_conv1d_rdna2.cu` | Live AOT @ `7779514b`. UPDATE dispatch order still Leave — [causal-conv1d.md](causal-conv1d.md). |
| **GDN decode + 5 prefill** | `csrc/rocm/gdn_*_rdna2.cu` | HIP Live. Replaces Triton `fused_recurrent_gated_delta_rule_packed_decode` + `chunk_gated_delta_rule` + (claimed) `fused_post_conv_prep`. Occupancy attr `(2,4)`, not FA leftover. |
| **`fused_post_conv_prep` Triton** | `vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py` | Leave **if** `gdn_prefill_prep_rdna2` is the dispatched object. Warmup still imports the Triton helper — that is leftover warmup, not a new Take, until a dump shows it still launches. |
| **W8A8-FP8 / W8A16 / mxfp4** | `csrc/rocm/gemm_w8a8_fp8_*`, `moe_w8a16_*`, `mxfp4_dot2_*` | HIP `fdot2`. Integer W8A8 `sdot4` is **spec, not on branch** — [w8a8.md](w8a8.md). Not a JIT graph-break. |
| **`TritonExperts` / `TritonWNA16Experts`** | `vllm/model_executor/layers/fused_moe/experts/triton_moe.py` | GEMM/MoE, not RMSNorm-class. HIP W4 covers default AWQ/GPTQ MoE. Leftover formats are Later, not AOT-elementwise. |
| **AITER** RMSNorm / rotary / MoE / FA | `vllm/kernels/aiter_ops.py`, `rocm_aiter_ops` | Dead on gfx1030 (`get_cdna_version` / `on_rdna()`). |
| **reshape_and_cache_flash** | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` | Cache write. Second-class vs conv1d/RoPE. `_C_cache_ops.reshape_and_cache*` exists; confirm which fires before HIP. |
| **LoRA Triton, topk_topp_triton, MHC triton, turboquant** | `vllm/lora/ops/triton_ops/*`, `vllm/v1/sample/ops/topk_topp_triton.py`, … | Not the serving graph for V620 extras. |
| **skinny / LLMM1 Triton GEMV** | ikantkode overlay | Leave GEMV. Take the **gate** only ([ikantkode-gfx1030.md](ikantkode-gfx1030.md)). |
| **`fused_qk_norm_rope`** | `vllm/_custom_ops.py` → `torch.ops._C.fused_qk_norm_rope` | **Leave unless a dump shows `_C` missing.** Binding is hipified `_C`, not Triton. Gemma `(1+w)` is a separate fold. |
| **`SiluAndMulWithClamp`** | `activation.py` | ROCm forces `forward_native` (eager Python), **not** Triton. Different class than RMSNorm JIT. |

## How to copy RMSNorm (when taking)

1. New `csrc/rocm/<op>.cu` — fp16, wave32, no WMMA/MFMA/FP8. Issue the real op (`fdot2` only if both sides half and it is a DOT; conv1d/RoPE/silu are **VALU**, not DOT).
2. Append to `VLLM_ROCM_EXT_SRC` (layernorm is unconditional next to EXL3; same is fine).
3. `ops.h` + `torch_bindings.cpp` at **global** scope (the `83de31cf` brace lesson). HIP `__shfl_*_sync` mask is **64-bit** (`f22a15b6`).
4. `_custom_ops.py` / `forward_rocm`: if `_rocm_C` op exists, never enter Triton.
5. Prove with a PIECEWISE probe that **this** kernel’s JIT no longer fires. Do **not** claim PIECEWISE is fixed.

LDS: keep the WG under 64 KiB. Occupancy = min(VGPR, LDS, SIMD). Do not pin `(N, 1)`.

## Wiki gaps this pass

- No `VERIFY` token left on DOT8 / bank pages. DOT8 stays **Explore / W4A4 i4×i4 only** ([w4a4.md](w4a4.md), [silicon/valu.md](../silicon/valu.md)). `ds_read_b128` cycle count still **unknown** ([silicon/lds-tiles.md](../silicon/lds-tiles.md) §8).
- Bank serial walk diagram now in [silicon/lds-tiles.md](../silicon/lds-tiles.md) §1.2 (was formula-only).
- Empty-ish kernel stubs (`lightning-indexer.md`, `mxfp4.md`, `skinny-gemm.md`, `flashkda.md`) are one-pagers by design, not blank.

## Does this change extras / UNC?

**No.** Wiki only. FA pin closed. RMSNorm AOT already on the tip; next HIP is a human extras commit, not this page.

## Sources

- extras commits: `8e35767f` (HIP AOT RMSNorm), `f22a15b6` (64-bit shfl + global-scope entries), `83de31cf` (namespace braces)
- extras files: `csrc/rocm/layernorm.cu`, `CMakeLists.txt` gfx1030 `VLLM_ROCM_EXT_SRC` (~1462–1479) + unconditional EXL3/`layernorm.cu` (~1495–1500), `vllm/_custom_ops.py` `rms_norm` → `_rocm_C` if present, `vllm/model_executor/layers/layernorm.py` `forward_rocm`
- Triton still on ROCm path @ `83de31cf`: `causal_conv1d.py` L16 `_causal_conv1d_fwd_kernel` + L775 `_update_kernel`; `rotary_embedding/common.py` `ApplyRotaryEmb.forward_hip`; `activation.py` `_swiglustep_and_mul_kernel`; `fused_moe/experts/triton_moe.py`; `third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py`; `v1/attention/ops/triton_*.py`
- HIP already: `fa_rdna2.cu`, `q_gemm_rdna2*.cu`, `gdn_*_rdna2.cu`, `exl3_dot2_*.cu`
- GDN HIP vs Triton: [gdn-decode.md](gdn-decode.md), [gdn-prefill.md](gdn-prefill.md)
