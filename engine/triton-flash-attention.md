# Stock Triton FlashAttention on gfx1030

Date: 2026-08-18. **Stock configs only.** vLLM `main` @ `49fb2ee`. Tuning knobs live in [triton-tuning.md](triton-tuning.md). Dispatch map: [triton-rocm.md](triton-rocm.md).

HIP `fa_rdna2` is a later A/B, not this page.

## Two stock FA paths

| Name | Prefill | Decode | gfx1030 |
|---|---|---|---|
| `TRITON_ATTN` | unified `kernel_unified_attention` | same kernel (2D or 3D split-softmax) | **Primary baseline** |
| `ROCM_ATTN` | Triton `context_attention_fwd` | Triton `kernel_paged_attention_2d` | Secondary. HIP `paged_attention_rocm` **does not fire** |

Do not label `ROCM_ATTN` "Triton prefill + HIP decode" on this box. That hybrid is CDNA / gfx11 / gfx12.

## `TRITON_ATTN` recorded stock

From `triton_unified_attention.py`:

| Knob | Stock |
|---|---|
| `BLOCK_M` | 16 if GQA≤16 else next pow2 |
| `BLOCK_Q` | `BLOCK_M / GQA` |
| decode `TILE_SIZE` | 16 |
| prefill `TILE_SIZE` | 32 |
| 3D split-softmax | `q==1` and seqs ≤ `128 // num_kv_heads` (graph-snapped), batch-invariance off |
| `NUM_PAR_SOFTMAX_SEGMENTS` | 16 |
| tensor descriptors | off unless XPU |

`--dtype half`. Backend advertises bf16; gfx1030 has no `v_dot2_f32_bf16`.

## `ROCM_ATTN` recorded stock

Prefill `prefix_prefill.py`:

| Knob | Stock |
|---|---|
| `BLOCK_M` / `BLOCK_N` | 128 / 64 (pow2 page); 32 / 32 (nonstandard) |
| `num_unroll_cache` | 4 |
| `num_unroll_request` | 1 |
| `num_warps` | 4 |
| `num_stages` | 1 |

Decode: `kernel_paged_attention_2d` in `chunked_prefill_paged_decode.py`. ikantkode overlay used `num_warps=8` on this kernel — **not stock**. Record stock first, then tune on the other page.

Native HIP gates (for the record, they fail here): CDNA head 64/128; gfx11/12 head 128, block 16, GQA 3–16, unquantized KV, no ALiBi/sinks.

## MLA (stock, unvalidated)

`triton_decode_attention.py` ROCm grouped: `BLOCK_N=16`, `BLOCK_H=16`, `num_warps=4`, `num_stages=1`, `waves_per_eu=1`, `matrix_instr_nonkdim=16`, `kpack=2`. Reduction: `waves_per_eu=4`, `num_stages=2`. Split min-work 512, cap `2*CUs`. No V620 number on this page.

## Compare later (not here)

`fa_rdna2` D=128/256, occupancy ticket, head-64 hole. Same usable KV, same graph mode, same dtype. Prefill and decode scored separately.

## Sources

- `vllm/v1/attention/backends/{triton_attn,rocm_attn}.py`
- `vllm/v1/attention/ops/{triton_unified_attention,prefix_prefill,chunked_prefill_paged_decode,triton_decode_attention}.py`
- `vllm/platforms/rocm.py::use_rocm_custom_paged_attention`
