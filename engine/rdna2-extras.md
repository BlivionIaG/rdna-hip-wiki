# `rdna2_extras` — release-based overlay

Date: 2026-08-19. Engine review of [BlivionIaG/vllm](https://github.com/BlivionIaG/vllm) branch **`rdna2_extras`** @ **`3e05abc9`**.

**Policy (locked):** human work goes on `rdna2_extras` = **vLLM release tag + gfx1030 overlay**. `perf/rdna2_w4a16` is the historical source, not the forward branch. Bots still do not PR upstream or edit the human branch. Occupancy still first.

## What landed

Base: **v0.27.1**. Overlay commit: `9ff87936` (merge). Tip: `3e05abc9` (all-reduce bypass, same intent as `0c59068e`).

Carried: W4A16 / W8A16 / W8A8-FP8-`fdot2` / mxfp4 sources, HIP sparse MLA decode+prefill, indexer radix top-k, `RDNA_ATTN` / `fa_rdna2`, DSv4 gfx10x fp16 routing, causal_conv1d NULL_BLOCK_ID bounds-check.

Merge resolutions that matter:

| File | Kept |
|---|---|
| `causal_conv1d.py` | Branch bounds-check (block-0 alias / MTP) + upstream PDL |
| `deepseek_v4/amd/rocm.py` | Upstream AITER FP8 (CDNA) **and** gfx10x fp16 inv-RoPE |
| `qwen_triton_warmup.py` | Upstream `kv_block_zeroer` (covers block-size 256) + `deepseek_v4` type |
| `spinloop.cpp` | Upstream clang/`x86intrin` |
| `RocmPlatform.use_custom_op_collectives` | **`return False`** — PYNCCL `_all_reduce_out_place` (torch 2.12 + venv-7.14). Broader than gfx10-only; ROCm-wide. Transport only. GEMM still does not own RCCL. |

## Not fixed by the rebase

- Occupancy trap on `fa_rdna2` + `skinny_gemms.cu`
- MLA prefill `load_row` OOB
- Stock skinny still gated (`on_gfx9() or on_gfx1x()`)
- GPU verify still pending on some GEMMs (original commits say so)

Do not treat the rebase as a tok/s win.

## Dispatch watch (v0.27.1)

In `vllm/platforms/rocm.py`:

```text
_ON_GFX10X = gfx10*
_ON_RDNA   = gfx11/gfx12  (not gfx10)
on_rdna()  is FALSE on V620
```

Use `on_gfx10x()` for our kernels. Any new upstream `on_rdna()` gate will **skip gfx1030**. `supports_fp8()` / `supports_mx()` still false here. Default MHA list is still `ROCM_ATTN` then Triton; `fa_rdna2` stays env-gated.

## Next release rebase

Replay list (minimum): `csrc/rocm/*rdna2*`, `fa_rdna2`, `skinny_gemms.cu`, `rocm_rdna2_mla_sparse.py`, `sparse_attn_indexer.py`, `deepseek_v4/amd/rocm.py`, `causal_conv1d.py`, `RocmPlatform.use_custom_op_collectives`, MoE/quant RDNA2 dispatchers.

## Cards

Retip In Progress work to `rdna2_extras` @ `3e05abc9`. Occupancy still first. Same Later list.

## Sources

- `9ff87936` merge message, `3e05abc9` all-reduce
- `vllm/platforms/rocm.py` on extras
- [coverage.md](coverage.md), [fork-delta.md](fork-delta.md)
