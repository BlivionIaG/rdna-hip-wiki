# `rdna2_extras` — release-based overlay (historical)

> **Dest today (2026-10-11):** [`opengfx1030/vllm-rdna` `rdna_extras`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras) = vLLM **v0.31.0** port ([PR #47](https://github.com/opengfx1030/vllm-rdna/pull/47)), tip `1dc3bac4` (contains `8abf4c97`). Tips and branch names quoted below (`rdna2_extras` on BlivionIaG/vllm, `perf/rdna2_w4a16`) are **historical snapshots**. Serve locks (ROCm pin, all-reduce path, prefix caching) live in [fork/serve-locks.md](../../fork/serve-locks.md) (fork PR #10) and [toolchain/pin/README.md](../../toolchain/pin/README.md); this page only says what the engine does.

Date: 2026-08-19. Engine review of [BlivionIaG/vllm](https://github.com/BlivionIaG/vllm) branch **`rdna2_extras`** @ **`3e05abc9`**.

**Policy (locked 2026-08-19, historical branch name):** human work went on `rdna2_extras` = **vLLM release tag + gfx1030 overlay**. `perf/rdna2_w4a16` is the historical source, not the forward branch. Bots still do not PR upstream or edit the human branch. Occupancy still first.

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
| `RocmPlatform.use_custom_op_collectives` | *(historical @ `3e05abc9`; current path: [fork/serve-locks.md](../../fork/serve-locks.md))* **`return False`** — PYNCCL `_all_reduce_out_place` (torch 2.12 + venv-7.14). Broader than gfx10-only; ROCm-wide. Transport only. GEMM still does not own RCCL. |

## Not fixed by the rebase

- Occupancy trap on `fa_rdna2` + `skinny_gemms.cu` (`__launch_bounds__(N, 1)` / `wvSplitKrc_` `(1,1)`)
- MLA prefill `load_row` OOB
- Stock skinny still gated (`on_gfx9() or on_gfx1x()`)
- GPU verify still pending on some GEMMs (original commits say so)
- **CMake gap:** `moe_w8a16_fp8_rdna2.cu` is on the branch but **not** in the gfx1030 source list — W8A16-FP8 MoE is sources-only / Fallback until listed

Do not treat the rebase as a tok/s win.

## Dispatch watch (v0.27.1)

In `vllm/platforms/rocm.py`:

```text
_ON_GFX10X = gfx10*
_ON_RDNA   = gfx11/gfx12  (not gfx10)
on_rdna()  is FALSE on V620
```

Use `on_gfx10x()` for our kernels. Any new upstream `on_rdna()` gate will **skip gfx1030**. `supports_fp8()` / `supports_mx()` still false here. Default MHA list is still `ROCM_ATTN` then Triton; `fa_rdna2` stays env-gated. Custom paged-attn now allows gfx10x — profiler-proof it is `fa_rdna2`.

## Sibling repos (access 2026-08-19)

| Repo | Role |
|---|---|
| [vllm-rdna-docker](https://github.com/BlivionIaG/vllm-rdna-docker) | Image build. Stay on extras, not ikantkode overlay. |
| [v620_toolbox](https://github.com/BlivionIaG/v620_toolbox) | Power + `pcie_p2p`. Later measured-P2P source for DeepEP/hetero. Not first. |
| [hippih](https://github.com/BlivionIaG/hippih) (private repo) | In-house HIP engine (README stub). Path #3 after extras + Llaminar. Contract: [hippih.md](../alt-engines/hippih.md). Do not start before occupancy. |

## Next release rebase

Replay list (minimum): `csrc/rocm/*rdna2*`, `fa_rdna2`, `skinny_gemms.cu`, `rocm_rdna2_mla_sparse.py`, `sparse_attn_indexer.py`, `deepseek_v4/amd/rocm.py`, `causal_conv1d.py`, `RocmPlatform.use_custom_op_collectives`, MoE/quant RDNA2 dispatchers, **CMake gfx1030 source list** (`moe_w8a16_fp8_rdna2.cu`).

## Cards

*(historical)* Retip In Progress work to `rdna2_extras` @ `3e05abc9`. Today's tip is `opengfx1030/vllm-rdna` `rdna_extras` @ `1dc3bac4`; tip bumps belong to [fork/](../../fork/README.md). Occupancy still first. Same Later list. CMake-gap issue on project 4 (W8A16-FP8 MoE not compiled).

## Sources

- `9ff87936` merge message, `3e05abc9` all-reduce
- `vllm/platforms/rocm.py` on extras
- RDNA2_Researcher HIP review (`kernels/overlays/rdna2-extras.md`)
- [coverage.md](../overview/coverage.md), [fork-delta.md](fork-delta.md), [hippih.md](../alt-engines/hippih.md)

Added in the 2026-10-11 sourcing pass:

- dest `docs/rdna2/v0.31.0-port-notes.md` — <https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/v0.31.0-port-notes.md>
- dest commit `8abf4c97` (v0.31 port merge tail; ancestor of tip `1dc3bac4`) — <https://github.com/opengfx1030/vllm-rdna/commit/8abf4c9719d588349b6ed5afc75e9d0133a43984>

