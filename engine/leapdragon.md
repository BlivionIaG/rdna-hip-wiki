# leapdragon/vllm-rdna2-recipe

Date: 2026-08-21. Engine review of [leapdragon/vllm-rdna2-recipe](https://github.com/leapdragon/vllm-rdna2-recipe) (GPL-3.0-or-later recipe, not a fork). Against **pristine vLLM 0.27.1** — same tag as `rdna2_extras`. Occupancy still first. Do **not** copy their tok/s onto [coverage.md](coverage.md).

Their box: 2× V620 in **×16** slots (measured P2P push **14.3 GB/s** / pull **5.7 GB/s**), TP=2, `Qwen3.8-27B-GPTQ-4bit` (`head_dim=256`, GQA 4, hybrid GDN+full attn), ROCm **7.2.3** in-container, int8-per-token-head KV. Not our 4× / 7.14 / extras HIP stack.

## Already ours

| Their piece | extras / wiki |
|---|---|
| 0001 `on_gfx10x()` + W4A16 gate (PR #52391) | Live on `rdna2_extras`. `on_rdna()` still gfx11/12. |
| 0005 `PYTORCH_ROCM_ARCH=gfx1030` | [vllm-rdna-docker](https://github.com/BlivionIaG/vllm-rdna-docker) / [rocm-host.md](rocm-host.md). |
| “No WMMA; `tl.dot` hits packed DOT” | [fp16-rdna2.md](fp16-rdna2.md) — we want **HIP** `fdot2`, not Triton as the path. |
| Qwen3.5/3.8 hybrid + LDS 64 KiB | [qwen35.md](qwen35.md) already: `TRITON_ATTN` blows 64 KB. |
| IC doesn’t hold 16 layers × 43k KV | [infinity-cache.md](infinity-cache.md): IC helps **this-step** pages, not the whole window. |

## Steal (intent, after occupancy)

Do **not** vendor their plugins into extras (GPL recipe + silent monkey-patch). Re-implement the *intent*.

| Item | Why |
|---|---|
| **0003 softmax segments** | Decode grid is `seqs × kv_heads × segments`. GQA-4 / TP=2 → 32 WGs on 36 WGP — idle. Scale segments to `MIN_LAUNCH_GRID_SIZE_2D`, cap 64. **Occupancy-class**, Triton-attn path. Their plateau: 16→64 segs. |
| **0002 LDS tile @ `head_dim≥256`** | Clamp Triton unified-attn `TILE_*` to 16, `launch_num_stages≤2`. Same 64 KiB cap as [qwen35.md](qwen35.md). `fa_rdna2` still needs its own 256-head LDS budget. |
| **0004 `BLOCK_KN` 128→256** | Exllama GPTQ: they measured 400 vs 365 GB/s; 64 worse, 512 starves small shapes. “Smaller blocks for skinny” was **backwards**. Sweep on extras skinny / W4, don’t copy 256 blind. |
| **Compile mode 3 + `GPU_MAX_HW_QUEUES=4`** | They claim mode 3 cuts elementwise 2553→338. Queues **8 is a 32% regression**. Tune on our 7.14 box, don’t copy. |
| **Force Exllama on stock** | `TritonW4A16LinearKernel` is the ~4 t/s trap if HIP/Exllama isn’t the dispatch. extras already has HIP W4 — profiler-proof it is ours, not Triton. |
| **Harness: IC benches lie** | One layer’s 43k KV (~45 MB) fits 128 MB and reports fake GB/s. Rotate ≥16 buffers. Measure **in-server**, not a Python loop. |
| **P2P alloc** | `hipExtMallocWithFlags(..., hipDeviceMallocUncached)` for exchange. **`hipDeviceMallocFinegrained` is a silent no-op** on this part. Handshake via `hipHostMallocCoherent`. Push, not pull. Later — [plx.md](plx.md) / [deepep.md](deepep.md). |

## Steal the idea, not the plugin

| Plugin | Idea | Our path |
|---|---|---|
| `fd_rdna2` | Flash-decode + byte-sliced `tl.dot` over int8 KV (520 B/entry). Context slope flatten. | [kv-int8.md](kv-int8.md) fused dequant **inside occupancy-fixed `fa_rdna2`**. Triton plugin is fallback / A/B only. |
| `ar_rdna2` | TP=2 **push** one-shot all-reduce. +1.9% on their box. | extras PYNCCL bypass first. Custom AR Later; import *push + uncached*, not their monkey-patch. |

## Dead ends they already paid for

Do not reopen without a new measurement:

- Wide `dwordx4` KV / 544 B align (loads already ~83% peak on their harness)
- Plain FMA instead of DOT (**4.9×** slower)
- Custom W4 GEMV *on top of a healthy Exllama stream* (they were at ~91% BW)
- `GPU_MAX_HW_QUEUES=8`
- Hoisting the attn range-mask (71% slower)

**Our occupancy card is not this dead-end.** Their “custom W4 GEMV is dead” assumes Exllama is already streaming. extras `fa_rdna2` / `skinny_gemms.cu` still sit on `__launch_bounds__(N, 1)` / `waves_per_eu(1,1)`.

## Pitfalls (keep)

- Plugins fail **silent** if logger isn’t `vllm.*`.
- `PYTORCH_TUNABLEOP_ENABLED=1` flips greedy **first run after boot** — not a kernel bug.
- Never mount host ROCm into the image (they mixed 7.1 + TheRock 7.14 → page faults). Live target stays **7.14.0 attested**, image may stay Ubuntu.
- Device-less `torch.cuda.get_arch_list()` returns `[]` with no error.

## Cards

No new first ticket. After occupancy:

1. Softmax-segment fill on the Triton-attn path (or prove `fa_rdna2` already fills the grid).
2. `head_dim=256` LDS clamp on whichever attn actually launches.
3. `BLOCK_KN` sweep on extras W4 — 256 is a hunch to verify.

MTP they left on the table; we already parked it ([mtp.md](mtp.md)).

## Sources

- https://github.com/leapdragon/vllm-rdna2-recipe (`00-HARDWARE.md`, `01-PATCHES.md`, `02-VERSIONS.md`, patches 0001–0005)
- Room 2026-08-21
