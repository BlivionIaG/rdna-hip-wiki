# Infinity Cache — how inference rides it

Date: 2026-08-19. Engine contract. SKU table: [silicon/infinity-cache.md](../silicon/infinity-cache.md). HIP knobs + fit math: [silicon/cache-policy.md](../silicon/cache-policy.md). Occupancy still first. Do not invent tok/s or IC TB/s.

**Yes, we use it.** Not as a switch. IC is already on every GPU-memory path (MALL). The only persist mechanism on gfx1030 is **fit the hot set + reread it + default loads + don't thrash**. HIP has no persist bit, no prefetch-into-IC, no IC bypass.

V620 = **128 MB** on-die L3, 64 B lines, after 4 MB L2, before GDDR6 512 GB/s. W7800 = 96 MB on MCDs. V340L = **none**.

## What actually speeds decode

| Traffic | Policy | Why |
|---|---|---|
| Layer shard that **fits** | default `load` | Want L2 LRU + MALL fill. Nontemporal would Stream L2 for nothing. |
| Layer shard that **misses** | `__builtin_nontemporal_load` | One-shot. Keep IC for `x[]`, scales, live KV. |
| Activation / scales | default | Tiny, reused. Not `__constant__` for a LUT. |
| KV pages **this step touches** | default | Paged walk of the whole cache still streams GDDR6. |
| Write-once residual / decode C | `__builtin_nontemporal_store` | GPUOpen Laplacian pattern. |

Continuous batching helps because the **same shard** is reread every token. Chunked prefill keeps decode skinny so we are not blowing IC with a fat GEMM at the same time.

## Fit on this box (TP=4, capacity not a promise)

From [cache-policy.md](../silicon/cache-policy.md) §4. Numbers are **bytes vs 128 MiB**, not measured hits.

| Workload | Shard vs IC | Weight path |
|---|---|---|
| 7B W4A16 / mxfp4 TP=4 | ~29–31 MiB — **fits**, leftover ~97 MiB | **keep** (default). Leftover can hold thousands of GQA KV tokens. |
| 7B W8A8 TP=4 | ~56 MiB — fits | **keep** |
| 7B FP16 TP=4 | ~111 MiB — fits, leftover ~17 MiB | **keep**, short KV only |
| 27B W4 / mxfp4 TP=4 | ~70–76 MiB — fits | **keep**. Leftover KV is hundreds of tokens, not the whole window. |
| 27B W8A8 TP=4 | 135 MiB — **7 MiB over** | **stream** weights, or drop to W4 |
| 27B FP16 TP=4 | 2.1× IC | **stream** |
| 7B W4 **no TP** | 115–125 MiB — tight | keep only if the GPU is quiet |

L2 (4 MB) holds **none** of these layers. L2 is the active panel / `x` / scales. A layer that “fits IC” still misses L2.

Prefill fat `M≥32` GEMM working sets are **larger than 128 MB**. IC does not make rocBLAS free. Do not retile prefill “for IC.”

## Occupancy is the IC knob we actually have

Extra waves on a miss-path GEMV multiply outstanding GDDR6 and **evict** the 128 MB. That is why `fa_rdna2` / `skinny_gemms.cu` occupancy is still first — a perfect fit with `__launch_bounds__(N, 1)` / `waves_per_eu(1,1)` is not the win.

Two HIP streams or SDMA on the same V620 **fight** the same 128 MB. A “layer lives in IC” claim is only true if that GPU is otherwise quiet. Collectives stay outside GEMM ([rdna2-extras.md](rdna2-extras.md) PYNCCL bypass).

Do **not** paste gfx940 `sc0`/`sc1` persist, gfx11 MALL NOALLOC (DLC), or `llvm.prefetch` (ignored on gfx1030).

## Engine rules

1. Occupancy flip first. IC reuse is a **working-set** property of that kernel, not a new first ticket.
2. `--dtype half`. W4A16 / mxfp4 on the keep path; W8A8 keep only when the shard fits.
3. Nontemporal **only** the miss-path stream. Never nontemporal a 7B W4 TP=4 shard.
4. FA: default loads on the live pages. INT8-KV ([kv-int8.md](kv-int8.md)) shrinks the window, not a persist API.
5. Hetero 2+8 ([multi-tier.md](multi-tier.md)): IC is **per GPU**. W7800 attn/KV is 96 MB MCD, not V620's 128. V340L has no IC — mix/FMA, own host.
6. hippih later: same policy, three TUs. Do not invent an IC-pin backend.
7. Do not invent TB/s. AMD “2.4×” is relative, not a V620 number.

## Not a card

No new first kernel. Occupancy + V620 HIP MoE still first. Nontemporal on the 27B miss path is a Later polish after occupancy is measured.

## Sources

- [silicon/infinity-cache.md](../silicon/infinity-cache.md), [silicon/cache-policy.md](../silicon/cache-policy.md)
- Room 2026-08-19 (user: can we use IC to speed inference)
