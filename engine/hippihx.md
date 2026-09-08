# hippihx — HIP op zoo (locked target)

Date: 2026-09-08. Repo: [BlivionIaG/hippihx](https://github.com/BlivionIaG/hippihx) (private).

**Not** [hippih.md](hippih.md) (older stub / Vega two-layer lab). This page is the V620 methodology lock: **b12x-style zoo**, HIP objects.

## Role

| Layer | Owns |
|---|---|
| **hippihx** | Tile contracts, HIP, fatbins, plan/bind/run, LDS/`__launch_bounds__` |
| **`rdna_extras`** | Thin V1/V2 wiring (`torch.ops`, `RDNA_ATTN`, envs, graphs) |
| **Produce** | EXL3 `-cb 3inst`, AWQ packs — outside the zoo |

Leave: fork of local-inference-lab/b12x CuTe/CE/NVFP4. Take: layering only.

## Tile contracts

- `attn/fa_fdot2`, `attn/gdn_scan`, `attn/kda_scan`, `attn/qsa_indexer`, `attn/dsa_nope`
- `gemm/` W4A16/`fdot2`, EXL3 `3inst`→`fdot2`
- `moe/` routed vs shared vs leftover BF16
- `comm/pcie/` Uncached+push Later (prefer INT8/Q8 wire; Leave `f8_dma` E4M3 without FP8 HW)
- `sequence/` `causal_conv` scalar FMA

Three fatbins: **gfx1030** (ROCm 7.14 wave32) / gfx1100 / gfx900 — no shared objects.

## Bind rules

One `torch.ops`/V1 entry per op (no Triton→HIP double-fire). Scratch from `plan`, zeroed for cudagraph. No D2H under capture.

## Migrate order (after UNC-26)

1. FA (`fa_fdot2`) — lock LDS first
2. EXL3 `3inst`
3. AWQ / `q_gemm` `fdot2`
4. GDN / `causal_conv`
5. Later KDA/QSA/DSA, `comm.pcie`

Do **not** move mid-UNC-26. Dest EXL3 stays In Progress on extras until Done.

## Cards

Project 4: **Later: hippihx HIP op zoo (thin rdna_extras)** + migrate FA / EXL3 / AWQ / GDN children. No new Linear UNC unless CoS boards one.

## Sources

- Room 2026-09-08 (GFX1030 Inference): b12x methodology; hippihx repo + scaffold agent
- [hippih.md](hippih.md) — historical in-house engine card; V620 zoo superseded by this page
- [rdna2-extras.md](rdna2-extras.md), [exl3.md](exl3.md), [coverage.md](coverage.md)
