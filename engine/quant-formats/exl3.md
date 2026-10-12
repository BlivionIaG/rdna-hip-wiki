# QTIP / EXL3 on gfx1030 — engine spec

Date: 2026-08-22 (live extras retip 2026-08-28). Engine contract. Silicon: [silicon/formats/exl3.md](../../silicon/formats/exl3.md) + [kernels/quant/exl3.md](../../kernels/quant/exl3.md). DSv4 apply: [dsv4-flash-run.md](../run-recipes/dsv4-flash-run.md). Occupancy still first.

**Verdict:** QTIP quality-for-size, native HIP. One kernel, codebook id + 16×16 pack. Infer does **not** retune the trellis. Produce `mul1` (`-cb mul1`, replaces `-cb 3inst`) (Blivion decision 2026-10-12, unsourced upstream). Still compile `3inst`/`mcg` (`cb=0/1`) for existing checkpoints such as 0xSero K216. Pair two states → `half2` → one `fdot2`.

## Live extras (2026-08-28)

*(historical)* `rdna2_extras` @ `e268c7d3` (BlivionIaG/vllm; current dest is `opengfx1030/vllm-rdna` `rdna_extras`). HIP files landed at `a2c8d5cf` (`exl3_dot2_dense` / `moe` / `hadamard`), not a rewrite of `mxfp4_dot2_moe`. `e268c7d3` is unused-var compile fix only. Python `38bdfec5` captures `*.mul1` / `*.mcg` markers and folds those layers (plus fused suh and bits-6 lm_head) to fp16; unmarked 2/3/4-bit single-shard stays on HIP. `VLLM_EXL3_FOLDED_CACHE` is the no-exllamav3 serve path. Not default AWQ/GPTQ.

## Frozen vs knobs

Frozen: 16×16 tiles, `trellis` + `suh`/`svh`. Hadamard-128 is **required** when `suh`/`svh` exist (cheap vs expert GDDR).

Runtime knobs: integer `K` as a **template** (one bpw per launch, don’t mix `K` in a WG) and skinny `BLOCK_M`.

Convert knobs:

| Knob | What |
|---|---|
| `-b` | Average bpw 1–8. We launch one integer `K` at a time. |
| `-hb` | lm_head bits 1–8 or 16 (default 6). |
| `-cb` | Produce **`mul1`** (Blivion decision 2026-10-12, unsourced upstream). Compile **`3inst`** and **`mcg`** for existing checkpoints. |
| `-hq` | **Off for DSv4** — leftover stays official MXFP8. |

RDNA2-optimal convert: REAP official MXFP4 experts → Viterbi keepers only at **3.0–3.5 bpw**, `-cb mul1` (Blivion decision 2026-10-12, unsourced upstream), no `-hq`. 2 bpw is decode-bound on 512 GB/s.

## Infer (native HIP)

- Bit-extract + codebook decode → pair states → `half2` → one `fdot2`.
- `cb==2` may use `dp4a` only as the codebook, not `sdot*` through K.
- Occupancy flip on the `mxfp4_dot2_moe` launch is the EXL3 flip. Do not land on `(1,1)`.

## Sources

- Note 2026-08-22: `K` template, `3inst` produce / `mcg` compile, no `mul1`, pair→`half2`

Added in the 2026-10-11 sourcing pass:

- exllamav3 repository (EXL3) — <https://github.com/turboderp-org/exllamav3>
- Tseng et al., *QTIP: Quantization with Trellises and Incoherence Processing* — <https://arxiv.org/abs/2406.11235>

