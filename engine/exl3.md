# QTIP / EXL3 on gfx1030 — engine spec

Date: 2026-08-22. Engine contract. Silicon: [silicon/exl3.md](../silicon/exl3.md) + [kernels/exl3.md](../kernels/exl3.md) (Later). DSv4 apply: [dsv4-flash-run.md](dsv4-flash-run.md). Occupancy still first.

**Verdict:** QTIP quality-for-size, native HIP. One kernel for a QTIP dump or EXL3 — codebook id + 16×16 pack. Infer is bit-extract + codebook decode → half → `fdot2` on `mxfp4_dot2_moe` (fatter unpack). Viterbi is quant-time. CUDA MMA is dead.

## Frozen vs knobs

Frozen in the format: 16×16 tiles, Hadamard-128, `trellis` + `suh`/`svh`. Infer does not re-Viterbi.

Convert knobs ([doc/convert.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md)):

| Knob | What |
|---|---|
| `-b` | Average bpw 1–8. Allocator picks integer `K` per layer. |
| `-hb` | lm_head bits 1–8 or 16 (default 6). |
| `-cb` | Codebook: `mcg` (default, `0xCBAC1FED`), `mul1` (`0x83DCD12D`), or `3inst`. |
| `-hq` | Bump attn / shared-expert bitrate. **Off for DSv4** — leftover stays official MXFP8. |

Same HIP kernel; dispatch is codebook id. 0xSero K216 is **`mcg`**. If we produce, pick **`3inst`** so unpack matches `decode_3inst`.

RDNA2-optimal convert: REAP official MXFP4 experts → Viterbi keepers only at **3.0–3.5 bpw**, `-cb 3inst`, no `-hq`. 2 bpw is decode-bound on 512 GB/s.

## Infer (native HIP)

- Weight path: bit-extract + `decode_3inst` (or mcg/mul1 LCG) → half → `fdot2`.
- `cb==2` may use `dp4a` **only as the codebook**, not `sdot*` through K.
- Hadamard + `suh`/`svh` are extra VALU around the GEMM.
- Shape: skinny `BLOCK_M` 1/2/4/8 like `mxfp4_dot2_moe`. Occupancy flip on that launch is the EXL3 flip. Do not land on `(1,1)`.

## Sources

- exllamav3 `convert.md` / `quantize.py`; room 2026-08-22 DSv4 knobs
