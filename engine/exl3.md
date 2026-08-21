# QTIP / EXL3 on gfx1030 — engine spec

Date: 2026-08-21. Engine contract. Silicon: [silicon/exl3.md](../silicon/exl3.md) + [kernels/exl3.md](../kernels/exl3.md) (Later). Human branch is **`rdna2_extras`**. Occupancy is still first. This is **Later / native HIP**, not a current subject.

**Verdict:** The interesting part is **QTIP quality-for-size**. Serve it with **native HIP**, not ExLlamaV3 / Cornell CUDA / Marlin. EXL3 is the convenient checkpoint (QTIP variant, HF-like names). Infer is bit-extract + `decode_3inst` → half → `fdot2`, skinny like `q_gemm_rdna2`. Viterbi is **quant-time only**. CUDA `FragB` / `mma.m16n8k16` is dead. Do not displace live W4A16.

## Quality vs engine

| Want | Take |
|---|---|
| QTIP / EXL3 PPL-per-byte (3–4 bpw first; 2 bpw is decode-bound) | **Yes, Later** |
| Native HIP GEMM (`on_gfx10x()`, occupancy attrs of `q_gemm_rdna2`) | **Yes** |
| ExLlamaV3 runtime, TabbyAPI, Aphrodite, CUDA `exl3_gemv`/`mgemm` | **No** |
| Cornell QTIP CUDA kernels | **No** |
| Convert-on-V620 | **No.** CUDA Viterbi stays the producer. Consume converted weights. |

Start **3–4 bpw**. 1.6 bpw is coherent on 70B but decode-bound on 512 GB/s GDDR6.

## How the weights are made (quant-time)

Source: [QTIP](https://arxiv.org/abs/2406.11235), [QuIP#](https://arxiv.org/abs/2402.04396), [exllamav3 doc/exl3.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md), `exl3_lib/quantize.py`.

1. Per-channel RMS + random sign flips (`su` / `sv`)
2. Blockwise Hadamard-128 (plus input Hadamard / scales) → stored as `suh` / `svh`
3. Hessian → LDLQ sweep
4. Per 16×16 tile: Viterbi tail-biting trellis over a **procedural** codebook
5. `pack_trellis`

| Tensor | Role |
|---|---|
| `trellis` | Packed states / K-bit indices |
| `suh` | Input scales + signs + Hadamard (fp16) |
| `svh` | Output scales + signs (fp16) |
| `mcg` / `mul1` | Codebook id (`0xCBAC1FED` / `0x83DCD12D`). LCG, no VRAM LUT |

EXL2 is a different format. Do not reuse vLLM `ExllamaLinearKernel`.

## Infer (native HIP — silicon lock)

Room 2026-08-21 @RDNA2_Researcher:

- Viterbi does **not** run at infer.
- Weight path: bit-extract + `decode_3inst` (mul + LOP3-emulate or byte-sum) → half → `fdot2`.
- `cb==2` may use `dp4a` **only as the codebook**, not `sdot*` through K.
- Hadamard-128 + `suh`/`svh` are extra VALU around the GEMM.
- CUDA `FragB` / `mma.m16n8k16` is dead on V620.

Engine shape: decode skinny M=1/2/4/8 like `q_gemm_rdna2`; prefill is an `fdot2` GEMM after the same unpack. Same occupancy contract. Do not land on `(1,1)`.

## Current vLLM support

| Surface | Status |
|---|---|
| Official vLLM | **None.** [Issue #19896](https://github.com/vllm-project/vllm/issues/19896) stale-closed. |
| `rdna2_extras` | **None.** `mixed_precision/exllama.py` is GPTQ/AWQ uint4/uint8. |
| Aphrodite `#1398` | Still WIP. CUDA. |
| Community CUDA forks | Not ours. |

## What we would need (if this becomes a subject)

1. Occupancy flip first.
2. Loader for `trellis`/`suh`/`svh`/`mcg` (or a QTIP-equivalent pack we own). Gate `on_gfx10x()`.
3. HIP: fused `decode_3inst` + `fdot2`, skinny + prefill. Refuse CUDA kernel names.
4. Smoke vs ExLlamaV3 PPL on one small 3–4 bpw EXL3 (not bit-exact). No copied tok/s.
5. Wiki only until someone asks for a Project 4 Later card.

`hippih` can own a three-ISA format later. extras is the first serving shell, not a second ISA tree.

## Not this ticket

Occupancy. Live W4A16 / W8A16 / mxfp4. EXL2. GGUF / ROCmFPX. Marlin. CUDA `exl3_mgemm`. Official upstream PR.

## Sources

- QTIP / QuIP# papers; exllamav3 `quantize.py` (`ldlq`, `pack_trellis`, `suh`/`svh`/`mcg`/`mul1`)
- Room 2026-08-21: quality-for-size + native HIP; silicon `decode_3inst` → `fdot2`
