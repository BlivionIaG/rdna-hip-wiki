# EXL3 on gfx1030 — engine spec

Date: 2026-08-21. Engine contract. Silicon tile / ISA dump lives with RDNA2_Researcher under [kernels/](../kernels/README.md) if they add it. Human branch is **`rdna2_extras`**. Occupancy is still first. This is **Later / steal-the-math**, not a current subject.

**Verdict:** EXL3 is a QTIP-variant *weight* format (1–8 bpw). Official vLLM does not serve it. `rdna2_extras` does not either — `exllama.py` is the old GPTQ/AWQ uint4/uint8 kernel. The CUDA runtime walks a tail-biting trellis into Tensor-Core 16×16 MMA. gfx1030 has no WMMA/TC. If we ever take it: unpack trellis → fp16 → `fdot2` (same pattern as mxfp4 / NVFP4), not a Marlin/TC port. Do not displace live W4A16.

## How it is made

Source of truth: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) [`doc/exl3.md`](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md) and [`exl3_lib/quantize.py`](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/quant/exl3_lib/quantize.py). Papers: [QTIP](https://arxiv.org/abs/2406.11235), [QuIP#](https://arxiv.org/abs/2402.04396).

Convert: HF weights + a target bitrate. Hessians computed on the fly. Fused Viterbi kernel. Minutes on small models, hours on 70B, one RTX 4090-class GPU. Not AQLM-expensive.

Offline recipe (locked from the quantizer, not invented):

1. Per-channel RMS + random sign flips (`su` / `sv`)
2. Blockwise Hadamard (output 128×128, plus input Hadamard / scales)
3. Hessian → LDLQ sweep (quantize reverse-order so error is compensated)
4. Per 16×16 tile: Viterbi / tail-biting trellis over a **procedural** codebook
5. Pack encoded states (`pack_trellis`)

Checkpoint tensors (safetensors, HF-like names — unlike EXL2):

| Tensor | Role |
|---|---|
| `trellis` | Packed trellis states / K-bit indices |
| `suh` | Input-channel scales + signs + Hadamard scale (fp16) |
| `svh` | Output-channel scales + signs (fp16) |
| `mcg` / `mul1` | Codebook id. `0xCBAC1FED` or `0x83DCD12D`. LCG, not a stored LUT |

Runtime: `LinearEXL3` → CUDA `exl3_gemv` / `exl3_mgemm`. Codebook is generated in registers. Decode is an FSM walk, then 16×16 Tensor-Core MMA (Marlin-shaped tiles). This is **not** LUT→`fdot2`.

EXL2 is a different format. Do not reuse the vLLM `ExllamaLinearKernel` path.

## Current vLLM support

| Surface | Status |
|---|---|
| Official vLLM | **None.** Quantization folder has no `exl3.py`. [Issue #19896](https://github.com/vllm-project/vllm/issues/19896) stale-closed 2026-04-27. |
| `rdna2_extras` | **None.** Only `vllm/model_executor/kernels/linear/mixed_precision/exllama.py` — GPTQ/AWQ `uint4b8` / `uint8b128`. |
| Aphrodite | [WIP PR #1398](https://github.com/aphrodite-engine/aphrodite-engine/pull/1398) still open (updated 2026-07-29). |
| Community CUDA | e.g. local-inference-lab/vllm PR #139 — `--quantization exl3` + Sparkinfer Trellis, Hopper/SM120 MoE. Not ours. |
| ExLlamaV3 itself | Native server (TabbyAPI). CUDA. |

## gfx1030 gap

| Piece | Take |
|---|---|
| Load / parse `trellis`+`suh`+`svh` | Later. New quant method + weight loader. |
| CUDA `exl3_*` + TC MMA | **Dead.** No WMMA, no Tensor Cores. |
| Unpack → fp16 → `fdot2` | Only plausible HIP path. Same family as mxfp4/NVFP4. |
| Prefill 16×16 TC tile | Does not map. Prefill would be an `fdot2` GEMM after unpack. |
| Decode skinny | Same occupancy contract as live W4. Do not land on `(1,1)`. |
| Convert-on-V620 | Quantizer is CUDA Viterbi. Do not assume a ROCm convert. Use already-converted HF EXL3. |

Zolotukhin (2026-05-17) already called 1.6 bpw EXL3 “no path to RDNA4 yet.” gfx1030 is a harder ISA, not an easier one.

## What we would need (if this becomes a subject)

1. Occupancy flip first. This card does not jump the queue.
2. Silicon: ISA the trellis walk (shifts / XOR / MAD vs a LUT). Confirm we can emit packed half2 and `v_dot2_f32_f16`. No `sdot4` on EXL3 weights (they are not i8).
3. Engine: `QuantizationConfig` + loader for `trellis`/`suh`/`svh`/`mcg`. Gate `on_gfx10x()`. Refuse CUDA kernel names.
4. HIP: fused dequant+GEMM, decode skinny M=1/2/4/8 vs prefill grouped. Same occupancy attributes as `q_gemm_rdna2`.
5. Smoke vs ExLlamaV3 on one small EXL3 (not bit-exact; bound PPL). No copied tok/s.
6. One Project 4 Later card. Do not file a pile.

## Not this ticket

Occupancy. Live W4A16 / W8A16 / mxfp4. EXL2. GGUF / ROCmFPX. Marlin. CUDA `exl3_mgemm`. Official upstream PR.

## Sources

- [exllamav3 doc/exl3.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md), `quantize.py` (`ldlq`, `pack_trellis`, `suh`/`svh`/`mcg`/`mul1`)
- vLLM `#19896` (stale-closed), Aphrodite `#1398` (WIP)
- `rdna2_extras` `mixed_precision/exllama.py` (not EXL3)
- Room 2026-08-21 EXL3 research
