---
title: "INT2 + mixed INT2/INT4 MoE — silicon / HIP contract"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/vllm-project/vllm/pull/40633
  - https://github.com/opengfx1030/vllm-rdna/blob/56f67111dd87847c5e92e14a7076fec2c9ab6f22/csrc/rocm/q_gemm_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/56f67111dd87847c5e92e14a7076fec2c9ab6f22/csrc/rocm/moe_q_gemm_rdna2.cu
---
# INT2 + mixed INT2/INT4 MoE — silicon / HIP contract

Spec only. Not in the fork. Occupancy still first. Cards: INT2, mixed INT2/INT4 MoE. Not sdot4-explore (that is W8A8 / Sage / W4A8).[^isa]

**No i2 DOT on gfx1030.** ISA has `V_DOT2_F32_F16`, `V_DOT4_I32_I8`, `V_DOT8_I32_I4`. INT2 is always **unpack, then an existing DOT**.[^isa]

Do not confuse with Triton INT2 KV (vLLM #40633 — Hadamard + centroids). That is cache, not weights.[^vpr-40633]

## Pack

16 × i2 per `uint32`, K-contiguous. `K % 16 == 0`. Symmetric signed `{−2,−1,0,1}` or the checkpoint’s codebook — **unknown until a real checkpoint is named**. Group scale (and optional ZP) like W4A16.[^isa]

```
// 16 codes / dword, LSB-first (lock this when a checkpoint lands)
i2 = (word >> (2*k)) & 0x3
// sign-extend if the codebook is signed two’s complement
```

No LUT in K$. Integer unpack in VGPR.[^isa]

## Dense — ship W2A16 first

Same job as W4A16, fatter pack:[^isa]

| Step | Op |
|---|---|
| Load W | `uint32` = 16×i2 [^isa] |
| unpack + scale | VGPR → `half2` [^isa] |
| A (fp16) · W | `V_DOT2_F32_F16` [^isa] |

Tile seed: same as W4A16 (`64×64×32` fp16 after unpack). Decode skinny like `q_gemm_rdna2`.[^f-56f67111-q-gemm-rdna2-cu]

W2A8 / W2A4 are later and ride the sdot4 explore list: unpack i2→i8 then `sdot4`, or i2→i4 then `sdot8` (W2A4 = both sides i4). Do not open those until W8A8 INT8 exists.[^isa]

## Mixed INT2/INT4 MoE

Some experts INT4, some INT2 (usually cold experts narrower). **One DOT family per kernel launch.**[^isa]

| Expert W | A fp16 path | Later A i8 path [^isa] |
|---|---|---|
| INT4 | existing W4A16 unpack → `fdot2` | W4A8 → `sdot4` [^isa] |
| INT2 | this unpack → `fdot2` | W2A8 → `sdot4` [^isa] |

Do **not** switch unpackers inside the K loop. Sort tokens by expert, then group experts by bitwidth (two tiles or two kernels). Shared MoE control flow from `moe_q_gemm_rdna2` (sorted tokens → expert tile → packed CAS / `output_topk`).[^f-56f67111-moe-q-gemm-rdna2-cu]

LDS: same 8 KB A+B seed after unpack. INT2 B tile is ½ the bytes of INT4 for the same BK — keep BK a multiple of 16 so both packs land on dword loads.[^isa]

## Do not

- Invent `sdot16` / i2 tensor core.[^isa]
- Marlin / CUTLASS / AITER fragments.[^isa]
- Mix INT2 and INT4 in one wave’s inner loop.[^isa]
- Land this on `waves_per_eu(1,1)`.[^isa]

## Done-when (when someone writes it)

- ISA dump: integer unpack + `v_dot2_f32_f16` (W2A16). No mystery DOT.[^isa]
- Writer and reader agree on 16-in-dword order + scales.[^isa]
- Mixed MoE: bitwidth grouping outside the K loop.
- Smoke vs fp16 / W4A16 experts. No tok/s from this page.[^isa]

## Sources

- RDNA 2 ISA 70648: DOT2 / DOT4 / DOT8 only
- Live W4A16: [w4a16.md](w4a16.md), `qdq_4_rdna2.cuh`
- sdot4 later: [sdot4-explore.md](sdot4-explore.md)

[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^vpr-40633]: vLLM PR #40633 — Triton INT4 / INT2 per-token-head KV cache quantization. <https://github.com/vllm-project/vllm/pull/40633>
[^f-56f67111-q-gemm-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/q_gemm_rdna2.cu` @ [`56f67111`](https://github.com/opengfx1030/vllm-rdna/blob/56f67111dd87847c5e92e14a7076fec2c9ab6f22/csrc/rocm/q_gemm_rdna2.cu).
[^f-56f67111-moe-q-gemm-rdna2-cu]: opengfx1030/vllm-rdna `csrc/rocm/moe_q_gemm_rdna2.cu` @ [`56f67111`](https://github.com/opengfx1030/vllm-rdna/blob/56f67111dd87847c5e92e14a7076fec2c9ab6f22/csrc/rocm/moe_q_gemm_rdna2.cu).
