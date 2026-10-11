# INT2 + mixed INT2/INT4 MoE — silicon / HIP contract

Spec only. Not in the fork. Occupancy still first. Cards: INT2, mixed INT2/INT4 MoE. Not sdot4-explore (that is W8A8 / Sage / W4A8).

**No i2 DOT on gfx1030.** ISA has `V_DOT2_F32_F16`, `V_DOT4_I32_I8`, `V_DOT8_I32_I4`. INT2 is always **unpack, then an existing DOT**.

Do not confuse with Triton INT2 KV (vLLM #40633 — Hadamard + centroids). That is cache, not weights.

## Pack

16 × i2 per `uint32`, K-contiguous. `K % 16 == 0`. Symmetric signed `{−2,−1,0,1}` or the checkpoint’s codebook — **unknown until a real checkpoint is named**. Group scale (and optional ZP) like W4A16.

```
// 16 codes / dword, LSB-first (lock this when a checkpoint lands)
i2 = (word >> (2*k)) & 0x3
// sign-extend if the codebook is signed two’s complement
```

No LUT in K$. Integer unpack in VGPR.

## Dense — ship W2A16 first

Same job as W4A16, fatter pack:

| Step | Op |
|---|---|
| Load W | `uint32` = 16×i2 |
| unpack + scale | VGPR → `half2` |
| A (fp16) · W | `V_DOT2_F32_F16` |

Tile seed: same as W4A16 (`64×64×32` fp16 after unpack). Decode skinny like `q_gemm_rdna2`.

W2A8 / W2A4 are later and ride the sdot4 explore list: unpack i2→i8 then `sdot4`, or i2→i4 then `sdot8` (W2A4 = both sides i4). Do not open those until W8A8 INT8 exists.

## Mixed INT2/INT4 MoE

Some experts INT4, some INT2 (usually cold experts narrower). **One DOT family per kernel launch.**

| Expert W | A fp16 path | Later A i8 path |
|---|---|---|
| INT4 | existing W4A16 unpack → `fdot2` | W4A8 → `sdot4` |
| INT2 | this unpack → `fdot2` | W2A8 → `sdot4` |

Do **not** switch unpackers inside the K loop. Sort tokens by expert, then group experts by bitwidth (two tiles or two kernels). Shared MoE control flow from `moe_q_gemm_rdna2` (sorted tokens → expert tile → packed CAS / `output_topk`).

LDS: same 8 KB A+B seed after unpack. INT2 B tile is ½ the bytes of INT4 for the same BK — keep BK a multiple of 16 so both packs land on dword loads.

## Do not

- Invent `sdot16` / i2 tensor core.
- Marlin / CUTLASS / AITER fragments.
- Mix INT2 and INT4 in one wave’s inner loop.
- Land this on `waves_per_eu(1,1)`.

## Done-when (when someone writes it)

- ISA dump: integer unpack + `v_dot2_f32_f16` (W2A16). No mystery DOT.
- Writer and reader agree on 16-in-dword order + scales.
- Mixed MoE: bitwidth grouping outside the K loop.
- Smoke vs fp16 / W4A16 experts. No tok/s from this page.

## Sources

- RDNA 2 ISA 70648: DOT2 / DOT4 / DOT8 only
- Live W4A16: [w4a16.md](w4a16.md), `qdq_4_rdna2.cuh`
- sdot4 later: [sdot4-explore.md](sdot4-explore.md)
