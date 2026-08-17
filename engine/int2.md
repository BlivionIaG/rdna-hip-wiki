# INT2 + mixed INT2/INT4 MoE — engine spec

Date: 2026-08-17. Engine contract. Silicon: [kernels/int2.md](../kernels/int2.md). Cards: INT2, mixed INT2/INT4 MoE on [project 4](https://github.com/users/BlivionIaG/projects/4). Not sdot4-explore (that is W8A8 / Sage / W4A8). Do not edit `perf/rdna2_w4a16` from this page. Occupancy still first.

**Verdict:** no i2 DOT on gfx1030. INT2 is a **weight pack**, not a new inner op. Unpack 16×i2/dword → fp16 → live `fdot2` (W2A16 first). Mixed MoE is the same DOT with **two unpackers**, bitwidth grouped **outside** the K loop. Not on the branch. Not INT2 KV (Triton #40633 is cache, different ticket family).

## Why this is an engine page

Silicon owns the unpack + tile. Engine owns: which checkpoint keys fire it, how MoE tokens are sorted before the kernel, and that we do **not** invent a second GEMM family.

| Piece | Spec |
|---|---|
| Dense | W2A16 — clone `q_gemm_rdna2` control flow. New unpacker only. |
| MoE homogeneous INT2 | clone `moe_q_gemm_rdna2`. Same `BLOCK_SIZE_M` ∈ `{1,2,4,8}`. TP>1: unreduced rows. |
| Mixed INT2/INT4 MoE | sort tokens by expert, **then** group experts by bitwidth. Two launches or two tiles. One DOT family per launch. |
| Dtype | fp16 activations. No bf16. |
| Gate | `on_gfx10x()` + an INT2 / mixed-bitwidth quant key. **Unknown until a checkpoint is named.** |
| Occupancy | no `waves_per_eu(1,1)`. Decode-class: `amdgpu_waves_per_eu(4, 8)`. |

Do not fall through to Marlin / CUTLASS / AITER / bitsandbytes. Do not rewrite the live W4A16 kernels — add an unpack path next to them.

## Mixed INT2/INT4 MoE (the explore mode)

Cold experts narrower (INT2), hot experts INT4. One router, two packs.

```
// keep
sort tokens → expert id
group experts by bitwidth          // outside K
INT4 tile: existing W4 unpack → fdot2
INT2 tile: this unpack       → fdot2
combine / output_topk as today

// drop
switch unpacker inside the K loop
mix INT2 and INT4 in one wave
W2A8 / W2A4 until W8A8 sdot4 exists
INT2 KV on this card
```

Shared MoE control flow stays `moe_q_gemm_rdna2` (sorted tokens → expert tile → packed CAS). Engine work is the **bitwidth grouping** + two method objects, not a new scheduler.

W2A8 / W2A4 ride the later sdot4 list (unpack i2→i8 then `sdot4`). Do not open those until W8A8 INT8 exists. See [kernels/sdot4-explore.md](../kernels/sdot4-explore.md).

## Dispatch vs live GEMMs

| Live today | This spec |
|---|---|
| W4A16 dense + MoE | same job, fatter pack |
| W8A16 / W8A16-FP8 / W8A8-FP8 | unrelated (`fdot2` already). Do not retarget. |
| mxfp4 / NVFP4 | E2M1 codebook, not i2. Separate cards. |
| W8A8 INT8 `sdot4` | later, not this page |

## Done-when

A named INT2 (or mixed) checkpoint loads on gfx1030 without Marlin. ISA dump is integer unpack + `v_dot2_f32_f16`. Mixed MoE groups bitwidth outside K. Smoke vs fp16 / W4A16 experts. No tok/s from this page.

## Not this ticket

Occupancy flip. Sage QK. MLA fat tile. MTP / DFlash / DSpark. INT8 KV. NVFP4. A new DOT opcode.

## Sources

- Silicon: [kernels/int2.md](../kernels/int2.md)
- Live W4A16: [kernels/w4a16.md](../kernels/w4a16.md), `q_gemm_rdna2` / `moe_q_gemm_rdna2`
- MoE engine: [moe.md](moe.md)
- Coverage: [coverage.md](coverage.md)
