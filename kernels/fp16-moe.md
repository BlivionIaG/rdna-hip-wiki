# Native HIP FP16 MoE on gfx1030 — silicon / kernel contract

Date: 2026-08-18. **Spec / Todo.** Engine: [engine/fp16-moe.md](../engine/fp16-moe.md). Target is gfx1030 wave32, fp16 A/B, fp32 accumulation. No MFMA, WMMA, AGPR or bf16.

## Hard ISA contract

```cpp
using h2 = __half2;
float acc = __builtin_amdgcn_fdot2(a2, b2, acc, false);
```

The inner instruction must disassemble to `V_DOT2_F32_F16`/`V_DOT2C_F32_F16`. Do not rely on `__hfma2` peepholing. K is contiguous, even, and addressed as aligned 32-bit `half2`; vector global transfers should be 16 B where aligned. Accumulators stay fp32 until the epilogue.

This is two kernel families. A single compromise tile is rejected.

## Native weight layout

Pack once after loading, never in forward:

- Expert-major outer dimension.
- K contiguous and even: each 32-bit word is two adjacent K values.
- N tiles contiguous so a wave streams coalesced weight words.
- `w13` physical layout `[E, N, 2, K/2]` in `half2`, where the `2` is adjacent gate/up. This permits the same thread to produce a gate/up pair and fuse `silu(gate)*up` without a transpose or second kernel.
- `w2` physical layout `[E, K_out, N_in/2]` in `half2` with output-N tiles contiguous.
- Version the packed layout and reject incompatible strides.

## Decode / tiny expert buckets (`M=1/2/4/8`)

Specialize M at compile time. Seed geometry:

| Item | Contract |
|---|---|
| WG | **128 threads = 4 wave32** for all four M variants |
| N tile | **512 outputs/WG**; each wave owns 128 N, each lane 4 columns |
| K stage | **256 fp16 values**; loop in 128 `half2` pairs |
| A LDS | `M x (256+8)` fp16 = 528 B (M1) to 4224 B (M8) |
| B | stream from global; never stage the FP16 expert tile in LDS |
| Acc/thread | `4*M` fp32 = 4/8/16/32 VGPR before overhead |
| Grid | `(expert row-bucket, ceil(N/512), splitK)` |

One weight `half2` is loaded and reused across all M routed rows before advancing K. That is the reason to keep M in a specialization rather than launching one GEMV per token. A is cooperatively staged/broadcast from LDS; B dominates bytes and is read coalesced.

Start with no split-K. Enable split-K only if `expert buckets x N tiles` cannot fill 72 CUs; reduction must use a separate fp32 workspace or an explicitly proven packed-CAS path, never fp16 accumulation.

### Decode occupancy gate

- Target **<=64 VGPR/lane** and no scratch/local-memory traffic. M8 has 32 accumulator VGPR and is the first pressure gate.
- Record RGA/LLVM VGPR, LDS, spills, waves/SIMD and runtime active blocks for every M specialization.
- Do not force `amdgpu_waves_per_eu(4,8)` until M8 compiles inside the budget. HIP waves-per-EU is not CUDA min-blocks.
- If M8 exceeds 64 VGPR, split it into two M4 passes before accepting spills.

## Prefill / verify grouped GEMM

First compile-and-sweep seed (not a claimed optimum):

| Item | Contract |
|---|---|
| Tile | **BM x BNout x BK = 64 x 64 x 32** |
| WG | **256 threads = 8 wave32** |
| Output ownership | 4x4 fp32 microtile/thread = 16 accumulators |
| LDS/stage | A 64x32 fp16 = 4 KiB; B 32x64 fp16 = 4 KiB |
| Double buffer | **16 KiB/WG** before padding; safely below 64 KiB/WG |
| Grid | grouped persistent work queue over `(expert, M tile, N tile)` |

`BNout=64` means 32 adjacent gate/up pairs for `w13`; for `w2` it is 64 ordinary output channels. Load A/B cooperatively with 16-byte transfers, then consume K as 16 `half2` pairs. Use fp32 accumulators and vector fp16 stores.

Required sweep after the seed works:

- `BM={32,64,128}`, `BNout={32,64}`, `BK={16,32,64}`.
- WG `{128,256}` where output ownership remains regular.
- Single versus double buffering.
- Keep any candidate under 32 KiB LDS if two WGs/WGP is the occupancy goal; reject local-memory spills before timing.

Do not start at 128x128: 64 fp32 outputs/thread at WG256 consumes 64 accumulator VGPR before addresses and staging.

## LDS / 64-bank rules

gfx1030 WGP LDS has **64 banks**, 4 B/bank. Consume LDS as aligned `half2`/dword, never scalar half. Bank is `(byte_addr/4) mod 64`.

- Pad K-major A rows initially by **+8 fp16** (`BK+8` or `Kstage+8`) and verify conflicts from the actual lane map.
- B must be stored/read so neighboring lanes consume neighboring dwords; avoid a column stride that is `0 mod 64` dwords.
- Padding is a sweep variable, not folklore: test `{0,4,8}` half plus an XOR-swizzled variant and retain the ISA/counter evidence.
- Separate `s_waitcnt vmcnt` for global loads from `lgkmcnt` for LDS; do not drain all counters every K step.

## Fused epilogues

- `w13`: same owner computes adjacent gate/up values, applies fp32 `silu(gate)*up`, then converts once to fp16 intermediate.
- `w2`: multiply route weight in fp32 before final fp16 conversion. Combine routed rows only when vLLM TP/EP semantics permit it; no grid-wide dependency inside either GEMM.
- Bias is optional and fused before activation.

## Correctness and performance gates

1. Guard-page/OOB tests for all M/N/K tails and expert sentinel `-1`.
2. Empty/skewed experts and `top_k={1,2,8}`.
3. Compare fp32-accum output and fused SiLU/route-scale against PyTorch.
4. TP=4 and CUDA/HIP graph safety: no allocation, packing or host sync in forward.
5. ISA dump proves `v_dot2_f32_f16`; no MFMA/WMMA and no scratch.
6. Benchmark rows/expert `{1,2,4,8,16,32,64,128}`; sweep crossover by rows **per expert**, not total tokens.
7. Report kernel-only and routed end-to-end, DRAM bytes, L2/Infinity-cache hit rate, VGPR/LDS/occupancy and launch count.

## Landing order

1. Reference op + packed-layout checker.
2. Decode M1, then M2/M4, then M8 after register gate.
3. Fused w13 and w2 epilogues.
4. 64x64x32 grouped prefill seed, then sweep.
5. TP=4 correctness and graph-safe dispatch.

This card remains after the current `fa_rdna2` occupancy fix.