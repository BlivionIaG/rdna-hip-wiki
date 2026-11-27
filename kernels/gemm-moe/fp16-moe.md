---
title: "Native HIP FP16 MoE on gfx1030 — silicon / kernel contract"
sources:
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/csrc/rocm/moe_q_gemm_rdna3.cu
  - https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/fa_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/8d141bdc80b3124315a4a1e52ab4485cb6974f5e
  - https://github.com/opengfx1030/vllm-rdna/commit/9a706f6a48120f00a3a80df731834fb900c0ca64
  - https://github.com/opengfx1030/vllm-rdna/pull/221
  - https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/csrc/rocm/moe_unquantized_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/vllm/model_executor/layers/fused_moe/experts/rdna2_unquantized_moe.py
  - https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/CMakeLists.txt
---
# Native HIP FP16 MoE on gfx1030 — silicon / kernel contract

Date: 2026-08-18. **Spec / Todo.** Engine: [engine/moe/fp16-moe.md](../../engine/moe/fp16-moe.md). Target is gfx1030 wave32, fp16 A/B, fp32 accumulation. No MFMA, WMMA, AGPR or bf16.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

## Hard ISA contract

```cpp
using h2 = __half2;
float acc = __builtin_amdgcn_fdot2(a2, b2, acc, false);
```

The inner instruction must disassemble to `V_DOT2_F32_F16`/`V_DOT2C_F32_F16`. Do not rely on `__hfma2` peepholing. K is contiguous, even, and addressed as aligned 32-bit `half2`; vector global transfers should be 16 B where aligned. Accumulators stay fp32 until the epilogue.[^isa]

This is two kernel families. A single compromise tile is rejected.

## Native weight layout

Pack once after loading, never in forward:

- Expert-major outer dimension.
- K contiguous and even: each 32-bit word is two adjacent K values.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- N tiles contiguous so a wave streams coalesced weight words.
- `w13` physical layout `[E, N, 2, K/2]` in `half2`, where the `2` is adjacent gate/up. This permits the same thread to produce a gate/up pair and fuse `silu(gate)*up` without a transpose or second kernel.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- `w2` physical layout `[E, K_out, N_in/2]` in `half2` with output-N tiles contiguous.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- Version the packed layout and reject incompatible strides.

## Decode / tiny expert buckets (`M=1/2/4/8`)

Specialize M at compile time. Seed geometry:

| Item | Contract |
|---|---|
| WG | **128 threads = 4 wave32** for all four M variants [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| N tile | **512 outputs/WG**; each wave owns 128 N, each lane 4 columns [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| K stage | **256 fp16 values**; loop in 128 `half2` pairs [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| A LDS | `M x (256+8)` fp16 = 528 B (M1) to 4224 B (M8) [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| B | stream from global; never stage the FP16 expert tile in LDS [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| Acc/thread | `4*M` fp32 = 4/8/16/32 VGPR before overhead [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| Grid | `(expert row-bucket, ceil(N/512), splitK)` [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |

One weight `half2` is loaded and reused across all M routed rows before advancing K. That is the reason to keep M in a specialization rather than launching one GEMV per token. A is cooperatively staged/broadcast from LDS; B dominates bytes and is read coalesced.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

Start with no split-K. Enable split-K only if `expert buckets x N tiles` cannot fill 72 CUs; reduction must use a separate fp32 workspace or an explicitly proven packed-CAS path, never fp16 accumulation.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

### Decode occupancy gate

- Target **<=64 VGPR/lane** and no scratch/local-memory traffic. M8 has 32 accumulator VGPR and is the first pressure gate.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- Record RGA/LLVM VGPR, LDS, spills, waves/SIMD and runtime active blocks for every M specialization.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- Do not force `amdgpu_waves_per_eu(4,8)` until M8 compiles inside the budget. HIP waves-per-EU is not CUDA min-blocks.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- If M8 exceeds 64 VGPR, split it into two M4 passes before accepting spills.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

## Prefill / verify grouped GEMM

First compile-and-sweep seed (not a claimed optimum):

| Item | Contract |
|---|---|
| Tile | **BM x BNout x BK = 64 x 64 x 32** [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| WG | **256 threads = 8 wave32** [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| Output ownership | 4x4 fp32 microtile/thread = 16 accumulators [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| LDS/stage | A 64x32 fp16 = 4 KiB; B 32x64 fp16 = 4 KiB [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| Double buffer | **16 KiB/WG** before padding; safely below 64 KiB/WG [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |
| Grid | grouped persistent work queue over `(expert, M tile, N tile)` [^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu] |

`BNout=64` means 32 adjacent gate/up pairs for `w13`; for `w2` it is 64 ordinary output channels. Load A/B cooperatively with 16-byte transfers, then consume K as 16 `half2` pairs. Use fp32 accumulators and vector fp16 stores.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

Required sweep after the seed works:

- `BM={32,64,128}`, `BNout={32,64}`, `BK={16,32,64}`.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- WG `{128,256}` where output ownership remains regular.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- Single versus double buffering.
- Keep any candidate under 32 KiB LDS if two WGs/WGP is the occupancy goal; reject local-memory spills before timing.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

Do not start at 128x128: 64 fp32 outputs/thread at WG256 consumes 64 accumulator VGPR before addresses and staging.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

## Dest kernel @ `8d141bdc`: grouped FP16 MoE prefill `moe_gemm_unquantized_prefill_kernel` (2026-11-27)

PR #221 (`9a706f6a`) adds a grouped fp16 prefill GEMM for unquantized MoE experts (the Flash-Next MTP layer) to `moe_unquantized_rdna2.cu`. It is a much smaller tile than the 64 × 64 × 32 / WG 256 seed above.[^t8d-tip][^t8d-c221]

**Occupancy first.** The VGPR count is the PR author's (ROCm 10.0 build of `cda6ef6e`), not from the 7.14.0 pin. Everything else is read from the source; the WG/WGP figures are arithmetic.[^t8d-m-kernel][^t8d-pr221]

| Item | At `8d141bdc` |
|---|---|
| Tile | BM 16 (sorted slots) × BN 64 × BK 32, TM 4 × TN 4 = 16 fp32 accumulators per thread |
| Block | 64 threads = 2 wave32 (TX 16 × TY 4); `__launch_bounds__(64)`, no `waves_per_eu` |
| Static LDS | `As[2][16][36]` + `Bs[2][64][36]` fp16 + `sh_slot[16]` + `sh_arow[16]` = 2,304 + 9,216 + 128 = **11,648 B** |
| WG/WGP (arith.) | LDS: floor(131,072 / 11,648) = **11** (22 waves). VGPR 61 → 64 → 16 waves/SIMD allows 32 such WGs. So LDS-bound at 11; any per-WGP workgroup-slot cap is not checked here (unsourced) |
| DOT | `__builtin_amdgcn_fdot2` only (fp16 × fp16 → fp32, no clamp), 2 per (i, j) per 4-K step, `#pragma unroll 1` over BK. No `sdot4` / `sdot8` / WMMA |
| Pipeline | double-buffered LDS, global loads for tile k+1 issued before compute on tile k, one barrier per K tile. A: 1 uint4 per thread, B: 4 uint4 per thread (rows clamped to `N_out − 1`) |
| Grid / output | `(num_blocks from moe_align_block_size(…, 16), ceil(N_out / 64))`. Every (slot, column) is written once, with no atomics. The down pass writes `slot_out[num_pairs, K]` (topk weight in the epilogue) and `ops.moe_sum` reduces it. `silu_and_mul` runs between the two GEMMs |

- **LDS row stride.** LD = BK + 4 = 36 halves = 72 B. Odd rows start at 8 mod 16 B, so the `uint4` stores into `As` / `Bs` are 8-B aligned, not 16-B aligned (arithmetic). This relies on gfx10 accepting unaligned 16-B LDS access. Whether hipcc emits `ds_write_b128` here or splits it, and the runtime's LDS alignment mode, are not checked (unsourced). The b64 fragment reads are 8-B aligned. Under the 64-bank formula, the `Bs` b64 read (16 distinct rows × 18-dword stride) is conflict-free, and `As` reads are 2-address broadcasts (arithmetic).[^t8d-m-kernel]
- **Default on, no env gate.** `Rdna2UnquantizedExperts` takes this path whenever `num_tokens ≥ 64`, K % 32 == 0, `n_dim` % 32 == 0 and the op is present. Below 64 tokens the pair-blocked GEMV stays. The PR reports relL2 = 0.000618 vs its reference, so it is not bit-identical to the old path (author's figure, not re-run here).[^t8d-m-py][^t8d-pr221]
- **No arch guard.** The file has no `#if __gfx…__` block. It is only in the gfx1030-only CMake list, but in a multi-arch build every listed arch's device pass compiles it. Every arch in `HIP_SUPPORTED_ARCHS` at this tip has a packed fp16 dot instruction, so this is not a build break today. Since `cda6ef6e`, the sibling W8A8 file has a gfx103x guard.[^t8d-m-kernel][^t8d-cmake][^t8d-cmake-archs]
- **Per-file `-O3`.** `moe_unquantized_rdna2.hip` is now in the `-O3` list (CMakeLists L1891), so the existing decode `gate_up_silu_kernel` / `down_kernel` in the same file are built at `-O3` too (RelWithDebInfo would give `-O2`). Their register use was not re-checked after this change.[^t8d-cmake]

## LDS / 64-bank rules

gfx1030 WGP LDS has **64 banks**, 4 B/bank. Consume LDS as aligned `half2`/dword, never scalar half. Bank is `(byte_addr/4) mod 64`.[^isa]

- Pad K-major A rows initially by **+8 fp16** (`BK+8` or `Kstage+8`) and verify conflicts from the actual lane map.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- B must be stored/read so neighboring lanes consume neighboring dwords; avoid a column stride that is `0 mod 64` dwords.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- Padding is a sweep variable, not folklore: test `{0,4,8}` half plus an XOR-swizzled variant and retain the ISA/counter evidence.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- Separate `s_waitcnt vmcnt` for global loads from `lgkmcnt` for LDS; do not drain all counters every K step.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

## Fused epilogues

- `w13`: same owner computes adjacent gate/up values, applies fp32 `silu(gate)*up`, then converts once to fp16 intermediate.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- `w2`: multiply route weight in fp32 before final fp16 conversion. Combine routed rows only when vLLM TP/EP semantics permit it; no grid-wide dependency inside either GEMM.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
- Bias is optional and fused before activation.

## Correctness and performance gates

1. Guard-page/OOB tests for all M/N/K tails and expert sentinel `-1`.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
2. Empty/skewed experts and `top_k={1,2,8}`.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
3. Compare fp32-accum output and fused SiLU/route-scale against PyTorch.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
4. TP=4 and CUDA/HIP graph safety: no allocation, packing or host sync in forward.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
5. ISA dump proves `v_dot2_f32_f16`; no MFMA/WMMA and no scratch.[^isa]
6. Benchmark rows/expert `{1,2,4,8,16,32,64,128}`; sweep crossover by rows **per expert**, not total tokens.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
7. Report kernel-only and routed end-to-end, DRAM bytes, L2/Infinity-cache hit rate, VGPR/LDS/occupancy and launch count.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

## Landing order

1. Reference op + packed-layout checker.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
2. Decode M1, then M2/M4, then M8 after register gate.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
3. Fused w13 and w2 epilogues.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
4. 64x64x32 grouped prefill seed, then sweep.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]
5. TP=4 correctness and graph-safe dispatch.[^isa][^f-49fb2ee-moe-q-gemm-rdna3-cu]

This card remains after the current `fa_rdna2` occupancy fix.[^f-83de31cf-fa-rdna2-cu]

## Sources

[^isa]: AMD, *"RDNA 2" Instruction Set Architecture Reference Guide* (doc 70648): "Feature Changes in RDNA2 Devices" dot-product list; VOP2 opcodes 2 `V_DOT2C_F32_F16` / 13 `V_DOT4C_I32_I8`; VOP3P opcodes 19–25 `V_DOT2_F32_F16` … `V_DOT8_U32_U4`; LDS "128kB memory per workgroup processor split up into 64 banks of dword-wide RAMs"; "a single workgroup may allocate up to 64kB of LDS". <https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture>
[^f-49fb2ee-moe-q-gemm-rdna3-cu]: vllm-project/vllm `csrc/rocm/moe_q_gemm_rdna3.cu` @ [`49fb2ee`](https://github.com/vllm-project/vllm/blob/49fb2ee3481246ce88224220442c6ef30f88b209/csrc/rocm/moe_q_gemm_rdna3.cu).
[^f-83de31cf-fa-rdna2-cu]: BlivionIaG/vllm `csrc/rocm/fa_rdna2.cu` @ [`83de31cf`](https://github.com/BlivionIaG/vllm/blob/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb/csrc/rocm/fa_rdna2.cu).
[^t8d-tip]: opengfx1030/vllm-rdna `rdna_extras` head [`8d141bdc`](https://github.com/opengfx1030/vllm-rdna/commit/8d141bdc80b3124315a4a1e52ab4485cb6974f5e) (merge of #229, committed 2026-11-27 08:45 Europe/Paris; read 2026-11-27 ~09:50 Paris).
[^t8d-c221]: commit [`9a706f6a`](https://github.com/opengfx1030/vllm-rdna/commit/9a706f6a48120f00a3a80df731834fb900c0ca64) (PR #221), in dest @ `8d141bdc`.
[^t8d-pr221]: PR #221 body, <https://github.com/opengfx1030/vllm-rdna/pull/221> (read 2026-11-27): `moe_gemm_unquantized_prefill_kernel` "61 VGPR, 30 SGPR, 11648 B LDS, 0 spills" from `llvm-readobj --notes` on a `cda6ef6e5e` build with ROCm 10.0; "relL2 = 0.000618". Author-reported.
[^t8d-m-kernel]: `moe_gemm_unquantized_prefill_kernel` and host wrapper: https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/csrc/rocm/moe_unquantized_rdna2.cu#LL229-L405
[^t8d-m-py]: dispatch: https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/vllm/model_executor/layers/fused_moe/experts/rdna2_unquantized_moe.py#LL179-L231
[^t8d-cmake]: `CMakeLists.txt` @ `8d141bdc`: gfx1030 list https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/CMakeLists.txt#LL1780-L1815; `-O3` list https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/CMakeLists.txt#LL1883-L1894
[^t8d-cmake-archs]: `HIP_SUPPORTED_ARCHS`: https://github.com/opengfx1030/vllm-rdna/blob/8d141bdc80b3124315a4a1e52ab4485cb6974f5e/CMakeLists.txt#LL55
