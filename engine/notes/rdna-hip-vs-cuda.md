# RDNA ROCm/HIP vs CUDA — gap report

Date: 2026-08-28. Scope: **4×/8× V620 gfx1030**, extras HIP, later hippih. Not Instinct. Not a CUDA port. Do not invent tok/s. Occupancy still first. Pin stays closed. No new ticket.

Contract index: [../coverage.md](../coverage.md). Silicon: [../../silicon/valu.md](../../silicon/valu.md), [../../silicon/rccl-p2p.md](../../silicon/rccl-p2p.md). Tickets: see [project 4](https://github.com/users/BlivionIaG/projects/4).

Three different “missing.” Do not mix them.

1. **ISA** — cannot close. Leave.
2. **TheRock / ROCm product** — AMD does not ship the GEMM/FA/profiler libs on gfx1030. Leave / work around.
3. **Engine / runtime** — CUDA serving assumes a vendor kernel library + NVLink. We write Wave32 DOT HIP or skip.

## 1. ISA (cannot add)

| CUDA (Ampere+ / Hopper / Blackwell) | gfx1030 V620 |
|---|---|
| Tensor cores, WGMMA, UMMA, TMA | Packed DOT only: `fdot2` 256, `sdot4` 512, `V_DOT8_I32_I4` 1024 FLOPS/clk/CU |
| FP8 / MXFP8 / NVFP4 / MXFP4 MMA | No unit. Unpack → `fdot2`. `supports_fp8()` / `supports_mx()` false |
| BF16 MMA | `v_dot2_f32_bf16` is RDNA3+. gfx1030 dots stay fp16. BF16 is emulated |
| AGPR / MFMA acc | No |
| NVLink / NVSwitch / XGMI | PCIe 4.0 x16. 88096 **PIX** on one board, **PHB** cross-board |
| TMA async copy, warp-specialized pipelines | No. Manual LDS + vector loads |

Same ISA in Rust or HIP. `llvm.amdgcn.fdot2` = `__builtin_amdgcn_fdot2`. Do not take f16-FMA ISel.

gfx1100 (secondary): WMMA exists. Do not pivot extras onto it. First box is V620.

## 1b. Kernel-writer silicon (not a second write list)

Same ISA peak as §1. What CUDA kernels get for free that we do not:

- No TMA / async-copy. Manual LDS + vector loads. LDS is 64 KiB/WG, **64 banks × 4 B** on RDNA2+ (32-bank is CDNA1–3, not gfx1030).
- No native `v_global_atomic_pk_add_f16` on gfx1030. Split-K / MoE epilogue is 64-bit CAS (`atomic_add_pk4_f16`).
- Infinity Cache has **no** persist/bypass (CUDA L2 persist is Leave).
- hipcc will **not** peephole `__hfma2` into `fdot2`. Issue `__builtin_amdgcn_fdot2`.
- `sdot4` is unused on extras. After occupancy: Sage / W8A8. INT8 KV is cvt + `fdot2`, not `sdot4` on KV.
- EXL3 GEMMs (`a2c8d5cf`) still have no `__launch_bounds__` / `waves_per_eu`. Lever is VGPR (`w0`+`w1` 4×16), not LDS.
- `rocprofiler-compute` roofline is gfx11-only (TheRock exclude). Occupancy dumps stay `llvm-objdump` / `amdhsa`.


## 2. ROCm 7.14 / TheRock gfx1030 excludes

Live target **7.14**. Stay. Do not bump to Instinct-centric 10.0 first.

**Excluded (CK = TheRock #4836, not #1245):** hipBLASLt, hipSPARSELt, composable_kernel, rocWMMA, hipTensor, rocprofiler-compute.

**Remain:** compiler, HIP runtime, ISA, rocBLAS / hipBLAS (not Lt), RCCL.

CUDA’s stand-in for that exclude list is CUTLASS + cublasLt + FA3 + Nsight roofline. We do not get a library fast GEMM or FA for DOT. FlyDSL can emit gfx1030 objects; shipped tiles are MFMA/WMMA — Gate 0–1 only.

## 3. Engine libs (CUDA-first → Leave)

| CUDA default | Our stand-in | Verdict |
|---|---|---|
| FlashInfer / FA3 / Sage2/3 | `fa_rdna2` + stock Triton FA baseline | Leave vendor. Write HIP |
| CUTLASS / CuTe / DeepGEMM | `q_gemm_rdna2` / `moe_q_gemm` / EXL3 HIP | Leave |
| Marlin / Machete / vendor NVFP4 | unpack → `fdot2` if we ever take the format | **Dead** as a port |
| TensorRT-LLM | extras / later hippih | Leave |
| bitsandbytes | — | **Dead** (AMD column ❌) |
| DeepEP / IBGDA / MORI | mapped-peer PCIe after measured P2P | Later. [../deepep.md](../deepep.md) |
| AITER / CK FA / shuffle MLA | — | **Dead** (CDNA) |
| Custom AR / QuickReduce | PYNCCL `_all_reduce_out_place` | gfx94/95 only. Already bypassed |
| CUDA Graphs + PDL | HIP FULL-graph glue | Live glue. Recaptures the `(1,1)` occupancy trap |
| NCCL over NVLink | RCCL Ring/Tree over PCIe | Transport only. GEMM does not own RCCL |

Triton exists on both. On this box it is a **compile tax**, not the home. Replace hot paths with HIP. Do not HIP-rewrite unused CUDA Triton (AITER FA, FA3, Marlin).

## 4. Stock vLLM / SGLang holes vs NVIDIA

Upstream does **not** list gfx1030. Forced build:

- `on_rdna()` = gfx11/12. V620 is `on_gfx10x()`. New `on_rdna()` gates skip us.
- Stock HIP paged-decode: `on_gfx1x`.
- Skinny `wvSplitK` / `LLMM1`: `on_gfx9() or on_gfx1x()`. `VLLM_ROCM_USE_SKINNY_GEMM` is a no-op here.
- `ROCM_ATTN` on gfx1030 is Triton prefill + Triton `kernel_paged_attention_2d`, not HIP decode.
- Custom collectives off → PYNCCL (locked, correct).
- Official FA / MoE whitelist is CDNA or gfx11/12.

SGLang overlay is **after occupancy**. hippih is after extras + SGLang + Llaminar. Do not start a CUDA-clone engine.

## 5. Already ours (do not re-list as missing)

Live / on-branch as of extras tip **`a2c8d5cf`** (28 Aug): W4A16 / W8A16 / W8A8-FP8-`fdot2` / mxfp4 sources, `fa_rdna2`, sparse MLA HIP, indexer top-k, **GDN HIP** + mixed-path decode + per-layer hybrid KV, **EXL3 HIP tree** (`exl3_dot2_*`, produce `3inst` → `fdot2`), PYNCCL TP, FULL-graph glue.

P2P is **attested** on the 4× V620 box. Still no measured `hipMemcpyPeer` GB/s.

Not fixed: occupancy `(1,1)` on `fa_rdna2` + `skinny_gemms.cu`, MLA prefill `load_row` OOB, some GEMM GPU-verify pending, W8A16-FP8 MoE CMake gap.

## 6. Add / improve for our needs

Same order as [../coverage.md](../coverage.md) v2 write list. Do **not** open a parallel CUDA-parity epic.

**Now (pin closed — current subjects only)**

1. Occupancy flip: drop min-blocks / `__launch_bounds__(N, 1)` / `amdgpu_waves_per_eu(1,1)`. Decode-class `waves_per_eu(4, 8)`. This is the real CUDA-graph / launch parity.
2. GPU-verify live W4 / mxfp4.
3. Finish EXL3 `3inst` → `fdot2` (EXL3 RDNA2 side project). Compile `mcg`, never produce `mul1`.

**Next writes (after occupancy)**

4. Sage QK `sdot4` prefill only.
5. Head-64 `fa_rdna2` (Triton decode hole).
6. W8A8 `sdot4` (real INT8 compute — W8A16 is not this).
7. INT8 KV `int8_per_token_head` fused into `fa_rdna2` (cvt + `fdot2` QK, no `sdot4` on KV).
8. MLA prefill OOB, then `fdot2` on prefill (both sides already half).
9. Native HIP FP16 / INT8 MoE. Do **not** rewrite `moe_q_gemm_rdna2`.
10. QSA indexer HIP if Flash-Next is a dest. Retip GDN if D/ratio change. No second tree.

**Later (not first)**

- Measure `B_P` / `hipMemcpyPeer` / RCCL `via P2P/IPC` on `v620_toolbox/pcie_p2p`. Then FreeToken **q\*** miss-split (policy only).
- DFlash / MTP / DSpark after MLA fat tile `q>1`.
- NVFP4 unpack → `fdot2`. INT2 unpack. W4A8 / integer W4A4 `sdot8` after W8A8.
- hippih. Not a FlashInfer or FreeToken fork.

**Do not add**

- Port FlashInfer, CUTLASS, AITER, TRT-LLM, Marlin, DeepEP-as-is.
- hipBLASLt / CK / rocWMMA on gfx1030.
- FA3 / TMA / Instinct FP8 MMA / W4A4-native FP4.
- Rust kernel rewrite (occupancy/LDS missing in rustc, not the DOT).
- ROCm 10.0 bump. `HSA_OVERRIDE_GFX_VERSION`. MI50 VBIOS flash.
- PD disagg or streaming decode-KV over PCIe as a throughput win.

## 7. One line

CUDA’s stack is tensor cores + NVLink + a vendor kernel library. Ours is Wave32 DOT + PCIe + kernels we write. Missing **libraries** are Leave. Missing **work** is occupancy, then `sdot4` / INT8 KV / head-64, then a measured PIX number.
