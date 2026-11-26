---
title: "Packed DOT / VALU dependency chain vs effective occupancy (gfx1030)"
sources:
  - https://gpuopen.com/download/RDNA_Architecture_public.pdf
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://gpuopen.com/learn/occupancy-explained/
  - https://gpuopen.com/learn/wmma_on_rdna3/
  - https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
  - https://llvm.org/docs/CommandGuide/llvm-mca.html
  - https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/SISchedule.td
  - https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/GCNProcessors.td
  - https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/SIInstrFormats.td
  - https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/VOP3PInstructions.td
  - https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/VOP2Instructions.td
  - https://github.com/llvm/llvm-project/blob/cd708029e0b2869e80abe31ddb175f7c35361f90/llvm/lib/Target/AMDGPU/SISchedule.td
  - https://github.com/ROCm/rocm-systems/blob/7c86e687bd651372deb4a5155933c5066b086a3c/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/derived_counters.xml
  - https://github.com/ROCm/rocm-systems/blob/7c86e687bd651372deb4a5155933c5066b086a3c/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/basic_counters.xml
---

# Packed DOT / VALU dependency chain vs effective occupancy (gfx1030)

Lock: **A dependent `fdot2` / `sdot4` accumulator chain is not a PIX / `llvm-calc-occupancy` MaxWaves limiter. It decides how many *ready* waves a SIMD32 needs to stay DOT-saturated.** RDNA issues **1 VALU instruction per cycle per SIMD32** and exposes **5 cycles** of dependent-result latency, which the hardware checks and which "can be filled by other waves." LLVM's gfx1030 model (`GFX10SpeedModel`) gives the DOT ops the ordinary `Write32Bit` class, latency 5 on the single `HWVALU` resource. So one wave whose inner K loop chains every DOT through one accumulator issues at most **1 DOT every 5 cycles** (llvm-mca: IPC 0.20). It needs about **5 ready waves**, or about **5 independent accumulators**, to issue every cycle. Once that product is reached, more waves add **no DOT throughput**: the SIMD is ALU-bound, and GPUOpen says that kind of workload "will not benefit from increased occupancy."[^rdna-arch][^llvm-sisched][^gpuopen-occupancy][^llvm-calc-occupancy]

Does **not** change extras HIP or UNC tickets. No tok/s. Do not restate the FA pin.

**Naming:**

| Term | Meaning on gfx1030 |
|---|---|
| **Packed DOT** | `V_DOT2C_F32_F16` / `V_DOT2_F32_F16` (`__builtin_amdgcn_fdot2`), `V_DOT4C_I32_I8` / `V_DOT4_I32_I8` (`sdot4`), `V_DOT8_I32_I4` (`sdot8`), plus the u-variants. ISA 70648 "Feature Changes in RDNA2 Devices": "Dot product ALU operations added accelerate inferencing and deep-learning" (sic). [^isa70648] |
| **`C` form** | VOP2 `D = dot(S0,S1) + D` ("accumulate with destination"). LLVM ties `$vdst = $src2`. It is 4 bytes and needs no source modifiers or clamp. The VOP3P `V_DOT2_F32_F16` / `V_DOT4_I32_I8` / `V_DOT8_I32_I4` forms are 8 bytes. [^isa70648][^llvm-vop2] |
| **Exposed latency** | Deck: "5 cycles of latency are exposed (automatic dependency check in hardware). Dependency stalls can be filled by other waves." A dependent VALU op waits for the result. No software NOPs are involved ([hazard-occupancy.md](hazard-occupancy.md)). [^rdna-arch] |
| **Chain / accumulator** | The sequence of DOTs that read and write the same accumulator VGPR. Its **reuse distance** is the number of VALU issues between two writes to the same accumulator. |
| **Ready wave** | A resident wave whose next instruction can issue: not parked on `s_waitcnt` ([waitcnt-occupancy.md](waitcnt-occupancy.md)), not stalled on I$ ([icache-occupancy.md](icache-occupancy.md)), and so on. **Resident ≠ ready.** |

Peak rates and the gfx1030 opcode table live in [valu.md](../arch-and-memory/valu.md). The 5-cycle craft one-liner lives in [hip-craft.md](../hip-craft/hip-craft.md) §7.5 and [lds-tiles.md](../lds-and-tiles/lds-tiles.md). This page adds only the **occupancy framing**, the LLVM model, and a box probe.

Date: **2026-10-13** Europe/Paris.

Companions: [occupancy-composite.md](occupancy-composite.md), [valu.md](../arch-and-memory/valu.md), [architecture.md](../arch-and-memory/architecture.md) §2.2 (VALU dest latency row), [hip-craft.md](../hip-craft/hip-craft.md) §4.2 / §7.5, [lds-tiles.md](../lds-and-tiles/lds-tiles.md), [trans-occupancy.md](trans-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [vgpr-occupancy.md](vgpr-occupancy.md), [icache-occupancy.md](icache-occupancy.md), [vgpr-bank-occupancy.md](vgpr-bank-occupancy.md), [hazard-occupancy.md](hazard-occupancy.md), [spi-ace-occupancy.md](spi-ace-occupancy.md).

## Take / Leave

| | |
|---|---|
| **Take** | Size DOT inner loops by **ready waves × independent accumulators ≥ 5** per SIMD32. This is the effective-occupancy rule. PIX / `llvm-calc-occupancy` only reserve slots (VGPR∩LDS∩WG/barrier). A theory of 12 waves/EU with one accumulator per wave still stalls when most waves sit on `vmcnt`. [^rdna-arch][^llvm-calc-occupancy] |
| **Take** | Split the accumulator **in source** (`c0..c4`, sum at the end). clang does **not** do it for you. Probe §3: a 64-step `fdot2` loop and a 64-step `sdot4` loop each compile to **64 back-to-back `v_dot*c` into one VGPR**. All 63 reuses have distance 1, even for the integer `sdot4` chain, which could be reassociated legally. |
| **Take** | Keep the **`C` form** in the hot loop: `fdot2(a, b, acc, false)` / `sdot4(…, false)` lower to 4-byte VOP2 `v_dot2c_f32_f16` / `v_dot4c_i32_i8`. `clamp=true` (or source modifiers) forces the 8-byte VOP3P `v_dot2_f32_f16 … clamp`. The rate class is the same, but the I$ footprint doubles ([icache-occupancy.md](icache-occupancy.md)). Probe §3. [^llvm-vop2] |
| **Take** | When a DOT kernel already saturates VALU issue, **stop buying waves**. Spend spare VGPRs on accumulators or K-unroll instead, and keep enough waves to cover VMEM latency, which DOT ILP does not hide. The deck: "Additional threads are needed … to hide memory latency." [^rdna-arch][^gpuopen-occupancy] |
| **Take** | Profile gfx1030 with counters that **exist for gfx10**. rocprofiler-sdk defines `SQ_INSTS_VALU` (SQ ev 64), `SQ_INSTS_WAVE32_VALU` (ev 75), `SQ_WAVE_CYCLES` (ev 26), `SQ_WAIT_INST_ANY` (ev 28) and `SQ_WAIT_ANY` (ev 37), plus the derived `WAVE_ISSUE_WAIT`, `WAVE_DEP_WAIT`, `MeanOccupancyPerActiveCU` and `VALUInsts`. The `VALUBusy` / `SALUBusy` derived metrics are defined only in the gfx8 / gfx9 / gfx90a blocks, **not** gfx10 / gfx1030. [^rocprof-derived][^rocprof-basic] |
| **Leave** | Do **not** put DOT rate, accumulator count or "DOT occupancy %" into `llvm-calc-occupancy` or PIX `WaveOccupancyLimiters`. Accumulators cost VGPRs, and only that VGPR count enters the min ([vgpr-occupancy.md](vgpr-occupancy.md)). [^llvm-calc-occupancy] |
| **Leave** | Do **not** read the llvm-mca numbers as silicon measurements. They come from LLVM's `GFX10SpeedModel`: latency 5, one `HWVALU`, one wave, no memory. They show the shape, not V620 timing. Silicon claims stay at the deck's "1/cycle, 5 exposed." [^llvm-mca][^llvm-sisched] |
| **Leave** | Do **not** invent per-DOT cycle tables, a DOT pipeline depth beyond 5, or separate rates for `sdot8` vs `fdot2`. The ISA lists semantics only. GPUOpen gives **peak ops/clk/CU** (FP16 256 / IU8 512 / IU4 1024 on RX 6950 XT), which fits 1 DOT/clk/SIMD with 2 SIMD/CU. LLVM models all three as `Write32Bit`. [^isa70648][^gpuopen-wmma][^llvm-vop3p] |
| **Leave** | Do **not** import VOPD dual-issue (`v_dual_dot2acc_f32_f16`), WMMA, MFMA, `fdot2_f32_bf16` or `sudot4` as ways to beat the chain. All are absent on gfx1030 ([valu.md](../arch-and-memory/valu.md)). TRANS co-issue is the only parallel VALU sibling ([trans-occupancy.md](trans-occupancy.md)). |
| **Leave** | Do **not** cite `VALUBusy` / `SALUBusy` as gfx1030 evidence until `rocprofv3 --list-avail` on a V620 shows them. The pinned rocprofiler-sdk XML does not define them for gfx10. *(Not checked on silicon this pass.)* [^rocprof-derived] |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. |

## 1. What the deck / ISA say

### 1.1 RDNA Architecture deck (issue + latency + ILP)

GPUOpen RDNA Architecture presentation, "Single-cycle instruction issue on RDNA" and the following slides:[^rdna-arch]

- "Each SIMD32 issues 1 instruction every cycle." "Vector instruction throughput is 1 every cycle (for Wave32)."
- "5 cycles of latency are exposed (automatic dependency check in hardware). Dependency stalls can be filled by other waves."
- "Single-cycle instruction issue: ILP and scheduling matters." The slide's two `v_fma_f32` / `v_fmac_f32` timelines put **dependent** pairs back to back in one and interleave three independent pairs in the other. The interleaved order finishes in visibly fewer cycles.
- "WGP requires 4*32 = 128 threads to be able to reach 100% ALU utilization. Only achieved with high instruction level parallelism (ILP)." "256 threads / WGP often reach >90% ALU utilization in practice" (footnote: observed in pixel shaders in a game trace, **not** compute DOT loops).
- "Additional threads are needed on both GCN and RDNA to hide memory latency. Unless you can fill the wait with ALU (this is extremely rare)."
- "What (not) to do for ILP": code bloat ("mind the I$ size!"), higher VGPR count. "Don't panic: extra waves are a really good source of parallelism."

Local text: `/workspace/rdna2-src/rdna-arch.txt` (pdftotext of the public PDF).

### 1.2 ISA (Document 70648)

| Claim | Where |
|---|---|
| RDNA 2 adds DOT ALU ops: `V_DOT2_F32_F16 / V_DOT2C_F32_F16`, `V_DOT2_I32_I16 / V_DOT2_U32_U16`, `V_DOT4_I32_I8 / V_DOT4C_I32_I8`, `V_DOT4_U32_U8`, `V_DOT8_I32_I4`, `V_DOT8_U32_U4` | "Feature Changes in RDNA2 Devices" |
| `V_DOT2C_F32_F16` (VOP2 op 2): `D.f32 = S0.f16[0]*S1.f16[0] + S0.f16[1]*S1.f16[1] + D.f32`, i.e. "accumulate with destination" | ch. 12 VOP2 |
| `V_DOT4C_I32_I8` (VOP2 op 13): four signed-byte products `+ D.i32` | ch. 12 VOP2 |
| `V_DOT2_F32_F16` (VOP3P op 19), `V_DOT8_I32_I4` (VOP3P op 24): same math with explicit `S2` | §12.10 VOP3P |
| Packed math = two 16-bit ops in one DWORD "as if they were separate threads"; VOP3P adds `op_sel` / `neg` and removes ABS / OMOD | §6.7 |
| "Shader hardware can resolve most data dependencies." Only long-latency memory-type results need `S_WAITCNT` | §4.4 |

The ISA gives **no** issue rate or latency per DOT. Rates come from the deck (1/cycle, 5 exposed) and GPUOpen (peak ops/clk/CU).[^isa70648][^gpuopen-wmma]

### 1.3 GPUOpen peak table

GPUOpen "How to accelerate AI applications on RDNA 3 using WMMA" lists **RX 6950 XT (RDNA 2) FLOPS/clock/CU: FP16 256, BF16 N/A, IU8 512, IU4 1024**. With 2 SIMD32 per CU and 32 lanes, that is 1 DOT2 / DOT4 / DOT8 per lane per clock per SIMD (4 / 8 / 16 ops each). That is the same 1-per-cycle issue as any other VALU op, which is why a DOT chain obeys the deck's 5-cycle rule unchanged. *(Derivation, same arithmetic as [valu.md](../arch-and-memory/valu.md).)*[^gpuopen-wmma]

## 2. What LLVM / HIP do

| Piece | Role |
|---|---|
| `GCNProcessors.td` | `def GFX1030 : AMDGPUProcessorModel<"gfx1030", GFX10SpeedModel, …>` [^llvm-gcnproc] |
| `SISchedule.td` `GFX10SpeedModel` | "The latency values are 1 / (operations / cycle). Add 1 stall cycle for VGPR read." `Write32Bit` → `[HWVALU, HWRC]`, latency **5**. `WriteTrans32` → `HWTransVALU`, 10. `WriteQuarterRate32` → 8. `HWVALU` is `ProcResource<1>`, `BufferSize = 1`. Identical at `llvmorg-19.1.7` (the probe toolchain). [^llvm-sisched][^llvm-sisched-19] |
| `SIInstrFormats.td` | `InstSI` default `let SchedRW = [Write32Bit];` [^llvm-instrformats] |
| `VOP3PInstructions.td` | `V_DOT2_F32_F16`, `V_DOT4_I32_I8`, `V_DOT8_I32_I4` sit under `let IsDOT = 1` with **no** `SchedRW` override, so they inherit `Write32Bit`. This is ordinary full-rate VALU, **not** TRANS or quarter-rate. [^llvm-vop3p] |
| `VOP2Instructions.td` | `V_DOT2C_F32_F16` / `V_DOT4C_I32_I8`: `Constraints = "$vdst = $src2"`, `IsDOT = 1`. `Dot2F32NoModsPat<AMDGPUfdot2, V_DOT2C_F32_F16_e32>` and the `int_amdgcn_sdot4 … DSTCLAMP.NONE → V_DOT4C_I32_I8_e32` pattern (`AddedComplexity = 30`) pick the 4-byte form whenever there are no mods or clamp. [^llvm-vop2] |
| `GCNSubtarget::computeOccupancy` / `llvm-calc-occupancy` | `min(WG+LDS, SGPR, VGPR)`. **No** DOT / VALU-latency term ([occupancy-composite.md](occupancy-composite.md)). [^llvm-calc-occupancy] |
| `llvm-mca` | Static throughput simulator over the target's scheduling model. Used in §3 for the **model** view only. [^llvm-mca] |

hipcc for `--offload-arch=gfx1030` owns emission. The machine scheduler can interleave **independent** chains you wrote. It does not split one chain into several.

## 3. Box probe (model + codegen, not silicon)

Toolchain: Debian clang / llvm-mca **19.1.7** (`apt llvm-19 clang-19` on the box, 2026-10-13). Target `amdgcn-amd-amdhsa -mcpu=gfx1030` (wave32 default). Probe tree: `/workspace/dot-probe/` (`k.c`, `k.s`, `sw_*.s`, `enc.s`). Plain C with clang builtins and `-nogpulib`, no HIP headers. *(Box probe; reproducible from the snippets below. Not a V620 measurement.)*

### 3.1 llvm-mca: accumulator reuse distance → IPC

120 back-to-back `v_dot*c` per block, accumulator `v{i mod N}` round-robin, 50 iterations:

| Independent accumulators N | `v_dot2c_f32_f16` IPC | `v_dot4c_i32_i8` IPC |
|---|---|---|
| 1 (single chain) | **0.20** | **0.20** |
| 2 | 0.40 | 0.40 |
| 3 | 0.60 | 0.60 |
| 4 | 0.80 | 0.80 |
| 5 | **1.00** | **1.00** |
| 6 / 8 / 10 | 1.00 | 1.00 |

In the model, IPC = min(1, N / 5): one `HWVALU` at 1/cycle with a 5-cycle `Write32Bit` latency. The deck's "filled by other waves" means silicon can also reach saturation with **W ready waves × N accumulators ≥ 5**. llvm-mca simulates one wave, so it cannot show that half. [^llvm-sisched][^rdna-arch]

```bash
# reproduce one row
for i in $(seq 0 119); do echo "v_dot2c_f32_f16 v$((i%N)), v$((40+2*(i%8))), v$((41+2*(i%8)))"; done > sw.s
llvm-mca -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -mattr=+wavefrontsize32 -iterations=50 sw.s | grep IPC
```

### 3.2 clang codegen: single chains are not split

```c
typedef _Float16 half2 __attribute__((ext_vector_type(2)));
__attribute__((amdgpu_kernel)) void dot_chain(const half2 *a, const half2 *b, float *out) {
  int t = __builtin_amdgcn_workitem_id_x(); float acc = 0.f;
  #pragma unroll
  for (int k = 0; k < 64; ++k) acc = __builtin_amdgcn_fdot2(a[k*256+t], b[k*256+t], acc, 0);
  out[t] = acc;
}
// sdot4_chain: same with int and __builtin_amdgcn_sdot4(…, acc, 0)
// dot_split5:  five accumulators c0..c4, stride-5 K loop, sum at end
// fdot2_clamp: __builtin_amdgcn_fdot2(a, b, 0.f, /*clamp=*/1)
```

| Kernel | Emitted DOT | Accumulator reuse distance (in DOT instructions) | Backend remark |
|---|---|---|---|
| `dot_chain` | 64 × `v_dot2c_f32_f16` into **one** VGPR | 63 / 63 at distance **1** | NumVgprs 70, Occupancy 12 |
| `sdot4_chain` | 64 × `v_dot4c_i32_i8` into **one** VGPR | 63 / 63 at distance **1** (integer, still not reassociated) | NumVgprs 70, Occupancy 12 |
| `dot_split5` | 65 × `v_dot2c_f32_f16` over 5 VGPRs | 26 at ≥ 5, 34 at 1–4 (scheduler interleaves the chains with loads / `s_waitcnt`, not strict round-robin) | NumVgprs 70, Occupancy 12 |
| `fdot2_clamp` | 1 × `v_dot2_f32_f16 … clamp` (VOP3P) | n/a | NumVgprs 3, Occupancy 16 |

The 70-VGPR / 12-wave figure comes from the full unroll keeping many of the 128 `global_load_dword` in flight ahead of their DOTs (25 loads before the first DOT, waits like `vmcnt(23)`), not from the accumulators: the chain and split kernels land on the same count. That is the VGPR-vs-ILP trade the deck warns about ("Higher VGPR count"), and it **is** in the min ([vgpr-occupancy.md](vgpr-occupancy.md)).[^rdna-arch]

### 3.3 Encoding size (`llvm-mc -show-encoding`, gfx1030)

| Instruction | Bytes |
|---|---|
| `v_dot2c_f32_f16 v0, v8, v9` | 4 (VOP2) |
| `v_dot4c_i32_i8 v0, v8, v9` | 4 (VOP2) |
| `v_dot2_f32_f16 v0, v8, v9, v0` | 8 (VOP3P) |
| `v_dot4_i32_i8 v0, v8, v9, v0` | 8 (VOP3P) |
| `v_dot8_i32_i4 v0, v8, v9, v0` | 8 (VOP3P). gfx1030 has **no** `v_dot8c` (llvm-mc rejects it) |
| `v_pk_fma_f16 v0, v8, v9, v10` | 8 (VOP3P) |

## 4. Relation to occupancy: outside the PIX min

```text
waves/EU = min(VGPR, SGPR→always 16, LDS+WG+barrier)       // unchanged by DOT chain shape
DOT issue/SIMD/cycle ≈ min(1, ready_waves × indep_accs / 5) // deck 1/cycle + 5 exposed (shape, not a formula AMD publishes)
```

| Resource | In PIX MaxWaves / `llvm-calc-occupancy`? | Effect [^llvm-calc-occupancy] |
|---|---|---|
| VGPR / LDS / WG / Barriers | **Yes** | Cap reserved wave slots. Extra accumulators count here only as VGPRs |
| **DOT / VALU dependency chain** | **No** | A wave whose next DOT depends on the previous one cannot issue for 5 cycles. The SIMD idles unless another **ready** wave has VALU work |
| Waitcnt / scoreboard | **No** | Removes waves from the ready set. This is what makes "theory 12" behave like "ready 2" in a load-heavy loop ([waitcnt-occupancy.md](waitcnt-occupancy.md)) |
| TRANS / SFU | **No** | Separate ¼-rate pipe that co-issues with DOT ([trans-occupancy.md](trans-occupancy.md)) |

Mental model:

```text
1 acc/wave, 12 resident, ~2 ready (rest on vmcnt)   → ~2/5 DOT issue, theory looks fine
5 accs/wave, 4 resident, ≥1 ready                   → DOT-saturated; more waves only for VMEM latency
VALU-saturated + still memory-parked often          → memory-bound: fix tiling/prefetch, not ALU ILP
```

## 5. Craft checklist (HIP → ISA)

1. Size occupancy first ([occupancy-composite.md](occupancy-composite.md)). Do **not** add a DOT term to `llvm-calc-occupancy`.
2. In K loops write **≥ 5 independent accumulators per output lane-group** (or rely on ≥ 5 *ready* waves). Merge them at the end of the tile. For `fdot2` the merge reorders FP32 additions, so validate numerics against the reference (`sdot4` integer sums are exact).
3. `--save-temps` and grep the hot loop: `v_dot2c_f32_f16` / `v_dot4c_i32_i8` (4-byte) expected. An unexpected `v_dot2_f32_f16 … clamp` / VOP3P DOT means you passed `clamp=true` or mods. Check that the destination VGPRs rotate.
4. Profile on gfx1030 with `SQ_INSTS_VALU`, `SQ_WAVE_CYCLES`, `SQ_WAIT_INST_ANY` (`WAVE_ISSUE_WAIT`, "waiting for any instruction issue") and `SQ_WAIT_ANY` (`WAVE_DEP_WAIT`, "waiting for anything").[^rocprof-derived][^rocprof-basic] No AMD doc maps these counters to VALU dependency stalls vs memory parks on gfx1030, so A/B the same kernel with 1 vs 5 accumulators rather than reading absolute values *(unsourced interpretation)*. Memory parks: [waitcnt-occupancy.md](waitcnt-occupancy.md).
5. Do not schedule against VOPD / CDNA 5-issue. Schedule against "1 VALU/cycle/SIMD, 5 exposed, TRANS co-issues" ([hip-craft.md](../hip-craft/hip-craft.md) §7.5).

## 6. What this page adds vs existing wiki

| Page | Already said | This lock adds |
|---|---|---|
| [hip-craft.md](../hip-craft/hip-craft.md) §4.2 / §7.5 | "hide 5-cycle dest latency with ≥ 5 independent accums or other waves" | **Ready ≠ resident**. LLVM model citation (`Write32Bit`=5 for DOT). Probe: clang does **not** split chains, even `sdot4`. gfx10 counter caveat |
| [lds-tiles.md](../lds-and-tiles/lds-tiles.md) | ≥ 5 independent FMAs / DOT2s or other waves | Occupancy framing: chain shape sits outside the PIX min |
| [valu.md](../arch-and-memory/valu.md) | Opcode / peak table; prefer `C` form | Encoding bytes + the `clamp` trap. DOTs are `Write32Bit`, not quarter-rate |
| [trans-occupancy.md](trans-occupancy.md) | TRANS ¼ rate, co-issue | Sibling for the *main* VALU pipe: latency-starved, not rate-starved |
| [occupancy-composite.md](occupancy-composite.md) | Out-of-min list | DOT / VALU dependency chain row |

## Sources

1. GPUOpen RDNA Architecture PDF: https://gpuopen.com/download/RDNA_Architecture_public.pdf (local `/workspace/rdna2-src/rdna-arch.pdf`). 1 instruction/cycle/SIMD32, 5 cycles exposed, ILP slides, 128/256 threads per WGP, memory-latency threads.
2. AMD "RDNA 2" ISA Reference Guide, Document **70648**: RDNA2 DOT feature list, ch. 12 VOP2 / §12.10 VOP3P DOT semantics, §6.7 packed math, §4.4 dependency resolution. Local `/workspace/rdna2-src/rdna2-isa.txt`. Landing: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
3. GPUOpen Occupancy explained: https://gpuopen.com/learn/occupancy-explained/ ("a workload that is not latency bound will not benefit from increased occupancy … an ALU bound workload").
4. GPUOpen WMMA on RDNA 3: https://gpuopen.com/learn/wmma_on_rdna3/ (RX 6950 XT FP16 256 / IU8 512 / IU4 1024 FLOPS/clock/CU).
5. LLVM AMDGPU at `e36df8573327ba8b9c2c829e8c236599fa4c946e`: `SISchedule.td` (GFX10SpeedModel), `GCNProcessors.td` (gfx1030 → GFX10SpeedModel), `SIInstrFormats.td` (default `Write32Bit`), `VOP3PInstructions.td` (DOT defs), `VOP2Instructions.td` (`C` forms + NoMods patterns). `SISchedule.td` at `llvmorg-19.1.7` (`cd708029e0b2…`) for the probe toolchain.
6. `llvm-calc-occupancy`: https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (no DOT term). `llvm-mca`: https://llvm.org/docs/CommandGuide/llvm-mca.html
7. rocprofiler-sdk counters in ROCm/rocm-systems at `7c86e687bd651372deb4a5155933c5066b086a3c`: `derived_counters.xml` (gfx10 block: `WAVE_ISSUE_WAIT`, `WAVE_DEP_WAIT`, `MeanOccupancyPerActiveCU`; `VALUBusy` / `SALUBusy` only in gfx8 / gfx9 / gfx90a blocks) and `basic_counters.xml` (gfx10 SQ events 26 / 28 / 37 / 64 / 75).
8. Box probe 2026-10-13: Debian clang / llvm-mca / llvm-mc 19.1.7, `/workspace/dot-probe/` (§3).
9. Wiki siblings: [valu.md](../arch-and-memory/valu.md), [hip-craft.md](../hip-craft/hip-craft.md), [lds-tiles.md](../lds-and-tiles/lds-tiles.md), [trans-occupancy.md](trans-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [occupancy-composite.md](occupancy-composite.md).

Footnotes:

[^rdna-arch]: GPUOpen, "RDNA Architecture" public PDF. https://gpuopen.com/download/RDNA_Architecture_public.pdf (checked 2026-10-13)
[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-13)
[^gpuopen-occupancy]: GPUOpen, "Occupancy explained". https://gpuopen.com/learn/occupancy-explained/ (checked 2026-10-13)
[^gpuopen-wmma]: GPUOpen, "How to accelerate AI applications on RDNA 3 using WMMA" (RX 6950 XT vs RX 7900 XTX FLOPS/clock/CU table). https://gpuopen.com/learn/wmma_on_rdna3/ (checked 2026-10-13)
[^llvm-calc-occupancy]: LLVM llvm-calc-occupancy command guide. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html (checked 2026-10-13)
[^llvm-mca]: LLVM llvm-mca command guide. https://llvm.org/docs/CommandGuide/llvm-mca.html (checked 2026-10-13)
[^llvm-sisched]: LLVM `SISchedule.td` at e36df8573327, `GFX10SpeedModel` block (L366–403) and `HWVALU` / `HWTransVALU` resources (L115–120). https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/SISchedule.td (checked 2026-10-13)
[^llvm-sisched-19]: LLVM `SISchedule.td` at llvmorg-19.1.7 (cd708029e0b2), `GFX10SpeedModel` block (L302ff, same Write32Bit=5 / WriteTrans32=10). https://github.com/llvm/llvm-project/blob/cd708029e0b2869e80abe31ddb175f7c35361f90/llvm/lib/Target/AMDGPU/SISchedule.td (checked 2026-10-13)
[^llvm-gcnproc]: LLVM `GCNProcessors.td` at e36df8573327 (L192 `def GFX1030 … GFX10SpeedModel`). https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/GCNProcessors.td (checked 2026-10-13)
[^llvm-instrformats]: LLVM `SIInstrFormats.td` at e36df8573327 (L251 `let SchedRW = [Write32Bit];` in `InstSI`). https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/SIInstrFormats.td (checked 2026-10-13)
[^llvm-vop3p]: LLVM `VOP3PInstructions.td` at e36df8573327 (L797–826 `IsDOT = 1` block: `V_DOT2_F32_F16`, `V_DOT4_I32_I8`, `V_DOT8_I32_I4`; no SchedRW override). https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/VOP3PInstructions.td (checked 2026-10-13)
[^llvm-vop2]: LLVM `VOP2Instructions.td` at e36df8573327 (L1295–1335: `V_DOT2C_F32_F16` / `V_DOT4C_I32_I8` with `$vdst = $src2`, `Dot2F32NoModsPat`, sdot4 → `V_DOT4C_I32_I8_e32`). https://github.com/llvm/llvm-project/blob/e36df8573327ba8b9c2c829e8c236599fa4c946e/llvm/lib/Target/AMDGPU/VOP2Instructions.td (checked 2026-10-13)
[^rocprof-derived]: ROCm/rocm-systems `projects/rocprofiler-sdk/source/share/rocprofiler-sdk/derived_counters.xml` at 7c86e687bd65 (gfx10 block L493–527; `VALUBusy` / `SALUBusy` at L83–84 gfx8, L123–124 gfx9, L368–369 gfx90a; `<gfx1030 base="gfx10">` adds none). https://github.com/ROCm/rocm-systems/blob/7c86e687bd651372deb4a5155933c5066b086a3c/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/derived_counters.xml (checked 2026-10-13)
[^rocprof-basic]: ROCm/rocm-systems `projects/rocprofiler-sdk/source/share/rocprofiler-sdk/basic_counters.xml` at 7c86e687bd65 (gfx10 block: SQ_WAVE_CYCLES ev 26, SQ_WAIT_INST_ANY ev 28, SQ_WAIT_ANY ev 37, SQ_INSTS_VALU ev 64, SQ_INSTS_WAVE32_VALU ev 75). https://github.com/ROCm/rocm-systems/blob/7c86e687bd651372deb4a5155933c5066b086a3c/projects/rocprofiler-sdk/source/share/rocprofiler-sdk/basic_counters.xml (checked 2026-10-13)
