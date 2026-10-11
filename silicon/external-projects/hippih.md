---
title: "hippih — silicon contract (empty repo, three ISAs)"
sources:
  - https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX900.html
  - https://gpuopen.com/learn/wmma_on_rdna3/
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://github.com/BlivionIaG/hippih
  - https://github.com/warpfront/hipfire
---

# hippih — silicon contract (empty repo, three ISAs)

Date: 2026-08-19. Repo: [BlivionIaG/hippih](https://github.com/BlivionIaG/hippih) @ `d1e34a7` is **LICENSE + one-line README only**. Engine page: [engine/hippih.md](../../engine/hippih.md) if present. **Do not pivot off `rdna2_extras`.** Occupancy still first.[^github-blivioniag-hippih]

Stated target: custom HIP inference engine for **gfx1030 + gfx1100 + gfx900**. That is three compile targets and three inner opcodes, not one “RDNA” kernel.

## Corrected plan

| SKU | LLVM | Wave | Fast FP16 inner op | Integer | Absent |
|---|---|---|---|---|---|
| V620 | **gfx1030** | wave32 WGP | `__builtin_amdgcn_fdot2` → `V_DOT2C_F32_F16` | `sdot4` / `sdot8` | WMMA, bf16 DOT, VOPD, `sudot4` [^isa70648] |
| W7800 | **gfx1100** | wave32 | `fdot2` decode; **WMMA 16×16×16** fat prefill | `sdot4` (IU8 peak same as RDNA2) | MFMA [^m-gpuopen-com-learn-wmma-on-rdna3] |
| V340L | **gfx900** Vega10 dual-die | wave64 CU | `v_mad_mix_f32` + `v_pk_fma_f16` | packed ALU, **no DOT** | `fdot2`, `sdot4`, `sdot8`, `fma_mix` [^m-vm-llvm-html-amdgpu-amdgpuasmgfx900-html] |

Fatbins are not interchangeable. `on_rdna()` (v0.27.1 = gfx11/12) must not gate hippih gfx1030/gfx900 paths. See [v340l.md](../other-hardware/v340l.md), [fp16-rdna2.md](../arch-and-memory/fp16-rdna2.md), [valu.md](../arch-and-memory/valu.md).

## What to take from extras (later)

Only after occupancy + `load_row` on the vLLM fork:

- gfx1030: skinny `fdot2` GEMV, 64×64×32 FP16 tile, W4A16/W8A16 dequant→`fdot2`, W8A8 `sdot4`, MLA after OOB fix.
- gfx1100: same decode DOT2; WMMA allowed for fat GEMM/attention.
- gfx900: new TU, not a retarget of extras objects.

Placement ideas (attn on W7800, experts on V620, activations-only) stay Later: [hetero-moe-w7800-v620.md](../interconnect/hetero-moe-w7800-v620.md). hippih may host that graph; it does not replace extras compute work.

## hipfire is not a hippih replacement

2026-08-21: do **not** fork [warpfront/hipfire](https://github.com/warpfront/hipfire) into this repo, and do **not** drop the SGLang overlay for it. hipfire's tuned path is gfx11/12 WMMA; gfx1030 is portable; Vega in that tree is **gfx906**, not gfx900. Strategy lock: [hipfire.md](hipfire.md).[^github-warpfront-hipfire]

hippih *may* later grow **tools / extra modes** (arch microbench, fail-closed graph replay). That is still this three-ISA contract, not a hipfire clone.

## First hippih kernel (when the repo exists)

1. Arch enum + `hipcc --offload-arch=` per TU.
2. gfx1030 microbench: `fdot2` vs `__hfma2` vs scalar FMA; ISA must show `v_dot2c`/`v_dot2`.
3. gfx900 microbench: `mad_mix` / `pk_fma_f16`; ISA must **not** show `v_dot*`.
4. gfx1100 microbench: WMMA tile vs `fdot2` skinny.
5. No shared occupancy attribute across the three.

Until those land, hippih is a name and a three-ISA contract, not an engine.

## Sources

Primary sources for this page, cited inline as footnotes:

[^github-blivioniag-hippih]: github.com/BlivionIaG/hippih. https://github.com/BlivionIaG/hippih (private repo (exists, verified via gh API as BlivionIaG on 2026-10-11))
[^github-warpfront-hipfire]: GitHub - warpfront/hipfire: RDNA-native LLM inference engine in Rust.. https://github.com/warpfront/hipfire (checked 2026-10-11)


[^isa70648]: AMD "RDNA 2" Instruction Set Architecture Reference Guide, doc 70648, 2020-11-30. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture (checked 2026-10-11)

[^m-gpuopen-com-learn-wmma-on-rdna3]: How to accelerate AI applications on RDNA 3 using WMMA - AMD GPUOpen. https://gpuopen.com/learn/wmma_on_rdna3/ (checked 2026-10-11)

[^m-vm-llvm-html-amdgpu-amdgpuasmgfx900-html]: Syntax of gfx900, gfx902, gfx909 and gfx90c Instructions — LLVM 22.0.0git documentation. https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX900.html (checked 2026-10-11)
