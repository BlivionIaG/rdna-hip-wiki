# Multi-tier MoE — placement vs engine base

Date: 2026-08-19. Engine contract. **Product path (locked):** optimize **vLLM fork first**, then **Llaminar**, then **hippih** as the in-house HIP engine. SGLang is not the next engine. Occupancy + V620 HIP MoE still first. Do not invent tok/s.

Silicon: [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md). V340L: [silicon/v340l.md](../silicon/v340l.md). Related: [moe.md](moe.md), [deepep.md](deepep.md), [alt-engines.md](alt-engines.md), [hippih.md](hippih.md).

## Topology (human plan)

- **Fast tier:** 2× W7800 48 GB (**gfx1100**) — attention, router, embeddings, LM head, **live KV**
- **Capacity tier:** 8× V620 32 GB (**gfx1030**) — expert GEMMs only
- **Bus rule:** ship **activations only** (fp16 on the hop unless both kernels consume a smaller dtype).
- **Not this work:** V340L is **Vega10 / gfx900**, dual-die on one PCIe 3.0 x16. Later. [silicon/v340l.md](../silicon/v340l.md).

This is **two HIP targets** for the W7800/V620 box (plus a third ISA if V340L is ever benched). gfx1100 may use WMMA/BF16 locally; V620 stays `fdot2`/`sdot4`. Do not park live KV on V620. Decode hop is tiny; **prefill is the bus**. W7800↔V620 P2P is **unmeasured** — bench later from [v620_toolbox](https://github.com/BlivionIaG/v620_toolbox) `pcie_p2p`.

## Engine base (locked 2026-08-19, corrected)

| Order | Engine | Why |
|---|---|---|
| **1. Now** | **vLLM fork** (`rdna2_extras`) | Stack that already compiles gfx1030 HIP. Occupancy, FA, MoE DOT, then placement glue. |
| **2. Next** | **Llaminar** | Heterogeneous domains + TP/PP + prefix-cache. Current ROCm is **gfx906 only** — we add gfx1030/gfx1100 HIP. Continuous batching still a plan (add it there). |
| **3. In-house** | **hippih** | Custom HIP engine for gfx1030 / gfx1100 / gfx900. Steal extras kernels + Llaminar placement. Repo is a stub — contract: [hippih.md](hippih.md). |
| **Not next** | SGLang | Serving-strong, Instinct/AITER. Demoted. Steal radix/CB ideas only. |

Do **not** block occupancy or HIP MoE on Llaminar or hippih.

## gfx900 HIP (V340L) — engine fire-list

Vega10 **does not** have the Vega20 DL DOT set. LLVM `fdot2.ll`: gfx900 emits `v_mad_mix_f32` / `v_fma_f16`; gfx906 with DL emits `v_dot2_f32_f16`. gfx900-specific VOP3P page is mix only (`v_mad_mix_f32`, `v_mad_mixhi/lo_f16`). Packed `v_pk_*_f16` lives in core GFX9, not as a DOT unit.

| Inner op | gfx900 (V340L) | gfx906 (Llaminar today) | gfx1030 (V620) |
|---|---|---|---|
| `v_mad_mix_f32` / mixhi/lo | **yes** | yes (often `v_fma_mix_*`) | not the path |
| `v_pk_fma/mul/add_f16` | **yes** (core GFX9) | yes | not the path |
| `v_dot2_f32_f16` / `__builtin_amdgcn_fdot2` | **no** | **yes** (DL) | **yes** (`fdot2`) |
| `v_dot4_i32_i8` / `sdot4` | **no** | **yes** (DL) | **yes** |
| `v_dot8_i32_i4` / `sdot8` | **no** | **yes** (DL) | **yes** |
| MFMA / WMMA | **no** | no | no |
| wave | 64 | 64 | 32 |

**HIP:** do not call `__builtin_amdgcn_fdot2` / `sdot4` on gfx900. Old HIP-Clang notes that say “fdot2 on gfx9+” mean Vega20+. On V340L the honest inner loop is packed FMA / `mad_mix` into fp32 accum. Llaminar gfx906 DOT objects **will not load**. A V340L backend is a third ISA, after occupancy + V620 HIP MoE. hippih owns that third backend when we write it.

@RDNA2_Researcher owns the silicon table; this is the engine dispatch constraint.

## Llaminar (next engine, after vLLM kernels)

[Llaminar/llaminar](https://github.com/Llaminar/llaminar) — C++ kernel-centric, alpha, GGUF.

| Piece | Status |
|---|---|
| Heterogeneous domains | Native — **why it is next** |
| TP / PP / MoE EP | EP WiP |
| Prefix cache | **Exists** |
| Continuous batching | **Must add** (plan / V1 HTTP non-goal) |
| ROCm | **gfx906 only** today |
| gfx1030 / gfx1100 HIP | **we write** |
| gfx900 / V340L | Later; packed mix/FMA, not their gfx906 DOT |

## hippih (in-house, after Llaminar lessons)

[hippih](https://github.com/BlivionIaG/hippih) — empty tree today. Same three-ISA split as this page. Do not start it before extras occupancy. Contract: [hippih.md](hippih.md).

## Implementation order

1. `fa_rdna2` occupancy + V620 HIP MoE (already first).
2. gfx1100 attention/router baseline on W7800.
3. Measured W7800↔V620 activation matrix (`v620_toolbox/pcie_p2p`).
4. Asymmetric activation dispatch/combine.
5. Optional KV overflow park — never live-KV on V620.
6. **Llaminar** as the hetero serving/runtime shell (gfx1030 + gfx1100 backends + CB).
7. **hippih** as the in-house engine (steal extras + Llaminar).
8. V340L gfx900 packed-mix backend — Later, not a V620 or gfx906 drop-in.

## Cards (for VLLM_FORK_Manager)

Occupancy still first.

- Multi-tier W7800/V620 placement — Later
- Llaminar after vLLM (gfx1030/gfx1100 HIP + CB) — Later, **this is #2**
- hippih in-house engine — Later, **this is #3** — [hippih.md](hippih.md)
- V340L gfx900 packed-mix / mad_mix — Later; no DOT objects
- SGLang — demoted, no first card

## Sources

- Room 2026-08-18: “vllm fork then laminar”; V340L HIP check
- Room 2026-08-19: hippih is the in-house engine, not a discard
- LLVM gfx900 VOP3P: https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX900.html
- LLVM gfx906 VOP3P (DOT): https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX906.html
- LLVM `fdot2.ll` (gfx900 → mix/FMA, gfx906 → `v_dot2_f32_f16`)
- Phoronix: Vega20 DL = fdot2/sdot4/sdot8, not Vega10
- [silicon/v340l.md](../silicon/v340l.md), [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md)
