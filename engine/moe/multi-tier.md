# Multi-tier MoE — placement vs engine base

Date: 2026-08-20. Retipped 2026-08-27 (Spark fast tier; W7800 dead). Engine contract. **Product path (locked):** optimize **vLLM fork first**, then **SGLang overlay** (parallel serving, after occupancy), then **Llaminar**, then **hippih**. Occupancy + V620 HIP MoE still first. Do not invent tok/s.

Silicon: [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md). V340L: [silicon/v340l.md](../silicon/v340l.md). PLX hop: [plx.md](plx.md), [silicon/plx-p2p-mmio.md](../silicon/plx-p2p-mmio.md). Related: [moe.md](moe.md), [deepep.md](deepep.md), [alt-engines.md](alt-engines.md), [hippih.md](hippih.md), [sglang-fork.md](sglang-fork.md).

## Topology (human plan)

- **Fast tier (retipped 2026-08-27):** 2× DGX Spark (**GB10** CUDA) — attention, router, embeddings, LM head, n-gram, **live KV**. W7800 gfx1100 row is **dead** (cards selling).
- **Capacity tier:** 8× V620 32 GB (**gfx1030**) — expert GEMMs only (`fdot2` / `sdot4`)
- **Bus rule:** ship **activations only**, **fp16 on the hop**. Spark FP4/FP8/BF16 converts **on Spark**. Two machines: Spark↔Spark is CX-7 NCCL; Spark→V620 is not HIP peer.
- **V340L:** 8 cards, Vega10 / **gfx900**, PCIe 3.0 x16 dual-die. **Later, separate host.** [silicon/v340l.md](../silicon/v340l.md).
- **R9700 / RDNA5:** Later, not a workstream. R9700 = gfx1201, 32 GB / 640 GB/s — expert SKU if bought, **not** Spark. New fatbin, no AITER RDNA4 import, no tok/s copy. RDNA5: wait for ISA. Silicon: [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md).

**Reference hardware:** two **5-slot x16 Gen4 88096** backplanes (10 GPU slots) + **8 V620**. Each board is CPU x16 + 5× GPU x16 = 96 lanes exact, PIX inside the board.

**Lane budget (corrected):** one 88096 still cannot do 2+8 x16 alone (176). **Two 5-slot boards can** (5+5 at x16). Cross-board is `PHB` unless the 88096s are cascaded (`PXB`). Occupancy box stays 4× V620 on **one** board. [plx.md](plx.md).

**Do not mix V340L + V620 on the same ROCm 7 host** (toolbox: remapped MMIO map fail). 10 slots cannot hold 8+8 cards anyway. V340L is hippih gfx900 mix/FMA, not extras.

Live form is **CUDA GB10 + HIP gfx1030**, not two HIP targets. extras cannot drive Spark. V620 stays `fdot2`/`sdot4`. Do not park live KV on V620. Decode hop is tiny; **prefill is the bus**. 88096 PIX P2P stays **V620-internal**; Spark→V620 is CX-7 / host-staged, not `hipMemcpyPeer`. Historical W7800↔V620 P2P is obsolete. Silicon: [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md).

## Engine base (locked 2026-08-20, corrected)

| Order | Engine | Why |
|---|---|---|
| **1. Now** | **vLLM fork** (`rdna2_extras`) | Stack that already compiles gfx1030 HIP. Occupancy, FA, MoE DOT, then placement glue. |
| **2. Parallel** | **SGLang rdna2 overlay** | Own serving path (radix + overlap). **Import extras HIP** after occupancy. [sglang-fork.md](sglang-fork.md). |
| **3. Next hetero** | **Llaminar** | **Leave the repo** (AGPL, still gfx906 / sm86 / GGUF). **Take** mixed-domain placement only. No gfx1030 DOT tree there. |
| **4. In-house** | **hippih** | Custom HIP engine for gfx1030 / gfx1100 / gfx900. Steal extras + SGLang serving + Llaminar placement. Repo is a stub — contract: [hippih.md](hippih.md). |

Do **not** block occupancy or HIP MoE on SGLang, Llaminar, or hippih. Do not start a second DOT tree for SGLang.

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

**HIP:** do not call `__builtin_amdgcn_fdot2` / `sdot4` on gfx900. Old HIP-Clang notes that say “fdot2 on gfx9+” mean Vega20+. On V340L the honest inner loop is packed FMA / `mad_mix` into fp32 accum. Llaminar gfx906 DOT objects **will not load**. A V340L backend is a third ISA, after occupancy + V620 HIP MoE. hippih owns that third backend when we write it. **Separate host from the V620 ROCm 7 box.**

@RDNA2_Researcher owns the silicon table; this is the engine dispatch constraint.

## Llaminar (hetero engine, after serving forks)

[Llaminar/llaminar](https://github.com/Llaminar/llaminar) — C++ kernel-centric, alpha, GGUF.

| Piece | Status |
|---|---|
| Heterogeneous domains | Native — **why it is next after serving** |
| TP / PP / MoE EP | EP WiP |
| Prefix cache | **Exists** |
| Continuous batching | **Must add** (plan / V1 HTTP non-goal) |
| ROCm | **gfx906 only** today |
| gfx1030 / gfx1100 HIP | **we write** |
| gfx900 / V340L | Later; packed mix/FMA, not their gfx906 DOT; **own host** |

## SGLang overlay (parallel, after occupancy)

Own rebase-on-release fork. Radix + overlap. Import extras HIP; refuse AITER/MFMA. Contract: [sglang-fork.md](sglang-fork.md).

## hippih (in-house, after extras + SGLang + Llaminar)

[hippih](https://github.com/BlivionIaG/hippih) — empty tree today. Same three-ISA split as this page. Do not start it before extras occupancy. Contract: [hippih.md](hippih.md).

## Implementation order

1. `fa_rdna2` occupancy + V620 HIP MoE (already first). 4× on **one** 5-slot 88096.
2. 8× V620 on two boards (5+3 or 4+4) only after PIX vs PHB is measured.
3. Spark GB10 attention/router/KV (their CUDA stack). extras does not compile for GB10.
4. Measured Spark→V620 activation hop (CX-7 / host). V620-internal PIX still from `v620_toolbox/pcie_p2p`.
5. Asymmetric activation dispatch/combine (NIC/host, not DeepEP BAR scatter).
6. Optional KV overflow park — never live-KV on V620.
7. **SGLang overlay** — radix/overlap shell + extras HIP import.
8. **Llaminar placement ideas only** — do not extend their repo / do not import AGPL.
9. **hippih** as the in-house engine (steal extras + SGLang + Llaminar).
10. V340L gfx900 packed-mix backend — Later, **separate host**, not a V620 or gfx906 drop-in.

## Cards (for VLLM_FORK_Manager)

Occupancy still first.

- Multi-tier Spark×2 + 8×V620 — Later; W7800 row dead; **two 5-slot 88096s still hold 8× V620** (PIX inside, PHB/PXB cross-board)
- V340L — Later, **own host**, do not mix with V620 on ROCm 7
- SGLang overlay — Later **after occupancy**, **this is #2 parallel** — [sglang-fork.md](sglang-fork.md)
- Llaminar after serving forks — Later **placement only**, **this is #3**; Leave the repo
- hippih in-house engine — Later, **this is #4** — [hippih.md](hippih.md)

## Sources

- Note 2026-08-18: “vllm fork then laminar”; V340L HIP check
- Note 2026-08-19: reference topology — two 5-slot 88096 backplanes; V340L separate host
- Note 2026-08-20: own SGLang path
- Note 2026-08-27: Spark×2 + 8×V620; Leave Llaminar repo
- LLVM gfx900 VOP3P: https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX900.html
- LLVM gfx906 VOP3P (DOT): https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX906.html
- LLVM `fdot2.ll` (gfx900 → mix/FMA, gfx906 → `v_dot2_f32_f16`)
- Phoronix: Vega20 DL = fdot2/sdot4/sdot8, not Vega10
- [silicon/v340l.md](../silicon/v340l.md), [silicon/hetero-moe-w7800-v620.md](../silicon/hetero-moe-w7800-v620.md), [plx.md](plx.md)
