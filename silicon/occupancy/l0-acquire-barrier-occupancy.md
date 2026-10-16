---
title: "WGP L0 acquire after grid / lookback barriers vs effective occupancy (gfx1030)"
updated: 2026-10-16
sources:
  - https://github.com/pytorch/pytorch/pull/198208
  - https://github.com/triton-lang/triton/pull/10906
  - https://github.com/triton-lang/triton/commit/eaad84b14826739f0a67c332cb23862c2a47d0ea
  - https://github.com/triton-lang/triton/pull/11911
  - https://github.com/llvm/llvm-project/pull/81450
  - https://github.com/llvm/llvm-project/commit/87d771193490e6604f132752438bd8404c83498c
  - https://github.com/llvm/llvm-project/blob/e4ef861a45c0af0e8cc842216a409e8f0774ff5f/llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_qsa_decode_rdna2.cu
  - https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
  - https://llvm.org/docs/AMDGPUUsage.html
---

# WGP L0 acquire after grid / lookback barriers vs effective occupancy (gfx1030)

Lock: **Per-CU vector L0 (TCP) invalidate on acquire is a correctness / effective-occupancy craft issue, not a PIX / `llvm-calc-occupancy` MaxWaves row.** In HIP default **WGP** mode a multi-wave workgroup can sit on **both** CUs of a WGP; those CUs have **separate** L0s. An acquire that runs `BUFFER_GL0_INV` on only one exec-masked wave leaves the other CU's readers on a stale line. Barrier arrive/wait alone does **not** fix that — every reading wave must still order its payload load after an acquire that invalidates **its** CU's L0. Completes the occupancy-adjacent set next to [l0-gl1-occupancy.md](l0-gl1-occupancy.md) (thrash), [barrier-occupancy.md](barrier-occupancy.md) (HW barrier slots), [waitcnt-occupancy.md](waitcnt-occupancy.md) (scoreboard park), and the toolchain compiler gotcha ([toolchain/compiler/README.md](../../toolchain/compiler/README.md) § “cross-workgroup flags in WGP mode”).[^pt198208][^llvm-simem][^llvm-calc-occupancy]

No tok/s. No new Linear tickets. Do not restate the FA pin. Does **not** change extras HIP or dest tip — MonoKernel's agent-scope grid barrier already runs the acquire on every wave after `__syncthreads` (read-only audit @ `1c1426d5`).[^mono-barrier]

Date: **2026-10-16** Europe/Paris.

Companions: [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [waitcnt-occupancy.md](waitcnt-occupancy.md), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md), [atomic-occupancy.md](atomic-occupancy.md), [cache-policy.md](../arch-and-memory/cache-policy.md) § fences / GLC, [hip-craft.md](../hip-craft/hip-craft.md), [../../kernels/persistent/monokernel-gfx1030-dest.md](../../kernels/persistent/monokernel-gfx1030-dest.md), [../../toolchain/compiler/README.md](../../toolchain/compiler/README.md).

## Take / Leave / Later

| Timing | Verdict | Claim |
|---|---|---|
| **Now · Take** | Craft | After a remote write + sync (lookback flag, split-K publish, custom **grid** barrier), every wave that will **plain-load** the payload must execute an acquire that invalidates **its** CU's L0 — typically `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` (or workgroup-scope acquire that still emits `BUFFER_GL0_INV` in WGP). Do not confine the invalidate to the lane that polled the flag. [^pt198208][^llvm-simem] |
| **Now · Take** | Craft | Prefer **one** acquire fence after the barrier (all waves) over acquire-loads on every poll iteration. MonoKernel's `grid_barrier` documents why: a per-poll acquire would invalidate GL0/GL1 of WGs still working. [^mono-barrier] |
| **Now · Take** | Coverage **Live** | Treat MonoKernel @ `1c1426d585c9b2653265947734225721cfc7ea6e` as already matching the Take: release fence → wave-0 relaxed poll of `gen` → `__syncthreads` → **all-thread** agent acquire → then phase loads. Same agent-acquire comment on QSA merge. [^mono-barrier][^mono-qsa] |
| **Now · Leave** | | Do **not** treat `__syncthreads` / `s_barrier` / a generation-counter arrive as sufficient by themselves for cross-CU (or cross-WG) plain loads of published payload. Barrier orders waves; L0 coherence on gfx10 is software. [^cache-policy][^pt198208] |
| **Now · Leave** | | Do **not** put L0 invalidate / fence cost into PIX MaxWaves or `llvm-calc-occupancy`. This is correctness + effective latency hiding (cold refill after inv), not a VGPR/LDS/barrier slot. [^llvm-calc-occupancy][^l0-gl1] |
| **Now · Leave** | | Do **not** flip dest to `-mcumode` solely to “share one L0.” CU mode collapses per-WG L0 sharing for **workgroup** scope (LLVM: no `BUFFER_GL0_INV` for WG acquire in CU mode), but MonoKernel residency / coop-launch contracts stay WGP; see [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md). [^llvm-simem][^tri11911] |
| **Later** | | Disassemble one MonoKernel barrier exit on the dest clang pin and confirm every wave emits `BUFFER_GL1_INV` then `BUFFER_GL0_INV` (agent acquire sequence from llvm#81450). Not required to keep the Live Take above. [^llvm81450] |
| **Dead** | | Inventing a hand-coded `BUFFER_GL0_INV` from HIP without going through the fence/atomic builtins — compiler owns the sequence; ISA opcode exists but craft is the fence. [^cache-policy][^isa70648] |

## 1. What the race is (WGP, per-CU L0)

Structural claim from pytorch#198208 (closed 2026-10-14 without merge; root cause still the silicon/compiler story the wiki needs):[^pt198208]

1. Writer: plain store of payload + release flag.
2. Reader: acquire on the flag, then plain `load` of payload.
3. Triton scalarised the atomic to **one lane** and broadcast through LDS, so `buffer_gl0_inv` ran under a **single-lane EXEC** while every wave loaded the payload.
4. On RDNA **WGP**, a workgroup with `num_warps` ≥ 4 spans **both CUs** of the WGP. Those CUs have **separate** L0 caches. Waves on the CU that never ran the invalidate read a **stale** line → silent wrong prefix (~0.5 relative error on 64-bit split-scan).
5. Fingerprint: `num_warps` 1–2 pass; 4–8 fail. Padding each descriptor to its own 128 B line, or a workgroup-wide fence before the payload load, also fixed it.

Reproduced on gfx1151 / gfx1200 / gfx1101; **not** on CDNA or NVIDIA (one WG → one L1/SM or one CU). gfx1030 was not in the author's soak list, but the WGP + per-CU L0 layout is the same HIP default on gfx1030 — already noted in the toolchain gotcha and locked by LLVM's own comment.[^pt198208][^llvm-simem][^toolchain-gotcha]

Triton fixed the numeric drift for RDNA by restricting the tagged `BarrierOpConversion` path to CDNA (triton#10906, merge `eaad84b14826739f0a67c332cb23862c2a47d0ea`); pytorch#198208 closed as already fixed on Triton 3.9. triton#11911 later allows the tagged barrier again **only in CU mode**, where one L0 covers the WG — WGP keeps the generic lowering that emits the invalidate.[^tri10906][^tri11911]

jeffdaily's review on #198208 (on a later LLVM pin) also notes: when the acquire fence lowers to an invalidate executed by **every** wave before the plain payload load, the stale-L0 path cannot happen. That is the hardware-facing form of the Take row above.[^pt198208]

## 2. How gfx1030 acquire actually invalidates

Already locked on [cache-policy.md](../arch-and-memory/cache-policy.md); repeated here for occupancy framing:[^cache-policy][^llvm-simem][^llvm81450]

| Scope (HIP / LLVM) | WGP (HIP default) | CU (`-mcumode`) |
|---|---|---|
| workgroup acquire load | **GLC** (bypass / miss L0 — L0 is per CU) | no L0 bypass for WG |
| agent / system acquire fence | `BUFFER_GL1_INV` then `BUFFER_GL0_INV` (outer→inner) | same agent sequence |
| workgroup acquire fence | `BUFFER_GL0_INV` if WGP | no GL0 inv if CU |

LLVM `SIGfx10CacheControl::insertAcquire` (file tip opened this pass):[^llvm-simem]

> “In WGP mode the waves of a work-group can be executing on either CU of the WGP. Therefore need to invalidate the L0 which is per CU.”

Agent/system order fixed by llvm#81450 (`87d771193490e6604f132752438bd8404c83498c`): invalidate L1 then L0 so L0 cannot refill stale data from L1.[^llvm81450]

ISA 70648: `BUFFER_GL0_INV` — “Write back and invalidate the shader L0. Returns ACK to shader.” Cycle cost **not** published — do not invent it; profile if a hot path acquire-polls.[^isa70648][^l0-gl1]

**Barrier alone:** `__syncthreads` → `s_barrier` + implicit workgroup fence. It does **not** flush L2/IC and does **not** substitute for a per-reading-wave L0 invalidate when the producer write was on another CU or another WG.[^cache-policy]

## 3. Occupancy framing (not a PIX row)

| Mechanism | PIX MaxWaves? | What it does to a held slot |
|---|---|---|
| HW barrier slots | **Yes** ([barrier-occupancy.md](barrier-occupancy.md)) | Caps multi-wave WGs per WGP |
| L0 / GL1 thrash from too many waves | **No** ([l0-gl1-occupancy.md](l0-gl1-occupancy.md)) | Raises miss latency → effective hiding collapses |
| Acquire `BUFFER_GL0_INV` / `BUFFER_GL1_INV` | **No** | Correctness; also **cold-refills** L0/GL1 after inv → same effective-occupancy family as thrash if you inv on every poll |
| waitcnt park during fence + following loads | **No** ([waitcnt-occupancy.md](waitcnt-occupancy.md)) | Wave stays reserved while scoreboard drains |

So: **Leave** treating this as a MaxWaves knob; **Take** placing the acquire once per sync epoch (all waves), not on the hot poll path.

## 4. MonoKernel agent-scope grid barrier (read-only @ `1c1426d5`)

Dest tip `1c1426d585c9b2653265947734225721cfc7ea6e` (PR #94 MonoKernel merge). `csrc/rocm/mono_k2_rdna2.cu` `grid_barrier`:[^mono-barrier]

1. `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` — publish prior phase stores.
2. Wave 0: agent-scope atomics on `Ctrl::{count` / `Ctrl::{gen`; **relaxed** poll of `gen` with `s_sleep` backoff. Comment: *“an acquire load per poll would invalidate the GL0/GL1 caches of WGs that are still working; the acquire fence after the barrier orders the data.”*
3. Waves 1–7: optional weight prefetch while waiting (`s_rel`).
4. `__syncthreads()` — WG-local join.
5. **`__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` on every thread** (no `threadIdx` / EXEC mask) — then return sticky ok flag.

`mono_qsa_decode_rdna2.cu` states the silicon intent explicitly before merging split partials:[^mono-qsa]

> “acquire at agent scope invalidates this CU's L0/L1, so the other splits' partials can be read with plain (pipelined) loads”

### Gap vs pytorch#198208?

| Failure mode (#198208) | MonoKernel `grid_barrier` |
|---|---|
| Invalidate only on the flag-polling lane | Acquire fence runs on **all** threads after `__syncthreads` |
| Within-WG lookback spanning two L0s | Grid barrier is **cross-WG**; each WG still needs its own per-CU inv — provided by the all-wave agent fence |
| Plain payload load without ordered inv | Phase loads / QSA merge loads sit **after** the acquire |

**Verdict for dest:** **no MonoKernel HIP gap** on this race class at `1c1426d5`. Route nothing to VLLM_FORK_Manager for an L0-inv patch. Awareness only: keep the all-wave acquire; do not “optimise” it down to wave 0; keep relaxed polling.

Coverage: barrier pattern **Live** on dest; this page does not change extras.

## 5. Rank summary (gfx1030)

| Item | Rank | Timing | Notes |
|---|---|---|---|
| All-wave agent acquire after custom grid / lookback sync | **Take** | **Now** | Matches MonoKernel + LLVM legalizer |
| Relaxed poll + single post-barrier acquire | **Take** | **Now** | Occupancy-aware; avoids inv thrash while peers work |
| Barrier / `__syncthreads` alone as cross-CU acquire | **Leave** | **Now** | Wrong correctness model |
| `-mcumode` to dodge L0 race | **Leave** | **Now** | Wrong lever for MonoKernel WGP residency |
| Hand `BUFFER_GL0_INV` from HIP | **Leave** | **Now** | Use fence builtins |
| Objdump confirm inv on dest clang pin | **Later** | **Later** | Verification, not a blocker |
| Change MonoKernel barrier for #198208 | **Dead** | — | Already correct pattern |

## Sources

1. pytorch#198208 — “[inductor] Fix race in 64-bit split-scan decoupled lookback on RDNA” (closed 2026-10-14; root-cause text + jeffdaily review + closing comment pointing at triton#10906). https://github.com/pytorch/pytorch/pull/198208
2. triton#10906 — merge `eaad84b14826739f0a67c332cb23862c2a47d0ea` (2026-07-31). https://github.com/triton-lang/triton/pull/10906
3. triton#11911 — “[AMD] Allow the tagged barrier lowering on RDNA in CU mode” (WGP retains invalidate path). https://github.com/triton-lang/triton/pull/11911
4. llvm#81450 — merge `87d771193490e6604f132752438bd8404c83498c` (GL1 then GL0 on acquire). https://github.com/llvm/llvm-project/pull/81450
5. LLVM `SIMemoryLegalizer.cpp` `SIGfx10CacheControl::insertAcquire` / WGP “L0 which is per CU” comments — blob opened this pass at `e4ef861a45c0af0e8cc842216a409e8f0774ff5f`. https://github.com/llvm/llvm-project/blob/e4ef861a45c0af0e8cc842216a409e8f0774ff5f/llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp
6. AMD RDNA 2 ISA Reference Guide 70648 — `BUFFER_GL0_INV` / `BUFFER_GL1_INV`, GLC/DLC. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
7. LLVM AMDGPUUsage — sync scopes `agent` / `workgroup`; memory model tables for gfx10. https://llvm.org/docs/AMDGPUUsage.html
8. dest MonoKernel — `mono_k2_rdna2.cu` `grid_barrier` L225–324 and `mono_qsa_decode_rdna2.cu` L257–259 @ `1c1426d585c9b2653265947734225721cfc7ea6e`. https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu
9. Wiki companions already locking topology / fences: [cache-policy.md](../arch-and-memory/cache-policy.md), [l0-gl1-occupancy.md](l0-gl1-occupancy.md), [barrier-occupancy.md](barrier-occupancy.md), [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md), [toolchain/compiler/README.md](../../toolchain/compiler/README.md), [monokernel-gfx1030-dest.md](../../kernels/persistent/monokernel-gfx1030-dest.md)

[^pt198208]: pytorch#198208 description (WGP separate L0s; single-lane `buffer_gl0_inv`; num_warps fingerprint) and timeline (closed 2026-10-14 as fixed by triton#10906 / Triton 3.9). https://github.com/pytorch/pytorch/pull/198208
[^tri10906]: triton#10906 merge commit `eaad84b14826739f0a67c332cb23862c2a47d0ea`. https://github.com/triton-lang/triton/commit/eaad84b14826739f0a67c332cb23862c2a47d0ea
[^tri11911]: triton#11911 PR body — CU mode shares one L0; WGP keeps generic lowering with invalidate (`buffer_gl0_inv` count drops in CU). https://github.com/triton-lang/triton/pull/11911
[^llvm81450]: llvm-project#81450 merge `87d771193490e6604f132752438bd8404c83498c` — agent/system acquire emits `BUFFER_GL1_INV` then `BUFFER_GL0_INV`. https://github.com/llvm/llvm-project/commit/87d771193490e6604f132752438bd8404c83498c
[^llvm-simem]: `SIGfx10CacheControl::insertAcquire` and WGP “L0 which is per CU” comments in `SIMemoryLegalizer.cpp` @ `e4ef861a45c0af0e8cc842216a409e8f0774ff5f`. https://github.com/llvm/llvm-project/blob/e4ef861a45c0af0e8cc842216a409e8f0774ff5f/llvm/lib/Target/AMDGPU/SIMemoryLegalizer.cpp
[^mono-barrier]: `grid_barrier` in `csrc/rocm/mono_k2_rdna2.cu` L225–324 @ `1c1426d585c9b2653265947734225721cfc7ea6e` (release fence; relaxed gen poll comment; all-thread agent acquire after `__syncthreads`). https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_k2_rdna2.cu#L225-L324
[^mono-qsa]: `mono_qsa_decode_rdna2.cu` L257–259 @ `1c1426d585c9b2653265947734225721cfc7ea6e`. https://github.com/opengfx1030/vllm-rdna/blob/1c1426d585c9b2653265947734225721cfc7ea6e/csrc/rocm/mono_qsa_decode_rdna2.cu#L257-L259
[^cache-policy]: Wiki [cache-policy.md](../arch-and-memory/cache-policy.md) — WGP WG acquire uses GLC; agent acquire emits GL1 then GL0 inv; `__syncthreads` does not flush L2/IC.
[^l0-gl1]: Wiki [l0-gl1-occupancy.md](l0-gl1-occupancy.md) — L0/GL1 not a PIX row; thrash vs effective occupancy.
[^llvm-calc-occupancy]: LLVM `llvm-calc-occupancy` / PIX limiters are VGPR, LDS, thread-group size, barriers — not cache invalidate. https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
[^isa70648]: AMD RDNA 2 ISA 70648 — `BUFFER_GL0_INV` wording. https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
[^toolchain-gotcha]: Wiki [toolchain/compiler/README.md](../../toolchain/compiler/README.md) § “Gotcha — cross-workgroup flags in WGP mode” (PR #30 domain pass), citing pytorch#198208 / triton#10906.
