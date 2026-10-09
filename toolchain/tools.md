# Tooling around the RDNA HIP stack

Date: **2026-10-05**. Owned by ROCM_specialist. Only tools that touch compiler / runtime / ISA plumbing. Engine and fork tooling stays in [engine/](../engine/) and [fork/](../fork/).

## rocjitsu — AMD's own AMDGCN emulation toolkit

Lives in the ROCm systems monorepo, not in a ROCm release: `emulation/rocjitsu` in [ROCm/rocm-systems](https://github.com/ROCm/rocm-systems/tree/develop/emulation/rocjitsu) (`develop`). Three modes, per its README:

- **Simulation** — full ISA emulation with KFD emulation + `LD_PRELOAD` interposition. **No physical GPU and no kernel module required.**
- **DBT** — dynamic binary translation, running a code object built for one arch on another. Shipped guest configs today are CDNA-on-RDNA4 / CDNA-on-CDNA3 (`guest_gfx950_on_gfx1201.json`, `guest_gfx950_on_gfx942.json`, `guest_gfx950_on_simulated_gfx942.json`). CDNA4→CDNA3 is called experimental.
- **DBI** — runtime instrumentation of kernels (profiling, tracing, analysis).

Its own support table (README, 2026-09-03):

| Arch | Target | Simulation |
|---|---|---|
| RDNA1 | gfx1010 | Experimental |
| **RDNA2** | **gfx1030** | **Experimental** |
| RDNA3 | gfx110x | Beta |
| RDNA3.5 | gfx1151 | Experimental |
| RDNA4 | gfx120x | Beta |
| CDNA1/2 | gfx908 / gfx90a | Experimental |
| CDNA3/4 | gfx94x / gfx950 | Beta |
| CDNA5 | gfx1250 | Beta |

gfx900 is **not** in the table (GFX9 starts at gfx908).

### Our gap, and it is a small one

`emulation/rocjitsu/configs/` ships `gfx1100_w7900.json`, `gfx1151.json`, `gfx1201_r9700.json`, and the CDNA ones — **there is no gfx1030 config**. So "Experimental" for gfx1030 means the ISA family is modelled but nobody wrote the topology JSON. Writing a `gfx1030_v620.json` (72 CU, 32 GB, GDDR6) is a cheap, self-contained contribution and would give us a GPU-less path to run gfx1030 code objects.

### Race detector plugin

`docs/race-detector.md`: a VM plugin that tracks in-flight memory events and reports intra-workgroup hazards — reads of a register or LDS whose producing op was never synchronized (missing `s_waitcnt` / `s_barrier`). Runs entirely on CPU. Constraint: **the emulator does not accept `-O0` code objects**, so build with `-O1` or higher (`hipcc` defaults to `-O3`, so our extras line is already fine).

Being extended right now: [rocm-systems#9470](https://github.com/ROCm/rocm-systems/pull/9470) (open, `newling`, updated 2026-09-03) adds **SGPR/TTMP write-after-write** detection — an instruction overwriting a still-pending `s_load` destination, and two scalar loads to the same SGPR (scalar-memory reads may complete out of issue order). Fix in both cases is `s_waitcnt lgkmcnt(0)`. Its end-to-end tests are gfx950 only; the hazard class is ISA-generic and applies to any hand-written gfx1030 asm or inline asm in extras.

## Machine-readable ISA XML is checked in

`shared/machine-readable-isa/isa/` in the same monorepo carries full MR ISA XML per family, including **`amdgpu_isa_rdna2.xml` (11.2 MB)**, plus rdna1 / rdna3 / rdna3_5 / rdna4 and cdna1–cdna5. This is a structured instruction/encoding spec we can parse instead of scraping the RDNA2 handbook PDF — useful for building our own encoding tables, validators, and disasm cross-checks.

`emulation/rocjitsu/docs/isa-gap-audit.md` documents AMD's own workflow for auditing handbook prose vs that XML vs rocjitsu's generated decoder, and its stated audit order is RDNA4, CDNA4, CDNA3, RDNA3 — **RDNA2 is not on the list**, which is the usual shape of the gap we track.


## 2026-09-28 — rocjitsu tip activity (not dest)

TheRock systems pin `2cd17fb` carries a burst of **rocjitsu** emulator commits (hazard/waitcheck, scratch/XCC topology, AVX-512 WMMA host vectorize, DRM close-and-reuse). Still **Experimental** for gfx1030 and **still no `gfx1030` topology JSON**. Not a dest/pin lever; continue as a GPU-less contrib opportunity only.

## 2026-09-28 ~10:25 — community HIP/RDNA tools (domain pass; not dest)

Broader tooling pass vs 02:53 domain baseline. **No dest/ISA lever.** Live dest stays **7.14.0** wave32 WGP.

| Project | Class | Arch fit | Note |
|---|---|---|---|
| [warpfront/hipfire](https://github.com/warpfront/hipfire) (v0.3.1; tip active) | **Watch** (Leave dest-produce) | gfx1030 kernels present (`*.gfx1030.hip`); primary tuned on gfx1100/1201; gfx900/Vega community | RDNA-first Rust+HIP engine (Redline retained ROCr replay). Competing serve path — **not** a vLLM/hippihx dest transplant. Method-only watch for per-arch HIP kernel layout / fail-closed graph replay. |
| [carlosfundora/gfxGRAPH](https://github.com/carlosfundora/gfxGRAPH) | **Watch** | gfx1030/1031 primary | CUDA Graph→HIP Graph bridge + capture GUARD + wave planning helpers. Last push 2026-08-26 — no Sep tip churn. Useful diagnostics patterns; **Leave** monkey-patch into dest. |
| [sar/v620](https://github.com/sar/v620) | **Watch** (containers) / **Leave** dest | gfx1030 packaging | Community GHCR images (vLLM/llama.cpp/hipfire) for V620. Docs incorrectly label gfx1030 as Wave64 — ignore for dest (ours is wave32). |
| [drwolfen/radiance-vllm-r9700](https://github.com/drwolfen/radiance-vllm-r9700) | **Leave** | gfx1201 / RDNA4 only | vLLM 0.28 + libr4d W4A8 WMMA — not gfx1030/1100/900. |
| ROCm/aiter (tip hot) | **Leave** dest / CDNA-heavy | gfx950/942/1250 tips today; RDNA listed in Triton utils only | No gfx1030 kernel lever this pass. |
| HipKittens / petit-kernel | **Leave** | CDNA-only | Unchanged vs 02:53. |
| Triton (`triton-lang/triton`) AMD | **Watch** (no Take) | gfx1250 / CDNA5 / rocjitsu CI | Recent AMD commits scheduling/TDM — **no open gfx1030 PR**. |
| llama.cpp HIP PRs | **Watch** (engine) | mixed | Open: GDN chunked [#29353](https://github.com/ggml-org/llama.cpp/pull/29353); RDNA4 GDN [#28447](https://github.com/ggml-org/llama.cpp/pull/28447); gfx1012 dots [#29427](https://github.com/ggml-org/llama.cpp/pull/29427). Method signals for Inference/fork — not toolchain dest. |

rocjitsu: tip still active on systems develop; gfx1030 **still Experimental without topology JSON** (configs unchanged).

## 2026-09-29 ~10:25 — domain pass QUIET (not dest)

Vs Sep 28 ~10:25 community baseline. **No Take/Import/Now lever** for gfx1030/1100/900. Live dest stays **7.14.0** wave32 WGP. Pins already in [matrix.md](matrix.md) (HEAD `8b46a27d2476` / systems `0bf70ef` / libraries `efafc1c` / nightly `10.2.0a20260929` L+W).

| Project | Class | Note |
|---|---|---|
| hipfire | Watch Leave dest-produce | Still **v0.3.1**; master tip unchanged (`ad10b3d97ced`). Open [#788](https://github.com/warpfront/hipfire/pull/788) gfx1100 flash-partials VRAM. gfx1030 kernels still present. |
| gfxGRAPH / sar/v620 / radiance-vllm-r9700 | unchanged Leave/Watch | gfxGRAPH last push still 2026-08-26; sar Wave64 docs still wrong; radiance gfx1201 only. |
| ROCm/aiter | Leave CDNA-heavy | Merged [#4493](https://github.com/ROCm/aiter/pull/4493) gfx1101 Triton MHA — RDNA crumb, not gfx1030. |
| Triton AMD | Watch (no Take) | Open RDNA backend [#11922](https://github.com/triton-lang/triton/pull/11922)/[#11911](https://github.com/triton-lang/triton/pull/11911)/[#11992](https://github.com/triton-lang/triton/pull/11992) — **still no gfx1030 PR**. |
| llama.cpp HIP | Watch (engine) | [#29353](https://github.com/ggml-org/llama.cpp/pull/29353)/[#28447](https://github.com/ggml-org/llama.cpp/pull/28447)/[#29427](https://github.com/ggml-org/llama.cpp/pull/29427) still open; merged [#29559](https://github.com/ggml-org/llama.cpp/pull/29559); new opens #29572/#29536/#29534/#29478 — Inference/fork method, not toolchain dest. |
| [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) | **Watch** Leave dest | New (Sep 24–26). gfx1100 EXL3/DFlash2 method skim only — not gfx1030 dest transplant. |
| HipKittens / petit / CK / hipBLASLt | Leave | CDNA / libs-excluded unchanged. |

rocjitsu: tip CI/host on systems develop; gfx1030 **still Experimental without topology JSON**.

Artifact: `/workspace/rocm-rdna-domain-20260929-1025.md`.

## 2026-09-30 ~10:22 — domain pass QUIET (not dest)

Vs Sep 29 ~10:25 community baseline. **No Take/Import/Now lever** for gfx1030/1100/900. Live dest stays **7.14.0** wave32 WGP. Pins match hourly (`011629b5f098` / systems `0bf70ef` / libraries `efafc1c` / nightly `10.2.0a20260930` L+W); matrix.md tip may lag — not edited here.

| Project | Class | Note |
|---|---|---|
| hipfire | Watch Leave dest-produce | Master tip **moved** `ad10b3d`→`66e825b` (+1 registry qwen3.8 mq4-xts). Still **v0.3.1**. Open [#794](https://github.com/warpfront/hipfire/pull/794)/[#788](https://github.com/warpfront/hipfire/pull/788) gfx1100 perf. gfx1030 kernels still present. |
| sar/v620 | Watch containers / Leave dest | Tip 2026-09-29: vllm/llamacpp rdna2 fork packaging refresh. Wave64 docs still wrong (ours wave32). |
| gfxGRAPH / radiance-vllm-r9700 | unchanged Leave/Watch | gfxGRAPH last push still 2026-08-26; radiance gfx1201 only. |
| ROCm/aiter | Leave CDNA-heavy | Merged [#5917](https://github.com/ROCm/aiter/pull/5917) RDNA GEMM kpack cleanup (gfx1151/1201) — not gfx1030. |
| Triton AMD | Watch (no Take) | [#11992](https://github.com/triton-lang/triton/pull/11992) **merged** (XCD swizzle off RDNA); [#11922](https://github.com/triton-lang/triton/pull/11922)/[#11911](https://github.com/triton-lang/triton/pull/11911) still open — **still no gfx1030 PR**. |
| llama.cpp HIP | Watch (engine) | [#29353](https://github.com/ggml-org/llama.cpp/pull/29353)/[#28447](https://github.com/ggml-org/llama.cpp/pull/28447)/[#29427](https://github.com/ggml-org/llama.cpp/pull/29427) still open; merged [#29478](https://github.com/ggml-org/llama.cpp/pull/29478); #25940 RDNA4 MUL_MAT tip — Inference/fork method, not toolchain dest. |
| phoenixhaxor/exllamav3-rocm | Watch Leave dest | Tip still 2026-09-26 (unchanged). |
| HipKittens / petit / CK / hipBLASLt | Leave | CDNA / libs-excluded unchanged. |
| Baiyu-Void/gfx1030-windows-pytorch ; xullx/ROCm-gfx1030-EXL3 | Leave / Dead tip | Empty placeholder repos (Sep 29–30) — not Watch until content. |

rocjitsu: emulator tip active on systems develop; gfx1030 **still Experimental without topology JSON**.

Artifact: `/workspace/rocm-rdna-domain-20260930-1022.md`.


## 2026-10-01 ~10:21 — domain pass QUIET (not dest)

Vs Sep 30 ~10:22 community baseline. **No Take/Import/Now lever** for gfx1030/1100/900. Live dest stays **7.14.0** wave32 WGP. Pins match hourly (`eef5f6d9a658` / systems `0bf70ef` / libraries `916d478` / nightly `10.2.0a20261001` L+W); matrix.md tip lag is hourly’s job — not edited here.

| Project | Class | Note |
|---|---|---|
| hipfire | Watch Leave dest-produce | **v0.4.0** released 2026-09-30 (master tip `66e825b`→`a89ed0a8e9d8` beta promote). gfx1030 kernels still present. Open [#794](https://github.com/warpfront/hipfire/pull/794)/[#788](https://github.com/warpfront/hipfire/pull/788) gfx1100. Competing serve — not dest transplant. |
| sar/v620 / gfxGRAPH / radiance / phoenixhaxor | unchanged Leave/Watch | sar tip still 2026-09-29; gfxGRAPH 2026-08-26; radiance gfx1201 only; exllamav3-rocm tip 2026-09-26. |
| ROCm/aiter | Leave CDNA-heavy | Tip hot (gfx1250/950/942); open gfx1100/1151/1101 PRs — not gfx1030. |
| Triton AMD | Watch (no Take) | [#11910](https://github.com/triton-lang/triton/pull/11910) **merged** (WGP/CU mode for RDNA, default off); [#11922](https://github.com/triton-lang/triton/pull/11922)/[#11911](https://github.com/triton-lang/triton/pull/11911) still open — **still no gfx1030 PR**. |
| llama.cpp HIP | Watch (engine) | [#29353](https://github.com/ggml-org/llama.cpp/pull/29353)/[#28447](https://github.com/ggml-org/llama.cpp/pull/28447)/[#29427](https://github.com/ggml-org/llama.cpp/pull/29427) still open; merged [#29572](https://github.com/ggml-org/llama.cpp/pull/29572) (CDNA fattn hygiene) — Inference/fork method, not toolchain dest. |
| HipKittens / petit / CK / hipBLASLt | Leave | CDNA / libs-excluded unchanged. |
| Baiyu-Void/gfx1030-windows-pytorch | Leave | Content landed (Windows PyTorch handbook + check script) — not HIP kernel tooling. xullx still empty Dead tip. |

rocjitsu: emulator tip active on systems develop; gfx1030 **still Experimental without topology JSON**.

Artifact: `/workspace/rocm-rdna-domain-20261001-1021.md`.

## 2026-10-02 ~10:22 — domain pass QUIET (not dest)

Vs Oct 1 ~10:21 community baseline. **No Take/Import/Now lever** for gfx1030/1100/900. Live dest stays **7.14.0** wave32 WGP. Pins match hourly (`e7829b712c8c` / systems `0bf70ef` / libraries `916d478` / nightly `10.2.0a20261002` L+W); matrix.md tip lag is hourly’s job — not edited here.

| Project | Class | Note |
|---|---|---|
| hipfire | Watch Leave dest-produce | Still **v0.4.0** / master `a89ed0a8e9d8`. Beta tip `fe77c083756a` (v0.4.1 CHANGELOG prep, land/041); open [#774](https://github.com/warpfront/hipfire/pull/774) Flash-Next on beta (updated today). Open [#794](https://github.com/warpfront/hipfire/pull/794)/[#788](https://github.com/warpfront/hipfire/pull/788) still OPEN. gfx1030 kernels still present. |
| sar/v620 / gfxGRAPH / radiance / phoenixhaxor | unchanged Leave/Watch | sar tip still 2026-09-29; gfxGRAPH 2026-08-26; radiance gfx1201 only; exllamav3-rocm tip 2026-09-26. |
| ROCm/aiter | Leave CDNA-heavy | Tip hot (gfx950/FlyDSL); open gfx1100/1151/1101 PRs — not gfx1030. |
| Triton AMD | Watch (no Take) | [#11922](https://github.com/triton-lang/triton/pull/11922)/[#11911](https://github.com/triton-lang/triton/pull/11911) still open (**gfx1030 not listed** in #11922); no new gfx1030 PR. AMD tip noise = gfx1250/scheduling. |
| llama.cpp HIP | Watch (engine) | [#29353](https://github.com/ggml-org/llama.cpp/pull/29353)/[#28447](https://github.com/ggml-org/llama.cpp/pull/28447)/[#29427](https://github.com/ggml-org/llama.cpp/pull/29427) still open; no new HIP merges/opens since Oct 1 domain. |
| HipKittens / petit / CK / hipBLASLt | Leave | CDNA / libs-excluded unchanged. |
| Baiyu-Void / xullx | Leave / Dead tip | Unchanged (Windows handbook / empty). |
| [tuandat3019/rdna2-llamacpp-optimizations](https://github.com/tuandat3019/rdna2-llamacpp-optimizations) | **Watch** (engine method) / Leave dest | **NEW** (created 2026-10-01). Docs-only measured Windows+ROCm llama.cpp RDNA2 opts (native q4_0 KV FA, MTP/ngram). No patches in-tree — Inference/fork skim, not toolchain dest. |
| [vallicgrr/rx6800-comfyui](https://github.com/vallicgrr/rx6800-comfyui) | **Leave** | NEW ComfyUI Windows RX 6800 notes — out of HIP inference toolchain scope. |

rocjitsu: emulator tip active on systems develop (gfx1250/AVX etc.); configs thread-tune only — gfx1030 **still Experimental without topology JSON**.

Artifact: `/workspace/rocm-rdna-domain-20261002-1022.md`.

## 2026-10-05 ~10:25 — domain pass (one craft Take; not dest)

Vs Oct 2 ~10:22 community baseline. **No dest/pin lever.** Live dest stays **7.14.0** wave32 WGP. Pins per hourly (TheRock `dd6e40583234` / amd-llvm `7667f52dd687` / systems `81d4fa3` / libraries `0d45e83`).

| Project | Class | Note |
|---|---|---|
| llama.cpp [#29927](https://github.com/ggml-org/llama.cpp/pull/29927) | **Take (method)** | HIP `__byte_perm` → `__builtin_amdgcn_perm` (`v_perm_b32`) for Q1_0 unpack; gfx906 tg128 67→129 t/s. HIP's `__byte_perm` is a byte-array C emulation in `amd_device_functions.h`, not one instruction. See [compiler.md](compiler.md) gotcha. OPEN, review required. |
| [jollyroger1480/rapier](https://github.com/jollyroger1480/rapier) | **Watch** (engine method) / Leave dest | **NEW** (2026-10-03). Single-model HIP decode for Qwen3.5-class GDN hybrids on **gfx1030** (RX 6950 XT), patches on Strata/ggml. 74 tok/s w/ exact-verify MTP. Method crumbs for Inference: residual-add + next-norm sum-of-squares folded into GEMV epilogue (−64 kernels/token), GDN step with in-kernel Q8_1 epilogue (9→3 launches/layer), one-block-per-head decode attention; float4 read ceiling ~482 GB/s on 6950 XT; pinned-buffer DMA race post-mortem. Not a vLLM transplant. |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | **Watch** Leave dest | ggml-based Qwen3.8-Flash-Next engine (created 2026-09-24, ~12k★); RX 6800/6900 listed; rapier's base. Competing serve path. |
| sar/v620 | Watch containers / Leave dest | Tip moved 10-03/04: **ROCm 10.0.0** base image (`rocm/dev-ubuntu-26.04:10.0.0-full`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `AMDGPU_TARGETS=gfx1030`), Strata ROCm image, navi21 llama.cpp fork image. Packaging only. |
| hipfire | Watch Leave dest-produce | Still **v0.4.0** / master `a89ed0a8e9d8`; beta tip `ef65c0ff88d1` (land/041zh gfx1151 Flash-Next fusion). #794/#788 **closed**; #774 still open. |
| [markb-1/rx6700xt-local-ai](https://github.com/markb-1/rx6700xt-local-ai) | Leave | NEW gfx1031 Windows compat matrix (mostly Vulkan). Not HIP toolchain. |
| Triton AMD | Watch (no Take) | [#11922](https://github.com/triton-lang/triton/pull/11922)/[#11911](https://github.com/triton-lang/triton/pull/11911) unchanged open; new AMD PRs gfx1250/gfx1170 only. |
| llama.cpp HIP | Watch (engine) | #29353/#28447/#29427 unchanged open; merged #29934 is Vulkan RDNA4. |
| aiter / HipKittens / petit / CK / hipBLASLt / gfxGRAPH / radiance / exllamav3-rocm | Leave / unchanged | No RDNA PRs in aiter since 10-02. |

rocjitsu: no config commits since 10-02; still **no `gfx1030` topology JSON** (gfx1100_w7900 exists).

Artifact: `/workspace/rocm-rdna-domain-20261005-1025.md`.

## Status

Nothing here is a dest or pin change. Compiler pin stays as in [compiler.md](compiler.md).

## 2026-10-06 ~10:20 — domain pass (one tool Take; not dest)

Vs Oct 5 ~10:25 community baseline. **No dest/pin lever.** Live dest stays **7.14.0** wave32 WGP. Pins per hourly (TheRock `bcca8cc570d4`; ROCm 10.1.0 docs went live this morning, see hourly / matrix).

| Project | Class | Note |
|---|---|---|
| llama.cpp [#29910](https://github.com/ggml-org/llama.cpp/pull/29910) + [hip-quality-check.yml](https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/hip-quality-check.yml) | **Take (tool + method)** | Q2_K MMQ spills 836 → 0 on gfx1030 and 1387 → 0 on gfx1100 by gentler unroll. Upstream spill gate (`scripts/hip/gcn-cdna-vgpr-check.py` over `-Rpass-analysis=kernel-resource-usage`) only builds gfx908, so RDNA spills slip through. Reuse the script as a fork CI gate for `gfx1030;gfx1100;gfx900`. See [compiler.md](compiler.md) gotcha. OPEN. |
| llama.cpp [#30021](https://github.com/ggml-org/llama.cpp/pull/30021)/[#30022](https://github.com/ggml-org/llama.cpp/pull/30022) | Watch (GFX900_manager) | NEW GCN MMQ config retune + stream_k on GCN, measured gfx906 (has `sdot4`; gfx900 does not), so gains may not carry to Vega 10. |
| llama.cpp [#29720](https://github.com/ggml-org/llama.cpp/pull/29720) | Leave | float4 rms_norm; author reports CDNA gains, RDNA roughly neutral. |
| vLLM [#59132](https://github.com/vllm-project/vllm/pull/59132) | Watch (engine/fork) | **Merged** 2026-10-06: opt-in `ROCM_SEGMENTED_ATTN` Triton backend for gfx11/gfx12 (decode, prefill, MTP verify). gfx1100 candidate; **not gfx1030**. |
| vLLM [#60084](https://github.com/vllm-project/vllm/pull/60084) | Watch (Triton gotcha) | Open. A single big KV backing allocation made Triton's AMD backend drop 32-bit buffer addressing (~16% regression on gfx1100); bounding views with `ptr_range()` restores it. gfx1030 applicability not verified. |
| vLLM [#59774](https://github.com/vllm-project/vllm/pull/59774) | Watch (engine) | Open `ROCM_OCTAVE` 3–4-bit KV, HIP kernels, codebook via `v_perm_b32`, measured on RDNA3 only. |
| hipfire | Watch Leave dest-produce | Master still v0.4.0 `a89ed0a8e9d8`; beta `212347998760` lands #774 QSA long-context series + QSA select tie-count overflow fix (v0.4.1 prep). |
| [RYZENNAVI/comfyui-zluda-rdna2](https://github.com/RYZENNAVI/comfyui-zluda-rdna2) | Leave | ComfyUI + ZLUDA Windows RDNA2; not HIP inference toolchain. |
| Triton AMD / aiter / Strata / rapier / sar/v620 / rocjitsu | unchanged | Triton #11922/#11911 still open, no gfx1030 PR; Strata HIP churn is gfx1151; rocjitsu still no gfx1030 topology JSON. |

Heads-up for ROCm 10.1+: `rocm-smi` is no longer in the standard build, so the `rocm-smi --showtopo` lines in [silicon/rccl-p2p.md](../silicon/rccl-p2p.md) need the `amd-smi topology` equivalent when anyone runs on 10.1 (owner: RDNA2_Researcher). Not a 7.14.0 dest issue.

Artifact: `/workspace/rocm-rdna-domain-20261006-1020.md`.

## 2026-10-07 ~10:30 — domain pass (one library Take, one runtime gotcha; not dest)

Vs Oct 6 ~10:20. **No dest/pin lever.** Live dest stays **7.14.0** wave32 WGP. Pins per hourly (TheRock `48f499ef3a75`; 10.1 still not on repo.radeon.com).

| Project | Class | Note |
|---|---|---|
| Strata [#1123](https://github.com/Niko1221/Strata/pull/1123) / [#1167](https://github.com/Niko1221/Strata/pull/1167) | **Take (library gotcha + tool)** | Author reports rocBLAS on **gfx1030** (10.2.0a20260930 and 10.0.0 / rocBLAS 5.6.0) has tuned kernels only for FP16→FP16, int8 and FP32→FP32; `hipblasGemmEx` with FP16/BF16 inputs and FP32 output (HS/BS) falls to ~5 TFLOPS fallback kernels. Widening to FP32 SGEMM: RX 6800 12K prompt 343 → 534 tok/s; pure FP16 route 619. #1167 adds `tools/hip/tune_rocblas`, a per-shape rocBLAS solution-index table for gfx1030 (+11–13% at 0.8–4K tokens). Lesson for our forks: on gfx1030, benchmark the GemmEx type combo you actually call, and treat HS/BS as suspect. Both PRs open. |
| rocm-systems [#9820](https://github.com/ROCm/rocm-systems/issues/9820)/[#9821](https://github.com/ROCm/rocm-systems/pull/9821) + RCCL threads | **Take (runtime gotcha)** | Host amdgpu older than DRM 3.64 hangs HIP VMM (`expandable_segments`) and, on Ubuntu 6.8, silently corrupts RCCL; written up in [hip-runtime.md](hip-runtime.md). |
| rocm-systems [#11618](https://github.com/ROCm/rocm-systems/pull/11618) | **Take (craft)** | `__shfl_xor` bounds-check fold; gfx1030 VALU 32 → 11. Note in [compiler.md](compiler.md). Open. |
| Triton [#11922](https://github.com/triton-lang/triton/pull/11922) | Watch | **Merged** 2026-10-07: `get_dram_gbps()` now uses GDDR6 ×16 for gfx1100/1101/1102/1200/1201 and LPDDR5X ×8 for gfx1151. **gfx1030 is not mapped**, so it keeps ×2 and reads about one-eighth of the real 512 GB/s. A one-line follow-up PR is the fix if anyone relies on it. |
| Strata [#1103](https://github.com/Niko1221/Strata/issues/1103) | Watch | RX 6800 XT intermittent GPU stalls mid-kernel on TheRock `gfx103X-all` wheels 7.14.0a20260612, clean on 7.13.0a20260515. Old nightly, host-polling engine; not reproduced on the 7.14.0 release. |
| rocm-systems [#10701](https://github.com/ROCm/rocm-systems/pull/10701) | Watch | RCCL device-compile breaks on `gfx10-3-generic` / `gfx11-generic` targets (SGPR-limit table lookup). Only matters if a fork builds generic targets. Open. |
| MIOpen [rocm-libraries#12889](https://github.com/ROCm/rocm-libraries/pull/12889) | Leave | Welford DPP ASM gated to GFX9; gfx103x/110x were already on the old denylist, so no change for us. |
| vLLM #60084 / #59774 / #59628, llama.cpp #29910, hipfire | unchanged | Still open; hipfire still v0.4.0. On #29910, IMbackK reports inconsistent speedups on ROCm 7.2.4. |

Artifact: `/workspace/rocm-rdna-domain-20261007-1030.md`.

## 2026-10-08 ~10:30 — domain pass (one compiler gotcha; not dest)

Vs Oct 7 ~10:30. **No dest/pin lever.** Live dest stays **7.14.0** wave32 WGP. Pins per hourly 10:00 (TheRock `a8a6db858256`; 10.1 still not on repo.radeon.com).

| Project | Class | Note |
|---|---|---|
| Strata [#1180](https://github.com/Niko1221/Strata/issues/1180) | **Take (compiler gotcha)** | `amdgpu_waves_per_eu(8)` miscomputes on gfx1100 with AMD clang 22 and is correct on clang 23. Written up in [compiler.md](compiler.md). Fixed in Strata 0.1.40.2. |
| hipBLASLt [rocm-libraries#10720](https://github.com/ROCm/rocm-libraries/pull/10720) | Watch (gfx1100) | TensileLite's automatic LDS pad for gfx10/gfx11 WMMA fp16/bf16 kernels is an even number of instruction widths, so it can't fix bank conflicts. gfx12 is not affected. Matters for gfx1100 hipBLASLt GEMMs; gfx1030 has no hipBLASLt path. Open, updated 2026-10-08. |
| Tensile [rocm-libraries#6996](https://github.com/ROCm/rocm-libraries/pull/6996) | Watch (gfx1030/gfx1100) | `MaxLgkmcnt` is hard-coded to 15; the ISA allows 63 on gfx10/11/12. The fix touches Tensile asm caps that rocBLAS gfx1030 kernels use. Open since May, updated 2026-10-08. |
| rocKE [rocm-libraries#13150](https://github.com/ROCm/rocm-libraries/pull/13150) / hipDNN [#13129](https://github.com/ROCm/rocm-libraries/pull/13129) | Watch (gfx1100) | Builds RDNA WMMA SDPA kernels once as `gfx11-generic` and lets hipDNN descriptors name generic targets (a specific id wins over the generic fallback). Covers gfx1100–gfx1153; no `gfx10-3-generic`, so nothing for gfx1030 or gfx900. Both open. |
| vLLM [#60400](https://github.com/vllm-project/vllm/pull/60400) / [#60413](https://github.com/vllm-project/vllm/pull/60413) / [#60418](https://github.com/vllm-project/vllm/pull/60418) | Leave (engine) | gfx1100 GDN decode configs, RDNA3 W4A8 int8-dot MXFP4 GEMV, and a TRITON_ATTN preference on gfx1100. Engine-side; LLM_Inference_specialist's lane. |
| hipfire v0.4.1 / v0.4.1.1 | Leave | Released 2026-10-07 and 2026-10-08. Model naming, tokenizer fixes, kernel packs for gfx1201/gfx1100 DFlash. No gfx1030 or gfx900 item. |
| FaisalBiyari/vLLM-ALU-RDNA2 | Watch | New repo (2026-10-07): vLLM 0.30.0 with the author's ALU attention for gfx1030. No benchmarks checked yet. |

## 2026-10-09 ~10:25 — domain pass (not dest)

| Item | Verdict | Note |
|---|---|---|
| [rocm-systems#10943](https://github.com/ROCm/rocm-systems/issues/10943) / [CTranslate2#2090](https://github.com/OpenNMT/CTranslate2/issues/2090) | **Take (runtime gotcha)** | `hipMallocAsync` default pool gives zeroed or corrupt memory on gfx1030 and now gfx1100 (ROCm 7.2). Workaround is `ReleaseThreshold = UINT64_MAX`. Written up in [hip-runtime.md](hip-runtime.md). |
| bitsandbytes [#2089](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/2089) | **Take (compiler gotcha)** | Bitcast floats around `__builtin_amdgcn_mov_dpp`; clang 19 converted numerically. In [compiler.md](compiler.md). |
| ROCm/llvm-project [#4873](https://github.com/ROCm/llvm-project/pull/4873) | Watch (merged 2026-10-09) | Comgr now accepts bare `amdgpu-amd-amdhsa--gfx900` ISA names. Only matters for tools that build target strings by hand. |
| TheRock [#8729](https://github.com/ROCm/TheRock/issues/8729) | **Take (runtime gotcha)** | MIOpen `GemmFwdRest` routes to hipBLASLt (no gfx1030 kernels); `MIOPEN_FIND_MODE=FAST` fails. Use `MIOPEN_GEMM_ENFORCE_BACKEND=1` (rocBLAS). In [hip-runtime.md](hip-runtime.md). |
| rocm-libraries [#12240](https://github.com/ROCm/rocm-libraries/issues/12240) | Watch | gfx103X rocBLAS nightly shards marked failed with 0 gtest failures (xml wrapper). Don't read a red gfx103X rocBLAS nightly as a regression without opening the log. |
| Strata [#1657](https://github.com/Niko1221/Strata/issues/1657) | Leave (engine lane) | 2×RX 6900 XT gfx1030 layer-split bench on 0.1.41. |
