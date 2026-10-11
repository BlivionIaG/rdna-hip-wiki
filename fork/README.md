---
title: Fork — opengfx1030/vllm-rdna
owner: VLLM_FORK_Manager
pinned_tip: 1dc3bac4cda3201a1b4fb345e044578e84a272cf
verified: 2026-10-11
sources:
  - https://github.com/opengfx1030/vllm-rdna
  - https://github.com/opengfx1030/vllm-rdna/commit/1dc3bac4cda3201a1b4fb345e044578e84a272cf
  - https://github.com/opengfx1030/vllm-rdna/pull/47
  - https://github.com/opengfx1030/vllm-rdna/commit/7813953b6f1c7c111ea9080bed08c21d1d703206
  - https://github.com/users/BlivionIaG/projects/4
  - https://github.com/BlivionIaG/vllm
---

# Fork

Owned by VLLM_FORK_Manager. This folder covers the gfx1030 vLLM fork **[opengfx1030/vllm-rdna](https://github.com/opengfx1030/vllm-rdna)**: its branches, the serve locks, how a change lands, and how a gfx1030 layer picks its kernel. Kernel internals live in [kernels/](../kernels/README.md) and [silicon/](../silicon/README.md), the ROCm pin and compiler in [toolchain/](../toolchain/README.md), and per-feature engine status in [engine/fork-delta.md](../engine/serving-stack/fork-delta.md).

**Snapshot (2026-10-11).** The dest branch is `rdna_extras`, the repo's default branch[^repo]. It is now on vLLM **v0.31.0** through PR #47, merge `7813953b`[^pr47][^m47]. The tip is **`1dc3bac4`**, the merge of PR #49 (INT4 PLE table)[^tip]. Every claim in these pages is pinned to that SHA unless a different one is named.

## Pages

| Page | What it answers |
|---|---|
| [branches.md](branches.md) | Which branch is dest, what happened to the 0.28 line, version-port branches, where PRs go (never upstream) |
| [serve-locks.md](serve-locks.md) | What every gfx1030 serve keeps fixed: ROCm 7.14.0, fp16, all-reduce path, prefix caching, W4A8 opt-in, EXL3 codebook, client boundary |
| [pr-lifecycle.md](pr-lifecycle.md) | Draft PR → review gates (silicon/ISA, current-stream launches, no WMMA/MFMA/bf16-dot) → merge into `rdna_extras`; tracking on Project 4 and Linear |
| [dispatch.md](dispatch.md) | How a gfx1030 layer picks its kernel: `on_rdna_family()`/`on_rdna2()`, the W4A16 dense and MoE priority lists, FA-RDNA2, compile-time arch macros, and the stock-vLLM gates the fork works around |

## Quick facts

- Dest: `rdna_extras` on `opengfx1030/vllm-rdna`[^repo]. Tip `1dc3bac4`[^tip].
- Base: upstream tag `v0.31.0` = `db9527a4`[^m47].
- Tickets: [GitHub Project 4 "GFX1030 inference"](https://github.com/users/BlivionIaG/projects/4) (private) and the Linear project **V620 inference** in team UNC. See [pr-lifecycle.md](pr-lifecycle.md#tracking).

## Moved from the old README

The old one-page README had a "Gates (stock vLLM)" list. It is now [dispatch.md § Stock vLLM gates](dispatch.md#stock-vllm-gates-v0310), re-checked against `v0.31.0` and the fork tip. It also named the human branch `perf/rdna2_w4a16` and a bot branch `perf/rdna2_w4a16_bot`. `perf/rdna2_w4a16` exists only on the old overlay repo [BlivionIaG/vllm](https://github.com/BlivionIaG/vllm), whose description now reads "MOVED: dest is opengfx1030/vllm-rdna (rdna_extras)"[^oldrepo]. `perf/rdna2_w4a16_bot` is on neither repo (`git ls-remote`, 2026-10-11). The current branch layout is in [branches.md](branches.md).

## Sources

[^repo]: Repo and default branch `rdna_extras`: <https://github.com/opengfx1030/vllm-rdna>
[^tip]: Tip commit `1dc3bac4`, "Merge feat/ple-int4: INT4 PLE table on the upstream PLE path (#49)": <https://github.com/opengfx1030/vllm-rdna/commit/1dc3bac4cda3201a1b4fb345e044578e84a272cf>
[^pr47]: PR #47 "[RDNA2] Port rdna_extras onto vLLM v0.31.0", merged 2026-10-11: <https://github.com/opengfx1030/vllm-rdna/pull/47>
[^m47]: Merge commit `7813953b`, parents `db9527a4` (= upstream tag `v0.31.0`) and `20edef53` (`rdna_extra/v0.31.0`): <https://github.com/opengfx1030/vllm-rdna/commit/7813953b6f1c7c111ea9080bed08c21d1d703206>
[^oldrepo]: Old overlay repo, description "MOVED: dest is opengfx1030/vllm-rdna (rdna_extras). This repo is the old rdna2_extras overlay archive.": <https://github.com/BlivionIaG/vllm>
