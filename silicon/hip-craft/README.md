---
title: "HIP craft and codegen"
updated: 2026-11-24
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/3e9ca702a10da840d29cbbc0f20872980e99c4a8
---
# HIP craft and codegen

waitcnt, scopes, kernarg, builtins, the codegen stack we use (HIP + DOT) and graph-capture rules.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [hip-craft.md](hip-craft.md) | waitcnt, scopes, kernarg, builtins, occupancy workflow; §5.5 dest `rdna/common` arch traits + conversions (`3e9ca702`) [^c-3e9ca702] |
| [codegen-stack.md](codegen-stack.md) | Use HIP+DOT; ignore CK XDL, hipBLASLt, rocWMMA, AITER |
| [graph-capture.md](graph-capture.md) | Capture vs page-commit guard: no D2H on the captured path; `Tensor!` on every HIP out arg |
| [warp-sync-mask-wave32.md](warp-sync-mask-wave32.md) | HIP `*_sync` masks are 64-bit (`static_assert`); wave32 drops the upper half; non-`NDEBUG` builds trap on mask ≠ active lanes; `__ballot`/`__activemask` 64-bit; ROCm 7.14.0 clang vs trunk shuffle ISA (2026-11-24) |

## Sources

Index rows summarise the linked pages; the one footnote below backs the §5.5 row. Primary sources are cited with footnotes and a `## Sources` section on each linked page.

[^c-3e9ca702]: opengfx1030/vllm-rdna commit [`3e9ca702`](https://github.com/opengfx1030/vllm-rdna/commit/3e9ca702a10da840d29cbbc0f20872980e99c4a8) (2026-10-08) — "refactor(rdna): csrc/rocm/rdna layout skeleton, move DSV4 kernels to rdna/dsv4" (adds `rdna/common/arch.cuh` and `convert.cuh`).
