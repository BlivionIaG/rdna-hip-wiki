---
title: "Quant formats"
sources:
  - https://github.com/opengfx1030/vllm-rdna/commit/e131562936f9c6f27c26449b00aaaeeb34159b4c
  - https://github.com/opengfx1030/vllm-rdna/commit/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0
  - https://github.com/opengfx1030/vllm-rdna/commit/30b0bd4e3604a6ea678244a0df74e452ac05fee6
  - https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb
  - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
  - https://github.com/opengfx1030/vllm-rdna/blob/30b0bd4e3604a6ea678244a0df74e452ac05fee6/csrc/rocm/exl3_dot2_dense.cu
  - https://arxiv.org/abs/2406.11235
  - https://github.com/opengfx1030/vllm-rdna/commit/c2cbf004772c2fa10f71e9858f98080e102107af
  - https://github.com/opengfx1030/vllm-rdna/commit/0835e3cbf57de062b09fbf6a4a804abaf6db71cb
  - https://github.com/opengfx1030/vllm-rdna/commit/f419d8d6c037003cc21c64e9a105f3fe9d06abe2
  - https://github.com/opengfx1030/vllm-rdna/commit/b74028935393d1e21f73f80f65f506c9ea4639af
---
# Quant formats

Weight/activation format contracts: pack, dequant, which DOT. Back to the [kernels index](../README.md).

| Page | Inner loop | Completeness |
|---|---|---|
| [w4a16.md](w4a16.md) | nibble dequant → `fdot2` | **Most complete** (dense). MoE GPTQ in-tree + resident skinny opt-in @ `e1315629`; MoE AWQ Not done. Modes: [engine/overview/coverage.md](../../engine/overview/coverage.md) (W4A16 row; no separate engine W4A16 page exists). K_STEP split repair @ `3a0786ea` [^c-e1315629][^c-3a0786ea] |
| [w4a8.md](w4a8.md) | int4 W × int8 A → `sdot4` | **Live / opt-in** @ `3a0786ea`; MoE epilogue CAS default / fp32 scratch opt-in @ `30b0bd4e`. Default = W4A16 [^c-3a0786ea][^c-30b0bd4e] |
| [w8a16.md](w8a16.md) | i8 → fp16 → `fdot2` | Incomplete. MoE in-tree; dense `.cu` absent at tip |
| [w8a16-fp8.md](w8a16-fp8.md) | E4M3 bit-trick → `fdot2` (LUT historical) | Dense built; MoE `.cu` excluded from build at `1dc3bac4` |
| [w8a8-fp8.md](w8a8-fp8.md) | both sides E4M3 → fp16 → `fdot2` | Incomplete. **Not** `sdot4` |
| [w8a8.md](w8a8.md) | extras vs contract: live FP8-`fdot2` / W8A16-`fdot2`; INT8 `sdot4` **absent** | **Spec / not on branch** @ `83de31cf` [^c-83de31cf] |
| [mxfp4.md](mxfp4.md) | E2M1 + E8M0 → `fdot2` | Incomplete [^ocp-mx] |
| [int2.md](int2.md) | i2 unpack → `fdot2` (later `sdot4`) | Spec. Mixed INT2/INT4 MoE = two unpackers, one DOT |
| [w4a4.md](w4a4.md) | i4×i4 `sdot8` | Explore. After W8A8. Not E2M1 |
| [sdot4-explore.md](sdot4-explore.md) | when `sdot4` is legal | Explore: W8A8 INT8, Sage QK; W4A8 → [w4a8.md](w4a8.md) Live opt-in |
| [w8a8-mxfp4.md](w8a8-mxfp4.md) | W8A8 INT8 = `sdot4` | **Spec / not added** |
| [nvfp4.md](nvfp4.md) | E2M1 + E4M3 mul → `fdot2` | Spec (no RDNA NVFP4 kernel at `1dc3bac4`) |
| [rocmfpx.md](rocmfpx.md) | codebook10 `perm` → i8 → `sdot4` | Later. llama.cpp GGUF, not a vLLM port |
| [rdna-hint-flex.md](rdna-hint-flex.md) | per-row INT2/4/8 + block (lo, step) → `fdot2` (F16) or `sdot4` (I8) | **Live, opt-in format** @ `b7402893`: F16 GEMV decode, dequant + rocBLAS prefill default, fused GEMM and I8 prefill opt-in / op-only, routed experts [^c-b740-hf] |
| [exl3.md](exl3.md) | trellis state → 3-inst → `fdot2` | Later. mul1 + K=1..8 + `exl3_project_rdna2` @ `30b0bd4e`; cb 0/1/2 all dispatch at `1dc3bac4`; decode-once v3 GEMM (`__launch_bounds__(128)`, gfx103x, bits 3/4/6, M ≤ 64) @ `c2cbf004` / `0835e3cb` / `f419d8d6`, unchanged at `64d59b89`. Occupancy + W4 first [^c-30b0bd4e][^f-30b0bd4e-exl3-dot2-dense-cu][^qtip][^c-c2cbf004] |

Do not conflate W8A16 / W8A16-FP8 / W8A8-FP8 (`fdot2`) with spec W8A8 INT8 (`sdot4`).

## Sources

Index rows summarise the linked pages; each page carries its own footnoted primary sources. Footnotes below back the diagram and any claim made on this index itself.

[^c-e1315629]: opengfx1030/vllm-rdna commit [`e1315629`](https://github.com/opengfx1030/vllm-rdna/commit/e131562936f9c6f27c26449b00aaaeeb34159b4c) (2026-09-23) — "Merge pull request #17 from GeorgeMA-Strong/codex/v620-baseline-port-20260922".
[^c-3a0786ea]: opengfx1030/vllm-rdna commit [`3a0786ea`](https://github.com/opengfx1030/vllm-rdna/commit/3a0786eaa70f37890b729ff4943e5cc9a0cb68c0) (2026-09-30) — "merge(rdna2): land w4a8-wiring onto rdna_extras (W4A8 opt-in; W4A16 default unchanged)".
[^c-30b0bd4e]: opengfx1030/vllm-rdna commit [`30b0bd4e`](https://github.com/opengfx1030/vllm-rdna/commit/30b0bd4e3604a6ea678244a0df74e452ac05fee6) (2026-10-01) — "test(exl3): backport loader unit tests + mul1 tile reformat".
[^c-83de31cf]: BlivionIaG/vllm commit [`83de31cf`](https://github.com/BlivionIaG/vllm/commit/83de31cf8c55ac2e63f538c2d08dd8f1668a52bb) (2026-09-01) — "fix(rocm/layernorm): close namespace braces before global-scope public entries".
[^ocp-mx]: Open Compute Project, *OCP Microscaling Formats (MX) v1.0 Specification* — MXFP4 = E2M1 elements with one shared E8M0 scale per 32-element block. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
[^f-30b0bd4e-exl3-dot2-dense-cu]: opengfx1030/vllm-rdna `csrc/rocm/exl3_dot2_dense.cu` @ [`30b0bd4e`](https://github.com/opengfx1030/vllm-rdna/blob/30b0bd4e3604a6ea678244a0df74e452ac05fee6/csrc/rocm/exl3_dot2_dense.cu).
[^qtip]: Tseng, Chee, Sun, Kuleshov, De Sa, *QTIP: Quantization with Trellises and Incoherence Processing*, arXiv:2406.11235 (CC BY 4.0) — bitshift trellis, computed codes 1MAD / 3INST, incoherence processing. <https://arxiv.org/abs/2406.11235>
[^c-c2cbf004]: opengfx1030/vllm-rdna commit [`c2cbf004`](https://github.com/opengfx1030/vllm-rdna/commit/c2cbf004772c2fa10f71e9858f98080e102107af) (2026-10-09) — "perf(rdna2): EXL3 decode-once GEMM for decode and MTP verify batches"; bits 6 added in [`0835e3cb`](https://github.com/opengfx1030/vllm-rdna/commit/0835e3cbf57de062b09fbf6a4a804abaf6db71cb), tile-word prefetch in [`f419d8d6`](https://github.com/opengfx1030/vllm-rdna/commit/f419d8d6c037003cc21c64e9a105f3fe9d06abe2).
[^c-b740-hf]: dest tip [`b7402893`](https://github.com/opengfx1030/vllm-rdna/commit/b74028935393d1e21f73f80f65f506c9ea4639af); new file `csrc/rocm/rdna_hint_flex_rdna2.cu` in [`8015ea48...b7402893`](https://github.com/opengfx1030/vllm-rdna/compare/8015ea485b7f6c0298d8f98ce6e824ce1c444c45...b74028935393d1e21f73f80f65f506c9ea4639af).
