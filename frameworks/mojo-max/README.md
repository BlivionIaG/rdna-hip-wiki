---
title: "Mojo / Modular MAX (sandbox)"
sources:
  - https://docs.modular.com/packages/
  - https://mojolang.org/docs/requirements/
  - https://forum.modular.com/t/programming-the-steam-decks-gpu-with-mojo-and-max/3282
  - https://github.com/modular/modular/blob/135c332fec9b326cab8e2f3951a0b2f31d017a5d/max/kernels/src/layout/tensor_core.mojo#L1961-L1966
  - https://max.modular.com/releases/v26.5/
  - https://max.modular.com/releases/v26.6/
---
# Mojo / Modular MAX (sandbox)

Date: 2026-09-28. Priority target: gfx1030 (V620).[^modular-packages]  
**Dest produce = HIP / MojoNotProduce.** Do not ship Mojo/MAX into vllm-rdna.[^modular-packages]

## Take / Leave

**Take (sandbox/zoo):** Mojo GPU programming on gfx1030 via HIP; open MAX kernel *ideas* → hand HIP later.[^modular-packages]  
**Leave:** MAX serve/runtime as a product path; Mojo in the fork; Instinct MXFP/MFMA/WMMA; any “drop-in MAX instead of vLLM” plan.[^modular-packages]

## AMD / gfx1030 facts

- Docs: AMD HIP backend; serving CI = MI355/MI300/MI325. Known-compatible **dev**: RX 6900 → **gfx1030**, Van Gogh → gfx1033.[^modular-packages][^mojo-req]  
  https://docs.modular.com/packages/ · https://mojolang.org/docs/requirements/ [^modular-packages][^mojo-req]
- 2026-06-27 Modular: Deck RDNA2 Mojo examples OK; MAX graphs zeros; **no RDNA2 model kernel paths yet**.[^modular-deck]  
  https://forum.modular.com/t/programming-the-steam-decks-gpu-with-mojo-and-max/3282 [^modular-packages]
- Code: RDNA1/2 tensor-core path is a compile-time failure — `comptime assert False, "RDNA1/RDNA2 tensor core support requires fallback paths (not yet implemented)"` (earlier spelled `constrained[False, …]`).[^modular-tc]
- MAX 26.5 (2026-08-11) / 26.6 (2026-09-17): `max.gpu` rehome; AMD work = CDNA4/MI355 + RDNA3+ WMMA.[^modular-packages][^mojo-req][^max-265][^max-266]  
  https://max.modular.com/releases/v26.5/ · https://max.modular.com/releases/v26.6/ [^max-265][^max-266]

## Harvest rules

- Ideas only: layout/tile, fused RMSNorm/RoPE, EP/dispatch structure.[^modular-packages]
- Re-implement in HIP for gfx1030 (`fdot2` / `sdot4`). No Mojo runtime on dest.[^modular-packages]

## Related

- Dest Triton baseline: [engine/stock-baseline/triton-rocm.md](../../engine/stock-baseline/triton-rocm.md)
- Portable zoo: [../portable/README.md](../portable/README.md)[^modular-packages]

## Sources

[^modular-packages]: Modular docs, packages / GPU support. <https://docs.modular.com/packages/>
[^mojo-req]: Mojo system requirements (known-compatible GPUs). <https://mojolang.org/docs/requirements/>
[^modular-deck]: Modular forum, *Programming the Steam Deck's GPU with Mojo and MAX* (2026-06-27). <https://forum.modular.com/t/programming-the-steam-decks-gpu-with-mojo-and-max/3282>
[^modular-tc]: modular/modular @ `135c332f`, `max/kernels/src/layout/tensor_core.mojo` L1961–L1966 (`comptime assert False` for RDNA1/RDNA2 tensor cores). <https://github.com/modular/modular/blob/135c332fec9b326cab8e2f3951a0b2f31d017a5d/max/kernels/src/layout/tensor_core.mojo#L1961-L1966>
[^max-265]: MAX 26.5 release notes (2026-08-11). <https://max.modular.com/releases/v26.5/>
[^max-266]: MAX 26.6 release notes (2026-09-17). <https://max.modular.com/releases/v26.6/>
