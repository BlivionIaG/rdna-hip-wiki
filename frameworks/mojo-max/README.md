# Mojo / Modular MAX (sandbox)

Date: 2026-09-28. Priority target: gfx1030 (V620).  
**Dest produce = HIP / MojoNotProduce.** Do not ship Mojo/MAX into vllm-rdna.

## Take / Leave

**Take (sandbox/zoo):** Mojo GPU programming on gfx1030 via HIP; open MAX kernel *ideas* → hand HIP later.  
**Leave:** MAX serve/runtime as a product path; Mojo in the fork; Instinct MXFP/MFMA/WMMA; any “drop-in MAX instead of vLLM” plan.

## AMD / gfx1030 facts

- Docs: AMD HIP backend; serving CI = MI355/MI300/MI325. Known-compatible **dev**: RX 6900 → **gfx1030**, Van Gogh → gfx1033.  
  https://docs.modular.com/packages/ · https://mojolang.org/docs/requirements/
- 2026-06-27 Modular: Deck RDNA2 Mojo examples OK; MAX graphs zeros; **no RDNA2 model kernel paths yet**.  
  https://forum.modular.com/t/programming-the-steam-decks-gpu-with-mojo-and-max/3282
- Code: RDNA1/2 tensor-core path `constrained[False, … not yet implemented]`.
- MAX 26.5 (2026-08-11) / 26.6 (2026-09-17): `max.gpu` rehome; AMD work = CDNA4/MI355 + RDNA3+ WMMA.  
  https://max.modular.com/releases/v26.5/ · https://max.modular.com/releases/v26.6/

## Harvest rules

- Ideas only: layout/tile, fused RMSNorm/RoPE, EP/dispatch structure.
- Re-implement in HIP for gfx1030 (`fdot2` / `sdot4`). No Mojo runtime on dest.

## Related

- Dest Triton baseline: [engine/triton-rocm.md](../../engine/triton-rocm.md)
- Portable zoo: [../portable/README.md](../portable/README.md)
