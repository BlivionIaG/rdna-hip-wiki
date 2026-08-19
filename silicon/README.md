# Silicon

How gfx1030 actually works, and what HIP can control.

| Page | Contents |
|---|---|
| [architecture.md](architecture.md) | WGP/CU, wave32/64, VGPR, L0/L1/L2/IC, no WMMA |
| [lds-tiles.md](lds-tiles.md) | 64-bank formula, GEMM/attn tile seeds, DP4A recipe |
| [cache-policy.md](cache-policy.md) | What HIP can set; IC fit; no persist/bypass |
| [rccl-p2p.md](rccl-p2p.md) | 4× V620 PCIe 4.0, RCCL, P2P attested, GB/s unmeasured |
| [plx-p2p-mmio.md](plx-p2p-mmio.md) | PEX88096/8749 lane budget, ACS `+0x6`, BAR0/MMIO, Linux dump |
| [hip-craft.md](hip-craft.md) | waitcnt, scopes, kernarg, builtins, occupancy workflow |
| [fa-occupancy.md](fa-occupancy.md) | Live `fa_rdna2` LDS/VGPR/`__launch_bounds__` |
| [codegen-stack.md](codegen-stack.md) | Use HIP+DOT; ignore CK XDL, hipBLASLt, rocWMMA, AITER |
| [valu.md](valu.md) | gfx1030 VALU: enc / size / issue / which DOT we fire. gfx1100 extras |
| [fp16-rdna2.md](fp16-rdna2.md) | Fastest FP16: explicit `fdot2` + occupancy; skinny decode, measure BLAS prefill |
| [flashkda.md](flashkda.md) | FlashKDA: no SM90/CUTLASS/bf16; 128×128 state = `fdot2` later |
| [flydsl-dot-atoms.md](flydsl-dot-atoms.md) | FlyDSL gfx1030 feasibility; formal `fdot2`/`sdot4` atoms and proof gates |
| [deepep-v620.md](deepep-v620.md) | DeepEP insight → mapped-peer PCIe scatter, not IBGDA |
| [hetero-moe-w7800-v620.md](hetero-moe-w7800-v620.md) | 2×W7800 + 8×V620: two ISAs, activations-only hop, KV stays on W7800 |
| [v340l.md](v340l.md) | V340L = Vega10 **gfx900**, dual-die; not a V620 drop-in |
| [hippih.md](hippih.md) | hippih stub: three ISAs (`fdot2` / WMMA / `mad_mix`); extras stays first |
