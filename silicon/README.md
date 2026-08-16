# Silicon

How gfx1030 actually works, and what HIP can control.

| Page | Contents |
|---|---|
| [architecture.md](architecture.md) | WGP/CU, wave32/64, VGPR, L0/L1/L2/IC, no WMMA |
| [lds-tiles.md](lds-tiles.md) | 64-bank formula, GEMM/attn tile seeds, DP4A recipe |
| [cache-policy.md](cache-policy.md) | What HIP can set; IC fit; no persist/bypass |
| [rccl-p2p.md](rccl-p2p.md) | 4× V620 PCIe 4.0, RCCL, P2P unproven |
| [hip-craft.md](hip-craft.md) | waitcnt, scopes, kernarg, builtins, occupancy workflow |
| [fa-occupancy.md](fa-occupancy.md) | Live `fa_rdna2` LDS/VGPR/`__launch_bounds__` |
| [codegen-stack.md](codegen-stack.md) | Use HIP+DOT; ignore CK XDL, hipBLASLt, rocWMMA, AITER |
