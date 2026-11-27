# Interconnect

4x V620 over PCIe 4.0: RCCL, P2P, PLX switch and MMIO, the custom one-shot all-reduce, DeepEP mapping and the hetero MoE layout.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [rccl-p2p.md](rccl-p2p.md) | 4× V620 PCIe 4.0, RCCL, P2P attested, GB/s unmeasured |
| [rdna-allreduce.md](rdna-allreduce.md) | T44 push AR: one-shot (`uint4` push), hier, and the opt-in two-shot mesh back @ `8d141bdc` [^c-8d141bdc]; Uncached staging + flags; opt-in VLLM_RDNA_AR=1 |
| [plx-p2p-mmio.md](plx-p2p-mmio.md) | PEX88096/8749 lane budget, ACS `+0x6`, BAR0/MMIO, Linux dump |
| [deepep-v620.md](deepep-v620.md) | DeepEP insight → mapped-peer PCIe scatter, not IBGDA |
| [hetero-moe-w7800-v620.md](hetero-moe-w7800-v620.md) | Spark×2 (GB10 CUDA) + 8×V620: activations-only hop, KV on Spark; W7800 row dead |

## Sources

This is an index page; it makes no technical claims of its own. Primary sources are cited with footnotes and a `## Sources` section on each linked page.

[^c-8d141bdc]: opengfx1030/vllm-rdna merge [`8d141bdc`](https://github.com/opengfx1030/vllm-rdna/commit/8d141bdc80b3124315a4a1e52ab4485cb6974f5e) (2026-11-27 08:45 Paris), merge of #229; range [`93c59de2...8d141bdc`](https://github.com/opengfx1030/vllm-rdna/compare/93c59de2b50ccea1b26f299e2ac5713861ffb2e3...8d141bdc80b3124315a4a1e52ab4485cb6974f5e) merges #221, #222, #224, #225, #228, #229. HIP changes are in `9a706f6a` + `cda6ef6e` (#221) and `750bbc8e` (#222). Launch bounds, LDS arithmetic and per-line permalinks are on the linked pages.
