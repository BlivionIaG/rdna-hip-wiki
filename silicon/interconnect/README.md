# Interconnect

4x V620 over PCIe 4.0: RCCL, P2P, PLX switch and MMIO, the custom one-shot all-reduce, DeepEP mapping and the hetero MoE layout.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [rccl-p2p.md](rccl-p2p.md) | 4× V620 PCIe 4.0, RCCL, P2P attested, GB/s unmeasured |
| [rdna-allreduce.md](rdna-allreduce.md) | T44 push one-shot AR on dest tip a4060647; Uncached+host flags; opt-in VLLM_RDNA_AR=1 |
| [plx-p2p-mmio.md](plx-p2p-mmio.md) | PEX88096/8749 lane budget, ACS `+0x6`, BAR0/MMIO, Linux dump |
| [deepep-v620.md](deepep-v620.md) | DeepEP insight → mapped-peer PCIe scatter, not IBGDA |
| [hetero-moe-w7800-v620.md](hetero-moe-w7800-v620.md) | Spark×2 (GB10 CUDA) + 8×V620: activations-only hop, KV on Spark; W7800 row dead |
