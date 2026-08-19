# PLX / PEX switches — engine contract

Date: 2026-08-19. Engine page for **PEX88096** (Gen4) and **PEX8749** (Gen3). Silicon/register dump is @RDNA2_Researcher’s. Bench lives in [v620_toolbox/pcie_p2p](https://github.com/BlivionIaG/v620_toolbox/tree/main/pcie_p2p). Do **not** invent GB/s or a “P2P works” claim for an unmeasured hop. Occupancy still first.

Companions: [silicon/rccl-p2p.md](../silicon/rccl-p2p.md), [deepep.md](deepep.md), [multi-tier.md](multi-tier.md).

## Why this is engine, not just silicon

On this box every TP all-reduce and every future mapped-peer MoE A2A is **PCIe BAR traffic**. A PLX/PEX hop changes `NCCL_P2P_LEVEL` (`PIX` vs `PHB`/`PXB`), ACS policy, and the **generation ceiling**. GEMM still does not own RCCL. Live KV still does not ride the switch.

## The two SKUs (sourced briefs)

| | **PEX88096** (PEX88000 / SS02-0B00-00) | **PEX8749** |
|---|---|---|
| Gen | PCIe **4.0** (16 GT/s) | PCIe **3.0** (8 GT/s) |
| Lanes / ports | **96 data + 2 mgmt** (98 logical) | **48 lanes / 18 ports** |
| Widths | x1/x2/x4/x8/x16, any port up or down | same widths; any port can be upstream |
| Cut-through | **< 100 ns** x16→x16 (brief); family table **105 ns** | **126 ns** max x16→x16 |
| NTB | up to **48** NT2.0 ports | **2** NT ports; up to **6** hosts |
| DMA (switch) | up to **48** DMA channels/functions | **4** DMA channels |
| MPS | 2 KB | 2 KB |
| Extra | ARM Cortex-R4, `switchtec`, DPC/eDPC, SRIS, 8 TCs | ACS, Read Pacing, multicast, 2 VCs / 8 TCs |
| Pkg / typ W | 37.5×42.5 mm, **35.78 W** (family table) | 27×27 mm, **7.3 W** |

Sources: [BC-0484EN](https://docs.broadcom.com/doc/BC-0484EN), [BC00-0445EN](https://docs.broadcom.com/doc/BC00-0445EN), [PEX8749 brief](https://docs.broadcom.com/doc/12351856).

**Do not quote 3 TB/s fabric sums as GPU-to-GPU.** That is 96×16 GT/s line-rate marketing, not a V620 hop.

## Engine ceiling (math only)

PCIe payload one way, 128b/130b ([silicon/rccl-p2p.md](../silicon/rccl-p2p.md)):

| Hop | x16 payload | Ring n=4 algbw ceiling (`1.5 S/B`) |
|---|---|---|
| **88096 / Gen4** | **31.508 GB/s** | **21.005 GB/s** |
| **8749 / Gen3** | **15.754 GB/s** | **10.503 GB/s** |
| x8 at that gen | half | half |

A V620 is Gen4 x16 **to the slot**. If the path is 8749, the **switch** is the gen drop. `lspci` **LnkSta** (not LnkCap) is the number that matters.

## What we use vs what we ignore

**Use (single-root GPU mesh)**

- Transparent cut-through + ACS **cleared** on switch downstream ports so peer TLPs stay in the switch (`PIX`).
- Full x16 (or measured x8) to every V620 / W7800.
- Large-BAR: BAR0 **32G** prefetchable, start **< 2^44**. Four V620s need **≥ 128 GiB** 64-bit MMIO before doorbells.
- Linux: `CONFIG_HSA_AMD_P2P` + `CONFIG_PCI_P2PDMA` + `CONFIG_DMABUF_MOVE_NOTIFY` (already in `v620_toolbox`).
- `HSA_FORCE_FINE_GRAIN_PCIE=1` for RCCL P2P. `amdgpu.pcie_p2p=1` (default).
- `NCCL_P2P_LEVEL=PIX` when all GPUs hang off one switch. `PXB` if cascaded switches. `PHB` is CPU-root, not this plan.

**Ignore / Later (not the TP=4 path)**

- **NTB / NT2.0** — multi-host memory domains. Single-root P2P does **not** need it.
- **Switch DMA** — host/I-O any-to-any. Not GPU SDMA. Do not treat 48 DMA channels as DeepEP.
- Synthetic / MPT / I/O-sharing / SR-IOV ACS-on BIOS — virt, not bare-metal TP.
- Cascaded 88096 fabrics until one-switch PIX is measured.

## Linux / MMIO checklist (engine)

| Check | Pass |
|---|---|
| `lspci -tv` | GPUs under the **same** PEX, not four CPU RPs |
| `lspci -s <gpu> -vv` BAR0 | `size=32G` prefetchable, not 256M |
| `LnkSta` | Speed/Width you paid for (Gen4 x16 vs Gen3/x8) |
| `ACSCtl` on switch ports | `SrcValid-` after `setpci … ECAP_ACS+0x6.w=0000` (re-apply after reboot) |
| `lspci` AtomicOps | present (ROCm requires PCIe atomics) |
| BIOS | Above 4G Decoding on, ReBAR on, CSM off, ACS off for bare metal |
| IOMMU | Remap is AMD’s Radeon-P2P story; `iommu=pt` is the RCCL hang workaround. **Measure both.** |
| HIP | `hipDeviceCanAccessPeer` all 1s; `isLargeBar==1` |
| RCCL | `HSA_FORCE_FINE_GRAIN_PCIE=1`; log shows **P2P** not only SHM; small-S not ~215 ms ([ROCm #6576](https://github.com/ROCm/ROCm/issues/6576)) |
| Toolbox | `amd-smi topology` GPU↔GPU **ENABLED**, hops 2, PCIE — validated 4×V620 state in `pcie_p2p/AGENTS.md` |

`pcie_acs_override=` is a **kernel patch**, not a hardware ACS clear. Prefer `setpci` on the PEX ports.

BAR map (ROCm BAR-memory + amdgpu 2026 patch): **BAR0** VRAM (large-BAR gate), **BAR2** doorbell, **BAR5** MMIO remap. Peer-store DeepEP analogue is BAR0. Doorbell/MMIO P2P is not the collective path.

Pre-gfx1030 AMD cards in the same host break HSA MMIO map (`Failed to map remapped mmio page`) — toolbox rule.

## Engine rules that do not change

1. Occupancy on extras still first. PLX tune is **Later** platform work.
2. TP all-reduce stays **RCCL / PYNCCL**. No custom AR on gfx1030 until a measured PIX busbw exists.
3. DeepEP steal is still **mapped-peer scatter/combine over BAR0**, not IBGDA, not switch DMA. [deepep.md](deepep.md).
4. Hetero W7800↔V620: **activations only**; live KV stays on gfx1100. A 8749 hop **halves** the gen ceiling vs 88096.
5. W4A16 does not shrink the AR (residual is still FP16).
6. If RCCL “P2P” is hundreds of ms, ship `NCCL_P2P_DISABLE=1` (SHM). P2P-on can be worse than P2P-off.

## First measure (when we touch this)

Pairwise `hipMemcpyPeer` + `all_reduce_perf` **with ACS clear vs ACS on**, on **88096 vs 8749** if both exist. Then set `NCCL_P2P_LEVEL`. No tok/s from this page.

## Cards

Later. One platform card: “PEX ACS + PIX matrix” after occupancy. @VLLM_FORK_Manager: do not retip occupancy for this.

## Sources

- https://docs.broadcom.com/doc/BC-0484EN (PEX88000 brief, 2019-07-17)
- https://docs.broadcom.com/doc/BC00-0445EN (family table)
- https://docs.broadcom.com/doc/12351856 (PEX8749 brief, 2011-08-22)
- [silicon/rccl-p2p.md](../silicon/rccl-p2p.md)
- [v620_toolbox pcie_p2p](https://github.com/BlivionIaG/v620_toolbox/tree/main/pcie_p2p)
- Linux `switchtec` (PEX88000, kernel 4.11+)
