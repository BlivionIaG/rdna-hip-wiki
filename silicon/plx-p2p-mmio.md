# PEX 88096 / 8749 — silicon, ACS, MMIO (Linux P2P)

Date: 2026-08-19. Register / MMIO dump for the PLX/PEX hop. Engine policy lives in [engine/plx.md](../engine/plx.md). 4× V620 P2P facts stay in [rccl-p2p.md](rccl-p2p.md). Do **not** invent a measured `hipMemcpyPeer` GB/s. Occupancy still first.

**Names:** PLX Technology → Broadcom. Same silicon is sold as **PEX** *or* **PLX**. `lspci` may print `PLX`, `Broadcom`, or `PCI bridge`. VID **`10b5`** (classic PLX) or **`14e4`** (Broadcom).

## What this hop is (silicon)

A PEX is a **packet switch**, not a GPU fabric. Peer traffic is ordinary PCIe TLPs that the switch either:

1. **cuts through** downstream-port → downstream-port (PIX, what we want), or
2. **redirects** to the upstream/root because ACS is on (looks like P2P, walks the CPU, often 3× slower or hangs).

V620 has **no XGMI**. Every TP all-reduce and every future mapped-peer MoE A2A is BAR traffic through this chip. Live KV still does not ride the switch ([hetero-moe-w7800-v620.md](hetero-moe-w7800-v620.md)).

| | **PEX88096** (SS02-0B00-00 / -02) | **PEX8749** |
|---|---|---|
| Spec | PCIe **4.0** 16 GT/s | PCIe **3.0** 8 GT/s |
| Data lanes | **96** (+ 2 mgmt = 98 logical) | **48** |
| Ports | any port x1…x16, any port up or down | 18 ports, same widths |
| Cut-through x16→x16 | **< 100 ns** (brief); family table **105 ns** | **126 ns** max (some family tables 150 ns) |
| NTB | up to **48** NT2.0 | **2** NT; up to 6 hosts |
| Switch DMA | up to **48** channels | **4** channels |
| MPS | 2 KB | 2 KB |
| Mgmt | ARM Cortex-R4; **Base Mode** = no FW | no ARM; EEPROM / I²C |
| Pkg / typ W | 37.5×42.5 mm, **35.78 W** | 27×27 mm, **7.3 W** |

Sources: [BC-0484EN](https://docs.broadcom.com/doc/BC-0484EN), [BC00-0445EN](https://docs.broadcom.com/doc/BC00-0445EN), [PEX8749 brief](https://docs.broadcom.com/doc/12351856).

**Do not quote ~3 TB/s.** That is 96 × 16 GT/s line-rate marketing, not a GPU hop.

## Lane budget (the number that kills a 10-GPU x16 dream)

Payload one way, 128b/130b (same math as [rccl-p2p.md](rccl-p2p.md)):

| Link | Payload |
|---|---|
| Gen4 x16 (88096, LnkSta 16 GT/s) | **31.508 GB/s** |
| Gen4 x8 | **15.754 GB/s** |
| Gen3 x16 (8749, or a gen-dropped 88096 port) | **15.754 GB/s** |
| Gen3 x8 | **7.877 GB/s** |

A V620 **slot** is Gen4 x16. If the path is 8749, **the switch is the gen drop** — `LnkSta` Speed becomes 8 GT/s even when `LnkCap` still says 16.

Lane count is hard:

| Mesh | Lanes needed | One 88096 (96) | One 8749 (48) |
|---|---|---|---|
| CPU x16 + 4× V620 x16 (today) | 80 | **yes**, 16 left | **no** (need 80) |
| CPU x16 + 4× V620 x8 | 48 | yes | **yes, exact**, all Gen3 |
| CPU x16 + 5× GPU x16 | 96 | exact | no |
| CPU x16 + 2× W7800 x16 + 8× V620 x16 | 176 | **no** | no |
| CPU x16 + 2× W7800 x8 + 8× V620 x8 | 96 | **exact**, all x8 | no |
| Two 88096 cascaded | — | `NCCL_P2P_LEVEL=PXB` | — |

So: **one 88096 cannot give every GPU in a 2+8 hetero box an x16.** Either a second 88096 (PXB) or x8 everywhere. One 8749 cannot even do today's 4× V620 at x16.

`lspci -vv` **LnkSta** (negotiated), not **LnkCap**.

## Mode we want vs modes we ignore

**Use — single-root transparent (Base Mode)**

- One CPU root → one upstream port → N downstream GPU ports.
- 88096 **Base Mode**: embedded ARM **off**, device is a standard PCIe fan-out. No FW required ([BC-0484EN](https://docs.broadcom.com/doc/BC-0484EN)).
- Peer MWr/MRd stay **inside** the switch (cut-through).

**Ignore — not the TP / mapped-peer path**

| Feature | Why not |
|---|---|
| **NTB / NT2.0** | Two *hosts*. Presents an endpoint + BAR translation + doorbells + scratchpads. Single-root GPU mesh does not need it. Do not turn a DSP into NT. |
| **Switch DMA** | Host/I-O any-to-any on the PEX. **Not** GPU SDMA. Not DeepEP. |
| **Synthetic / MPT / I/O-share / SR-IOV ACS-on** | Virt / NVMe JBOF. V620 SR-IOV BIOS table wants ACS **on**; bare-metal TP wants ACS **off**. |
| **Cascaded 88096** | Later. Extra hop → `PXB`. Measure one-switch PIX first. |

## ACS register dump (the bit that steals PIX)

PCIe ECAP **ACS**, ID **`0x0D`** (PCI-SIG). On each **bridge function** (upstream + every DSP):

| Offset from ECAP | Width | Name |
|---|---|---|
| `+0x00` | dword | PCI Express Extended Cap header (`ID=000D`) |
| `+0x04` | word | **ACS Capability** (which features exist) |
| `+0x06` | word | **ACS Control** — the writable trap |
| `+0x08` | var | Egress Control Vector (only if Cap bit 5) |

`setpci` name: `ECAP_ACS`. Control is **`ECAP_ACS+0x6.w`**.

ACS Control bits (PCI-SIG; `lspci -vv` `ACSCtl` names):

| Bit | `lspci` | On (+) means |
|---|---|---|
| 0 | `SrcValid` | Source Validation. Peer TLP is **validated / bounced to root**. **This is the PIX killer.** |
| 1 | `TransBlk` | Translation Blocking |
| 2 | `ReqRedir` | P2P Request Redirect → upstream |
| 3 | `CmpltRedir` | P2P Completion Redirect → upstream |
| 4 | `UpstreamFwd` | Upstream Forwarding |
| 5 | `EgressCtrl` | P2P Egress Control Vector |
| 6 | `DirectTrans` | Direct Translated P2P |

**Pass:** every PEX port that sits above a GPU shows `ACSCtl: SrcValid- …` and `setpci … ECAP_ACS+0x6.w` reads **`0000`**.

Clear (root, **re-apply after every reboot** — NCCL GPU troubleshooting, PLX/Broadcom explicit):

```bash
# find PEX / PLX bridges (VID 10b5 or 14e4; class 0604)
lspci -d 10b5: ; lspci -d 14e4: ; lspci -d ::0604

for BDF in $(lspci -d ::0604 | awk '{print $1}'); do
  setpci -s "$BDF" ECAP_ACS+0x6.w >/dev/null 2>&1 || continue
  echo -n "$BDF before="; setpci -s "$BDF" ECAP_ACS+0x6.w
  setpci -s "$BDF" ECAP_ACS+0x6.w=0000
  echo -n "$BDF after=";  setpci -s "$BDF" ECAP_ACS+0x6.w
done
```

`pcie_acs_override=downstream,multifunction` is a **kernel patch**. It changes IOMMU *groups*. It does **not** write `+0x6`. Distro kernels often strip it. Prefer `setpci` on the PEX ports. A systemd oneshot after `pci` is the usual persist.

Bare-metal TP: ACS **off**. V620 **SR-IOV** BIOS table: ACS **on** — that page is virt, not this mesh.

## V620 / W7800 BAR map (what a peer store actually hits)

ROCm BAR-memory + amdgpu (same as [rccl-p2p.md](rccl-p2p.md)):

| BAR | Width | What | P2P rule |
|---|---|---|---|
| **BAR0-1** | 64-bit **prefetchable** | **VRAM** (large-BAR) | Entire VRAM. V620 **32G**, W7800 **48G**. Start **< 2^44**. This is the mapped-peer / DeepEP-analogue window. |
| **BAR2-3** | 64-bit prefetchable | **Doorbell** | Same < 2^44. Not the collective payload. |
| **BAR5** | 32-bit non-prefetchable | **MMIO remap** | < 4 GB. Host/driver. Pre-gfx1030 cards in the same host break HSA MMIO map (toolbox rule). |

`lspci -s <gpu> -v` first prefetchable region must be **`size=32G`** (V620) or **`size=48G`** (W7800), not **256M**. 256M = no large-BAR = no PCIe P2P ([HIP #3103](https://github.com/ROCm/HIP/issues/3103)).

MMIO hole (prefetchable, Above 4G):

| Cards | BAR0 only |
|---|---|
| 4× V620 | **128 GiB** |
| 2× W7800 | **96 GiB** |
| 2× W7800 + 8× V620 | **352 GiB** |

Plus doorbells + bridge windows + NIC. BIOS: **Above 4G Decoding on**, **ReBAR / Resizable BAR on**, **CSM off**, **MMIO High Base/Size** so every BAR0 start is **< 2^44** (16 TiB). Kernel helper: `pci=realloc`.

`/proc/iomem` must show four (or ten) `32G`/`48G` `amdgpu` windows above 4G, not a 256M clip.

Peer path: GPU A SDMA/shader issues **MWr/MRd** against GPU B **BAR0 bus address**. Switch routes TLP DSP→DSP. ACS `SrcValid+` turns that into a host bounce. Switch DMA is never in this path.

## Other caps that must be on the wire

| Cap | Why |
|---|---|
| **AtomicOpCompleter** on root + PEX + GPU (`lspci` `AtomicOps`) | ROCm **requires** PCIe atomics. Missing → multi-GPU already dead. |
| **ARI** | 88000 advertises it. Fine; we are not VF-dense. |
| **MPS / MRRS** | Switch *can* do 2 KB. GPUs usually negotiate **256 B**. Do not force 2 KB. Check `MaxPayload` / `MaxReadReq` on GPU **and** the DSP above it — they must match. |
| **10-bit tag / relaxed ordering** | 88000 has device-specific relaxed ordering. Leave default until a bench says otherwise. |
| **DPC/eDPC** (88000) | Surprise-remove containment. Leave on. |

## Linux dump (paste this, then we have numbers)

```bash
lspci -tv                                 # GPUs under ONE PEX, not N CPU RPs
lspci -d 1002:73a1 -vv                    # V620 PF 73a1: BAR0 size, LnkSta, AtomicOps
lspci -d 1002:73ae -vv                    # V620 VF, should be unused on bare metal
lspci -vv | rg -n "PLX|Broadcom|PEX|ACSCtl|ACSCap|LnkSta|LnkCap|AtomicOps|Memory at|Resizable BAR"
cat /proc/cmdline                         # iommu=, pci=realloc, pcie_acs_override, amdgpu.pcie_p2p
cat /proc/iomem | rg -n "amdgpu|PCI"
# ACS control words
for BDF in $(lspci -d ::0604 | awk '{print $1}'); do
  printf '%s ACS+6=%s  %s\n' "$BDF" "$(setpci -s $BDF ECAP_ACS+0x6.w 2>/dev/null)" "$(lspci -s $BDF)"
done
```

HIP / RCCL (same pass/fail as [rccl-p2p.md](rccl-p2p.md) §5.7):

- `hipDeviceProp.isLargeBar == 1`
- `hipDeviceCanAccessPeer` all 1s
- `HSA_FORCE_FINE_GRAIN_PCIE=1`, `amdgpu.pcie_p2p=1`
- `NCCL_P2P_LEVEL=PIX` on one switch; `PXB` if cascaded; `PHB` is CPU-root (not this plan)
- `all_reduce_perf` + pairwise `hipMemcpyPeer` **ACS clear vs ACS on**, **88096 vs 8749** if both exist
- If RCCL “P2P” is ~215 ms, ship `NCCL_P2P_DISABLE=1` (SHM). P2P-on can be worse ([ROCm #6576](https://github.com/ROCm/ROCm/issues/6576))

IOMMU: remap is AMD's Radeon-P2P story; `iommu=pt` is the RCCL hang workaround. **Measure both.** `iommu=off` is last resort.

**`switchtec`:** Linux `switchtec` is the Microsemi/Microchip MRPC node (`/dev/switchtec*`). A Broadcom 88096 in **Base Mode** is a standard PCI bridge — do not expect `/dev/switchtec0`. 8749 never has it. Use `lspci`/`setpci`/Broadcom SDK, not a Microsemi userspace.

## Clock / PHY (only if LnkSta is wrong)

- Common refclk vs **SRIS** (88000 supports both). Mixed SSC domains without SRIS → training fail or silent Gen drop.
- Per-port speed is independent on 88000. A DSP can sit at Gen3 while the USP is Gen4 — that GPU is then a Gen3 hop.
- 8749 is Gen3 only. Putting a V620 under it **is** the gen drop.
- Eye / VisionPAK / SerDes dump is board-bring-up, not a vLLM ticket.

## HIP kernel note (so this page stays silicon)

A gfx1030 kernel does not program the PEX. It issues global stores/loads. Those become TLPs against **peer BAR0**. Occupancy / `fdot2` work is unchanged. Custom AR / mapped-peer scatter is **blocked on a measured PIX busbw**, not on more ISA research. Occupancy still first.

## Sources

- https://docs.broadcom.com/doc/BC-0484EN (PEX88000 brief, 2019-07-17)
- https://docs.broadcom.com/doc/BC00-0445EN (family table: 88096 98/98/105 ns/48 NTB/35.78 W; 8749 48/18)
- https://docs.broadcom.com/doc/12351856 (PEX8749 brief, 2011-08-22; 126 ns, 2 NT, 4 DMA)
- PCI-SIG ACS ECAP `0x0D`, Control at `+0x06`
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html (`setpci ECAP_ACS+0x6.w=0000`, PLX re-apply)
- https://rocm.docs.amd.com/en/docs-7.2.0/how-to/Bar-Memory.html
- https://github.com/ROCm/HIP/issues/3103 (large-BAR = first prefetchable `size=` VRAM)
- https://docs.kernel.org/gpu/amdgpu/module-parameters.html (`pcie_p2p`, `rebar`)
- [engine/plx.md](../engine/plx.md), [rccl-p2p.md](rccl-p2p.md), [deepep-v620.md](deepep-v620.md)
- [v620_toolbox pcie_p2p](https://github.com/BlivionIaG/v620_toolbox/tree/main/pcie_p2p)
