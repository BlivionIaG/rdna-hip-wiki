---
title: "RCCL and PCIe P2P on 4× Radeon PRO V620 (gfx1030)"
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/platforms/rocm.py
  - https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html
  - https://www.activeport.com.au/wp-content/uploads/2023/01/V620-Datasheet_V4_5.pdf
  - https://ir.amd.com/news-events/press-releases/detail/1030/amd-radeon-pro-v620-gpu-delivers-powerful-multi-purpose-data-center-visual-performance-for-todays-demanding-cloud-workloads
  - https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/w6000-v620.html
  - https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html
  - https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/iommu.html
  - https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/system-requirements.html
  - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
  - https://docs.kernel.org/gpu/amdgpu/module-parameters.html
  - https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/Kconfig
  - https://rocm.docs.amd.com/en/docs-7.2.0/how-to/Bar-Memory.html
  - https://github.com/ROCm/HIP/issues/3103
  - https://github.com/ROCm/hip/blob/rocm-7.0.0/include/hip/hip_runtime_api.h#L249
  - https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/install-faq.html
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
  - https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/hip_runtime_api/modules/peer_to_peer_device_memory_access.html
  - https://github.com/ROCm/hip/blob/96b5aa0a/docs/how-to/hip_runtime_api/multi_device.rst
  - https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html
  - https://github.com/ROCm/legacy-rocm-build/issues/2616
  - https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.0/how-to/rccl-usage-tips.html
  - https://rocm.docs.amd.com/projects/rccl/en/latest/what-is-rccl.html
  - https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
  - https://pcisig.com/what-bit-rates-does-pcie-50-specification-support-and-how-does-it-compare-prior-pcie-generations
  - https://pcisig.com/faq?field_category_value%5B%5D=pci_express_4.0
  - https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
  - https://github.com/vllm-project/vllm/issues/38107
  - https://github.com/ROCm/legacy-rocm-build/issues/6576
  - https://github.com/ROCm/ROCm/issues/787
  - https://github.com/intel/compute-runtime/issues/935
  - https://github.com/ROCm/legacy-rocm-build/issues/1714
  - https://github.com/ROCm/rccl-tests/issues/162
  - https://github.com/ROCm/rccl/issues/1454
  - https://github.com/ROCm/HIP/issues/3352
  - https://github.com/vllm-project/vllm/issues/24160
  - https://github.com/vllm-project/vllm/issues/18805
  - https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html
  - https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py
  - https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/quick_all_reduce/
  - https://arxiv.org/abs/2307.09288
  - https://arxiv.org/html/2312.04333v4
  - https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/apidocs/bridge/bridge.models.llama.llama_provider.html
  - https://openreview.net/pdf?id=ypvmPqTWi8
  - https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html
---

# RCCL and PCIe P2P on 4× Radeon PRO V620 (gfx1030)

Audience: someone writing custom HIP for vLLM TP=4 on this box. Date of this pass: **2026-08-17** (P2P/RCCL). Host/OS retip: **2026-08-19**. ROCm 10.0 note: **2026-08-27**.

Rule: every concrete number is attributed to a URL that was opened. If a figure is not in those sources, it is marked **unknown**. Infinity Cache is not Infinity Fabric. Consumer RX 6800/6900 XT shares the ISA, not the firmware or the official support claim.

Companion notes already on disk (not re-derived here):
- `silicon/arch-and-memory/architecture.md` — silicon, no MFMA
- `engine/vllm-sglang-map.md` — vLLM/SGLang path map; custom AR gated to gfx94/95

---

## 0. What is true on 4× V620

Lead with the facts that decide the kernel, not the Instinct brochure.

**Attested (this box, 2026-08-17):** PCIe P2P works on the reference 4× V620. Treat P2P as available here. Do **not** invent a GB/s figure until a `hipMemcpyPeer` bench is pasted. Sources below still describe the general Radeon failure modes.


| Claim | Status on this SKU | Source |
|---|---|---|
| Host interconnect | **PCIe 4.0 x16 only.** No GPU-to-GPU fabric on the card. | AMD product page: Bus Type `PCIe® 4.0 x16`. Partner datasheet: `PCI Express® Interface PCIe® Gen4 x16`. |
| Infinity Fabric / XGMI | **Absent.** Discrete Radeon PRO add-in card. Infinity Cache (128 MB) is on-die last-level cache, not a chip-to-chip link. AMD’s XGMI table lists Instinct SKUs and marks **Radeon PRO V710 = N/A**; V620 is not even a row. | Product page (Infinity Cache 128 MB). Instinct virt-drv XGMI table. [^rocm-gpu-arch-specs] |
| Official ROCm (V620 footnote) | **QA scope:** gfx1030 ✅, Ubuntu **24.04.x / 22.04.5 only**. RHEL is in the *general* ROCm table; the V620 footnote **excludes** it. Not a measured fail. | ROCm 7.2 / 7.14 system-requirements footnote [8]/[9]. Engine split: [../engine/rocm-host.md](../../engine/rocm-host.md). |
| Attested host (this box, 2026-08-19) | ROCm **7.2.0** on **Fedora 43** works; **RHEL 10** works; live **7.14.0** works. Live target is **7.14**. Do not file AMD bugs as if Fedora/RHEL were supported SKUs. | Lab note. |
| RCCL in ROCm 7.2.0 | **2.27.7**. Intra-node path on this box is **PCIe only** (no xGMI). | ROCm 7.2.0 release notes component table. |
| vLLM `dist_backend` | **`"nccl"`** (RCCL under the hood). | `RocmPlatform.dist_backend = "nccl"` in `vllm/platforms/rocm.py`. [^extras-rocm-py] |
| vLLM custom all-reduce | **Off.** `use_custom_allreduce()` is gfx94/gfx95 only. | Same file, comment: “We only enable custom allreduce for MI300 series”. |
| vLLM Quick Reduce | **Off.** Arch gate is gfx94/gfx95. Also refuses `world_size > 2` unless `is_fully_connected` (XGMI, `amdsmi` link type **2**, hops **1**). | `quick_all_reduce.py`; `rocm.py` `is_fully_connected`. |
| vLLM `is_fully_connected` | **False** on V620. That helper is XGMI 1-hop, not PCIe. | `rocm.py` `is_fully_connected`. |
| What actually all-reduces | **RCCL Ring/Tree over PCIe**, via `PyNcclCommunicator`. If P2P is up: GPU BAR DMA. If not: **host bounce / SHM**. | RCCL usage tips + HIP multi-device docs. |
| Theoretical PCIe 4.0 x16 payload | **31.508 GB/s** one way (`16.0 GT/s × 16 × 128/130`). | PCI-SIG: 16.0 GT/s/lane/direction. 128b/130b is the PCIe 3.0+ line code. [^kfd-crat][^pcisig-pcie4] |
| Ring all-reduce, n=4, if the bus is B | Time `t = (S/B) × 2(n−1)/n = 1.5 S/B`. Ceiling **algbw = B/1.5**. If B = 31.508 GB/s, **algbw ≤ 21.005 GB/s**. | NCCL-tests PERFORMANCE.md. [^pcisig-pcie4] |
| W4A16 vs all-reduce | **Does not shrink the collective.** TP all-reduce is residual / row-parallel output, still **FP16** (2 B/elem). Weight quant is local. | Megatron-style TP; vLLM `RowParallelLinear`. |
| Instinct vs this box | Instinct GPU–GPU P2P is **XGMI** and “don’t use PCI/PCIe for peer-to-peer DMA”. Navi 21 P2P is **PCIe BAR + large-BAR + chipset**, optional, often broken. | amdgpu IOMMU page. |
| Custom gfx1030 AR | **P2P is attested on this 4× V620 box** (2026-08-17). Still no measured `hipMemcpyPeer` GB/s in this wiki. A gfx1030 custom AR is no longer blocked on “does P2P exist”; it is blocked on a bandwidth number and on vLLM’s 4+ GPU custom-AR policy. | Lab attestation. `custom_all_reduce.py` `should_custom_ar`. |

That is the whole box. The rest of this note is the evidence and the cost model.

---

## 1. V620 official interconnect

### 1.1 What AMD prints

AMD product page:[^v620-product]
https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html

| Field | Value |
|---|---|
| Name | AMD Radeon™ PRO V620 |
| Architecture | RDNA™ 2 |
| Stream processors | 4608 |
| Compute units | 72 |
| Ray accelerators | 72 |
| Dedicated memory | 32 GB GDDR6 |
| Infinity Cache | 128 MB [^rocm-gpu-arch-specs] |
| Peak memory bandwidth | 512 GB/s |
| Total board power | 300 W |
| Form factor | PCIe® Add-in Card |
| **Bus Type** | **PCIe® 4.0 x16** |

Partner datasheet (same numbers, extra board fields):[^activeport-01-v620-datasheet-v4-5-pdf]
https://www.activeport.com.au/wp-content/uploads/2023/01/V620-Datasheet_V4_5.pdf

| Field | Value |
|---|---|
| GPUs per board | 1 |
| Stream processors | 4608 |
| Compute units | 72 |
| Ray accelerators | 72 |
| Peak clock | 2200 MHz |
| Memory | 32 GB GDDR6, 256-bit, 16 Gbps, 512 GB/s |
| Display connectors | **None** |
| **PCI Express® Interface** | **PCIe® Gen4 x16** |
| Peak board power | 300 W |
| Power | 2× 8-pin |
| Form factor | Passive, full height, dual slot, 10.5" |

AMD IR press (2021-11-04): 4608 SPs, 72 CUs, 32 GB @ 16 Gbps, 512 GB/s, 256-bit.[^ir-1030-amd-radeon-pro-v620-gpu-delivers-powerfu]
https://ir.amd.com/news-events/press-releases/detail/1030/amd-radeon-pro-v620-gpu-delivers-powerful-multi-purpose-data-center-visual-performance-for-todays-demanding-cloud-workloads

There is **no** second connector, no XGMI cage, no “Infinity Fabric link” row on any official V620 page opened here.

ROCm 7.2.0 RDNA2 system-optimization page identifies the tested card as **RDNA2 V620 (D603GLXE)**, PF `Device 73a1`, VF `Device 73ae`.[^rocm-system-optimization-w6000-v620-html]
https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/w6000-v620.html

### 1.2 Infinity Cache ≠ Infinity Fabric / XGMI

- **Infinity Cache** on V620 is a 128 MB on-die cache in the memory hierarchy (product page, press, datasheet).[^rocm-gpu-arch-specs]
- **XGMI** is “AMD’s high-speed GPU-to-GPU interconnect based on Infinity Fabric™”. Physical XGMI connector on supported Instinct GPUs; joins peer VRAM into a homogeneous space. “XGMI enables P2P transfers that are generally faster than those over PCIe.”[^instinct-userguides-xgmi-configuration-html]
  https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html

XGMI support table on that page (complete as published):

| GPU | Infinity Fabric (XGMI) |
|---|---|
| Instinct MI350X/MI355X | 2/4/8 GPUs |
| Instinct MI325X | 2/4/8 GPUs |
| Instinct MI300X | 2/4/8 GPUs |
| Instinct MI210X | 4/8 GPUs |
| **Radeon Pro V710** | **N/A** |

V620 is a discrete Radeon PRO add-in card. It is not in the XGMI table. The later V710 (RDNA 3, still a discrete PRO) is explicitly **N/A**. Treat V620 the same: **no XGMI**.

amdgpu IOMMU page, same point from the other side:

> AMD Instinct GPUs are connected via XGMI links and don’t use PCI/PCIe for peer-to-peer DMA.[^instinct-conceptual-iommu-html]

https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/iommu.html

V620 is the opposite: **all** GPU–GPU traffic is PCIe.

### 1.3 ROCm support — official matrix vs this box

Split lives in [../engine/rocm-host.md](../../engine/rocm-host.md). Do **not** read the Ubuntu footnote as “this box cannot run.”

**Official (AMD QA, do not erase).** https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/system-requirements.html — and the same footnote on [7.14 latest](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).[^rocm-reference-system-requirements-html][^rocm-reference-system-requirements-html-2]

AMD Radeon PRO table: **V620, RDNA2, gfx1030, ✅**, footnote [8] (7.2) / [8]–[9] (7.14): **Ubuntu 24.04.x and 22.04.5 only**. RHEL is listed in the *general* OS table; the V620 footnote **excludes** it (`RHEL … except AMD Radeon PRO V620`). That is **QA scope**, not physics. Fedora is not in the matrix.

(Later 7.2.3 docs bump the Ubuntu 24.04 point release to 24.04.4. The official V620 restriction stays Ubuntu-only.)

W6800 is the other official gfx1030 PRO SKU (footnote [7], Ubuntu + RHEL). Consumer RX 6800/6900 XT are **not** in the Linux support table. Same ISA, not the same claim.

**Attested (2026-08-19).** ROCm **7.2.0** on **Fedora 43** works. **RHEL 10** works. Live stack is ROCm **7.14.0**. Host/runtime only — no tok/s, no P2P GB/s from this retip. Live target: **7.14**. Docker images may still be Ubuntu (image policy, not a host ban). Do not mix V340L onto this ROCm 7 host ([v340l.md](../other-hardware/v340l.md)).

CPU requirement (same page): **PCIe atomics**. “Modern CPUs after the release of 1st generation AMD Zen CPU and Intel Haswell support PCIe atomics.” If the root port does not advertise atomics, ROCm multi-GPU is already in a bad state (see §5).

Same page, multi-GPU hang warning:

> Systems with multiple GPUs may require `iommu=pt` to be set at boot time to prevent application hangs, as described in Issue #5.

---

## 2. amdgpu / HIP P2P — what ROCm actually supports

### 2.1 The kernel switch is `amdgpu.pcie_p2p`, not `amdgpu.p2p`

Linux amdgpu module parameters:[^docs-amdgpu-module-parameters-html]
https://docs.kernel.org/gpu/amdgpu/module-parameters.html

| Parameter | Meaning |
|---|---|
| **`pcie_p2p`** (bool) | “Enable PCIe P2P (**requires large-BAR**). Default value: **true (on)**.” |
| **`use_xgmi_p2p`** (int) | “Enables/disables XGMI P2P interface (0 = disable, 1 = enable).” Irrelevant on V620. |
| `rebar` (int) | “Allow BAR resizing. Disable this to prevent the driver from attempting to resize the BAR if the GPU supports it and there is available MMIO space.” BIOS may already have resized. |
| `debug_mask` bit `0x2` | Simulate large-BAR on a small-BAR card by **lying about VRAM size** (visible size, usually **256 MB**). Test-only. |

There is **no** `amdgpu.p2p` parameter in that table. People say `amdgpu.p2p`; the knobs are `pcie_p2p` and `use_xgmi_p2p`. Boot form: `amdgpu.pcie_p2p=1` (default).

Kconfig (`HSA_AMD_P2P`):[^github-amdkfd-kconfig]
https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/Kconfig

> Enable peer-to-peer (P2P) communication between AMD GPUs **over the PCIe bus** … only enabled on **compatible chipsets**, and between GPUs with **large memory BARs** that expose the **entire VRAM** in PCIe bus address space **within the physical address limits of the GPUs**.

Required kernel options (amdgpu IOMMU page):

- `CONFIG_PCI_P2PDMA`
- `CONFIG_DMABUF_MOVE_NOTIFY`
- `CONFIG_HSA_AMD_P2P`

Zen and later: “peer-to-peer DMA is fully supported.” Pre-Zen: writes only.

### 2.2 Large-BAR is a hard gate

ROCm 7.2.0 BAR-memory page:[^rocm-how-to-bar-memory-html]
https://rocm.docs.amd.com/en/docs-7.2.0/how-to/Bar-Memory.html

> P2P DMA only works when one device can directly access the local BAR memory of another. If the memory address of a BAR memory exceeds the physical addressing limit of a device, the device will not be able to access that BAR.

Official BAR layout (same page):

| BAR | Value | P2P constraint |
|---|---|---|
| BAR0-1 | 64-bit, prefetchable, GPU memory | “8 GB or 16 GB depending on GPU. **Set to less than 2^44** to support P2P access from other GPUs with a 44-bit physical address limit.” [^rocm-how-to-bar-memory-html] |
| BAR2-3 | 64-bit, prefetchable, doorbell | Same **< 2^44** rule. |
| BAR5 | 32-bit, non-prefetchable, MMIO | “Set to less than 4 GB.” |

`2^44` = **17,592,186,044,416** bytes (16 TiB). The “8 GB or 16 GB” wording is Instinct-era table text. On V620 the VRAM is **32 GB**; a working large-BAR maps **the entire 32 GB** (HIP #3103 method: `lspci -v` first prefetchable region **size=32G**, not 256M).[^github-issues-3103]

BIOS knobs the same page names: **Above 4G Decoding = Enabled**, **MMIO High Base / MMIO High Size** so the aperture sits inside the device physical-address limit.

HIP `hipDeviceProp_t.isLargeBar`: **1** if large PCI BAR, else **0**.[^github-hip-hip-runtime-api-h]
https://github.com/ROCm/hip/blob/rocm-7.0.0/include/hip/hip_runtime_api.h#L249

AMD engineer on HIP #3103 (MI100, same rule for any PCIe P2P):[^github-issues-3103]

> You might also want to make sure that you have PCIe Large BAR turned on … `> 4G Decoding` … `lspci -v` … first memory region … **size=32G** … If this is smaller (e.g. **size=256M**) then you do not have large BAR enabled. **In that case, we cannot support PCIe peer-to-peer transfers.**[^github-issues-3103]

https://github.com/ROCm/HIP/issues/3103

On 4× 32 GB V620 you need **four 32 GiB prefetchable BARs** in the MMIO hole (plus doorbells/MMIO). That is **≥ 128 GiB** of 64-bit MMIO before other devices. BIOS: **Above 4G Decoding** on, **Resizable BAR / Re-Size BAR** on if the firmware exposes it, **CSM off**. `pci=realloc` is the usual kernel helper when the BIOS left the BARs small.[^v620-product]

HIP also exposes `hipDeviceAttributeIsLargeBar`.

### 2.3 IOMMU: two official stories, both published

amdgpu IOMMU page (https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/iommu.html):[^instinct-conceptual-iommu-html]

| Mode | Cmdline | Official recommendation |
|---|---|---|
| Enabled (remap) | default | **“Recommended for AMD Radeon GPUs that need peer-to-peer DMA.”** Per-device IOVA. |
| Passthrough | `iommu=pt` | **“Recommended for AMD Instinct GPUs and for AMD Radeon GPUs that don’t need peer-to-peer DMA.”** Interrupt remap on, I/O remap off. Shared platform address space. |
| Off | `iommu=off` | Not recommended. Only for old kernels that cannot do P2P with an IOMMU. |

ROCm 7.2.0 **install FAQ Issue #5** (https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/install-faq.html):[^rocm-reference-install-faq-html]

> NCCL WARN Missing `"iommu=pt"` from kernel command line which can lead to system instability or hang!

**Solution they print:** add `iommu=pt` to GRUB. Points at RCCL #1129.

System-requirements page repeats: “Systems with multiple GPUs may require `iommu=pt` … Issue #5.”

ROCm BAR-memory page (different purpose): enabling IOMMU in **non-passthrough** mode creates a virtual I/O address space and keeps those addresses inside the device physical-address limit — the BAR-placement workaround, not the RCCL hang workaround.

So: **remap is what AMD says Radeon P2P wants; `iommu=pt` is what RCCL/ROCm install says multi-GPU needs to not hang.** On Instinct this is consistent (XGMI, no PCIe P2P). On 4× V620 it is a real fork. Measure both; do not assume the Instinct default is the P2P default.

### 2.4 ACS

PCIe ACS (Access Control Services) on a switch/root port can **redirect** peer TLPs up to the root complex instead of switching them locally. Linux `pci_p2pdma` then sees a host-bridge hop; if that bridge is not whitelisted, P2P is `NOT_SUPPORTED`.

NCCL GPU troubleshooting (RCCL inherits the same PCIe mechanics and the same `NCCL_P2P_*` knobs):[^docs-troubleshooting-gpu-troubleshooting-html]
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html

> IO virtualization (also known as VT-d or IOMMU) can interfere with GPU Direct by redirecting all PCI point-to-point traffic to the CPU root complex, causing a significant performance reduction or even a hang.

Check:

```
sudo lspci -vvv | grep ACSCtl
```

`SrcValid+` means ACS source validation is on. Clear on switch ports (must be re-applied after reboot on many PLX/Broadcom switches):

```
sudo setpci -s <BDF> ECAP_ACS+0x6.w=0000
```

`pcie_acs_override=downstream,multifunction` is a **kernel patch**, not a PCI-SIG feature. It changes IOMMU grouping / software policy; it does **not** clear hardware ACS bits.

Two AMD “ACS Enable” contexts that look contradictory and are not:

- ROCm 7.2.0 V620 **SR-IOV** BIOS table sets **ACS Enable = Enabled** (VF isolation). That page is virtualization, not bare-metal TP=4.[^rocm-system-optimization-w6000-v620-html]
  https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/w6000-v620.html
- Bare-metal P2P wants ACS **off** on switch downstream ports so peer TLPs stay in the switch.

NCCL: “Virtual machines require ACS to function, hence disabling ACS is not an option.” Do not expect full P2P bandwidth inside a VM.

Level1Techs forum thread on 2× R9700 (gfx1201) behind a PEX88096 switch (secondary, user-reported, **not V620**): RCCL all-reduce **5.41 → 19.60 GB/s** at 128M after clearing ACS on the switch downstream ports (3.6×).[^l1t-249623] Treat as “ACS can cost a factor of three,” not as a V620 number.

### 2.5 HIP APIs

https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/hip_runtime_api/modules/peer_to_peer_device_memory_access.html[^rocm-modules-peer-to-peer-device-memory-access-h]

| API | Contract |
|---|---|
| `hipDeviceCanAccessPeer(&flag, device, peer)` | **1** if `device` can DMA `peer` VRAM; **0** if not; **0** if `device==peer`. `hipErrorInvalidDevice` if either id is invalid. |
| `hipDeviceEnablePeerAccess(peer, 0)` | Map peer allocations into this device’s VA. |
| `hipDeviceDisablePeerAccess(peer)` | Undo that mapping. |
| `hipMemcpyPeer(dst, dstDev, src, srcDev, n)` | Copy between peer-accessible devices. |
| `hipMemcpyPeerAsync(...)` | Same, async. |

HIP 5.4.4 doxygen still said **“PeerToPeer support is experimental.”** 7.2 no longer prints that sentence on the same page. The hardware/platform caveats did not go away.

HIP multi-device how-to (https://github.com/ROCm/hip/blob/96b5aa0a/docs/how-to/hip_runtime_api/multi_device.rst):[^github-hip-runtime-api-multi-device-rst]

> If peer-to-peer access is not activated, the call to `hipMemcpy` still works but internally uses a **staging buffer in host memory**, which incurs a performance penalty.

That is the host-bounce path. RCCL’s SHM transport is the collective version of the same idea.

HIP tests now **skip** when `hipDeviceCanAccessPeer` is 0 (`SWDEV-209747`, `SWDEV-527806`). P2P is not assumed.

A **0** from `hipDeviceCanAccessPeer` is decisive for HIP memcpy-peer. It is **not** decisive for every RCCL path: some Radeon stacks fall back to HSA dmabuf import and still attempt a “P2P” transport that is not HIP peer access (see §5.6). Always read the RCCL log.

### 2.6 `HSA_ENABLE_SDMA` / `HSA_ENABLE_PEER_SDMA`

ROCR env (https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html):[^rocm-api-reference-environment-variables-html]

| Var | Default | Effect |
|---|---|---|
| `HSA_ENABLE_SDMA` | **1** | DMA engines for H2D / D2H / D2D on `hsa_memory_copy`, `hsa_amd_memory_async_copy`, `hsa_amd_memory_async_copy_on_engine`, `hsa_amd_memory_fill`. **0** = blit/compute copy. |
| `HSA_ENABLE_PEER_SDMA` | **1** | DMA engines for **D2D only**. Ignored if `HSA_ENABLE_SDMA=0`. |

ROCm #2616 (OpenMP missed sync, listed **Radeon Pro W6800** among GPUs): workaround `HSA_ENABLE_SDMA=0`, “Performance impact may be observed.” Closed as fixed in 6.3. On 7.2 leave SDMA **on** unless you are chasing a copy-sync bug.[^github-issues-2616]

`HSA_ENABLE_SDMA=0` does **not** create P2P. It only changes who copies (SDMA vs shader).

### 2.7 RCCL’s extra PCIe switch: `HSA_FORCE_FINE_GRAIN_PCIE=1`

ROCm 7.2.0 RCCL usage tips:

> To enable peer-to-peer access on machines with **PCIe-connected GPUs**, set `HSA_FORCE_FINE_GRAIN_PCIE=1`. This feature requires GPUs that support peer-to-peer access along with **proper large BAR addressing support**.[^rocm-how-to-rccl-usage-tips-html]

https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.0/how-to/rccl-usage-tips.html

This is the RCCL-specific enable, not a substitute for large-BAR or `hipDeviceCanAccessPeer`.

MSCCL on the same page: **enabled by default on MI300X**; other platforms need `RCCL_MSCCL_FORCE_ENABLE=1`. MSCCL++ is `RCCL_MSCCLPP_ENABLE=1`, default threshold **1 MB** (`RCCL_MSCCLPP_THRESHOLD`). **Do not expect MSCCL++ on gfx1030.**

### 2.8 Navi 21 vs Instinct, in one paragraph

| | Instinct (MI210/MI300/…) | Navi 21 V620 |
|---|---|---|
| GPU–GPU link | XGMI / Infinity Fabric (hundreds of GB/s class; MI300X CPX allreduce **~340 GB/s** busbw in AMD’s own RCCL plot) | **None** |
| PCIe P2P | Used for NIC / cross-hive, not the primary intra-node path | **The only intra-node path** |
| `use_xgmi_p2p` | Meaningful | No-op |
| `HSA_FORCE_FINE_GRAIN_PCIE` | Usually irrelevant intra-node | **Required for RCCL P2P** |
| IOMMU default AMD wants | `iommu=pt` | Remap if you want P2P; `pt` if you want the RCCL hang workaround |
| Official multi-GPU story | First-class | Supported GPU, **not** a first-class fabric |

Navi 21 **can** do PCIe P2P in the driver model (large-BAR + `HSA_AMD_P2P` + compatible chipset). It is not the Instinct path and it is not guaranteed on a random 4-slot board.

---

## 3. RCCL and what vLLM actually calls

### 3.1 Algorithms

RCCL “What is RCCL?”:

> The collective operations are implemented using **ring and tree** algorithms … optimized … topology awareness, high-speed interconnects, and RDMA-based collectives.
> RCCL uses **PCIe and xGMI** … InfiniBand, RoCE, and TCP/IP …[^rocm-latest-what-is-rccl-html]

https://rocm.docs.amd.com/projects/rccl/en/latest/what-is-rccl.html

ROCm 7.2.0 ships **RCCL 2.27.7** (release notes component table; “Compatibility with NCCL 2.27.7”). That tree still documents **MSCCL / MSCCL++**. Default MSCCL is **MI300X**. `NCCL_ALGO` / `NCCL_PROTO` force algorithm/protocol. Unset unless you are measuring.

Small messages → Tree (latency). Large → Ring (bandwidth). That is the NCCL/RCCL tuner, not a V620 special case.

### 3.2 NCCL-compat env (the ones that matter here)

> **Boundary (2026-10-11):** RCCL/NCCL env knobs are version-specific and owned by [toolchain/comms/README.md](../../toolchain/comms/README.md) (ROCM_specialist). This section is kept until toolchain/ absorbs it; then it shrinks to a one-line summary plus that link. Silicon keeps the PCIe/P2P topology and all-reduce mechanics.

RCCL accepts the NCCL names. Canonical text:[^docs-docs-env-html]
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

| Variable | Values | What it does on this box |
|---|---|---|
| `NCCL_P2P_DISABLE` | unset / `1` | `1` disables the P2P transport (direct access over NVLink **or PCI**). RCCL then uses SHM/host. Reported workaround when P2P init crashes in `hipIpcGetMemHandle` (rccl-tests #162)[^rccl-tests-162-disable] **or** when P2P is pathologically slow (legacy-rocm-build #6576, §5.6)[^github-issues-6576]. |
| `NCCL_P2P_LEVEL` | `LOC`/`0` never; `NVL` NVLink/XGMI-class; `PIX`/`1` same PCI switch; `PXB`/`2` via PCI switches; `PHB`/`3` same NUMA (through CPU); `SYS`/`4` cross-NUMA | Cutoff for the **flat P2P schedule**. On V620 there is no `NVL`. Typical 4-GPU workstation is `PHB` or `SYS`. `PIX` only if a real switch sits between cards. |
| `NCCL_SHM_DISABLE` | unset / `1` | `1` forbids the host-SHM fallback. Only useful as a forced-P2P experiment; can hang if P2P is broken. |
| `NCCL_IB_DISABLE` | `0` use IB/RoCE verbs; `1` force TCP sockets | Single-node TP=4: IB is unused. `1` is fine and avoids a useless verbs probe. |
| `NCCL_MIN_NCHANNELS` | integer | Floor on channels (CUDA/HIP blocks). AMD documents it for **MI300X with 2 or 4 GPUs** (RCCL uses **32** channels on 2× MI300X, **24** on 4×; example `export NCCL_MIN_NCHANNELS=32`). Not a V620 default. Raising it uses more CUs for comm. |
| `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS` | `INFO` / `INIT,P2P,GRAPH,TUNING` | You want to see P2P channels, not only SHM/NET. |

RCCL 2.27.7 / ROCm 7.2.0 is the older world vs later “symmetric memory” notes: vLLM on gfx1030 is not on that path anyway (`use_custom_allreduce` false, no MI300 symmetric stack).

### 3.3 vLLM dispatch on ROCm (main, 2026-08)

`RocmPlatform` (`vllm/platforms/rocm.py`, opened raw):[^extras-rocm-py]

```python
dist_backend: str = "nccl"

def use_custom_allreduce(cls) -> bool:
    # We only enable custom allreduce for MI300 series
    return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])
```

`is_navi()` is `"gfx1" in _GCN_ARCH`, so **gfx1030 counts as Navi**. That flag is for attention/kernel selection, not for custom AR.

`CudaCommunicator.all_reduce` order (ROCm): Quick Reduce → (FlashInfer, CUDA) → AITER/custom AR → `pynccl`. On gfx1030 the first three never construct. **Every TP all-reduce is `PyNcclCommunicator` → RCCL.**

Quick Reduce (`quick_all_reduce.py`):

- Arch gate: gfx94 / gfx95 only
- “Custom quick allreduce is only supported on ROCm MI300 series.”
- `world_size > 2 and not self.fully_connected` → disabled (“not supported on more than two PCIe-only GPUs”)
- Env: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` default **`NONE`**

Custom AR (`custom_all_reduce.py`) even on CUDA:

> for 4 or more non NVLink-capable GPUs, custom allreduce provides little performance improvement over NCCL.

And it only runs for `world_size == 2` or `fully_connected`. **TP=4 PCIe would not use it even if someone compiled the HIP.**

`is_fully_connected`: `amdsmi_topo_get_link_type`, require `hops == 1` and `type == 2` (XGMI). Four V620s fail that check.

---

## 4. Realistic TP=4 all-reduce cost

### 4.1 Bus

PCI-SIG: PCIe 4.0 added **16.0 GT/s per lane per direction** of raw bandwidth.[^pcisig-what-bit-rates-does-pcie-50-specification]
https://pcisig.com/what-bit-rates-does-pcie-50-specification-support-and-how-does-it-compare-prior-pcie-generations[^pcisig-field-category-value-5b-5d-pci-express-4]
https://pcisig.com/faq?field_category_value%5B%5D=pci_express_4.0

PCIe 3.0+ line code is **128b/130b**. Payload one way on x16:

```
16e9 × 16 × (128/130) / 8 = 31.508 GB/s
```

Raw (no encoding, the way PCI-SIG’s 5.0 FAQ quotes 128 GB/s on x32 @ 32 GT/s): `16 × 16 / 8 = 32.0 GB/s`. Use **31.508 GB/s** as the payload ceiling.[^pcisig-pcie4]

That is **per link, one direction**. A ring all-reduce is not a single memcpy of S bytes.

`lspci -vv` **LnkSta** is the number that matters, not **LnkCap**. Four dual-slot 300 W cards in a board that only wires x8/x8/x8/x8 are already at **15.754 GB/s** payload before software.[^pcisig-what-bit-rates-does-pcie-50-specification]

### 4.2 Ring math (n = 4)

NCCL-tests PERFORMANCE.md (https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md):[^github-doc-performance-md]

An all-reduce of S bytes on n ranks, point-to-point algorithms, needs **2(n−1)** transfers of S/n per rank:

```
t ≥ (S / B) × 2(n−1)/n
algbw = S / t
busbw = algbw × 2(n−1)/n
```

For **n = 4**: factor **1.5**.

| If the bottleneck B is… | Ceiling algbw (S/t) | Ceiling busbw |
|---|---|---|
| 31.508 GB/s (PCIe 4.0 x16, one way, 100%) | **21.005 GB/s** | 31.508 GB/s [^pcisig-pcie4] |
| 15.754 GB/s (x8, or x16 at 50%) | 10.503 GB/s | 15.754 GB/s |
| Host-staged (see §4.4) | **unknown** on this box; expect **≪ P2P**, often several× | — |

Tree is **not** bandwidth-optimal; NCCL says use `algbw × 2` if you want a link estimate. Tree wins **small** S (decode). Ring wins **large** S (prefill).

These are **ceilings**. Protocol, ACS, NUMA, and RCCL channel count eat the rest. A measured 70% of 31.508 is **22.1 GB/s busbw** → **14.7 GB/s algbw**. That is an example, not a V620 bench. **unknown: measured busbw on 4× V620.** AMD has not published an rccl-tests table for this SKU.[^pcisig-pcie4]

### 4.3 Bytes on the wire — residual is still FP16

TP all-reduce in vLLM is the Megatron residual: after **row-parallel `o_proj`** and after **row-parallel `down_proj`**. Two all-reduces per layer, each of shape `[num_tokens, hidden]`.

Element size for `--dtype float16`: **2 bytes**. gfx1030 has **no hardware bf16** (vLLM #38107; hardware bf16 is RDNA 3). Use FP16.[^github-issues-38107]

**W4A16 / AWQ / GPTQ does not change this.** Weights are 4-bit locally. The tensor that is reduced is the activation / residual, still FP16. Quantizing the collective (Quick Reduce INT8/INT4) is an MI300 path and is off.

```
S = num_tokens × hidden × 2
```

Hidden sizes:

| Model | hidden | n_layers | Source |
|---|---|---|---|
| Llama 2 7B | **4096** | 32 | Llama 2 paper (arXiv:2307.09288) table; recap in arXiv:2312.04333 [^arxiv-abs-2307-09288] |
| Llama 2 70B | **8192** | **80** | Same |
| Llama 3 / 3.1 8B | **4096** | 32 | Megatron-Bridge `Llama3ModelProvider8B` |
| Llama 3 / 3.1 / 3.3 70B | **8192** | 80 | Same, `Llama3ModelProvider70B` |
| Qwen2-72B | **8192** | — | Qwen2 paper table (OpenReview) |

Worked S (FP16):

| Workload | tokens | hidden | S per AR | 2 AR / layer |
|---|---|---|---|---|
| Decode, bs=1 | 1 | 4096 | **8 KiB** | 16 KiB |
| Decode, bs=1 | 1 | 8192 | **16 KiB** | 32 KiB |
| Decode, bs=32 | 32 | 8192 | **512 KiB** | 1 MiB |
| Prefill | 2048 | 4096 | **16 MiB** | 32 MiB |
| Prefill | 2048 | 8192 | **32 MiB** | 64 MiB |
| Prefill | 8192 | 8192 | **128 MiB** | 256 MiB |

Time if the bus really is 31.508 GB/s and Ring n=4 (`t = 1.5 S / B`):[^pcisig-pcie4]

| S | t (ceiling) |
|---|---|
| 16 KiB | **0.73 µs** (you will never see this; launch + latency dominate) |
| 512 KiB | **23 µs** (still latency-ish) |
| 32 MiB | **1.46 ms** |
| 128 MiB | **5.83 ms** |

Decode TP=4 on PCIe is a **latency** problem (Tree, few channels). Prefill is a **bandwidth** problem (Ring, P2P vs host).

70B, 80 layers, 2 AR/layer, decode bs=1: 160 × 16 KiB = **2.5 MiB** of AR payload per token. At 1.5× on a 31.5 GB/s bus the bandwidth term is ~0.12 ms/token — **not** the story. The story is **per-call overhead × 160**. Host bounce makes that overhead much worse. A broken P2P path at **~215 ms per collective** (ROCm #6576, §5.6) would be **160 × 215 ms ≈ 34 s/token**. That is not theoretical.[^github-issues-6576]

Prefill 2048 tokens, 70B: 160 × 32 MiB = **5.12 GiB** of AR payload. At 21 GB/s algbw ≈ **244 ms** of pure AR if fully exposed (it is not; compute overlaps some). Host bounce can turn that into a large fraction of prefill time.

### 4.4 When P2P works vs host staging

**P2P path (what you want):**

1. `lspci -v` BAR0 = **32G** on every card, BAR start **< 2^44**.
2. `hipDeviceCanAccessPeer(i,j) == 1` for all i≠j (or at least the pairs RCCL will use).
3. `HSA_FORCE_FINE_GRAIN_PCIE=1`.
4. `amdgpu.pcie_p2p=1` (default), `CONFIG_HSA_AMD_P2P=y`.
5. Chipset actually routes peer TLPs (same switch **or** a root complex that does P2P; ACS not redirecting).
6. RCCL log shows **P2P** transport, not only SHM — **and** `all_reduce_perf` times are microseconds-to-milliseconds, not hundreds of milliseconds.

Then Ring/Tree issue `hipMemcpy`/`SDMA` against the **peer BAR**. Ceiling = §4.2.

**Host bounce (what you get if any of the above fails):**

HIP: `hipMemcpy` uses a **host staging buffer** (multi-device.rst). RCCL: **SHM** transport — GPU0 → host → GPU1, twice per hop in a ring.

Cost model (order of magnitude, not a bench):

- Each hop is **two** PCIe copies (D2H + H2D) instead of one P2P.
- Host memory and the root complex become the shared bottleneck for **all four** GPUs.
- Small S: extra latency (map, CPU involvement, extra interrupt).
- Large S: algbw often lands in the **few GB/s** class, not 20. **unknown: measured SHM algbw on this box.**

`NCCL_P2P_DISABLE=1` **forces** this path. Use it to confirm a hang is P2P, or as a stability workaround. On some Radeon boxes it is **faster** than the “P2P” transport RCCL selected (§5.6). That is the check, not a slogan.

### 4.5 Topology classes for 4 GPUs (NCCL names)

| Layout | `NCCL_P2P_LEVEL` needed | What the bytes do |
|---|---|---|
| All four under one PLX/PEX switch, ACS clear | `PIX` | Best PCIe P2P. Switch bisection must hold 2–4 concurrent x16 streams; a cheap switch will not. |
| Pairs on two switches | `PXB` | Extra hop. Still P2P if ACS/IOMMU allow. |
| Four CPU root ports, same NUMA | `PHB` | “Through the CPU.” Works on Zen+ if the RC routes peer TLPs. Often slower than a switch. This is the topology that produced the 7000× P2P stall on dual W7800 (§5.6). |
| Split across NUMA / sockets | `SYS` | Crosses UPI/Infinity Fabric **CPU** links. Worst. Pin ranks to local GPUs. |

`rocm-smi --showtopo` / `amd-smi topology` / `NCCL_DEBUG_SUBSYS=GRAPH` tell you which of those you have. **unknown: this user’s board topology** (not in the prompt).

AMD’s own V620 SR-IOV test host was a **Supermicro AS-4124GS-TNR**, EPYC 7552, Ubuntu 20.04.3 — a dual-socket 4U with many CPU root ports, not a single PLX switch. That is a `PHB`/`SYS` class machine if you put four cards in it. Do not cargo-cult that BIOS table (ACS Enable, SR-IOV) onto a bare-metal TP=4 box.

---

## 5. Known Radeon P2P failures (sourced)

### 5.1 Large-BAR / 256 MB window

- HIP #3103: no large-BAR → **no PCIe P2P**. `lspci` **256M** vs **32G**.[^github-issues-3103]
- ROCm BAR-memory: BAR above **2^44** → peer cannot DMA it. Historical ROCm #787: BAR at `0x380000000000` (> 2^44) → peers do not show up even with a shared root complex.[^github-issues-787]
- Consumer default BAR is historically **256 MB**. PRO/data-center VBIOS is more likely to request a full 32 GB BAR, but the **platform** still has to assign it (4×32 GB).
- `amdgpu` `debug_mask=0x2` fakes large-BAR by reporting **256 MB** VRAM — useless for production.

### 5.2 `iommu=pt` vs remap

- AMD IOMMU page: **remap for Radeon that need P2P**; **`pt` for Instinct and Radeon that do not**.
- ROCm 7.2 FAQ + RCCL #1129: missing `iommu=pt` → **hang**, 100% GPU, no temp rise.
- You cannot treat Instinct GRUB as gospel on V620. If P2P is 0 with `pt`, try remap; if RCCL hangs with remap, you may be stuck with `pt` + host bounce.

### 5.3 ACS / switch vs CPU root

- ACS redirect → `pci_p2pdma` host-bridge path → not whitelisted → P2P denied, or P2P “works” at a fraction of link rate (NCCL GPU troubleshooting; Intel compute-runtime #935, same Linux core).[^github-issues-935]
- `pcie_acs_override` is a **downstream kernel patch**. Distro kernels often strip it.
- 4-GPU **without** a switch: four root ports. P2P is then a **CPU RC** feature. Consumer RCs are the ones that fail or silently corrupt. Server/EPYC + whitelist is the hopeful case. **unknown: whether this user’s RC is in `pci_p2pdma_whitelist`.**

### 5.4 Consumer vs PRO firmware

- ROCm #1714 (AMD): official gfx1030 Linux QA is **V620 and W6800**. RX 6800/6900 XT “run on a stack that has undergone full QA of the **ISA**” but **no official support**. Firmware, SR-IOV, VBIOS BAR, and display IP differ.[^github-issues-1714]
- V620: **no display connectors**, SR-IOV / MxGPU story, 32 GB. That is the PRO/data-center VBIOS, not Adrenalin consumer.[^v620-product]
- Consumer + `HSA_OVERRIDE_GFX_VERSION` is a foot-gun (ROCm #5238). Do not set it on V620.

### 5.5 PCIe atomics

- ROCm 7.2.0 system requirements: **PCIe atomics required**.
- Proxmox/QEMU: root ports that do not advertise atomics break ROCm multi-GPU. Bare metal: `dmesg` + `lspci` capability (`AtomicOps` / `AtomicOpCompleter`). If atomics are missing, fix the platform; do not write a kernel around it.

### 5.6 HIP / RCCL software bugs (the ones that kill TP)

> **Boundary (2026-10-11):** HIP/RCCL release bugs and their workarounds are version-specific and owned by [toolchain/comms/README.md](../../toolchain/comms/README.md) (ROCM_specialist). This section is kept until toolchain/ absorbs it; then it shrinks to a one-line summary plus that link. Silicon keeps the PCIe/P2P topology and all-reduce mechanics.

These are **Radeon / PCIe** failures, not Instinct XGMI failures. None of them is a published V620 rccl-tests table. They are the failure modes that show up when RCCL tries P2P on discrete Radeon.

**A. `hipIpcGetMemHandle failed : invalid argument` (RCCL `src/transport/p2p.cc`)**

- ROCm/rccl-tests #162: **4× Radeon AI PRO R9700** (gfx1201) `all_reduce_perf` dies at `p2p.cc:256` with `hipIpcGetMemHandle failed : invalid argument`.[^github-issues-162]
- ROCm/rccl #1454: same WARN across IOMMU / `CONFIG_HSA_AMD_P2P` / dmabuf matrices; “RCCL is completely broken in most configurations.”[^github-issues-1454]
- Workarounds, by thread: in rccl-tests #162 an AMD engineer asked the reporter to try `NCCL_P2P_DISABLE=1`, and the reporter confirmed `all_reduce_perf` then runs (forcing the SHM path).[^rccl-tests-162-disable] In rccl #1454 the workarounds discussed are `HSA_ENABLE_IPC_MODE_LEGACY=0` (reported ineffective on ROCm 6.2.x) and a patch gist; `NCCL_P2P_DISABLE` is **not** proposed there.[^rccl-1454-ipc-legacy] The #1454 reporter's log pairs an RX 6800 XT (Navi 21) with an MI100.[^github-issues-1454] If TP=4 crashes in P2P setup, trying `NCCL_P2P_DISABLE=1` first is a reasonable check *(our advice, not a vendor recommendation)*.

**B. P2P selected, 7000× slower than SHM (ROCm 7.2.3, RCCL 2.27.7)**[^github-issues-6576]

https://github.com/ROCm/legacy-rocm-build/issues/6576

- Dual **W7800 48 GB (gfx1100)**, separate CPU root ports (`03:00.0` / `07:00.0`), **Gen4 x8/x8**, large BAR, Ubuntu 24.04, kernel 6.8, **RCCL 2.27.7 (ROCm 7.2.3 host)** and 2.28.3.[^github-issues-6576]
- `rocm-smi --showtopo`: Weight **40**, Hops **2**, Link **PCIE**. RCCL still selects **direct P2P/IPC**.
- Healthy SHM (`NCCL_P2P_DISABLE=1`): small collectives **30–41 µs**, 33 MB **26 GB/s**.
- Broken P2P: **fixed ~215 ms per collective**. vLLM TP=2 probe **42.43 s → 0.029 s** (1477×) after disable. 12-layer model ~5.3 s/step (24 × 221 ms); 64-layer ~27 s (128 × 211 ms).
- Single-process two-GPU rccl-tests with P2P on that 7.2.3 host: **illegal memory access + amdgpu UTCL2 page fault** on the second GPU.
- Reproduced with IOMMU fully off. Not an IOMMU-translation bug.

This is the failure that makes “P2P is on” worse than “P2P is off.” On 4× V620, **measure both**. If RCCL P2P is hundreds of milliseconds, force SHM.

**C. HIP API surface is uneven**

- HIP #3352: `hipDeviceCanAccessPeer` returns 0; `hipMemcpy2D` D2D SIGSEGV; `hipMemcpyDtoD` ok.[^github-issues-3352]
- ROCR-Runtime #355 (2× R9700, gfx1201, mainline 6.19, ROCm 7.2.3): `hipDeviceCanAccessPeer` returned 1 and `hipMemcpyPeer` worked only after five fixes (kernel `pci_p2pdma` whitelist patch, BAR resize, DMA mask, `NCCL_PROTO=Simple`). `hipIpcGetMemHandle` then fails in legacy mode, and with `HSA_ENABLE_IPC_MODE_LEGACY=0` `hipIpcOpenMemHandle` returns error 17, which blocks the RCCL P2P transport; forcing P2P with `NCCL_SHM_DISABLE=1` hung the node.[^rocr-355]
- *(unsourced)* A gfx906 P2P sample failure that reportedly does not affect gfx1030. The earlier citation, HIP #1902, is a PR adding half `abs` APIs and does not support this; no replacement source was found. Do not generalize a Vega P2P bug to Navi 21.

**D. MSCCL + HIP-graph capture OOM (vLLM)**

- vLLM #24160 / #18805: RCCL all-reduce OOM during CUDA/HIP graph capture. Workaround `RCCL_MSCCL_ENABLE=0`. Default MSCCL is MI300X, so gfx1030 should not hit this unless someone forced MSCCL on.[^github-issues-24160][^github-issues-18805]

### 5.7 What to run before writing any kernel

```
lspci -d 1002: -vv | rg -n "VGA|Display|Memory at|LnkSta|LnkCap|Atomics|ACS"
cat /proc/cmdline          # iommu=, amdgpu.pcie_p2p, pci=realloc
# isLargeBar via HIP sample or hipDeviceGetAttribute
# hipDeviceCanAccessPeer matrix 4×4
# hipMemcpyPeer 64 MB pairwise GB/s
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,GRAPH \
  HSA_FORCE_FINE_GRAIN_PCIE=1 \
  ./all_reduce_perf -b 8 -e 128M -f 2 -g 4
# then the same with NCCL_P2P_DISABLE=1
```

Pass criteria: BAR **32G** and **< 2^44**, peer matrix **all 1s**, `hipMemcpyPeer` in the **tens of GB/s** (not 1–3), RCCL log **P2P**, `all_reduce_perf` busbw not an order of magnitude below 31.5 GB/s on large S, **and** small-S times in microseconds (not ~215 ms). If disable-P2P is faster, ship with disable-P2P.[^pcisig-pcie4]

---

## 6. What a custom kernel author should assume

### 6.1 Assume this, until the §5.7 checklist passes

1. **No vLLM custom AR. No Quick Reduce. No AITER AR.** gfx1030 is not in those gates (`rocm.py`, `quick_all_reduce.py`).
2. **All TP sync is RCCL** (`dist_backend=nccl`).
3. **P2P is optional and platform-defined.** If `hipDeviceCanAccessPeer` is 0, RCCL **will** host-bounce — or worse, select a broken P2P/IPC path (§5.6 B). Your kernel cannot fix that.
4. **A gfx1030 custom AR that maps peer VRAM is the same P2P problem.** IPC handles, `hipIpcOpenMemHandle`, or raw BAR pointers all need the same large-BAR + ACS + IOMMU + RC path. If RCCL P2P is down, your AR is down.
5. **Even if P2P is up, vLLM’s own rule is: custom AR is for 2-GPU or fully-connected (XGMI/NVLink).** TP=4 PCIe is the case they explicitly skip on CUDA because it “provides little performance improvement over NCCL.” A hand-rolled gfx1030 AR is unlikely to beat a healthy RCCL Ring on 4× PCIe 4.0.
6. **W4A16 does not create a smaller AR.** Write compute kernels (DOT/WMMA-less GEMM, attention). Do not start with a collective.

### 6.2 When a custom AR *would* be worth it

Only after:

- peer matrix is 1s,
- `hipMemcpyPeer` ≈ PCIe,
- RCCL already uses P2P **and** is not the 215 ms path,
- **and** you have a profile showing RCCL launch/latency (not bandwidth) eating decode.

Then the useful work is a **small-S, 2- or 4-rank, IPC + LDS reduce**, not a Ring rewrite. That is still a research project. It is not the first kernel to write on this box.

### 6.3 Practical env for vLLM TP=4 on V620 (checklist, not a promise)

```
# platform (pick one IOMMU story after you measure)
# iommu=pt          # RCCL hang workaround (Instinct default)
# (remap / no pt)   # AMD Radeon-P2P recommendation
# pci=realloc
# amdgpu.pcie_p2p=1   # default; there is no amdgpu.p2p

export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_ENABLE_SDMA=1
export NCCL_IB_DISABLE=1          # single node
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,GRAPH
# NCCL_P2P_DISABLE=1             # if P2P setup crashes OR is ~215 ms/collective
# NCCL_P2P_LEVEL=PHB             # or PIX if you have a switch
# NCCL_MIN_NCHANNELS=...         # only after a tuner log; not an Instinct cargo-cult
# RCCL_MSCCL_ENABLE=0            # only if someone forced MSCCL and graphs OOM
```

vLLM: `--dtype float16` (not bf16), `--tensor-parallel-size 4`. Custom AR / Quick Reduce env vars do nothing useful on gfx1030.

---

## Sources

_Sources (opened this pass)_


**V620 / interconnect**
- https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html
- https://www.activeport.com.au/wp-content/uploads/2023/01/V620-Datasheet_V4_5.pdf
- https://ir.amd.com/news-events/press-releases/detail/1030/amd-radeon-pro-v620-gpu-delivers-powerful-multi-purpose-data-center-visual-performance-for-todays-demanding-cloud-workloads
- https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html
- https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/w6000-v620.html
- https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/system-requirements.html
- https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html (7.14 V620 Ubuntu-only footnote)
- [../engine/rocm-host.md](../../engine/rocm-host.md) — official vs attested (Fedora 43 / RHEL 10 / 7.14.0)
- https://github.com/ROCm/legacy-rocm-build/issues/1714

**P2P / IOMMU / BAR / HIP / ACS**
- https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/iommu.html
- https://rocm.docs.amd.com/en/docs-7.2.0/how-to/Bar-Memory.html
- https://docs.kernel.org/gpu/amdgpu/module-parameters.html
- https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/Kconfig
- https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/hip_runtime_api/modules/peer_to_peer_device_memory_access.html
- https://github.com/ROCm/HIP/issues/3103
- https://github.com/ROCm/hip/blob/96b5aa0a/docs/how-to/hip_runtime_api/multi_device.rst
- https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html
- https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/install-faq.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html
- https://github.com/ROCm/legacy-rocm-build/issues/2616
- https://github.com/intel/compute-runtime/issues/935 (ACS / pci_p2pdma behavior)

**RCCL / NCCL / PCIe rate**
- https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html (RCCL 2.27.7)
- https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.0/how-to/rccl-usage-tips.html
- https://rocm.docs.amd.com/projects/rccl/en/latest/what-is-rccl.html
- https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- https://pcisig.com/faq?field_category_value%5B%5D=pci_express_4.0 (16.0 GT/s)
- https://pcisig.com/what-bit-rates-does-pcie-50-specification-support-and-how-does-it-compare-prior-pcie-generations

**Known Radeon P2P failures**
- https://github.com/ROCm/legacy-rocm-build/issues/6576 (W7800 P2P 215 ms vs SHM 30 µs; RCCL 2.27.7 / ROCm 7.2.3)
- https://github.com/ROCm/rccl-tests/issues/162 (4× R9700 `hipIpcGetMemHandle`)
- https://github.com/ROCm/rccl/issues/1454
- https://github.com/ROCm/ROCm/issues/787 (BAR > 2^44)
- https://github.com/ROCm/HIP/issues/3352
- https://github.com/vllm-project/vllm/issues/24160
- https://github.com/vllm-project/vllm/issues/18805

**vLLM**
- https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py
- https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py
- https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/quick_all_reduce/
- https://github.com/vllm-project/vllm/issues/38107

**Model widths**
- https://arxiv.org/abs/2307.09288 (Llama 2; 70B hidden 8192, 80 layers)
- https://arxiv.org/html/2312.04333v4 (same numbers restated)
- https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/apidocs/bridge/bridge.models.llama.llama_provider.html
- https://openreview.net/pdf?id=ypvmPqTWi8 (Qwen2)

**Unknown (not opened / not measured)**
- Measured `hipMemcpyPeer` GB/s and RCCL busbw on **this** 4× V620.
- This machine’s PCIe tree (switch vs four CPU roots, NUMA, ACS bits, LnkSta width).
- Whether this kernel build has `CONFIG_HSA_AMD_P2P` and `pcie_acs_override`.
- A published gfx1030 RCCL Ring/Tree tuning table (AMD’s published channel numbers are MI300X).

Footnotes:

[^extras-rocm-py]: `vllm/platforms/rocm.py`, opengfx1030/vllm-rdna rdna_extras at 1dc3bac4 (2026-10-11 tip); the page may describe an older commit named inline. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/platforms/rocm.py
[^v620-product]: AMD Radeon PRO V620 product page. https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html (checked 2026-10-11)
[^activeport-01-v620-datasheet-v4-5-pdf]: www.activeport.com.au/wp-content/uploads/2023/01/V620-Datasheet_V4_5.pdf. https://www.activeport.com.au/wp-content/uploads/2023/01/V620-Datasheet_V4_5.pdf (checked 2026-10-11)
[^ir-1030-amd-radeon-pro-v620-gpu-delivers-powerfu]: AMD Radeon™ PRO V620 GPU Delivers Powerful, Multi-Purpose Data Center Visual Performance for Today’s Demanding Cloud Workloads :: Advanced Micro Devices, Inc. (AMD). https://ir.amd.com/news-events/press-releases/detail/1030/amd-radeon-pro-v620-gpu-delivers-powerful-multi-purpose-data-center-visual-performance-for-todays-demanding-cloud-workloads (checked 2026-10-11)
[^rocm-system-optimization-w6000-v620-html]: AMD RDNA2 system optimization — ROCm Documentation. https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/w6000-v620.html (checked 2026-10-11)
[^instinct-userguides-xgmi-configuration-html]: XGMI Configuration — AMD Instinct Virtualization Driver. https://instinct.docs.amd.com/projects/virt-drv/en/latest/userguides/XGMI_configuration.html (checked 2026-10-11)
[^instinct-conceptual-iommu-html]: Input-Output Memory Management Unit (IOMMU) — AMD GPU Driver (amdgpu) 31.60.0. https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/conceptual/iommu.html (checked 2026-10-11)
[^rocm-reference-system-requirements-html]: System requirements (Linux) — ROCm installation (Linux). https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/system-requirements.html (checked 2026-10-11)
[^rocm-reference-system-requirements-html-2]: Redirecting… — ROCm installation (Linux). https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html (checked 2026-10-11)
[^docs-amdgpu-module-parameters-html]: Module Parameters — The Linux Kernel documentation. https://docs.kernel.org/gpu/amdgpu/module-parameters.html (checked 2026-10-11)
[^github-amdkfd-kconfig]: linux/drivers/gpu/drm/amd/amdkfd/Kconfig at master · torvalds/linux. https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/Kconfig (checked 2026-10-11)
[^rocm-how-to-bar-memory-html]: Troubleshoot BAR access limitation — ROCm Documentation. https://rocm.docs.amd.com/en/docs-7.2.0/how-to/Bar-Memory.html (checked 2026-10-11)
[^github-issues-3103]: verification of the simpleP2P example in HIP · Issue #3103 · ROCm/hip. https://github.com/ROCm/HIP/issues/3103 (checked 2026-10-11)
[^github-hip-hip-runtime-api-h]: hip/include/hip/hip_runtime_api.h at rocm-7.0.0 · ROCm/hip. https://github.com/ROCm/hip/blob/rocm-7.0.0/include/hip/hip_runtime_api.h#L249 (checked 2026-10-11)
[^rocm-reference-install-faq-html]: Installation troubleshooting — ROCm installation (Linux). https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/install-faq.html (checked 2026-10-11)
[^docs-troubleshooting-gpu-troubleshooting-html]: GPU troubleshooting — NCCL 2.32.3 documentation. https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html (checked 2026-10-11)
[^rocm-modules-peer-to-peer-device-memory-access-h]: Peer to peer device memory access — HIP 7.2.53210 Documentation. https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/reference/hip_runtime_api/modules/peer_to_peer_device_memory_access.html (checked 2026-10-11)
[^github-hip-runtime-api-multi-device-rst]: hip/docs/how-to/hip_runtime_api/multi_device.rst at 96b5aa0a6d9730f9268fd2f333d1fba7315bbe2f · ROCm/hip. https://github.com/ROCm/hip/blob/96b5aa0a/docs/how-to/hip_runtime_api/multi_device.rst (checked 2026-10-11)
[^rocm-api-reference-environment-variables-html]: Environment variables — ROCR 1.21.0 Documentation. https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/api-reference/environment_variables.html (checked 2026-10-11)
[^github-issues-2616]: Missed synchronization between kernel completion and subsequent dependent data transfer results in an error · Issue #2616 · ROCm/legacy-rocm-build. https://github.com/ROCm/legacy-rocm-build/issues/2616 (checked 2026-10-11)
[^rocm-how-to-rccl-usage-tips-html]: RCCL usage tips — RCCL 2.27.7 Documentation. https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.0/how-to/rccl-usage-tips.html (checked 2026-10-11)
[^rocm-latest-what-is-rccl-html]: What is RCCL? — RCCL 2.30.7 Documentation. https://rocm.docs.amd.com/projects/rccl/en/latest/what-is-rccl.html (checked 2026-10-11)
[^docs-docs-env-html]: Environment Variables — NCCL 2.32.3 documentation. https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html (checked 2026-10-11)
[^pcisig-what-bit-rates-does-pcie-50-specification]: What bit rates does the PCIe 5.0 specification support and how does it compare to prior PCIe generations? | PCI-SIG. https://pcisig.com/what-bit-rates-does-pcie-50-specification-support-and-how-does-it-compare-prior-pcie-generations (checked 2026-10-11)
[^pcisig-field-category-value-5b-5d-pci-express-4]: FAQ | PCI-SIG. https://pcisig.com/faq?field_category_value%5B%5D=pci_express_4.0 (checked 2026-10-11)
[^github-doc-performance-md]: nccl-tests/doc/PERFORMANCE.md at master · NVIDIA/nccl-tests. https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md (checked 2026-10-11)
[^github-issues-38107]: (Bug): Abnormally bad performance on AMD ROCM gfx1030 (W6800, V620, 6900XT 6800XT) · Issue #38107 · vllm-project/vllm. https://github.com/vllm-project/vllm/issues/38107 (checked 2026-10-11)
[^github-issues-6576]: (Issue): RCCL P2P/IPC transport ~7000x slower than SHM on dual gfx1100 (separate CPU root ports): fixed ~215 ms per collective, NCCL_P2P_DISABLE=1 restores full speed · Issue #6576 · ROCm/legacy-rocm-build. https://github.com/ROCm/legacy-rocm-build/issues/6576 (checked 2026-10-11)
[^github-issues-787]: P2P support · Issue #787 · ROCm/legacy-rocm-build. https://github.com/ROCm/ROCm/issues/787 (checked 2026-10-11)
[^github-issues-935]: (GSD-12874) (BMG / Arc Pro B70) Is cross-root-port P2P expected? L0 dev-to-dev copy returns 0x70000003; kernel refuses at P2PDMA (host bridge not whitelisted) — and can the capability be made queryable? · Issue #935 · intel/compute-runtime. https://github.com/intel/compute-runtime/issues/935 (checked 2026-10-11)
[^github-issues-1714]: Which devices are even supported? (HIP/ROCm) · Issue #1714 · ROCm/legacy-rocm-build. https://github.com/ROCm/legacy-rocm-build/issues/1714 (checked 2026-10-11)
[^github-issues-162]: (Issue): Radeon AI PRO R9700 4card failed in rccl-tests · Issue #162 · ROCm/rccl-tests. https://github.com/ROCm/rccl-tests/issues/162 (checked 2026-10-11)
[^github-issues-1454]: (Issue): RCCL is compleatly broken in most configurations · Issue #1454 · ROCm/rccl. https://github.com/ROCm/rccl/issues/1454 (checked 2026-10-11)
[^github-issues-3352]: Multi-GPU, P2P access, Memcpy2D, SIGSEGV · Issue #3352 · ROCm/hip. https://github.com/ROCm/HIP/issues/3352 (checked 2026-10-11)
[^github-issues-24160]: (Bug): (AMD GPU)(OOM)(RCCL Allreduce)(NCCL error: unhandled cuda error) Got RCCL allreduce OOM error when running Llama405B TP8 on AMD GPUs · Issue #24160 · vllm-project/vllm. https://github.com/vllm-project/vllm/issues/24160 (checked 2026-10-11)
[^github-issues-18805]: (Usage): NCCL error when using tow AMD GPUs ( gfx1100 ) · Issue #18805 · vllm-project/vllm. https://github.com/vllm-project/vllm/issues/18805 (checked 2026-10-11)
[^rocm-about-release-notes-html]: ROCm 7.2.0 release notes — ROCm Documentation. https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html (checked 2026-10-11)
[^github-platforms-rocm-py]: vllm/vllm/platforms/rocm.py at main · vllm-project/vllm. https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py (checked 2026-10-11)
[^github-device-communicators-custom-all-reduce-py]: vllm/vllm/distributed/device_communicators/custom_all_reduce.py at main · vllm-project/vllm. https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/custom_all_reduce.py (checked 2026-10-11)
[^docs-device-communicators-quick-all-reduce]: docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/quick_all_reduce/. https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/quick_all_reduce/ (checked 2026-10-11)
[^arxiv-abs-2307-09288]: (2307.09288) Llama 2: Open Foundation and Fine-Tuned Chat Models. https://arxiv.org/abs/2307.09288 (checked 2026-10-11)
[^arxiv-html-2312-04333v4]: Is Bigger and Deeper Always Better? Probing LLaMA Across Scales and Layers. https://arxiv.org/html/2312.04333v4 (checked 2026-10-11)
[^docs-bridge-bridge-models-llama-llama-provider-h]: bridge.models.llama.llama_provider — Megatron Bridge. https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/apidocs/bridge/bridge.models.llama.llama_provider.html (checked 2026-10-11)
[^openreview-id-ypvmpqtwi8]: openreview.net/pdf?id=ypvmPqTWi8. https://openreview.net/pdf?id=ypvmPqTWi8 (returns 403 to scripted fetch on 2026-10-11 (bot block); not re-read)
[^rocm-rocm-x-blog-readme-html]: ROCm 10.0: A Decade of Open Compute, Built for the Age of Agentic AI — ROCm Blogs. https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html (checked 2026-10-11)

## ROCm 10.0 RCCL / GIN (2026-08-27)

> **Boundary (2026-10-11):** RCCL release features (ROCm 10.0 / GIN) are version-specific and owned by [toolchain/comms/README.md](../../toolchain/comms/README.md) (ROCM_specialist). This section is kept until toolchain/ absorbs it; then it shrinks to a one-line summary plus that link. Silicon keeps the PCIe/P2P topology and all-reduce mechanics.

[ROCm 10.0 blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html): RCCL merges NCCL 2.30.4, adds symmetric memory and a **GIN** device API (“lets the GPU kick off network transfers directly over GDA and SDMA paths instead of bouncing through the CPU”). The blog does not mention DeepEP or IBGDA, so no comparison to them is made here. **Not** 88096 PIX BAR `hipMemcpyPeer`. Do not retip [deepep-v620.md](deepep-v620.md). Live host stays **7.14.0**. Do not invent V620 GIN bandwidth.[^rocm-rocm-x-blog-readme-html]
[^kfd-crat]: Linux v6.12 amdkfd kfd_crat.c, sienna_cichlid_cache_info (Navi 21: TCP 16 KB/CU, I$ 32 KB/2 CU, K$ 16 KB/2 CU, GL1 128 KB/10 CU, L2 4096 KB, L3 128 MB). https://github.com/torvalds/linux/blob/v6.12/drivers/gpu/drm/amd/amdkfd/kfd_crat.c#L556 (checked 2026-10-11)
[^pcisig-pcie4]: PCI-SIG FAQ, PCI Express 4.0 (16 GT/s per lane). https://pcisig.com/faq?field_category_value%5B%5D=pci_express_4.0 (checked 2026-10-11)
[^rocm-gpu-arch-specs]: ROCm docs 7.2.4, "GPU architecture specifications" (Radeon PRO V620 row: gfx1030, 32 GiB, 72 CU, wave 32/64, LDS 128 KiB, Infinity Cache 128 MiB, L2 4 MiB, GL1 128 KiB, L0 vector 16 KiB, L0 scalar 16 KiB, L0 instruction 32 KiB, VGPR file 512 KiB). https://rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html (checked 2026-10-11)
[^l1t-249623]: Level1Techs forum, "Does the LRLink PCIe Bridge Fix RCCL's Broken P2P on Dual R9700? (Update: Yes it does!)", opened 2026-05-01; status table "RCCL allreduce 19.60 GB/s at 128M (was 5.41)" after ACS clear. https://forum.level1techs.com/t/does-the-lrlink-pcie-bridge-fix-rccls-broken-p2p-on-dual-r9700-update-yes-it-does/249623 (checked 2026-10-11)
[^rccl-tests-162-disable]: ROCm/rccl-tests #162, AMD reply asking to try `NCCL_P2P_DISABLE=1` and reporter's successful run. https://github.com/ROCm/rccl-tests/issues/162#issuecomment-3648839425 and https://github.com/ROCm/rccl-tests/issues/162#issuecomment-3652639807 (checked 2026-10-11)
[^rccl-1454-ipc-legacy]: ROCm/rccl #1454 comments on `HSA_ENABLE_IPC_MODE_LEGACY=0` and its ineffectiveness on ROCm 6.2.x. https://github.com/ROCm/rccl/issues/1454#issuecomment-2544057419 and https://github.com/ROCm/rccl/issues/1454#issuecomment-2551226542 (checked 2026-10-11)
[^rocr-355]: ROCm/ROCR-Runtime #355, "gfx1201 (RDNA 4): hipIpcOpenMemHandle returns error 17 (dmabuf path, mainline kernel)" (closed). https://github.com/ROCm/ROCR-Runtime/issues/355 (checked 2026-10-11)
