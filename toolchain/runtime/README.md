---
owner: ROCM_specialist
updated: 2026-10-12
sources:
  - https://github.com/ROCm/rocm-systems/issues/10943
  - https://github.com/OpenNMT/CTranslate2/issues/2090
  - https://github.com/ROCm/rocm-systems/issues/9820
  - https://github.com/ROCm/rocm-systems/pull/9821
  - https://github.com/ROCm/rocm-systems/issues/12638
  - https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html
  - https://github.com/ROCm/rocm-systems/issues/12528
  - https://github.com/ROCm/rocm-systems/pull/12670
  - https://github.com/Niko1221/Strata/issues/1705
  - https://github.com/ROCm/rocm-systems/blob/2d8b4d61b89499019698982be71926a4f9c4907e/projects/rocr-runtime/libhsakmt/src/fmm.c
  - https://github.com/ROCm/rocm-systems/issues/13234
---

# HIP runtime

Date: **2026-10-12**. Live dest stays ROCm **7.14.0** ([pin](../pin/README.md)). TheRock systems-pin chronology is [../log/runtime-scans.md](../log/runtime-scans.md). MIOpen's GEMM backend is a library routing fact: [libraries/](../libraries/README.md).

## Allocator paths

```mermaid
flowchart TD
  call["device allocation"]
  malloc["hipMalloc / hipFree"]
  async["hipMallocAsync on the default pool"]
  vmm["hipMemAddressReserve, hipMemCreate, hipMemMap, hipMemSetAccess"]
  call --> malloc
  call --> async
  call --> vmm
  malloc --> mallocok["reported control: 0 of 600 mismatches"]
  async --> asyncbad["reported: buffer reads back as zeros, hipSuccess"]
  async --> thr["hipMemPoolAttrReleaseThreshold = UINT64_MAX: 0 of 600"]
  vmm --> hang["DRM 3.61: hipMemSetAccess waits in DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT"]
  vmm --> vmmok["DRM 3.64: open fix keeps the timeline wait"]
```

### `hipMallocAsync` default pool

[rocm-systems#10943](https://github.com/ROCm/rocm-systems/issues/10943) (open) reports that on ROCm 7.2 / HIP 7.2.53211, memory from `hipMallocAsync` on the device default pool can come back **zeroed while still in use**, with every call returning `hipSuccess`. The device still reports `hipDeviceAttributeMemoryPoolsSupported = 1`. The pure-HIP repro (allocate on stream 1, free on stream 2 after an event) failed 151 of 600 iterations; `hipMalloc` / `hipFree` failed 0 of 600; the same async pattern with `hipMemPoolAttrReleaseThreshold = UINT64_MAX` failed 0 of 600. The report's hypothesis is that the pool releases physical backing at a sync point (default threshold 0) before queued work finishes.[^malloc10943]

The same issue's opening note says gfx1032 and gfx1201 were reported to crash at load through CTranslate2, and that the author had no gfx11 report at filing time. A later comment on that issue adds **gfx1100** (RX 7900 XTX, HIP 7.2.53211): GPU page faults (`Page not present`) and a NULL read in `libamdhip64.so.7.2.53211` at process exit.[^malloc10943]

[CTranslate2#2090](https://github.com/OpenNMT/CTranslate2/issues/2090) (open) is the gfx1030 consumer report: official ROCm wheels, HIP 7.2.53211, faster-whisper silently drops words (the table's default-allocator rows sit between about 5% and 68% similarity to the reference, and the result changes every run). `CT2_CUDA_ALLOCATOR=cub_caching` made three runs byte-identical.[^ct2090]

The reports are on HIP 7.2.53211. This page has no 7.14.0 repro. Treat the bug as open for the 7.14.0 dest until someone runs the two-stream repro there. In HIP code, keep hot scratch on `hipMalloc` (or a cache in front of it) unless `ReleaseThreshold` is raised first. The W4A16 split-K allocator choice is [silicon/attention/fa-occupancy.md](../../silicon/attention/fa-occupancy.md). Whether PyTorch's default HIP caching allocator calls `hipMalloc`, and whether `PYTORCH_HIP_ALLOC_CONF=backend:cudaMallocAsync` selects this pool, was not re-opened this pass (unsourced). `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is the virtual-memory path in the next section, which is a different failure.[^vmm9820]

### Virtual memory hangs below DRM 3.64

[rocm-systems#9820](https://github.com/ROCm/rocm-systems/issues/9820) (open): with ROCm 7.14 userspace on amdgpu DRM 3.61, `hipMemSetAccess` blocks forever in `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`. libhsakmt puts GEM_VA timeline output in the ioctl; DRM 3.64 added that output, and older amdgpu accepts the larger struct and ignores the extra fields, so the wait never completes. PyTorch reaches it through `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. Turning expandable segments off avoids the path.[^vmm9820]

[rocm-systems#9821](https://github.com/ROCm/rocm-systems/pull/9821) (open) gates the timeline syncobj on DRM ≥ 3.64 and uses the synchronous VA op below that. The PR's test notes say a physical DRM 3.61 host reproduced the hang, and DRM 3.64 completed the timeline waits.[^vmm9821] Host kernel versions for the RCCL symptom on the same DRM boundary are [comms/](../comms/README.md).

## Host memory pressure: pinned copies and queue eviction

[rocm-systems#12528](https://github.com/ROCm/rocm-systems/issues/12528) (open) reports that `hipMemcpy` from pageable (mmap'd) host memory stalls for minutes when the host is under reclaim. clr pins each pageable chunk with `hsa_amd_memory_lock`, which creates a KFD userptr buffer object. Under reclaim the MMU notifier invalidates that range and KFD evicts the process's GPU queues. In the reporter's gfx1200 run, queues were evicted for 146.5 s of about 195 s and the load never finished in 180 s. The same load forced through clr's staging path with `GPU_PINNED_MIN_XFER_SIZE=100000` finished in about 36 s with about 2 s of eviction. The reporter had not measured what staging costs without memory pressure.[^pin12528] An AMD engineer (`schung-amd`, 2026-10-01) wrote that they reproduced it on **gfx1100** with TheRock nightlies.[^pin12528c] A 2026-10-12 comment adds a gfx1201 report where pinned host memory (`hipHostMalloc` / `hipHostRegister`) also sees queue eviction during page migration.[^pin12528c]

[rocm-systems#12670](https://github.com/ROCm/rocm-systems/pull/12670) (open, not merged) would skip pinning in `DmaBlitManager::getBuffer()` when both `MemAvailable` is low and memory PSI avg10 is above 5%. Its gfx1100 (RX 7900 XTX, nightly `10.2.0a20261001`) test went from 3 of 3 runs timing out at 180 s to about 15 s.[^pin12670]

Strata [#1705](https://github.com/Niko1221/Strata/issues/1705) (open) is an engine-side gfx1100 report (RX 7900 XTX, Linux, ROCm wheels `7.10.0a20251120`): nine verify-timeout and prompt-watchdog deaths in one day during long-context serving under host reclaim, already with `GPU_PINNED_MIN_XFER_SIZE=1048576` set. Adding `HSA_USERPTR_FOR_PAGED_MEM=0` stopped them in the next run (283 requests over 4 h 54 m, up to 245 625 tokens). The issue ties it to the same mechanism as the gfx1030 verify timeouts in Strata #649, and names Pop!_OS's default `vm.swappiness=180` zram setup as the trigger on that box.[^st1705] In libhsakmt, userptr for paged memory is on unless `HSA_USERPTR_FOR_PAGED_MEM` is set to `0`; with it off, paged memory is backed by SVM unless `HSA_SVM_FOR_PAGED_MEM=0`.[^fmm]

For our HIP code: when an engine streams mmap'd weights or keeps a large page-locked expert set while the host is short on RAM, a stall with an idle GPU and one spinning CPU core is this failure first, not a kernel hang. Try `GPU_PINNED_MIN_XFER_SIZE` (staging) or `HSA_USERPTR_FOR_PAGED_MEM=0`, and measure copy throughput without pressure before making either a default. No 7.14.0 repro on this page; no pin change.

## Graphs

[rocm-systems#12638](https://github.com/ROCm/rocm-systems/issues/12638) (open) reports that on Core SDK **10.0.0**, a retained fork/join graph whose root segment holds only empty nodes can hang on first replay. `GraphExecSegmented::BuildSyncPlan()` allocates a completion signal for that segment and never attaches a producer, so the successor's barrier waits. The issue names released source commit `6b0e43f341195e203754e08f850e437ff2fc09f9` (`therock-10.0`).[^graph12638] The 7.14.0 release notes are dated 2026-07-15, before that 10.0.0 report.[^rn7140] Check this before any 10.x move if a capture uses empty join roots. No 7.14.0 pin change.

[rocm-systems#13234](https://github.com/ROCm/rocm-systems/issues/13234) (open, filed 2026-10-10) reports that `hipGraphLaunch` never returns when one graph segment holds at least `2 * ROC_AQL_QUEUE_SIZE` AQL packets: the host spins at 100% CPU and the GPU sits idle. The issue traces it to `VirtualGPU::ReserveAqlSlots` in clr publishing the whole packet count before writing bodies, with no cap at the ring size. With a 16384-slot ring, 32767 launches completed and 32768 hung; the report notes that a batch just above one ring passes. Reproduced on gfx1151 with ROCm 7.2.1 and 10.0.0.[^graph13234] The code it names is in clr's ROCm device path, not arch-specific code, but this page has no gfx1030 or gfx1100 repro. Keep captured decode graphs well under two rings per segment, or raise `ROC_AQL_QUEUE_SIZE`.

## Sources

[^malloc10943]: rocm-systems#10943, open, including the gfx1100 follow-up comment. https://github.com/ROCm/rocm-systems/issues/10943
[^ct2090]: CTranslate2#2090, open. https://github.com/OpenNMT/CTranslate2/issues/2090
[^vmm9820]: rocm-systems#9820, open. https://github.com/ROCm/rocm-systems/issues/9820
[^vmm9821]: rocm-systems#9821, open. https://github.com/ROCm/rocm-systems/pull/9821
[^graph12638]: rocm-systems#12638, open. https://github.com/ROCm/rocm-systems/issues/12638
[^rn7140]: ROCm Core SDK 7.14.0 release notes, dated 2026-07-15. https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html
[^pin12528]: rocm-systems#12528, open, issue body (gfx1200 measurements, staging workaround). https://github.com/ROCm/rocm-systems/issues/12528
[^pin12528c]: rocm-systems#12528 comments by `schung-amd` (2026-10-01, gfx1100 repro) and `briansp2020` (2026-10-12, pinned host memory on gfx1201). https://github.com/ROCm/rocm-systems/issues/12528
[^pin12670]: rocm-systems#12670, open. https://github.com/ROCm/rocm-systems/pull/12670
[^st1705]: Strata#1705, open. https://github.com/Niko1221/Strata/issues/1705
[^fmm]: libhsakmt `fmm.c` at rocm-systems `2d8b4d61`, `HSA_USERPTR_FOR_PAGED_MEM` and `HSA_SVM_FOR_PAGED_MEM` handling. https://github.com/ROCm/rocm-systems/blob/2d8b4d61b89499019698982be71926a4f9c4907e/projects/rocr-runtime/libhsakmt/src/fmm.c
[^graph13234]: rocm-systems#13234, open. https://github.com/ROCm/rocm-systems/issues/13234
