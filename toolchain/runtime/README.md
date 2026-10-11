---
owner: ROCM_specialist
updated: 2026-10-11
sources:
  - https://github.com/ROCm/rocm-systems/issues/10943
  - https://github.com/OpenNMT/CTranslate2/issues/2090
  - https://github.com/ROCm/rocm-systems/issues/9820
  - https://github.com/ROCm/rocm-systems/pull/9821
  - https://github.com/ROCm/rocm-systems/issues/12638
  - https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html
---

# HIP runtime

Date: **2026-10-11**. Live dest stays ROCm **7.14.0** ([pin](../pin/README.md)). TheRock systems-pin chronology is [../log/runtime-scans.md](../log/runtime-scans.md). MIOpen's GEMM backend is a library routing fact: [libraries/](../libraries/README.md).

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

The reports are on HIP 7.2.53211. This page has no 7.14.0 repro. Treat the bug as open for the 7.14.0 dest until someone runs the two-stream repro there. In HIP code, keep hot scratch on `hipMalloc` (or a cache in front of it) unless `ReleaseThreshold` is raised first. The W4A16 split-K allocator choice is [silicon/fa-occupancy.md](../../silicon/fa-occupancy.md). Whether PyTorch's default HIP caching allocator calls `hipMalloc`, and whether `PYTORCH_HIP_ALLOC_CONF=backend:cudaMallocAsync` selects this pool, was not re-opened this pass (unsourced). `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is the virtual-memory path in the next section, which is a different failure.[^vmm9820]

### Virtual memory hangs below DRM 3.64

[rocm-systems#9820](https://github.com/ROCm/rocm-systems/issues/9820) (open): with ROCm 7.14 userspace on amdgpu DRM 3.61, `hipMemSetAccess` blocks forever in `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`. libhsakmt puts GEM_VA timeline output in the ioctl; DRM 3.64 added that output, and older amdgpu accepts the larger struct and ignores the extra fields, so the wait never completes. PyTorch reaches it through `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. Turning expandable segments off avoids the path.[^vmm9820]

[rocm-systems#9821](https://github.com/ROCm/rocm-systems/pull/9821) (open) gates the timeline syncobj on DRM ≥ 3.64 and uses the synchronous VA op below that. The PR's test notes say a physical DRM 3.61 host reproduced the hang, and DRM 3.64 completed the timeline waits.[^vmm9821] Host kernel versions for the RCCL symptom on the same DRM boundary are [comms/](../comms/README.md).

## Graphs

[rocm-systems#12638](https://github.com/ROCm/rocm-systems/issues/12638) (open) reports that on Core SDK **10.0.0**, a retained fork/join graph whose root segment holds only empty nodes can hang on first replay. `GraphExecSegmented::BuildSyncPlan()` allocates a completion signal for that segment and never attaches a producer, so the successor's barrier waits. The issue names released source commit `6b0e43f341195e203754e08f850e437ff2fc09f9` (`therock-10.0`).[^graph12638] The 7.14.0 release notes are dated 2026-07-15, before that 10.0.0 report.[^rn7140] Check this before any 10.x move if a capture uses empty join roots. No 7.14.0 pin change.

## Sources

[^malloc10943]: rocm-systems#10943, open, including the gfx1100 follow-up comment. https://github.com/ROCm/rocm-systems/issues/10943
[^ct2090]: CTranslate2#2090, open. https://github.com/OpenNMT/CTranslate2/issues/2090
[^vmm9820]: rocm-systems#9820, open. https://github.com/ROCm/rocm-systems/issues/9820
[^vmm9821]: rocm-systems#9821, open. https://github.com/ROCm/rocm-systems/pull/9821
[^graph12638]: rocm-systems#12638, open. https://github.com/ROCm/rocm-systems/issues/12638
[^rn7140]: ROCm Core SDK 7.14.0 release notes, dated 2026-07-15. https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html
