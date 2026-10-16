---
owner: ROCM_specialist
updated: 2026-10-14
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
  - https://github.com/invoke-ai/InvokeAI/issues/9487
  - https://github.com/ROCm/rocm-systems/issues/4817
  - https://github.com/torvalds/linux/commit/b89d58b6595d79dc3fe75e213e1f4c5efd0251d4
  - https://github.com/ROCm/legacy-rocm-build/issues/6003
  - https://github.com/ROCm/rocm-systems/blob/2d8b4d61b89499019698982be71926a4f9c4907e/projects/rocr-runtime/libhsakmt/src/openclose.c
  - https://github.com/ROCm/legacy-rocm-build/issues/6698
  - https://github.com/Niko1221/Strata/issues/2245
  - https://github.com/Niko1221/Strata/issues/2305
  - https://github.com/ROCm/rocm-systems/blob/e6527207c5057d6d5c992203e7970b3170e581b7/projects/clr/rocclr/device/rocm/rocdevice.cpp
  - https://github.com/ROCm/rocm-systems/blob/1ec2a772e4febc55079ad51ddb2398cf4e4ed230/projects/clr/rocclr/device/rocm/rocsettings.cpp
  - https://github.com/ROCm/rocm-systems/blob/therock-10.1/projects/clr/rocclr/utils/flags.hpp
  - https://github.com/ROCm/rocm-systems/blob/therock-7.14.1/projects/clr/rocclr/utils/flags.hpp
  - https://github.com/ROCm/rocm-systems/pull/8485
  - https://github.com/ROCm/legacy-rocm-build/issues/6407
---

# HIP runtime

Date: **2026-10-14**. Live dest stays ROCm **7.14.0** ([pin](../pin/README.md)). TheRock systems-pin chronology is [../log/runtime-scans.md](../log/runtime-scans.md). MIOpen's GEMM backend is a library routing fact: [libraries/](../libraries/README.md).

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

[InvokeAI#9487](https://github.com/invoke-ai/InvokeAI/issues/9487) (open) is a framework-side data point on gfx1100 (RX 7900 XTX, ROCm torch wheels): `.to('cuda')` on tensors that are still views into an mmap'd safetensors file ran at about 2 MB/s, against about 7 GB/s after a `clone()` into ordinary RAM. The stall sat in `c10::hip::memcpy_and_sync` → HSA runtime → ioctl with one CPU core at 100% and the GPU and disk idle. A 2026-10-13 comment confirms it on an R9700 with ROCm 7.2 and says the clone-first patch fixes it. The issue names no mechanism below the ioctl, so tying it to the userptr eviction above is our reading, not the report's.[^inv9487]

For our HIP code: when an engine streams mmap'd weights or keeps a large page-locked expert set while the host is short on RAM, a stall with an idle GPU and one spinning CPU core is this failure first, not a kernel hang. Try `GPU_PINNED_MIN_XFER_SIZE` (staging) or `HSA_USERPTR_FOR_PAGED_MEM=0`, and measure copy throughput without pressure before making either a default. No 7.14.0 repro on this page; no pin change.

### mmap'd sources: SVM split fault before Linux 7.2

[rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817) (open) reports `hipMemcpyAsync` failing with `Memory access fault ... Page not present` on valid parameters, first through llama.cpp checkpoint restore (MI100 and RX 7900 XTX, ROCm 7.2.0).[^svm4817] An AMD engineer (`harkgill-amd`, 2026-10-06) wrote that the cause is a KFD bug in how pinned SVM ranges are split, which left the head or tail of a valid range unmapped while SDMA was still reading it. With the fix, the same restore flow went from crashing in under 10 restores to passing more than 250, and they named amdgpu-dkms 7.1.9 as carrying it.[^svm4817c] The upstream commit is `b89d58b6595d` ("drm/amdkfd: Use exclusive bounds for SVM split alignment checks"): `ALIGN_DOWN(prange->last, 512)` treated a range ending one page before a 2 MB boundary as if its last 2 MB block did not exist, so a split there could drop the head or tail from the remap list.[^kfdsvm] GitHub's compare view puts the commit in Linux `v7.2-rc1` and `v7.2-rc2`, and not in `v7.1`.[^kfdsvm]

[legacy-rocm-build#6003](https://github.com/ROCm/legacy-rocm-build/issues/6003) (open) is the RDNA2 thread: random `hipErrorIllegalAddress` in ComfyUI and PyTorch on an RX 6700 XT (gfx1031 run as `HSA_OVERRIDE_GFX_VERSION=10.3.0`), reported again on ROCm 10.1.0 on 2026-10-12. One commenter traced their llama.cpp crashes to copying mmap'd model files under host memory pressure and avoided them with llama.cpp `--load-mode none`; AMD answered that this matches the #4817 fix. Another commenter, on 3× RX 6800 (gfx1030) with a 6.18 LTS kernel, reports that `HSA_USE_SVM=0` avoided the fault (about 300 restores, 0 faults, no measurable cost on their host).[^lrb6003] In libhsakmt, `HSA_USE_SVM=0` turns off SVM API support for the process.[^oc]

For the V620 host: check whether the amdgpu module in use is Linux 7.2 or later, or a DKMS build that carries `b89d58b6595d`. If it is neither, load weights from non-mmap'd buffers, or try `HSA_USE_SVM=0` and measure. Our reading, untested: `HSA_USERPTR_FOR_PAGED_MEM=0` from the section above backs paged memory with SVM, so on an unfixed kernel it may trade the eviction stall for this fault. This is a host-kernel floor, like the DRM 3.64 note above, not a 7.14.0 userspace pin change.

## Streams and hardware queues

clr does not give every HIP stream its own hardware queue. Per priority level it creates new HSA queues until it holds `GPU_MAX_HW_QUEUES` of them (default 4, same in `therock-7.14.1` and `therock-10.1`). After that, a new stream reuses an existing queue: an unshared queue first, then the lowest load. Under `DEBUG_HIP_DYNAMIC_QUEUES=1` (the default) "load" is a metric built from queue depth and user count; under `0` it is the user count only.[^q-flags][^q-pool] So which streams share a queue depends on creation order and on how busy each queue is at that moment. It is not fixed by the code alone.

Stream priority changed between releases. In `therock-7.14.1` and `therock-10.0`, `DEBUG_HIP_IGNORE_STREAM_PRIORITY` defaults to `false`, so a high-priority stream gets a queue from a separate high-priority pool. In `therock-10.1` it defaults to `true` ("Treat all streams as normal priority").[^q-flags][^q-flags714] The change is [rocm-systems#8485](https://github.com/ROCm/rocm-systems/pull/8485) (merged 2026-08-06). Its description says KFD does not create queues with different hardware priority, only an extra queue that competes for the same pipe, and that apps should raise `GPU_MAX_HW_QUEUES` if they need more queues.[^q-8485] An engine that relies on a high-priority side stream gets plain normal-priority streams on a 10.1 runtime unless it sets `DEBUG_HIP_IGNORE_STREAM_PRIORITY=0`.

### gfx1030 cross-queue stall (V620)

[legacy-rocm-build#6698](https://github.com/ROCm/legacy-rocm-build/issues/6698) (open, filed 2026-10-13) reports a silent hang on 2× Radeon Pro V620 (gfx1030, Linux 7.0 in-tree amdgpu, TheRock wheels 7.14.0a20260612 and 7.13.0a20260515). Two hardware queues hold valid pending AQL packets, the MEC pipes that own them are idle, no waves exist, and no error is raised anywhere; the process has to be killed. `amdgpu.cwsr_enable=0` did not help.[^q-6698] The workload is Strata's two-GPU layer split with an MTP drafter and two overlapped verify windows.[^q-2245]

The reporter's 2026-10-13 update: `GPU_MAX_HW_QUEUES=2` avoided the stall in 116 of 116 requests, while 4 (default) and 6 stalled. 8 did not stall but decoded about 40% slower, and 16 about 95% slower. With 2 queues, the drafter stream and the stage-1 verify stream share one hardware queue, so their per-window event hand-off never crosses queues or pipes. The reporter reads the trigger as that cross-queue hand-off; AMD had not replied when we read it.[^q-6698] In the Strata thread, one specific drafter attention kernel (8 cells per step with DPP lane exchanges) made the hang near-certain, and every variant without it ran clean; the reporter calls it a timing trigger, not a math bug. A gfx1100 user on Windows reported the same hang where that kernel does not run.[^q-2245] A separate R9700 (gfx1201) report found the drafter fast or slow depending on the engine start, and `GPU_MAX_HW_QUEUES=2` made every start fast. That is a throughput effect, not a hang.[^q-2305]

For our HIP code: if two streams hand work back and forth through events every step, assume their queue placement can differ from run to run. When a multi-stream engine on gfx1030 hangs with an idle GPU and no fault, try `GPU_MAX_HW_QUEUES=2` before blaming a kernel, then measure decode against the default. Putting both stages on one stream does not depend on clr's mapping. No 7.14.0 pin change.

### `multiProcessorCount` counts WGPs on RDNA

clr reads the agent's CU count and halves it when WGP mode is on. `GPU_ENABLE_WGP_MODE` defaults to `true`, and WGP mode is used on every gfx10+ target except gfx1250 and later.[^q-wgp][^q-flags] So `hipDeviceProp_t.multiProcessorCount` and `hipDeviceAttributeMultiprocessorCount` return the WGP count. [legacy-rocm-build#6407](https://github.com/ROCm/legacy-rocm-build/issues/6407) (open, docs) reports `16` on an RX 7600 with 32 CUs.[^q-6407] Our arithmetic from the code: a 72-CU V620 reports 36. A grid sized "one block per CU" from this value gets half the CUs. Size persistent or resident grids from the CU count (`rocminfo`, or 2× this value in WGP mode), and say which one a kernel note uses.

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
[^inv9487]: InvokeAI#9487, open, issue body (gfx1100, ~2 MB/s vs ~7 GB/s, py-spy stack) and `axxelh` comment 2026-10-13 (R9700, ROCm 7.2). https://github.com/invoke-ai/InvokeAI/issues/9487
[^svm4817]: rocm-systems#4817, open, issue body by `IMbackK`. https://github.com/ROCm/rocm-systems/issues/4817
[^svm4817c]: rocm-systems#4817 comment by `harkgill-amd`, 2026-10-06 (root cause, >250 restores, amdgpu-dkms 7.1.9). https://github.com/ROCm/rocm-systems/issues/4817#issuecomment-6023275073
[^kfdsvm]: Linux commit `b89d58b6595d` "drm/amdkfd: Use exclusive bounds for SVM split alignment checks", authored 2026-06-16; GitHub compare read 2026-10-13: `v7.2-rc1` and `v7.2-rc2` contain it, `v7.1` does not. https://github.com/torvalds/linux/commit/b89d58b6595d79dc3fe75e213e1f4c5efd0251d4
[^lrb6003]: legacy-rocm-build#6003, open, comments by `Jollym` (2026-10-03), `harkgill-amd` (2026-10-06), `Rhynri` (2026-10-07), `pkrasicki` (2026-10-12, ROCm 10.1.0). https://github.com/ROCm/legacy-rocm-build/issues/6003
[^oc]: libhsakmt `openclose.c` at rocm-systems `2d8b4d61`, `HSA_USE_SVM` handling. https://github.com/ROCm/rocm-systems/blob/2d8b4d61b89499019698982be71926a4f9c4907e/projects/rocr-runtime/libhsakmt/src/openclose.c
[^q-flags]: clr `flags.hpp` at tag `therock-10.1`: `GPU_MAX_HW_QUEUES` 4, `GPU_ENABLE_WGP_MODE` true, `DEBUG_HIP_DYNAMIC_QUEUES` 1, `DEBUG_HIP_IGNORE_STREAM_PRIORITY` true. https://github.com/ROCm/rocm-systems/blob/therock-10.1/projects/clr/rocclr/utils/flags.hpp
[^q-flags714]: clr `flags.hpp` at tag `therock-7.14.1`: `GPU_MAX_HW_QUEUES` 4, `DEBUG_HIP_IGNORE_STREAM_PRIORITY` false ("Ignore priority streams"); `therock-10.0` also false (read 2026-10-14). https://github.com/ROCm/rocm-systems/blob/therock-7.14.1/projects/clr/rocclr/utils/flags.hpp
[^q-pool]: clr `rocdevice.cpp` at `e6527207`, `Device::getQueueFromPool` and `Device::acquireQueue` (reuse only at the limit; unshared first; mode 1+ load metric, mode 0 refCount). Same functions present at tag `therock-7.14`. https://github.com/ROCm/rocm-systems/blob/e6527207c5057d6d5c992203e7970b3170e581b7/projects/clr/rocclr/device/rocm/rocdevice.cpp
[^q-8485]: rocm-systems#8485, "Disable Stream priorities in HIP to reflect KFD functionality", merged 2026-08-06. https://github.com/ROCm/rocm-systems/pull/8485
[^q-6698]: legacy-rocm-build#6698, open, issue body and `metzenwest` comment 2026-10-13 22:04 UTC (`GPU_MAX_HW_QUEUES` table). https://github.com/ROCm/legacy-rocm-build/issues/6698
[^q-2245]: Strata#2245, open, issue body (bisect to 6126f8d, kernel variant table) `bcrawford-cpm` comment 2026-10-13 16:17 UTC (gfx1100), and `metzenwest` comment 2026-10-13 17:46 UTC. https://github.com/Niko1221/Strata/issues/2245
[^q-2305]: Strata#2305, open (R9700 gfx1201, `GPU_MAX_HW_QUEUES` table). https://github.com/Niko1221/Strata/issues/2305
[^q-wgp]: clr `rocdevice.cpp` at `e6527207` (`maxComputeUnits_ / 2` when `enableWgpMode_`) and `rocsettings.cpp` at `1ec2a772` (WGP mode for gfx10+, off for gfx1250+). https://github.com/ROCm/rocm-systems/blob/1ec2a772e4febc55079ad51ddb2398cf4e4ed230/projects/clr/rocclr/device/rocm/rocsettings.cpp
[^q-6407]: legacy-rocm-build#6407, open (documentation). https://github.com/ROCm/legacy-rocm-build/issues/6407
