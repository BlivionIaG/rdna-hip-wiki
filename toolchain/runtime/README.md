---
owner: ROCM_specialist
updated: 2026-10-26
sources:
  - https://github.com/ROCm/TheRock/issues/9484
  - https://github.com/ROCm/TheRock/issues/9141
  - https://github.com/ROCm/TheRock/issues/9475
  - https://github.com/ROCm/rocm-systems/issues/14234
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
  - https://github.com/ROCm/TheRock/issues/9306
  - https://github.com/ROCm/TheRock/pull/9432
  - https://github.com/ROCm/TheRock/pull/9440
  - https://github.com/Niko1221/Strata/issues/2245
  - https://github.com/Niko1221/Strata/issues/2305
  - https://github.com/ROCm/rocm-systems/blob/e6527207c5057d6d5c992203e7970b3170e581b7/projects/clr/rocclr/device/rocm/rocdevice.cpp
  - https://github.com/ROCm/rocm-systems/blob/1ec2a772e4febc55079ad51ddb2398cf4e4ed230/projects/clr/rocclr/device/rocm/rocsettings.cpp
  - https://github.com/ROCm/rocm-systems/blob/therock-10.1/projects/clr/rocclr/utils/flags.hpp
  - https://github.com/ROCm/rocm-systems/blob/therock-7.14.1/projects/clr/rocclr/utils/flags.hpp
  - https://github.com/ROCm/rocm-systems/pull/8485
  - https://github.com/ROCm/legacy-rocm-build/issues/6407
  - https://github.com/pytorch/pytorch/issues/200608
  - https://github.com/pytorch/pytorch/pull/200774
  - https://github.com/Niko1221/Strata/issues/2729
  - https://github.com/ROCm/rocm-systems/pull/10639
  - https://github.com/ROCm/TheRock/pull/8259
  - https://github.com/ROCm/TheRock/pull/9021
  - https://github.com/ROCm/TheRock/pull/8833
  - https://github.com/ROCm/rocm-systems/pull/11277
  - https://github.com/ROCm/rocm-libraries/pull/13639
  - https://github.com/ROCm/rocm-systems/pull/5792
---

# HIP runtime

Date: **2026-10-16**. Live dest stays ROCm **7.14.0** ([pin](../pin/README.md)). TheRock systems-pin chronology is [../log/runtime-scans.md](../log/runtime-scans.md). MIOpen's GEMM backend is a library routing fact: [libraries/](../libraries/README.md).

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
  vmm --> vmmok["DRM 3.64: fix keeps the timeline wait"]
```

### `hipMallocAsync` default pool

[rocm-systems#10943](https://github.com/ROCm/rocm-systems/issues/10943) (open) reports that on ROCm 7.2 / HIP 7.2.53211, memory from `hipMallocAsync` on the device default pool can come back **zeroed while still in use**, with every call returning `hipSuccess`. The device still reports `hipDeviceAttributeMemoryPoolsSupported = 1`. The pure-HIP repro (allocate on stream 1, free on stream 2 after an event) failed 151 of 600 iterations; `hipMalloc` / `hipFree` failed 0 of 600; the same async pattern with `hipMemPoolAttrReleaseThreshold = UINT64_MAX` failed 0 of 600. The report's hypothesis is that the pool releases physical backing at a sync point (default threshold 0) before queued work finishes.[^malloc10943]

The same issue's opening note says gfx1032 and gfx1201 were reported to crash at load through CTranslate2, and that the author had no gfx11 report at filing time. A later comment on that issue adds **gfx1100** (RX 7900 XTX, HIP 7.2.53211): GPU page faults (`Page not present`) and a NULL read in `libamdhip64.so.7.2.53211` at process exit.[^malloc10943]

[CTranslate2#2090](https://github.com/OpenNMT/CTranslate2/issues/2090) (open) is the gfx1030 consumer report: official ROCm wheels, HIP 7.2.53211, faster-whisper silently drops words (the table's default-allocator rows sit between about 5% and 68% similarity to the reference, and the result changes every run). `CT2_CUDA_ALLOCATOR=cub_caching` made three runs byte-identical.[^ct2090]

The reports are on HIP 7.2.53211. This page has no 7.14.0 repro. Treat the bug as open for the 7.14.0 dest until someone runs the two-stream repro there. In HIP code, keep hot scratch on `hipMalloc` (or a cache in front of it) unless `ReleaseThreshold` is raised first. The W4A16 split-K allocator choice is [silicon/attention/fa-occupancy.md](../../silicon/attention/fa-occupancy.md). Whether PyTorch's default HIP caching allocator calls `hipMalloc`, and whether `PYTORCH_HIP_ALLOC_CONF=backend:cudaMallocAsync` selects this pool, was not re-opened this pass (unsourced). `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is the virtual-memory path in the next section, which is a different failure.[^vmm9820]

### Virtual memory hangs below DRM 3.64

[rocm-systems#9820](https://github.com/ROCm/rocm-systems/issues/9820) (open): with ROCm 7.14 userspace on amdgpu DRM 3.61, `hipMemSetAccess` blocks forever in `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT`. libhsakmt puts GEM_VA timeline output in the ioctl; DRM 3.64 added that output, and older amdgpu accepts the larger struct and ignores the extra fields, so the wait never completes. PyTorch reaches it through `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. Turning expandable segments off avoids the path.[^vmm9820]

[rocm-systems#9821](https://github.com/ROCm/rocm-systems/pull/9821) (merged to `develop` on 2026-10-29; not in a tagged ROCm release yet) gates the timeline syncobj on DRM ≥ 3.64 and uses the synchronous VA op below that. The PR's test notes say a physical DRM 3.61 host reproduced the hang, and DRM 3.64 completed the timeline waits.[^vmm9821] Host kernel versions for the RCCL symptom on the same DRM boundary are [comms/](../comms/README.md).

[pytorch#200608](https://github.com/pytorch/pytorch/issues/200608) (open) is a second expandable-segments failure, in PyTorch's allocator rather than the kernel interface. On WSL2 (RX 7800 XT, ROCm 7.2.4, torch 2.12.1+rocm7.2) the allocator hits `TORCH_INTERNAL_ASSERT(!handles_.at(i))` in `ExpandableSegment::map()` with 4 to 5 GiB still free. A 2026-10-15 comment from bare-metal Linux (RX 7900 XT gfx1100, kernel 7.0, torch 2.14.1+rocm7.2, HIP 7.2.53211) saw no assert. Instead the same seed gave a different output every run and the next prompt hit `hipErrorIllegalAddress`, with an amdgpu page fault at a 2 MiB-aligned address no longer mapped in the process. `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` did not help. With expandable segments off: 0 faults and 5 of 5 runs bit-identical.[^pt200608] [pytorch#200774](https://github.com/pytorch/pytorch/pull/200774) (open) says the cause is `map()` recording new handles before `hipMemMap` / `hipMemSetAccess` succeed, so a failed map leaves slots occupied, and makes the mapping roll back on failure.[^pt200774] The comment does not give the DRM version, so whether its failed map is the DRM 3.64 hang above is not known. Until #200774 is in a wheel we use, leave `expandable_segments` off on ROCm.

## Host memory pressure: pinned copies and queue eviction

[rocm-systems#12528](https://github.com/ROCm/rocm-systems/issues/12528) (open) reports that `hipMemcpy` from pageable (mmap'd) host memory stalls for minutes when the host is under reclaim. clr pins each pageable chunk with `hsa_amd_memory_lock`, which creates a KFD userptr buffer object. Under reclaim the MMU notifier invalidates that range and KFD evicts the process's GPU queues. In the reporter's gfx1200 run, queues were evicted for 146.5 s of about 195 s and the load never finished in 180 s. The same load forced through clr's staging path with `GPU_PINNED_MIN_XFER_SIZE=100000` finished in about 36 s with about 2 s of eviction. The reporter had not measured what staging costs without memory pressure.[^pin12528] An AMD engineer (`schung-amd`, 2026-10-01) wrote that they reproduced it on **gfx1100** with TheRock nightlies.[^pin12528c] A 2026-10-12 comment adds a gfx1201 report where pinned host memory (`hipHostMalloc` / `hipHostRegister`) also sees queue eviction during page migration.[^pin12528c]

[rocm-systems#12670](https://github.com/ROCm/rocm-systems/pull/12670) (open, not merged) would skip pinning in `DmaBlitManager::getBuffer()` when both `MemAvailable` is low and memory PSI avg10 is above 5%. Its gfx1100 (RX 7900 XTX, nightly `10.2.0a20261001`) test went from 3 of 3 runs timing out at 180 s to about 15 s.[^pin12670]

Strata [#1705](https://github.com/Niko1221/Strata/issues/1705) (open) is an engine-side gfx1100 report (RX 7900 XTX, Linux, ROCm wheels `7.10.0a20251120`): nine verify-timeout and prompt-watchdog deaths in one day during long-context serving under host reclaim, already with `GPU_PINNED_MIN_XFER_SIZE=1048576` set. Adding `HSA_USERPTR_FOR_PAGED_MEM=0` stopped them in the next run (283 requests over 4 h 54 m, up to 245 625 tokens). The issue ties it to the same mechanism as the gfx1030 verify timeouts in Strata #649, and names Pop!_OS's default `vm.swappiness=180` zram setup as the trigger on that box.[^st1705] In libhsakmt, userptr for paged memory is on unless `HSA_USERPTR_FOR_PAGED_MEM` is set to `0`; with it off, paged memory is backed by SVM unless `HSA_SVM_FOR_PAGED_MEM=0`.[^fmm]

Strata [#2729](https://github.com/Niko1221/Strata/issues/2729) (open, filed 2026-10-15) is a second gfx1100 box (RX 7900 XT 20 GB, 31 GiB RAM, CachyOS kernel 7.2.9, zram) with a 14 to 21 GiB expert pool held as one paged SVM region. Before the env var, a `kworker` stuck in `svm_range_restore_work` took the box down. With `HSA_USERPTR_FOR_PAGED_MEM=0` the amdkfd hang went away, but `kswapd` order-0 allocation failures (the reclaim pressure itself) stayed, and a suspend with the SVM mapping active never resumed. The report's point is that the env var removes the hang on a 125 GiB box (#1705) but not the pressure on a 31 GiB one.[^st2729] So on a RAM-tight host, size the resident set to fit RAM; the env var alone is not enough.

[InvokeAI#9487](https://github.com/invoke-ai/InvokeAI/issues/9487) (open) is a framework-side data point on gfx1100 (RX 7900 XTX, ROCm torch wheels): `.to('cuda')` on tensors that are still views into an mmap'd safetensors file ran at about 2 MB/s, against about 7 GB/s after a `clone()` into ordinary RAM. The stall sat in `c10::hip::memcpy_and_sync` → HSA runtime → ioctl with one CPU core at 100% and the GPU and disk idle. A 2026-10-13 comment confirms it on an R9700 with ROCm 7.2 and says the clone-first patch fixes it. The issue names no mechanism below the ioctl, so tying it to the userptr eviction above is our reading, not the report's.[^inv9487]

For our HIP code: when an engine streams mmap'd weights or keeps a large page-locked expert set while the host is short on RAM, a stall with an idle GPU and one spinning CPU core is this failure first, not a kernel hang. Try `GPU_PINNED_MIN_XFER_SIZE` (staging) or `HSA_USERPTR_FOR_PAGED_MEM=0`, and measure copy throughput without pressure before making either a default. No 7.14.0 repro on this page; no pin change.

### Small pageable H2D copies on gfx11

[rocm-systems#10639](https://github.com/ROCm/rocm-systems/pull/10639) (open) says `hipMemcpyAsync` can silently drop a small host-to-device copy from pageable memory, reproduced intermittently on gfx1101. Small pageable copies go through a pinned staging slot that a blit shader reads. clr bump-allocates slots at the cache line size the agent reports, 64 bytes on gfx11, while L2 tracks 128-byte lines. The first copy's shader can pull the neighbor slot into L2 before the CPU fills it, so the second copy reads stale data; in the report every lost copy is odd-numbered and reads back as zero. The PR floors the reported line at 128 bytes on gfx11.[^clr10639] It names gfx11 only; gfx1030 is not discussed. For gfx1100 code that sends many small per-step uploads from pageable memory, send them from pinned (`hipHostMalloc`) buffers or check them on readback until the PR is in a pin.

### mmap'd sources: SVM split fault before Linux 7.2

[rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817) (open) reports `hipMemcpyAsync` failing with `Memory access fault ... Page not present` on valid parameters, first through llama.cpp checkpoint restore (MI100 and RX 7900 XTX, ROCm 7.2.0).[^svm4817] An AMD engineer (`harkgill-amd`, 2026-10-06) wrote that the cause is a KFD bug in how pinned SVM ranges are split, which left the head or tail of a valid range unmapped while SDMA was still reading it. With the fix, the same restore flow went from crashing in under 10 restores to passing more than 250, and they named amdgpu-dkms 7.1.9 as carrying it.[^svm4817c] The upstream commit is `b89d58b6595d` ("drm/amdkfd: Use exclusive bounds for SVM split alignment checks"): `ALIGN_DOWN(prange->last, 512)` treated a range ending one page before a 2 MB boundary as if its last 2 MB block did not exist, so a split there could drop the head or tail from the remap list.[^kfdsvm] GitHub's compare view puts the commit in Linux `v7.2-rc1` and `v7.2-rc2`, and not in `v7.1`.[^kfdsvm]

[legacy-rocm-build#6003](https://github.com/ROCm/legacy-rocm-build/issues/6003) (open) is the RDNA2 thread: random `hipErrorIllegalAddress` in ComfyUI and PyTorch on an RX 6700 XT (gfx1031 run as `HSA_OVERRIDE_GFX_VERSION=10.3.0`), reported again on ROCm 10.1.0 on 2026-10-12. One commenter traced their llama.cpp crashes to copying mmap'd model files under host memory pressure and avoided them with llama.cpp `--load-mode none`; AMD answered that this matches the #4817 fix. Another commenter, on 3× RX 6800 (gfx1030) with a 6.18 LTS kernel, reports that `HSA_USE_SVM=0` avoided the fault (about 300 restores, 0 faults, no measurable cost on their host).[^lrb6003] In libhsakmt, `HSA_USE_SVM=0` turns off SVM API support for the process.[^oc]

For the V620 host: check whether the amdgpu module in use is Linux 7.2 or later, or a DKMS build that carries `b89d58b6595d`. If it is neither, load weights from non-mmap'd buffers, or try `HSA_USE_SVM=0` and measure. Our reading, untested: `HSA_USERPTR_FOR_PAGED_MEM=0` from the section above backs paged memory with SVM, so on an unfixed kernel it may trade the eviction stall for this fault. This is a host-kernel floor, like the DRM 3.64 note above, not a 7.14.0 userspace pin change.

## Streams and hardware queues

clr does not give every HIP stream its own hardware queue. Per priority level it creates new HSA queues until it holds `GPU_MAX_HW_QUEUES` of them (default 4, same in `therock-7.14.1` and `therock-10.1`). After that, a new stream reuses an existing queue: an unshared queue first, then the lowest load. Under `DEBUG_HIP_DYNAMIC_QUEUES=1` (the default) "load" is a metric built from queue depth and user count; under `0` it is the user count only.[^q-flags][^q-pool] So which streams share a queue depends on creation order and on how busy each queue is at that moment. It is not fixed by the code alone.

Stream priority changed between releases. In `therock-7.14.1` and `therock-10.0`, `DEBUG_HIP_IGNORE_STREAM_PRIORITY` defaults to `false`, so a high-priority stream gets a queue from a separate high-priority pool. In `therock-10.1` it defaults to `true` ("Treat all streams as normal priority").[^q-flags][^q-flags714] The change is [rocm-systems#8485](https://github.com/ROCm/rocm-systems/pull/8485) (merged 2026-08-06). Its description says KFD does not create queues with different hardware priority, only an extra queue that competes for the same pipe, and that apps should raise `GPU_MAX_HW_QUEUES` if they need more queues.[^q-8485] An engine that relies on a high-priority side stream gets plain normal-priority streams on a 10.1 runtime unless it sets `DEBUG_HIP_IGNORE_STREAM_PRIORITY=0`.

### gfx1030 cross-queue stall (V620)

[TheRock#9306](https://github.com/ROCm/TheRock/issues/9306) (open, filed 2026-10-13 as legacy-rocm-build#6698, whose URL now redirects here; labelled `status: triage` and assigned to an AMD engineer, no AMD reply yet as of 2026-10-23) reports a silent hang on 2× Radeon Pro V620 (gfx1030, Linux 7.0 in-tree amdgpu, TheRock wheels 7.14.0a20260612 and 7.13.0a20260515). Two hardware queues hold valid pending AQL packets, the MEC pipes that own them are idle, no waves exist, and no error is raised anywhere; the process has to be killed. `amdgpu.cwsr_enable=0` did not help.[^q-6698] The workload is Strata's two-GPU layer split with an MTP drafter and two overlapped verify windows.[^q-2245]

The reporter's 2026-10-13 update: `GPU_MAX_HW_QUEUES=2` avoided the stall in 116 of 116 requests, while 4 (default) and 6 stalled. 8 did not stall but decoded about 40% slower, and 16 about 95% slower. With 2 queues, the drafter stream and the stage-1 verify stream share one hardware queue, so their per-window event hand-off never crosses queues or pipes. The reporter reads the trigger as that cross-queue hand-off; AMD had not replied when we read it.[^q-6698] In the Strata thread, one specific drafter attention kernel (8 cells per step with DPP lane exchanges) made the hang near-certain, and every variant without it ran clean; the reporter calls it a timing trigger, not a math bug. A gfx1100 user on Windows reported the same hang where that kernel does not run.[^q-2245] A separate R9700 (gfx1201) report found the drafter fast or slow depending on the engine start, and `GPU_MAX_HW_QUEUES=2` made every start fast. That is a throughput effect, not a hang.[^q-2305]

For our HIP code: if two streams hand work back and forth through events every step, assume their queue placement can differ from run to run. When a multi-stream engine on gfx1030 hangs with an idle GPU and no fault, try `GPU_MAX_HW_QUEUES=2` before blaming a kernel, then measure decode against the default. Putting both stages on one stream does not depend on clr's mapping. No 7.14.0 pin change.

### gfx1100 idle power with a pending barrier (MES)

[TheRock#9484](https://github.com/ROCm/TheRock/issues/9484) (open, filed 2026-10-26 01:39 UTC, no AMD reply yet) reports that on an RX 7900 XTX (gfx1100, MES firmware 0x80), a single HSA queue holding one unsatisfied AQL `BARRIER_AND` packet raises idle board power from 9–10 W to 92–94 W at 100% GFX activity, with no compute wave resident. Drained queues idle normally. The reporter reproduced it on ROCm 7.2.4, 7.14 and 10.1, and says `sched_policy=2` and the August gfx11 MES timer correction did not fix it.[^q-9484] The reporter links it to [TheRock#9141](https://github.com/ROCm/TheRock/issues/9141), an older 100%-utilisation report where an AMD contributor reproduced stuck 100% GPU use on gfx1100 with three or more llama.cpp instances and suggested `GPU_MAX_HW_QUEUES=1` as a partial workaround.[^q-9141]

Our reading, not tested here: a HIP stream that waits on an event that is not yet recorded (`hipStreamWaitEvent` ahead of the producer) leaves exactly this kind of barrier packet in the queue, so a server that parks streams on cross-stream events between requests may burn power on gfx1100 while idle. Check board power on the W7800 and 7900 XTX boxes with an idle engine before treating it as a load problem. No gfx1030 report so far. No 7.14.0 pin change.

### `multiProcessorCount` counts WGPs on RDNA

clr reads the agent's CU count and halves it when WGP mode is on. `GPU_ENABLE_WGP_MODE` defaults to `true`, and WGP mode is used on every gfx10+ target except gfx1250 and later.[^q-wgp][^q-flags] So `hipDeviceProp_t.multiProcessorCount` and `hipDeviceAttributeMultiprocessorCount` return the WGP count. [legacy-rocm-build#6407](https://github.com/ROCm/legacy-rocm-build/issues/6407) (open, docs) reports `16` on an RX 7600 with 32 CUs.[^q-6407] Our arithmetic from the code: a 72-CU V620 reports 36. A grid sized "one block per CU" from this value gets half the CUs. Size persistent or resident grids from the CU count (`rocminfo`, or 2× this value in WGP mode), and say which one a kernel note uses.

## Hostcall: device `printf` and `assert`

Device `printf` and `assert` reach the host through a hostcall buffer. The device side (ockl in device-libs) and the host side (clr) must speak the same protocol. TheRock [#8259](https://github.com/ROCm/TheRock/pull/8259) (merged 2026-10-02) turned the new hostcall implementation on by default by setting `THEROCK_USE_NEW_HOSTCALL_IMPL` to ON, which TheRock passes to both device-libs and clr; the old path stayed selectable with `=OFF`.[^tr8259] [#9021](https://github.com/ROCm/TheRock/pull/9021) (merged 2026-10-15 20:07 UTC) reverts it because it "breaks HSA and ROCGDB" (two internal P1 tickets). A reviewer wrote that it broke things on 110X and gfx90a, and that `rocgdb-gpu` tests were clean on 110X, gfx90a and gfx1151 with the revert.[^tr9021] A related TheRock PR, [#8833](https://github.com/ROCm/TheRock/pull/8833) (closed 2026-10-15 without merge), was about Triton wheels shipping an OCKL that did not match the SDK, so device-side hostcalls in Triton kernels hung.[^tr8833]

Our reading, not checked against nightly contents: TheRock nightlies built between those two merges carry the new hostcall default. If device `printf`, `assert`, or rocgdb misbehaves on a gfx1100 box with such a nightly, that is the first suspect, not the kernel. The 7.14.0 dest predates #8259.

[#9432](https://github.com/ROCm/TheRock/pull/9432) (merged 2026-10-22 22:31 UTC) relands it: it reverts #9021 and sets `THEROCK_USE_NEW_HOSTCALL_IMPL` back to ON for device-libs and clr. Its description says "Land this after 10.2 snaps" and does not say what fixed the HSA and ROCgdb breakage.[^tr9432] About two and a half hours earlier, [#9440](https://github.com/ROCm/TheRock/pull/9440) raised the main-branch version to 10.3.0.[^tr9440] Our reading: TheRock main nightlies from 2026-10-23 on carry the new hostcall again, and a 10.2 release cut from before #9432 would not. The same triage as above applies to those nightlies, and `-DTHEROCK_USE_NEW_HOSTCALL_IMPL=OFF` is the switch for a source build. The 7.14.0 dest is unaffected.

[TheRock#9475](https://github.com/ROCm/TheRock/issues/9475) (open, filed 2026-10-25) is a Windows report on an RX 6800 (gfx1030, TheRock `rocm-sdk` 10.1.0 wheels): each hostcall takes seconds because the device's mailbox interrupt never seems to reach the host listener, which only wakes on its 4–16 s timeout under the default `USE_NEW_HOSTCALL_IMPL=OFF` in that build. One `printf` per thread takes 12–16 s; two per thread get the kernel killed after about 20 s, `hipDeviceSynchronize` still returns success, and the next `hipMemcpy` fails with `hipErrorLaunchFailure`. Building with `-mprintf-kind=buffered` avoids it.[^hc9475] It is Windows-only so far; on Linux gfx1030 this page has no such report. `-mprintf-kind=buffered` is also the low-risk choice for debug builds of our own kernels.

[rocm-systems#11277](https://github.com/ROCm/rocm-systems/pull/11277) (open) covers a separate hostcall failure. A kernel whose metadata declares `hidden_hostcall_buffer` cannot be dispatched on a device without PCIe AtomicOps to the host, and today clr refuses at dispatch with `hipErrorIllegalState` and only a debug-level log line. The PR says any kernel that links `printf` or `assert` declares the buffer whether or not it runs a hostcall, and that RCCL 2.30.4 declares it on thirteen kernels; it cites RCCL reports on gfx1100 and gfx1201 that reach this failure. The PR names the refused kernel and adds an opt-in to proceed.[^clr11277] For our kernels: keep `printf` and `assert` out of release builds, so a host without PCIe atomics can still launch them. Whether the V620 host's PCIe path exposes AtomicOps was not checked here.

## Mixed-architecture hosts

A box with a gfx1030 and a gfx1100 in one process has two open runtime and library gaps.

- **hipBLASLt.** [rocm-libraries#13639](https://github.com/ROCm/rocm-libraries/pull/13639) (open, review required) says hipBLASLt loads kernels for one architecture per process, whichever device uses it first. Other devices then get zero GEMM algorithms (success with no results), wrong code objects for ExtOps (Softmax, LayerNorm, AMax), or failing MatrixTransform. If the first device has no kernels at all, every device gets `HIPBLAS_STATUS_INVALID_VALUE`. Its example is a gfx1100 + gfx1201 PyTorch host where fp8 `torch._scaled_mm` fails on the gfx1201 after the gfx1100 ran a GEMM. The PR says rocBLAS fixed the same problem in [#4781](https://github.com/ROCm/rocm-libraries/pull/4781).[^lt13639] gfx1030 has no hipBLASLt kernels in the 10.0.0 wheel ([libraries/](../libraries/README.md)), so on a gfx1030 + gfx1100 host, a gfx1030 touching hipBLASLt first could leave the gfx1100 with none. That last case is our reading of the PR, not a test.
- **Code object load.** [rocm-systems#5792](https://github.com/ROCm/rocm-systems/pull/5792) (open) says `hipGetFuncBySymbol` fails with `hipErrorInvalidDeviceFunction` on every device when any device in the system has no compatible code object, because the fat-binary loop stops at the first device without one. The report was gfx906 + gfx908; `ROCR_VISIBLE_DEVICES` isolation worked around it.[^clr5792] A binary built only for `gfx11-generic` on a host that also has a gfx1030 fits that pattern.

Until both land: build every binary for every arch in the box, and run one process per architecture with `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`.

## Graphs

[rocm-systems#12638](https://github.com/ROCm/rocm-systems/issues/12638) (open) reports that on Core SDK **10.0.0**, a retained fork/join graph whose root segment holds only empty nodes can hang on first replay. `GraphExecSegmented::BuildSyncPlan()` allocates a completion signal for that segment and never attaches a producer, so the successor's barrier waits. The issue names released source commit `6b0e43f341195e203754e08f850e437ff2fc09f9` (`therock-10.0`).[^graph12638] The 7.14.0 release notes are dated 2026-07-15, before that 10.0.0 report.[^rn7140] Check this before any 10.x move if a capture uses empty join roots. No 7.14.0 pin change.

[rocm-systems#13234](https://github.com/ROCm/rocm-systems/issues/13234) (open, filed 2026-10-10) reports that `hipGraphLaunch` never returns when one graph segment holds at least `2 * ROC_AQL_QUEUE_SIZE` AQL packets: the host spins at 100% CPU and the GPU sits idle. The issue traces it to `VirtualGPU::ReserveAqlSlots` in clr publishing the whole packet count before writing bodies, with no cap at the ring size. With a 16384-slot ring, 32767 launches completed and 32768 hung; the report notes that a batch just above one ring passes. Reproduced on gfx1151 with ROCm 7.2.1 and 10.0.0.[^graph13234] The code it names is in clr's ROCm device path, not arch-specific code, but this page has no gfx1030 or gfx1100 repro. Keep captured decode graphs well under two rings per segment, or raise `ROC_AQL_QUEUE_SIZE`.

[rocm-systems#14234](https://github.com/ROCm/rocm-systems/issues/14234) (open, filed 2026-10-26) reports a SIGSEGV inside `libamdhip64` 7.2.4 (`hip-runtime-amd 7.2.53211.70204`) on an RX 7900 XTX (gfx1100): an internal kernarg arena allocator returns NULL and the caller writes to address 0 with no check. The backtrace runs through llama.cpp's CUDA-graph capture path during MTP speculative decoding. The reporter's core analysis shows the arena alive with a fresh, never-used 4 MiB chunk and a zero cursor, and does not name a root cause.[^graph14234] Watch it if a captured decode graph crashes in `libamdhip64` with a NULL destination; it is not reported on 7.14.0.

## Sources

[^malloc10943]: rocm-systems#10943, open, including the gfx1100 follow-up comment. https://github.com/ROCm/rocm-systems/issues/10943
[^ct2090]: CTranslate2#2090, open. https://github.com/OpenNMT/CTranslate2/issues/2090
[^vmm9820]: rocm-systems#9820, open. https://github.com/ROCm/rocm-systems/issues/9820
[^vmm9821]: rocm-systems#9821, merged 2026-10-29T21:53Z into develop. https://github.com/ROCm/rocm-systems/pull/9821
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
[^q-6698]: TheRock#9306 (formerly legacy-rocm-build#6698), open, issue body and `metzenwest` comment 2026-10-13 22:04 UTC (`GPU_MAX_HW_QUEUES` table); labels and assignee read 2026-10-23. https://github.com/ROCm/TheRock/issues/9306
[^q-2245]: Strata#2245, open, issue body (bisect to 6126f8d, kernel variant table) `bcrawford-cpm` comment 2026-10-13 16:17 UTC (gfx1100), and `metzenwest` comment 2026-10-13 17:46 UTC. https://github.com/Niko1221/Strata/issues/2245
[^q-2305]: Strata#2305, open (R9700 gfx1201, `GPU_MAX_HW_QUEUES` table). https://github.com/Niko1221/Strata/issues/2305
[^q-wgp]: clr `rocdevice.cpp` at `e6527207` (`maxComputeUnits_ / 2` when `enableWgpMode_`) and `rocsettings.cpp` at `1ec2a772` (WGP mode for gfx10+, off for gfx1250+). https://github.com/ROCm/rocm-systems/blob/1ec2a772e4febc55079ad51ddb2398cf4e4ed230/projects/clr/rocclr/device/rocm/rocsettings.cpp
[^q-6407]: legacy-rocm-build#6407, open (documentation). https://github.com/ROCm/legacy-rocm-build/issues/6407
[^pt200608]: pytorch#200608, open, issue body (WSL2, RX 7800 XT) and `davidk-croesus` comment 2026-10-15 01:15 UTC (bare metal gfx1100). https://github.com/pytorch/pytorch/issues/200608
[^pt200774]: pytorch#200774, open, "[ROCm] Make expandable segment mapping failures retryable". https://github.com/pytorch/pytorch/pull/200774
[^st2729]: Strata#2729, open, issue body (events 1-4, `HSA_USERPTR_FOR_PAGED_MEM=0`, suspend). https://github.com/Niko1221/Strata/issues/2729
[^clr10639]: rocm-systems#10639, open, "fix(clr): set gfx11 cache line size to 128 bytes" (gfx1101, AIRUNTIME-2666). https://github.com/ROCm/rocm-systems/pull/10639
[^tr8259]: TheRock#8259, merged 2026-10-02 20:22 UTC. https://github.com/ROCm/TheRock/pull/8259
[^tr9021]: TheRock#9021, merged 2026-10-15 20:07 UTC; description, and `lumachad` comments 2026-10-15 07:42 and 15:03 UTC. https://github.com/ROCm/TheRock/pull/9021
[^tr8833]: TheRock#8833, "Fix Triton hostcalls by packaging SDK-matched OCKL", closed 2026-10-15 without merge. https://github.com/ROCm/TheRock/pull/8833
[^tr9432]: TheRock#9432, "feat(hostcall): Reland: enable the new hostcall implementation by default", merged 2026-10-22 22:31 UTC; description read 2026-10-23. https://github.com/ROCm/TheRock/pull/9432
[^tr9440]: TheRock#9440, "Raise version to 10.3.0", merged 2026-10-22 20:02 UTC. https://github.com/ROCm/TheRock/pull/9440
[^clr11277]: rocm-systems#11277, open. https://github.com/ROCm/rocm-systems/pull/11277
[^lt13639]: rocm-libraries#13639, open, "fix(hipblaslt): load kernels per GPU architecture on mixed-architecture systems". https://github.com/ROCm/rocm-libraries/pull/13639
[^clr5792]: rocm-systems#5792, open, "[clr] Fix hipGetFuncBySymbol failure in heterogeneous multi-GPU systems". https://github.com/ROCm/rocm-systems/pull/5792
[^q-9484]: TheRock#9484, "[gfx1100][MES] Pending AQL BARRIER_AND forces 100% GFX / ~95W idle power on RX 7900 XTX", open, read 2026-10-26. https://github.com/ROCm/TheRock/issues/9484
[^q-9141]: TheRock#9141, "100% GPU usage for multiple models with pipeline parallel", open; comments by zichguan-amd 2025-08-14 to 2025-12-10. https://github.com/ROCm/TheRock/issues/9141
[^hc9475]: TheRock#9475, "gfx1030 on Windows: each hostcall takes seconds, so a kernel calling printf twice times out", open, read 2026-10-26. https://github.com/ROCm/TheRock/issues/9475
[^graph14234]: rocm-systems#14234, open, read 2026-10-26 with two reporter follow-ups. https://github.com/ROCm/rocm-systems/issues/14234
