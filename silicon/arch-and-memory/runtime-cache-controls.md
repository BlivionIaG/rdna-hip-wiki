---
title: "Runtime and command-stream cache controls on gfx1030 (HIP persisting L2, SDMA MALL policy, PM4 GCR)"
sources:
  - https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/cache-residency.md
  - https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/sdma/copy.md
  - https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/sdma/atomics.md
  - https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/pm4/cache.md
  - https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/clr/hipamd/src/hip_device.cpp#L767-L800
  - https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L2745-L2782
  - https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/rocr-runtime/libhsakmt/src/memory.c#L289-L325
  - https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/clr/hipamd/src/hip_stream.cpp#L948-L985
  - https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/clr/hipamd/src/hip_graph_internal.hpp#L2064-L2097
  - https://github.com/ROCm/amdgpu/blob/5081d704e60e807e9541c355c53df57d88ea8cde/include/uapi/drm/amdgpu_drm.h#L632-L650
  - https://github.com/torvalds/linux/blob/fe2ec83746e501645709761605c2464a44fd2929/include/uapi/drm/amdgpu_drm.h#L589-L607
  - https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp#L354-L450
  - https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10_merged_sdma_packets.h#L746-L846
  - https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L228-L273
  - https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L600-L715
---

# Runtime and command-stream cache controls on gfx1030

Up: [arch-and-memory/](README.md) · Kernel-side ISA bits (GLC/SLC/DLC, no persist bit): [cache-policy.md](cache-policy.md) · IC sizing: [infinity-cache.md](infinity-cache.md)

[cache-policy.md](cache-policy.md) covers what a **kernel** can set. This page covers what the **runtime, copy engine and command processor** can set on gfx10.3. Mostly it covers what they cannot set. Source: ROCm/hrx-system `docs/reference/amd/gpu` at `0c93b424f24dc266dcee6ad31b8f84c94921a6d6`. Those files are byte-identical to `39d683333c2b5d393ddd51aef1eb3c1d01e9a07e`: `git log 39d68333..0c93b424 -- docs/reference/amd/gpu` is empty. Each claim also cites the PAL, ROCm or Linux line range that hrx-system cites.

## TL;DR (Take / Leave)

| Item | Verdict | Why |
|---|---|---|
| `hipDeviceProp_t::persistingL2CacheMaxSize` as a "can pin L2" signal | **Leave / Dead** | HIP fills it from the ordinary L2 size (`info.l2CacheSize_`)[^hip-props]. It does not report a reservable size. |
| CUDA-style `accessPolicyWindow` (stream/graph/launch attribute) to keep KV or weights resident | **Leave / Dead** | `accessPolicyMaxWindowSize = 0`[^hip-props]. Stream attribute rejects the window[^hip-stream]. Graph node rejects any nonempty window[^hip-graph]. |
| ROCr `HSA_AMD_AGENT_ATTRIBUTE_REQUEST_PERSISTING_L2_CACHE_SIZE` | **Leave / Dead** on the cited drivers | The thunk sends `AMDGPU_VM_OP_GL2_PERSISTING_L2_CACHE` on the render node[^thunk]. Upstream Linux and ROCm amdgpu (incl. `releases/therock-7.14`) define only RESERVE/UNRESERVE_VMID there, with no `size` field[^vm-uapi][^res]. KFD exposes no `persisting_cache_size_max`, so ROCr's maximum is 0 and ROCr rejects any positive request[^res][^rocr-setter]. |
| SDMA gfx103Plus `cache_policy` / `llc_policy` / `cpv` fields (MALL no-alloc on copies) | **Take Later (awareness)** | The packet has the bits. PAL fills them only when MALL is supported, for Navi2x, with a non-default `sdmaBypassMall` panel setting and a valid KMD L2 policy[^sdma-copy][^pal-policy]. HIP does not expose them. Do not plan a "copy-without-polluting-IC" path on them. |
| PM4 `ACQUIRE_MEM` GCR GFX10 fields (GL2_WB/INV, GLM, GLK, GLV, GL1, range ≤ 64 KiB heuristic) | **Take (reference)** | Useful when reading rocprofv3/thread-trace or HIP runtime barrier packets. It is not a kernel knob[^pm4-gcr][^pal-gcr]. |

Net effect for the dest stack: **residency on V620 is still working-set design + reuse order** ([cache-policy.md](cache-policy.md): "HIP default is already the try-to-keep-it policy"). No runtime knob overrides it. This matches the existing kernel-side verdict. It **does not change extras HIP or any UNC card**.

## 1. HIP "persisting L2" is a capacity echo

At rocm-systems `105dd4ff`, `hip_device.cpp` sets[^hip-props]:

```cpp
deviceProps.persistingL2CacheMaxSize = info.l2CacheSize_;
...
deviceProps.accessPolicyMaxWindowSize = 0;
```

`l2CacheSize_` comes from the ordinary `HSA_AGENT_INFO_CACHE_SIZE` L2 entry. That entry is the partition's L2 capacity, not the HSA persisting maximum[^res]. So on gfx1030 a nonzero `persistingL2CacheMaxSize` reports the 4 MiB L2 (see [l2-occupancy](../occupancy/l2-occupancy.md)). It is **not** an admission of reservation support[^res].

Consumers of the window[^res]:

- `hipDeviceSetLimit`: stack, heap and scratch threshold only. No persisting-cache case[^res].
- `hipStreamSetAttribute`: an access-policy window gets the invalid-value result[^hip-stream].
- `hipGraphKernelNodeSetAttribute`: `num_bytes` is bounded by `accessPolicyMaxWindowSize` (0), so any nonempty window is rejected[^hip-graph].
- Extended launch attributes: an access-policy window gets the unsupported-attribute result[^res].

**Porting note:** CUDA code that calls `cudaStreamSetAttribute(..., cudaStreamAttributeAccessPolicyWindow, ...)` for KV/weights must drop that call on HIP/gfx1030. Do not guard it with `persistingL2CacheMaxSize > 0`, because on HIP that test is always true.

## 2. ROCr persisting-size request has no native backend on the cited drivers

ROCr exposes a setter attribute (0) and queries `0xA125` (last requested) / `0xA126` (max)[^res]. `GpuAgent::SetAgentAttribute` rejects a request above the discovered max. It then calls the driver and stores the value only on success[^rocr-setter]. The thunk:

- uses the render-node fd with `DRM_IOCTL_AMDGPU_VM` and op `AMDGPU_VM_OP_GL2_PERSISTING_L2_CACHE`. It sets `args.in.size`, with no rounding and no completion wait[^thunk];
- has no counterpart in upstream Linux `fe2ec837`, ROCm amdgpu `82021279` or `releases/therock-7.14` `5081d704`. Their `drm_amdgpu_vm_in` has only `op` and `flags`[^vm-uapi][^res];
- reads its maximum from `persisting_cache_size_max`, which those KFD cache-property producers omit. The record is zero-initialized, so the max is 0[^res].

The "requested" query returns the stored value. It does not read back hardware state[^res]. Even if a later driver adds the op, the request carries no address range, queue or stream. It cannot express "keep this KV slab resident"[^res].

## 3. SDMA copy policy fields on gfx10.3 (PAL view)

PAL's `gfx103Plus` SDMA COPY_LINEAR view adds destination cache policy bits 20:18, source cache policy bits 28:26 and `cpv` at bit 19[^sdma-copy][^pal-layout]. The atomic packet places `cache_policy` at 22:20 and `cpv` at 24. The fence/increment packet uses `l2_policy` 25:24, `llc_policy` 26 and `cpv` 28[^sdma-atomics]. Field positions differ per packet, so one packet's layout does not carry over to another[^sdma-atomics].

PAL's policy logic[^sdma-copy][^pal-policy]:

- The fields are written only when `supportsMall`. Otherwise they stay zero.
- `GetCachePolicy` returns nonzero only for **Navi2x** with the matching read/write bypass setting. Bit 2 is **LLC no-alloc** (MALL/Infinity Cache bypass) and bits 1:0 are the KMD L2 policy.
- `cpv` is set only when `sdmaBypassMall` is not its default and `sdmaL2PolicyValid` is true. PAL's own comment says that by default it disables the MALL on products that have these control bits[^pal-policy].

These hints change cache placement only. They do not provide release or acquire ordering[^sdma-copy]. This page has no source for what ROCr's Linux SDMA blit path puts in these bits on gfx1030 (unsourced; not claimed). Treat SDMA MALL bypass as a **PAL/driver setting, not a HIP/vLLM control**.

## 4. PM4 ACQUIRE_MEM GCR word on GFX10 (reading aid)

Word 7 GCR bits, GFX10/GFX11 PAL layout[^pm4-gcr][^pal-gcr]:

| Bits | Field |
|---|---|
| 1:0 | `GLI_INV` (0 none, 1 all, 2 range, 3 first/last) |
| 3:2 | `GL1_RANGE` |
| 4 / 5 | `GLM_WB` / `GLM_INV` (metadata cache) |
| 6 / 7 | `GLK_WB` / `GLK_INV` (scalar K$) |
| 8 | `GLV_INV` (vector L0) |
| 9 | `GL1_INV` |
| 10 | `GL2_US` |
| 12:11 | `GL2_RANGE` (adds 1 = `VOL`) |
| 13 / 14 / 15 | `GL2_DISCARD` / `GL2_INV` / `GL2_WB` |
| 17:16 | `SEQ` (0 parallel, 1 forward, 2 reverse) |

PAL uses range mode only for nonzero base and size ≤ 64 KiB. hrx-system notes this is a runtime heuristic, not a hardware limit. Ranged maintenance walks page translations, so a big range can cost more than whole-cache maintenance, and an unmapped gap can fault[^pm4-gcr][^pal-acquire]. PAL requests forward `SEQ` when K$ WB and GL2 WB occur together[^pm4-gcr][^pal-acquire].

Kernel-writer relevance: none of these is reachable from HIP device code. They explain why host↔device or cross-queue handoffs cost what they cost. They do not give a way to keep data in L2 or Infinity Cache.

## Sources

[^hip-props]: ROCm/rocm-systems `105dd4ff35798f95646353bc08f6c885416ae17e`, `projects/clr/hipamd/src/hip_device.cpp` L767–L800 (fetched and checked 2026-11-19): https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/clr/hipamd/src/hip_device.cpp#L767-L800
[^hip-stream]: rocm-systems `105dd4ff35798f95646353bc08f6c885416ae17e`, `hip_stream.cpp` L948–L985, via hrx-system cache-residency.md: https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/clr/hipamd/src/hip_stream.cpp#L948-L985
[^hip-graph]: rocm-systems `105dd4ff35798f95646353bc08f6c885416ae17e`, `hip_graph_internal.hpp` L2064–L2097: https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/clr/hipamd/src/hip_graph_internal.hpp#L2064-L2097
[^rocr-setter]: rocm-systems `105dd4ff35798f95646353bc08f6c885416ae17e`, `amd_gpu_agent.cpp` L2745–L2782: https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L2745-L2782
[^thunk]: rocm-systems `105dd4ff35798f95646353bc08f6c885416ae17e`, `libhsakmt/src/memory.c` L289–L325: https://github.com/ROCm/rocm-systems/blob/105dd4ff35798f95646353bc08f6c885416ae17e/projects/rocr-runtime/libhsakmt/src/memory.c#L289-L325
[^vm-uapi]: ROCm/amdgpu `5081d704e60e807e9541c355c53df57d88ea8cde` (`releases/therock-7.14`) `include/uapi/drm/amdgpu_drm.h` L632–L650: https://github.com/ROCm/amdgpu/blob/5081d704e60e807e9541c355c53df57d88ea8cde/include/uapi/drm/amdgpu_drm.h#L632-L650 ; upstream Linux `fe2ec83746e501645709761605c2464a44fd2929` L589–L607: https://github.com/torvalds/linux/blob/fe2ec83746e501645709761605c2464a44fd2929/include/uapi/drm/amdgpu_drm.h#L589-L607
[^res]: ROCm/hrx-system `0c93b424f24dc266dcee6ad31b8f84c94921a6d6` (identical at `39d683333c2b5d393ddd51aef1eb3c1d01e9a07e`), `docs/reference/amd/gpu/cache-residency.md`: https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/cache-residency.md
[^sdma-copy]: hrx-system `0c93b424f24dc266dcee6ad31b8f84c94921a6d6`, `docs/reference/amd/gpu/sdma/copy.md` (header table and "PAL policy and compression selection"): https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/sdma/copy.md
[^sdma-atomics]: hrx-system `0c93b424f24dc266dcee6ad31b8f84c94921a6d6`, `docs/reference/amd/gpu/sdma/atomics.md`: https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/sdma/atomics.md
[^pal-policy]: GPUOpen-Drivers/pal `c5e800072a32f68b6ccc4422936d96167c6e0728`, `src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp` L354–L450 (`GetSettingByPassMall`, `GetCpvFromLlcPolicy`, `GetMallBypass`; fetched and checked 2026-11-19): https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp#L354-L450
[^pal-layout]: pal `c5e800072a32f68b6ccc4422936d96167c6e0728`, `gfx10_merged_sdma_packets.h` L746–L846: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10_merged_sdma_packets.h#L746-L846
[^pm4-gcr]: hrx-system `0c93b424f24dc266dcee6ad31b8f84c94921a6d6`, `docs/reference/amd/gpu/pm4/cache.md` ("Acquire GCR fields", "Range and sequencing policy"): https://github.com/ROCm/hrx-system/blob/0c93b424f24dc266dcee6ad31b8f84c94921a6d6/docs/reference/amd/gpu/pm4/cache.md
[^pal-gcr]: pal `c5e800072a32f68b6ccc4422936d96167c6e0728`, `src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp` L228–L273: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L228-L273
[^pal-acquire]: pal `c5e800072a32f68b6ccc4422936d96167c6e0728`, `gfx9CmdUtil.cpp` L600–L715: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L600-L715
