---
owner: ROCM_specialist
updated: 2026-11-06
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/pinned-host-evictions-2026-11-05.md
  - https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna/serving.md
  - https://github.com/opengfx1030/vllm-rdna/commit/6e647aeb1e4990119628b945a601b09790f778d5
  - https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/hwq-abba-2026-11-06.md
  - https://github.com/opengfx1030/vllm-rdna/commit/50570499f83e70e038a631ffa223be4e26fa6590
  - https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/rdna-ar-wedge-2026-11-04.md
  - https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md
  - https://github.com/ROCm/rocm-systems/issues/14778
---

# Pinned-host mlock, `GPU_MAX_HW_QUEUES`, and TheRock wheel lib path

Date: **2026-11-06**. Sourced from `opengfx1030/vllm-rdna` at commit `50570499` (and the mlock commit `6e647aeb` it includes). Mechanism is generic for ROCm dGPUs; do not read fleet inventory into it. Related background on pageable/pinned H2D eviction under reclaim: [README.md § Host memory pressure](README.md#host-memory-pressure-pinned-copies-and-queue-eviction). Related `GPU_MAX_HW_QUEUES` pool behaviour: [README.md § Streams and hardware queues](README.md#streams-and-hardware-queues).

## Pinned host memory and KFD queue evictions

On a dGPU, ROCr backs paged `hipHostMalloc` memory (including torch `pin_memory=True` buffers) with a **KFD userptr buffer object** over ordinary shmem pages. Those pages are not pinned in the `VmPin` sense: `VmPin` reads 0 on a live worker. Reclaim unmapping a page for swap-out, or compaction migrating it, reaches `amdgpu_hmm_invalidate_hsa`, and KFD **evicts every GPU queue of that process** until the restore worker re-validates the BO. `evicted_ms` grows when the queues run again, so a large invalidated BO shows up as seconds.[^phe-mech][^wedge-mech]

### Mitigation: `mlock` the pinned mappings

Commit [`6e647aeb`](https://github.com/opengfx1030/vllm-rdna/commit/6e647aeb1e4990119628b945a601b09790f778d5) adds a worker pass that, after model load and graph capture, walks `/proc/self/maps`, `mlock()`s every `rw-s /dev/zero (deleted)` mapping (the hipHostMalloc / torch pinned-allocator blocks), smallest first, and rescans every 2 s for blocks that appear later. A failed `mlock` only logs a warning.[^mlock-commit][^phe-fix]

Control env var `VLLM_ROCM_MLOCK_PINNED_HOST`:[^envs-mlock][^serving-mlock]

| Value | Effect |
|---|---|
| `auto` (default) | on for ROCm **gfx10x** only |
| `1` | on for any ROCm GPU |
| `0` | off |

`mlock` keeps reclaim off a page. Locked pages are protected from compaction **only** when `vm.compact_unevictable_allowed=0` (kernel default is 1). A Linux VM check in the investigation moved pages of locked mappings under the default and 0 of them with the sysctl at 0.[^phe-compact][^serving-mlock]

`RLIMIT_MEMLOCK` must sit above the pinned total. A TP4 Flash-Next worker in that write-up held about **25 GiB** locked (exact-size PLE table plus the other pinned blocks). Below the limit the worker logs `mlock ... failed` / `mlock failed for N pinned host mappings` and serves without those locks.[^envs-mlock][^serving-mlock] Check with `ulimit -l`. For a container whose soft/hard memlock is capped, raise it (Docker spelling: `--ulimit memlock=-1`). The fork docs name `ulimit -l` / `limits.d`, not the Docker flag; the flag is the usual Docker form of unlimited memlock.[^serving-mlock]

The locker only mlocks mappings **in the process that runs it**. Mappings owned by any other process are outside that pass (the investigation lists what it does not cover, including anything in a process that is not running this locker).[^phe-fix]

**Possibly related (hypothesis, not shown by the eviction proof):** [rocm-systems#14778](https://github.com/ROCm/rocm-systems/issues/14778) reports that `ROCR_VISIBLE_DEVICES` still opens every GPU's DRM render node and acquires a VM on each, so `hipHostRegister` maps into the hidden GPUs too. If a probe or neighbour process holds KFD contexts on cards it never runs work on, that process is outside the worker's mlock and can still carry userptr BOs on shared hosts. The issue states the VM-on-every-GPU behaviour; linking it as a cause of the pinned-host eviction finding is a hypothesis, not a measurement in the eviction doc.[^vis14778] (Already noted under [README.md § ROCR_VISIBLE_DEVICES](README.md#rocr_visible_devices-does-not-isolate-memory).)

### How to check a worker

From the investigation docs:[^phe-check][^wedge-check][^serving-mlock]

1. **`VmLck`** in `/proc/<pid>/status` (also logged by the locker as `VmLck A -> B GiB`). After a successful pass it should rise by the locked GiB and stay flat while the worker runs.
2. **KFD `evicted_ms`:** `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/evicted_ms` (same value in every `stats_*` of a process). A jump means queues were evicted and restored. Raw KFD SMI `QUEUE_EVICTION` records with trigger USERPTR match the same events.
3. **fdinfo vs locked size:** each render-node fdinfo of the worker has `drm-total-cpu` (system-memory BOs in that GPU's VM, userptr included). On the worker's own GPU it equals the sum of its `rw-s /dev/zero (deleted)` mappings. After a full lock, `drm-total-cpu − VmLck` should be about 0. The fork's `tools/rdna/port_v031/pinned_fdinfo_check.py PID...` prints that comparison.

## `GPU_MAX_HW_QUEUES` 2 vs 4 (paired ABBA)

A TP4 gfx1030 deployment measured `GPU_MAX_HW_QUEUES=4` against `2` for `flashnext-mtp2` (TP4 / EP4, MTP 2, AWQ-W4A16) in a paired ABBA of four fresh boots (order A1, B1, B2, A2; A=2, B=4) on an exclusive four-card slot. Tree `c6b1ff354e` (mlock merge); no `csrc` / cmake delta vs the prior tip.[^hwq]

| Result | Detail |
|---|---|
| ITL median | **−4.2 % to −6.9 %** for B vs A across the five cells (1k/512 at c=1/6/8, 16k/1024 at c=1/8); every B boot faster than both A boots in every cell |
| TTFT / prefill aggregate | within within-condition spread in every cell (decision metric) |
| KFD queues | **3** compute queues per worker at 2; **5** at 4 (before and after the cells) |
| dmesg | no `Runlist is getting oversubscribed` line; no new amdgpu fault / timeout / reset / AER |
| Evictions | 0 ms KFD eviction on all four workers through all 20 cells |

**Scope / caveat (from the source):** Slot B only; one model and one prompt family (random tokens); the rdna_ar stress that originally motivated 2 was not rerun. The measurement is one `flashnext-mtp2` TP4 server on four exclusive cards — stacking servers on the same GPU was not part of this ABBA.[^hwq]

## TheRock pip wheel library layout

TheRock's Python packaging puts the runtime core package under a top-level `_rocm_sdk_core*` directory in site-packages (sibling of `_rocm_sdk_libraries*` / devel).[^tr-pack]

In the fork's harness scripts at `50570499`, `LD_LIBRARY_PATH` had pointed at `$SITE/_rocm_sdk_core/lib/core/lib`, which **does not exist**. The fix uses `$SITE/_rocm_sdk_core/lib` (alongside the existing `host-math/lib` and `rocm_sysdeps/lib` entries). With the bad path, `libamdhip64` / `libamd_comgr` resolved from `/opt/rocm/...` instead, and under one venv layout that pulled a second LLVM and aborted `import vllm` with a duplicate LLVM option registration.[^ldpath]

TheRock's published packaging doc names the `_rocm_sdk_core` runtime package and the site-lib sibling layout; it does not spell the `lib/` vs `lib/core/lib` path in prose. The missing `core/lib` and the correct `_rocm_sdk_core/lib` path are from the fork commit above.[^tr-pack][^ldpath]

## Sources

[^phe-mech]: `docs/rdna2/pinned-host-evictions-2026-11-05.md` at `50570499`, § Mechanism (userptr BO, `amdgpu_hmm_invalidate_hsa`, process-wide queue eviction, `evicted_ms`). https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/pinned-host-evictions-2026-11-05.md
[^wedge-mech]: `docs/rdna2/rdna-ar-wedge-2026-11-04.md` at `50570499`, userptr / `evicted_ms` / `QUEUE_EVICTION` trigger USERPTR. https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/rdna-ar-wedge-2026-11-04.md
[^mlock-commit]: vllm-rdna commit `6e647aeb` "fix(rdna): mlock every pinned host mapping after warm-up". https://github.com/opengfx1030/vllm-rdna/commit/6e647aeb1e4990119628b945a601b09790f778d5
[^phe-fix]: `docs/rdna2/pinned-host-evictions-2026-11-05.md` at `50570499`, § The fix (scan/mlock/rescan, what it does not cover: other processes / non-`rw-s /dev/zero` registrations). https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/pinned-host-evictions-2026-11-05.md
[^envs-mlock]: `vllm/envs.py` at `50570499`, `VLLM_ROCM_MLOCK_PINNED_HOST` (`auto`/`1`/`0`; ~25 GiB per TP4 Flash-Next worker; failed mlock warns only). https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/vllm/envs.py
[^serving-mlock]: `docs/rdna/serving.md` at `50570499`, § Host settings for the pinned PLE table (`auto` gfx10x, `RLIMIT_MEMLOCK`, `vm.compact_unevictable_allowed=0`, `ulimit -l` / sysctl checks). https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna/serving.md
[^phe-compact]: `docs/rdna2/pinned-host-evictions-2026-11-05.md` at `50570499`, § Checking the mechanism on a real kernel (compaction moves locked pages only when `vm.compact_unevictable_allowed=1`). https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/pinned-host-evictions-2026-11-05.md
[^phe-check]: `docs/rdna2/pinned-host-evictions-2026-11-05.md` at `50570499`, § What a worker pins / Evidence tooling (`VmLck`, `drm-total-cpu`, `pinned_fdinfo_check.py`, sampler `evicted_ms`). https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/pinned-host-evictions-2026-11-05.md
[^wedge-check]: `docs/rdna2/rdna-ar-wedge-2026-11-04.md` at `50570499`, `evicted_ms` path under `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu_id>/`. https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/rdna-ar-wedge-2026-11-04.md
[^vis14778]: rocm-systems#14778, open: with `ROCR_VISIBLE_DEVICES` set, process still acquires a VM on every GPU and maps `hipHostRegister`'ed memory into the hidden ones. https://github.com/ROCm/rocm-systems/issues/14778
[^hwq]: `docs/rdna2/hwq-abba-2026-11-06.md` at `50570499` (method, ITL table, KFD queue counts, dmesg, Scope and caveats). https://github.com/opengfx1030/vllm-rdna/blob/50570499f83e70e038a631ffa223be4e26fa6590/docs/rdna2/hwq-abba-2026-11-06.md
[^ldpath]: vllm-rdna commit `50570499` "fix(rdna-tools): LD_LIBRARY_PATH uses _rocm_sdk_core/lib, not the missing core/lib". https://github.com/opengfx1030/vllm-rdna/commit/50570499f83e70e038a631ffa223be4e26fa6590
[^tr-pack]: TheRock `docs/packaging/python_packaging.md` (runtime packages expand into site-lib; `_rocm_sdk_core` sibling layout). https://github.com/ROCm/TheRock/blob/main/docs/packaging/python_packaging.md
