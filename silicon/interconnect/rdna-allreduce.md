---
title: "gfx1030 push one-shot all-reduce (`rdna_ar`) — silicon lock"
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/3b59ee16e553168b1e3809c9a141088662171448
  - https://github.com/opengfx1030/vllm-rdna/commit/a4060647cfbb32d3b907eba6018501bb8b251d22
  - https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/distributed/device_communicators/rdna_all_reduce.py
  - https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/envs.py
  - https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cuh
  - https://github.com/opengfx1030/vllm-rdna/commit/00fcc97ff5015158da93f52bd32f2c70757f0f8a
  - https://github.com/opengfx1030/vllm-rdna/commit/611063c62a400ba96e15e8168e2b2418494fdfdd
  - https://github.com/opengfx1030/vllm-rdna/commit/a43be93b10f921bae65e047b23d4ba1d33d6aa44
  - https://github.com/opengfx1030/vllm-rdna/commit/64d59b8942d3b982f1d91f0f3e625dfda4418106
  - https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_allreduce.cu
  - https://github.com/opengfx1030/vllm-rdna/commit/b54e626104e248be16eb59ffe916116170a069f9
  - https://github.com/opengfx1030/vllm-rdna/commit/9aa08d40a66b180ca61ee1133e464618a0b0a463
  - https://github.com/opengfx1030/vllm-rdna/commit/03287a53f0b895be66adef219c1e8ed97035ed81
  - https://github.com/opengfx1030/vllm-rdna/commit/19a010b476a82455aa3b54ef63b91f6e5fa9d3f4
  - https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/csrc/rocm/rdna_allreduce.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/tools/rdna/recipes/flashnext-mtp2.env
---

# gfx1030 push one-shot all-reduce (`rdna_ar`) — silicon lock

## Update at dest tip `c4a05415` (checked 2026-11-02)

`rdna_ar` now also carries two decode-step collectives besides the all-reduce. The all-reduce math, gate, dtypes, buffers and wait bound below are unchanged.[^ar-c4a0-19a0]

| Item | State at `c4a05415` |
|---|---|
| Shared exchange | Phases 1-2 (push into every peer's Uncached staging, posted flag store, local flag wait, abort handling) are one device function, `rdna_ar_exchange`, and each collective is a finisher over it. *(Historical @ `b7402893`: the push and wait lived inline in `rdna_ar_oneshot`.)*[^ar-c4a0-cuh] |
| All-gather | `rdna_ag_oneshot`: last-dim gather, `[rows, cols]` to `[rows, world*cols]` with rank j at columns `[j*cols, (j+1)*cols)`. It is a pure copy, so the result matches RCCL's `all_gather_into_tensor` layout bit for bit.[^ar-c4a0-cuh] |
| Vocab argmax | `rdna_ar_argmax_oneshot`: each rank pushes one 8-byte `(float value, int index)` pair per row (`RdnaArPair`); the finisher keeps `torch.argmax` semantics (NaN wins, ties go to the lowest rank) and writes int64 indices.[^ar-c4a0-cuh] |
| Eligibility | Both use the all-reduce one-shot gate (`rdna_ar_fits`: staged bytes ≤ min(`max_bytes`, `oneshot_max`), so 64 KiB by default). All-gather needs a contiguous fp16/bf16/fp32 tensor; argmax needs 1-D contiguous values plus int64 indices of the same length, sized by `sizeof(RdnaArPair)`.[^ar-c4a0-cu] |
| Launch / LDS | Same launch shape as the all-reduce (`<<<nblocks, 256>>>`, no dynamic LDS); no DOT, no tile, no new `__launch_bounds__`.[^ar-c4a0-cu] |
| Gate | New env `VLLM_RDNA_AR_GATHER`, read with `os.environ` and default `"1"`; `0` keeps all-gather and argmax on RCCL. Both routes also need the backend enabled, and `VLLM_RDNA_AR` still defaults to `"0"`, so a default serve does not take them.[^ar-c4a0-py][^ar-c4a0-envs] |
| Verification claimed | The commit message reports bit-exact results against a reference for worlds 2 and 4 over fp16/bf16/fp32, ties, NaN and CUDA-graph replay on 4x V620. This page does not copy its latency figures.[^ar-c4a0-19a0] |

## Update at dest tip `b7402893` (checked 2026-10-29)

The only `rdna_allreduce.cu` change from `8015ea48` to `b7402893` is the peer-access error order in native init (PR #142). Protocol, gate, dtypes, buffers and wait bound in the next section are unchanged.[^ar-b740-cmp]

| Item | State at `b7402893` |
|---|---|
| Failed peer access | `hipGetLastError()` is now called right after `hipDeviceEnablePeerAccess` and **before** the `TORCH_CHECK` that can throw, "so the RCCL fallback does not inherit a failed peer-access probe". *(Historical @ `1dc3bac4`…`8015ea48`: the clear ran after the check, so it only covered "already enabled"; a real failure threw with the error still in HIP's sticky last-error slot.)*[^ar-b740-cu] |
| How it landed | `b54e6261` first consumed the error from Python via `ctypes` (`libamdhip64.so.7`), with the comment that it "would surface at torch's next launch as `invalid device ordinal` and kill worker init after the graceful disable"; `9aa08d40` moved the fix into the `.cu` and removed the Python workaround (net Python change zero).[^ar-b54e][^ar-9aa0] |
| Silicon impact | None on the kernel: no launch, LDS, Uncached-staging or flag-page change. It only matters when peer access is refused and the backend falls back to RCCL.[^ar-b740-cu] |

## Current state at dest tip `1dc3bac4` (checked 2026-10-11; unchanged at `64d59b89`, rechecked 2026-10-12)

This section supersedes the two-shot and persist-out rows further down; those are kept as history. `csrc/rocm/rdna_allreduce.cu` / `.cuh` have no change from `1dc3bac4` to tip `64d59b89`.[^c-64d59b89-ar]

| Item | State at `1dc3bac4` |
|---|---|
| Algorithm | **One-shot push only.** The two-shot kernel (`rdna_ar_twoshot`) and its selection were removed in `611063c6`: it returned zeros above the one-shot gate and made the boot self-test disable the whole backend. `rdna_ar_can()` also caps at the one-shot gate, so the kernel never sees a larger message.[^c-611063c6][^dest-envs-1dc3][^dest-cuh-1dc3][^dest-py-1dc3] |
| Gate | **Runtime setting: `VLLM_RDNA_AR_ONESHOT_KB`** (KiB, default 64), read by the host at init and by the Python gate; eligible messages are ≤ min(`VLLM_RDNA_AR_MAX_KB`, `VLLM_RDNA_AR_ONESHOT_KB`), both 64 KiB by default; larger messages go to RCCL.[^dest-cu-env-1dc3][^dest-py-gate-1dc3][^dest-envs-1dc3] `RDNA_AR_ONESHOT_MAX` (65536) is a **compile-time constant** in `rdna_allreduce.cuh`: it is only the default that `VLLM_RDNA_AR_ONESHOT_KB` overrides, not a setting you can change at serve time.[^dest-cuh-1dc3] |
| Dtypes | fp16, bf16, fp32 (`at::kHalf`, `at::kFloat`, `at::kBFloat16`).[^dest-envs-1dc3][^dest-cu-elig-1dc3] |
| Recipe gate (`b30757c2`) | The `flashnext-mtp2` recipe sets `VLLM_RDNA_AR_MAX_KB=256` and `VLLM_RDNA_AR_ONESHOT_KB=256` (one-shot up to 256 KiB, RCCL above); commit `641e0c22` raised it from 64 KiB because MTP2 verify batches exceed 64 KiB. The library default `_ONESHOT_DEFAULT_KB` is still 64, and `docs/rdna2/tuning.md` says the other recipes stay at 64 until measured.[^ar-b307-recipe][^ar-b307-py][^ar-b307-tuning] |
| Two-shot requests | `VLLM_RDNA_AR_ALGO=twoshot`, or a `MAX_KB` above the one-shot gate, is ignored with one warning.[^dest-py-1dc3] |
| Output buffer | Each call returns a **fresh output tensor** (`at::empty_like`). Since `00fcc97f` (PR #47), the old shared persistent buffer (`rdna2_persist_zeros`) is gone: it was clobbered by the next all-reduce while callers still held the result (the embedding output kept as the decoder residual).[^extras-rdna-allreduce-cu][^dest-00fcc97f] |
| Wait bound | Each one-shot wait is bounded in wall-clock time by `VLLM_RDNA_AR_WAIT_MS` (default `RDNA_AR_WAIT_MS` = 120000 ms), replacing the old poll-count cap (`VLLM_RDNA_AR_SPIN_CAP` / `RDNA_AR_SPIN_CAP` no longer exist at the tip); waits over `RDNA_AR_SLOW_MS` (2000 ms) that complete are logged as a late peer. This landed in `a43be93b`, which also traced the old "peer flag never arrived" wedge to rank skew rather than a lost P2P write: the 2e6-poll cap was only ~0.8 s on gfx1030. `a43be93b` left two-shot on `SPIN_CAP` polls; `611063c6` then removed two-shot and the `SPIN_CAP` knob.[^dest-cuh-wait-1dc3][^dest-py-1dc3][^c-a43be93b][^c-611063c6] |
| Default | Off: `VLLM_RDNA_AR` defaults to `"0"` and is not implied by `VLLM_FORCE_CUSTOM_ALL_REDUCE`.[^dest-envs-1dc3] |

## Historical: extras lock 2026-09-28 — tip `bfd5286d` (`101a16c8` / prior `22d6e346`+`c782696a`)

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. Occupancy still first (FA prefill leftover). No tok/s.

| Item | Lock |
|---|---|
| `RDNA_AR_ONESHOT_MAX` | **65536** (64 KiB) compile-time default in `csrc/rocm/rdna_allreduce.cuh` — was 32768; env-less callers stay one-shot ≤ 64 KiB. Runtime override is `VLLM_RDNA_AR_ONESHOT_KB` (still true at `1dc3bac4`). [^dest-cuh-1dc3][^dest-cu-env-1dc3] |
| Default path | *(historical; two-shot removed in `611063c6`, first locked at `1dc3bac4`)*[^c-611063c6] One-shot ≤ 64 KiB, RCCL above. Two-shot only if explicitly opted in (wide `VLLM_RDNA_AR_MAX_KB` without matching oneshot / algo override). Serve scripts + library `max_kb` default back to **64**. |
| Two-shot kernel | *(historical; removed in `611063c6`, first locked at `1dc3bac4`)*[^c-611063c6] Was present (`rdna_ar_twoshot`: push reduce-scatter + allgather, abort phases 3/4, bf16, flag row `RDNA_AR_FLAG_STRIDE`). |
| Two-shot race | *(historical; kernel removed in `611063c6`, first locked at `1dc3bac4`)*[^c-611063c6] Peer-flag handshake wedges under co-tenant PCIe load (`rdna_ar_check` / wedge marker → RCCL). **Later/rework** — not dest-safe. Do not re-arm wide gate until phases 2/4 fixed. |
| Gate | Still opt-in `VLLM_RDNA_AR=1`; serve default stays PYNCCL. No DOT / FA `__launch_bounds__` / LDS tile. |

Prior protocol lock (Uncached staging, VRAM flags, on-device seq, T44b abort, push fanout) below still holds for the one-shot path.

---

Date: **2026-09-18**. Dest tip: `opengfx1030/vllm-rdna` `rdna_extras` @ `3b59ee16` (PR #13 leap T44b). Prior silicon lock tip: `a4060647`. Occupancy still first (FA prefill leftover). Do **not** copy tok/s.[^commit-3b59ee16][^commit-a4060647]

Companions: [rccl-p2p.md](rccl-p2p.md) (why stock custom AR is off), [leapdragon.md](../external-projects/leapdragon.md) (push vs pull GB/s, Uncached), [graph-capture.md](../hip-craft/graph-capture.md) (no frozen seq in kernarg), [plx-p2p-mmio.md](plx-p2p-mmio.md) (root-complex burst / fabric knobs).

## Status

| Item | Lock |
|---|---|
| Files | `csrc/rocm/rdna_allreduce.cuh` (kernel), `csrc/rocm/rdna_allreduce.cu` (host), `ops.h` / `torch_bindings.cpp`, `vllm/distributed/device_communicators/rdna_all_reduce.py`, hook in `cuda_communicator.py` [^extras-rdna-allreduce-cu][^extras-rdna-all-reduce-py] |
| CMake | `rdna_allreduce.cu` in base `VLLM_ROCM_EXT_SRC` (not the gfx1030-only append list) |
| Gate | **Opt-in.** `VLLM_RDNA_AR=1` + world 2..8. Default env remains `"0"`. Serve default stays PYNCCL / stock path; `VLLM_FORCE_CUSTOM_ALL_REDUCE` does **not** arm this. |
| Attribution | Protocol / abort-record / wedge marker from leapdragon/vllm-rdna2-qwen T44/T44b (Aron Hsiao). Persist-out (`rdna2_persist_zeros`; *historical*, replaced by a fresh output per call in `00fcc97f`), integer device index, PIX logging, default-off are this fork. |
| DOT / LDS / launch_bounds | **None of the FA leftover.** Tiny `__shared__ int` only (seq + abort). No `fdot2`, no tile pad, no `__launch_bounds__`. |

## Protocol (T44b @ tip `3b59ee16`)

Keeps the WS2 / leapdragon findings, with the 2026-08-30 / T44b flag + abort updates:

1. **Push, never pull** — each rank writes its slice into every peer’s staging slot (`src = our rank`). Peer order is rank-staggered: `j = (rank + k) % world`.
2. **Staging is Uncached** — `hipExtMallocWithFlags(..., hipDeviceMallocUncached)`. Coarse-grained peer stores otherwise leave the owner reading stale L2.
3. **Flags live in each rank’s OWN uncached device memory** — 4 KiB flag page appended to the same Uncached staging alloc (`RDNA_AR_FLAG_PAGE`). Peer announce is one posted P2P store into `peers.flags[j][rank]`; we poll **local** uncached slots (no PCIe read traffic on the wait). Host-coherent `shm_open` flag page is **gone** (signature keeps `shm_name` but ignores it).[^m-78e84a272cf-csrc-rocm-rdna-allreduce-cuh]
4. **Seq on-device** — `seqbuf` atomic load+1 inside the kernel; never a kernarg (graph-capture safe). Dual parity staging: `stage[(parity * W + src) * max_elems + i]`.
5. **Bounded wait + T44b abort record** — *(historical at `3b59ee16`: `VLLM_RDNA_AR_SPIN_CAP`, default `RDNA_AR_SPIN_CAP` 2e6 polls; neither name exists at `1dc3bac4`, where the bound is wall-clock `VLLM_RDNA_AR_WAIT_MS`, default 120000 ms[^dest-cuh-wait-1dc3])*; first block to hit the bound `atomicCAS`es device `timeout` and writes a 64-bit code into a **host-mapped** `report` (plain store, no PCIe atomics). Python `rdna_ar_timeout_info` / `rdna_ar_check()` read with a plain host load (no D2H sync). Poll backoff: `__builtin_amdgcn_s_sleep(8)`. (ISA: ≈ 448–512 clocks out of arbitration; slot stays reserved — [wave-priority-occupancy.md](../occupancy/wave-priority-occupancy.md).)
6. **Wedge marker** — on abort, write `VLLM_CACHE_ROOT/rdna_ar_wedged` and fail the step; next boot stays on RCCL until the marker is deleted.
7. **Fixed-order fp32 reduce** — rank 0 .. W−1 accumulate in `float`, then store `T` (`half` or `float`). Bit-identical across ranks for TP.

Abort code layout (bits) at `1dc3bac4`: `0-7` aborted flag; `8-11` phase (`1` = own grid barrier, `2` = peer flag missing); `12-15` peer; `16-31` measured wall-clock wait in 16 ms units (saturating); `32-63` sequence. *(At `3b59ee16`, bits 16-31 were spins/1024, labelled "~ms"; historical.)*[^dest-cuh-abort-1dc3]

World max: `RDNA_AR_MAX_WORLD = 8`. Launch: **256** threads[^dest-cu-launch-1dc3]; blocks auto by payload size, capped by `VLLM_RDNA_AR_BLOCKS`. `VLLM_RDNA_AR_PACE` (0..127) paces strided pushes. `VLLM_RDNA_AR_MAX_KB` default **64** (decode-sized; prefill stays on RCCL/CUSTOM).

Eligibility at `1dc3bac4`: contiguous fp16 / bf16 / fp32, `nbytes ≤` the one-shot gate (min of `VLLM_RDNA_AR_MAX_KB` and `VLLM_RDNA_AR_ONESHOT_KB`).[^dest-cu-elig-1dc3][^dest-py-gate-1dc3] Larger / other dtypes stay on stock RCCL path. When armed, eligible tensors dispatch **ahead of** stock CUSTOM / PYNCCL.


```mermaid
flowchart TB
  T["all_reduce(tensor)"] --> G{"VLLM_RDNA_AR=1,<br/>world 2..8,<br/>no wedge marker?"}
  G -- "no (serve default)" --> R["stock path: PYNCCL / RCCL"]
  G -- "yes" --> E{"contiguous fp16/bf16/fp32<br/>and nbytes ≤ min(VLLM_RDNA_AR_MAX_KB,<br/>VLLM_RDNA_AR_ONESHOT_KB), both default 64 KiB?<br/>(RDNA_AR_ONESHOT_MAX = compile-time default only)"}
  E -- "no" --> R
  E -- "yes" --> O["one-shot push"]
  O --> P1["push own slice into every peer's<br/>Uncached staging (rank-staggered)"]
  P1 --> P2["posted P2P flag store;<br/>poll local uncached flags"]
  P2 --> P3["fixed-order fp32 reduce rank 0..W-1"]
  P2 -- "VLLM_RDNA_AR_WAIT_MS bound hit" --> AB["abort record + wedge marker<br/>next boot stays on RCCL"]
```

*Diagram: Dispatch split and the one-shot protocol as written in the Status, Protocol and eligibility paragraphs above (tip `3b59ee16`), updated to dest `1dc3bac4`: one-shot only; the gate is the runtime env `VLLM_RDNA_AR_ONESHOT_KB` (capped by `VLLM_RDNA_AR_MAX_KB`), both 64 KiB by default, RCCL above; `RDNA_AR_ONESHOT_MAX` is only the compiled-in default. Two-shot no longer exists.*[^extras-rdna-allreduce-cu][^commit-3b59ee16][^dest-cuh-1dc3]

## What this is not

- Not a replacement for FA occupancy work.
- Not the production default on V620 TP=4 PIX — leave `VLLM_RDNA_AR=0`; PYNCCL remains the dest-safe Uncached-P2P backend for mixed prefill/decode.
- Not XGMI / Instinct custom AR; vLLM’s gfx94/95 custom + QuickReduce stay off on V620 ([rccl-p2p.md](rccl-p2p.md)).
- Do not paste T44 µs figures or leapdragon tok/s into dest claims; cite source comments only when needed, and keep our own benches separate.

## Take / Leave

| | |
|---|---|
| **Take** | Uncached staging + **VRAM uncached flags** + on-device seq + push fanout + fixed-order reduce + T44b host-mapped abort record / `rdna_ar_check()` / wedge marker. Fabric knobs (`BLOCKS` / `PACE`; wall-clock `WAIT_MS` replaced `SPIN_CAP`). Library default `MAX_KB=64` (the `flashnext-mtp2` recipe sets 256 since `641e0c22`, locked at `b30757c2`[^ar-b307-recipe]). Fresh output per call (persist-out is historical since `00fcc97f`). |
| **Leave** | Default-on; host-coherent flag page; mid-serve silent spin-to-cap without wedge fail; any claimed µs/GB/s until remeasured on this 4×/8× 88096 box; auto-arming because P2P is on. |

## Sources

- opengfx1030/vllm-rdna commits `611063c6` (two-shot removal), `a43be93b` (wall-clock one-shot wait), `00fcc97f` (fresh output); tip `64d59b89` (read 2026-10-12)

Primary sources for this page, cited inline as footnotes:

[^extras-rdna-allreduce-cu]: `csrc/rocm/rdna_allreduce.cu`, opengfx1030/vllm-rdna rdna_extras at 1dc3bac4 (2026-10-11 tip); the page may describe an older commit named inline. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cu
[^commit-3b59ee16]: opengfx1030/vllm-rdna commit `3b59ee16e553`. https://github.com/opengfx1030/vllm-rdna/commit/3b59ee16e553168b1e3809c9a141088662171448 (exists, verified via GitHub API 2026-10-11)
[^commit-a4060647]: opengfx1030/vllm-rdna commit `a4060647cfbb`. https://github.com/opengfx1030/vllm-rdna/commit/a4060647cfbb32d3b907eba6018501bb8b251d22 (exists, verified via GitHub API 2026-10-11)
[^extras-rdna-all-reduce-py]: `vllm/distributed/device_communicators/rdna_all_reduce.py`, opengfx1030/vllm-rdna rdna_extras at 1dc3bac4 (2026-10-11 tip); the page may describe an older commit named inline. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/distributed/device_communicators/rdna_all_reduce.py
[^m-78e84a272cf-csrc-rocm-rdna-allreduce-cuh]: github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cuh. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cuh (checked 2026-10-11)
[^dest-envs-1dc3]: `vllm/envs.py`, `VLLM_RDNA_AR` comment block, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/envs.py#L1469-L1476
[^dest-cuh-1dc3]: `csrc/rocm/rdna_allreduce.cuh`, `RDNA_AR_ONESHOT_MAX` and its comment, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cuh#L58-L70
[^dest-py-1dc3]: `vllm/distributed/device_communicators/rdna_all_reduce.py` module docstring, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/distributed/device_communicators/rdna_all_reduce.py#L15-L30
[^dest-00fcc97f]: opengfx1030/vllm-rdna commit 00fcc97f "fix(rocm): rdna_ar returns a fresh output tensor per all-reduce" (PR #47; an ancestor of 1dc3bac4). https://github.com/opengfx1030/vllm-rdna/commit/00fcc97ff5015158da93f52bd32f2c70757f0f8a
[^dest-cu-env-1dc3]: `csrc/rocm/rdna_allreduce.cu` `getenv("VLLM_RDNA_AR_ONESHOT_KB")` overriding `oneshot_max = RDNA_AR_ONESHOT_MAX`, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cu#L84-L93
[^dest-py-gate-1dc3]: `resolve_gate_kb()` in `vllm/distributed/device_communicators/rdna_all_reduce.py` (`VLLM_RDNA_AR_ONESHOT_KB`, `_ONESHOT_DEFAULT_KB = 64`), opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/vllm/distributed/device_communicators/rdna_all_reduce.py#L52-L80
[^dest-cuh-wait-1dc3]: `RDNA_AR_WAIT_MS` / `RDNA_AR_SLOW_MS` and their comment in `csrc/rocm/rdna_allreduce.cuh`, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cuh#L51-L65
[^dest-cuh-abort-1dc3]: abort-record comment and `rdna_ar_ms_code` in `csrc/rocm/rdna_allreduce.cuh`, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cuh#L72-L96
[^dest-cu-launch-1dc3]: `rdna_ar_oneshot<T><<<nblocks, 256, 0, stream>>>` in `csrc/rocm/rdna_allreduce.cu`, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cu#L196
[^dest-cu-elig-1dc3]: dtype check (`kHalf` / `kFloat` / `kBFloat16`) and `at::empty_like` output in `csrc/rocm/rdna_allreduce.cu`, opengfx1030/vllm-rdna at 1dc3bac4. https://github.com/opengfx1030/vllm-rdna/blob/1dc3bac4cda3201a1b4fb345e044578e84a272cf/csrc/rocm/rdna_allreduce.cu#L215-L231
[^c-611063c6]: opengfx1030/vllm-rdna commit [`611063c6`](https://github.com/opengfx1030/vllm-rdna/commit/611063c62a400ba96e15e8168e2b2418494fdfdd) (2026-10-09) — "fix(rdna): remove the two-shot all-reduce from rdna_ar". Deletes `rdna_ar_twoshot`, `VLLM_RDNA_AR_ALGO` twoshot selection and `RDNA_AR_SPIN_CAP` / `VLLM_RDNA_AR_SPIN_CAP`; message: two-shot "returned zeros above the one-shot gate", `twoshot` / larger `MAX_KB` "ignored with one warning", `rdna_ar_can()` caps at the one-shot gate.
[^c-a43be93b]: opengfx1030/vllm-rdna commit [`a43be93b`](https://github.com/opengfx1030/vllm-rdna/commit/a43be93b10f921bae65e047b23d4ba1d33d6aa44) (2026-10-09) — "fix(rdna): rdna_ar one-shot waits tolerate late peers (wall-clock bound)". Message: root cause is rank skew; the 2e6-poll cap "on gfx1030 is only ~0.8 s"; waits use `wall_clock64` at `hipDeviceAttributeWallClockRate`; `VLLM_RDNA_AR_WAIT_MS` default 120 s; "Two-shot is untouched (still VLLM_RDNA_AR_SPIN_CAP polls)". An ancestor of `611063c6`.
[^c-64d59b89-ar]: opengfx1030/vllm-rdna tip [`64d59b89`](https://github.com/opengfx1030/vllm-rdna/commit/64d59b8942d3b982f1d91f0f3e625dfda4418106); `git diff --stat 1dc3bac4cda3201a1b4fb345e044578e84a272cf 64d59b8942d3b982f1d91f0f3e625dfda4418106 -- csrc` lists only `csrc/rocm/moe_w4a8_rdna2.cu`.
[^ar-b740-cmp]: Compare [`8015ea48...b7402893`](https://github.com/opengfx1030/vllm-rdna/compare/8015ea485b7f6c0298d8f98ce6e824ce1c444c45...b74028935393d1e21f73f80f65f506c9ea4639af): `csrc/rocm/rdna_allreduce.cu` +3/−4 lines, `.cuh` unchanged; PR #142 merged as [`03287a53`](https://github.com/opengfx1030/vllm-rdna/commit/03287a53f0b895be66adef219c1e8ed97035ed81).
[^ar-b740-cu]: [rdna_allreduce.cu:114–125](https://github.com/opengfx1030/vllm-rdna/blob/b74028935393d1e21f73f80f65f506c9ea4639af/csrc/rocm/rdna_allreduce.cu#L114-L125) @ b74028935393d1e21f73f80f65f506c9ea4639af.
[^ar-b54e]: Commit [`b54e6261`](https://github.com/opengfx1030/vllm-rdna/commit/b54e626104e248be16eb59ffe916116170a069f9) "fix(rdna2): consume HIP sticky error after failed peer-access enable" (`vllm/distributed/device_communicators/rdna_all_reduce.py` +10).
[^ar-9aa0]: Commit [`9aa08d40`](https://github.com/opengfx1030/vllm-rdna/commit/9aa08d40a66b180ca61ee1133e464618a0b0a463) "fix(rdna2): clear peer-access error before native init throws" (`rdna_allreduce.cu` +3/−4, `rdna_all_reduce.py` −10).
[^ar-c4a0-19a0]: opengfx1030/vllm-rdna commit [`19a010b4`](https://github.com/opengfx1030/vllm-rdna/commit/19a010b476a82455aa3b54ef63b91f6e5fa9d3f4) (2026-11-01) "feat(rdna2): HIP one-shot all-gather and fused vocab argmax on rdna_ar"; an ancestor of dest tip `c4a05415cbda030c784bdc91c7d32e55c287a828`.
[^ar-c4a0-cuh]: `csrc/rocm/rdna_allreduce.cuh` L198 (`rdna_ar_exchange`), L334 (all-reduce finisher), L360-L388 (`rdna_ag_oneshot`), L392-L440 (`RdnaArPair`, `rdna_ar_pair_wins`, `rdna_ar_argmax_oneshot`) @ c4a05415cbda030c784bdc91c7d32e55c287a828. https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/csrc/rocm/rdna_allreduce.cuh#L198-L440
[^ar-c4a0-cu]: `csrc/rocm/rdna_allreduce.cu` L281-L282 (`rdna_ar_fits`), L286-L304 (launches, 256 threads), L311-L316 (`rdna_ar_can_all_gather`), L353-L362 (`rdna_ar_can_argmax`) @ c4a05415cbda030c784bdc91c7d32e55c287a828. https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/csrc/rocm/rdna_allreduce.cu#L281-L362
[^ar-c4a0-py]: `vllm/distributed/device_communicators/rdna_all_reduce.py` L150-L152 (`VLLM_RDNA_AR_GATHER`, default "1") @ c4a05415cbda030c784bdc91c7d32e55c287a828. https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/vllm/distributed/device_communicators/rdna_all_reduce.py#L150-L152
[^ar-c4a0-envs]: `vllm/envs.py` L184 and L1511 (`VLLM_RDNA_AR` default "0") @ c4a05415cbda030c784bdc91c7d32e55c287a828. https://github.com/opengfx1030/vllm-rdna/blob/c4a05415cbda030c784bdc91c7d32e55c287a828/vllm/envs.py#L1511

[^ar-b307-recipe]: `tools/rdna/recipes/flashnext-mtp2.env` @ `b30757c2` (L38-L44): https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/tools/rdna/recipes/flashnext-mtp2.env#L38-L44 ; commit [`641e0c22`](https://github.com/opengfx1030/vllm-rdna/commit/641e0c2288db9aae34b74428632b80bc8643adfb) "perf(rdna2): flashnext-mtp2 one-shot all-reduce gate 64 -> 256 KiB" (in compare https://github.com/opengfx1030/vllm-rdna/compare/59b0f5eb227998194c5ac8ca7bebf555f1e8ae32...b30757c257c25c737d671179025bbc5c2ec1ee50).
[^ar-b307-py]: `_ONESHOT_DEFAULT_KB = 64` and `resolve_gate_kb()` @ `b30757c2`: https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/vllm/distributed/device_communicators/rdna_all_reduce.py#L58-L80
[^ar-b307-tuning]: `docs/rdna2/tuning.md` all-reduce env table @ `b30757c2`: https://github.com/opengfx1030/vllm-rdna/blob/b30757c257c25c737d671179025bbc5c2ec1ee50/docs/rdna2/tuning.md
