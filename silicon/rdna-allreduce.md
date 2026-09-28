# gfx1030 push one-shot all-reduce (`rdna_ar`) — silicon lock

## extras lock 2026-09-28 — tip `bfd5286d` (`101a16c8` / prior `22d6e346`+`c782696a`)

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. Occupancy still first (FA prefill leftover). No tok/s.

| Item | Lock |
|---|---|
| `RDNA_AR_ONESHOT_MAX` | **65536** (64 KiB) in `csrc/rocm/rdna_allreduce.cuh` — was 32768; env-less callers stay one-shot ≤ 64 KiB. |
| Default path | One-shot ≤ 64 KiB, RCCL above. Two-shot only if explicitly opted in (wide `VLLM_RDNA_AR_MAX_KB` without matching oneshot / algo override). Serve scripts + library `max_kb` default back to **64**. |
| Two-shot kernel | Present (`rdna_ar_twoshot`: push reduce-scatter + allgather, abort phases 3/4, bf16, flag row `RDNA_AR_FLAG_STRIDE`). |
| Two-shot race | Peer-flag handshake wedges under co-tenant PCIe load (`rdna_ar_check` / wedge marker → RCCL). **Later/rework** — not dest-safe. Do not re-arm wide gate until phases 2/4 fixed. |
| Gate | Still opt-in `VLLM_RDNA_AR=1`; serve default stays PYNCCL. No DOT / FA `__launch_bounds__` / LDS tile. |

Prior protocol lock (Uncached staging, VRAM flags, on-device seq, T44b abort, push fanout) below still holds for the one-shot path.

---

Date: **2026-09-18**. Dest tip: `opengfx1030/vllm-rdna` `rdna_extras` @ `3b59ee16` (PR #13 leap T44b). Prior silicon lock tip: `a4060647`. Occupancy still first (FA prefill leftover). Do **not** copy tok/s.

Companions: [rccl-p2p.md](rccl-p2p.md) (why stock custom AR is off), [leapdragon.md](leapdragon.md) (push vs pull GB/s, Uncached), [graph-capture.md](graph-capture.md) (no frozen seq in kernarg), [plx-p2p-mmio.md](plx-p2p-mmio.md) (root-complex burst / fabric knobs).

## Status

| Item | Lock |
|---|---|
| Files | `csrc/rocm/rdna_allreduce.cuh` (kernel), `csrc/rocm/rdna_allreduce.cu` (host), `ops.h` / `torch_bindings.cpp`, `vllm/distributed/device_communicators/rdna_all_reduce.py`, hook in `cuda_communicator.py` |
| CMake | `rdna_allreduce.cu` in base `VLLM_ROCM_EXT_SRC` (not the gfx1030-only append list) |
| Gate | **Opt-in.** `VLLM_RDNA_AR=1` + world 2..8. Default env remains `"0"`. Serve default stays PYNCCL / stock path; `VLLM_FORCE_CUSTOM_ALL_REDUCE` does **not** arm this. |
| Attribution | Protocol / abort-record / wedge marker from leapdragon/vllm-rdna2-qwen T44/T44b (Aron Hsiao). Persist-out (`rdna2_persist_zeros`), integer device index, PIX logging, default-off are this fork. |
| DOT / LDS / launch_bounds | **None of the FA leftover.** Tiny `__shared__ int` only (seq + abort). No `fdot2`, no tile pad, no `__launch_bounds__`. |

## Protocol (T44b @ tip `3b59ee16`)

Keeps the WS2 / leapdragon findings, with the 2026-08-30 / T44b flag + abort updates:

1. **Push, never pull** — each rank writes its slice into every peer’s staging slot (`src = our rank`). Peer order is rank-staggered: `j = (rank + k) % world`.
2. **Staging is Uncached** — `hipExtMallocWithFlags(..., hipDeviceMallocUncached)`. Coarse-grained peer stores otherwise leave the owner reading stale L2.
3. **Flags live in each rank’s OWN uncached device memory** — 4 KiB flag page appended to the same Uncached staging alloc (`RDNA_AR_FLAG_PAGE`). Peer announce is one posted P2P store into `peers.flags[j][rank]`; we poll **local** uncached slots (no PCIe read traffic on the wait). Host-coherent `shm_open` flag page is **gone** (signature keeps `shm_name` but ignores it).
4. **Seq on-device** — `seqbuf` atomic load+1 inside the kernel; never a kernarg (graph-capture safe). Dual parity staging: `stage[(parity * W + src) * max_elems + i]`.
5. **Bounded spin + T44b abort record** — `VLLM_RDNA_AR_SPIN_CAP` (default `RDNA_AR_SPIN_CAP` 2e6); first block to hit the cap `atomicCAS`es device `timeout` and writes a 64-bit code into a **host-mapped** `report` (plain store, no PCIe atomics). Python `rdna_ar_timeout_info` / `rdna_ar_check()` read with a plain host load (no D2H sync). Poll backoff: `__builtin_amdgcn_s_sleep(8)`.
6. **Wedge marker** — on abort, write `VLLM_CACHE_ROOT/rdna_ar_wedged` and fail the step; next boot stays on RCCL until the marker is deleted.
7. **Fixed-order fp32 reduce** — rank 0 .. W−1 accumulate in `float`, then store `T` (`half` or `float`). Bit-identical across ranks for TP.

Abort code layout (bits): `0-7` aborted flag; `8-11` phase (`1` = own grid barrier, `2` = peer flag missing); `12-15` peer; `16-31` spins/1024 (~ms); `32-63` sequence.

World max: `RDNA_AR_MAX_WORLD = 8`. Launch: **256** threads; blocks auto by payload size, capped by `VLLM_RDNA_AR_BLOCKS`. `VLLM_RDNA_AR_PACE` (0..127) paces strided pushes. `VLLM_RDNA_AR_MAX_KB` default **64** (decode-sized; prefill stays on RCCL/CUSTOM).

Eligibility: contiguous CUDA `half`/`float`, `nbytes ≤ max_bytes`. Larger / other dtypes stay on stock RCCL path. When armed, eligible tensors dispatch **ahead of** stock CUSTOM / PYNCCL.

## What this is not

- Not a replacement for FA occupancy work.
- Not the production default on V620 TP=4 PIX — leave `VLLM_RDNA_AR=0`; PYNCCL remains the dest-safe Uncached-P2P backend for mixed prefill/decode.
- Not XGMI / Instinct custom AR; vLLM’s gfx94/95 custom + QuickReduce stay off on V620 ([rccl-p2p.md](rccl-p2p.md)).
- Do not paste T44 µs figures or leapdragon tok/s into dest claims; cite source comments only when needed, and keep our own benches separate.

## Take / Leave

| | |
|---|---|
| **Take** | Uncached staging + **VRAM uncached flags** + on-device seq + push fanout + fixed-order reduce + T44b host-mapped abort record / `rdna_ar_check()` / wedge marker. Fabric knobs (`BLOCKS` / `PACE` / `SPIN_CAP`). Default `MAX_KB=64`. Persist-out for graph safety. |
| **Leave** | Default-on; host-coherent flag page; mid-serve silent spin-to-cap without wedge fail; any claimed µs/GB/s until remeasured on this 4×/8× 88096 box; auto-arming because P2P is on. |
