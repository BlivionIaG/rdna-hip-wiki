# gfx1030 push one-shot all-reduce (`rdna_ar`) — silicon lock

Date: **2026-09-09**. Dest tip: `opengfx1030/vllm-rdna` `rdna_extras` @ `a4060647` (`[review-only]` cherry-pick leapdragon rdna_ar fabric knobs onto rdna_extras #1). Prior tip: `d71721c7`. Occupancy still first (FA prefill leftover). Do **not** copy tok/s.

Companions: [rccl-p2p.md](rccl-p2p.md) (why stock custom AR is off), [leapdragon.md](leapdragon.md) (push vs pull GB/s, Uncached), [graph-capture.md](graph-capture.md) (no frozen seq in kernarg), [plx-p2p-mmio.md](plx-p2p-mmio.md) (root-complex burst / fabric knobs).

## Status

| Item | Lock |
|---|---|
| Files | `csrc/rocm/rdna_allreduce.cuh` (kernel), `csrc/rocm/rdna_allreduce.cu` (host), `ops.h` / `torch_bindings.cpp`, `vllm/distributed/device_communicators/rdna_all_reduce.py`, hook in `cuda_communicator.py` |
| CMake | `rdna_allreduce.cu` added to base `VLLM_ROCM_EXT_SRC` (not the gfx1030-only append list) |
| Gate | **Opt-in.** `VLLM_RDNA_AR=1` + `on_gfx10x()` + world 2..8. Default env is `"0"` (wiring in `cuda_communicator.py`). Module docstring that says “enabled by default” is wrong vs the gate. |
| Review | Commit subject is `[review-only]` — on dest tip, not a silent occupancy bump. |
| DOT / LDS / launch_bounds | **None of the FA leftover.** Kernel uses tiny `__shared__ int` only (seq + abort). No `fdot2`, no tile pad, no `__launch_bounds__`. |

## Protocol (from `.cuh`)

Keeps the WS2 / leapdragon findings:

1. **Push, never pull** — each rank writes its slice into every peer’s staging slot (`src = our rank`). Peer order is rank-staggered: `j = (rank + k) % world`.
2. **Staging is Uncached** — `hipExtMallocWithFlags(..., hipDeviceMallocUncached)`. Coarse-grained peer stores otherwise leave the owner reading stale L2.
3. **Flags are host-coherent** — 4096 B `shm_open` page, `hipHostRegisterMapped | Portable`, device pointer via `hipHostGetDevicePointer`. Device-memory flags cannot be polled across PCIe.
4. **Seq on-device** — `seqbuf` atomic load+1 inside the kernel; never a kernarg (graph-capture safe). Dual parity staging: `stage[(parity * W + src) * max_elems + i]`.
5. **Bounded spin** — `RDNA_AR_SPIN_CAP` (2e6); sticky `*timeout` + shared abort instead of hanging the GPU.
6. **Fixed-order fp32 reduce** — rank 0 .. W−1 accumulate in `float`, then store `T` (`half` or `float`). Bit-identical across ranks for TP.

World max: `RDNA_AR_MAX_WORLD = 8`. Launch: **256** threads; blocks auto by payload size (`≤8 KiB → 4`, `≤32 KiB → 16`, else `32`), capped by `VLLM_RDNA_AR_BLOCKS`. `VLLM_RDNA_AR_PACE` (0..127) inserts `__builtin_amdgcn_s_sleep(1)` between strided stores (fabric-friendliness into the receiving root complex). `VLLM_RDNA_AR_MAX_KB` default **512**.

Eligibility: contiguous CUDA `half`/`float`, `nbytes ≤ max_bytes`. Larger / other dtypes stay on stock RCCL path.

## What this is not

- Not a replacement for FA occupancy work.
- Not XGMI / Instinct custom AR; vLLM’s gfx94/95 custom + QuickReduce stay off on V620 ([rccl-p2p.md](rccl-p2p.md)).
- Do not paste T44 µs figures or leapdragon tok/s into dest claims; cite source comments only when needed, and keep our own benches separate.

## Take / Leave

| | |
|---|---|
| **Take** | Uncached staging + host-coherent flags + on-device seq + push fanout + fixed-order reduce. Fabric knobs (`BLOCKS` / `PACE`) for multi-device root-complex courtesy. |
| **Leave** | Default-on marketing in the Python docstring; any claimed µs/GB/s until remeasured on this 4×/8× 88096 box. |
