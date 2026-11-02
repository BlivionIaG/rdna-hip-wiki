---
title: "Device-side spin waits vs GPU_MAX_HW_QUEUES=2 (gfx1030 / V620)"
sources:
  - https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_k2_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/rdna_allreduce.cuh
  - https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/rdna_allreduce.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/hc_prefill_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_gdn_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_qsa_decode_rdna2.cu
  - https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/vllm/models/qwen4_exp/amd/mono_decode_rdna2.py
  - https://github.com/opengfx1030/vllm-rdna/commit/bbe36aab973a41b56b9e70ab45a5d647c0bcee86
  - https://github.com/opengfx1030/vllm-rdna/commit/7860e9b4ec2a9ef301fc4ff4e9beba913429d452
  - https://github.com/ROCm/TheRock/issues/9306
---

# Device-side spin waits vs `GPU_MAX_HW_QUEUES=2` (gfx1030 / V620)

Lock: **At `rdna_extras` tip `5bda4c245024508f8e9ce59c43c5387c98b61228`, none of the fork's device-side spin waits form a same-GPU cross-stream producer/waiter pair that `GPU_MAX_HW_QUEUES=2` can serialize into a deadlock.** Same-kernel spins are either co-resident-capped or last-arriver (no wait). Peer-flag spins wait on another GPU. The `=2` vs `=4` choice still matters for CLR queue sharing (MonoKernel TP skew) and for the separate TheRock#9306 MEC cross-queue stall; those are not spin-deadlock of these sites.[^tip-5bda4c24]

No tok/s. No ms. Do not restate the FA pin. Do not bump dest / UNC from this page.

Date: **2026-11-02** Europe/Paris. Read tip: `opengfx1030/vllm-rdna` `rdna_extras` @ `5bda4c245024508f8e9ce59c43c5387c98b61228`.

Companions: [fence-occupancy.md](fence-occupancy.md) (HC ticket fence shape), [wave-priority-occupancy.md](wave-priority-occupancy.md) (`s_sleep` backoff), [rdna-allreduce.md](../interconnect/rdna-allreduce.md), [toolchain/runtime/README.md](../../toolchain/runtime/README.md) (CLR `GPU_MAX_HW_QUEUES`, TheRock#9306 / MEC), [kernels/persistent/monokernel-gfx1030-dest.md](../../kernels/persistent/monokernel-gfx1030-dest.md).

## Take / Leave

| | |
|---|---|
| **Take** | Treat a device spin as a **hwq hazard** only when the producer is a **different kernel or stream on the same GPU** that CLR could place behind the waiter on a shared HSA queue. Same-kernel co-resident waits and cross-GPU peer-flag waits are not that class.[^tip-5bda4c24] |
| **Take** | MonoKernel `grid_barrier` and fused Stage-D peer wait at `5bda4c24`: **safe** under `=2` for spin-deadlock. Barrier spins same-grid; grid is capped to the residency limit and launched on one stream. Peer wait is cross-rank (other GPU).[^mono-coreside][^mono-barrier][^mono-peer][^mono-launch] |
| **Take** | `rdna_ar` / all-gather / fused argmax exchange at `5bda4c24`: **safe** under `=2` for spin-deadlock. Intra-rank `arrive[p]` spin is same-kernel (block 0 waits for the other blocks of that launch; width 4–32). Peer-flag spin is cross-GPU. Launch is on the current stream.[^ar-arrive][^ar-peer][^ar-launch] |
| **Take** | HC / small-M split-K ticket (`hc_prefill_rdna2.cu`) and Mono GDN / QSA arrival counters: **safe**. Last-arriver pattern — non-last blocks return; nobody spins on the ticket.[^hc-ticket][^gdn-ticket][^qsa-ticket][^fence-page] |
| **Take** | `flashnext-mtp2` `GPU_MAX_HW_QUEUES=2` (`bbe36aab`) is consistent with this spin reading when `--monokernel` is off. The ABBA result is documented in `7860e9b4`; `--monokernel` still raises workers to 4 for **skew**, not because these spins deadlock at 2.[^bbe36][^7860][^mono-hwq] |
| **Leave** | Do **not** equate TheRock#9306's reporter-claimed `=2` workaround with "spins are unsafe at 4". That issue is idle MEC pipes with pending AQL across queues; the real mitigation called out there is MEC firmware `0x86`. Record `mec_fw_version` on soaks. See [toolchain/runtime/README.md](../../toolchain/runtime/README.md).[^therock-9306] |
| **Leave** | Do **not** open UNC / retip extras / edit `rdna_extras` from this page. Queue recipe and monokernel floor stay with VLLM_FORK_Manager. |
| **Unknown (out of dest path)** | Stock `csrc/rocm/quickreduce/` `wait_sync_flag` spins on peer flags (cross-rank). Not traced against the Flash-Next / `rdna_ar` live path here. (unsourced for dest serve) |

## 1. Hazard rule

CLR maps many HIP streams onto at most `GPU_MAX_HW_QUEUES` HSA queues per priority, then shares by load.[^runtime-hwq] A device wave that spins waiting for a flag written by **another kernel on another stream of the same GPU** deadlocks if that producer packet sits behind the waiter on the same hardware queue: the waiter never yields the CP, so the producer never runs.

Not a hwq hazard:

- Producer and waiter are **blocks of the same launch**, and the grid is **guaranteed co-resident** (or the pattern is last-arriver with no spin).
- Producer is on a **peer GPU** (P2P / system-scope flag). Local HW queues do not order the peer's CP.

## 2. Site table at `5bda4c24`

| Site | File:lines @ `5bda4c24` | Producer | Guard | Verdict vs `=2` |
|---|---|---|---|---|
| MonoKernel `grid_barrier` | `mono_k2_rdna2.cu:289–332` | Same kernel (other WGs bump `ctrl->gen`) | Grid ≤ residency cap (code-object VGPR/LDS × CU); one stream; timeout + sticky err | **safe** |
| MonoKernel Stage D peer flags | `mono_k2_rdna2.cu:2037–2051` | Peer ranks' mono / AR publish | Cross-GPU; `s_sleep(8)`; timeout | **safe** (local hwq) |
| `rdna_ar_exchange` arrive | `rdna_allreduce.cuh:258–268` | Same kernel (`atomicAdd` on `arrive[p]`) | `nblocks` 4–32; block 0 polls; abort on wait budget | **safe** |
| `rdna_ar_exchange` peer flags | `rdna_allreduce.cuh:289–299` | Peer ranks' AR kernels | Cross-GPU; `RDNA_AR_POLL_PAUSE`; sticky abort | **safe** (local hwq) |
| HC split-K ticket | `hc_prefill_rdna2.cu:429–431` | Same kernel (last `atomicAdd`) | Last-arriver; non-last `return`; no spin | **safe** |
| Mono GDN norm arrivals | `mono_gdn_rdna2.cu:366` | Same kernel last tile | Last-arriver; re-arm | **safe** |
| Mono QSA split arrivals | `mono_qsa_decode_rdna2.cu:256` | Same kernel last split | Last-arriver; re-arm | **safe** |
| QuickReduce `wait_sync_flag` | `quickreduce/base.h:366–368` | Peer ranks | Cross-rank spin; not dest-traced here | **unknown** (unsourced for dest) |

## 3. What `=2` vs `=4` still means

| Concern | `=2` | `=4` | Spin-deadlock of §2? |
|---|---|---|---|
| MonoKernel + multi-stream decode under FG=1 | CLR shares queues; side work skews TP ranks (~10 % attention-side on unlucky ranks). Engine raises floor to 4 when `--monokernel` is on.[^mono-hwq] | Removes that sharing skew on the measured 4× V620 setup.[^mono-hwq] | **No** — skew / overlap, not a stuck spin |
| `flashnext-mtp2` without `--monokernel` | Shipped in `bbe36aab` after ABBA preferred TTFT; decode regresses a few percent per `7860e9b4`.[^bbe36][^7860] | Prior recipe value | **No** per §2 |
| TheRock#9306 V620 multi-queue stall | Reporter-claimed workaround (drafter∥verify cross-queue hand-off) | Default / higher counts stalled in that report | **Different failure** (idle MEC, no waves). Prefer MEC `0x86`; keep `=2` as fallback on `0x74`.[^therock-9306] |

## Sources

[^tip-5bda4c24]: opengfx1030/vllm-rdna `rdna_extras` tip [`5bda4c245024508f8e9ce59c43c5387c98b61228`](https://github.com/opengfx1030/vllm-rdna/commit/5bda4c245024508f8e9ce59c43c5387c98b61228) — "rdna: MTP verify glue batch 3 - HC decode tile retune, rdna_gemm_smallm" (read 2026-11-02). Parent of the HWQ docs tip `7860e9b4`.
[^mono-coreside]: `csrc/rocm/mono_k2_rdna2.cu` at `5bda4c24`, L32–L34: plain launch, one stream; grid must be co-resident; WG count is the residency cap (2 WGs/CU, 144 on a V620). https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_k2_rdna2.cu#L32-L34
[^mono-barrier]: Same file, `grid_barrier` poll on `c->gen` with `s_sleep` backoff, L289–L332. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_k2_rdna2.cu#L289-L332
[^mono-peer]: Same file, Stage D peer-flag wait, L2037–L2051. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_k2_rdna2.cu#L2037-L2051
[^mono-launch]: Same file, host launch caps `G` by code-object occupancy × CU count, L2440–L2463. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_k2_rdna2.cu#L2440-L2463
[^ar-arrive]: `csrc/rocm/rdna_allreduce.cuh` at `5bda4c24`, arrive poll L258–L268. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/rdna_allreduce.cuh#L258-L268
[^ar-peer]: Same file, peer-flag poll L289–L299. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/rdna_allreduce.cuh#L289-L299
[^ar-launch]: `csrc/rocm/rdna_allreduce.cu` at `5bda4c24`, `rdna_ar_blocks` 4/16/32 and `getCurrentCUDAStream()` launch, L227–L256. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/rdna_allreduce.cu#L227-L256
[^hc-ticket]: `csrc/rocm/hc_prefill_rdna2.cu` at `5bda4c24`, ticket `atomicAdd` / early return / re-arm, L429–L457. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/hc_prefill_rdna2.cu#L429-L457
[^gdn-ticket]: `csrc/rocm/mono_gdn_rdna2.cu` at `5bda4c24`, L366 / L399. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_gdn_rdna2.cu#L366
[^qsa-ticket]: `csrc/rocm/mono_qsa_decode_rdna2.cu` at `5bda4c24`, L256 / L296. https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/csrc/rocm/mono_qsa_decode_rdna2.cu#L256
[^fence-page]: [fence-occupancy.md](fence-occupancy.md) — HC ticket fences verified correct at `b30757c2` (still the ticket shape at `5bda4c24`).
[^mono-hwq]: `vllm/models/qwen4_exp/amd/mono_decode_rdna2.py` `set_worker_hw_queue_budget`, L343–L369 at `5bda4c24`: `=2` shares queues and skews TP under persistent kernel; raises floor to `hw_queues` (default 4). https://github.com/opengfx1030/vllm-rdna/blob/5bda4c245024508f8e9ce59c43c5387c98b61228/vllm/models/qwen4_exp/amd/mono_decode_rdna2.py#L343-L369
[^bbe36]: Commit [`bbe36aab973a41b56b9e70ab45a5d647c0bcee86`](https://github.com/opengfx1030/vllm-rdna/commit/bbe36aab973a41b56b9e70ab45a5d647c0bcee86) — `perf(rdna2): flashnext-mtp2 GPU_MAX_HW_QUEUES 4 -> 2`.
[^7860]: Commit [`7860e9b4ec2a9ef301fc4ff4e9beba913429d452`](https://github.com/opengfx1030/vllm-rdna/commit/7860e9b4ec2a9ef301fc4ff4e9beba913429d452) — docs the ABBA 4-vs-2 result; notes `--monokernel` stays at 4.
[^therock-9306]: [ROCm/TheRock#9306](https://github.com/ROCm/TheRock/issues/9306) — V620 multi-queue stall; `=2` as reporter workaround; MEC `0x86` mitigation. Summarized in [toolchain/runtime/README.md](../../toolchain/runtime/README.md).
[^runtime-hwq]: [toolchain/runtime/README.md](../../toolchain/runtime/README.md) — clr `GPU_MAX_HW_QUEUES` / `getQueueFromPool` sharing.
