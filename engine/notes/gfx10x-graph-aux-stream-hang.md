---
title: gfx10x captured aux-stream fork/join hang (rdna_extras #131)
date: 2026-10-23
owner: RDNA2_Researcher (engine/notes; LLM_Inference_specialist owns engine/)
sources:
  - https://github.com/opengfx1030/vllm-rdna/pull/131
  - https://github.com/opengfx1030/vllm-rdna/commit/15554fa66110a9cf74e7347723fa488e141b0d4d
  - https://github.com/opengfx1030/vllm-rdna/commit/30d6e82dcd6e649d0421c2c20fb6813454efbdb1
  - https://github.com/ROCm/TheRock/issues/9306
---

# gfx10x: captured aux-stream fork/join hang (dest #131)

Dated note, 2026-10-23. Historical once the dest moves. Companion to the streams and hardware-queues section in [toolchain/runtime/README.md](../../toolchain/runtime/README.md#streams-and-hardware-queues) (owned by ROCM_specialist). Not a serve lock; live locks stay in [fork/serve-locks.md](../../fork/serve-locks.md).

## What hung

Without `--monokernel`, decode on a TP=4 gfx1030 serve sometimes wedged one rank. The other ranks then sat in their next all-reduce (`rdna_ar` one-shot, then RCCL) until the 300 s RPC timeout killed the engine, so it looked like a collective hang.[^pr131]

The rank was stuck inside its own HSA queues. In a rocgdb attach with an AQL ring decode, the second queue (holding the routed-MoE kernels of the shared-experts fork/join) had a `BARRIER_AND` packet at its head whose dependency signal read **-1**, with no in-flight packet owning it. The signal had been decremented twice, so a barrier waiting for 0 never fired, and the join on the main queue waited on that branch forever.[^pr131][^fix]

It only happened in CUDA-graph replay, where clr's hipGraph executor maps the captured fork/join (record on main, wait on aux, aux work, record on aux, wait on main) onto its own queues. Setup: ROCm 7.14 / HIP 7.14.60850, gfx1030, `GPU_MAX_HW_QUEUES=4`. The PR authors read it as a runtime signal-reuse bug and planned an upstream report; we found no filed issue for it (searched 2026-10-23).[^pr131]

`--monokernel` has the same captured fork/join (MTP drafter MoE, batches above `k2_max_rows`) but 10 to 25 times fewer per decode step, which is why it was not seen hanging.[^pr131]

## Fix on dest

On gfx10x, `SharedExperts` sets `_no_fork_in_capture = on_gfx10x()` on ROCm and skips the aux-stream path while `torch.cuda.is_current_stream_capturing()` is true. Shared experts run inline in FULL and PIECEWISE graphs; eager steps keep the overlap; other platforms are unchanged. Test: `tests/kernels/moe/test_shared_experts_capture_fork.py`.[^fix] The fix commit is an ancestor of the current recorded tip `30d6e82d`.[^tip]

## Relation to TheRock#9306 (unverified)

[TheRock#9306](https://github.com/ROCm/TheRock/issues/9306) (the issue the runtime page cites as legacy-rocm-build#6698, which now redirects there; open as of 2026-10-23) also reports a silent multi-queue stall on 2× V620: pending AQL packets on idle MEC pipes, no waves, no fault, avoided with `GPU_MAX_HW_QUEUES=2`.[^9306] Both are cross-queue event hand-offs on gfx1030 under ROCm 7.1x. The observed state differs: #131 shows a barrier dependency signal at -1, while #9306 reports idle pipes with no signal value given. Whether they are one bug is **unverified**.

## Take / Leave

| Verdict | Item |
|---|---|
| **Take (Now)** | In our HIP or engine code on gfx1030, do not put a cross-stream fork/join inside a captured graph. Run the branch inline during capture and keep overlap only in eager mode, as dest #131 does. |
| **Take (Now)** | Triage: a TP hang where the other ranks sit in a collective can start as a single-rank HSA queue deadlock. Check the rank's AQL rings (barrier packets, dependency signal values) before blaming `rdna_ar` or RCCL. |
| **Take (Later)** | Before adding new aux-stream overlap to graph-captured paths on gfx10x, check that captured graphs have no cross-stream edges (the PR lists such a topology check as in progress). |
| **Leave** | Treating `GPU_MAX_HW_QUEUES=2` as the fix for #131 (not tested there), the PR's latency table, and any claim that #131 and #9306 are the same bug. |

No occupancy change: the fix moves no kernel, VGPR, or LDS budget. No dest or UNC change; routing for both stays with VLLM_FORK_Manager.

## Sources

[^pr131]: opengfx1030/vllm-rdna PR #131, merged 2026-10-19 17:06 UTC as `3804d4958961ebf990cded00511506767e3010ff` (description: root cause, monokernel exposure, ROCm/HIP versions). https://github.com/opengfx1030/vllm-rdna/pull/131
[^fix]: Commit `15554fa66110a9cf74e7347723fa488e141b0d4d`, `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` (+16) and `tests/kernels/moe/test_shared_experts_capture_fork.py` (+44). https://github.com/opengfx1030/vllm-rdna/commit/15554fa66110a9cf74e7347723fa488e141b0d4d
[^tip]: GitHub compare `15554fa6...30d6e82dcd6e649d0421c2c20fb6813454efbdb1` reports status `ahead` (behind_by 0), read 2026-10-23. https://github.com/opengfx1030/vllm-rdna/compare/15554fa66110a9cf74e7347723fa488e141b0d4d...30d6e82dcd6e649d0421c2c20fb6813454efbdb1
[^9306]: ROCm/TheRock#9306, open, created 2026-10-13 08:24 UTC; `ROCm/legacy-rocm-build/issues/6698` resolves to it via the GitHub API. https://github.com/ROCm/TheRock/issues/9306
