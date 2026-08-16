# SageAttention on gfx1030 (prefill QK)

Date: 2026-08-17. Locked in GFX1030 Inference. Check this page against a real HIP implementation later. Do not invent tok/s.

Paper: Zhang et al., *SageAttention: Accurate 8-bit attention for Plug-and-Play Inference Acceleration*, [arXiv 2410.02367](https://arxiv.org/abs/2410.02367).

Silicon packing: RDNA2_Researcher. Dispatch: this page + [attention-dispatch.md](attention-dispatch.md). Ticket: existing Sage card on [project 4](https://github.com/users/BlivionIaG/projects/4) — prefill-only, **after occupancy**. No new ticket. Do not edit `perf/rdna2_w4a16` for this.

## Verdict

The only sourced “faster than FA2” path that matches this silicon is **INT8 QK via `sdot4`**, and **only for prefill**.

- QK is compute-bound on fat prefill. IU8 peak is 512 vs FP16 256 FLOPS/clock/CU (GPUOpen RX 6950 XT table).
- PV stays `fdot2` / FP16.
- Decode is GDDR6-bound. INT8 QK does not cut the KV read. Decode stays `fa_rdna2`.
- Live tree: QK is **fdot2**. Sage is absent.

## Algorithm (what to implement, not NVIDIA MMA)

From the paper, keep:

1. Quantize **Q and K to INT8** (not FP8 — `supports_fp8()` is false; no FP8 unit).
2. **Smooth K** (channel-wise outlier removal) before quant. Paper: accuracy fix, <0.2% time on their CUDA kernel — re-measure here.
3. **QK** in INT8, accumulate i32 through K, **scale in the epilogue**.
4. Softmax in FP32 (or the FA2 online-softmax you already have).
5. **PV in FP16** with `fdot2`. Do not quantize P,V to INT8/FP8 on this chip.

From RDNA2_Researcher (2026-08-17):

- Pack D as **4×i8**. Gate `D % 4 == 0`. D=128 and D=256 are fine.
- `sdot4` through the K loop. No `sudot4` (gfx11+).
- Scale in the epilogue, not per-dot.
- Do not import NVIDIA `mma(u8.u8.s32)` or FA3 two-level FP32 accum.

## Do not study / do not port

| Thing | Why |
|---|---|
| FlashAttention-3 | Hopper TMA / WGMMA / FP8 |
| SageAttention2 ([2411.10958](https://arxiv.org/abs/2411.10958)) | INT4 QK + FP8 PV |
| SageAttention3 ([2505.11594](https://arxiv.org/abs/2505.11594)) | Blackwell FP4 |
| vLLM FA3 Sage2 path | Hopper FA3, not this tree |
| SGLang sage_attn PR #17679 | Closed draft |

## Dispatch lock

| Phase | Take |
|---|---|
| Prefill, large q, D=128/256 | Candidate Sage QK + fdot2 PV, **after** occupancy flip |
| Short extend (q=2–32) | Stay fa_rdna2 short until measured |
| Decode q=1 | `fa_rdna2_decode_paged`. No Sage |
| Head-64 | Still a FA2 hole first. `D%4==0` so Sage can follow, not lead |
| MLA | Off until fat tile |

## Checklist when a kernel lands

Mark each against the implementation, not this prose.

- [ ] Prefill only. Decode path unchanged.
- [ ] QK is `sdot4` on packed i8, i32 through K, scale in epilogue.
- [ ] PV is still `fdot2`.
- [ ] `D % 4 == 0` gate. 128/256 work.
- [ ] K-smoothing present or an explicit “skip + quality note”.
- [ ] Occupancy / LDS / VGPR do not occupancy-0 mixed P+D. Occupancy flip already landed first.
- [ ] No second `__launch_bounds__` min-blocks (HIP waves/EU).
- [ ] Quality: perplexity or a product eval vs fa_rdna2 FP16 QK. Do not ship on kernel time alone.
- [ ] Timing: prefill QK-bound shapes only (long q, D=128/256). Report vs `fa_rdna2_prefill_*` on the same box. No copied 4090 2.1×.
- [ ] Branch is `perf/rdna2_w4a16_bot` or a ticket — not a silent edit of `perf/rdna2_w4a16`.

## Unknowns (do not fill)

- Any tok/s or TOPS on V620.
- Whether K-smoothing is required for the models we serve.
- Occupancy of a sdot4 QK tile vs current fdot2 prefill (D=256 LDS is already tight at 59620).
- Short-extend crossover where QK stops being compute-bound.
