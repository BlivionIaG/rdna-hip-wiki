# ikantkode gfx1030 overlays — silicon review

Sourced from [ikantkode/gfx1030-vllm-0.26](https://github.com/ikantkode/gfx1030-vllm-0.26) (file-mount patches on `blivioniag/vllm-rdna:v0.26.0`) and the `-vd` checkpoint [ikantkode/Qwen3.5-4B-AWQ-vd](https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd). Their numbers, not ours. Occupancy still first.

**Not our HIP path.** Triton AWQ GEMV + gate flips. No `fdot2` / `sdot4` / `fa_rdna2`. Do not port the Triton kernels. Steal the **gates and occupancy lessons**.

## What they actually changed (silicon)

| Rung | Change | Opcode / launch | Take |
|---|---|---|---|
| 3 | Unlock AMD **LLMM1** for gfx1030 (`on_gfx9 \|\| on_gfx1x \|\| gfx1030`) | in-tree skinny n==1, `k≤8192` | **Keep.** Highest-ROI line. Check the fork has this gate. |
| 3 | **Disable `wvSplitK` on gfx1030** | device-assert | **Keep off.** Same skinny card. Do not “enable MFMA.” |
| 6–10, 15 | M==1 AWQ GEMV, then K-split + per-(N,K) table | Triton `tl.sum` fp32 FMA, **not** `tl.dot` / `fdot2` | Lesson only: fill 72 CUs when `cdiv(N,BN)` is tiny. HIP W4A16 already owns this job. |
| 7 | fp16 GEMV for `n==1, k>8192` (GDN `out_proj` 2560×9216) | Triton scalar FMA | LLMM1 cannot launch there. HIP skinny / W4A16 should cover k>8192. |
| 13 | Fused Gemma RMSNorm | 1 Triton vs 10–13 ATen; IR gate wants `weight.dtype==x.dtype`, Gemma weight is fp32 | **New Todo** if we want it. Launch tax, not DOT. |
| 14 | paged-attn Triton `num_warps=8` | 4 WG on 72 CUs was latency-bound (226→112 µs) | Same occupancy ticket as `fa_rdna2`. |
| 8 | Re-quant attn + GDN to INT4 | checkpoint, not a kernel | Engine. `(1+w)` LN-fold, not Llama `w`. |

Their GEMV inner loop is `acc += tl.sum(w_f32 * x_f32)` after AWQ nibble unpack. Same “unpack then scalar FMA” we already refused to treat as a new DOT on MLA.

## Confirmed landmines (matches our ISA)

- LDS **64 KB/WG**: `TRITON_ATTN` asked 139264 and died. They stay on `ROCM_ATTN`.
- `wvSplitK` asserts on gfx1030 (our `skinny_gemms.cu` MFMA/`waves_per_eu(1,1)` trap).
- No bf16; `--dtype float16`.
- AITER off.
- MTP **slower** on this M=1-optimized decode — same fat-tile (`q>1`) blocker we already filed.
- torch.compile/inductor can freeze gfx1030 (ROCm #5572); they use breakable cudagraphs.

## Tickets

Reuse: occupancy (`fa_rdna2` + skinny), W4A16 dense (HIP GEMV, not their Triton), MTP Todo.

**Add**
1. **LLMM1 gfx1030 gate** — one-line `on_gfx10x()` in `rocm_unquantized_gemm_impl`. Confirm whether `perf/rdna2_w4a16` already has it.
2. **Fused Gemma RMSNorm** — fp32 gain `(1+w)`, one kernel, kill the native chain. Spec.

**Do not add:** a Triton AWQ GEMV card, a GDN/FLA HIP card (their FLA is 0.31 ms/token), copying their per-shape table into HIP.

## Sources

- https://github.com/ikantkode/gfx1030-vllm-0.26 README + CHANGELOG + `patches/utils.py` + `patches/awq_triton.py`
- https://huggingface.co/ikantkode/Qwen3.5-4B-AWQ-vd
- RDNA 2 ISA: 64 KB LDS/WG; no bf16 DOT
