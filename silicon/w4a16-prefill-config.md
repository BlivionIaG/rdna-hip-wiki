# W4A16 prefill config lock (gfx1030)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip **`56f67111`**.

## Take
- Large-M AWQ/GPTQ prefill → **ConfigA** (`THREADS=256`, `N_TILE=1024`, `M_TILE=16`, **`K_STEP=32`**, `LDS=0`).
- AWQ M>32 dispatcher → GPTQ `gptq_gemm_rdna2_prefill` (not a separate AWQ prefill object).
- Same `fdot2` / `qdq_4_rdna2`; AWQ `zero_offset=0` via `use_v2_format`.
- Dead `q_gemm_rdna2_awq_prefill.cu` removed (`1046782`).
- Inner loops use `K_STEP/8`; **per-j group refresh** (`if (k + 8*j == nextgroup)`) so mid-block group boundaries are correct when `K_STEP > groupsize` (ConfigA with `K_STEP==groupsize` is behavior-neutral).

## Leave
- **ConfigA_Large** (`THREADS=128`, `N_TILE=512`, `M_TILE=32`, `K_STEP=32`) and **ConfigP** (register double-buffered prefetch): deleted from tip `56f67111` after measuring slower than ConfigA on fat-M AWQ shapes (fewer M-blocks did not cut HBM / Infinity Cache already absorbs weight re-reads; THREADS=128 + fat `block_c` hit spill; ConfigP cut workgroups/CU). Do not resurrect without beating ConfigA on fat-M greedy.
- Old **ConfigH** (`K_STEP=64`) name remains gone. Do not reintroduce K_STEP=64 dispatch until fat-M numerics match ConfigA.
- Resurrect AWQ prefill object, tok/s claims, EXL3 DOT changes (ticket-26 stays `-cb 3inst`).

## Infra (`56f67111`)
- `VLLM_RDNA2_PREFILL_FORCE_CONFIG` (`0=V1`, `1=A`, `3=C`) for kernel bisection; default unset → natural dispatch. Slot `4` (A_Large) is gone.
- `VLLM_RDNA2_PREFILL_FORCE_SPLIT_K` debug override kept; natural `split_k` stays. Commit notes natural split_k=4 sits near the measured curve — fp16 CAS-loop atomic epilogue is not the bottleneck for a redesign.
- Persist zeros keepalive unchanged. Occupancy still FA-first.

Companion: [kernels/w4a16.md](../kernels/w4a16.md).


## K_STEP split-K repair (`3a0786ea`)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` @ **`3a0786ea`**. Object: `csrc/rocm/q_gemm_rdna2_prefill.cu` `compute_split_k` (gfx1030 W4A16 prefill; same TU W4A8 falls back into).

### Take — repair-only policy
- Kernel walks K in `K_STEP`-wide chunks and does **not** clamp the last chunk to `k_per_split`. A split with `(size_k / split) % K_STEP != 0` over-reads the split’s LDS row / global K (NaN/garbage; e.g. `k=4352`, legacy `split=16` → `k_per_split=272` at ConfigV1).
- **Keep** the pre-W4A8 powers-of-two search (`legacy`) whenever usable: `size_k % legacy == 0 && (size_k / legacy) % K_STEP == 0`.
- **Else only:** enumerate every usable split `s ∈ [1..16]` with `size_k % s == 0 && (size_k / s) % K_STEP == 0`, then apply the same LDS-budget / grid-growth heuristics. `size_k % K_STEP == 0` at entry ⇒ `split=1` always usable.
- Debug: `VLLM_RDNA2_PREFILL_DEBUG` prints chosen split; force override still `VLLM_RDNA2_PREFILL_FORCE_SPLIT_K`.

### Audit counts (grounded; no tok/s)
From `bench_results/2026-09-29_w4a16-regression-audit/SUMMARY.md` + `split_table.txt` over 60 production shapes:

| Class | Count | Lock |
|---|---:|---|
| Old invalid → repaired | **8** | `(225,4352,{3584,4096,5120,8704})`, `(2001,4352,8704)`, `(2001,8704,8704)`, `(2048,4352,8704)`, `(2048,8704,8704)` |
| Legacy-valid retunes reverted | **13** | earlier full-enumeration path had changed them; repair restores legacy |
| Deviation on legacy-valid | **0** | policy: repair, don’t re-tune |

### Leave
- Replacing powers-of-two with full enumeration for shapes that were already valid
- Copying ms / % from the audit CSVs into dest claims
- Changing ConfigA / `K_STEP=32` tile geometry (unchanged)

Companion W4A8 opt-in: [kernels/w4a8.md](../kernels/w4a8.md).
