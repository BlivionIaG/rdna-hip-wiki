# W4A16 prefill config lock (gfx1030)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip **`7ac98a26`**.

## Take
- Large-M AWQ/GPTQ prefill → **ConfigA** (`THREADS=256`, `N_TILE=1024`, `M_TILE=16`, **`K_STEP=32`**, `LDS=0`).
- AWQ M>32 dispatcher → GPTQ `gptq_gemm_rdna2_prefill` (not a separate AWQ prefill object).
- Same `fdot2` / `qdq_4_rdna2`; AWQ `zero_offset=0` via `use_v2_format`.
- Dead `q_gemm_rdna2_awq_prefill.cu` removed (`1046782`).

## Leave
- **ConfigH** (`K_STEP=64`): present in enum/switch but unused. Doubling K step without matching packed DOT coverage was incorrect at M>256.
- Resurrect AWQ prefill object, tok/s claims, EXL3 DOT changes (UNC-26 stays `-cb 3inst`).

## Why ConfigH failed
Outer loop advanced by 64 K while the inner packed loop originally covered 32 K (`j < 4`). Tip kept `K_STEP/8` loop fix for future work but reverted dispatch to ConfigA. Do not re-enable H without a fat-M greedy that matches ConfigA.

Companion: [kernels/w4a16.md](../kernels/w4a16.md). Occupancy still FA-first.
