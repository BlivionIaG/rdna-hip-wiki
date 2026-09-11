# W4A16 prefill config lock (gfx1030)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip **`441ec054`**.

## Take
- Large-M AWQ/GPTQ prefill → **ConfigA** (`THREADS=256`, `N_TILE=1024`, `M_TILE=16`, **`K_STEP=32`**, `LDS=0`).
- AWQ M>32 dispatcher → GPTQ `gptq_gemm_rdna2_prefill` (not a separate AWQ prefill object).
- Same `fdot2` / `qdq_4_rdna2`; AWQ `zero_offset=0` via `use_v2_format`.
- Dead `q_gemm_rdna2_awq_prefill.cu` removed (`1046782`).
- Inner loops use `K_STEP/8`; **per-j group refresh** (`if (k + 8*j == nextgroup)`) so mid-block group boundaries are correct when `K_STEP > groupsize` (ConfigA with `K_STEP==groupsize` is behavior-neutral).

## Leave
- **ConfigA_Large** (`THREADS=128`, `N_TILE=512`, `M_TILE=32`, `K_STEP=32`, `LDS=0`): dormant enum slot `ConfigId_A_Large=4`, never selected by natural `select_config`. Reachable only via `VLLM_RDNA2_PREFILL_FORCE_CONFIG=4`. Measured slower than ConfigA on fat-M AWQ shapes (fewer M-blocks did not cut HBM; THREADS=128 + fat `block_c` hit spill). Do not promote without beating ConfigA on fat-M greedy.
- Old **ConfigH** (`K_STEP=64`) name is gone from the tip; replaced by the A_Large slot above. Do not reintroduce K_STEP=64 dispatch until fat-M numerics match ConfigA.
- Resurrect AWQ prefill object, tok/s claims, EXL3 DOT changes (UNC-26 stays `-cb 3inst`).

## Infra (441ec054)
- `VLLM_RDNA2_PREFILL_FORCE_CONFIG` (`0=V1`, `1=A`, `3=C`, `4=A_Large`) for kernel bisection; default unset → natural dispatch.
- Persist zeros keepalive unchanged. Occupancy still FA-first.

Companion: [kernels/w4a16.md](../kernels/w4a16.md).
