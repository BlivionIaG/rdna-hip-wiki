# llama.cpp side project — ROCmFPX on V620

Date: 2026-08-17. **Not the vLLM fork.** Occupancy on `perf/rdna2_w4a16` still first. Contract: [rocmfpx.md](rocmfpx.md). Source: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX).

**Why a side project:** their HIP kernels already target gfx1030 (`scripts/build-rdna2.sh`). We can measure codebook→integer-dot on the same V620 without touching the vLLM tree. Steal confirmed ISA into W4A8 / a later codebook GEMM. Do not import ggml into vLLM.

## Scope

| Do | Don't |
|---|---|
| Clone ROCmFPX `main`, `env JOBS=N scripts/build-rdna2.sh` | Merge into `BlivionIaG/vllm` |
| ISA dump MMVQ/MMQ inner loop on gfx1030 | Guess `sdot4` vs FMA from the README |
| One small GGUF (Qwen 4B-class ROCmFP4 FAST or STRIX_LEAN) | Copy their gfx1151 tok/s onto coverage |
| Note `amdgcn_perm` + UE4M3 vs our mxfp4/NVFP4 LUTs | Port Vulkan, MTP, TurboQuant |
| File findings back to [rocmfpx.md](rocmfpx.md) + silicon | Bind-mount their `.cu` into the vLLM docker |

## Done-when

1. Binary from `build-rdna2/` runs on a V620 (`HSA_OVERRIDE_GFX_VERSION=10.3.0` if needed).
2. ISA of the hot `MUL_MAT` / MMVQ kernel shows the real opcode (`v_dot4c_i32_i8` vs scalar FMA).
3. One short decode smoke (greedy, fixed tokens). Numbers stay on this page / the note, not [coverage.md](coverage.md).
4. One paragraph: keep / drop vs W4A8 `sdot4`.

## Card

@VLLM_FORK_Manager: **one Later / side-project card**, not a vLLM In Progress. Title: `llama.cpp ROCmFPX on V620`. Points at this page + [rocmfpx.md](rocmfpx.md). Does not block occupancy.

## Sources

- `scripts/build-rdna2.sh` → `gfx1030`
- `docs/BUILD-AMD-ARCHITECTURES.md`
- [alt-engines.md](alt-engines.md) (llama.cpp MMVQ / `ggml_cuda_dp4a`)
