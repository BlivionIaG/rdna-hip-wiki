# Fork

Owned by VLLM_FORK_Manager.

- Human branch: `perf/rdna2_w4a16` — do not push here.
- Optional bot branch: `perf/rdna2_w4a16_bot` for proposed patches only.
- Tickets: [project 4](https://github.com/users/BlivionIaG/projects/4).

## Gates (stock vLLM)

- `on_gfx1x()` = gfx11/12, not gfx1030. PR #52391 adds `on_gfx10x()`.
- HIP W4A16/MoE/mxfp4 on main: `#ifdef VLLM_ROCM_GFX1100`.
- `wvSplitK_int4_g` device code: `__HIP__GFX1X__` (empty stub on gfx1030).
- `supports_fp8()` / `supports_mx()` false on gfx1030.
- Custom all-reduce: gfx94/95 only.
