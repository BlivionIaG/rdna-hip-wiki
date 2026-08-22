# gfx1030 HIP codegen stack: use / ignore

Audience: someone writing **W4A16 / W8A8 / mxfp4 HIP** for **gfx1030** (no WMMA, no MFMA) who may later automate tile search.

Date of this pass: **2026-08-17**. Companion notes on this box (not re-derived here):

- `/workspace/rdna2-architecture-brief.md` — silicon, DOT vs WMMA/MFMA
- `/workspace/rdna2-lds-tiles.md` — LDS banks, seed tiles, `sdot4`
- `/workspace/rdna-vllm-sglang-map.md` — vLLM/SGLang dispatch gates

Rule: every concrete claim is attributed. If a figure was not in a page opened this pass, it is marked **unknown**.

---

## 0. Use / ignore (one page)

| Tool | Verdict on gfx1030 | Why | What to do instead |
|---|---|---|---|
| **Hand-written HIP + packed DOT** (`v_dot2*`, `v_dot4c_i32_i8` / `__builtin_amdgcn_sdot4`, optional `v_dot8*_i4`) | **USE.** This is the real stack. | No WMMA/MFMA on RDNA2. ISA + llama.cpp `#8629` put `sdot4` on all gfx103x. | Write the kernel. Take XOR/tile *ideas*, Leave CK instances. |
| **rocBLAS Tensile** (old Tensile, not TensileLite) | **USE** for dense FP16 GEMM you do not want to own. | rocBLAS ships Tensile; gfx1030 is in Tensile's arch list. This is the library that actually runs `torch.nn.functional.linear` after hipBLASLt refuses. | Keep `PYTORCH_TUNABLEOP_ENABLED=1` and **`PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`**. |
| **PyTorch TunableOp** | **USE** for library GEMM, not for your custom quant kernel. | Benchmarks rocBLAS (and hipBLASLt if enabled) solutions per `(M,N,K,dtype,trans)`. | Tune once, reuse `tunableop_results.csv`. Do not let it try hipBLASLt on this card. |
| **Triton AMD backend (compiler)** | **USE as a portable compiler.** Do not treat it as a tuned RDNA2 product. | `HIPOptions` sets `warp_size=32` for `gfx_major >= 10`. `tl.dot` lowers to `llvm.amdgcn.fdot2` / `llvm.amdgcn.sdot4` / `llvm.fmuladd`, **not** WMMA. | Autotune `BLOCK_{M,N,K}`, `num_warps`, `num_stages`, `waves_per_eu`. Ignore `matrix_instr_nonkdim` / `kpack` (MFMA knobs). |
| **Triton as a vLLM/AITER FA product** | **IGNORE as a first-class gfx1030 path.** | vLLM `flash_attn_triton_available()` requires `on_gfx1x()` (gfx11/12). AITER `#210` (2025-03): “If you can run Triton kernels [on] gfx1030, then yes.” That is a compiler answer, not a quality claim. | Hand HIP attention, or a *your* Triton kernel you autotune. Do not wait for AITER. |
| **Composable Kernel — `DeviceGemmDl` / `DeviceGemmDpp`** | **OPTIONAL reference**, not a product you ship. | Official hello-world: XDL GEMM **refuses** on gfx1030; `example_gemm_dl_fp16` is the NAVI2x path. README: DL/DPP “useful on NAVI2x”. | Read the DL instance for VALU tiling. Do not take a CK dependency in TheRock/PyTorch wheels — gfx103x is blacklisted. |
| **CK Tile DSL (coords, XOR preshuffle)** | **TAKE THE IDEA. Do not ship the library.** | XOR / `make_xor_transform` is the right bank-conflict tool. CK's published formula is the **32-bank (CDNA) modulus**. gfx1030 WGP mode is **64 banks**. | Re-implement XOR with `% 64`. See companion LDS note. |
| **CK GEMM XDL / CK Tile universal GEMM / CK FMHA** | **IGNORE on gfx1030.** | XDL = MFMA. Tile GEMM/FMHA pipelines are XDL/WMMA. vLLM PR `#32944` (2026-01): “Flash Attention's CK backend only supports CDNA (gfx90a/gfx942/gfx950).” CK `#886` (2024-10): FA not on gfx1030/gfx1100. CK 1.2.0 later added **gfx11** FMHA — still not gfx10. | Triton FA (hand-rolled) or HIP. |
| **hipBLASLt / TensileLite** | **IGNORE in default ROCm.** Community force-build is a science project. | TheRock `#1062`: gfx10xx “technically supported by tensilelite” but **excluded** (extops need gfx90a+ `.amdhsa_accum_offset`). hipBLASLt `#648`: ExtOp asm fails on gfx10; PyTorch falls back to hipBLAS. | rocBLAS Tensile + TunableOp. Do not spend weeks unblocking hipBLASLt unless you *are* that community porter. |
| **rocWMMA** | **IGNORE.** | Official targets: gfx9 MFMA + gfx11/gfx12 WMMA. TheRock `#1944`: gfx103X “fundamentally incompatible.” rocm-libraries `#8209` (software WMMA via `v_dot2c`+DPP) was **rejected** — rocWMMA is a hardware-matrix abstraction, not a VALU emulator. | Call `sdot4` / `fdot2` yourself. |
| **llvm-calc-occupancy** | **USE** before you launch a sweep. | Same `GCNSubtarget` math the compiler uses. | `-mcpu=gfx1030 -mattr=+wavefrontsize32`. |
| **hipcc `-Rpass-analysis=kernel-resource-usage`** | **USE** every compile. | Prints VGPR/SGPR/LDS/occupancy/spills on stdout. | Gate the sweep: drop configs that spill. |
| **RGA (Radeon GPU Analyzer)** | **USE offline** for ISA + live VGPR. | Binary-analysis mode reads HIP code objects. Live-VGPR column tells you which line bought the extra register block. | `rga -s bin` on the `.hsaco` / code object. Not a runtime profiler. |
| **RGP (Radeon GPU Profiler)** | **USE if you are on Windows HIP**, or treat as optional. | Best *measured* occupancy + instruction-latency view on RDNA. AMD HPC blog: HIP/OCL RGP is a **Windows** story; Linux Instinct workflow is rocprof*. | Linux: `rocprofv3` + `LDSBankConflict` / occupancy. |
| **AITER (CK/ASM library)** | **IGNORE.** | vLLM gates AITER to CDNA3+ (`get_cdna_version() > 2`). Official AITER GPU table is gfx90a/942/950. | — |
| **mxfp4 hardware / CK MX pipelines** | **IGNORE as hardware.** | MX FP4/FP8 in CK/Triton is gfx950 / CDNA4. gfx1030 has no scale-MFMA. | Unpack to f16/i8 in software; treat like W4A16. |

**Read the table this way:** the codegen stack you *own* is HIP + DOT + a tiny tile harness. The stack you *borrow* is rocBLAS Tensile (via TunableOp) and Triton's FMA/DOT compiler. Everything that assumes MFMA or WMMA is a distraction.

---

## 1. Composable Kernel

### 1.1 What actually compiles per arch

CK is not one library. It is three instruction families plus a newer Tile frontend.

| Instance family | Instruction | gfx1030 (RDNA2) | gfx1100 (RDNA3) | gfx942 (CDNA3) |
|---|---|---|---|---|
| `DeviceGemmDl` / `batched_gemm_multi_d_dl` | VALU “DL” (`v_dot2` / FMA-class) | **Yes** — this is the documented NAVI2x path | Builds; slower than WMMA | Builds; slower than XDL |
| `DeviceGemmDpp` | DL + DPP on the load | **Yes** — README: “slightly better fp16 GEMM on NAVI2x” | Same | Same |
| `DeviceGemmXdl*` / `DeviceGemm_Xdl_CShuffle*` | XDL = MFMA | **No.** Hello-world: `DeviceGemmXdl<…> does not support this problem` | **No** (no MFMA) | **Yes** — first-class |
| `DeviceGemmWmma_CShuffle` / CK Tile WMMA GEMM | WMMA 16×16×16 | **No** (no WMMA) | **Yes** (PR `#2466` and later) | N/A (use XDL) |
| CK Tile universal GEMM (XDL pipelines, MX, preshuffle) | MFMA / MX-MFMA | **No** | Partial (WMMA Tile GEMM, not MX) | **Yes** (gfx942/gfx950) |
| CK / CK Tile FMHA (Flash Attention) | MFMA or WMMA softmax-GEMM | **No.** CK `#886` (2024-10, AMD): FA only on “recent Instinct.” vLLM `#32944` (2026-01): CK FA = gfx90a/942/950. | **As of CK 1.2.0 / ROCm 7.13: gfx11 FMHA added.** That post-dates `#32944`. Still WMMA, still not gfx10. | **Yes** — default FA-2 backend on Instinct |
| MX FP8/FP4 GEMM / FMHA | `v_mfma_scale_*` | **No** | **No** | gfx950 (CDNA4), not gfx942 |

Sources:

- Hello-world, gfx1030 vs XDL vs DL: https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.1/tutorial_hello_world.html
- README `DISABLE_DL_KERNELS` / `DISABLE_DPP_KERNELS` / `GPU_ARCHS`: https://github.com/ROCm/composable_kernel/blob/develop/README.md
- CK `#886` FA: https://github.com/ROCm/composable_kernel/issues/886
- vLLM `#32944`: https://github.com/vllm-project/vllm/pull/32944
- CK changelog 1.2.0 (ROCm 7.13) “Added gfx11 support for FMHA”; WMMA gfx12 FMHA; MX on gfx950: https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- CK Tile GEMM WMMA gfx11/12: https://github.com/ROCm/composable_kernel/pull/2466
- Example GEMM families (`DeviceGemmDl` / `Xdl` / `Wmma`): https://github.com/ROCm/composable_kernel/tree/develop/example/01_gemm

Build flags that matter if you *do* compile CK yourself:

```text
cmake -D GPU_ARCHS="gfx1030" -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ...
# DL/DPP are ON by default as of CK 1.1.0 / ROCm 7.0 (“DL and DPP kernels are now enabled by default”).
# Older trees needed -D DL_KERNELS=ON.
# Mixed families (gfx1030;gfx1100;gfx942) must use GPU_ARCHS, not GPU_TARGETS.
```

TheRock / packaged ROCm will not hand you this. `therock_amdgpu_targets.cmake` lists `composable_kernel` under `EXCLUDE_TARGET_PROJECTS` for the gfx103X family (link target: TheRock `#4836` / `#3931` / `#4768`). `#3931` (2026-03): CK configure/instance factory blows up on gfx1030 (`add_device_grouped_conv3d_fwd_xdl_*` undeclared). `#4768`: PyTorch wheel + gfx103x hits `CK_BUFFER_RESOURCE_3RD_DWORD` undeclared — “CK has not been ported to or validated for RDNA2.” The allowlist quoted in `#4836` is `gfx908 gfx90a gfx942 gfx950 gfx1150… gfx1200 gfx1201` — **no gfx1030, and at that snapshot no gfx1100 either** (gfx1100 landed later via TheRock `#4941`).

### 1.2 Tile DSL and bank XOR

CK Tile (`include/ck_tile`) is a compile-time coordinate / distributed-tensor DSL: `load_tile` / `store_tile`, `StaticDistributedTensor`, `make_xor_transform`. Docs: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/lds_index_swapping.html and https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html

What is reusable on gfx1030:

- The **XOR preshuffle** idea: permute column index by a function of the row so a wave's simultaneous LDS accesses hit different banks, **without padding**.
- The pipeline `XOR → unmerge → merge`, all constexpr.
- Typical knob: `MLdsLayer` (1/2/4) so small tiles still spread across banks.

What is *not* copy-pasteable:

- CK's published bank formula is **`bank = (addr/4) mod 32`** (CDNA / 32-bank). gfx1030 **WGP mode is 64 banks** (RDNA 2 ISA §2.3.1; companion LDS note). If you take CK's XOR as-is you will *move* conflicts, not kill them.
- TileWindow “automatically applies XOR” is true inside CK Tile GEMM/FMHA, which you are not running.
- Examples in the XOR doc are “A Block GEMM on MI300” — MFMA fragment layout, not VALU DOT.

Practical take for a HIP author:

```text
// WGP mode, 64 dword banks. Do not use CK's % 32.
dword = byte_addr >> 2
bank  = dword % 64
// XOR preshuffle (same structure as CK, modulus corrected):
k0'   = k0 ^ (m % (KPerBlock / KPack))
```

Pad-by-1 is the dumb alternative (wastes LDS, 64 KB/WG cap). Prefer XOR. Verify with `rocprofv3 --pmc LDSBankConflict` or RGP, not by staring at the formula.

### 1.3 Is CK GEMM / FA usable, or CDNA-only?

| Question | Answer |
|---|---|
| Can I call CK FA from vLLM on a 6800? | **No.** `#32944` text is explicit: CK FA = CDNA. RDNA3/4 ViT uses **Triton FA**, and only if `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` *and* `on_gfx1x()`. gfx1030 is gfx10. |
| Can I build CK FA for gfx1030? | **No** as of the pages opened. `#886` removed FA tests for gfx1030. Changelog FMHA multi-arch line is `gfx9, gfx950, gfx12` then later “gfx11 support” — never gfx10. |
| Can I use CK GEMM as my W4A16/W8A8 engine? | **Not as a library.** `DeviceGemmDl` is fp16/fp32/int8 VALU GEMM, not a dequant-GEMM. There is no CK W4A16 instance for NAVI2x in the pages opened. int8 DL exists (example table: int8 ✔️, fp8 ❌, bf16 ❌ on DL). |
| Is the Tile DSL worth learning? | Only as a **pattern book**. A gfx1030 author will spend less time writing 200 lines of HIP XOR+DOT than fighting CK's instance factory and TheRock blacklist. |

**Bottom line:** CK is CDNA-first, RDNA3-second (WMMA + recent FMHA), RDNA2-legacy (DL/DPP GEMM only). For W4A16/W8A8/mxfp4 on gfx1030, ignore CK as a runtime and keep the XOR write-up.

---

## 2. Triton AMD backend

### 2.1 Compiler support quality, 2025–2026

Two different claims exist in the wild. They are not the same sentence.

**Claim A — hipBLASLt `#648` (2024-11-30, TheTrustedComputer), talking to an RX 5500 XT (gfx1012) user:**

> “You can still use the latest PyTorch on the 5500 XT as long you compile it for the GPU's architecture and disable flash and memory-efficient attention (**Triton has no support for gfx101x and gfx103x arches**).”

Source: https://github.com/ROCm/hipBLASLt/issues/648

That sentence is about **RDNA1 + a 2024 PyTorch/xFormers/memory-efficient-attention stack**, and it **lumps gfx103x with gfx101x**. It is **not** Triton's current compiler contract.

**Claim B — Triton source (main, 2026):**

```python
# third_party/amd/backend/compiler.py  HIPOptions.__post_init__
gfx_major = int(self.arch[3:-2])   # "gfx1030" -> 10
warp_size = 32 if gfx_major >= 10 else 64
```

`num_warps` default is 4; must be a power of two. `waves_per_eu` default 0. `num_stages` default 2. `matrix_instr_nonkdim` / `kpack` exist as MFMA packing knobs (clamped/meaningful on gfx9/gfx950).

Triton PR `#3083` (2024): warp size 32 for gfx10/11, 64 for gfx9. Review: “we don't have navi CI.” Manual test was an RX 7900 XT (gfx1100), not a 6800.

FMA/DOT lowering landed **2025-01-31**: triton-lang/triton `#4594` / commit `2926ae7` — “Emit AMD specific intrinsics for FMA dot”: `AccelerateAMDMatmul` emits FMA for `i8×i8→i32` and `fp16×fp16→fp32`; codegen uses `v_dot` for those dtypes.

AITER `#210` (2025-03-17, still open): “Does `triton_mha_fwd_kernel` support rdna2/gfx1030?” AMD reply (`rahulbatra85`): **“If you can run Triton kernels rdna2/gfx1030, then yes it should run.”** That is “the Python will JIT,” not “we tuned it.”

DeepWiki's 2026 AMD-backend architecture table (derived from the same `compiler.py`) lists first-class rows as **gfx942, gfx950, gfx110x/115x, gfx1200/1250**. gfx1030 is not in that table. Feature flags in `parse_options` special-case gfx942 (TF32) and gfx950 (FP8 deprecations). Nothing special-cases gfx1030 except the warp-size rule.

**Quality summary for a 2025–2026 gfx1030 author:**

| Layer | Status |
|---|---|
| JIT / HSACO for `tl.load` / `tl.store` / `tl.dot` | Works if your ROCm+Triton wheel was built with gfx1030 (or you override arch). |
| `tl.dot` → hardware | VALU `fdot2` / `sdot4` / `fmuladd`. **Not WMMA. Not MFMA.** |
| Navi CI / official perf guide | Historically absent. AMD Triton perf wiki + `optimizing-triton-kernel` docs are **MI300X / MFMA** documents. |
| AITER / vLLM product kernels | Not enabled for gfx1030. `#210` is a shrug. |
| `#648` “no Triton on gfx103x” | **Stale / wrong** for the compiler in 2026. Still directionally right for “nobody shipped a tuned FA/GEMM product.” |

### 2.2 How `tl.dot` maps on RDNA2

From Triton's AMD FMA lowering (`third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp`, main):

| `tl.dot` dtypes | Intrinsic | Hardware |
|---|---|---|
| f16×f16 → f32 | `llvm.amdgcn.fdot2` | `v_dot2_f32_f16` / `v_dot2c_f32_f16` (vector size 2) |
| bf16×bf16 → f32 | `llvm.amdgcn.fdot2.f32.bf16` | `v_dot2` bf16 variant |
| i8×i8 → i32 | `llvm.amdgcn.sdot4` | `v_dot4_i32_i8` / `v_dot4c_i32_i8` (vector size 4) |
| f16→f16, f32→f32, f64 | `llvm.fmuladd.*` | plain FMA, vector size 1 |

`AccelerateAMDMatmul.cpp` still talks about `instrShape` of “wmma/mfma hardware instruction” when *planning warps*. On gfx1030 there is no such instruction; the FMA path is what `#4594` added so small dots do not require a matrix core. If you dump TTGIR and see `amd_mfma` / `amd_wmma` layouts, you are looking at a gfx9/gfx11 compile, not a 6800 compile.

Tensile's own capability dump (pasted in hipBLASLt `#648` for gfx1012, same table has a gfx1030 column) is the hardware truth:

```text
HasMFMA     gfx1030 = 0
HasWMMA     gfx1030 = 0          # gfx1100 = 1
v_dot2_f32_f16 / v_dot2c_f32_f16 = 1
VOP3v_dot4_i32_i8 / v_dot4c_i32_i8 = 1
HasWave32   = 1
VgprBank    = 1
```

### 2.3 Autotune knobs that matter (and that do not)

`num_warps=4` on RDNA **means 4 × 32 = 128 threads**, not 4 × 64. Triton's own `Config` docstring still says “8 * 32 = 256” (NVIDIA-centric). On gfx9 it would be 4 × 64; on gfx1030 it is 4 × 32. Source: `compiler.py` `warp_size` + https://triton-lang.org/main/python-api/generated/triton.Config.html

| Knob | On gfx1030 | Sweep? |
|---|---|---|
| `BLOCK_M` / `BLOCK_N` / `BLOCK_K` | Tile of the `tl.dot`. This is 90% of the search. | **Yes. First.** |
| `num_warps` | 2 / 4 / 8. Workgroup threads = `num_warps * 32`. `num_warps=4` → 128. | **Yes. Second.** |
| `num_stages` | Software pipeline of `tl.load`→`tl.dot`. Default 2. LDS × stages. | Yes, 1 vs 2. 3 often blows 64 KB. |
| `waves_per_eu` | Hint: LLVM `amdgpu-waves-per-eu`. 0 = compiler default. 2/4 asks for lower VGPR so more waves fit. | Yes if occupancy-limited. |
| `GROUP_SIZE_M` | Persistent/grouped launch (L2 reuse). | Yes for fat GEMM; **no** for skinny decode. |
| `matrix_instr_nonkdim` (16 vs 32) | MFMA shape. **No-op / ignore** on gfx1030. | No. |
| `kpack` | MFMA K packing. gfx950 clamps to 1. | No. |
| `num_ctas` > 1 | Cluster launch. `compiler.py` errors if the arch does not support it. | No. |

Dump winner: `TRITON_PRINT_AUTOTUNING=1`. Dump ISA: Triton's AMD wiki (generate `amdgcn`, grep `.vgpr_count`, `ttg.shared`, `ttg.num-warps`).

AITER `#210` does **not** give you a recommended config. If you use Triton at all on this card, you own the autotune list.

---

## 3. hipBLASLt / TensileLite / rocBLAS Tensile / TunableOp

### 3.1 Default exclusion

TheRock `#1062` (open, 2025-07 → 2026-07) quotes hipBLASLt CMake:

> gfx10XX architectures (e.g. gfx1010, gfx1011, gfx1030, …) are **technically supported by tensilelite**, but are **NOT included in the default "all" build** in hipBLASLt. This is because **"extops" builds are not supported for legacy devices**.

Supported TensileLite targets in that error: `gfx908; gfx90a; gfx942; gfx950; gfx1100; gfx1101; gfx1103; gfx1150; gfx1151; gfx1200; gfx1201`. **No gfx1030. No gfx1102** either (that was the original failure).

`therock_amdgpu_targets.cmake` gfx103X family `EXCLUDE_TARGET_PROJECTS` includes **hipBLASLt** (`#1062`), **rocWMMA** (`#1944`), **composable_kernel**, hipSPARSELt, hipTensor.

### 3.2 Why force-build dies

hipBLASLt `#648` (closed, migrated to rocm-libraries `#321`): community attempt to compile for gfx1012. ExtOp generator emits:

```text
.amdhsa_accum_offset 40   ; error: directive requires gfx90a+
```

Same class of failure on TheRock gfx103x CI (`#1002`, `#1062`): `L_256_4_1_gfx1032.s` + `-mwavefrontsize64` + accum-offset. TensileLite *library* generation then hits `IndexError: list index out of range` because there is no Logic file for that ISA.

AngryLoki on `#648`: hipBLASLt and rocWMMA are tied to **MFMA (gfx9) or WMMA (gfx11)**. Workaround for PyTorch's *link* dependency: build hipBLASLt for a dummy supported arch (e.g. gfx940) or Gentoo's `BUILD_WITH_TENSILE=OFF` stub; at runtime PyTorch sees “unsupported architecture” and falls back to hipBLAS/rocBLAS. `TORCH_BLAS_PREFER_HIPBLASLT=0` is optional after the 2.4.0 fallback fix.

### 3.3 Community force-build — status as of 2026-07

GrecAndrei on TheRock `#1062` (2026-07-09), RX 6650 XT / **gfx1032**:

- Silent load failure was not just “unsupported”: `TensileLibrary_lazy_gfx1032.dat` present but msgpack `Enum not found! gfx1032` because `AMDGPU::Processor` lacked 1031/1032/1034/1035.
- Open PRs: rocm-libraries `#9075` (Processor enum), `#9076` (register gfx1032 as Tensile arch), `#9077`, `#9219`.
- Local proof after enum fix + Tensile lib: **FP16 GEMM 2048³ ≈ 28.7 TFLOPS** on 6650 XT.
- Next TheRock step: remove hipBLASLt from `EXCLUDE_TARGET_PROJECTS` for gfx1032 **after those land and CI exists**. Until then the exclude is correct for default builds.

That is a **gfx1032 community porter** report, not an AMD product. gfx1030 (6800-class) is the same TensileLite ISA family, so the same patches *might* apply, but you still need Logic files, ExtOp asm that does not use `accum_offset`, and a Processor enum. **Do not start a W4A16 project by force-building hipBLASLt.**

### 3.4 rocBLAS Tensile is the actual FP16 GEMM backend

rocBLAS design notes: https://rocm.docs.amd.com/projects/rocBLAS/en/latest/conceptual/rocblas-design-notes.html

- rocBLAS embeds **Tensile** (old generator) and *may* call hipBLASLt on arches that have it (gfx12 default).
- `ROCBLAS_USE_HIPBLASLT=0` forces Tensile.
- Tensile's *rocBLAS* arch list (TheRock `#1443` paste of `TensileSupportedArchitectures.cmake`) **includes gfx1030** (and 1031/1032/1034/1035). That is a different CMake file from hipBLASLt's TensileLite list.

So: `torch.matmul` / `F.linear` on a 6800 with hipBLASLt disabled or rejected → **rocBLAS → Tensile asm/source GEMM**. That path is real, tuned years ago for NAVI2x, and is what TunableOp should search.

### 3.5 TunableOp (you already have this right)

PyTorch `aten/src/ATen/cuda/tunable/README.md`:

| Env | Default | gfx1030 setting |
|---|---|---|
| `PYTORCH_TUNABLEOP_ENABLED` | 0 | **1** |
| `PYTORCH_TUNABLEOP_TUNING` | 1 | 1 for a capture run, then 0 |
| `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` | 1 | **0** (your `HIPBLASLT=0`) |
| `PYTORCH_TUNABLEOP_ROCBLAS_ENABLED` | 1 | **1** |
| `PYTORCH_TUNABLEOP_FILENAME` | `tunableop_results.csv` | set it |
| `PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE` | L2 size | **0** if you hit the view-GEMM fault (`pytorch#140278`, filed on an RX 6900 XT) |

TunableOp does **not** tune your W4A16 kernel. It only wraps `at::cuda::blas::gemm`. Use it so the *unquant* / epilogue GEMMs (and any path that still goes through `F.linear`) are not stuck on a random Tensile default. Offline flow: `RECORD_UNTUNED=1` → `tunable.tune_gemm_in_file(...)`.

Also set `TORCH_BLAS_PREFER_HIPBLASLT=0` if you still see “Attempting to use hipBLASLt on an unsupported architecture.”

---

## 4. rocWMMA

Official README targets (https://github.com/ROCm/rocWMMA):

- CDNA matrix cores: gfx908 / 90a / 942 / 950 as `gfx9`
- RDNA AI: gfx1100/1101/1102/1151 as `gfx11`; gfx1200/1201 as `gfx12`

TheRock `#1944`: **gfx103X fundamentally incompatible** — “do not support the matrix instructions that rocWMMA facilitates.” Will not be supported.

rocm-libraries `#8209` (community, gfx1032 software WMMA via `v_dot2c_f32_f16` + DPP8 + `ds_bpermute`): AMD rejected it. Quote: rocWMMA's purpose is a C++ abstraction over **hardware** MFMA/WMMA; a VALU fallback has no bound. Suggestion: publish the microkernel as a **standalone header**.

**Do not `#include <rocwmma/rocwmma.hpp>` on gfx1030.** If you want a 16×16×16 f16 software fragment, copy the DPP8 idea into your own header and own it. For W8A8, `sdot4` is the native op — you do not need a WMMA-shaped API.

---

## 5. Practical automation: sweep tiles

### 5.1 Hardware budget (so the sweep is legal)

From the companion silicon brief + GPUOpen occupancy + LLVM IsaInfo (not re-derived):

| Resource | gfx1030 (WGP mode, wave32) | Why it gates tiles |
|---|---|---|
| Wave | 32 threads | `num_warps=4` → 128 threads; HIP WG=256 → 8 waves |
| Wave slots | 16 / SIMD32 | Occupancy = assigned / 16 |
| VGPR file | 1024 / SIMD32, granule 16 (LLVM IsaInfo; companion brief) | 64 VGPR → 16 waves; 128 VGPR → 8 waves |
| LDS | 128 KB / WGP; **≤ 64 KB / workgroup**; granule 1 KB | Double-buf + pad must stay ≤ 64 KB |
| LDS banks | **64** dword banks in WGP mode | XOR modulus 64, not CK's 32 |
| Max WG | 1024 threads | Stay at 128 or 256 |
| DOT | `v_dot2*` (f16), `v_dot4c_i32_i8` (sdot4), `v_dot8*_i4` | W8A8 BK % 4 == 0; W4 can use `v_dot8` i4 or unpack→f16 |
| Infinity Cache | 128 MB on 6800/6900-class | Skinny decode is cache-resident more often than on RDNA1 |

`llvm-calc-occupancy` is the source of truth for *your* binary:

```bash
# same math the AMDGPU backend uses (GCNSubtarget)
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 \
  --wg-size=256 --vgprs=72 --sgprs=40 --lds=16k

# occupancy ladder (how many VGPR you may spend for N waves/EU)
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --limits
```

Docs: https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html

Every HIP compile:

```bash
hipcc --offload-arch=gfx1030 -O3 \
  -Rpass-analysis=kernel-resource-usage \
  -save-temps \
  w4a16.hip -o w4a16
# remarks: VGPRs, SGPRs, LDS, Occupancy [waves/SIMD], spills
# reject any config with VGPR/SGPR spill
```

RGA (offline ISA, live VGPR — no GPU required):

```bash
# Binary analysis of a HIP code object (RGA 2.9+)
rga -s bin --isa out/isa.txt --livereg out/livereg.txt --analysis out/stats.txt \
    my_kernel.co
```

RGA does not run the kernel. Use it when occupancy math says “cut 8 VGPR to gain a wave” and you need to know *which* live range.

RGP (measured occupancy, instruction latency, wave32 vs wave64, limiter hint on the Pipeline tab): Windows HIP via Radeon Developer Panel (RGP 1.14+). Linux HPC blog (https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html): RGP HIP is a Windows story; use **rocprofv3**.

```bash
rocprofv3 --pmc LDSBankConflict,VALUBusy,SALUBusy,MemUnitStalled,GPUBusy \
          --output-format csv -- ./harness --tile 64,64,64
```

### 5.2 Tiny harness (what to automate)

One binary, one CSV row per config. Do not start with Triton autotune if the kernel is already HIP.

```text
fields: kernel, M, N, K, BM, BN, BK, waves, wg, lds_bytes, vgpr, occ_waves,
        t_us, tflops_or_gbps, lds_conflict, spill
reject: spill != 0, lds_bytes > 65536, occ_waves < 4 unless ILP-only win
repeat: 50 warmup / 200 iters, rotating buffers (or TunableOp-style pool)
shapes: capture the real vLLM decode/prefill (M,N,K) first — do not sweep 4k³
```

Occupancy pre-filter (no GPU):

1. Compile tile → read VGPR/LDS from `-Rpass-analysis` or the `.s` `.amdhsa_*` block.
2. `llvm-calc-occupancy` → drop configs below your floor (start at 8 waves/SIMD for memory-bound, 4 for compute-bound).
3. Run survivors. Rank by time, not by occupancy. GPUOpen occupancy article: more waves can trash Infinity Cache / L0.

### 5.3 What to sweep first

Two different problems. Do not share a config list.

#### W4A16 skinny (decode GEMV / “M = 1…16, N = hidden, K = hidden”)

This is **memory-bound**, often Infinity-Cache resident. hipfire's gfx1030 note (community, not AMD): RDNA2's 128 MB IC changes the occupancy/ILP trade — **fewer waves, more work per wave** beat gfx1010-style `launch_bounds(32,20)` high-occupancy tunings.

Sweep **in this order**:

1. **`BLOCK_N` (output channels per WG)** — 64, 128, 256. Skinny M means you have almost no M-parallelism; N-tile is how you buy memory coalescing and IC reuse on the weight row.
2. **`BLOCK_K` / vector load width** — 32, 64, 128. Prefer `dwordx4` on packed u4 (32 nibbles/load). K-tile is the dequant reuse distance.
3. **Dequant placement** — scales/zeros in SGPR vs LDS vs per-thread. Do this before you touch occupancy.
4. **`num_warps` / WG** — 4 (128 thr) then 8 (256) then 2. `num_warps=4` is the Triton-shaped default and a good HIP `__launch_bounds__(128, …)` seed.
5. **Occupancy / `__launch_bounds__(WG, min_waves)`** — (128, 8) vs (128, 4) vs (256, 4). Measure; do not assume 16-wave is faster.
6. **LDS staging of B (weights)** — register-only vs 8–16 KB LDS. Skinny M often wants **no LDS for A** (A is the activation vector, broadcast).
7. Ignore `GROUP_SIZE_M`, `matrix_instr_nonkdim`, hipBLASLt, CK XDL.

Seed from the companion LDS note (recipe 1 / 9 scaled to skinny M): **BM=16 or 32, BN=64, BK=64, WG=128, LDS ≤ 16 KB**. Unpack nibble → f16 then `v_dot2`, *or* keep i4 packed and use `v_dot8_i32_i4` if you already trust that path (ISA exists; quality **unknown** in vLLM — llama.cpp is the existence proof).

Do **not** start with a 128×128×32 GEMM tile. You will be occupancy-dead and launch-bound.

#### W8A8 sdot4 (prefill / fatter M, or a real GEMM)

`v_dot4c_i32_i8` / `__builtin_amdgcn_sdot4(a,b,c,false)` — 4×i8→i32 per instruction. BK **must be a multiple of 4**. Packed layout walks LDS as `int`, not `half` (companion LDS note recipe 9).

Sweep **in this order**:

1. **`BLOCK_K`** — 32, 64, 128 (all % 4 == 0). This is what keeps `sdot4` fed. 64 is the companion seed (4 KB+4 KB LDS at 64×64).
2. **`BLOCK_M` × `BLOCK_N`** — (64,64), (128,64), (64,128), (128,128). Stop at 128×128 until VGPR < ~80 and no spill.
3. **XOR vs pad on LDS** — sequential `int` along K is already conflict-free for wave32 on 64 banks; XOR matters when you *transpose* for the inner product. Measure `LDSBankConflict` before adding XOR.
4. **`num_warps` 4 vs 8** (128 vs 256). 256 is the companion default for the 64×64×64 seed.
5. **`num_stages` / double-buffer** — 8 KB → 16 KB. Still two WG/WGP. Do not triple-buffer.
6. **`waves_per_eu` / launch_bounds** only after the tile is stable.

Seed: **64×64×64, WG=256, LDS 8 KB (16 KB double-buf), `sdot4`, no WMMA header.**

#### mxfp4

No hardware scale-dot on gfx1030. Sweep it as **W4A16** (unpack to f16 + `fdot2`) or as **packed i4** (`v_dot8`) plus a software scale. Do not import a CK/Triton MX pipeline.

### 5.4 Suggested loop (HIP, not Triton)

```text
for shape in captured_vllm_shapes:          # decode skinny vs prefill fat, separately
  for (BM,BN,BK) in seeds_for_that_shape:
    for wg in (128, 256):
      for min_occ in (4, 8, 12):
        compile with -Rpass-analysis
        if spill or lds>64K: skip
        occ = llvm-calc-occupancy(...)
        if occ.waves < min_occ: skip        # optional floor
        run harness, write CSV
  # only then: XOR on/off, double-buf on/off, on the top 3 tiles
```

Triton equivalent if you prototype in Python first: `@triton.autotune` over `BLOCK_*` + `num_warps` + `num_stages` + `waves_per_eu`, `key=['M','N','K']`, `TRITON_PRINT_AUTOTUNING=1`. Same reject rules. Do not add `kpack` / `matrix_instr_nonkdim`.

---

## 6. Sources (opened this pass)

**CK**

- https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.1/tutorial_hello_world.html
- https://github.com/ROCm/composable_kernel/blob/develop/README.md
- https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- https://github.com/ROCm/composable_kernel/issues/886
- https://github.com/ROCm/composable_kernel/pull/2466
- https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/lds_index_swapping.html
- https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html
- https://github.com/vllm-project/vllm/pull/32944
- https://github.com/ROCm/TheRock/issues/3931
- https://github.com/ROCm/TheRock/issues/4768
- https://github.com/ROCm/TheRock/issues/4836

**Triton**

- https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/FMA.cpp
- https://github.com/triton-lang/triton/commit/2926ae791c39c776e623602b15d2545424890b49 (`#4594`, 2025-01-31)
- https://github.com/openai/triton/pull/3083
- https://github.com/ROCm/aiter/issues/210
- https://github.com/ROCm/hipBLASLt/issues/648 (Triton-support comment; see §2.1)
- https://triton-lang.org/main/python-api/generated/triton.Config.html
- https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html (MI300X knobs; use only to see what *not* to copy)

**hipBLASLt / Tensile / TunableOp**

- https://github.com/ROCm/TheRock/issues/1062
- https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake
- https://github.com/ROCm/hipBLASLt/issues/648
- https://rocm.docs.amd.com/projects/rocBLAS/en/latest/conceptual/rocblas-design-notes.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md
- https://github.com/pytorch/pytorch/issues/140278
- https://github.com/pytorch/pytorch/pull/128753

**rocWMMA**

- https://github.com/ROCm/rocWMMA
- https://github.com/ROCm/TheRock/issues/1944
- https://github.com/ROCm/rocm-libraries/issues/8209
- https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.1.0/api-reference/api-reference-guide.html

**Occupancy / RGA / RGP**

- https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
- https://gpuopen.com/learn/occupancy-explained/
- https://gpuopen.com/rga/
- https://gpuopen.com/manuals/rga_manual/quickstart/
- https://gpuopen.com/rgp/
- https://gpuopen.com/learn/rgp_1_14/
- https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
- https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/performance_guidelines.html
- https://github.com/ROCm/ROCm/issues/1609 (`-Rpass-analysis=kernel-resource-usage`)

**DOT / W4A16 / W8A8**

- llama.cpp `#8629` / commit `46e4741` — `sdot4` on all RDNA2
- LLVM D158468 — `llvm.amdgcn.sdot4` → `v_dot4_i32_i8`
- Companion `/workspace/rdna2-lds-tiles.md` recipe 9
- hipfire `545e6ec` (community RDNA2 GEMV occupancy note; not an AMD doc)

---

## 7. One-paragraph recommendation

On gfx1030 you are a **VALU-DOT HIP author**. Use rocBLAS Tensile + TunableOp (hipBLASLt off) for dense FP16; use Triton only as a compiler whose `tl.dot` is `fdot2`/`sdot4`; take CK's XOR idea with a **64-bank** modulus; ignore CK FA, CK XDL, hipBLASLt, rocWMMA, AITER, and every MFMA autotune knob. Sweep W4A16 skinny on **N and K plus launch_bounds**, W8A8 on **BK then square tiles**, with `llvm-calc-occupancy` + `-Rpass-analysis` as the pre-filter and rocprofv3/RGP as the lie detector.
