# vLLM / SGLang on RDNA 2 (gfx1030) and RDNA 3 (gfx1100) vs CDNA

Audience: someone writing custom HIP kernels for LLM inference. Date of this pass: **2026-08-17**.

Rule: every concrete claim is attributed to a URL that was opened. If a figure is not in those sources, it is marked **unknown**. This is not a performance paper and not a fake support matrix.

Companion silicon notes (already on disk; not re-derived here):
- `silicon/architecture.md` — no MFMA, no WMMA on gfx1030; packed DOT only
- `silicon/lds-tiles.md` — LDS tiles; llama.cpp `sdot4` on all RDNA2

How to read the status words:

| Word | Meaning in this brief |
|---|---|
| **first-class** | Named in the engine's or AMD's official GPU list, and shipped in a prebuilt image/wheel for that family |
| **buildable** | Present in a compile-time arch list (`HIP_SUPPORTED_ARCHS`, `PYTORCH_ROCM_ARCH`) so a from-source build can emit code |
| **community-run** | GitHub issues show people serving models; not an official support claim |
| **gated off** | Source explicitly refuses the path (arch check, `sys.exit`, decorator) |
| **hardware-absent** | ISA / GPUOpen: the instruction does not exist |
| **fallback** | What actually runs when the fast path is off |
| **unknown** | Not in any page opened this pass |

vLLM's own `on_gfx1x()` / `_ON_RDNA` mean **gfx11 or gfx12**, not gfx1030. That is the most common source of false "RDNA includes 6800" claims. Quote: `_ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])` in https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py

---

## 0. One page — what is real on Radeon

| Path | gfx1030 (RX 6800 / 6900 / W6800) | gfx1100 (RX 7900 / W7900) | CDNA (MI250 gfx90a / MI300 gfx942 / MI350 gfx950) | Fallback if the fast path is off |
|---|---|---|---|---|
| **Official engine support** | **Not listed.** vLLM docs name MI200/MI300/MI350 + RX 7900 (gfx1100/1101) + RX 9000 + Ryzen AI. AMD ROCm AI page lists Instinct + 7900/7800/7700/7600/9070. **No 6800/6900.** Buildable: `HIP_SUPPORTED_ARCHS` includes `gfx1030`. Community-run: issues #38107, #41622, #22590. | **First-class in vLLM docs.** First-class on AMD ROCm vLLM/SGLang pages. AMD ships a separate `rocm/vllm:…_rdna_…` image and `torch[device-gfx1100]` wheels. | **First-class.** MI200/MI300/MI350 in vLLM docs. AMD `rocm/vllm:…_cdna_…` image + AITER wheel. | From-source `PYTORCH_ROCM_ARCH=gfx1030` (vLLM). SGLang: patch `sgl-kernel` allowlist (not on `main`). |
| **AITER (library)** | **Gated off.** `is_aiter_found_and_supported()` requires `get_cdna_version() > 2` (gfx942+). Official AITER GPU table is gfx90a / gfx942 / gfx950 only. | **Gated off for CK/ASM.** Same CDNA3+ check. `is_rdna_aiter_enabled()` is **RDNA4 (gfx12) only**. AMD SGLang page: disable AITER on Radeon. | **First-class** on gfx942/gfx950. gfx90a is in AITER's table but vLLM's `get_cdna_version()>2` excludes it from the "supported" helper. | Triton / ROCM_ATTN / torch. |
| **GEMM: hipBLASLt** | **Not a first-class hipBLASLt target.** hipBLASLt README required HW: gfx90a, gfx94x, gfx110x. TheRock #1062: gfx10xx "technically supported by tensilelite" but **excluded from default hipBLASLt** (no extops). hipBLASLt #648: PyTorch falls back to hipBLAS/rocBLAS when the GPU is missing. | **Listed** (gfx110x). WMMA-capable. | **Listed** (gfx90a / gfx94x / gfx950). MFMA. | rocBLAS Tensile / hipBLAS / `torch.nn.functional.linear`. |
| **GEMM: AITER CK / hipb_mm / Triton GEMM** | **Gated off** (`@if_aiter_supported`). | **Gated off** (not CDNA3+, not RDNA4). | **On** when `VLLM_ROCM_USE_AITER=1`. hipb_mm needs CDNA version > 2. | PyTorch linear / hipBLAS. |
| **GEMM: vLLM skinny (`wvSplitK`)** | **Not in the gfx9 / gfx1x dispatch.** PR #34709 enabled gfx11/gfx12; previously gfx9-only. gfx1030 is neither. | **Yes if that PR is in your tree** (`on_gfx1x()`). MFMA path still `#ifdef __HIP__GFX9__`. | **Yes** (original). MFMA inner. | `torch.nn.functional.linear` (unquant) / `torch._scaled_mm` (FP8, where FP8 exists). |
| **GEMM: CK / CUTLASS-style** | **unknown** as a vLLM-selected first-class path on gfx1030. CK FA is CDNA-only (vLLM PR #32944). | **unknown** for dense GEMM. WMMA exists in hardware (GPUOpen). | CK + AITER assembly. | — |
| **Attention: AITER FA / MLA / unified** | **Gated off.** `is_mha_enabled()` / `is_mla_enabled()` use `@if_aiter_supported`. PR #32944: AITER FA strictly gfx9. Tests: `ROCM_AITER_FA` raises on gfx1x. | **Gated off** (same). AMD opt guide calls `ROCM_ATTN` / `TRITON_MLA` the "Radeon / fallback". | **Recommended** with `VLLM_ROCM_USE_AITER=1` (`ROCM_AITER_FA` / `ROCM_AITER_MLA`). | ROCM_ATTN or TRITON_ATTN / TRITON_MLA. |
| **Attention: CK FlashAttention** | **No.** PR #32944: "Flash Attention's CK backend only supports CDNA (gfx90a/gfx942/gfx950)." | **No** (same). | **Yes** (default FA-2 backend on Instinct). | Triton FA if opted in; else SDPA / ROCM_ATTN. |
| **Attention: Triton FA (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`)** | **Not in the enablement path.** `flash_attn_triton_available()` requires `on_gfx1x()`. gfx1030 is gfx10. Whether a hand-built Triton FA kernel *could* run is **unknown** (AITER #210: "If you can run Triton kernels [on] gfx1030, then yes"). | **Opt-in for ViT** (PR #32944). Main decode still ROCM_ATTN / TRITON_ATTN, not this flag. AMD pip install for Radeon sets the env var. | Unused; CK / AITER FA win. | Torch SDPA (ViT). |
| **Attention: ROCM_ATTN** | **Intended Radeon/fallback backend** (AMD opt guide; vLLM blog 2026-02-27). Prefill: Triton. Decode: HIP paged-attn **only if** `use_rocm_custom_paged_attention` is true — on non-CDNA that requires `_ON_GFX1X`. **gfx1030 therefore does not take the custom HIP decode kernel.** | **Yes.** Custom HIP decode if head=128, block=16, GQA 3–16, fp16/bf16, no alibi, `kv_cache_dtype=auto`. Else Triton decode. | Also available; AITER FA is preferred when enabled. | Triton decode when HIP constraints miss. |
| **Attention: TRITON_ATTN / TRITON_MLA** | **Portable fallback.** Users force `--attention-backend TRITON_ATTN` on gfx1030 (#38107, #41622). | **Portable fallback.** | Baseline; slower than AITER on Instinct (AMD opt guide). | — |
| **Paged KV write** | Native HIP reshape for block 16/32; else Triton writer (`rocm_attn.py`). Custom *decode* HIP: no (see above). | Same write path. Custom decode HIP: yes under the gfx1x constraints. | Broader custom-decode constraints (head 64 or 128, block 16 or 32, GQA 1–16). | Triton writer / Triton decode. |
| **Fused GEMM-epilogue (SiLU, RMSNorm, RoPE)** | **AITER fused ops gated off.** Default IR priority is `vllm_c` / `native`, not `aiter` (RMSNorm aiter only if AITER on and not RDNA4). | Same AITER gate. WMMA epilogue fusion in hipBLASLt: **unknown** as a vLLM-selected path. | AITER RMSNorm, fused add+quant, fused QK-norm-RoPE-cache, SiLU+mul+FP8 quant. | Unfused torch / Triton. |
| **Quant: FP8** | **`supports_fp8()` is false** (`on_cdna() or on_rdna4()`). Do not write an FP8 kernel expecting vLLM to dispatch it. | **`supports_fp8()` is false** (RDNA3 is not RDNA4). Hardware FP8 on gfx1100: **unknown** in pages opened here; vLLM will not claim FP8. | **Yes.** FNUZ on gfx94; E4M3FN otherwise. AITER W8A8 / blockscale / hipb_mm. | — |
| **Quant: INT8 / INT4 / DP4A** | **Hardware: `V_DOT4` / `V_DOT8` / `sdot4` (ISA).** llama.cpp uses `__builtin_amdgcn_sdot4` on all RDNA2 (#8629). **vLLM has no first-class DP4A GEMM on gfx1030** (RDNA3 GPTQ kernels are `#ifdef VLLM_ROCM_GFX1100`). | Hardware DOT4 + WMMA IU8/IU4. vLLM: `gptq_gemm_rdna3` / `_wmma` / `moe_gptq_gemm_rdna3` under `VLLM_ROCM_GFX1100`. | MFMA INT8/INT4 + AITER quant GEMM. | Triton AWQ (`VLLM_USE_TRITON_AWQ=1` forced on ROCm). Generic GPTQ HIP if compiled. |
| **Quant: AWQ / GPTQ / GGML-style** | AWQ: Triton (ROCm forces `VLLM_USE_TRITON_AWQ`). GPTQ: generic HIP if present; **not** the gfx1100 WMMA/DOT2 kernels. GGML-style: **not a vLLM/SGLang path** — that is llama.cpp. | AWQ: Triton. GPTQ/W4A16: dedicated RDNA3 HIP (PR #44075) when the binary was built with `gfx1100`. | AWQ Triton or AITER; GPTQ HIP; Quark FP8/MXFP4. | Triton WNA16 MoE if no native kernel. |
| **MoE** | AITER fused-MoE **gated off**. No `moe_gptq_gemm_rdna3`. Triton fused-MoE if the model path uses it — **untuned; unknown quality on gfx1030**. | AITER MoE gated off. Native W4A16 MoE HIP if gfx1100 build (PR #44075). Else Triton. | AITER fused-MoE default with `VLLM_ROCM_USE_AITER=1`. | Triton `fused_moe_kernel_gptq_awq`. |
| **All-reduce / RCCL** | `use_custom_allreduce()` is **gfx94/gfx95 only**. Quick Reduce is documented for MI300-class. Multi-GPU = RCCL / NCCL-compat. | Same: no custom AR, no Quick Reduce claim. | Custom AR + AITER fused AR+RMSNorm; Quick Reduce; `NCCL_MIN_NCHANNELS=112` on MI300. | RCCL. |
| **SGLang `sgl-kernel`** | **Gated off on `main`.** `setup_rocm.py`: `if amdgpu_target not in ["gfx942","gfx950"]: sys.exit(1)`. gfx1030 PR #20925 closed (fictional SKU name). **Not on AMD's SGLang GPU list.** | **Not on `main` allowlist.** AMD ROCm 7.14: "initial SGLang support for Radeon" + **disable AITER**. Tracking issue #30599 (open, 2026-07). Community: builds after a one-line allowlist; decode still broken in the reports opened here. | **First-class** gfx942/gfx950. | Triton / torch_native after patching; AITER must stay off on Radeon. |

**Read the table this way:** on a 6800/6900 you are not on the AITER/MFMA/FP8/custom-AR island, and you are also not on vLLM's `on_gfx1x()` island (custom paged-attn, Triton-FA opt-in, skinny GEMM, GPTQ-RDNA3). What remains is Triton, generic HIP/rocBLAS, and whatever you write. On a 7900 you are first-class in the *docs*, still off AITER/FP8/custom-AR, but you do get WMMA, gfx1x paged-attn, and (if built) RDNA3 GPTQ/MoE HIP.

---

## 1. vLLM ROCm support matrix

### 1.1 What the official docs list

vLLM GPU install (opened): https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html and https://docs.vllm.cc/en/latest/getting_started/installation/gpu/

> GPU: **MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201), Ryzen AI MAX / AI 300 Series (gfx1151/1150)**
> ROCm 6.3 or above. MI350 needs ROCm 7.0+. Ryzen AI needs ROCm 7.0.2+.

**gfx1030 / RX 6800 / 6900 / W6800 is not in that sentence.**

AMD ROCm AI ecosystem vLLM page (opened): https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html

Instinct: MI355X/MI350X/MI350P (gfx950), MI325X/MI300X/MI300A (gfx942).

Radeon: RX 9070* (gfx1201), RX 9060* (gfx1200), **RX 7900 XTX/XT/GRE (gfx1100)**, RX 7800/7700 (gfx1101), RX 7600 (gfx1102), PRO W7900/W7800 (gfx1100), W7700/V710 (gfx1101), AI PRO R9700/R9600.

**No RX 6800 / 6900 / W6800 / gfx1030.**

vLLM source-build examples in the same docs set `PYTORCH_ROCM_ARCH` to `gfx942` or `gfx90a;gfx942` — Instinct examples, not Radeon.

### 1.2 Docker tags

| Image | Who | What it is for |
|---|---|---|
| `vllm/vllm-openai-rocm:latest` / `:nightly` | vLLM project | Official ROCm serve image (docs, 2026). AMD's older `rocm/vllm` / `rocm/vllm-dev` marked **deprecated** in favor of this (docs: "Prior to January 20th, 2026…"). |
| `rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0` | AMD | Instinct. Pip path also installs an **AITER** wheel from `vllm-cdna/`. |
| `rocm/vllm:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0` | AMD | Radeon (the SKUs in §1.1). Pip path: `vllm-rdna/` wheel, **no AITER wheel**, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. Flash-attn wheel is `py3-none-any` (Triton), not the CDNA `cp314` build. |

Source: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html and https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html

There is **no** `*_gfx1030_*` or `*_rdna2_*` tag in the pages opened.

### 1.3 Buildable vs first-class

vLLM `CMakeLists.txt` (opened via search hits on `main` / recent commits):

```
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;…;gfx1200;gfx1201")
```

https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt

So **gfx1030 is a legal HIP offload target**. That is not the same as "first-class." Users who set `ARG_PYTORCH_ROCM_ARCH=gfx1030` have hit CMake falling back to gfx906/gfx942 when `GPU_TARGETS` was not propagated (issue #22590, PR #31079).

Community-run evidence (not a support promise):
- https://github.com/vllm-project/vllm/issues/38107 — W6800 / V620 / 6900 XT / 6800 XT, `dtype=bfloat16` → single-digit tok/s (no HW BF16).
- https://github.com/vllm-project/vllm/issues/41622 — W6800 gfx1030, LoRA + TP crash; `--attention-backend TRITON_ATTN --dtype float16`.
- https://github.com/vllm-project/vllm/issues/22590 — Docker build with `ARG_PYTORCH_ROCM_ARCH=gfx1030`.

### 1.4 How vLLM classifies the chip at runtime

From https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py (opened raw):

```
_ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
_ON_GFX1100 = "gfx1100" in _GCN_ARCH
_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
_ON_CDNA = any(arch in _GCN_ARCH for arch in ["gfx9", "gfx1250"])
_ON_RDNA = _ON_GFX1X and not _ON_CDNA
_ON_RDNA4 = any(arch in _GCN_ARCH for arch in ["gfx1200", "gfx1201"])
```

| Query | gfx1030 | gfx1100 | gfx942 |
|---|---|---|---|
| `on_cdna()` | no | no | yes |
| `on_gfx1x()` / `on_rdna()` | **no** | yes | no |
| `on_rdna4()` | no | no | no |
| `supports_fp8()` | **no** | **no** | yes |
| `supports_mx()` | no | no | gfx950 only |
| `use_custom_allreduce()` | no | no | yes (gfx94/95) |
| `is_navi()` (`"gfx1" in arch`) | **yes** (string match) | yes | no |

`is_navi()` matching gfx1030 is a string accident (`gfx1030` contains `gfx1`). Do not treat it as "Navi 3." The functions that matter for kernels are `on_gfx1x()` and `on_cdna()`.

`get_cdna_version()`: gfx90a→2, gfx942→3, gfx950→4, else **0**.

BF16 check uses `has_device_capability(80)`. gfx1030 parses as capability **(10, 3)**, so the check **passes** even though GPUOpen's WMMA table lists BF16 as **N/A** on RX 6950 XT. That is the #38107 foot-gun: vLLM will accept `--dtype bfloat16` / `auto` and then every GEMM emulates.

---

## 2. vLLM hot-path map

AITER gate used everywhere below, from https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py (opened raw):

```
def is_aiter_found_and_supported() -> bool:
    … return get_cdna_version() > 2   # gfx942+

def if_aiter_supported(func):
    … if is_aiter_found_and_supported(): return func(...)
    return None

# is_enabled / is_linear_enabled / is_mha_enabled / is_mla_enabled /
# is_fused_moe_enabled / is_triton_rotary_embed_enabled / …
# all carry @if_aiter_supported

def is_rdna_aiter_enabled() -> bool:
    return on_rdna4() and cls._AITER_ENABLED   # gfx12 only
```

AMD's own opt guide (Instinct-titled, but it names the Radeon fallback): https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html

> The Radeon/fallback backends (`ROCM_ATTN`, `TRITON_MLA`) … do not use AITER and do not require the env var.

vLLM blog (2026-02-27; search snippet, fetch timed out — treat quotes as from the search extract): https://vllm.ai/blog/2026-02-27-rocm-attention-backend

> Radeon GPU support: Along with `TRITON_ATTN`, this backend [`ROCM_ATTN`] supports Radeon GPUs—useful for consumer hardware deployments where AITER primitives aren't available.

### 2.1 GEMM / hipBLASLt / rocBLAS / CK / Triton / AITER

| Mechanism | gfx1030 | gfx1100 | CDNA | Source |
|---|---|---|---|---|
| **AITER linear / W8A8 CK / hipb_mm / Triton unquant GEMM** | off | off | on if `VLLM_ROCM_USE_AITER=1` | `_aiter_ops.py` `@if_aiter_supported`; hipb_mm also `get_cdna_version()>2` |
| **hipBLASLt as a library** | excluded from default supported-target list; TensileLite "legacy, no extops" | gfx110x required-HW | gfx90a / gfx94x | https://github.com/ROCm/hipBLASLt README; https://github.com/ROCm/TheRock/issues/1062 |
| **PyTorch GEMM** | hipBLASLt missing → **hipBLAS / rocBLAS Tensile** (or generic) | hipBLASLt preferred if `TORCH_BLAS_PREFER_HIPBLASLT=1` | same + AITER wrappers | hipBLASLt #648 (AngryLoki: "tied to mfma (gfx9) or wmma (gfx11)"; PyTorch falls back); AMD opt guide |
| **vLLM skinny `wvSplitK`** | **not** `on_gfx9()` and **not** `on_gfx1x()` | yes after PR #34709 | yes (MFMA) | https://github.com/vllm-project/vllm/pull/34709 — "Previously these kernels were gfx9-only … RDNA falling through to `torch.nn.functional.linear`" |
| **CK GEMM inside vLLM** | unknown as a selected backend | unknown | used via AITER (`gemm_a8w8_CK`) | `_aiter_ops.py` |
| **CUTLASS-style** | vLLM CUTLASS is a CUDA fetch path in the CUDA docs. ROCm equivalent is CK/AITER. **No gfx1030 CUTLASS claim.** | — | — | vLLM GPU install "Use the local cutlass" is under NVIDIA. |

rocBLAS itself: "hipBLASLt is used as the default backend for problems on the gfx12 architecture"; Tensile otherwise. https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html — **does not name gfx1030 as a hipBLASLt default.**

### 2.2 Attention

Backend priority for dense MHA (`_get_backend_priorities` in `rocm.py`, opened):

1. `ROCM_ATTN` (unless KV-connector)
2. `ROCM_AITER_FA` if `is_mha_enabled()` — **False on Radeon**
3. `ROCM_AITER_UNIFIED_ATTN` if aiter found+supported **or** `is_rdna_aiter_enabled()` — **False on gfx1030 and gfx1100**
4. `TRITON_ATTN`
5. `TURBOQUANT`

MLA: AITER MLA family if `is_mla_enabled()`, else `TRITON_MLA` only.

Custom HIP paged-attention (`use_rocm_custom_paged_attention`):

- **CDNA:** fp16/bf16, head 64 or 128, block 16 or 32, GQA 1–16, seq ≤ 128k, no sinks.
- **else:** requires `_ON_GFX1X` **and** head==128, block==16, GQA 3–16, no alibi, `kv_cache_dtype=="auto"`.
- **gfx1030 hits the else branch and fails `_ON_GFX1X` → returns False.**

RDNA4 later relaxed GQA/block and added software FP8 KV (commit ebaeda8, 2026-05-07). **gfx11 keeps the tight limits; gfx1030 is not in that commit.**

ROCM_ATTN implementation notes (https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py):
- Native C++ paged attn: block 16 or 32 (LDS).
- Head sizes listed: 32…256.
- Prefill: Triton. Decode: HIP when the custom kernel accepts the shape, else Triton.
- `fused_rope_kvcache_supported()` returns `rocm_aiter_ops.is_enabled()` — **False on Radeon**.

ViT:
- CDNA: AITER FA if MHA on, else CK `flash_attn`.
- gfx1x: Triton FA if `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`.
- else: Torch SDPA.
- **gfx1030 → SDPA** unless someone later extends `on_gfx1x()`.

CK vs Triton FA (ROCm model-accel page): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html — CK default, Triton if `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. ROCm/flash-attention README: Triton backend "supports AMD's CDNA (MI200, MI300) **and RDNA** GPUs" — that sentence does **not** name gfx1030. vLLM only *opts in* on gfx1x.

### 2.3 Fused epilogues

| Op | gfx1030 / gfx1100 | CDNA |
|---|---|---|
| AITER RMSNorm / fused add+dynamic quant / SiLU+mul+FP8 group quant / fused QK-norm-RoPE-cache | off (`@if_aiter_supported`; FP8 extra-dead on these chips) | on with AITER flags |
| Default IR priority | `native` (inductor) or `vllm_c, native` | `aiter` prepended for RMSNorm when AITER+cudagraph and not RDNA4 |
| hipBLASLt fused epilogue | unknown as a vLLM-selected path | used via AITER hipb_mm on CDNA3+ |

`get_default_ir_op_priority` in `rocm.py` (opened).

### 2.4 Quantization

`RocmPlatform.supported_quantization` lists awq, gptq, fp8, quark, mxfp4, … — that is a **method name allowlist**, not a per-arch kernel guarantee. The real gates:

| Method | gfx1030 | gfx1100 | CDNA |
|---|---|---|---|
| **FP8 W8A8 / PTPC / KV-cache** | `supports_fp8()` false | `supports_fp8()` false | yes; FNUZ on gfx94 |
| **MXFP4 / MXFP8** | `supports_mx()` false | false | gfx950 / gfx1250 |
| **AWQ** | Triton (`verify_quantization` forces `VLLM_USE_TRITON_AWQ=1`) | same | same, plus AITER options |
| **GPTQ dense** | generic HIP if compiled; **not** `gptq_gemm_rdna3` | `gptq_gemm_rdna3` + `_wmma` if `VLLM_ROCM_GFX1100` | generic / AITER |
| **GPTQ/AWQ MoE** | Triton fallback | `moe_gptq_gemm_rdna3` (DOT2 + exllama dequant) if gfx1100 build | AITER fused-MoE |
| **INT8 DP4A / sdot4** | **silicon yes, vLLM dispatch no** | silicon yes (DOT4 + WMMA IU8); vLLM W4A16 uses DOT2 not DP4A | MFMA |
| **GGML Q4/Q5/Q6/Q8** | not a vLLM path | not a vLLM path | not a vLLM path |

`csrc/rocm/torch_bindings.cpp` (search extract): `gptq_gemm_rdna3*` and `moe_gptq_gemm_rdna3` behind `#ifdef VLLM_ROCM_GFX1100`.

PR #44075: native HIP W4A16 MoE on gfx1100 using `v_dot2_f32_f16` / `v_dot2_f32_bf16`; falls through to Triton WNA16 otherwise.

### 2.5 MoE, all-reduce

- AITER fused-MoE: `@if_aiter_supported` → CDNA3+ only.
- Custom all-reduce: `gfx94` / `gfx95` only (`rocm.py`).
- Quick Reduce: documented under MI300-class in the AMD opt guide. **No Radeon claim.**
- Multi-GPU on Radeon: RCCL via the NCCL process-group path (`dist_backend = "nccl"` in `RocmPlatform`). XGMI "fully connected" is an Instinct topology.

---

## 3. SGLang map

### 3.1 Official support

AMD ROCm AI SGLang page (opened): https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html

- Instinct: same MI355/MI350/MI325/MI300 list as vLLM.
- Radeon: 9070/9060/7900/7800/7700/7600 + PRO W7900/W7800 — **no gfx1030**.
- Version named: SGLang 0.15.3post1. Image: `rocm/sgl-dev:v0.5.13.post1-ubuntu24.04-py3.14-rocm7.14` (same tag for Instinct and Radeon in the page).
- **Known issue (Radeon):** "ROCm 7.14 introduces **initial** SGLang support for AMD Radeon GPUs. Radeon GPU users should **disable AITER** and unset `SGLANG_ROCM_FUSED_DECODE_MLA`." Some MoE (GPT-OSS-20B, MiniMax-M2.7) and Qwen3-ASR "may not function correctly on Radeon."

SGLang's own hardware docs URL `https://docs.sglang.io/docs/hardware-platforms/amd_gpu.html` **404'd** this pass. Cookbooks that loaded (GLM-5 / GLM-5.1) only name **MI300X/MI325X/MI355X**.

### 3.2 `sgl-kernel` allowlist (`main`)

https://github.com/sgl-project/sglang/issues/30599 (opened) quotes current `main`:

```
# sgl-kernel/setup_rocm.py:75-79
if amdgpu_target not in ["gfx942","gfx950"]:
    sys.exit(1)
```

Same text appears in the search extract of https://github.com/sgl-project/sglang/blob/3b26644b/sgl-kernel/setup_rocm.py (LDS budget comments: gfx942 48 KB dynamic smem; gfx95x 128 KB).

**Consequence:** `moe_align_block_size`, `topk_softmax`, `silu_and_mul`, `rotary_embedding`, … **do not build for RDNA on `main`.** Loading a gfx942 wheel on gfx1100 produced garbage `expert_ids` and a page-fault (#30245, cited in #30599).

gfx1030 PR https://github.com/sgl-project/sglang/pull/20925 — closed. Author admitted the SKU name was fictional; the gfx1030 allowlist change was not re-landed under an honest title in the pages opened here.

### 3.3 gfx1100 community status (2026-07, still open)

Issue #30599 (tracking, open):

| Piece | Status on `main` (as of comments 2026-07-17) |
|---|---|
| Official RDNA3/4 support | **No.** Roadmap #23494 names Instinct only. |
| Allowlist for gfx1100 | Merged to `amd_march` (PR #30415, 2026-07-07), **not `main`**. |
| Build + `import sgl_kernel` on gfx1100 | Works after adding gfx1100 to `RDNA_TARGETS` + wave32 `WARP_SIZE` (PR #31137 cluster). |
| Serve prefill | Reported working (Qwen2.5-7B, `--attention-backend triton`). |
| Serve decode | **Still failing** in the opened reports (hang / illegal memory access). Not pinned to a named kernel. |
| AITER | Eager imports in ~36 files; must be optional/disabled on Radeon. |
| Custom / quick all-reduce | Guarded off on RDNA (wave64 / peer-IPC). Fall back to RCCL. |

Issue #27519: gfx1101 (7800 XT) compiles and serves LFM2.5 with a one-line whitelist. That is a community recipe, not `main`.

### 3.4 SGLang hot paths (what exists vs what Radeon gets)

| Path | gfx1030 | gfx1100 | CDNA |
|---|---|---|---|
| **sgl-kernel HIP** (silu_and_mul, RoPE, topk_softmax, moe_align, …) | no (`sys.exit`) | no on `main`; yes after allowlist + wave32 | yes |
| **AITER FA / fused decode MLA / fused MoE** | n/a | **must disable** (AMD page) | default in the ROCm image |
| **Triton attention** | unknown (no official recipe) | intended Radeon path; decode still reported broken on gfx1100 in #30599 | available; AITER preferred |
| **FlashInfer** | CUDA wheels; must uninstall on ROCm (#30599) | same | not the AMD path |
| **RMSNorm** | would need torch-native; vLLM fused op signature mismatch on HIP (#31403) | same | AITER |
| **FP8 / MXFP4** | no | no first-class claim; AMD page does not list FP8 for Radeon SGLang | yes (cookbooks) |
| **All-reduce** | RCCL if you get that far | RCCL; custom AR gated | AITER custom AR (EAGLE needs `--disable-custom-all-reduce` on Instinct — GLM-5.1 cookbook) |

---

## 4. Concrete holes — CDNA kernels that die or degrade on gfx1030

Quoted from files/issues that were opened. This is the "where a custom kernel is a real win" list.

### 4.1 Hardware holes (cannot enable; do not try)

| Missing | Why | Source |
|---|---|---|
| **MFMA / AGPR** | CDNA-only. GPUOpen WMMA table: RDNA2 has packed DOT, not matrix cores. | https://gpuopen.com/learn/wmma_on_rdna3/ ; companion brief |
| **WMMA 16×16×16** | gfx11+. | same |
| **BF16 as a first-class matrix type** | GPUOpen: BF16 `N/A` on RX 6950 XT. vLLM still *accepts* bf16 via capability 10.3 ≥ 8.0. | GPUOpen; `rocm.py` `check_if_supports_dtype`; issue #38107 |
| **FP8 dispatch** | `supports_fp8() = on_cdna() or on_rdna4()`. | `rocm.py` |
| **buffer→LDS async copy** | AITER PR #3336: `vmem-to-lds-load-insts` is gfx940/941/942/950. RDNA 10/11/12 fall back to load-VGPR-then-LDS or fail to compile. | https://github.com/ROCm/aiter/pull/3336 |
| **AITER `fmha_v3` ASM** | Gated `gfx942`/`gfx950`. | https://github.com/ROCm/aiter/pull/3330 |
| **hipBLASLt extops / default build** | gfx10xx excluded; extops use `.amdhsa_accum_offset` (gfx90a+). | TheRock #1062; hipBLASLt #648 |

### 4.2 Software gates that fire on gfx1030 even when the silicon could do *something*

| Gate | What you lose | Quote / file |
|---|---|---|
| `@if_aiter_supported` / `get_cdna_version()>2` | Entire AITER surface: FA, MLA, fused MoE, W8A8 GEMM, RMSNorm fused quant, RoPE Triton, fused AR+RMS, hipb_mm | `_aiter_ops.py` |
| `on_gfx1x()` false | Custom HIP paged-attn decode; ViT Triton FA opt-in; skinny `wvSplitK`; RDNA4 AITER-Triton analog | `rocm.py` `use_rocm_custom_paged_attention`, `flash_attn_triton_available`, PR #34709 |
| `#ifdef VLLM_ROCM_GFX1100` | `gptq_gemm_rdna3`, `gptq_gemm_rdna3_wmma`, `moe_gptq_gemm_rdna3` | `csrc/rocm/torch_bindings.cpp` |
| `use_custom_allreduce()` | HIP custom AR | `rocm.py`: `return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])` |
| CK FlashAttention | FA-2 CK | PR #32944: "CK backend only supports CDNA" |
| SGLang `setup_rocm.py` allowlist | All `sgl_kernel` HIP ops | #30599 quote |
| AITER official GPU table | No RDNA row | https://rocm.github.io/aiter/ — CDNA 2/3/3.5 only |

### 4.3 Degrade-not-disable (you still run, slowly)

| Symptom | What actually runs | Source |
|---|---|---|
| Decode GEMM M=1..4 | `torch.nn.functional.linear` (no wvSplitK) | PR #34709 |
| Attention decode | Triton (custom HIP paged-attn false) | `rocm.py` + AMD opt guide |
| AWQ | Triton AWQ | `verify_quantization` |
| Prefill attention | Triton (ROCM_ATTN prefill path) | vLLM blog extract; `rocm_attn.py` |
| BF16 model / `dtype=auto` | Software BF16 GEMM, single-digit tok/s | #38107 |
| hipBLASLt-shaped PyTorch | hipBLAS/rocBLAS Tensile, not LT | hipBLASLt #648 |
| SGLang (if forced) | Missing HIP ops → wrong expert ids / decode hang | #30599, #30245 |

AITER #210 (Triton MHA on gfx1030): maintainer said "If you can run Triton kernels rdna2/gfx1030, then yes it should run." That is **not** a vLLM enablement, and Triton-on-gfx1030 is not independently confirmed in this pass.

---

## 5. What to write first on gfx1030 (ranked)

Conservative rule: only rank a kernel if (a) the map shows the engine is **not** already shipping a first-class path, (b) the silicon **does** have the op, and (c) the hole is on the decode or quant hot path. No invented speedups.

llama.cpp contrast (known, opened): `ggml_cuda_dp4a` → `__builtin_amdgcn_sdot4` on `CDNA || RDNA2` (commit #8629, all gfx103x). https://github.com/ggml-org/llama.cpp/commit/46e47417aa4f18c08738afd4d9a3e838e97ca03f and https://github.com/ggml-org/llama.cpp/blob/6f165c1c/ggml/src/ggml-cuda/common.cuh

| Rank | Write this | Why the map says it is a win | Do not confuse with |
|---|---|---|---|
| **1** | **INT8 / INT4 GEMM via `sdot4` / `sdot8` (`v_dot4c_i32_i8`, `v_dot8_i32_i4`), wave32, LDS tile ≤ 32 KB** | Silicon is there (RDNA2 ISA). llama.cpp already uses it on every RDNA2 SKU. vLLM/SGLang have **no** DP4A/sdot4 dispatch on gfx1030. AWQ/GPTQ go through Triton. This is the largest engine-vs-silicon gap. | gfx1100 `gptq_gemm_rdna3` (WMMA/DOT2, `#ifdef VLLM_ROCM_GFX1100`). AITER W8A8 CK (CDNA). |
| **2** | **Decode skinny GEMM, M=1..4, FP16→F32 accum, `V_DOT2_F32_F16`** | wvSplitK is gfx9 + gfx1x only. gfx1030 decode linear is `torch.nn.functional.linear`. Decode *is* the serving hot path. hipBLASLt is not a first-class gfx1030 backend. | AITER hipb_mm. WMMA 16×16 (gfx1100). MFMA wvSplitK (CDNA). |
| **3** | **Paged-attention decode HIP (VALU / DOT2, head 64 and 128, block 16, GQA ≥ 1)** | `use_rocm_custom_paged_attention` is false on gfx1030, so ROCM_ATTN decode is Triton. CDNA gets a broader HIP kernel; gfx1x gets a narrow one. You are writing the missing third row. Keep LDS ≤ 64 KB; start at WG=128 for d=256 (llama.cpp #24672 occupancy 0). | AITER `pa_fwd_asm` / `paged_attention_common` (CDNA). gfx1x WMMA paged-attn (commit ebaeda8). |
| **4** | **W4A16 GPTQ/AWQ fused decode (dequant + DOT2), optional MoE routing in-kernel** | gfx1100 already got this (PR #44075) because Triton WNA16 was the bottleneck. gfx1030 still on Triton AWQ. Same dequant+DOT2 idea, **without** WMMA. | `moe_gptq_gemm_rdna3` (will not register). AITER fused-MoE. |
| **5** | **Fused `silu_and_mul` + RMSNorm + RoPE (plain VALU, wave32)** | AITER fused epilogues are CDNA. SGLang's HIP versions do not build. These are small kernels but they are launch-bound on decode. Only rank this after 1–4; the map does not prove they dominate GEMM/attn. | AITER fused QK-norm-RoPE-cache (CDNA, often FP8). |

**Explicitly not ranked** (map does not support a win claim):

- FP8 anything — `supports_fp8()` is false; no HW claim opened for gfx1030.
- AITER FA / CK FA / `fmha_v3` — gated + often ASM/MFMA.
- Custom all-reduce / Quick Reduce — MI300-class only.
- BF16 GEMM — hardware N/A; force `--dtype float16` instead.
- SGLang decode kernel — the remaining gfx1100 fault is **unidentified** in #30599; do not "fix SGLang decode" until that kernel is named.
- WMMA tiles — gfx1100-only. On gfx1100, rank WMMA GEMM / FA *after* you confirm hipBLASLt already covers the shape (it is a listed gfx110x target).

gfx1100 addendum (not the gfx1030 ranking): first-class vLLM + hipBLASLt + WMMA + gfx1x paged-attn + optional GPTQ HIP. The remaining first-class holes vs CDNA are **AITER FA/MoE/RMS**, **FP8**, **custom AR**. A custom kernel on 7900 is a win only where those are the bottleneck *and* hipBLASLt/ROCM_ATTN already miss — measure before writing.

---

## 6. llama.cpp HIP (contrast only)

llama.cpp is not vLLM. It is the existence proof that RDNA2 silicon can serve LLMs at tens of tok/s when the inner loop is DOT/sdot4 and dtype is fp16/quant, not bf16.

| Item | llama.cpp HIP | vLLM / SGLang on gfx1030 |
|---|---|---|
| `sdot4` / DP4A | all RDNA2 (`#8629`) | no dispatch |
| FA tile | custom HIP; occupancy pitfalls at 256 thr / d=256 (#24672) | Triton / no custom paged-attn |
| WMMA FA | gfx1100+ (`GGML_HIP_ROCWMMA_FATTN`) | n/a on gfx1030 |
| Official arch | you pass `-DAMDGPU_TARGETS=gfx1030` | not in the support sentence |

---

## 7. Unknowns (do not invent)

| Question | Status |
|---|---|
| Does a stock vLLM wheel (`vllm/vllm-openai-rocm` or `wheels.vllm.ai/rocm`) contain gfx1030 code objects? | **unknown.** Docs do not say. CMake *can* build it. |
| Exact Tensile `MT*` list shipped for gfx1030 in the ROCm you run | **unknown.** rocBLAS historically is not an MFMA/WMMA first-class target here. |
| Triton language support quality on gfx1030 (compiler + FA kernel) | **unknown.** AITER #210 is a conditional "if Triton runs." One hipBLASLt #648 comment claims "Triton has no support for gfx101x and gfx103x" — that is a 2024 user comment, not a Triton project statement. |
| hipBLASLt *if* you force a TensileLite gfx1030 library (TheRock #1062 community PRs) | community WIP; not default. Do not depend on it. |
| vLLM first-class FA Br/Bc table for gfx1030 | not in opened pages (already noted in the LDS brief) |
| Whether SGLang decode-on-gfx1100 is paged-KV, Triton attn, or something else | **unconfirmed** (#30599 author stopped short) |
| FP8 hardware on gfx1100 | vLLM `supports_fp8()` is false regardless |
| SGLang `docs/platforms/amd_gpu.md` current text | URL 404 this pass |

---

## 8. Sources actually opened

### Official docs / blogs
1. https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html — vLLM ROCm GPU list, wheels, `vllm/vllm-openai-rocm`, deprecated `rocm/vllm`
2. https://docs.vllm.cc/en/latest/getting_started/installation/gpu/ — same list
3. https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html — Instinct vs Radeon SKUs, `_cdna_` / `_rdna_` Docker + pip, AITER wheel on CDNA only
4. https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html — SGLang SKUs, disable AITER on Radeon, ROCm 7.14 "initial" Radeon support
5. https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html — AITER flags, Radeon fallback `ROCM_ATTN` / `TRITON_MLA`, skinny GEMM env, Quick Reduce, FP8/MX on Instinct
6. https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html — FA CK vs Triton env var
7. https://rocm.github.io/aiter/ — AITER GPU table: gfx90a / gfx942 / gfx950 only
8. https://rocm.docs.amd.com/projects/rocBLAS/en/latest/how-to/what-is-rocblas.html — Tensile vs hipBLASLt (gfx12 default)
9. https://gpuopen.com/learn/wmma_on_rdna3/ — no WMMA/MFMA on RDNA2; BF16 N/A
10. https://vllm.ai/blog/2026-02-27-rocm-attention-backend — search extract (full fetch timed out): ROCM_ATTN / TRITON_ATTN are the Radeon backends
11. https://docs.vllm.ai/en/stable/features/ — Feature×Hardware is a single "AMD" column; not per-gfx

### vLLM source / issues / PRs
12. https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py — arch flags, fp8, custom AR, paged-attn, backend priority, ViT FA
13. https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py — `@if_aiter_supported`, RDNA4-only aiter analog
14. https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt — `HIP_SUPPORTED_ARCHS` includes gfx1030 and gfx1100
15. https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/rocm_attn.py — ROCM_ATTN 2-path, block 16/32
16. https://github.com/vllm-project/vllm/blob/7c2acd38/csrc/rocm/torch_bindings.cpp — `VLLM_ROCM_GFX1100` GPTQ/MoE ops
17. https://github.com/vllm-project/vllm/pull/32944 — CK FA CDNA-only; Triton FA opt-in on RDNA3/4; AITER FA gfx9-only
18. https://github.com/vllm-project/vllm/pull/34709 — wvSplitK gfx9 → gfx1x; gfx1030 still excluded
19. https://github.com/vllm-project/vllm/pull/44075 — W4A16 MoE HIP gfx1100
20. https://github.com/vllm-project/vllm/pull/38454 — tests: AITER FA rejected on gfx1x
21. https://github.com/vllm-project/vllm/issues/38107 — gfx1030 bf16 disaster
22. https://github.com/vllm-project/vllm/issues/41622 — gfx1030 LoRA/TP; TRITON_ATTN
23. https://github.com/vllm-project/vllm/issues/22590 + PR #31079 — Docker `PYTORCH_ROCM_ARCH=gfx1030`
24. https://github.com/vllm-project/vllm/issues/4514 — historical gfx1100 Triton FA stack-frame (2024)
25. https://github.com/vllm-project/vllm/commit/ebaeda88562bfde2d870e6f1f9d31d4df82e0106 — RDNA4 paged-attn FP8; gfx11 limits unchanged

### SGLang
26. https://github.com/sgl-project/sglang/issues/30599 — RDNA3/4 tracking; allowlist quote; decode still broken on gfx1100
27. https://github.com/sgl-project/sglang/issues/27519 — gfx1101 whitelist recipe
28. https://github.com/sgl-project/sglang/pull/20925 — gfx1030 PR closed
29. https://github.com/sgl-project/sglang/blob/3b26644b/sgl-kernel/setup_rocm.py — search extract: gfx942/gfx950 only

### AITER / hipBLASLt / llama.cpp
30. https://github.com/ROCm/aiter/issues/210 — Triton MHA on gfx1030: conditional yes
31. https://github.com/ROCm/aiter/pull/3330 — fmha_v3 gfx942/950; RDNA scalar fallbacks
32. https://github.com/ROCm/aiter/pull/3336 — buffer-to-LDS CDNA-only
33. https://github.com/ROCm/hipBLASLt README — required HW gfx90a / gfx94x / gfx110x
34. https://github.com/ROCm/TheRock/issues/1062 — gfx10xx excluded from default hipBLASLt
35. https://github.com/ROCm/hipBLASLt/issues/648 — gfx10+ request; MFMA/WMMA tie; PyTorch fallback
36. https://github.com/ROCm/flash-attention README — Triton FA "CDNA and RDNA" (no gfx1030 name)
37. https://github.com/ggml-org/llama.cpp/commit/46e47417aa4f18c08738afd4d9a3e838e97ca03f — sdot4 on all RDNA2
38. https://github.com/ggml-org/llama.cpp/blob/6f165c1c/ggml/src/ggml-cuda/common.cuh — `ggml_cuda_dp4a`
39. https://github.com/ggml-org/llama.cpp/issues/24672 — gfx1030 FA occupancy

### Local
40. `silicon/architecture.md`
41. `silicon/lds-tiles.md`
