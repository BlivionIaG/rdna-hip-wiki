# gfx1030 HIP kernel-craft brief (RDNA 2)

Audience: someone about to write **W4A16 / W8A8 / mxfp4** HIP for gfx1030. This is the compiler / ISA sheet, not the silicon floorplan and not a tile recipe.

Companions (already on disk; not re-derived):
- `silicon/architecture.md` — WGP/CU, DOT vs WMMA/MFMA, caches
- `silicon/lds-tiles.md` — 64-bank math, seed tiles
- `silicon/codegen-stack.md` — what to use / ignore
- `kernels/w4a16.md` — packing, dequant, decode vs prefill

Rule: every concrete number is attributed. If a figure was not in a page opened this pass, it is marked **unknown**.

Date of this pass: **2026-08-17**.

---

## 0. One-page checklist

Do these before the first `hipcc` of a W4A16 / W8A8 / mxfp4 kernel.

| # | Do | Why | Source |
|---|---|---|---|
| 1 | `hipcc --offload-arch=gfx1030`. Default is **wave32 + WGP**. Do **not** add `-mwavefrontsize64` or `-mcumode` until you have a measured reason. | HIP: “HIP doesn’t support `warpSize` of 64 on gfx10 and above.” LLVM default is `-mno-wavefrontsize64`, `-mno-cumode`. | HIP C++ language extensions; LLVM AMDGPUUsage Target Features |
| 2 | Pin occupancy: `__launch_bounds__(256, 4)` **or** `__attribute__((amdgpu_waves_per_eu(4, 8)))` + `__attribute__((amdgpu_flat_work_group_size(256, 256)))`. EU = **one SIMD32**. | Unconstrained compile assumes 1024-thread blocks and will over-allocate VGPR. 4 waves/SIMD ⇒ VGPR ≤ 256; 8 waves ⇒ VGPR ≤ 128; 16 waves ⇒ VGPR ≤ 64. | HIP `__launch_bounds__`; Clang `amdgpu_waves_per_eu`; LLVM `getTotalNumVGPRs` = 1024 |
| 3 | Align `__shared__` tiles: `alignas(16)` on every 128-bit view; allocation is **1 KB granule, 1 KB aligned**. Cap **64 KB/WG**. Before capture: `TILE×head×el×stages+256 ≤ 65536` ([lds-tiles.md](lds-tiles.md) Radiance clamp). | ISA §3.6.6. `ds_read_b128` needs 16 B alignment (LLVM D92767). No HIP builtin for `ds_read_b128`. | RDNA 2 ISA 70648; LLVM `ds_read_b128`; radiance `patch_unified_attention_lds.py` @ 22c69cd |
| 4 | In the K-loop: wait **`lgkmcnt`** for LDS, **`vmcnt`** for vector loads, **`vscnt`** for vector stores. Never `s_waitcnt vmcnt(0)` “to be safe” — that does **not** wait for stores on RDNA. | RDNA split the store counter. Non-atomic stores are fire-and-forget. `s_waitcnt vscnt(0)` belongs before `s_barrier` / atomics. | GPUOpen RDNA architecture deck |
| 5 | `__syncthreads()` is enough for LDS reuse. Before a **global** atomic or a store that another WG will read: wait `vscnt(0)` (compiler usually does this). `atomicAdd` = **device/agent**. `atomicAdd_system` + `__threadfence_system` only on **fine-grained** memory. `hipMalloc` is **coarse-grained**; system-scope on it is **downgraded**. | HIP fences; ROCm gpu-atomics; HIP coherence control. | HIP C++ language extensions; https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html |
| 6 | `export HIP_FORCE_DEV_KERNARG=1` (HIP 7.14 **default is already 1**). Keep kernarg + `__constant__` small; I$ is **32 KB/WGP**. | Device kernarg = 2–3 µs launch win. Giant unrolled epilogues miss I$. | HIP env vars; kfd_crat / RDNA deck |
| 7 | Issue the **builtin**. `__builtin_amdgcn_fdot2` (W4A16), `__builtin_amdgcn_sdot4` (W8A8), optional `__builtin_amdgcn_sdot8` (packed i4). hipcc will **not** peephole `__hfma2` into DOT2. | vLLM `q_gemm_rdna3.cu`; LLVM `dot10-insts` / `dot1-insts`. | Clang AMDGPU builtins; llama.cpp #8629 |
| 8 | Occupancy gate: `llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --wg-size=256 --vgprs=N --lds=K` **and** a runtime occupancy query. Reject spills. A **203 VGPR / 256-thread** tile is **occupancy 0** on gfx1030. | Compiler occupancy ≠ launchable occupancy. | LLVM CommandGuide; llama.cpp #24672 |
| 9 | Force `--dtype float16`. No WMMA, no MFMA, no bf16 DOT, no CDNA 5-issue, no `is_navi()` / `"gfx1"` string match. | GPUOpen WMMA table; HIP hardware-implementation CDNA paragraph; vLLM `rocm.py`. | §7 |

**Build line**

```
hipcc --offload-arch=gfx1030 -O3 \
  -Rpass-analysis=kernel-resource-usage \
  -save-temps \
  kernel.hip -o kernel
# optional, measure both, do not ship on a guess:
#   -mcumode
#   -mwavefrontsize64   # LLVM yes; HIP runtime says no on gfx10+
```

**Pre-filter**

```
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 \
  --wg-size=256 --vgprs=N --lds=K
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --limits
```

Reject: VGPR/SGPR spill, LDS > 65536, `hipOccupancyMaxActiveBlocksPerMultiprocessor` == 0.

---

## 1. Wave32 vs wave64, WGP vs CU mode

### 1.1 Hardware (ISA, not HIP paraphrase)

RDNA 2 ISA §2.1 (Document ID **70648**): hardware is **natively wave32**. Wave64 VALU/VMEM is the same instruction issued twice (low 32, then high 32). Either half may be skipped if `EXEC` for that half is 0 (except VALU ops that write an SGPR/VCC).

RDNA 2 ISA §2.3.1 / §10.3:

| | CU mode | WGP mode |
|---|---|---|
| SIMDs used by one workgroup | 2 (one CU) | 4 (whole WGP) |
| LDS visible | 64 KB half attached to that CU | full 128 KB address space |
| Workgroup LDS cap | still **64 KB** | still **64 KB** |
| ISA reason | “higher LDS memory bandwidth”; both halves run in parallel | more ALU + texture bandwidth for a WG of ≥ 4 waves |
| Far-side LDS | illegal | legal, “performance may be lower in some cases” |

A single workgroup **cannot** allocate more than 64 KB even in WGP mode. The extra 64 KB is so **two** workgroups (or the two CU-mode halves) can each hold 64 KB.

VGPR allocation (ISA §3.6.4): groups of **16 dwords for wave32**, **8 dwords for wave64**. Addressable `V0–V255`. Physical file 1024 VGPR/SIMD32 in wave32 accounting (LLVM `getTotalNumVGPRs`; RDNA deck).

Wave slots: **16 per SIMD32** on gfx1030 (LLVM `getMaxWavesPerEU`; GPUOpen Occupancy explained). SGPRs do **not** limit occupancy on GFX10+ (LLVM `isSGPROccupancyLimited` false; GPUOpen).

### 1.2 LLVM / hipcc flags

LLVM AMDGPUUsage “Target Features” (https://llvm.org/docs/AMDGPUUsage.html):

| Feature | Clang flag | Default on gfx1030 | Meaning |
|---|---|---|---|
| `wavefrontsize64` | `-m[no-]wavefrontsize64` | **off** → wave32 | “When disabled native wavefront size 32 is used, when enabled wavefront size 64 is used.” |
| `cumode` | `-m[no-]cumode` | **off** → WGP | “When disabled native WGP wavefront execution mode is used, when enabled CU wavefront execution mode is used.” |

`gfx1030` processor row lists both features as **supported**, not as defaults. Triple: `amdgcn-amd-amdhsa--gfx1030` / `amdgpu-amd-amdhsa`. Generic: `gfx10-3-generic` covers gfx1030–1036 with **no ISA restrictions**.

HIP C++ language extensions (https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html):

- `warpSize` is **32** on gfx10+. “HIP doesn’t support `warpSize` of 64 on gfx10 and above.”
- “`mwavefrontsize64` compiler option is not supported by HIP runtime.”
- ROCm 7.0+: `warpSize` is early-folded (usable in loop bounds) but is **not** a host-side compile-time constant. Query `hipDeviceAttributeWarpSize`.

So: LLVM will compile `-mwavefrontsize64`. The **HIP runtime** does not advertise wave64 on gfx10+. Treat wave64 as an LLVM experiment, not a shippable HIP default. Measure only if you have a wide-reduction reason; it doubles VGPR cost per wave and turns every VALU into two beats (ISA §2.1).

Dedicated occupancy page: [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md) (launch-mode axis; sibling of [wave-size-occupancy.md](wave-size-occupancy.md)).

**When `-mcumode` is worth measuring:** the kernel is **LDS-bandwidth bound**, the workgroup is happy on 2 SIMD32s, and you are not using the second CU’s ALUs. Attention softmax scratch is the candidate. A VALU-bound 8-wave GEMM that wants 4 SIMD stays in WGP mode.

### 1.3 `amdgpu_waves_per_eu` vs `__launch_bounds__`

Two knobs. They are not the same sentence.

**Clang `amdgpu_waves_per_eu`** (https://clang.llvm.org/docs/AttributeReference.html#amdgpu-waves-per-eu):

```c
__attribute__((amdgpu_waves_per_eu(min[, max])))
__attribute__((amdgpu_flat_work_group_size(min, max)))
```

- Optimization hint. Backend **limits VGPR / LDS / scratch** so that at least `min` and at most `max` waves fit on an EU.
- `0, 0` = no limit. Warning if the backend cannot meet it; **error** if values violate the subtarget or conflict with other attributes.
- `amdgpu_num_vgpr` / `amdgpu_num_sgpr` are **deprecated**; use `waves_per_eu`.
- `amdgpu_flat_work_group_size(0, 0)` implies default **128, 256**. Helps barrier codegen and scratch promotion.
- LLVM IR: `"amdgpu-waves-per-eu"="m,n"`, `"amdgpu-flat-work-group-size"="min,max"`.
- If the range is incompatible with `"amdgpu-flat-work-group-size"`, **the workgroup-size occupancy bound wins** (LLVM AMDGPUUsage Attributes).

**HIP `__launch_bounds__`** (HIP C++ language extensions):

```c
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)
kernel(...);
```

- `MAX_THREADS_PER_BLOCK`: programmer guarantee. Default = device max (1024). HIP **validates the launch** and errors if the block is larger. Reducing it lets the compiler spend more VGPR/thread.
- `MIN_WARPS_PER_EXECUTION_UNIT`: optional, **defaults to 1**. Lower bound on occupancy. Compiler derives a VGPR cap ≈ `available_registers / MIN_WARPS`.
- “The compiler can only use the hints to manage **register** usage, and does **not** automatically reduce shared memory usage. The compilation **fails** if the compiler cannot generate code that satisfies the launch bounds.”
- GPUOpen lab notes (Laplacian part 3): default second argument is 1; set the first argument to your real block size (they use 256).

**EU on RDNA = one SIMD32**, not an ISA CU. GPUOpen Occupancy explained and LLVM `getMaxWavesPerEU` both count per SIMD32. HIP’s own `__launch_bounds__` page then says “On AMD GPUs a Compute Unit consists of 4 Execution Units” — that is the **GCN/WGP** picture (4 SIMD32 = one WGP), not the ISA CU (2 SIMD32). `rocminfo` “SIMDs per CU: 4” is the same confusion (architecture brief §1.1). Read `MIN_WARPS_PER_EXECUTION_UNIT` as **waves per SIMD32**.

Wave32 occupancy ladder (1024 VGPR, granule 16):

| `waves_per_eu` / `__launch_bounds__` 2nd arg | Max VGPR/lane | Waves/WGP (4 SIMD) | 256-thr WGs/WGP (8 waves) |
|---|---|---|---|
| 16 | 64 | 64 | 8 (LDS will cap first) |
| 8 | 128 | 32 | 4 |
| 4 | 256 | 16 | 2 |
| 2 | 256 (file cap) | 8 | 1 |

Seeds for this project:

```c
// decode skinny (W4A16 M=1..4): want 16 waves if you can stay ≤ 64 VGPR
__attribute__((amdgpu_waves_per_eu(8, 16)))
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void __launch_bounds__(256, 8) w4a16_decode(...);

// prefill / W8A8 sdot4: 4–8 waves is the realistic band
__attribute__((amdgpu_waves_per_eu(4, 8)))
__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void __launch_bounds__(256, 4) w8a8_prefill(...);
```

Do **not** ship `__launch_bounds__(1024)` on a 256-thread kernel. GPUOpen: the compiler then budgets registers for a block you will never launch.

### 1.4 `__shared__` alignment

ISA §3.6.6: LDS allocations are **256 dwords (1024 B), 256-dword aligned**, no wrap. A “33-column” pad still rounds the *workgroup* allocation up to a kilobyte.

HIP: `__shared__` is LDS. Static size is compile-time; `extern __shared__` size is the 5th launch-config argument. HIP does **not** document a byte-alignment guarantee beyond C++ type alignment.

Craft rules that *are* sourced:

| Rule | Why |
|---|---|
| `alignas(16)` on every `__shared__` array you intend to load as `int4` / `float4` / `half8` | LLVM: `ds_read_b128` requires **16-byte** alignment unless `unaligned-access-mode` (D92767). Otherwise you get `ds_read2_b64` or a split. |
| Do not walk LDS as scalar `half` | Bank map is dword-based; 2-way conflict (LDS brief §1.5). Pack `half2`. |
| Dynamic `extern __shared__` + several views | Declare `extern __shared__ alignas(16) char smem[];` then slice. ISA allocation base is 1 KB aligned, so the *start* is 16-byte aligned; your *offsets* may not be. |
| Cap 64 KB | ISA §2.3.1. Two WGs/WGP ⇒ each ≤ 32 KB after 1 KB rounding. |

There is **no** HIP / Clang builtin named `__builtin_amdgcn_ds_read_b128`. Exposure is: aligned vector load from `addrspace(3)`, or inline asm. Historical LLVM generated `ds_read_b128` only under `-amdgpu-ds128` (D44210, 2018); current selection prefers `ds_read_b128` over `ds_read2_b64` when alignment is 16 B (D92767). **Verify the `.s`.** Cycle count of `ds_read_b128` on gfx1030 is **unknown** in the ISA (LDS brief).

---

## 2. `s_waitcnt`: `vmcnt` vs `vscnt` vs `lgkmcnt`

### 2.1 The counters (gfx1030)

RDNA split the GCN combined load/store counter. GPUOpen RDNA architecture deck (https://gpuopen.com/download/RDNA_Architecture_public.pdf), “Load / Store Queues - RDNA”:

- Vector **loads** increment **VMCNT**.
- Vector **stores** increment **VSCNT**.
- “Non atomic stores are now true **fire and forget**.”
- “It’s very likely that you will see `s_waitcnt vscnt(0)` only in front of a `s_barrier` or in front of atomic operations.”
- `s_endpgm` implicitly waits on all counters.

LLVM gfx1030 waitcnt operand (https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.0.0/LLVM/llvm/html/AMDGPU/gfx1030_waitcnt.html):

| Field | Bits | Counts | Range |
|---|---|---|---|
| `vmcnt` | 15:14 + 3:0 | outstanding **vector memory** ops (loads / atomics that return) | 0..63 |
| `expcnt` | 6:4 | exports (graphics; ignore in compute) | 0..7 |
| `lgkmcnt` | 13:8 | **LDS, GDS, Constant (SMEM), Message** | 0..63 |

`vscnt` is **not** a field of `s_waitcnt` on gfx10. It is a separate instruction:

```
s_waitcnt_vscnt null, N     ; wait until VSCNT <= N
```

LLVM `SIInsertWaitcnts.cpp` names: `LOAD_CNT` = VMcnt prior to gfx12; `DS_CNT` = LGKMcnt prior to gfx12; `STORE_CNT` = VScnt on gfx10/gfx11. `vscnt` is used to resolve **memory** dependencies (SIMemoryLegalizer), not data dependencies. The compiler does **not** wait `vscnt(0)` on function entry/return (llvm-project `f2c164c`).

`s_waitcnt` syntax (LLVM):

```
s_waitcnt vmcnt(0)                    ; all vector loads done
s_waitcnt lgkmcnt(0)                  ; all LDS / SMEM / messages done
s_waitcnt vmcnt(1) lgkmcnt(2)         ; leave 1 VMEM and 2 LGKM outstanding
s_waitcnt_vscnt null, 0               ; all vector stores done
```

`N` means “wait until **at most N** ops of that class are still outstanding.” `vmcnt(0)` = drain. `vmcnt(2)` = software-pipeline: issue 3 loads, wait until 2 remain, use the oldest.

### 2.2 Which wait, when

| Situation | Wait | Do **not** |
|---|---|---|
| About to **use** a `global_load` / `buffer_load` / `flat_load` result | `s_waitcnt vmcnt(k)` | `lgkmcnt` (wrong pipe) |
| About to **use** an LDS / `ds_*` / `__shared__` result | `s_waitcnt lgkmcnt(k)` | `vmcnt` |
| About to **use** an `s_load` / `__constant__` / kernarg dword | `s_waitcnt lgkmcnt(k)` | — |
| About to `__syncthreads()` / `s_barrier` after LDS **stores** | compiler emits `lgkmcnt(0)` (+ `vmcnt(0)` + `vscnt(0)` if the backend has no auto-wait-before-barrier) | relying on `vmcnt(0)` alone to publish LDS |
| About to issue a **global atomic** that must see prior stores to the same address | `s_waitcnt_vscnt null, 0` (and usually `vmcnt(0)`) | assuming GCN-style `vmcnt(0)` covers stores |
| About to **reuse** an LDS address another wave just wrote | `__syncthreads()` (implies the LDS wait). Intra-wave: `lgkmcnt(0)` is enough; no barrier. | a global fence |
| End of kernel | `s_endpgm` waits everything | an extra drain in the epilogue “for safety” |

LLVM on `s_barrier` (no `hasAutoWaitcntBeforeBarrier`, no `supportsBackOffBarrier`): inserts `allZero` **including vscnt** (`SIInsertWaitcnts`). You normally do **not** write the wait yourself next to `__syncthreads()`. You **do** write it (or check the `.s`) when you use inline asm, or when you publish a global buffer that another workgroup will consume without a kernel boundary.

HIP `__syncthreads()` “implicitly include[s] a threadfence, thereby ensuring visibility of memory accesses for the threads in the group” (HIP C++ language extensions). That is **block** scope, not device scope.

### 2.3 Inner-loop pattern (DOT2 / sdot4)

```
; software-pipeline, wave32
ds_read_b64     v[a0:a1], v[ldsA]          ; or two b32 / one b128 if aligned
ds_read_b64     v[b0:b1], v[ldsB]
s_waitcnt       lgkmcnt(0)
v_dot2c_f32_f16 acc, a, b                  ; or v_dot4c_i32_i8
; prefetch next K into LDS from global — do NOT vmcnt(0) here
global_load_dwordx4 v[...], ...
; wait only the loads you are about to dequant
s_waitcnt       vmcnt(N)                   ; N = still-in-flight prefetch
```

Companion LDS brief already said this. The foot-gun is copying a GCN snippet that uses `s_waitcnt vmcnt(0)` as a universal drain.

---

## 3. Memory scopes and HIP atomics

### 3.1 AMDHSA scopes (LLVM)

LLVM AMDGPUUsage “Memory Scopes” (https://llvm.org/docs/AMDGPUUsage.html), OS `amdhsa`. Inclusive HRF-indirect (HSA). Concurrent atomics only compose if scopes include each other.

| LLVM syncscope | HIP surface | Who it includes |
|---|---|---|
| `system` (default / none) | `__threadfence_system`, `atomicAdd_system` | all agents (this GPU, peer GPUs, host) |
| `agent` | `__threadfence`, `atomicAdd` on device memory | threads on **this device** |
| `workgroup` | `__threadfence_block`, `__syncthreads` | this workgroup |
| `wavefront` | (implicit in warp shuffles / DPP) | this wave |
| `singlethread` | — | this lane (signal handlers) |
| `*-one-as` | — | same as the parent scope, **one address space only** |

`cluster` exists in the table; on targets without cluster launch it **behaves like `agent`**. gfx1030 has no cluster.

### 3.2 HIP fences

HIP C++ language extensions:

| Fence | Scope |
|---|---|
| `__threadfence_block()` | all threads in the **block** |
| `__threadfence()` | all threads on the **device** |
| `__threadfence_system()` | all threads in the **system** (other devices + host) |

`__syncthreads()` = block barrier **plus** a block-scope fence.

W4A16 decode that CAS-reduces into a pre-zeroed C (vLLM `atomic_add_pk4_f16`): that is **device-scope** on `hipMalloc` memory. `__threadfence()` after the CAS is only needed if another thread on this GPU must see it *before* kernel end. Another kernel sees it at the stream sync. Do **not** upgrade to `_system` “to be safe” — that is a different, slower path (below).

### 3.3 `atomicAdd` vs `atomicAdd_system`

HIP: unsuffixed atomics operate on **shared or global memory on the executing device**, depending on the address space. `_system` suffix = system scope (host + other GPUs).

ROCm “Hardware atomics operation support” (https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html), covers gfx9/10/11/12:

| Memory | What atomics do |
|---|---|
| **Coarse-grained** (`hipMalloc` default; `hipExtMallocWithFlags` + `hipDeviceMallocDefault`) | Cacheable. Atomics process in **L2**. Legal for **device-scope** sync inside one kernel. **System-scope is downgraded to device-scope.** |
| **Fine-grained device** (`hipExtMallocWithFlags` + `hipDeviceMallocFinegrained`) | Write-uncacheable via page tables. Atomics that hit it go to **Infinity Fabric**. Needed for true system-scope. |
| **Fine-grained system** (typical `hipHostMalloc` / `hipMallocManaged`) | Device-scope atomics still hit L2; system-scope **bypasses L2** to Infinity Fabric. |

HIP coherence control (https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html):

| API | Flag | Coherence |
|---|---|---|
| `hipMalloc` / `hipExtMallocWithFlags` + default | — | **Coarse-grained** |
| `hipExtMallocWithFlags` | `hipDeviceMallocFinegrained` | Fine-grained |
| `hipHostMalloc` | default | Fine-grained |
| `hipHostMalloc` | `hipHostMallocNonCoherent` | Coarse-grained |
| `hipMallocManaged` | default | Fine-grained |
| `hipMallocManaged` | `hipMemAdviseSetCoarseGrain` | Coarse-grained |

Fine-grained “can be useful if both host and device operate on the same data.” Cost: “many AMD GPUs use a limited cache policy, such as leaving these allocations uncached by the GPU or making them read-only.” Coarse-grained uses the **full L2**.

**gfx1030 craft:**

- Weight / activation / C buffers: `hipMalloc` = coarse, device-scope `atomicAdd` / 64-bit CAS. This is the vLLM W4A16 path. Correct.
- Host-visible progress flags / multi-GPU: allocate **fine-grained**, use `atomicAdd_system` + `__threadfence_system`. Check `rocminfo` `Pool Info` for `FINE GRAINED` on the GPU agent — consumer Navi 21 often has **only coarse-grained** on the device agent (the architecture brief’s rocminfo caveat; confirm on the V620). If the pool is missing, you cannot get a real system-scope device atomic.
- LLVM metadata for native system-scope integer atomics: `!amdgpu.no.fine.grained.memory` and `!amdgpu.no.remote.memory` (AMDGPUUsage). HIP’s `_system` entry points set this for you when the allocation is coarse; do not fight it.
- FP atomics: default HIP-Clang is **safe** (CAS loop). `-munsafe-fp-atomics` / `unsafeAtomicAdd` emit the HW op and are **unsafe on fine-grained** (HIP C++ language extensions; ORNL AMD memory-model notes). W4A16 C reduction is integer CAS into packed f16, not `atomicAdd<float>`. Keep it that way.

### 3.4 What a W4A16 / W8A8 kernel actually needs

| Event | Scope |
|---|---|
| LDS A/B tile reuse | `__syncthreads()` (workgroup) |
| Intra-wave reduce (DPP / `permlane` / `__shfl_xor`) | wavefront; no fence |
| K-split CAS into C | **device** atomic on coarse `hipMalloc` |
| Next kernel reads C | stream / device synchronize (coarse visibility at kernel boundary) |
| Host polling a flag while the kernel runs | fine-grained + `_system` — **do not put this on the GEMM path** |

---

## 4. Kernarg / SMEM / I$

### 4.1 Where arguments live

AMDHSA kernarg is a **scalar** buffer. Waves load it with `s_load_*` into SGPRs; that traffic hits the **scalar data cache (K$)**, not the vector L0. LLVM address space 4 (`constant`) is the same path. RDNA deck: I$ **32 KB/WGP**, K$ **16 KB/WGP**, 64 B lines. Confirmed for Navi 21 by `kfd_crat.c` (architecture brief §4.3).

HIP env var (https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html):

| Var | Default (HIP 7.14) | Meaning |
|---|---|---|
| `HIP_FORCE_DEV_KERNARG` | **1** | “Forces kernel arguments to be stored in **device memory** to reduce latency. Can improve performance by **2–3 µs** for some kernels.” `0` disables. |

vLLM ROCm optimization page: set it explicitly on bare metal; Docker images already set it. Same sentence in the MI300X workload guide — the mechanism is not CDNA-specific.

**Why it matters on gfx1030:** default-off (older ROCm) left kernarg in host-visible memory and every launch paid a tiny extra hit. Decode launches are small and frequent; 2–3 µs is a real fraction of a 30–50 µs qkv. Current HIP defaults it **on**. Still set it in the process environment so a wheel built against an older runtime cannot surprise you.

Hidden / implicit args (`llvm.amdgcn.implicitarg.ptr`, dispatch ptr, queue ptr) sit in the same segment. Attributes `amdgpu-no-implicitarg-ptr`, `amdgpu-no-dispatch-ptr`, … let the backend drop unused ones (LLVM AMDGPUUsage). HIP kernels that never call `gridDim` / `clock64` / printf still often keep the implicit blob — check the `.amdhsa_kernarg_size` in the `.s`.

### 4.2 I$ 32 KB — the unroll foot-gun

RDNA deck: “mind the I$ size.” 32 KB/WGP is shared by 4 SIMD32. A fully unrolled 128×128 epilogue plus four dequant helpers will miss it; every miss is a scalar-instruction stall, not a VMEM stall, and occupancy will not hide it the way it hides GDDR6.

Dedicated locks: [icache-occupancy.md](icache-occupancy.md) (I$; not a PIX MaxWaves row; SQC I$ counters / Take–Leave); [kcache-occupancy.md](kcache-occupancy.md) (K$ / kernarg / `__constant__`; SQC DCache counters / Take–Leave).

Craft:

- Unroll the **inner K** (8–16 DOT2 / sdot4) to cover the 5-cycle VALU dest latency (RDNA deck).
- Do **not** unroll the whole M/N epilogue.
- Keep dequant in one header; vLLM already learned that growing `qdq_4_rdna3.cuh` regresses the decode TU (`q_gemm_rdna3_wmma.cu`).
- Put problem sizes, scales-base, strides in SGPRs / kernarg. Do **not** reload them as vector loads every iteration.
- `__constant__` is the same K$ as kernarg. Uniform scale tables belong there; a **per-lane nibble LUT** in `__constant__` serializes the inner loop (llama.cpp #24438 mechanism). W4A16 uses the `0x64006400` bit-trick so you never do that.

SMEM alignment: SGPRs used as 64-bit / SMEM bases must be **even**; SMEM returns ≥ 4 dwords need **quad** alignment (ISA §3.6.2). The compiler handles this if you do not hand-pack SGPRs in asm.

---

## 5. Intrinsics cheat sheet

Clang AMDGPU builtins (https://clang.llvm.org/docs/AMDGPUBuiltinReference.html) and LLVM AMDGPUUsage IR intrinsics. Feature names are the Clang `TARGET_BUILTIN` predicates. **Clamp is a compile-time constant.**

### 5.1 Packed DOT — this is the inner loop

| Builtin | LLVM intrinsic | ISA | Feature | Use on gfx1030 |
|---|---|---|---|---|
| `float __builtin_amdgcn_fdot2(half2 a, half2 b, float c, bool clamp)` | `llvm.amdgcn.fdot2` | `V_DOT2_F32_F16` / `V_DOT2C_F32_F16` | `dot10-insts` | **W4A16 / FP16 GEMM.** Issue this. hipcc will not emit it from `__hfma2`. |
| `int __builtin_amdgcn_sdot4(int a, int b, int c, bool clamp)` | `llvm.amdgcn.sdot4` | `V_DOT4_I32_I8` / `V_DOT4C_I32_I8` | `dot1-insts` | **W8A8.** 4×i8 → i32. BK % 4 == 0. llama.cpp #8629: all gfx103x. |
| `unsigned __builtin_amdgcn_udot4(unsigned a, unsigned b, unsigned c, bool clamp)` | `llvm.amdgcn.udot4` | `V_DOT4_U32_U8` | `dot7-insts` | Unsigned twin. |
| `int __builtin_amdgcn_sdot8(int a, int b, int c, bool clamp)` | `llvm.amdgcn.sdot8` | `V_DOT8_I32_I4` / `V_DOT8C_I32_I4` | `dot1-insts` | 8×i4 → i32. GPUOpen: **1024** IU4 ops/clk/CU. **Not** a drop-in for W4A16 (per-group fp16 scale). mxfp4 / integer-accum experiment. |
| `unsigned __builtin_amdgcn_udot8(...)` | `llvm.amdgcn.udot8` | `V_DOT8_U32_U4` | `dot7-insts` | Unsigned twin. |
| `int __builtin_amdgcn_sdot2(short2, short2, int, bool)` | `llvm.amdgcn.sdot2` | `V_DOT2_I32_I16` | `dot2-insts` | Packed i16. Rare for LLM. |
| `__builtin_amdgcn_fdot2_f32_bf16` | `llvm.amdgcn.fdot2.f32.bf16` | `V_DOT2_F32_BF16` | **`dot12-insts` (gfx11+)** | **ABSENT.** Do not call. |
| `__builtin_amdgcn_sudot4` / `sudot8` | `llvm.amdgcn.sudot4` | `V_DOT4_I32_IU8` | gfx11 | **ABSENT.** llama.cpp RDNA3 path. |
| `__builtin_amdgcn_wmma_*` | `v_wmma_*` | RDNA 3 ISA §7.9 | `wmma-256b-insts` | **ABSENT.** hipcc errors. |
| `v_mfma_*` / AGPR | — | CDNA | `FeatureMAIInsts` | **ABSENT.** |

Signatures (Clang RFC / BuiltinsAMDGPU.td, HIP/C++ takes `_Float16` for `fdot2` as of llvm-project `ba1b867`):

```c
float acc = __builtin_amdgcn_fdot2(a_h2, b_h2, acc, false);
int   acc = __builtin_amdgcn_sdot4(a_i32, b_i32, acc, false);
int   acc = __builtin_amdgcn_sdot8(a_i32, b_i32, acc, false);
```

`clamp=false` lowers to the `C` (carry / VOP2) form when legal (`v_dot2c_*`, `v_dot4c_i32_i8`). Same math, cheaper encoding.

GPUOpen WMMA-on-RDNA3 table (official RDNA2 vs RDNA3 FLOPS/clk/CU): RX 6950 XT FP16 **256**, BF16 **N/A**, IU8 **512**, IU4 **1024**.

### 5.2 DPP / permlane / reduce

| Builtin / intrinsic | ISA | Notes |
|---|---|---|
| `__builtin_amdgcn_permlane16(old, src, src1, src2, fi, bc)` | `v_permlane16_b32` | Gather inside a 16-lane row. `src1`/`src2` scalar; packed into a 64-bit select. `fi`/`bc` compile-time. |
| `__builtin_amdgcn_permlanex16(...)` | `v_permlanex16_b32` | Cross the two 16-lane rows of a wave32. |
| `__builtin_amdgcn_permlane64` | `v_permlane64_b32` | Swaps wave64 halves. **No-op in wave32.** |
| `__builtin_amdgcn_update_dpp(old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)` | `v_mov_b32` + DPP | Preferred over deprecated `mov.dpp`. |
| `llvm.amdgcn.wave.reduce.{add,fadd,min,fmin,max,fmax,and,or,xor}` | DPP or iterative | Hint: 0 default, 1 iterative, 2 DPP. |
| HIP `__shfl` / `__shfl_xor` / `__shfl_down` | backend-picked | Portable. Masks are **64-bit** in HIP even on wave32 (high bits unused). |
| HIP `__reduce_*_sync` | backend-picked | ROCm 7+; extra types behind `HIP_ENABLE_EXTRA_WARP_SYNC_TYPES`. |
| `__builtin_amdgcn_readlane` / `readfirstlane` / `writelane` | `v_readlane_b32` etc. | Uniform-ize a pointer / scale. |

Intra-wave softmax / row-reduce: **DPP or `permlane`, not LDS** (LDS brief §2.2). vLLM skinny `REDUCE_SUM_DPP_WAVE32` encodings were written for gfx11; **verify against ISA 70648 DPP table** before copying (W4A16 brief §8).

### 5.3 LDS 128-bit — if exposed

| Path | Exposed? |
|---|---|
| ISA `ds_read_b128` / `ds_write_b128` | Yes (RDNA 2 ISA ch. 10; LLVM `AMDGPUAsmGFX1030`) |
| Clang `__builtin_amdgcn_ds_read_b128` | **No** (not in AMDGPUBuiltinReference) |
| HIP `int4` / `float4` load from `alignas(16) __shared__` | Yes, if the backend selects `ds_read_b128` (alignment 16 B; D92767) |
| Inline asm `"ds_read_b128 $0, $1"` | Yes, last resort |

ISA peak wording is “32 × 32-bit or `read2`/`write2` 64-bit.” A 128-bit op is **width-limited even without a bank conflict**. Prefer two `b64` if the `.s` shows `b128` stretching the loop. Cycle count: **unknown**.

### 5.4 Other builtins you will actually touch

| Builtin | Use |
|---|---|
| `__builtin_amdgcn_kernarg_segment_ptr()` | AS(4) pointer to kernarg. Rarely needed; HIP lowering does this. |
| `__builtin_amdgcn_implicitarg_ptr()` | Implicit blob after explicit args. |
| `__builtin_amdgcn_s_barrier` | Under `__syncthreads()`. Do not call unless you are in asm. |
| `__builtin_amdgcn_sched_barrier(mask)` / `sched_group_barrier` | Instruction-group scheduling. Mask bits include VALU / VMEM / DS / TRANS. MFMA/WMMA bit is dead on gfx1030. |
| `__builtin_amdgcn_s_buffer_load_*` | Scalar buffer load. “Should generally be avoided” (Clang): separate cache, not coherent with vector stores. Legal for **read-only** descriptors / scales if you accept that contract. |
| `__builtin_amdgcn_global_load_b128` | gfx9+ global 128-bit. Not LDS. |

---

## 6. Occupancy workflow

### 6.1 The math (do not re-derive)

From the architecture brief + LLVM `IsaInfo` + GPUOpen Occupancy explained:

| Resource | gfx1030 wave32 WGP |
|---|---|
| Wave slots / SIMD32 | **16** |
| VGPR file | **1024**, granule **16** |
| SGPR | fixed 106+VCC; **not an occupancy limiter** |
| LDS | 128 KB/WGP, **≤ 64 KB/WG**, granule 1 KB |
| Barriers | 32 in WGP mode, 16 in CU mode (LLVM `getMaxWorkGroupsPerCU`) |
| Max flat WG | 1024 |

Occupancy = `min(slots, VGPR budget, LDS budget, WG-size / barrier slots)`.

### 6.2 `llvm-calc-occupancy`

https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html — “thin front-end over the same occupancy math the AMDGPU backend uses (`GCNSubtarget`).”

```
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 \
  --wg-size=256 --vgprs=72 --sgprs=40 --lds=16k

# VGPR/SGPR caps per occupancy level
llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --limits
```

`--lds` accepts `k`/`kb` (base 1024). Omit a resource to leave it unconstrained. `--wg-size=MIN:MAX` reports a range.

Always pass `-mattr=+wavefrontsize32` so the tool does not silently assume wave64 (the CommandGuide example is `gfx90a` / wave64).

### 6.3 Compile-time remarks

```
hipcc --offload-arch=gfx1030 -O3 -Rpass-analysis=kernel-resource-usage ...
```

Prints VGPR, SGPR, LDS, occupancy (waves/SIMD), spills (ROCm #1609). **Reject any config with VGPR or SGPR spill.** Scratch on gfx1030 is dword-interleaved through L0/L2/IC (LLVM Private address space) — fatal in an inner loop.

RGA (offline, no GPU): `rga -s bin --isa --livereg --analysis` on the `.hsaco`. Live-VGPR column tells you which line bought the extra 16-VGPR granule (GPUOpen RGA).

### 6.4 RGP / rocprof — measured occupancy and spills

GPUOpen Occupancy explained + RGP manuals:

- **Pipeline tab**: VGPR/SGPR/LDS, **spills**, wave32 vs wave64, limiter hint.
- Occupancy is *capacity* to hide latency, not a performance guarantee. Extra waves on a memory-bound KV / weight stream **thrash L0/L1/IC**.
- Linux HPC blog (https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html): RGP HIP is a **Windows** story; Linux Instinct/ROCm workflow is **rocprofv3**.

```
rocprofv3 --pmc LDSBankConflict,VALUBusy,SALUBusy,MemUnitStalled,GPUBusy \
          --output-format csv -- ./harness
```

### 6.5 203 VGPR tile death (llama.cpp #24672)

https://github.com/ggml-org/llama.cpp/issues/24672 and https://github.com/ggml-org/llama.cpp/discussions/23310 — gfx1030/1031 HIP FlashAttention:

| Config | Threads | VGPR | LDS | Compiler occ. | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
|---|---|---|---|---|---|
| decode `<256,256,1,8>` | 128 | 102 | 21 504 B | 6 waves/SIMD | **2** (runs) |
| prefill `<256,256,4,8>` | 256 | **203** | 37 888 B | 4 waves/SIMD | **0** (assert) |

203 VGPR, granule 16 → 208 allocated. 1024/208 = 4 waves/SIMD *in isolation*. A 256-thread WG is 8 waves. 8 waves × 208 VGPR = 1664 > 1024, so the WG **does not fit on one SIMD** — and SPI must place the whole WG on one WGP with enough **per-SIMD** VGPR for the waves that land there. The compiler’s “4 waves/SIMD” assumes it can spread 8 waves across 4 SIMD32s (2 waves/SIMD). That is legal **only if** VGPR/wave × waves_on_that_SIMD ≤ 1024. Two waves × 208 = 416, which fits; the **runtime** still returned 0. Possible causes named in the issue: occupancy API using CU-mode / wave64 accounting, or an extra hidden VGPR/SGPR/scratch that the compiler remark omitted. **The operational rule is not the theory:** if the runtime occupancy query is 0, the kernel does not launch, regardless of `llvm-calc-occupancy`.

Gate:

1. `-Rpass-analysis` — no spill, VGPR ≤ 128 for a 256-thread WG unless you have measured a 2-wave/SIMD win.
2. `llvm-calc-occupancy` — waves/EU ≥ 4 as a starting floor.
3. `hipOccupancyMaxActiveBlocksPerMultiprocessor` (or the CUDA-compat name llama.cpp used) — **≥ 1**.
4. Stay at **128 threads** for d=256 attention / fat prefill until step 3 is green (W4A16 brief §5.3).

---

## 7. Foot-guns

### 7.1 bf16 emulation

GPUOpen WMMA table: BF16 **N/A** on RX 6950 XT. RDNA 2 ISA feature list has `V_DOT2_F32_F16` and the integer DOT family — **no** `V_DOT2_F32_BF16`. That opcode is gfx11 `dot12-insts` (`__builtin_amdgcn_fdot2_f32_bf16`).

vLLM #38107: gfx1030 `dtype=auto` is accepted because HIP reports capability **(10, 3) ≥ 8.0**, then bf16 is **emulated**. Single-digit tok/s. Force `--dtype float16`.

The fp16 dequant bit-trick (`0x64006400`) **overflows** a bf16 mantissa (7 bits vs 10). Every bf16 path uses a different magic (`0x43004300`) and a right-shift — and that path has no DOT2 on this card.

### 7.2 `rocminfo` lying

Architecture brief §1.1, sourced to HIP issue #2238 (https://github.com/ROCm/HIP/issues/2238) gfx1030 dump:

| Field | Dump | Reality (ISA + kernel tables) |
|---|---|---|
| Compute Unit | 80 | **Correct** (ISA CU count) |
| SIMDs per CU | 4 | **WGP**, not ISA CU (CU = 2×SIMD32) |
| Max Waves Per CU | 64 | **WGP** (4 × 16), not CU (2 × 16 = 32) |
| Shader Engines | 8 | Counting **shader arrays**, not marketing SEs (Navi 21 = 4 SE × 2 SA) |
| Wavefront Size | 32 | Correct as the **runtime default** |

HIP `__launch_bounds__` docs repeat “a Compute Unit consists of 4 Execution Units.” Same GCN-era picture. Use ISA + LLVM `IsaInfo`, not rocminfo field names, when you size tiles.

Also check `rocminfo` **Pool Info** before you believe fine-grained device memory exists. A device agent that only lists `COARSE GRAINED` cannot host a real system-scope GPU atomic.

### 7.3 `is_navi()` matches gfx1030 via `"gfx1"`

vLLM `vllm/platforms/rocm.py`:

```python
def on_gfx1x() -> bool:
    return any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])

def is_navi(cls) -> bool:
    return "gfx1" in _GCN_ARCH
```

`"gfx1" in "gfx1030"` is **True**. `on_gfx1x()` is gfx11/12 only and is the **correct** WMMA / HybridW4A16 gate. `is_navi()` is the **wrong** gate: it lumps Navi 21 (no WMMA) with Navi 31 (WMMA). Custom paged-attention historically used `ON_NAVI = "gfx1" in gcnArchName` to **disable** a CDNA kernel — that part is accidentally right for gfx1030, but the same helper used as a positive “this is RDNA3” test is a silent miscompile / garbage-output bug (lemonade-sdk/vllm-rocm #24: gfx1151 running gfx1100 WMMA objects).

In *your* code: match `gfx1030` / `gfx1031` / `gfx10-3` explicitly, or parse major/minor the way `_capability_from_gcn_arch` now does (`gfx` + 2-digit major). Never `s.find("gfx1")`.

### 7.4 Assuming WMMA

There is no `v_wmma_*`, no rocWMMA, no AITER FA, no CK XDL/FMHA, no hipBLASLt ExtOp (codegen-stack brief). hipcc errors on `__builtin_amdgcn_wmma_*`. A 16×16 fragment layout copied from `q_gemm_rdna3_wmma.cu` is not “close.” Prefill is tiled **DOT2**, seed 64×64×32 (W4A16 brief §5.3).

### 7.5 Assuming CDNA 5-issue

HIP Hardware implementation (https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) describes a CDNA CU whose “issue arbiter can issue five instructions per cycle (VALU + VMEM + SALU/SMEM + LDS + branch).” That paragraph is **not** an RDNA 2 WGP fact.

RDNA deck: **1 VALU instruction / cycle / SIMD32**; SALU and VMEM are separate pipes; exact issue-group width beyond that is **not quantified in the ISA**. Transcendentals are ¼ rate and can co-issue with non-transcendental VALU. There is **no** VOPD (`v_dual_*`, LLVM `FeatureVOPDInsts`, gfx11+).

Schedule the inner loop as “one DOT2/sdot4 per cycle per SIMD, hide 5-cycle dest latency with ≥ 5 independent accums or other waves.” Do not write a 5-issue dependency graph and expect it to map.

### 7.6 Other craft traps (short)

| Trap | Source |
|---|---|
| `s_waitcnt vmcnt(0)` as a universal drain | RDNA deck: stores are `vscnt` |
| hipcc not emitting `v_dot2` from `__hfma2` | vLLM `q_gemm_rdna3.cu` |
| `__launch_bounds__` 2nd arg default 1 → compiler spends VGPR like you asked for 1 wave/SIMD | HIP docs |
| HIP “4 EU per CU” when sizing `__launch_bounds__` | HIP docs vs ISA CU = 2 SIMD32 |
| `ds_read_b128` without `alignas(16)` | LLVM D92767 |
| `BLOCK_K > group_size` (wrong scale, silent) | vLLM PR #39705 |
| Capability `(10,3) ≥ 8.0` enabling bf16 | vLLM #38107 |
| Fine-grained `unsafeAtomicAdd` | HIP unsafe-fp-atomics note |
| GDS | LLVM: not implemented for AMDHSA |
| `tgsplit` | gfx1030 does not list it; waves of a WG stay on one WGP/CU |

---

## 8. Unknowns (do not invent)

| Question | Status |
|---|---|
| `ds_read_b128` cycle count / phase groups on gfx1030 | ISA only gives 1–64 cycles for indexed LDS; CK phases are CDNA wave64 |
| Whether current hipcc enables `ds_read_b128` by default without `-amdgpu-ds128` | Selection prefers it at 16 B align (D92767); **verify the `.s`** |
| HIP runtime actually launching a `-mwavefrontsize64` object on gfx1030 | Docs say the option is unsupported; LLVM will still compile it |
| Fine-grained **device** pool on V620 / Navi 21 | Confirm with `rocminfo` Pool Info; do not assume CDNA-style fine-grained HBM |
| gfx1030 native FP64/FP32 atomic throughput | Not opened this pass; W4A16 uses integer CAS |
| DPP `0x118/0x114/0x112/0x111` wave32 reduce encodings vs ISA 70648 | Not verified this pass (W4A16 brief) |
| Exact ACE count / SPI width | HIP says “multiple ACEs”, no number |

---

## 9. Sources actually opened

**ISA / silicon**
1. AMD “RDNA 2” ISA 70648 — https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture — §2.1 wave32/64, §2.3.1 / §10.3 WGP vs CU, §3.6.2 SGPR, §3.6.4 VGPR granule, §3.6.6 LDS alloc, ch. 10 `ds_*`, feature list DOT
2. GPUOpen RDNA architecture deck — https://gpuopen.com/download/RDNA_Architecture_public.pdf — VMCNT/VSCNT split, fire-and-forget stores, 5-cycle dest, I$/K$ 32/16 KB
3. GPUOpen Occupancy explained — https://gpuopen.com/learn/occupancy-explained/ — 16 slots, SGPR not limiting, RGP
4. GPUOpen WMMA on RDNA 3 — https://gpuopen.com/learn/wmma_on_rdna3/ — no WMMA/MFMA/bf16-matrix on RDNA2; FLOPS table
5. Companion `silicon/architecture.md`, `silicon/lds-tiles.md`, `silicon/codegen-stack.md`, `kernels/w4a16.md`

**LLVM / Clang**
6. LLVM AMDGPUUsage — https://llvm.org/docs/AMDGPUUsage.html — `gfx1030` features, `cumode` / `wavefrontsize64`, address spaces, AMDHSA scopes, `amdgpu-waves-per-eu`, fine-grained metadata, DOT / permlane / DPP intrinsics
7. LLVM `llvm-calc-occupancy` — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
8. LLVM gfx1030 waitcnt — https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.0.0/LLVM/llvm/html/AMDGPU/gfx1030_waitcnt.html
9. LLVM `SIInsertWaitcnts` / `f2c164c` — vscnt is memory-legalizer only; barrier waits include vscnt
10. Clang AttributeReference `amdgpu_waves_per_eu` / `amdgpu_flat_work_group_size` — https://clang.llvm.org/docs/AttributeReference.html#amdgpu-waves-per-eu
11. Clang AMDGPU builtins — https://clang.llvm.org/docs/AMDGPUBuiltinReference.html
12. LLVM D44210 / D92767 — `ds_read_b128` generation and 16-byte alignment
13. llvm-project `ba1b867` — `fdot2` HIP `_Float16` signature

**HIP / ROCm**
14. HIP C++ language extensions — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html — `__launch_bounds__`, `__shared__`, `warpSize`, fences, atomics, unsafe FP atomics, “no warpSize 64 on gfx10+”
15. HIP env vars — https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html — `HIP_FORCE_DEV_KERNARG` default 1, 2–3 µs
16. HIP coherence control — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html
17. ROCm hardware atomics — https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html — coarse vs fine, system-scope downgrade, L2 vs Infinity Fabric
18. HIP Hardware implementation — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html — LDS 64 banks, CDNA 5-issue paragraph (do not reuse)
19. vLLM ROCm optimization — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html — `HIP_FORCE_DEV_KERNARG=1`
20. GPUOpen lab notes, Laplacian part 3 — https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-finite-difference-docs-laplacian_part3/ — `__launch_bounds__(256)`

**Bugs / kernels**
21. llama.cpp #24672 / #23310 — 203 VGPR occupancy 0
22. llama.cpp #8629 — `sdot4` on all gfx103x
23. llama.cpp #24438 — K$ LUT serialization
24. vLLM #38107 — gfx1030 bf16 emulation
25. vLLM `platforms/rocm.py` — `is_navi()` = `"gfx1" in arch`; `on_gfx1x()` = gfx11/12
26. HIP #2238 — gfx1030 rocminfo dump
27. lemonade-sdk/vllm-rocm #24 — gfx1151 WMMA garbage
28. vLLM `q_gemm_rdna3.cu` — hipcc does not peephole DOT2
