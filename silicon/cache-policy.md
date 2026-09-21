# RDNA 2 (gfx1030) cache policy for HIP kernel writers

Audience: someone writing custom HIP for weight-only decode (W4A16 / W8A8 / mxfp4) on **4× AMD Radeon PRO V620**. Not a consumer GPU review.

Companion briefs (sizes are taken from these; this note does **not** re-derive them unless a source contradicts):

- `silicon/architecture.md` — L0 16 KB/CU, L1 128 KB/SA, L2 4 MB, IC 128 MB 64 B lines, GDDR6 512 GB/s
- `kernels/w4a16.md` — decode geometry (A in LDS, B streamed, DOT2, no WMMA)
- `silicon/lds-tiles.md` — LDS banks and GEMM/attention tiles

Rule: every concrete number is attributed. If a figure is not in a source that was opened, it is marked **unknown**. Quote only text that was opened. Do **not** invent Infinity Cache TB/s or any cache associativity.

Date of this pass: **2026-08-17**.

---

## 0. What you can actually control from HIP

This is the whole note in one table. Everything below is evidence for these rows.

| Knob | Exposed on gfx1030 from HIP? | What it actually does | Use for decode |
|---|---|---|---|
| Ordinary `load` / `store` of `__global__` | yes (default) | GLC=0, SLC=0, DLC=0. L0 hit-LRU; L1 hit-LRU; L2 LRU. Writes write-through L0/L1 to L2. | **Default for anything you want to reuse** (a layer that fits in IC, the live KV window, activations). You do **not** have a “persist in Infinity Cache” bit. Reuse + working-set size is the persist mechanism. |
| `__builtin_nontemporal_load` / `_store` | yes (Clang builtin; AMD-staff HIP issue #3506) | Compiler sets SLC (and, for stores, GLC). ISA: L2 **Stream** (hit leaves the line, does not reset age) or **Hit No Allocate** if DLC is also set. HIP has **no** `-Xptxas -dlcm=cg` equivalent. | **Streaming** weights that will not be reused this kernel (cold 27B shard, decode of a layer that already missed IC). Do **not** mark a layer you are trying to keep in L2/IC. |
| `__threadfence_block` / `__threadfence` / `__threadfence_system` | yes | LLVM scopes `workgroup` / `agent` / `system`. Software coherence. Does **not** pin data in IC. | Cross-CU visibility of writes (decode metadata, page tables). Not a cache-residency API. |
| HIP atomics (`atomicAdd`, …) / `_system` | yes | Execute at **L2** (HIP Hardware implementation). GLC on atomics means “return pre-op value”, not “bypass”. | Rare in a weight-only GEMV. W4A16 K-split C uses a 64-bit CAS loop (no packed atomic add on gfx10). |
| `__launch_bounds__` / `amdgpu_waves_per_eu` | yes | Occupancy cap. Extra waves on a memory-bound decode kernel thrash L0/L1/IC (GPUOpen Occupancy explained). | Cap occupancy on the weight stream. |
| `__constant__` | yes | Scalar (K$) path; invalidated of volatile data at kernel boundary (LLVM AMDGPUUsage). | Kernel args, scales that are wave-uniform. Not a 100 MB weight array. |
| `llvm.amdgcn.raw.ptr.buffer.load` `cpol` (GLC/SLC/DLC bits) | LLVM intrinsic, not a HIP language feature | gfx10/11: bit0=GLC, bit1=SLC, bit2=DLC. **Not** gfx940 `sc0`/`nt`/`sc1`. | Only if you drop to buffer intrinsics. Same bits HIP already emits for nontemporal / atomics / fences. |
| Persist / MALL NOALLOC / IC bypass | **no** on gfx1030 | Persist/`sc0`/`sc1`/`nt` is **gfx940** (CDNA 3) encoding. GFX11 re-purposes DLC as MALL NOALLOC; GFX10 DLC is **L1 bypass**. LLVM: MALL is coherent and “has no impact on system coherence”; all GPU-memory traffic goes through it. | You cannot pin or skip Infinity Cache from a HIP kernel on V620. |
| `llvm.prefetch` / `global_prefetch` | **ignored** on gfx1030 (LLVM: “Implemented on gfx1250, ignored on earlier targets”) | — | Do not write a prefetch loop expecting IC fill. |
| `s_dcache_wb` | **not in the RDNA 2 ISA** | RDNA 2 has `S_DCACHE_INV` (invalidate scalar K$), `BUFFER_GL0_INV` (“Write back and invalidate the shader L0”), `S_GL1_INV` / `BUFFER_GL1_INV`. | Compiler emits these for fences. You do not call them from HIP. |
| `"amdgpu-memory-bound"` | **not a user knob** | LLVM: “Set internally by backend.” | Ignore. |
| Image T# `LLC No-alloc` (ISA bits 201:200) | image descriptor only | Last-level no-alloc, including `PTE.NoAlloc`. Not a HIP `__global__` load. | Not your decode path unless you bind weights as an image. |
| `S_ATC_PROBE` / `S_ATC_PROBE_BUFFER` | ISA, not HIP | “Probe or prefetch an address into the **SQC data cache**” (scalar K$, 16 KB/WGP). | Useless for a weight tensor. |
| Buffer V# **cache swizzle** (bit 62) | buffer descriptor, not HIP C++ | “Optionally, swizzle texture cache TC L0 cache banks.” | Only if you build a V# yourself. Default HIP global loads do not set this. |
| Working-set size, coalescing, occupancy, no concurrent thrashers | yes, by construction | The only reliable “keep it in IC” method on gfx1030. | Quantize / TP-shard until the **hot** set ≤ 128 MB; stream the rest with nontemporal. |

**HIP default is already the “try to keep it” policy.** There is no `persist` on this ISA. There is a **stream** hint. There is no **prefetch-into-IC**. There is no **bypass-IC**. Design the working set; do not hunt for a magic bit.

---

## 1. Official cache topology and coherence

### 1.1 Topology (already established; restated so the policy bits have a home)

```
VGPR
 → LDS 128 KB/WGP (64 KB/WG)          companion LDS brief; RDNA 2 ISA §2.3.1
 → L0 vector 16 KB/CU, 128 B line     kfd_crat “TCP L1 Cache per CU”; RDNA deck; ROCm gpu-arch-specs
 → L1 / GL1 128 KB/SA, 128 B line     kfd_crat “GL1 Data Cache per SA”, 10 CU on full Navi 21
 → L2 4 MB, 128 B line                kfd_crat Sienna Cichlid 4096 KB; ROCm V620 row
 → Infinity Cache / MALL / L3 128 MB, 64 B line
 → GDDR6 512 GB/s
```

ISA §2.4 (70648, AMD RDNA2 ISA 70648 PDF):

> “On the primary read path, the device consists of multiple channels of L2 cache that provides data to Read-only L1 caches, and finally to L0 caches per WGP.”

HIP Hardware implementation (https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) on the **vector** cache:

> “Write-through design (writes go directly to L2)”
> “Coherent with other CUs through software management”
> “Typical size of 16 KB per CU”
> “The L2 cache serves as the coherence point for all GPU memory accesses”
> “Atomic operation support: Atomics execute directly in the L2 cache for coherence.”

HIP’s RDNA “three-level” sentence places L1 “shared between CUs in a WGP”. The kernel table and the RDNA deck put **GL1 on the shader array**. Use the kernel + ISA for placement (companion architecture brief §4.3). HIP’s “32 channels, 256-byte interleave” sentence is in a **CDNA** paragraph — do not reuse it for Navi 21. L2 channel count on gfx1030 remains **unknown**.

I$ is 32 KB/WGP, K$ (sL1D) 16 KB/WGP, both 64 B lines, **not** hit-on-miss (HIP + kfd_crat). L2 **is** hit-on-miss (HIP).

ROCm `gpu-arch-specs` (docs-6.2.1, opened) V620 row, used as a second official size table (not a product-page substitute for CU/VRAM/IC, but it prints the on-die caches the product page omits):

| Field | V620 |
|---|---|
| LLVM target | gfx1030 |
| VRAM | 32 GiB |
| CU | 72 |
| LDS | 128 KiB |
| Infinity Cache | 128 MiB |
| L2 | 4 MiB |
| Graphics L1 | 128 KiB |
| L0 vector | 16 KiB |
| L0 scalar | 16 KiB |
| L0 instruction | 32 KiB |

Source: https://rocm.docs.amd.com/en/docs-6.2.1/reference/gpu-arch-specs.html

**Associativity:** **unknown** in the ISA, the RDNA deck, HIP, `kfd_crat.c`, the V620 product page, and this ROCm table. Do not invent 16-way.

### 1.2 Write-through L0/L1, L2 as coherence point

RDNA 2 ISA §8.1, GLC field on MUBUF (same wording in §8.1.10):

```
READ
  GLC = 0  Reads can hit on the L0 and persist across wavefronts
  GLC = 1  Reads miss the L0 and force fetch to L2. No L0 persistence across waves.
WRITE
  GLC = 0  Writes miss the L0, write through to L2, and persist in L0 across wavefronts.
  GLC = 1  Writes miss the L0, write through to L2. No persistence across wavefronts.
ATOMIC
  GLC = 0  Previous data value is not returned.
  GLC = 1  Previous data value is returned.
  Note: GLC means "return pre-op value" for atomics.
```

“Persist” in that table is **L0 line retention across waves**, not Infinity Cache persist.

ISA §8.1.10, stores:

> “For both GLC==0 and GLC==1, write data are combined across work-items of the wavefront store clause … dirtied lines are written to the L2 cache automatically and invalidated.”
> “For stores and atomics, the L1 cache is bypassed (but is coherent).”
> “For stores the L0 cache is always Miss-Evict.”

HIP “Memory coherence”:

> “Write-through L1 caches: All writes update both L1 and L2, ensuring L2 always has the latest data.”
> “Software-managed coherence: Coherence between CUs requires explicit synchronization through: Memory fences for ordering; Cache invalidation instructions; Atomic operations (executed at L2 level); Kernel boundaries (implicit synchronization).”
> “Write combining: … write masks indicating which bytes to update.”

So: **L2 is the first GPU-wide copy of truth.** L0/L1 are write-through filters. Two CUs do not snoop each other’s L0. If CU A stores and CU B loads the same address in the same kernel without a fence/atomic/kernel boundary, B may see its own stale L0 line.

ISA §2.4 also mentions “Specific cache-less load instructions can force data to be retrieved from device memory” — that is the GLC=1 / DLC=1 path, still via L2 unless the store-side SLC/DLC table says Bypass (below). It is **not** a documented IC bypass.

### 1.3 Atomics at L2

HIP: atomics execute in L2. ISA: GLC on an atomic is the return-old-value flag, not a cache-scope flag. DLC/SLC still apply to the L2 policy of the atomic (Table 39).

W4A16 decode on gfx1030 has **no** `v_global_atomic_pk_add_f16` (landed gfx940 / gfx1250 per vLLM `q_gemm_rdna3.cu`). K-split C is a 64-bit CAS into pre-zeroed fp16. That CAS is an L2 atomic; it does not pin the output in IC.

### 1.4 Software fences and HIP / LLVM / ISA scopes

HIP C++ language extensions (https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html):

> “`__threadfence_block()` orders memory accesses for all threads within a thread block.”
> “`__threadfence()` orders memory accesses for all threads on a device.”
> “`__threadfence_system()` orders memory accesses for all threads in the system, making writes to memory visible to other devices and the host.”
> “Synchronization functions … implicitly include a threadfence.”

Current HIP device header (`amd_device_functions.h`, opened):

```
__threadfence()         → __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent")
__threadfence_block()   → __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup")
__threadfence_system()  → __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "")   // LLVM default = system
```

LLVM AMDGPUUsage “AMDHSA LLVM Sync Scopes” (inclusive scopes, HSA / HRF-indirect):

| LLVM syncscope | HIP name | Who it includes |
|---|---|---|
| `wavefront` | (no HIP `__threadfence_wave`; wave ops are implicit) | same wave |
| `workgroup` | `__threadfence_block` / `__syncthreads` | same workgroup |
| `agent` | `__threadfence` (HIP “device”) | same GPU agent |
| `system` (empty string) | `__threadfence_system` | host + other devices |
| `cluster` | — | gfx12.5+; on gfx1030 “behaves like `agent`” |
| `singlethread` | — | same lane |
| `*-one-as` | — | same scope, one address space |

HIP does **not** spell “wave” as a fence. Wave-level ordering is the SIMD (lockstep on RDNA) plus any `s_waitcnt` the compiler already emits.

WGP vs CU mode matters for **workgroup** scope. LLVM `SIMemoryLegalizer.cpp` `SIGfx10CacheControl::enableLoadCacheBypass` (opened):

> “In WGP mode the waves of a work-group can be executing on either CU of the WGP. Therefore need to bypass the L0 which is per CU. Otherwise in CU mode all waves of a work-group are on the same CU, and so the L0 does not need to be bypassed.”

Default HIP is **WGP mode**. A workgroup-scoped acquire load on gfx10 WGP therefore gets **GLC** (miss L0). In CU mode (`-mcumode`) it does not. For agent/system, gfx10 sets **GLC+DLC** (miss L0 and L1). Same file: `insertAcquire` for agent/system emits `BUFFER_GL1_INV` then `BUFFER_GL0_INV` (“outer in … to avoid L0 pulling in stale data from L1”). `insertWriteback` on gfx10 **returns false** — there is no L2 writeback instruction in the gfx10 legalizer.

Compiler-inserted invalidates matching the ISA:

| ISA op | What the ISA says |
|---|---|
| `BUFFER_GL0_INV` | “Write back and invalidate the shader L0. Returns ACK to shader.” |
| `BUFFER_GL1_INV` / `S_GL1_INV` | “Invalidate the GL1 cache only.” |
| `S_DCACHE_INV` | “This instruction invalidates the entire scalar cache.” / “Invalidate the scalar data L0 cache.” |

There is **no** `s_dcache_wb` in the RDNA 2 ISA opcode list that was opened. That name is a different generation’s scalar writeback. Do not emit it by hand on gfx1030.

`__syncthreads()` is workgroup arrive+wait (`s_barrier`) plus the implicit workgroup fence. It does not flush L2 or IC.

---

## 2. Infinity Cache / MALL: names, and what HIP/LLVM actually expose

### 2.1 Driver / compiler names

| Name | Who uses it | Opened source |
|---|---|---|
| Infinity Cache | AMD product / press | V620 product page: “AMD Infinity Cache Technology 128 MB”; 2021-11-04 V620 press; RDNA 2 Explained (W6000) “global cache … seen by the entire graphics core” |
| L3 | `amdkfd` `kfd_crat.c` | “L3 Data Cache per GPU (Infinity Cache)”, 128×1024 KB, line **64 B** |
| MALL (memory-attached last level) | LLVM AMDGPUUsage memory model; Linux/LLVM engineers | LLVM D114076 / AMDGPUUsage: “On GFX10.3 a memory attached last level (MALL) cache exists for GPU memory.” |

The RDNA 2 ISA **does not** use the words MALL or Infinity Cache. The ISA’s “third level” shows up only as the SLC/DLC table’s L2 policies and as image-descriptor `LLC No-alloc`.

LLVM AMDGPUUsage (D114076, later “GFX10.3 and GFX11”), quote:

> “On GFX10.3 a memory attached last level (MALL) cache exists for GPU memory. The MALL cache is fully coherent with GPU memory and has no impact on system coherence. All agents (GPU and CPU) access GPU memory through the MALL cache.”

Implications that are in that paragraph, and not more:

- CPU and GPU both see GPU memory **through** MALL. It is not a private GPU-only victim cache you can desync from the host.
- It does **not** change system-scope coherence rules. System fences still exist; MALL is not a substitute for `__threadfence_system`.
- There is no “map this VA range to skip MALL” in the LLVM gfx10 model text that was opened. (LLVM does say ranges of VAs can be set up to **bypass L2** “to ensure system coherence” on some targets — that sentence is about L2 vs other agents, not about MALL.)

**Internal IC bandwidth (TB/s):** **not published** in the ISA, the V620 product page, the 2021-11-04 press, ROCm `gpu-arch-specs`, or LLVM. Marketing “up to 2.4×” / “2.5×” is relative, not an absolute GB/s. Do not quote 1.8–2.0 TB/s.

### 2.2 ISA cache-policy bits on gfx1030 (GLC / DLC / SLC)

RDNA 2 ISA §8.1.10 Tables 38–39 (opened):

**Vector loads**

| SLC | DLC | L2 | L1 |
|---|---|---|---|
| 0 | 0 | LRU | Hit LRU |
| 0 | 1 | LRU | Miss Evict |
| 1 | 0 | Stream | Hit LRU |
| 1 | 1 | Hit No Allocate | Miss Evict |

**Vector stores and atomics** (L1 already bypassed)

| SLC | DLC | L2 |
|---|---|---|
| 0 | 0 | LRU |
| 0 | 1 | **Bypass** |
| 1 | 0 | Stream — “Hit leaves line in cache but do not reset age.” |
| 1 | 1 | Hit No Allocate |

DLC definition (ISA): “Device Level Coherent. When set, accesses are forced to miss in level 1.”
SLC definition (ISA): “System Level Coherent. Used in conjunction with DLC to determine L2 cache policies.”

ISA §8.1.10 also: “The Device Level Coherent bit (DLC) and System Level Coherent (SLC) bits control the behavior of the **second and third level caches**.” That is the closest the ISA comes to talking about a level past L2. It does **not** name MALL, and it does **not** give a fourth bit for “persist in IC”.

That is the **entire** programmable policy surface for vector memory on gfx1030.

### 2.3 What LLVM does with those bits (gfx10 vs gfx11 vs gfx940)

LLVM intrinsic `cpol` / cachepolicy immediate (IntrinsicsAMDGPU.td, opened via llvm-commits #87364 / #78768):

```
gfx10/gfx11:  bit 0 = glc, bit 1 = slc, bit 2 = dlc
gfx90a:       bit 4 = scc
gfx940:       bit 0 = sc0, bit 1 = nt, bit 4 = sc1
gfx12+:       bits [0-2] = th, bits [3-4] = scope
```

**`sc0` / `sc1` / `nt` are not gfx1030 encodings.** They are CDNA 3 (gfx940) / later. Do not copy an MI300 persist sample onto V620.

LLVM commit b0a3849 (DLC-for-GFX11), quote:

> “In GFX10 dlc controlled L1 cache bypass. In GFX11 it has been repurposed to control MALL NOALLOC, and glc controls L1 as well as L0 cache bypass.”

And `SIMemoryLegalizer.cpp` `SIGfx10CacheControl` (opened):

> “Note: there is no L2 cache coherent bypass control at the ISA level.”
> GFX11 only: “Set MALL NOALLOC for both load and store instructions” via DLC.
> Nontemporal on GFX10 (shipping legalizer this pass): SLC for loads; GLC+SLC for stores. DLC is **omitted** on GFX10 for the nontemporal sequence (“If GFX10, omit dlc=1” in the AMDGPUUsage table / D127405).

So on **V620 / gfx1030**:

| Intent | Bits LLVM actually sets | Hits IC? |
|---|---|---|
| Normal load | none | yes, if it missed L2 (MALL is on the path) |
| Nontemporal load | SLC (=1), no DLC | L2 Stream. **Still goes through MALL.** Stream makes L2 a poor holder; IC behavior under Stream is **not stated** in the ISA or LLVM gfx10 model. |
| Volatile / agent-scope acquire load | GLC+DLC | miss L0 and L1, L2 LRU. Still through MALL. |
| Nontemporal store | GLC+SLC | L2 Stream. Still through MALL. |
| Store SLC=0 DLC=1 | L2 Bypass (ISA Table 39) | LLVM does **not** expose this as a HIP builtin. Even then, “L2 Bypass” is not documented as “MALL Bypass”. |
| gfx11 MALL NOALLOC | DLC on gfx11 | **not present** on gfx10 |

### 2.4 Persist

**Not exposed. Not in the RDNA 2 ISA.** The word “persist” in the ISA is L0-across-waves (GLC=0). The CDNA 3 MALL persist / `sc*` policy is a different encoding on a different target.

HIP issue #3506 (AMD staff, opened): there is **no** compiler flag or environment variable to disable/bypass L1. The supported hammer is `__builtin_nontemporal_{load,store}` on every access.

> “Unfortunately, HIP doesn't have compiler flags or environment variables that allow for disabling/bypassing L1 cache directly. … manually replacing every load and store … with `__builtin_nontemporal_load(ptr)` / `__builtin_nontemporal_store(value, ptr)`.”

### 2.5 Bypass

What you can bypass from HIP/LLVM on gfx1030:

- **L0** — GLC=1 (fences, atomics-with-return, volatile, WGP workgroup-scope loads).
- **L1** — DLC=1 (same sequences; ISA “forced to miss in level 1”).
- **L2** — only the store/atomic cell “SLC=0 DLC=1 → Bypass” in Table 39. LLVM’s legalizer comment says there is **no coherent L2 bypass** for the load sequences it emits. HIP does not give you that store cell as a named builtin.
- **MALL / IC** — **no** HIP, LLVM, or ISA bit on gfx1030.

### 2.6 Prefetch

| Mechanism | gfx1030? | Target |
|---|---|---|
| `llvm.prefetch` | ignored (LLVM AMDGPUUsage / PR #157949: “Implemented on gfx1250, ignored on earlier targets”) | gfx1250 `global_prefetch_b8` → GL2 |
| `S_ATC_PROBE` / `S_ATC_PROBE_BUFFER` | ISA yes | **SQC** (scalar K$), 16 KB/WGP |
| `S_INST_PREFETCH` | ISA yes | I$, 1–3 × 64 B lines |
| Software prefetch = ordinary load into a VGPR you do not use yet | yes | whatever the load’s GLC/SLC/DLC say; still no IC-specific destination |

A decode kernel that “prefetches the next layer into IC” is **an ordinary LRU load of a working set that fits**, issued early, not a prefetch opcode.

I$ vs occupancy (32 KB/WGP, out of PIX min): [icache-occupancy.md](icache-occupancy.md).

### 2.7 Non-temporal

HIP/Clang (GPUOpen lab notes, Laplacian part 3, opened):

```
T    __builtin_nontemporal_load(T *addr);
void __builtin_nontemporal_store(T value, T *addr);
```

> “The AMD clang compiler provides two overloaded builtins allowing generation of non-temporal loads and stores.”
> “these intrinsics are specific to AMD GPUs.”

GPUOpen used the **store** to keep a write-once output out of L2 so the reused stencil could occupy L2. That is the right mental model for decode: **nontemporal the stream you will not reread; leave the reused set on the default path.**

LLVM models nontemporal as metadata on the load/store; the legalizer turns it into SLC (and GLC on stores). It is a **hint**.

HIP has no `__nontemporal__` qualifier on pointers. You wrap the access.

### 2.8 `llvm.amdgcn.buffer.load` / raw.ptr.buffer.load

Old `llvm.amdgcn.buffer.*` is removed (llvm-project #113250). Current:

```
llvm.amdgcn.raw.ptr.buffer.load  (ptr addrspace(8), voffset, soffset, imm offset, imm cpol)
```

`cpol` on gfx1030 is the GLC/SLC/DLC immediate above. Same hardware as a HIP global load. Using the intrinsic does **not** unlock persist or IC bypass.

### 2.9 `"amdgpu-memory-bound"`

LLVM AMDGPUUsage attributes table: **“Set internally by backend.”** Not a persist/bypass hint you pass from HIP.

---

## 3. What a weight-only decode kernel can actually do

Goal: keep **one layer** (or the live KV window) in the 128 MB IC, and **stream** everything that will not be reused.

Decode geometry (from `kernels/w4a16.md`, not re-argued here): A staged in LDS (`M_COUNT × (256+8) × 2 B`), packed B streamed from global, dequant in VGPR, inner product `__builtin_amdgcn_fdot2`. There is no WMMA and no bf16 DOT on gfx1030. That already decides most of the cache traffic: **A never hits IC after the first LDS fill; B is the IC question.**

### 3.1 The only persist mechanism: fit + reuse + don’t thrash

MALL is on every GPU-memory path (LLVM). If the bytes you reread still reside in the 128 MB, the second read does not go to GDDR6. If they don’t, you pay 512 GB/s.

You cannot lock a line. You can:

1. **Make the hot set ≤ 128 MB** (quantize, TP-shard, layer-at-a-time, don’t pin the whole model).
2. **Reread it** before something else evicts it (same kernel, or the next kernel before a copy/another stream blows 128 MB).
3. **Not mark it nontemporal.**
4. **Not oversubscribe occupancy** so 36 WGPs × 4 SIMD × 16 waves all pull different cold lines through the same 128 MB (GPUOpen Occupancy explained: memory-bound kernels can lose from extra waves via cache thrash).
5. **Not run a concurrent kernel or SDMA** that walks tens of MB of some other buffer. IC is GPU-wide (AMD “seen by the entire graphics core”).

Two concurrent HIP streams on one V620 **fight for the same 128 MB**. A “layer lives in IC” claim is only true if that GPU is otherwise quiet.

### 3.2 Stream vs keep — concrete policy for W4A16 / W8A8 / mxfp4

| Buffer | Policy | Why |
|---|---|---|
| Packed weights of a layer that **fits** (see §4) | default loads, coalesced | You want L2 LRU + MALL fill. Nontemporal would Stream L2 and give you nothing in return. |
| Packed weights of a layer that **does not fit** | `__builtin_nontemporal_load` on the weight stream | One-shot. Keep L2/IC for x[], scales, residual, KV. |
| Per-group scales / zp (a few MB) | default loads; prefer `__constant__` or SMEM if wave-uniform | Tiny, reused every tile. Do **not** put a nibble→fp16 LUT in `__constant__` (llama.cpp #24438: per-lane K$ index serializes the inner loop). The W4A16 bit-trick exists so you never do that. |
| Activation `x` (d × 2 B, ~7 KB on Qwen2.5-7B) | default; usually already in LDS/VGPR | Fits in L0. W4A16 decode stages A in LDS even at M=1. |
| KV window you will touch again this step | default | Fits or it doesn’t — see §4.5. |
| Decode output / residual write | `__builtin_nontemporal_store` if you will not reread it in this kernel | GPUOpen Laplacian pattern. |
| Scratch / spill | don’t | A spill is an L0/L2/IC hit in the inner loop (companion brief). |

Do **not** nontemporal the weights of a 7B W4 layer that already fits. That is how you turn a ~115 MB IC-resident GEMV into a 512 GB/s stream.

W8A8 is the same policy with 1 B/param instead of ~0.52. mxfp4 is the same policy with 4.25 b/param (OCP MX block 32×E2M1 + 1 B E8M0). There is **no** MX decode unit on gfx1030; unpack is VALU. Cache policy does not change because the unpack is in-register.

### 3.3 64 B vs 128 B lines

| Level | Line | Source |
|---|---|---|
| L0 / L1 / L2 | **128 B** | kfd_crat; RDNA deck; HIP “128-byte cache lines … Wave32 × 4 B = 128 B” |
| Infinity Cache | **64 B** | kfd_crat L3 `.cache_line_size = 64` |

Programming consequences that are in those numbers:

- Happy HIP load: wave32, consecutive `threadIdx.x`, 4 B/lane → **one L0/L1/L2 line**. That is the native path. Wave64 × 4 B = two L2 lines.
- An IC miss/fill is a **64 B** transaction. A 128 B L2 line that misses IC is **two** IC lines. Align persistent buffers to **128 B** (satisfies both). 64 B alignment is enough for IC but splits an L2 line if you start at +64.
- Packed W4: 32 × 4-bit = 16 B per 32 weights. A 128 B line holds 256 W4 weights (plus you still want the scale nearby). An IC 64 B line holds 128 W4 weights. **Group-128 W4A16** is one IC line of payload + a scale that should sit in the same 64/128 B neighborhood or in K$.
- W8A8: 128 B = 128 int8 weights. Same coalescing rule (wave32 × 4 B).
- mxfp4 block is 17 B (32×4-bit + 1 B E8M0; OCP MX v1.0). That is **not** a line size. Pad/pack so 4 blocks (68 B) or 8 blocks (136 B) meet 64/128 B — exact on-disk layout is a kernel choice; the hardware only sees the addresses you emit.

### 3.4 Buffer V# / RVA and cache swizzle

RDNA 2 ISA Table 37, buffer resource descriptor (the 128-bit V#; LLVM `ptr addrspace(8)`):

| Bits | Name | ISA text |
|---|---|---|
| 47:0 | Base address | “Byte address.” |
| 61:48 | Stride | Bytes 0–16383 |
| **62** | **Cache swizzle** | “Buffer access. Optionally, swizzle texture cache TC L0 cache banks.” |
| 63 | Swizzle enable | “Swizzle AOS according to stride, index_stride, and element_size, else linear” |

HIP `float*` global loads do **not** go through a V# you control; they are `global_load` / flat. Cache swizzle is only live if you build a buffer resource (`llvm.amdgcn.make.buffer.rsrc` / `raw.ptr.buffer.load`).

What the ISA actually says about bit 62: it swizzles **L0 (TC) banks**, not IC, not L2. It is a bank-conflict knob for structured/AOS buffer fetches. It is **not** a persist/bypass/IC-map knob. The AOS swizzle (bit 63) is an addressing swizzle, also not a cache-residency control.

If you stay in HIP C++ `half*` / `uint4*` global pointers, you can ignore V# swizzle. Coalesce. That is the whole L0 story.

Image T# bits 201:200 `LLC No-alloc` (ISA) talk to last-level allocation, including `PTE.NoAlloc`. That is an **image** descriptor. A HIP decode kernel loading a linear weight buffer does not set it. Whether the amdgpu driver ever sets `PTE.NoAlloc` on a `hipMalloc` mapping is **unknown** in the sources opened here.

### 3.5 Occupancy vs IC

V620 is **72 CU = 36 WGP** (AMD V620: 72 CU; ISA CU = half a WGP). Full slots = 36 × 4 SIMD × 16 waves = **2304 wave32**. A 256-thread workgroup is 8 waves → 288 workgroups to fill the chip.

For a weight-streaming GEMV, that many waves all missing IC will just multiply outstanding GDDR6. Prefer `__launch_bounds__(256, 4)` / `amdgpu_waves_per_eu(4, 8)` and **measure** (companion LDS brief). The architecture brief already warned this; it is the cache-policy version of the same fact.

W4A16 decode seed (w4a16 brief): `THREADS=256`, `BLOCK_KN=256`, 4 N-cols/thread. On 72 CU, `BLOCK_KN=512` is the gfx1100 lesson inverted — fewer `gridDim.z` leaves CUs idle. Occupancy 0 is a real gfx1030 failure mode (llama.cpp FA: 256 thr / 203 VGPR). Stay ≤ 64 VGPR on the decode TU if you want 16 waves/SIMD.

### 3.6 What you should not do

- Do not emit `s_dcache_wb`.
- Do not copy gfx940 `sc0`/`sc1` persist samples.
- Do not expect `llvm.prefetch` to fill IC.
- Do not assume HIP `__constant__` can hold a layer (logical constant space is small; it is the K$ path).
- Do not assume L1 (128 KB / SA, 10 CU on full Navi 21) holds a layer. It holds a **K-panel**. V620 is a 72-CU harvest; per-SA CU count is **not** independently published — treat L1 as “small, shared, not your layer cache”.
- Do not assume two kernels in flight “share” IC cooperatively. They compete.
- Do not put a dequant LUT in K$ / `__constant__` and index it per-lane.

---

## 4. Worked math — V620, 7B / 27B, TP=4

### 4.1 V620 silicon (official, not inferred from 6900 XT)

Opened AMD sources, this pass:

| Item | Value | Source |
|---|---|---|
| Architecture | RDNA 2, Navi 21 / gfx1030 | LLVM Processors table lists “Radeon PRO V620” under `gfx1030`; AMD 2021-11-04 press “AMD RDNA 2 architecture”; ROCm gpu-arch-specs |
| Compute units | **72** | AMD product page https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html “Compute Units 72”; AMD 2021-11-04 press table |
| Stream processors | **4608** | same press (72 × 64) |
| Ray accelerators | **72** | product page |
| Peak engine clock | 2.2 GHz | partner V620 datasheet (ActivePort V4.5, opened via search); product page in prior fetch |
| Peak FP32 (arithmetic) | 72 × 64 × 2 × 2.2 GHz = **20.28 TFLOPs** | product of official CU × 64 SP/CU × FMA × clock. Vector, not a matrix peak — there is no WMMA. |
| Infinity Cache | **128 MB** | product page “AMD Infinity Cache Technology 128 MB”; 2021-11-04 press body; ROCm gpu-arch-specs 128 MiB |
| Memory | **32 GB GDDR6**, 16 Gbps, **256-bit**, **512 GB/s**, ECC | product page (32 GB, 512 GB/s, 128 MB IC); 2021-11-04 press table (32 GB @ 16 Gbps, 512 GB/s, 256-bit) |
| L2 | **4 MB** | ROCm gpu-arch-specs V620 row; `kfd_crat.c` Sienna Cichlid 4096 KB. Product page does **not** print L2. |
| L0 vector / L1 / I$ / K$ | 16 / 128 / 32 / 16 KB | ROCm gpu-arch-specs; kfd_crat. Product page does **not** print these. |
| Bus | PCIe 4.0 x16 | product page / datasheet |

L2 **4 MB** and IC line **64 B** are the Sienna Cichlid (Navi 21) `kfd_crat.c` numbers from the architecture brief. V620 is a 72-CU / 32 GB harvest of that die; no opened AMD page prints a different L2 or IC. Use 4 MB / 128 MB / 512 GB/s.

CU count 72 ⇒ **36 WGP**. Full Navi 21 is 80 CU / 8 SA × 10 CU (kfd_crat `num_cu_shared = 10`). V620’s exact SA population is **unknown** (harvest). Do not invent “7.2 SA”.

IC absolute TB/s: **unknown**. Do not quote 1.8–2.0 TB/s.
Associativity: **unknown**. Do not quote 16-way.

### 4.2 One transformer layer — definitions

“7B” = **Qwen2.5-7B** (the model this box’s W4A16 brief already uses). Official HF `Qwen/Qwen2.5-7B` `config.json` (opened via HF blob): `hidden_size=3584`, `intermediate_size=18944`, `num_hidden_layers=28`, `num_attention_heads=28`, `num_key_value_heads=4`, `hidden_act=silu` (SwiGLU), `tie_word_embeddings=false`. README: 7.61B total / **6.53B non-embedding**, “Attention QKV bias”, GQA 28/4, head dim 128. Same FFN width on `Qwen2.5-7B-Instruct`.

“27B” = **Gemma 2 27B**. Config: DeepMind `Gemma2_27B` / HF `gemma2_27b_config` (`hidden_size=4608`, `intermediate_size=36864`, 32 heads, 16 KV heads, `head_dim=128`, 46 layers, GeGLU, `use_post_attn_norm=True`, `use_post_ffw_norm=True` → **4** RMSNorms/layer). Paper Table 1 “Feedforward dim 73728” is **2×36864** (gate+up concatenated); the three-matrix count uses 36864. 46 × 566,249,472 = 26,047,475,712, which matches the paper’s 26,047,480,320 non-embedding to within one 4608-vector (the final RMSNorm lives outside the layer stack).

There is **no** official Qwen2.5-27B. Qwen2.5-32B is a different shape (`hidden=5120`, 64 layers) and is not used below.

Bytes below are **one layer’s weights only** (no embed, no lm_head, no KV).

| | Qwen2.5-7B | Gemma 2 27B |
|---|---|---|
| Q | 3584² = 12,845,056 | 4608×4096 = 18,874,368 |
| K | 3584×512 = 1,835,008 | 4608×2048 = 9,437,184 |
| V | 1,835,008 | 9,437,184 |
| O | 12,845,056 | 18,874,368 |
| QKV bias | 3584+512+512 = 4,608 | 0 |
| Attention params | **29,364,736** | **56,623,104** |
| FFN (gate+up+down) | 3 × 3584 × 18944 = **203,685,888** | 3 × 4608 × 36864 = **509,607,936** |
| RMSNorm | 2 × 3584 = 7,168 | 4 × 4608 = 18,432 |
| **Params / layer** | **233,057,792** | **566,249,472** |

Sanity: 28 × 233,057,792 = 6,525,618,176 ≈ official 6.53B non-embedding (plus the final RMSNorm).

Generic decoder-only layer (so you can plug another 27B):

```
Q = hidden * (n_q * d_h)
K = hidden * (n_kv * d_h)
V = hidden * (n_kv * d_h)
O = (n_q * d_h) * hidden
FFN = 3 * hidden * intermediate     # SwiGLU / GeGLU
```

Storage assumptions (stated, not invented as “the” format):

| Format | Bytes / param | What is included |
|---|---|---|
| FP16 | 2 | raw |
| W8A8 | 1 | int8 weights; per-channel FP16 scales are ~80–200 KB and ignored in the table (they do not change IC fit) |
| W4A16 g128 | 0.5 + 2/128 | int4 payload + **one FP16 scale per group of 128**; no zero-point. AWQ-with-zeros adds another 2/128. |
| W4A16 g32 | 0.5 + 2/32 | same, group 32 (vLLM RDNA3 / GPTQ common). `gs ∈ {32,64,128}` in the W4A16 brief. |
| mxfp4 | 4.25/8 = 17/32 | OCP MX v1.0 concrete MXFP4: block **k=32**, element FP4 E2M1 (4 b), scale E8M0 (8 b) → 136 b / 32 = 4.25 b. Spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf |

QKV bias on Qwen2.5-7B is 9 KB in FP16. It is in the param count; it does not move any “fits / misses” line.

### 4.3 Bytes / layer and what fits in 128 MiB IC

IC capacity used here: **128 × 1024² = 134,217,728 B** (kfd_crat 128×1024 KB). Product marketing says “128 MB”; if someone uses 128e6 the conclusions do not change.

| Model | Format | Bytes / layer | vs 128 MiB IC | vs 4 MB L2 |
|---|---|---|---|---|
| Qwen2.5-7B | FP16 | 466,115,584 (444.52 MiB) | **3.47× — miss** | miss |
| Qwen2.5-7B | W8A8 | 233,057,792 (222.26 MiB) | **1.74× — miss** | miss |
| Qwen2.5-7B | W4A16 g128 | 120,170,424 (114.60 MiB) | **0.90× — fits** | miss |
| Qwen2.5-7B | W4A16 g128+zeros | 123,811,952 (118.07 MiB) | **0.92× — fits** | miss |
| Qwen2.5-7B | W4A16 g32 | 131,095,008 (125.02 MiB) | **0.98× — fits, leftover ~3 MiB** | miss |
| Qwen2.5-7B | W4A16 g32+zeros | 145,661,120 (138.91 MiB) | **1.09× — miss** | miss |
| Qwen2.5-7B | mxfp4 | 123,811,952 (118.07 MiB) | **0.92× — fits** | miss |
| Gemma 2 27B | FP16 | 1,132,498,944 (1080.04 MiB) | **8.44× — miss** | miss |
| Gemma 2 27B | W8A8 | 566,249,472 (540.02 MiB) | **4.22× — miss** | miss |
| Gemma 2 27B | W4A16 g128 | 291,972,384 (278.45 MiB) | **2.18× — miss** | miss |
| Gemma 2 27B | W4A16 g32 | 318,515,328 (303.77 MiB) | **2.37× — miss** | miss |
| Gemma 2 27B | mxfp4 | 300,820,032 (286.88 MiB) | **2.24× — miss** | miss |

**Single-GPU, no TP:** only **7B W4A16 (g128, or g32 without zeros) and 7B mxfp4** put a whole layer in IC. 7B W4A16 g32 is **tight** — leftover ~3 MiB is not a KV window; one extra buffer or a second stream will evict. 7B W8A8 and everything 27B stream (or you keep a **slice** — one matrix, not the layer).

L2 (4 MB) holds none of these layers. L2 is for the active GEMV panel / `x` / scales, not the layer.

### 4.4 TP=4 (this box) — per-GPU shard

Tensor-parallel split of the seven matrices (HF `base_model_tp_plan`: Q/K/V/gate/up colwise, O/down rowwise). Norms and Qwen QKV bias replicate (~KB). Ignore them. Shard ≈ **1/4** of the layer bytes.

| Model | Format | Bytes / GPU | vs 128 MiB IC | Leftover IC |
|---|---|---|---|---|
| Qwen2.5-7B | FP16 | 116,528,896 (111.13 MiB) | **fits (0.87×)** | 16.9 MiB |
| Qwen2.5-7B | W8A8 | 58,264,448 (55.56 MiB) | **fits (0.43×)** | 72.4 MiB |
| Qwen2.5-7B | W4A16 g128 | 30,042,606 (28.65 MiB) | **fits (0.22×)** | 99.3 MiB |
| Qwen2.5-7B | W4A16 g32 | 32,773,752 (31.26 MiB) | **fits (0.24×)** | 96.7 MiB |
| Qwen2.5-7B | mxfp4 | 30,952,988 (29.52 MiB) | **fits (0.23×)** | 98.5 MiB |
| Gemma 2 27B | FP16 | 283,124,736 (270.01 MiB) | **2.11× — miss** | — |
| Gemma 2 27B | W8A8 | 141,562,368 (135.00 MiB) | **1.05× — miss** (6.9 MiB over) | — |
| Gemma 2 27B | W4A16 g128 | 72,993,096 (69.61 MiB) | **fits (0.54×)** | 58.4 MiB |
| Gemma 2 27B | W4A16 g32 | 79,628,832 (75.94 MiB) | **fits (0.59×)** | 52.1 MiB |
| Gemma 2 27B | mxfp4 | 75,205,008 (71.72 MiB) | **fits (0.56×)** | 56.3 MiB |

Read that table as **capacity**, not as a promise the driver will keep the shard resident. Capacity + default loads + no competing traffic is the whole persist story (§3.1).

Practical decode policy on this box:

| Workload | Weight path | IC leftover (order of) | What to put in the leftover |
|---|---|---|---|
| 7B W4 / mxfp4, TP=4 | default loads, **keep** | ~97–99 MiB | KV window (below) |
| 7B W8A8, TP=4 | default, **keep** | ~72 MiB | KV |
| 7B FP16, TP=4 | default, **keep** | ~17 MiB | short KV / scales only |
| 7B W4 g32, **no TP** | default, **keep if the GPU is quiet** | ~3 MiB | nothing else |
| 27B W4 / mxfp4, TP=4 | default, **keep** | ~52–58 MiB | KV |
| 27B W8A8, TP=4 | **stream** (nontemporal weights) or drop to W4 | shard is 7 MiB over IC; scales + `x` + code will evict | do not pretend it fits |
| 27B FP16, TP=4 | **stream** | 2.1× IC | nontemporal weights; IC for `x` + KV slice |

### 4.5 KV window (FP16 K/V, weight-only activations)

Per-token KV bytes, **full model**, FP16, no quantization:

```
Qwen2.5-7B:  28 layers × 2 × 4 KV heads × 128 × 2 B = 57,344 B/token   (56 KiB)
Gemma 2 27B: 46 layers × 2 × 16 KV heads × 128 × 2 B = 376,832 B/token (368 KiB)
```

Qwen’s GQA (4 KV heads) is why 7B KV/token is **much** smaller than Llama-2-7B MHA (512 KiB/token). Do not reuse a Llama-2 KV table on this model.

| | Tokens of KV in 128 MiB IC (weights not resident) | After a TP=4 KV-head split |
|---|---|---|
| Qwen2.5-7B | **2340** (÷ 57,344 B) | **9362** (1 KV head / GPU = 14,336 B/token) |
| Gemma 2 27B | **356** | **1424** (4 KV heads / GPU) |

If the **layer shard is also resident**, subtract its bytes first. KV here is the **TP=4** column (this box):

| Resident shard | Leftover | Tokens of TP=4 KV |
|---|---|---|
| 7B W4A16 g128 | 99.3 MiB | ~7266 |
| 7B W4A16 g32 | 96.7 MiB | ~7079 |
| 7B mxfp4 | 98.5 MiB | ~7206 |
| 7B W8A8 | 72.4 MiB | ~5300 |
| 7B FP16 | 16.9 MiB | ~1235 |
| 27B W4A16 g128 | 58.4 MiB | ~649 |
| 27B W4A16 g32 | 52.1 MiB | ~579 |
| 27B mxfp4 | 56.3 MiB | ~626 |

A multi-thousand-token KV cache on 27B does **not** fit next to the shard. Paged attention that walks the whole cache every token will stream GDDR6; IC only helps the **working** pages (the pages this step actually touches). That is the same “fit the hot set” rule.

On 7B GQA + TP=4, leftover IC after a W4 shard is large enough for **thousands** of tokens of KV. That is the interesting decode case on this box: keep the W4 shard **and** a long-ish KV window, both on the default (LRU) path.

### 4.6 Bandwidth floor when IC misses

GDDR6 **512 GB/s** (AMD V620 product page + press). A 27B FP16 TP=4 shard is 283 MB → ≥ 0.55 ms/layer at peak if it misses everything, before decode math. A 7B W4 TP=4 shard is 30 MB → ≥ 59 µs/layer at peak. Those are **roofline floors**, not measurements. IC TB/s is unknown, so the hit-path roofline is unknown.

---

## 5. Unknowns (do not invent)

| Question | Status | Where it might live |
|---|---|---|
| Infinity Cache absolute TB/s | unknown in ISA, V620 page, press, LLVM, ROCm gpu-arch-specs | Hot Chips / ISSCC; measure |
| L0/L1/L2/IC associativity | unknown (architecture brief) | microbench; not 16-way from an AMD PDF |
| L2 channel count / interleave on Navi 21 | unknown (HIP 32×256 B is CDNA text) | amdgpu TCC; Hot Chips |
| IC behavior under L2 Stream / Hit-No-Allocate | ISA table stops at L2; §8.1.10 says DLC/SLC “control the second and third level caches” but does not define IC Stream | does Stream also age IC lines? **unknown** |
| Whether `hipMalloc` pages ever get `PTE.NoAlloc` | unknown | amdgpu VM / MALL PTE bits |
| V620 SA count / CUs per SA (72-CU harvest) | unknown | 10 CU/SA is full Sienna Cichlid |
| `s_dcache_wb` | **not in RDNA 2 ISA** | do not use |
| Persist / MALL NOALLOC from HIP on gfx1030 | **does not exist** | gfx11 DLC; gfx940 sc* |
| `llvm.prefetch` on gfx1030 | ignored | gfx1250 |
| Cycle cost of `BUFFER_GL0_INV` / `S_GL1_INV` | not in ISA | — |
| Whether nontemporal load should also set GLC on gfx10 | LLVM PR #89739 disputed; shipping legalizer this pass uses SLC-only for gfx10 loads, GLC+SLC for stores | check `llvm-objdump` of **your** ROCm |
| mxfp4 decode throughput on gfx1030 | no MX unit; you unpack with VALU / `V_DOT8` | measure |

---

## 6. Sources actually opened

1. Companion `silicon/architecture.md`, `kernels/w4a16.md`, `silicon/lds-tiles.md`
2. AMD “RDNA 2” ISA 70648 AMD RDNA2 ISA 70648 PDF — §2.4, §7.2.2 `S_DCACHE_INV`, §8.1 GLC/DLC/SLC, §8.1.10 Tables 38–39, Table 37 V# bits 62–63, image `LLC No-alloc` bits 201:200, `BUFFER_GL0_INV` / `BUFFER_GL1_INV`, `S_GL1_INV`, `S_ATC_PROBE`, `S_INST_PREFETCH`
3. HIP Hardware implementation — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (write-through L0/L1, L2 coherence point, atomics at L2, software fences)
4. HIP C++ language extensions — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (`__threadfence*`)
5. HIP `amd_device_functions.h` — https://raw.githubusercontent.com/ROCm/clr/develop/hipamd/include/hip/amd_detail/amd_device_functions.h (`__builtin_amdgcn_fence` scopes)
6. LLVM AMDGPUUsage — https://llvm.org/docs/AMDGPUUsage.html (scopes, gfx1030 = V620, `llvm.prefetch` gfx1250-only, `amdgpu-memory-bound` internal, buffer addrspace 8)
7. LLVM D114076 / commit 6d28dff — MALL paragraph for GFX10.3
8. LLVM commit b0a3849 / D127405 — “In GFX10 dlc controlled L1 cache bypass. In GFX11 it has been repurposed to control MALL NOALLOC”
9. LLVM IntrinsicsAMDGPU.td cachepolicy comments (PRs #78768, #87364) — gfx10 glc/slc/dlc vs gfx940 sc0/nt/sc1
10. `SIMemoryLegalizer.cpp` (main, opened) — `SIGfx10CacheControl`: WGP GLC, no L2 coherent bypass, gfx10 nontemporal = SLC (loads) / GLC+SLC (stores), gfx11 DLC = MALL NOALLOC, gfx10 `insertWriteback` is a no-op
11. HIP issue #3506 — https://github.com/ROCm/hip/issues/3506 (no L1-bypass flag; use `__builtin_nontemporal_*`)
12. GPUOpen Laplacian part 3 — https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-finite-difference-docs-laplacian_part3/ (nontemporal builtins)
13. AMD Radeon PRO V620 product page — https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html (72 CU, 128 MB IC, 32 GB, 512 GB/s)
14. AMD 2021-11-04 V620 press — https://www.amd.com/en/newsroom/press-releases/2021-11-4-amd-radeon-pro-v620-gpu-delivers-powerful-multi-.html and IR copy https://ir.amd.com/news-events/press-releases/detail/1030/amd-radeon-pro-v620-gpu-delivers-powerful-multi-purpose-data-center-visual-performance-for-todays-demanding-cloud-workloads (4608 SP, 72 CU, 32 GB @ 16 Gbps, 512 GB/s, 256-bit)
15. ROCm gpu-arch-specs 6.2.1 — https://rocm.docs.amd.com/en/docs-6.2.1/reference/gpu-arch-specs.html (V620: gfx1030, 72 CU, 32 GiB, IC 128, L2 4, L1 128, L0 16)
16. `kfd_crat.c` Sienna Cichlid — architecture brief (L2 4 MB, L3 128 MB / 64 B)
17. Qwen2.5-7B dims — https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json (`hidden=3584`, `intermediate=18944`, 28 layers, GQA 28/4); README 6.53B non-embedding
18. Gemma 2 27B dims — DeepMind `gemma/gm/nn/_gemma.py` `Gemma2_27B`; paper Table 1 https://arxiv.org/html/2408.00118 ; Keras export `gemma2_27b_config` (intermediate_size=36864)
19. OCP MX v1.0 — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
20. GPUOpen Occupancy explained — https://gpuopen.com/learn/occupancy-explained/ (cache thrash from extra waves)
21. LLVM Processors table — `gfx1030` example products include Radeon PRO V620
22. LLVM PR #157949 — `llvm.prefetch` “Implemented on gfx1250, ignored on earlier targets”
