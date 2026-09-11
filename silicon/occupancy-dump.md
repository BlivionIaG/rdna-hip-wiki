# gfx1030 occupancy dump (NT_AMDGPU_METADATA)

Reusable recipe to pull compiled VGPR / SGPR / LDS / scratch from a HIP `.so` / `.hip_fatbin` / `.co`. Occupancy-first. Does **not** change extras HIP or UNC cards. FA numbers live on [fa-occupancy.md](fa-occupancy.md) §8; the math lives on [hip-craft.md](hip-craft.md) §6; silicon limits on [architecture.md](architecture.md).

This page is the dump path. Use it on **any** gfx1030 kernel (EXL3 GEMM, GDN, W4, RMSNorm, skinny), not only FA.

Fold page: [occupancy-composite.md](occupancy-composite.md).

## 1. Where the numbers live

HIP embeds one Clang offload bundle per host binary in ELF section **`.hip_fatbin`**. Unbundle the gfx1030 entry to a code object (HSACO / `.co`). Code object V3+ stores AMDHSA kernel props in an ELF note:

| Field | Value |
|---|---|
| Owner / name | `AMDGPU` |
| Type | **`NT_AMDGPU_METADATA` = 32** (types 0–31 reserved) |
| Payload | MessagePack map |

Root keys: `amdhsa.version`, `amdhsa.target`, `amdhsa.kernels[]`. Each kernel map is what occupancy reads. LLVM emits the occupancy fields in `AMDGPUHSAMetadataStreamer.cpp` (`getHSAKernelProps`).

Occupancy keys (integers unless noted):

| Key | Required? | Meaning for occupancy |
|---|---|---|
| `.name` / `.symbol` | yes | kernel name; symbol is `<name>.kd` |
| `.vgpr_count` | yes | VGPRs **per work-item**. Occupancy uses this, rounded to the gfx1030 wave32 alloc granule **16**. Addressable file is still 256; physical file is 1024/SIMD32. |
| `.sgpr_count` | yes | SGPRs per wave, including VCC / FLAT_SCRATCH specials. **Not an occupancy limiter on GFX10+** ([hip-craft.md](hip-craft.md) §6.1). Still dump it. |
| `.vgpr_spill_count` | optional | count of VGPR→spill stores from the allocator |
| `.sgpr_spill_count` | optional | count of SGPR→spill stores |
| `.group_segment_fixed_size` | yes | **static** LDS bytes (`ProgramInfo.LDSSize`) |
| `.private_segment_fixed_size` | yes | **scratch** bytes (`ProgramInfo.ScratchSize`, LLVM private AS 5) |
| `.wavefront_size` | yes | 32 on HIP gfx1030 default; 64 only if compiled `+wavefrontsize64` |
| `.max_flat_workgroup_size` | yes | compiler’s WG cap (from `amdgpu_flat_work_group_size`) |
| `.uses_dynamic_stack` | COV5+ | boolean; dynamic call stack still needs a private-segment budget |

`.agpr_count` is **not** emitted on gfx1030 (no MAI / AGPR). Ignore CDNA AGPR columns.

Code-object V3 docs still phrase `.vgpr_count` / `.sgpr_count` as “GFX6–GFX9”; LLVM still writes the same keys for GFX10. Believe the streamer, not the V3 sentence.

## 2. Extract the gfx1030 code object

From a host `.so` / executable (Clang HIP Support; Offload Bundler):

```
llvm-objcopy --dump-section=.hip_fatbin=fatbin.bin _rocm_C.abi3.so

clang-offload-bundler --list --type=o --input=fatbin.bin
# look for hip-amdgcn-amd-amdhsa--gfx1030  (empty env field → trailing --)

clang-offload-bundler --unbundle --type=o \
  --input=fatbin.bin \
  --targets=hip-amdgcn-amd-amdhsa--gfx1030 \
  --output=gfx1030.co
```

`llvm-objdump --offloading` on the host ELF lists / extracts HIP offload-bundle entries (LLVM CommandGuide; PR #140128). There is **no** `--amdgpu-code-object-metadata` flag on upstream llvm-objdump (LLVM 22 CommandGuide). After you have a `.co`, dump the note:

```
llvm-readelf --notes gfx1030.co
# Owner: AMDGPU   Type: NT_AMDGPU_METADATA (32)
```

Some ROCm `llvm-readelf` / `readelf` builds pretty-print the MessagePack. If the note is hex, decode it with the snippet below.

A raw `.hsaco` / `.co` (device-link output, no fatbin) is already the code object — skip unbundle.

## 3. pyelftools + msgpack snippet

```python
# pip: pyelftools msgpack
from elftools.elf.elffile import ELFFile
import msgpack, sys

NT_AMDGPU_METADATA = 32
KEYS = (
    ".vgpr_count", ".sgpr_count",
    ".vgpr_spill_count", ".sgpr_spill_count",
    ".group_segment_fixed_size", ".private_segment_fixed_size",
    ".wavefront_size", ".max_flat_workgroup_size",
)

def notes(path):
    with open(path, "rb") as f:
        elf = ELFFile(f)
        for sec in elf.iter_sections():
            if sec.header["sh_type"] != "SHT_NOTE":
                continue
            for n in sec.iter_notes():
                if n["n_type"] == NT_AMDGPU_METADATA and n["n_name"].rstrip("\0") == "AMDGPU":
                    yield n["n_descdata"] if "n_descdata" in n else n["n_desc"]

def dump_co(path):
    for blob in notes(path):
        doc = msgpack.unpackb(blob, raw=False, strict_map_key=False)
        for k in doc.get("amdhsa.kernels", []):
            row = {key: k.get(key, 0) for key in KEYS}
            row["name"] = k.get(".name")
            print(row)

if __name__ == "__main__":
    dump_co(sys.argv[1])
```

Walk a fatbin the same way only **after** unbundle. pyelftools on the host `.so` sees the `.hip_fatbin` **section**, not the nested device ELF notes.

FA §8 used this path on `_rocm_C.abi3.so` (venv-7.14.0). Reuse it; do not re-derive.

## 4. Occupancy from the dump (gfx1030 wave32)

Do not re-derive the silicon table — [hip-craft.md](hip-craft.md) §6.1 / [architecture.md](architecture.md): **16 waves/SIMD32**, **1024 VGPR/SIMD**, granule **16**, LDS **128 KB/WGP** but **≤ 64 KB/WG**, SGPR not limiting.

```
vgpr_alloc = ceil(.vgpr_count / 16) * 16          # 0 if count == 0
waves_vgpr = min(16, 1024 // vgpr_alloc)          # LLVM getOccupancyWithNumVGPRs

lds_bytes  = host_dynamic_smem or .group_segment_fixed_size
lds_alloc  = ceil(lds_bytes / 1024) * 1024        # ISA 1 KB granule
wgs_lds    = 65536 // lds_alloc                   # per 64 KB WG cap; 0 if lds > 64 KB

waves_wg   = wg_threads / .wavefront_size         # HIP launch, not metadata
```

Then occupancy = min(wave slots, VGPR budget, LDS budget, WG-size / barrier slots) as in hip-craft §6.1.

`llvm-calc-occupancy -mcpu=gfx1030 -mattr=+wavefrontsize32 --wg-size=N --vgprs=V --lds=K` is the same backend math ([hip-craft.md](hip-craft.md) §6.2). Always pass `+wavefrontsize32`.

## 5. Gotcha — dynamic `extern __shared__`

`.group_segment_fixed_size` is **only** the compiler’s static LDS (`ProgramInfo.LDSSize`). HIP kernels that take `extern __shared__` + a launch `smem` argument compile that field as **0**.

FA §8: every `fa_*` tile uses dynamic shared; metadata LDS = 0; occupancy LDS came from the host `size_t smem` formulas. Same trap on any kernel that stages A/K/V in dynamic LDS.

Rule: **LDS occupancy uses the bytes HIP actually reserves at launch**, not `.group_segment_fixed_size` when that field is 0. Static `__shared__` arrays do show up in the metadata. Report both.

COV5+ may also emit `hidden_dynamic_lds_size` in `.args` when `isDynamicLDSUsed()`; that is a hidden kernarg, not the byte count.

## 6. Gotcha — spill / scratch reject

Reject an inner-loop gfx1030 kernel if **any** of:

- `.vgpr_spill_count` > 0
- `.sgpr_spill_count` > 0
- `.private_segment_fixed_size` > 0
- COV5+ `.uses_dynamic_stack` == true

Scratch is LLVM private address space 5. Hardware allocates per-wave backing and lanes access it with **dword (4 B) interleaving** (LLVM AMDGPUUsage, Private):

```
wavefront-scratch-base
  + (private-address / 4) * wavefront-size * 4
  + wavefront-lane-id * 4
  + (private-address % 4)
```

That traffic is L0 / L2 / Infinity Cache, not a register file. It will not hide behind occupancy the way a VGPR load does. hip-craft §6.3: scratch in an inner loop is fatal.

gfx1030’s scratch ABI is **Absolute flat scratch** (LLVM Processors table, `amdgpu10.30` / `gfx10-3-generic`). Flat access to scratch needs FLAT_SCRATCH_LO/HI setup in the kernel prolog. That is extra SGPR pressure, not extra occupancy slots — SGPRs still do not limit GFX10 occupancy — but it is one more reason a spilled kernel is not “almost fine.”

`.vgpr_count` is **not** rounded in metadata (LLVM: “not rounded up to the allocation granularity”). Occupancy rounding is the dump reader’s job (granule 16). A 203-VGPR compile is 208 allocated — see hip-craft §6.5 (llama.cpp #24672 occupancy-0). Do not treat compiler-remark occupancy as launchable occupancy; still require `hipOccupancyMaxActiveBlocksPerMultiprocessor` ≥ 1.

`.wavefront_size` must be **32** for HIP gfx1030 as shipped. 64 doubles VGPR cost per wave (physical file 512 in wave64 accounting) and is not the HIP default.

## 7. What this page is not

- Not a FA pin change. FA compiled numbers stay on [fa-occupancy.md](fa-occupancy.md) §8.
- Not a `__launch_bounds__` / `amdgpu_waves_per_eu` rewrite. That lowering is hip-craft §1.3.
- **Does not change extras HIP.** **Does not change UNC cards.** No new Linear ticket. No tok/s.

Apply the dump to kernels that still lack a resource table (EXL3 dense/MoE GEMM, GDN remaining, W4, RMSNorm) before arguing occupancy.

## Sources

1. LLVM AMDGPUUsage — ELF note `NT_AMDGPU_METADATA` = 32, MessagePack, private AS 5 dword-interleave, Absolute flat scratch on `gfx1030` — https://llvm.org/docs/AMDGPUUsage.html
2. LLVM `AMDGPUHSAMetadataStreamer.cpp` `getHSAKernelProps` — keys `.group_segment_fixed_size` / `.private_segment_fixed_size` / `.wavefront_size` / `.sgpr_count` / `.vgpr_count` / `.sgpr_spill_count` / `.vgpr_spill_count` — https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUHSAMetadataStreamer.cpp
3. LLVM AMDGPUMetadataVerifier — required vs optional kernel integers — https://github.com/llvm/llvm-project/blob/main/llvm/lib/BinaryFormat/AMDGPUMetadataVerifier.cpp
4. LLVM r346992 / D53445 — V3 note type 32, MessagePack — https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20181112/603452.html
5. Clang HIP Support — `.hip_fatbin` dump + `clang-offload-bundler --unbundle` — https://clang.llvm.org/docs/HIPSupport.html
6. Clang Offload Bundler — bundle ID `hip-amdgcn-amd-amdhsa--<target-id>` — https://clang.llvm.org/docs/ClangOffloadBundler.html
7. llvm-objdump `--offloading` — https://llvm.org/docs/CommandGuide/llvm-objdump.html
8. llvm-calc-occupancy — https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html
9. GPUOpen Occupancy explained — 16 slots on RDNA2, SGPR not limiting — https://gpuopen.com/learn/occupancy-explained/
10. [fa-occupancy.md](fa-occupancy.md) §8, [hip-craft.md](hip-craft.md) §6, [architecture.md](architecture.md)
