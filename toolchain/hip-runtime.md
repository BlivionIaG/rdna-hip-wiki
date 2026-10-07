# HIP runtime / ROCm pin

Date: **2026-10-07**.

| Claim | Status |
|---|---|
| Live target | ROCm **7.14.0** on Fedora 43 and/or RHEL 10 |
| 7.14.1 quality | Real release 2026-09-02. GitHub has **no** `rocm-7.14.1`; TheRock **`therock-7.14.1` → `f51dc6c91e0d`**. Production pin stays **`rocm-7.14.0` @ `830cc1b5e90d`** |
| Latest AMD drop | Core SDK **10.0.0** (2026-08-26) |
| Do we bump? | **No.** Not to 7.14.1 (RCCL IB + amdflang only) and not to 10.0 |

V620 official footnote remains Ubuntu-only. TheRock `gfx103X-dgpu` is a real compiler target; libraries stay excluded.

## Gotcha — the host kernel's amdgpu DRM version is part of the pin (2026-10-07)

Two independent upstream threads say ROCm 7.14-era userspace misbehaves on older in-kernel amdgpu, and both clear on an amdgpu that reports **DRM 3.64** or newer. Check what the host runs with `dmesg | grep "Initialized amdgpu"` (the line ends in the DRM version, e.g. `3.64.0`) before blaming HIP, RCCL or the engine.

- **Silent RCCL corruption (gfx1030 and gfx1100).** AMD (`lucbruni-amd`) attributes wrong `all_gather` / `reduce_scatter` data on the first-enumerated GPU over SHM, and gfxhub page faults at vLLM TP=8, to an amdgpu regression in the Ubuntu 24.04 stock **6.8** kernel (`6.8.0-139`), not to RCCL. On 8×V620 the 6.8 kernel failed 29/43 TP=8 runs; HWE **7.0.0-34** (in-tree amdgpu 3.64.0) ran 10/10 clean with rccl-tests OK. Fix is the HWE 7.0 kernel or `amdgpu-dkms`; `HSA_DISABLE_CACHE=1` is the stopgap. Threads: [legacy-rocm-build#6565](https://github.com/ROCm/legacy-rocm-build/issues/6565), [rocm-systems#12145](https://github.com/ROCm/rocm-systems/issues/12145) (both open).
- **HIP VMM hang below DRM 3.64.** `hipMemAddressReserve` → `hipMemCreate` → `hipMemMap` → `hipMemSetAccess` blocks forever in `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` with 7.14 userspace on DRM 3.61, because libhsakmt asks for GEM_VA timeline output that older amdgpu silently ignores. PyTorch hits it through `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`; turning that off is the workaround. Fix is open, not merged: [rocm-systems#9820](https://github.com/ROCm/rocm-systems/issues/9820) / [#9821](https://github.com/ROCm/rocm-systems/pull/9821) (gates the timeline path on DRM ≥ 3.64).

What this means for us: the live dest stays **7.14.0**, but a host on an older stock kernel can produce RCCL or allocator failures that look like our bugs. Record the amdgpu DRM version next to the ROCm version in any bring-up or bug report. No pin or dest change.

## Watch — ROCm 10.0 HIP graph hang on empty fork/join roots (2026-10-07)

[rocm-systems#12638](https://github.com/ROCm/rocm-systems/issues/12638) (open): on Core SDK 10.0.0, a retained fork/join graph whose root segment holds only empty nodes can hang on first replay, because the segmented executor allocates a completion signal for that segment and never attaches a producer. Not on our 7.14.0 dest; check it before any 10.x move if the engine captures graphs with empty join nodes.

## 2026-10-02 — TheRock systems `bc176d4` (#8588; tip / soak-rebuild watch, not dest)

[TheRock#8588](https://github.com/ROCm/TheRock/pull/8588) **merged** `rocm-systems` **`0bf70ef` → `bc176d4`** (+82). HEAD **`c9bde5b3d052` → `6447c1fb73b3`** (also CI/security and profiler-hub reverts; the material event is the systems pin). Libraries still `916d478` ([#8607](https://github.com/ROCm/TheRock/pull/8607)). Compiler pin unchanged (ww-37-SMP1.1 / amd-llvm `4f43f4746ede`; hipify `501cd6c1`; spirv `2c14c774`). Nightly **`10.2.0a20261001` → `10.2.0a20261002` L+W** (core + libraries + device-gfx1030/1100/900).

CLR/runtime notes now in the TheRock systems pin (still **not** live dest):

- **HIP `__half`**: [rocm-systems#12269](https://github.com/ROCm/rocm-systems/pull/12269) — integral assign.
- **ROCR vmem**: [rocm-systems#12201](https://github.com/ROCm/rocm-systems/pull/12201) — vmem handle flags.
- **ROCR blit**: [rocm-systems#12209](https://github.com/ROCm/rocm-systems/pull/12209) — publish blit kernel code after AssembleShader.
- **Windows SVM**: [rocm-systems#11414](https://github.com/ROCm/rocm-systems/pull/11414) — default aperture 256 GiB → 4 TiB.

**No gfx1030/1100/900 ISA or support-matrix change** (gfx1030/1100 still Release Ready; gfx900 still Build Passing). [#8696](https://github.com/ROCm/TheRock/pull/8696) closed unmerged. Open watches: libraries [#8697](https://github.com/ROCm/TheRock/pull/8697) (`916d478`→`762c581`, CI failing); draft SMP ww38.1.2 [#8569](https://github.com/ROCm/TheRock/pull/8569) (CI failing). Live dest stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP.

## 2026-09-29 — TheRock libraries `efafc1c` (#8561; tip, not dest)

[TheRock#8561](https://github.com/ROCm/TheRock/pull/8561) **merged** `rocm-libraries` **`6054c51` → `efafc1c`**. HEAD **`8b46a27d2476`**. Systems still `0bf70ef`. Compiler pin unchanged (ww-37-SMP1.1). Nightly still **`10.2.0a20260929` L+W**. Math/library tip only (hipdnn/rocprim-gfx1250/ci/fmha-gfx1250/test; no CLR/HIP runtime pin move; no gfx1030 ISA lever). Live dest stays **7.14.0**.

#8576/#8587 closed unmerged. Open: libraries [#8594](https://github.com/ROCm/TheRock/pull/8594); systems [#8588](https://github.com/ROCm/TheRock/pull/8588)/[#8583](https://github.com/ROCm/TheRock/pull/8583).

## 2026-09-29 — TheRock systems `0bf70ef` (#8567; tip, not dest)

[TheRock#8567](https://github.com/ROCm/TheRock/pull/8567) **merged** `rocm-systems` **`0ad5f73` → `0bf70ef`**. HEAD **`182f764488ee`**. Compiler pin unchanged (ww-37-SMP1.1). Libraries still `6054c51`. Nightly still **`10.2.0a20260929` L+W**.

CLR/runtime notes now in the TheRock systems tip (still **not** live dest):

- **Batch copy enqueue**: [rocm-systems#11650](https://github.com/ROCm/rocm-systems/pull/11650) — `hipMemcpyBatchAsync` reserves only the descriptor size needed per dispatch (avoids host wait after 16 dispatches).
- **rocddi**: [rocm-systems#12228](https://github.com/ROCm/rocm-systems/pull/12228) — Rust AMDF/HSA systems layer (validated gfx1201; not a gfx1030 dest lever).

#8557 closed unmerged (superseded by #8567). Open: systems [#8583](https://github.com/ROCm/TheRock/pull/8583); libraries [#8576](https://github.com/ROCm/TheRock/pull/8576)/[#8561](https://github.com/ROCm/TheRock/pull/8561). Live dest stays **7.14.0**.

## 2026-09-28 — TheRock systems `0ad5f73` + libraries `6054c51` (tip, not dest)

[TheRock#8533](https://github.com/ROCm/TheRock/pull/8533) **merged** `rocm-systems` **`2cd17fb` → `0ad5f73`**. [#8530](https://github.com/ROCm/TheRock/pull/8530) **merged** `rocm-libraries` **`459a4ec` → `6054c51`**. HEAD **`7440cb8578f4`**. Compiler pin unchanged (ww-37-SMP1.1). Nightly **`10.2.0a20260928` Linux-only** (L+W tip still `10.2.0a20260927`).

CLR/runtime notes now in the TheRock systems tip (still **not** live dest):

- **Same-device D2D**: [rocm-systems#12250](https://github.com/ROCm/rocm-systems/pull/12250) — batch-swap path.
- **RCCL LL**: [rocm-systems#10670](https://github.com/ROCm/rocm-systems/pull/10670) — explicit 128-bit vector loads/stores (tested gfx1100/gfx1201/gfx950; preserves GFX11 inline-asm b128).

Open follow-ons: systems [#8557](https://github.com/ROCm/TheRock/pull/8557) (`0ad5f73`→`f939bbe`); libraries [#8561](https://github.com/ROCm/TheRock/pull/8561). Live dest stays **7.14.0**.

## 2026-09-28 — TheRock systems `2cd17fb` + libraries `459a4ec` (tip, not dest)

[TheRock#8508](https://github.com/ROCm/TheRock/pull/8508)/[#8525](https://github.com/ROCm/TheRock/pull/8525) **merged** `rocm-systems` **`b0aa8f2` → `2cd17fb`**. [#8512](https://github.com/ROCm/TheRock/pull/8512) **merged** `rocm-libraries` **`ecb3f35` → `459a4ec`**. Compiler pin unchanged (ww-37-SMP1.1). Nightly **`10.2.0a20260927` L+W**.

CLR/runtime notes now in the TheRock systems pin (still **not** live dest):

- **Graph deps**: [rocm-systems#12076](https://github.com/ROCm/rocm-systems/pull/12076) — barrier-value packets for single graph dependencies.
- **rocjitsu**: multiple emulator harden/vectorize/hazard-detect commits (still Experimental for gfx1030; no topology JSON).
- **rocSHMEM**: [rocm-systems#12195](https://github.com/ROCm/rocm-systems/pull/12195) — replace gfx1100 with general GFX11.
- **Blit**: Accelerated Blit Copy Engine (ABCE) intro ([#12161](https://github.com/ROCm/rocm-systems/pull/12161)).
- **hiprtc**: remove `HIPRTC_USE_RUNTIME_UNB…` env ([#12239](https://github.com/ROCm/rocm-systems/pull/12239)).

Open follow-ons: systems [#8542](https://github.com/ROCm/TheRock/pull/8542) (`2cd17fb`→`e4826ad`, includes CLR same-device batch swap). Live dest stays **7.14.0**.

## 2026-09-25 — TheRock libraries `ecb3f35` (#8472; tip, not dest)

[TheRock#8472](https://github.com/ROCm/TheRock/pull/8472) **merged** 2026-09-25 ~13:49 Europe/Paris (`fac0dbc97d76`). `rocm-libraries` **`5911365` → `ecb3f35`**. Systems still `b0aa8f2`. Compiler pin unchanged (ww-37-SMP1.1). Nightly still **`10.2.0a20260925` L+W**. Math/comms library tip only (no CLR/HIP runtime pin move; no gfx1030 ISA lever). Live dest stays **7.14.0**.

## 2026-09-25 — TheRock systems `b0aa8f2` (#8470; tip, not dest)

[TheRock#8470](https://github.com/ROCm/TheRock/pull/8470) **merged** 2026-09-25 ~07:02 Europe/Paris (`49f90b5394`). `rocm-systems` **`9799b78` → `b0aa8f2`**. Compiler pin unchanged (ww-37-SMP1.1). Libraries still `5911365`. Nightly **`10.2.0a20260925` L+W**.

CLR/runtime notes now in the TheRock systems pin (still **not** live dest):

- **Unified-memory blit**: [rocm-systems#10962](https://github.com/ROCm/rocm-systems/pull/10962) — staging copy instead of host pinning on unified-memory devices (`rocblit.cpp`).
- **Module API**: [rocm-systems#11025](https://github.com/ROCm/rocm-systems/pull/11025) — `hipModuleEnumerateFunctions`.
- **Code object load**: [rocm-systems#11542](https://github.com/ROCm/rocm-systems/pull/11542) — CO loading fix in `program.cpp`.
- **Graph profiler**: [rocm-systems#11742](https://github.com/ROCm/rocm-systems/pull/11742) — report BARRIER_AND/OR packets on CLR profiler timeline.
- **ROCR**: [rocm-systems#11617](https://github.com/ROCm/rocm-systems/pull/11617) — increase fallback cache line size.
- **RCCL**: registration/teardown harden ([#11954](https://github.com/ROCm/rocm-systems/pull/11954)); same-domain NET path keyed on physical device ([#12051](https://github.com/ROCm/rocm-systems/pull/12051)).
- **Profiler**: rocprofiler-sdk WSL2 compute/profiling for RDNA 3 ([#7016](https://github.com/ROCm/rocm-systems/pull/7016)) — gfx110x WSL path; not a gfx1030 dest lever.

Follow-on open: [#8506](https://github.com/ROCm/TheRock/pull/8506) systems `b0aa8f2`→`a46ee26`. Live dest stays **7.14.0**.

## 2026-09-17 — CLR HIP minor 7.17 (upstream tip, not dest)

[ROCm/clr `1cb204b7`](https://github.com/ROCm/clr/commit/1cb204b7) (2026-09-17 ~15:04 Europe/Paris): HIP minor **7.16 → 7.17**, OpenCL **3684 → 3686** for new APIs in ROCm **10.2**. Not in live dest; production pin stays **7.14.0**.

## 2026-09-17 — TheRock systems `d378a17` (#8258; tip, not dest)

[TheRock#8258](https://github.com/ROCm/TheRock/pull/8258) **merged** 2026-09-17 15:56 Europe/Paris (`bbd401f0`). `rocm-systems` **`1091c91` → `d378a17`**. Compiler pin unchanged (ww36-2.1). Libraries still `320d658`. Nightly still **`10.2.0a20260917` L+W** (no `a20260918`).

CLR/runtime notes now in the TheRock systems pin (still **not** live dest):

- **Graph capture**: [rocm-systems#11517](https://github.com/ROCm/rocm-systems/pull/11517) — skip fork bookkeeping when a stream waits on its **own** captured event (was infinite recurse / SIGSEGV in `hipStreamEndCapture`). Device-scoped invalidation (#8338) **landed then reverted** (#11699) — do not rely on it.
- **Uncached pools**: [rocm-systems#11343](https://github.com/ROCm/rocm-systems/pull/11343) — `HSA_AMD_MEMORY_POOL_UNCACHED_FLAG` limited to **gfx12.0** (`gfx1200`/`gfx1201`) on legacy + VMM paths. **gfx1030 / gfx1100 / gfx900 do not get that flag** under this tip; gfx1250+ uses extended fine-grain instead. Relevant to Uncached P2P page-pool policy — do not assume Uncached flag behavior from CDNA/gfx12 docs on RDNA2.
- OpenCL build number in pin **3581 → 3684**. Standalone CLR tip `1cb204b7` (HIP minor **7.17**) is still **ahead** of this systems pin.

Drop `#8288` (closed unmerged; superseded). Still open: systems follow-on `#8300`/`#8294`, libraries `#8289`, COT+ASAN `#8248`/`#8291`. Live dest stays **7.14.0**.

