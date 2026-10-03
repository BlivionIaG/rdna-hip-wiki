# HIP runtime / ROCm pin

Date: **2026-10-02**.

| Claim | Status |
|---|---|
| Live target | ROCm **7.14.0** on Fedora 43 and/or RHEL 10 |
| 7.14.1 quality | Real release 2026-09-02. GitHub has **no** `rocm-7.14.1`; TheRock **`therock-7.14.1` → `f51dc6c91e0d`**. Production pin stays **`rocm-7.14.0` @ `830cc1b5e90d`** |
| Latest AMD drop | Core SDK **10.0.0** (2026-08-26) |
| Do we bump? | **No.** Not to 7.14.1 (RCCL IB + amdflang only) and not to 10.0 |

V620 official footnote remains Ubuntu-only. TheRock `gfx103X-dgpu` is a real compiler target; libraries stay excluded.

## 2026-10-02 — TheRock systems `bc176d4` (#8588; tip / soak-rebuild watch, not dest)

[TheRock#8588](https://github.com/ROCm/TheRock/pull/8588) **merged** `rocm-systems` **`0bf70ef` → `bc176d4`** (+82). HEAD **`c9bde5b3d052` → `6447c1fb73b3`** (also CI/security and profiler-hub reverts; the material event is the systems pin). Libraries still `916d478` ([#8607](https://github.com/ROCm/TheRock/pull/8607)). Compiler pin unchanged (ww-37-SMP1.1 / amd-llvm `4f43f4746ede`; hipify `501cd6c1`; spirv `2c14c774`). Nightly **`10.2.0a20261001` → `10.2.0a20261002` L+W** (core + libraries + device-gfx1030/1100/900).

CLR/runtime notes now in the TheRock systems pin (still **not** live dest):

- **HIP `__half`**: [rocm-systems#12269](https://github.com/ROCm/rocm-systems/pull/12269) — integral assign.
- **ROCR vmem**: [rocm-systems#12201](https://github.com/ROCm/rocm-systems/pull/12201) — vmem handle flags.
- **ROCR blit**: [rocm-systems#12209](https://github.com/ROCm/rocm-systems/pull/12209) — publish blit kernel code after AssembleShader.
- **Windows SVM**: [rocm-systems#11414](https://github.com/ROCm/rocm-systems/pull/11414) — default aperture 256 GiB → 4 TiB.

**No gfx1030/1100/900 ISA or support-matrix change** (gfx1030/1100 still Release Ready; gfx900 still Build Passing). [#8696](https://github.com/ROCm/TheRock/pull/8696) closed unmerged. Open watches: libraries [#8697](https://github.com/ROCm/TheRock/pull/8697) (`916d478`→`762c581`, CI failing); draft SMP ww38.1.2 [#8569](https://github.com/ROCm/TheRock/pull/8569) (CI failing). Live dest stays **7.14.0** `hipcc --offload-arch=gfx1030 -O3` wave32 WGP.

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

