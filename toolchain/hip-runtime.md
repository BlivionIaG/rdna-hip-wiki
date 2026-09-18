# HIP runtime / ROCm pin

Date: **2026-09-03**.

| Claim | Status |
|---|---|
| Live target | ROCm **7.14.0** on Fedora 43 and/or RHEL 10 |
| 7.14.1 quality | Real release 2026-09-02. GitHub has **no** `rocm-7.14.1`; TheRock **`therock-7.14.1` → `f51dc6c91e0d`**. Production pin stays **`rocm-7.14.0` @ `830cc1b5e90d`** |
| Latest AMD drop | Core SDK **10.0.0** (2026-08-26) |
| Do we bump? | **No.** Not to 7.14.1 (RCCL IB + amdflang only) and not to 10.0 |

V620 official footnote remains Ubuntu-only. TheRock `gfx103X-dgpu` is a real compiler target; libraries stay excluded.

## 2026-09-17 — CLR HIP minor 7.17 (upstream tip, not dest)

[ROCm/clr `1cb204b7`](https://github.com/ROCm/clr/commit/1cb204b7) (2026-09-17 ~15:04 Europe/Paris): HIP minor **7.16 → 7.17**, OpenCL **3684 → 3686** for new APIs in ROCm **10.2**. Not in live dest; production pin stays **7.14.0**.

## 2026-09-17 — TheRock systems `d378a17` (#8258; tip, not dest)

[TheRock#8258](https://github.com/ROCm/TheRock/pull/8258) **merged** 2026-09-17 15:56 Europe/Paris (`bbd401f0`). `rocm-systems` **`1091c91` → `d378a17`**. Compiler pin unchanged (ww36-2.1). Libraries still `320d658`. Nightly still **`10.2.0a20260917` L+W** (no `a20260918`).

CLR/runtime notes now in the TheRock systems pin (still **not** live dest):

- **Graph capture**: [rocm-systems#11517](https://github.com/ROCm/rocm-systems/pull/11517) — skip fork bookkeeping when a stream waits on its **own** captured event (was infinite recurse / SIGSEGV in `hipStreamEndCapture`). Device-scoped invalidation (#8338) **landed then reverted** (#11699) — do not rely on it.
- **Uncached pools**: [rocm-systems#11343](https://github.com/ROCm/rocm-systems/pull/11343) — `HSA_AMD_MEMORY_POOL_UNCACHED_FLAG` limited to **gfx12.0** (`gfx1200`/`gfx1201`) on legacy + VMM paths. **gfx1030 / gfx1100 / gfx900 do not get that flag** under this tip; gfx1250+ uses extended fine-grain instead. Relevant to Uncached P2P page-pool policy — do not assume Uncached flag behavior from CDNA/gfx12 docs on RDNA2.
- OpenCL build number in pin **3581 → 3684**. Standalone CLR tip `1cb204b7` (HIP minor **7.17**) is still **ahead** of this systems pin.

Drop `#8288` (closed unmerged; superseded). Still open: systems follow-on `#8300`/`#8294`, libraries `#8289`, COT+ASAN `#8248`/`#8291`. Live dest stays **7.14.0**.

