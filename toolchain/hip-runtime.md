# HIP runtime / ROCm pin

Date: **2026-09-03**.

| Claim | Status |
|---|---|
| Live target | ROCm **7.14.0** on Fedora 43 and/or RHEL 10 |
| 7.14.1 quality | Real release 2026-09-02. GitHub has **no** `rocm-7.14.1`; TheRock **`therock-7.14.1` → `f51dc6c91e0d`**. Production pin stays **`rocm-7.14.0` @ `830cc1b5e90d`** |
| Latest AMD drop | Core SDK **10.0.0** (2026-08-26) |
| Do we bump? | **No.** Not to 7.14.1 (RCCL IB + amdflang only) and not to 10.0 |

V620 official footnote remains Ubuntu-only. TheRock `gfx103X-dgpu` is a real compiler target; libraries stay excluded.
