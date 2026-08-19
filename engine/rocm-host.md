# ROCm host — official matrix vs this box

Date: 2026-08-19. Engine contract. Do **not** treat AMD’s OS footnote as “this box cannot run.” Occupancy still first.

## Official (AMD docs — do not erase)

ROCm **7.2.0** and **7.14.0** system-requirements: V620 / gfx1030 is ✅ with footnote **Ubuntu 24.04.x and 22.04.5 only**. RHEL is in the *general* ROCm table; the V620 footnote **excludes** it (`RHEL 10.1 / 9.7 … except AMD Radeon PRO V620`).

Sources: [7.2 system-requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/reference/system-requirements.html) footnote [8]; [latest](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) footnote [8] / [9].

That is **QA scope**, not physics. Fedora is not in the matrix. “Not on RHEL” in [silicon/rccl-p2p.md](../silicon/rccl-p2p.md) meant the footnote, not a measured fail.

## Attested (operator, 2026-08-19)

| Stack | Status |
|---|---|
| ROCm **7.2.0** on **Fedora 43** | Works |
| **RHEL 10** | Works |
| ROCm **7.14.0** (current) | Works |

No tok/s claimed. This is host/runtime, not a kernel win.

## Engine rule

- **Live target:** ROCm **7.14.0** on this host (Fedora 43 and/or RHEL 10 as used).
- **Official matrix:** keep quoting Ubuntu-only for V620 so we do not file AMD bugs as if Fedora/RHEL were supported SKUs.
- Docker (`vllm-rdna-docker`) may still be Ubuntu — that is image policy, not a host ban.
- `rdna2_extras` PYNCCL bypass (`3e05abc9`) was for Torch 2.12 + a 7.14-class venv. Stay on that path unless a 7.14 dispatcher is re-measured.
- Do not mix V340L onto this ROCm 7 host ([plx.md](plx.md)).

@RDNA2_Researcher: retip the “Not on RHEL” cell in `silicon/rccl-p2p.md` to **official Ubuntu-only / attested Fedora 43 + RHEL 10 + 7.14**.

## Sources

- Room 2026-08-19 (operator)
- AMD ROCm 7.2 / 7.14 install system-requirements (V620 footnote)
