# Occupancy

Hard limiters (VGPR, LDS, barriers, WG size, wave size, WGP/CU mode) and the effective-occupancy pages: things that are not a PIX MaxWaves row but park or starve waves inside a held slot.

Up: [silicon/](../README.md)

| Page | Contents |
|---|---|
| [occupancy-composite.md](occupancy-composite.md) | Fold: min(VGPR, LDS, WG/barrier); llvm-calc-occupancy; PIX↔LLVM; measured vs theory |
| [occupancy-dump.md](occupancy-dump.md) | NT_AMDGPU_METADATA VGPR/SGPR/LDS/scratch dump recipe |
| [vgpr-occupancy.md](vgpr-occupancy.md) | VGPR waves/EU: 1024 file, granule 16, launch_bounds EU=SIMD32; WG spanning |
| [sgpr-occupancy.md](sgpr-occupancy.md) | SGPR is **not** a limiter on GFX10+; descriptor SGPR count must be 0; 128 always allocated |
| [scratch-occupancy.md](scratch-occupancy.md) | Scratch/private AS 5 is not a waves/SIMD limiter; ROCr may cut waves_per_cu |
| [barrier-occupancy.md](barrier-occupancy.md) | Barrier slots 32 WGP / 16 CU; single-wave free; not FA leftover |
| [wg-size-occupancy.md](wg-size-occupancy.md) | PIX Thread Group Size: atomic WG lifetime; flat_work_group_size range; ≠ Barriers |
| [wave-size-occupancy.md](wave-size-occupancy.md) | Wave32 vs wave64: rescales VGPR/N/beats; HIP ships wave32; not a PIX row |
| [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md) | WGP vs CU (`-mcumode`): LDS pool / barriers / SIMDs-per-WG; HIP ships WGP |
| [spi-ace-occupancy.md](spi-ace-occupancy.md) | SPI/ACE: measured ≤ theory (lack of work / launch-rate); V620 fill 36 WGP × 4 × 16 = 2304; not a PIX MaxWaves row |
| [icache-occupancy.md](icache-occupancy.md) | I$/SQC 32 KB/WGP: not a PIX MaxWaves row; fetch-stall vs effective occupancy |
| [kcache-occupancy.md](kcache-occupancy.md) | K$/SQC DCache 16 KB/WGP: not a PIX MaxWaves row; scalar/kernarg miss vs effective occupancy |
| [l0-gl1-occupancy.md](l0-gl1-occupancy.md) | Vector L0/TCP 16 KB/CU + GL1 128 KB/SA: not a PIX MaxWaves row; thrash vs effective occupancy |
| [l0-acquire-barrier-occupancy.md](l0-acquire-barrier-occupancy.md) | WGP per-CU L0 acquire race after grid/lookback barriers (pytorch#198208): not a PIX MaxWaves row; all-wave agent inv vs single-lane; MonoKernel Live |
| [l2-occupancy.md](l2-occupancy.md) | L2 4 MiB GPU-wide (mid vs IC): not a PIX MaxWaves row; thrash vs effective occupancy |
| [infinity-cache-occupancy.md](infinity-cache-occupancy.md) | IC/MALL 128 MiB GPU-wide (last-level vs L2): not a PIX MaxWaves row; thrash vs effective occupancy |
| [hard-clause-occupancy.md](hard-clause-occupancy.md) | gfx10+ `S_CLAUSE` hard clauses: not a PIX MaxWaves row; arbiter lock / burst vs effective interleave |
| [waitcnt-occupancy.md](waitcnt-occupancy.md) | waitcnt / VMCNT+VSCNT+LGKMCNT scoreboard: not a PIX MaxWaves row; park vs effective occupancy |
| [exec-divergence-occupancy.md](exec-divergence-occupancy.md) | EXEC / lane divergence: not a PIX MaxWaves row; lane-idle vs effective occupancy |
| [trans-occupancy.md](trans-occupancy.md) | Transcendental / SFU (¼ rate, 8-wide, co-issue): not a PIX MaxWaves row; TRANS starve vs effective occupancy |
| [vgpr-bank-occupancy.md](vgpr-bank-occupancy.md) | VGPR 4-bank (`vN mod 4`) / SGPR 8 paired-bank operand reads: not a PIX MaxWaves row; LLVM dropped bank reassign (D101313) — footnote, never trade VGPRs |
| [wave-priority-occupancy.md](wave-priority-occupancy.md) | Wave priority (`S_SETPRIO`, SPI+USER prio, age tiebreak) / `S_SLEEP` / `S_WAKEUP`: not a PIX MaxWaves row; arbiter bias / voluntary park inside reserved slots; LLVM `-amdgpu-set-wave-priority` is an off-by-default A/B |
| [atomic-occupancy.md](atomic-occupancy.md) | Atomics: gfx1030 has **no** global/flat FP add atomic — fp32 `atomicAdd` is a `global_atomic_cmpswap` retry loop (even with `-munsafe-fp-atomics`); LDS `ds_add_f32` native; global fmin/fmax native only with no-fine-grained md; not a PIX MaxWaves row (vmcnt parks + EXEC shrink inside a held slot) |
| [dpp-crosslane-occupancy.md](dpp-crosslane-occupancy.md) | Cross-lane ops: DPP `row_shr`/`row_xmask` + `v_permlanex16` are VALU; `__shfl*` = `ds_bpermute` (LDS pipe, 0 LDS bytes, `lgkmcnt` park); `wave_reduce_*` = serial readlane loop on clang 22; `row_bcast`/`wave_*` gone in GFX10; extras `REDUCE_SUM_DPP_WAVE32` (`0x118…0x111`) verified correct; not a PIX MaxWaves row |
| [hazard-occupancy.md](hazard-occupancy.md) | Software hazard waitstates: gfx1030 has HW interlocks and none of the gfx101x hazard features, so clang emits 0 `s_nop`/`s_waitcnt_depctr`/`vscnt null` (probe); gfx1013/gfx101x builds of the same source get them, so gfx1030 objects are unsafe on BC-250; don't copy NOP padding; not a PIX MaxWaves row |

## Sources

This is an index page; it makes no technical claims of its own. Primary sources are cited with footnotes and a `## Sources` section on each linked page.
