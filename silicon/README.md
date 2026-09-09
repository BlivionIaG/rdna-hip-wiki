# Silicon

How gfx1030 actually works, and what HIP can control.

| Page | Contents |
|---|---|
| [architecture.md](architecture.md) | WGP/CU, wave32/64, VGPR, L0/L1/L2/IC, no WMMA |
| [lds-tiles.md](lds-tiles.md) | 64-bank formula, GEMM/attention tile seeds, DP4A recipe |
| [cache-policy.md](cache-policy.md) | What HIP can set; IC fit; no persist/bypass |
| [infinity-cache.md](infinity-cache.md) | IC is RDNA2 intro; V620 128 MB / W7800 96 MB; not XGMI |
| [rccl-p2p.md](rccl-p2p.md) | 4× V620 PCIe 4.0, RCCL, P2P attested, GB/s unmeasured |
| [rdna-allreduce.md](rdna-allreduce.md) | T44 push one-shot AR on dest tip a4060647; Uncached+host flags; opt-in VLLM_RDNA_AR=1 |
| [plx-p2p-mmio.md](plx-p2p-mmio.md) | PEX88096/8749 lane budget, ACS `+0x6`, BAR0/MMIO, Linux dump |
| [hip-craft.md](hip-craft.md) | waitcnt, scopes, kernarg, builtins, occupancy workflow |
| [fa-occupancy.md](fa-occupancy.md) | Live `fa_rdna2` LDS/VGPR/`__launch_bounds__` |
| [occupancy-dump.md](occupancy-dump.md) | NT_AMDGPU_METADATA VGPR/SGPR/LDS/scratch dump recipe |
| [scratch-occupancy.md](scratch-occupancy.md) | Scratch/private AS 5 is not a waves/SIMD limiter; ROCr may cut waves_per_cu |
| [barrier-occupancy.md](barrier-occupancy.md) | Barrier slots 32 WGP / 16 CU; single-wave free; not FA leftover |
| [lds-occupancy.md](lds-occupancy.md) | LDS pool 128 KB WGP / 64 KB WG; LLVM 512 B vs ISA 1 KB granule; MaxWGsLDS ladder |
| [vgpr-occupancy.md](vgpr-occupancy.md) | VGPR waves/EU: 1024 file, granule 16, launch_bounds EU=SIMD32; WG spanning |
| [wg-size-occupancy.md](wg-size-occupancy.md) | PIX Thread Group Size: atomic WG lifetime; flat_work_group_size range; ≠ Barriers |
| [graph-capture.md](graph-capture.md) | Capture vs page-commit guard: no D2H on the captured path; `Tensor!` on every HIP out arg |
| [codegen-stack.md](codegen-stack.md) | Use HIP+DOT; ignore CK XDL, hipBLASLt, rocWMMA, AITER |
| [valu.md](valu.md) | gfx1030 VALU: enc / size / issue / which DOT we fire. gfx1100 extras |
| [fp16-rdna2.md](fp16-rdna2.md) | Fastest FP16: explicit `fdot2` + occupancy; skinny decode, measure BLAS prefill |
| [flashkda.md](flashkda.md) | FlashKDA: no SM90/CUTLASS/bf16; 128×128 state = `fdot2` later |
| [sglang-fork.md](sglang-fork.md) | SGLang overlay imports extras HIP; no second DOT / AITER tree |
| [leapdragon.md](leapdragon.md) | leapdragon recipe: grid fill, LDS@256, BLOCK_KN sweep; not our occupancy card |
| [hipfire.md](hipfire.md) | hipfire: capability dispatch + sdot4 MMQ on gfx10; WMMA is gfx11/12 only |
| [flydsl-dot-atoms.md](flydsl-dot-atoms.md) | FlyDSL gfx1030 feasibility; formal `fdot2`/`sdot4` atoms and proof gates |
| [deepep-v620.md](deepep-v620.md) | DeepEP insight → mapped-peer PCIe scatter, not IBGDA |
| [hetero-moe-w7800-v620.md](hetero-moe-w7800-v620.md) | Spark×2 (GB10 CUDA) + 8×V620: activations-only hop, KV on Spark; W7800 row dead |
| [v340l.md](v340l.md) | V340L = Vega10 **gfx900**, dual-die; not a V620 drop-in |
| [v340l-rocm-714.md](v340l-rocm-714.md) | Adopted 2026-09-03: TheRock nightly ROCm 10 `device-gfx900` (Path A). Official 7.14/10.0 debs ❌ Vega |
| [v340l-macos-tb.md](v340l-macos-tb.md) | Locked: 1× UT4G + 1× 88096 + 8× V340L. Repo BlivionIaG/v340l-macos |
| [v340l-tune.md](v340l-tune.md) | 8× V340L Linux tune: COMPUTE+MCLK lock, 110 W/die, 8 GB packing |
| [hippih.md](hippih.md) | hippih stub: three ISAs (`fdot2` / WMMA / `mad_mix`); extras stays first |
| [exl3.md](exl3.md) | EXL3/QTIP: Viterbi is quant-time; infer is 3-inst codebook → half → `fdot2`. Later |
| [../kernels/gdn-decode.md](../kernels/gdn-decode.md) | GDN decode HIP `(2,4)`, no LDS |
| [../kernels/gdn-prefill.md](../kernels/gdn-prefill.md) | GDN prefill 5 HIP; kkt no `fdot2`; wy ~58 KB LDS |
| [../kernels/layernorm.md](../kernels/layernorm.md) | HIP AOT RMSNorm, tiny LDS, no DOT; not FA leftover |
| [dsv4-flash.md](dsv4-flash.md) | DSv4 Flash: mxfp4 expert tile is the decode kernel; leftover is MXFP8 + e8m0/128², not W8A16-FP8 |
| [wafer-gpu-perf.md](wafer-gpu-perf.md) | wafer-ai list: ISA is 70648 + valu, not their MI350/CDNA4 |
| [petit-kernel.md](petit-kernel.md) | causalflow petit FP4: Take shuffle/LDS/denorm caveats; Leave MFMA/CDNA |
| [curvedinf-int8-vllm.md](curvedinf-int8-vllm.md) | curvedinf INT8 fork: Take PTH-KV + GDN fp32; Leave CK/UA/XGMI; `sdot4` Later |
