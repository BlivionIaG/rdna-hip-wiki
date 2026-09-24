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
| [lds-bank-occupancy.md](lds-bank-occupancy.md) | Bank conflicts = LDS *latency*, not MaxWGsLDS; pad vs XOR at occupancy cliff |
| [vgpr-occupancy.md](vgpr-occupancy.md) | VGPR waves/EU: 1024 file, granule 16, launch_bounds EU=SIMD32; WG spanning |
| [sgpr-occupancy.md](sgpr-occupancy.md) | SGPR is **not** a limiter on GFX10+; descriptor SGPR count must be 0; 128 always allocated |
| [wg-size-occupancy.md](wg-size-occupancy.md) | PIX Thread Group Size: atomic WG lifetime; flat_work_group_size range; ≠ Barriers |
| [wave-size-occupancy.md](wave-size-occupancy.md) | Wave32 vs wave64: rescales VGPR/N/beats; HIP ships wave32; not a PIX row |
| [wgp-cu-mode-occupancy.md](wgp-cu-mode-occupancy.md) | WGP vs CU (`-mcumode`): LDS pool / barriers / SIMDs-per-WG; HIP ships WGP |
| [occupancy-composite.md](occupancy-composite.md) | Fold: min(VGPR, LDS, WG/barrier); llvm-calc-occupancy; PIX↔LLVM; measured vs theory |
| [icache-occupancy.md](icache-occupancy.md) | I$/SQC 32 KB/WGP: not a PIX MaxWaves row; fetch-stall vs effective occupancy |
| [kcache-occupancy.md](kcache-occupancy.md) | K$/SQC DCache 16 KB/WGP: not a PIX MaxWaves row; scalar/kernarg miss vs effective occupancy |
| [l0-gl1-occupancy.md](l0-gl1-occupancy.md) | Vector L0/TCP 16 KB/CU + GL1 128 KB/SA: not a PIX MaxWaves row; thrash vs effective occupancy |
| [l2-occupancy.md](l2-occupancy.md) | L2 4 MiB GPU-wide (mid vs IC): not a PIX MaxWaves row; thrash vs effective occupancy |
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
| [v340l-macos-tb.md](v340l-macos-tb.md) | Locked: 1× UT4G + 1× 88096 + 8× V340L. Repo [BlivionIaG/v340l-macos](https://github.com/BlivionIaG/v340l-macos) |
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
- [w4a16-prefill-config.md](w4a16-prefill-config.md) — ConfigA K_STEP=32 dest; ConfigH Leave (`7ac98a26`)
| [fa-gqa.md](fa-gqa.md) | FA GQA subgroup default; true variant dropped |
| [../kernels/qwen4exp-flash-next-hip.md](../kernels/qwen4exp-flash-next-hip.md) | Flash-Next HIP glue T46–T49 + mrope fixes @ 5c3c0c6f |
