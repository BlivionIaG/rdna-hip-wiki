# KV cache quantization and offloading

Date: 2026-08-17. Sourced only.

## Quant

Shipped elsewhere: vLLM FP8 / per-token-head INT8/INT4 / TurboQuant / NVFP4. SGLang FP8 + experimental FP4. Official ROCm dtype is `fp8_e4m3` on MI-class. gfx11 is documented "no FP8" (PR #34741). gfx1030: `supports_fp8()` is false. No sourced FP8-KV kernel.

If you write KV quant here, start from **INT8 + fused dequant into paged-decode**. Unfused dequant erases the memory win. Native cvt is the honest path; do not plan an FP8 path.

IC leftover after a resident weight shard is hundreds of KV tokens, not the full cache. Paged-decode that walks the whole cache still streams GDDR6; IC only helps the hot pages this step touches. See [silicon/cache-policy.md](../silicon/cache-policy.md).

## Offload

GPU L1 → host L2 → Mooncake/NIXL/LMCache L3. Helps **prefix reuse and preemption**. Does **not** hide per-token decode H2D. HiCache hides H2D behind **prefill**, not per-token decode. ktransformers refuses the scan-KV-over-PCIe path (sparse CPU attn on DRAM). See [alt-engines.md](alt-engines.md).

OffloadingConnector is GPU→CPU, not PD. Streaming decode KV over PCIe every token is the wrong physics.

## Unknowns

- Specdec / FP8-KV quality on gfx1030.
- Whether any fused INT8-KV + fa_rdna2 path exists in the unpushed tree.
