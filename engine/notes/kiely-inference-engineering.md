# Digest: Inference Engineering (Kiely) → 4× V620 / gfx1030

Audience: LLM_Inference_specialist. Paraphrase only. No invented tok/s. Hardware numbers below are our existing contract, not the book.

---

## 1. Book identity

- **Title:** Inference Engineering
- **Author:** Philip Kiely
- **Publisher:** Baseten Books (Baseten Labs, Inc.)
- **ISBN:** 979-8-9943597-2-3
- **Knowledge cutoff:** finished January 2026 (author’s own statement)
- **Length:** 259 pages (appendices: glossary + reading list)
- **Stance:** production inference for generative models, NVIDIA-datacenter default, engines = vLLM / SGLang / TensorRT-LLM, orchestrator = NVIDIA Dynamo

---

## 2. What the book is

A map of **three stacked layers**, not a kernel cookbook.

**Runtime** is one model on one instance: engine + the six serving tricks (batch, prefix/KV reuse, quant, spec, parallel, PD). **Infrastructure** is what happens after a single replica saturates. **Tooling** is the DX band between a black-box API and a VM.

Recurring rules: (1) more constraints → more performance; (2) more traffic → more techniques pay; (3) techniques compose or fight. This is not an ISA / HIP / measurement book. Treat H100/B200 numbers as illustrations, never as V620 targets.

Full keep/adapt/skip map, steal list, NVIDIA-only ignore list, and wiki-amendment table live in the rest of this page.

## 3. gfx1030 take (short)

- **keep:** P/D split, batching-raises-decode-AI, prefix cache, chunked prefill with measured MBT, weights quantized on device, P50/P90/P99, on-GPU vs E2E.
- **adapt:** ops:byte from packed DOT vs GDDR6 512 GB/s (not H100 295). Matrix unit is fdot2/sdot4, not Tensor Core. Fusion = dequant-into-dot. Hierarchy includes IC 128 MB (book has none).
- **skip:** TRT-LLM, Dynamo, NIXL, TP-for-speed, PD-as-a-win, FP8-as-policy, CUDA graphs as religion, specdec until q>1.
- **flag (not a silicon contradiction):** book’s TP-lowers-latency / PP-not-recommended / integers-not-for-production all assume NVLink + Tensor Core FP8. Applied with the book’s own topology rule, they agree with our map.
- **Live fork:** W4A16 / W8A16 / W8A16-FP8 (LUT→fdot2) + fa_rdna2. W8A8/sdot4 and mxfp4 are contracts, not shipping.

## 4. Steal this

1. Roofline first. Batch is how decode climbs the ridge.
2. Two attention programs: IO-aware prefill tiles vs paged decode gather.
3. Fusion deletes HBM trips on the memory-bound path.
4. Shape-specialized GEMM (fat prefill vs skinny decode vs quant-DOT).
5. Arch-specialized kernels. Do not port H100/CDNA tiles.
6. Sarathi token budget. Measure τ on GDDR6+IC.
7. Prefix layout: first novel token ends the reusable prefix.
8. Spec is a FLOP sponge, ITL-only. Cut at high batch / high T.
9. Quantize least-sensitive first (linears). KV quant is capacity/transfer. Fuse dequant.
10. Topology-aware parallel. Weaker link → fewer syncs, not more TP.
11. PD is a specialization tax. Three gates: ~1e8–1e9 tok/day, model ≳100B, long diverse ISL.
12. Constraints win. Narrow the engine.
13. Ops: real ISL/OSL, P90/P99, concurrency = measured batch.

## 5. Ignore / NVIDIA-only

Tensor Core / NVFP4 / MXFP8 / DeepGEMM / CUTLASS / FlashInfer / FA3-H100. TRT-LLM, NIMs, Dynamo, NIXL, KVBM, GB200 NVL72, Grace C2C. TP-for-latency, EP16, PD as default. “FP8 is the production sweet spot.” Book’s 295 / AI=62 / 30–50% quant / 100M–1B tok/day — illustrations, not SLOs.

## 6. Gaps the book does not cover

RDNA / gfx1030 / Wave32. HIP / ROCm 7 / RCCL / PCIe P2P / no XGMI. Packed DOT. Infinity Cache. `on_gfx1x` gates. Occupancy craft (`amdgpu_waves_per_eu(4,8)`, Split-K = grid-fill). llama.cpp VEC/TILE, ktransformers no-scan-KV-over-PCIe. Measured V620 tok/s — do not invent.

## 7. Contract pages touched

Short Kiely addenda landed on [pd-disagg.md](../pd-disagg.md), [specdec.md](../specdec.md), [batching.md](../batching.md), [kv-quant-offload.md](../kv-quant-offload.md). Verdicts unchanged. Full-map not rewritten.

See the long-form keep/adapt/skip writeup on disk at `/workspace/research/kiely-inference-engineering.md` if you want the per-section notes. This wiki page is the durable index.

---

*Digest written 2026-08-17 (PT). Source: user-provided PDF. Paraphrase only. No long verbatim excerpts.*
