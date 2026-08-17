# Engine notes

Working notes and source digests. Not the silicon / engine contract.

Owned by LLM_Inference_specialist. Paraphrase only. Do not invent tok/s.

| Page | What |
|---|---|
| [session-2026-08-17.md](session-2026-08-17.md) | Full lock dump for 2026-08-17/18 (branch tip, MoE, Triton, FlyDSL, DeepEP) |
| [kiely-inference-engineering.md](kiely-inference-engineering.md) | Kiely, *Inference Engineering* (Baseten, Jan 2026) mapped onto gfx1030 |
| [ikantkode-qwen35.md](ikantkode-qwen35.md) | ikantkode Qwen3.5-4B-AWQ-vd + gfx1030-vllm-0.26 overlay on Blivion docker |
| [rocmfpx.md](rocmfpx.md) | charlie12345/ROCmFPX digest — GGUF codebook, not a vLLM port |

Contract pages stay in [../](../README.md). If a note changes a verdict, patch the contract page and say so here.

Qwen3.5 digest → contract [../qwen35.md](../qwen35.md) (LLMM1 gate, `(1+w)` RMSNorm, AWQ-vd recipe). Their tok/s stay on the note.

ROCmFPX digest → [../rocmfpx.md](../rocmfpx.md) + [../llamacpp-rocmfpx.md](../llamacpp-rocmfpx.md). Stock GGUF loader ≠ their custom types.
