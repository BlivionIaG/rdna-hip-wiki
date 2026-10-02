# RDNA2 project sweep 2026-10-02 (morning)

Weekday sweep vs 2026-10-01. No dest tip/UNC bump from this pass. Dest produce stays HIP gfx1030 packed `fdot2`/`sdot4` under hipcc 7.14. Mojo≠produce.

## Take Later (ideas only — rewrite; no blind merge)

- [tuandat3019/rdna2-llamacpp-optimizations](https://github.com/tuandat3019/rdna2-llamacpp-optimizations) tip [`f94e8f38`](https://github.com/tuandat3019/rdna2-llamacpp-optimizations/commit/f94e8f38c2efc4b0f968abc658ed25caaac9527a) (2026-10-01): measured RDNA2 llama.cpp — native q4_0 KV in FA tile (no F16 staging), MTP+ngram-mod gates, ADD/RMS/MUL fusion ports; custom FA archived as no TPS gain. **Leave** llama.cpp runtime/product.
- [leapdragon/vllm-rdna2-qwen](https://github.com/leapdragon/vllm-rdna2-qwen) tip [`4499c6ff`→`2a3ae3c2`](https://github.com/leapdragon/vllm-rdna2-qwen/commit/2a3ae3c2f7e5e72629d3e065cc8a1118fe301eb4): `PLE_OFFLOAD_ANON` Engram/n-gram host-RAM path (anon copy vs page-cache thrash under KV offload) + SimpleCPUOffload QSA-ring docs. Aligns with V4.1 Engram-in-host-RAM intent as methodology only. **Leave** fork merge.

## Leave / Dead / ignore

- Triton [PR #12042](https://github.com/triton-lang/triton/pull/12042) **merged** — remove RDNA1/RDNA2 targets → Triton-on-RDNA2 upstream path **Dead/Leave** for gfx1030 Produce.
- [manuelbaez/v620-inference-engine](https://github.com/manuelbaez/v620-inference-engine) tip→[`e575e7e0`](https://github.com/manuelbaez/v620-inference-engine/commit/e575e7e07cdcd174d9b6b5a3ab52439de0900eb1): server/API + SSD docs only (**Leave** product).
- [idragonfly-ai/rocm-patches](https://github.com/idragonfly-ai/rocm-patches): README claims gfx1030; tree is gfx906 Tensile/hsaco only → **ignore**.
- `abihsoro/qwen38-flash-next-v620` → **404 gone** (public leftover: `abihsoro/qwen38-flash-next-mi50`).
- Watchlist frozen: hipfire `a89ed0a8`, furnace `a7665356`, MicoinSmith `833739d2`, windows-rdna2 `71b787e6`, recipe `8e1da3ae`, flashnext-pp3 `5a6d1c2a`, 2kiss `28ff8153`.

## Dest tip (confirm only — not mutated here)

`opengfx1030/vllm-rdna` `rdna_extras` tip [`bb40498c`](https://github.com/opengfx1030/vllm-rdna/commit/bb40498cc00255281be56973ce61b743885f37a9) (PR#34 GQA prefill HEAD_DIM + EXL3 MTP=2 + serve recipes). Any dest/UNC call → **VLLM_FORK_Manager**.
