# Idle upstream awareness 2026-09-25

Window ≈ 2026-09-24 16:14 → 2026-09-25 ~15:56 Europe/Paris. Prior watches + topic PRs (`gh` search API). Portable merges (align hash-block prompt-tail APC restore, batched Mamba2 SSM saves, hybrid padded tails as spec-decode, PLE prefetch outside CG pool, HiCache drain-before-Mamba-writeback / relative donation / batch D2H, DP draft sync + mixed-chunk DP-spec, MTP one-layer plan, UnreducedOutput deferred FFN AR, physical page KV budgets); **no** extras/UNC move. No clones. No invented tok/s.

## Verdict

**No dest bump.** Pin stays **7.14**. UNC-26 EXL3/`-cb 3inst` untouched. APC stays **off** until Flash-Next align-state copy fixed (third APC path) — open [#57807](https://github.com/vllm-project/vllm/pull/57807) / [#55507](https://github.com/vllm-project/vllm/pull/55507) / [#57908](https://github.com/vllm-project/vllm/pull/57908) **still reinforce**, do not lift. No new UNC. PD remains **Dead** for PCIe gfx1030. Expert DRAM offload remains **Dead** when MoE fits after quant. Lab all-reduce stays **PYNCCL** (`VLLM_RDNA_AR=0`) — Leave custom-AR [#57182](https://github.com/vllm-project/vllm/pull/57182).

## Take Later (portable ideas only)

- Align-grid vs hash-block mismatch: restore hybrid Mamba/GDN+MTP prompt-tail prefix-cache hits when `hash_block_size` < Mamba `block_size` — [vLLM #58368](https://github.com/vllm-project/vllm/pull/58368) **MERGED** (same family as APC-off reason; **not** Flash-Next align-state copy fix).
- Prefill SSM state stores are one batched indexed write (no per-seq GPU↔CPU sync) — [vLLM #49371](https://github.com/vllm-project/vllm/pull/49371) **MERGED**.
- Partial-block `BlockStored` must carry every multimodal extra key — [vLLM #58288](https://github.com/vllm-project/vllm/pull/58288) **MERGED**.
- Hybrid padded prompt tails over prior state classify as **spec-decode** rows (placeholders roll back; prefill would fold them into recurrent state) — [vLLM #58434](https://github.com/vllm-project/vllm/pull/58434) **MERGED**.
- Breakable CG side-stream payloads that outlive the eager-break segment need strong pinned storage **outside** the graph pool — [vLLM #58489](https://github.com/vllm-project/vllm/pull/58489) **MERGED**.
- Huge embedding tables can stay host-pinned with UVA (Inductor must not materialize a device copy) — [vLLM #57497](https://github.com/vllm-project/vllm/pull/57497) **MERGED** (**Leave** Qwen4Exp/MI300X sizes).
- Connector `wait_for_save` runs on empty scheduler steps, not only forwards — [vLLM #57775](https://github.com/vllm-project/vllm/pull/57775) **MERGED**.
- Sparse MLA packed block-stride alignment lives on `KVCacheSpec` (LCM of constraints) — [vLLM #55528](https://github.com/vllm-project/vllm/pull/55528) **MERGED**.
- Non-DCP Mamba donation is **prefix-relative** (not absolute radix-page grid) — [SGLang #40907](https://github.com/sgl-project/sglang/pull/40907) **MERGED** (**Take Later** donation gate; **Leave** HiCache/AMD product).
- Drain outstanding HiCache backups before internal Mamba write-back (mates [#40680](https://github.com/sgl-project/sglang/pull/40680)) — [SGLang #41092](https://github.com/sgl-project/sglang/pull/41092) **MERGED**.
- Mixed-chunk prefill × DP-spec admission (local 1-tok decode progress without mixed verify) — [SGLang #41179](https://github.com/sgl-project/sglang/pull/41179) **MERGED**.
- HiCache buffer-only batch D2H within a flush — [SGLang #40960](https://github.com/sgl-project/sglang/pull/40960) **MERGED** (**Leave** HiCache product).
- Sidecar SWA/Mamba storage transfers need a contiguous `prefix_keys` chain — [SGLang #40456](https://github.com/sgl-project/sglang/pull/40456) **MERGED** (**Leave** HiCache product).
- DP×spec draft scopes inherit the target's DP sync slot / pad width — [SGLang #41062](https://github.com/sgl-project/sglang/pull/41062) **MERGED**.
- MTP/NextN drafts are one-layer layout plans (first+last), not middle layers under attn-TP scatter — [SGLang #41194](https://github.com/sgl-project/sglang/pull/41194) **MERGED**.
- Deferred FFN all-reduce completion is an explicit `UnreducedOutput` value, not a tensor attribute — [SGLang #41196](https://github.com/sgl-project/sglang/pull/41196) **MERGED**.
- Finished-req tree insert ≠ release ownership (`release_kv_cache` for abort/retract) — [SGLang #40988](https://github.com/sgl-project/sglang/pull/40988) **MERGED**.
- KV budgets price physical page size (`page_bytes`), not logical latent bytes — [SGLang #41091](https://github.com/sgl-project/sglang/pull/41091) **MERGED**.
- Status flip: [#40907](https://github.com/sgl-project/sglang/pull/40907) MERGED (Take Later relative donation gate; Leave HiCache product). New open watches: vLLM [#58642](https://github.com/vllm-project/vllm/pull/58642)/[#58676](https://github.com/vllm-project/vllm/pull/58676); SGLang [#41218](https://github.com/sgl-project/sglang/pull/41218)/[#41171](https://github.com/sgl-project/sglang/pull/41171)/[#41173](https://github.com/sgl-project/sglang/pull/41173)/[#41182](https://github.com/sgl-project/sglang/pull/41182); llama [#29353](https://github.com/ggml-org/llama.cpp/pull/29353). Still open core: vLLM #57158/#57183/#57180/#57128/#56953/#56971/#57010/#56936/#56957/#57182/#56455/#56243/#56177/#56869/#57908/#57785/#57881/#57943/#57959/#57569/#54743/#56466/#53520/#56089/#57184/#57170/#58108/#57807/#55507/#58196/#58198/#58168; SGLang #39799/#39605/#39575/#39606/#39815/#38887/#38663/#40166/#40159/#39787/#39803/#39960/#39792/#40571/#40543/#40857; llama.cpp #28873/#28927/#29019/#29191/#29208; ExL #379/#315/#392; KT #2190; NIXL #2196/#2244/#2252/#2241/#2225.

## Leave

- Custom ROCm AR [#57182](https://github.com/vllm-project/vllm/pull/57182); AITER/ROCm MXFP8 gfx950 / TRTLLM defer finalize [#58635](https://github.com/vllm-project/vllm/pull/58635) / XQA SpecDec [#32269](https://github.com/sgl-project/sglang/pull/32269) / DeepEP per-rank bound [#41201](https://github.com/sgl-project/sglang/pull/41201) / PD deferred release [#41023](https://github.com/sgl-project/sglang/pull/41023) / `none` retraction [#41103](https://github.com/sgl-project/sglang/pull/41103) as product; HiCache/Mooncake/NIXL/MoRIIO/UCX product; Ascend/NPU; DecodeBench [#58472](https://github.com/vllm-project/vllm/pull/58472); ExL codebook suffix [#391](https://github.com/turboderp-org/exllamav3/pull/391); Engram THP [#56926](https://github.com/vllm-project/vllm/pull/56926) as product; HIP ISA; claimed tok/s.

## Sources

Linked inline on this page (each URL checked 2026-10-11):

- <https://github.com/vllm-project/vllm/pull/57807>
- <https://github.com/vllm-project/vllm/pull/55507>
- <https://github.com/vllm-project/vllm/pull/57908>
- <https://github.com/vllm-project/vllm/pull/57182>
- <https://github.com/vllm-project/vllm/pull/58368>
- <https://github.com/vllm-project/vllm/pull/49371>
- <https://github.com/vllm-project/vllm/pull/58288>
- <https://github.com/vllm-project/vllm/pull/58434>
- <https://github.com/vllm-project/vllm/pull/58489>
- <https://github.com/vllm-project/vllm/pull/57497>
- <https://github.com/vllm-project/vllm/pull/57775>
- <https://github.com/vllm-project/vllm/pull/55528>
- <https://github.com/sgl-project/sglang/pull/40907>
- <https://github.com/sgl-project/sglang/pull/40680>
- <https://github.com/sgl-project/sglang/pull/41092>
- <https://github.com/sgl-project/sglang/pull/41179>
- <https://github.com/sgl-project/sglang/pull/40960>
- <https://github.com/sgl-project/sglang/pull/40456>
- <https://github.com/sgl-project/sglang/pull/41062>
- <https://github.com/sgl-project/sglang/pull/41194>
- <https://github.com/sgl-project/sglang/pull/41196>
- <https://github.com/sgl-project/sglang/pull/40988>
- <https://github.com/sgl-project/sglang/pull/41091>
- <https://github.com/vllm-project/vllm/pull/58642>
- <https://github.com/vllm-project/vllm/pull/58676>
- <https://github.com/sgl-project/sglang/pull/41218>
- <https://github.com/sgl-project/sglang/pull/41171>
- <https://github.com/sgl-project/sglang/pull/41173>
- <https://github.com/sgl-project/sglang/pull/41182>
- <https://github.com/ggml-org/llama.cpp/pull/29353>
- <https://github.com/vllm-project/vllm/pull/58635>
- <https://github.com/sgl-project/sglang/pull/32269>
- <https://github.com/sgl-project/sglang/pull/41201>
- <https://github.com/sgl-project/sglang/pull/41023>
- <https://github.com/sgl-project/sglang/pull/41103>
- <https://github.com/vllm-project/vllm/pull/58472>
- <https://github.com/turboderp-org/exllamav3/pull/391>
- <https://github.com/vllm-project/vllm/pull/56926>

