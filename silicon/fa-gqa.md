# FA GQA (gfx1030)

## extras lock 2026-09-28 — tip `e0112c55` (PR #29 split decode + row gate)

Dest: `opengfx1030/vllm-rdna` `rdna_extras` (`6ed39093`→`e0112c55`, +4 incl. merge of PR #29). Occupancy first. No tok/s. No fabricated occupancy %.

`csrc/rocm/fa_rdna2.cu` + ops/bindings + `rdna_attn.py` dispatch:

| Surface | Lock |
|---|---|
| Decode multi-token | Device helper `fa_decode_token_seq`: with optional `cu_query_lens`, map query token → sequence and per-token causal `kv_len = seq_len - (end - 1 - tok)` (graph padding past `cu[-1]` → 0). Without it, one query token per sequence (prior contract). |
| Kernel signature | Decode kernels (D=128 `__launch_bounds__(128)`, D=256 / GQA `__launch_bounds__(256)`) take `cu_query_lens` + `num_seqs`; `block_table` / `seq_lens` indexed by **sequence**, not token. |
| Host ops | `fa_rdna2_decode_paged(..., out, cu_query_lens=None)`; fp8/int8 decode paths pass `nullptr, 0` (no multi-token yet). |
| Dispatch | Replaces the per-position verify-through-decode Python loop (`83e6af80`) with one `fa_rdna2_decode_paged` launch over decode-first tokens + `decode_query_start_loc`. Mixed batches: decode slice then `_forward_prefill` on the rest. |
| Row gate | Split decode only when `num_decode_tokens * num_heads <= 256` (`_SPLIT_DECODE_MAX_ROWS`); above that route the batch through prefill. Env `VLLM_FA_RDNA2_SPLIT_DECODE` defaults **1**. |
| Capture | With split decode + `VLLM_FA_RDNA2_VERIFY_FULL_GRAPH=1` (default **on**): `UNIFORM_BATCH`. Capture zeros `seq_lens` and `query_start_loc`. hippihx V1 single-token hook + `supports_draft_decode_metadata_update` retained. |
| Unchanged ISA | Still `__launch_bounds__(128|256)` (prefill often `(N, 1)`); `__builtin_amdgcn_fdot2`; XOR LDS swizzle; `fa_clip_kv_walk` / `fa_masked` / GQA register softmax. CMake gfx1030 list unchanged. |
| Later | One CTA per request owning all 1+k verify tokens (K/V read once) — not this tip. |

Do not invent numbers. Do not copy tok/s.

---

## extras lock 2026-09-28 — tip `ae5bedfd` (production launcher FA defaults)

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. Occupancy first. No tok/s. No fabricated occupancy %.

**No `csrc/rocm` / CMake gfx1030 / `__launch_bounds__` / `fdot2` / LDS tile delta** — production launcher script only. Kernel ISA stays the PR #28 + MTP verify-decode locks below.

| Surface | Lock |
|---|---|
| Production launcher FA | `serve_gfx1030_flashnext.sh`: `VLLM_USE_RDNA2_FA` defaults **1**; `VLLM_FA_RDNA2_GQA_DECODE` defaults **1** (overridable; `VLLM_USE_RDNA2_FA=0` falls back to Triton). Mirrors MTP launcher defaults at `83e6af80`. |
| Tree pin | Same script exports `PYTHONPATH=<script's tree>` first so the served code matches the launcher (avoids a copied venv's other-tree editable finder winning). |
| Unchanged ISA | Still `__launch_bounds__(128|256)` (prefill often `(N, 1)`); `__builtin_amdgcn_fdot2`; XOR LDS swizzle; `fa_clip_kv_walk` / `fa_masked` / GQA register softmax; MTP verify-through-decode as at `83e6af80`. |

Production as-shipped confirmation run on this launcher was still pending free GPUs at commit time — do not invent numbers. Do not copy tok/s.

---

## extras lock 2026-09-28 — tip `83e6af80` (PR #28 follow-up item 3)

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. Occupancy first. No tok/s. No fabricated occupancy %.

**No `csrc/rocm` / CMake gfx1030 / `__launch_bounds__` / `fdot2` / LDS tile delta** — Python + MTP launcher only. Kernel ISA stays the PR #28 lock below.

| Surface | Lock |
|---|---|
| MTP verify path | Uniform short causal batches (`1 < max_seqlen_q <= 8` and `sum(q_lens) == nseq * max`) run one `fa_rdna2_decode_paged` per position with `seq_len_i = S - (L-1-i)` — exact per-token causal prefix; reuses validated decode kernel (no prefill walk). |
| Capture metadata | `RdnaAttentionMetadataBuilder` `_cudagraph_support` = `UNIFORM_BATCH` (was `UNIFORM_SINGLE_TOKEN_DECODE`); `supports_draft_decode_metadata_update = True`. Capture zeros `seq_lens` only; keeps capture-shaped `query_start_loc`. |
| MTP launcher defaults | `serve_gfx1030_flashnext_mtp.sh`: `ATTN` default **fa** (was triton; `ATTN=triton` remains fallback). When fa: `VLLM_USE_RDNA2_FA=1` and `VLLM_FA_RDNA2_GQA_DECODE` defaults **1** (was opt-in off). |
| Unchanged ISA | Still `__launch_bounds__(128|256)` (prefill often `(N, 1)`); `__builtin_amdgcn_fdot2`; XOR LDS swizzle; `fa_clip_kv_walk` / `fa_masked` / GQA register softmax as at `bfd5286d`. |

QSA still blocks the fused draft path (`supports_draft_decode_metadata_update` is inert there). Do not invent numbers. Do not copy tok/s.

---

## extras lock 2026-09-28 — tip `bfd5286d` (PR #28 / `690cbf02`)

Dest: `opengfx1030/vllm-rdna` `rdna_extras`. Occupancy first. No tok/s. No fabricated occupancy %.

`fa_prefill_paged_varlen_gqa_kernel_256` + shared helpers in `csrc/rocm/fa_rdna2.cu`:

| Surface | Lock |
|---|---|
| Prefill KV walk | `fa_clip_kv_walk` clips causal/sliding-window K/V tile range per CTA; tile boundaries unchanged → bit-identical vs walk-and-mask. |
| Prefill sliding window | `fa_masked`: keep keys with `q - k < w` (was `k >= q - w`); matches decode + FA/Triton. |
| GQA prefill D=256 softmax | Online softmax in registers via BC-lane `__shfl_xor` (BC=16); one barrier fewer per tile; **no** `sM`/`sL`/`sD` LDS. Launcher smem = `sQ+sK+sV+sP` only; O stays `o_acc[RDS]` registers (prior register-O lock). |
| Decode | Whole-tile splits keep `uint4` V load on every split; keys left of sliding window skipped (`kv_lo`); combine zeros `seq_len==0` (graph padding) instead of NaN. |
| GQA decode kernel | `fa_decode_paged_splitk_gqa_kernel_256` opt-in `VLLM_FA_RDNA2_GQA_DECODE=1` (default **off**). |
| Host / persist | Ops take model softmax `scale` + caller-provided `out` (RDNA_ATTN in-place); drop staging memset/`copy_`; `rdna2_persist_empty` for fully-overwritten decode/split-K scratch (GEMM/AR `rdna2_persist_zeros` unchanged). |
| Unchanged ISA | Still `__launch_bounds__(128|256)` (prefill often `(N, 1)`); `__builtin_amdgcn_fdot2`; XOR LDS swizzle where previously present. Default ATTN remains triton; FA path opt-in `VLLM_USE_RDNA2_FA`. |

CMake gfx1030 append list: **unchanged** this tip window. `97fbae58` (hippihx V1) and `f22fee02` (HIP MoE MTP) are Python-only — no `csrc/rocm` object contract.

---

Dest: `opengfx1030/vllm-rdna` `rdna_extras` tip **`50120e13`** (was through `5c3c0c6f`; silicon delta `d1b200b1`). No tok/s.

## Prefill

- Templated multi-q-head CTA sharing K/V smem.
- **Production default = subgroup** (`HEADS_PER_CTA=2`, BR=8).
- `true` GQA variant dropped from dest (`13a3daf4`); env `VLLM_FA_RDNA2_GQA_MODE` = `subgroup|off` (subgroup default). Do not resurrect true-GQA without new soak.

### extras lock 2026-09-17 (tip `50120e13`, HIP `d1b200b1`)

`fa_prefill_paged_varlen_gqa_kernel_256` in `csrc/rocm/fa_rdna2.cu`:

| Surface | Delta |
|---|---|
| O accumulator | Left LDS (`sO`) for per-thread registers `o_acc[RDS]`. Thread `t` owns row `t/RP` and `RDS` consecutive dims (`RP = THREADS/ROWS`, `RDS = 256/RP`). Online-softmax rescale is register-only (no full sO sweep). |
| P·V | Hoists `sP[row*BC+k]` out of the dim loop; V strip via `uint4` (16 B) loads into FMA on `o_acc`. |
| S = Q·Kᵀ | Same `acc0`/`acc1` `fdot2` partition; loads widened to `uint4` / 8-half. fp16 accumulation order unchanged. |
| Launcher smem | Drops dead `sO` term; `sM`/`sL`/`sD` counted as `* 3` (was under-counted by one row-set). |

Still fdot2 / half2. No `__launch_bounds__` change on this kernel. LDS pressure falls by the old `HEADS*BR*256` float O tile — occupancy-relevant for this GQA prefill shape, but the broader FA prefill `(N, 1)` leftover on other kernels is **not** closed. Do not invent numbers. Do not copy tok/s.

## Decode

- `fa_decode_paged_splitk_gqa_kernel_256` with `GQA_MAX_G=8`, `__launch_bounds__(256)`, LDS K/V tiles + fdot2 QK.

## Occupancy

Prefill kernels still carry `__launch_bounds__(128|256, 1)` — **occupancy still first**. Register-O is an LDS shrink on this GQA kernel only, not the occupancy-card close. Same leftover as [fa-occupancy.md](fa-occupancy.md).
