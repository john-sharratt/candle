# DeepSeek-V4-Flash Support — Design Document

**Branch:** `deepseek-flash`
**Goal:** run DeepSeek-V4-Flash-0731 (284B total / 13B activated, MXFP4 experts) on the
dev machine (RTX PRO 5000 Blackwell 72 GB, 189 GB RAM, NVMe D:) through our batched
inference engine, ported test-first from the `quantized_qwen3_moe` implementation.

**Primary references**
- Model card: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731>
- Reference implementation: `inference/{model.py, kernel.py, convert.py, generate.py, config.json}`
  in the model repo (studied in full; semantics reproduced below)
- Technical report: "DeepSeek-V4: Towards Highly Efficient Million-Token Context
  Intelligence", <https://arxiv.org/abs/2606.19348>
- GGUF arch reference: llama.cpp `deepseek4` (merged PR
  <https://github.com/ggml-org/llama.cpp/pull/24162>, release ≥ b10173)
- GGUF artifact: <https://huggingface.co/bartowski/DeepSeek-V4-Flash-0731-GGUF> (MXFP4, 156.38 GB)

---

## 0. Implementation status

**Current state.** DeepSeek-V4-Flash runs through the batched engine. The model is
`candle-transformers/src/models/deepseek4.rs` (`impl Arch for DeepSeekV4`: config defaults, GGUF
metadata keys, tensor names, latent geometry, and the `dflash` drafter arch), the engine family is
`latent_moe/` (layers, paged-latent kernel path, provenance gallery, wave engine; AGENTS.md "Sparse-Latent
MoE Engine"). `latent_moe::Engine::load` builds the resident engine from the offline-prepared
`MXFP4_KO` GGUF (the routed experts as the exact 4-bit int8 twin, §3.2; non-expert weights resident
as int8-KO), and `latent_moe::BatchedEngine` (`latent_moe/wave/`) wraps it as a `ManagedBatchedModel`
that the scheduler binds for `ModelArch::DeepSeekV4`
(`candle-conversation/src/models/builder/load_arch.rs`). Attention runs on the paged-latent kernels
(`candle-kernels/src/paged-latent/`) over arena KV; the compressed corpus lives in a `FloatGallery`;
the MoE runs through the live dispatch (`docs/moe_live_dispatch_design.md`); DSpark speculative decode
is built (`deepseek_v4_speculative_decode.md`). The end-to-end gate is
`candle-conversation/tests/deepseek_rung4.rs` (Paris through the conversation engine, persist → reboot).
The remaining sections keep their first-pass structure; §5-§7 carry status lines where the build
departed from the plan. The first-pass status follows as a record.

### First pass (2026-08-05): the block-streaming reference forward

> **THE FULL MODEL RUNS ON REAL WEIGHTS (2026-08-05).** The complete 43-layer, 284B
> DeepSeek-V4-Flash-0731 executes end-to-end on the real merged GGUF and produces sane logits
> (`full_model_streaming_forward`: logits [1,6,129280], max|logit|≈38, ~107 s/forward). Every
> novel component is validated on **real weights**, not just tiny-config: MXFP4 experts, CSA
> attention, the sqrtsoftplus/noaux MoE with 256 experts, and the mHC hyper-connection block
> (`real_attention_layer_runs`, `real_moe_layer_runs`, `real_block_runs`). This is the
> **correctness-complete** path via a **block-streaming** reference forward (one ~3.4 GB block
> resident at a time — so 145 GB never needs to fit in VRAM), using `QMatMul` dequant→matmul
> for MXFP4 experts (no native FP4 MMA on sm_120, §6.1).
>
> **Generation quality (`full_model_generate_coherent`):** greedy-decoding "The capital of
> France is" yields on-topic, grammatical English (" the capital of …") — the model attends to
> context and predicts sensible tokens, confirming the forward is numerically *close to*
> correct — but it degenerates into repetition rather than answering "Paris". That points to
> either the missing BOS/chat template (an instruct model prompted as a raw base LM) or a
> residual quality bug that a tiny-config parity test can't catch (candidates: the grouped
> 8-way output projection, the mHC combine across 43 layers, or a scale). Closing the quality
> gap needs an activation-level comparison against the reference — the next debugging pass. The
> engine forward is proven; full generation quality is not yet nailed.
>
> **Enabling changes landed (all tested):** the single-file GGUF merger (`merge_split_ggufs`),
> `moe_bucketize` 128→256 experts, and the `QMatMul` MXFP4 dequant→matmul path
> (`cuda_mxfp4_qmatmul_dequant_path`). Performance is I/O-bound in this reference (it reloads
> all experts per forward); the batched/wave engine with the resident `ExpertCache` (§6.4) is
> the remaining work for speed.

> **Course correction that unblocked this (2026-08-05).** The first pass wrongly reported a
> "streaming wall". The 147 GB of MXFP4 experts fit in VRAM (72 GB) **+ host RAM (189 GB)**;
> block-streaming proved it directly. The reference math ports into the engine's
> attention/layer/MoE hooks for the perf path.

Built and tested (branch `deepseek-flash`), bottom-up and TDD:

- **MXFP4 codec — CPU + CUDA, bit-exact.** `GgmlDType::MXFP4` (block `{e:u8, qs:[u8;16]}`,
  17 B/32 elems, GGUF file code 39) with quantize/dequantize matching ggml exactly (E2M1
  integer table × E8M0-half scale); raw-byte unit tests in `k_quants.rs`. A standalone
  CUDA dequant kernel (`run_dequantize_mxfp4`) decodes to F32/F16/BF16, validated bit-exact
  vs the CPU codec (`cuda_mxfp4_dequant_matches_cpu`) — kept off the locked `QTYPE_COUNT`
  tables. GGUF load path reads MXFP4 (+ ggml's standard int codes 24–28 for the I32
  `tid2eid`).
- **Full reference forward** — `latent_moe/`, one concern per file, every
  module parity-tested against a scalar transcription of `model.py`: `rope` (YaRN +
  inverse), `hyper` (mHC + Sinkhorn), `compressor` (gated overlap pooling), `indexer`
  (CSA scoring + top-k), `attention` (latent single-KV + sink softmax over
  window‖compressed keys), `moe` (sqrtsoftplus/noaux_tc + hash routing + clamped SwiGLU),
  `transformer` (mHC-wrapped blocks + generate), `streaming` (`StreamingModel`, the block-streaming forward). Tiny-config end-to-end forward+generate
  runs deterministically. The reference recomputes over the full prefix each step
  (prefill path) — exact for complete compressed groups, no incremental KV state.
- **Metadata loader + real-weight validation** — `config_from_gguf` reads the real
  `deepseek4.*` metadata (verified, §4). A real MXFP4 expert tensor decodes to sane values on
  the GPU (`real_mxfp4_expert_decodes_on_gpu`). *(The multi-split `GgufModel` file handling is
  superseded — the model will load from one merged GGUF via the engine's single-mmap path, §6.5.)*

**What the first pass got wrong (now corrected in the plan):**
- Treated the expert set as un-streamable. It isn't — the engine streams it (§5.1).
- Built a standalone dense reference instead of integrating into the engine. The math is
  reused, but the container was wrong (§6.4).
- Proposed dequant-then-BF16-matmul for experts. A native FP4 tensor-core GEMM was the first
  replacement; it is unavailable on sm_120 (§6.1), so the experts run as the `MXFP4_KO` int8 twin on the
  int8 grouped GEMM, which keeps the RAM footprint at 147 GB instead of ballooning (§3.3, §6.1).

**What the first pass left, and where it landed:** the single merged GGUF (`merge_split_ggufs`,
`candle-core/src/quantized/gguf_file.rs`, then the offline `prepare_ko_gguf`,
`candle-core/src/quantized/prepare.rs`); the expert GEMM (no FP4 tensor-core GEMM — the experts run
as the `MXFP4_KO` int8 twin on the int8 grouped GEMM, §6.1); the engine seams (a `ManagedBatchedModel`
implemented directly, not the defaulted trait methods of §6.4); the MoE kernel deltas (§6.3); the
batched-forwarding gate (the rung-4 and wave gates above). See §6–§7.

---

## 1. Model overview

| Hyperparameter | Value | Notes |
|---|---|---|
| `model_type` | `deepseek_v4` | GGUF arch string: `deepseek4` |
| Total / activated params | 284B / 13B | 304B on HF including the DSpark (MTP) module |
| `num_hidden_layers` | 43 | + 3 DSpark draft layers (`n_mtp_layers: 3`) |
| `hidden_size` (`dim`) | 4096 | |
| `vocab_size` | 129280 | |
| `num_attention_heads` | 64 | query heads |
| `num_key_value_heads` | **1** | single shared KV vector per token (MQA-latent) |
| `head_dim` | **512** | of which the last `qk_rope_head_dim = 64` dims carry RoPE |
| `q_lora_rank` | 1024 | low-rank Q: `wq_a` 4096→1024, `wq_b` 1024→64·512 |
| `o_lora_rank` / `o_groups` | 1024 / 8 | grouped low-rank output projection |
| `sliding_window` | 128 | every layer keeps a 128-slot ring cache |
| `compress_ratios` | `[0,0,(4,128)×20,4,0,0,0]` | per-layer: 0 = SWA-only, 4 = CSA, 128 = HCA |
| `index_n_heads` / `index_head_dim` / `index_topk` | 64 / 128 / 512 | CSA lightning indexer |
| `n_routed_experts` / `num_experts_per_tok` / `n_shared_experts` | **256** / 6 / 1 | every layer is MoE |
| `moe_intermediate_size` | 2048 | per-expert FFN width |
| `num_hash_layers` | 3 | layers 0–2 route by token id (`tid2eid` table) |
| `scoring_func` / `topk_method` | `sqrtsoftplus` / `noaux_tc` | `routed_scaling_factor = 1.5`, `norm_topk_prob = true` |
| `swiglu_limit` | 10.0 | gpt-oss-style clamp in every expert |
| `hc_mult` / `hc_sinkhorn_iters` | 4 / 20 | Manifold-Constrained Hyper-Connections (mHC) |
| `max_position_embeddings` | 1,048,576 | YaRN ×16 from `original_seq_len = 65536` |
| `rope_theta` / `compress_rope_theta` | 10000 / 160000 | SWA layers use 10000 without YaRN |
| Weights (dense) | FP8 E4M3, 128×128 block scales, `ue8m0` scale format | |
| Weights (routed experts) | **FP4 (E2M1), per-32 E8M0 scales = MXFP4** | FP4 QAT — this is the native trained format |
| KV cache | FP8 E4M3 (non-RoPE dims, per-64 blocks), BF16 RoPE dims | `--kv-cache-dtype fp8` in vLLM |
| DSpark | `dspark_block_size 5`, target layers `[40,41,42]`, markov rank 256 | speculative decoding module |

Layer schedule (indices into `compress_ratios`): L0–L1 SWA-only; L2–L42 alternate
CSA (ratio 4, odd positions starting L2) and HCA (ratio 128) — 21 CSA + 20 HCA
layers; the 3 DSpark layers are SWA-only. Params check: 43 layers × 256 experts
× 3 × 4096 × 2048 ≈ 277B routed + ~6.5B dense/attention + ~1.1B embed/head ≈ 284B ✓.

### Why this model fits our engine

The architecture is *natively unbounded-context*: at 1M tokens the entire KV state is
≈ 3.5 GB/session (§3.2), the per-token attention cost is bounded (window 128 + top-512
compressed + T/128 HCA keys), and the model does its own learned retrieval (the
indexer) — a hardware-native cousin of our provenance-selected attention. The FP4 QAT
experts (147 GB) stream from RAM/NVMe through the existing `ExpertCache`, and the
hash-routed layers + DSpark draft blocks give *perfect* expert prefetch information —
a direct amplifier for our Markov expert prediction machinery.

---

## 2. Architecture deep-dive (reference semantics)

Everything in this section was extracted from the official `inference/model.py` +
`kernel.py` and must be treated as the ground truth for the port.

### 2.1 Attention (shared by all layer kinds)

Single latent KV vector per token; 64 query heads read it directly (no per-head K/V
expansion — unlike `deepseek2.rs` MLA, there is **no** `kv_b_proj`):

```
qr = rms_norm(wq_a(x))                       # 4096 -> 1024, q_norm
q  = wq_b(qr).unflatten(64, 512)             # 1024 -> 64*512
q *= rsqrt(mean(q², dim=-1) + eps)           # per-head unweighted RMS normalization
rope(q[..., -64:])                           # rope on last 64 dims only

kv = rms_norm(wkv(x))                        # 4096 -> 512 (ONE vector, k and v identical)
rope(kv[..., -64:])
fp8_fake_quant(kv[..., :-64], block=64)      # QAT-matched: non-rope dims through E4M3

o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, 512**-0.5)
rope_inverse(o[..., -64:])                   # de-rotate: values carry rotated dims
o = o.view(bsz, seq, 8, 8*512)               # 8 groups of 8 heads
o = einsum(o, wo_a.view(8, 1024, 4096))      # per-group low-rank: 4096 -> 1024
x = wo_b(o.flatten(2))                       # 8*1024 -> 4096
```

Key facts:
- **K and V are the same 512-dim vector.** `sparse_attn` computes scores `q·kvᵀ` and
  output `p·kv` against the *same* rows. Softmax scale `512**-0.5`.
- **Learned attention sinks**: per-head scalar `attn_sink` added to the softmax
  denominator as `exp(sink − max)` — exactly the gpt-oss mechanism.
- **Output de-rotation** (`apply_rotary_emb(o[..., -64:], freqs, inverse=True)`): the
  value rows carry rotated RoPE dims, so the output's rope-span must be counter-rotated
  by the *query* position. Attention output for the rope dims therefore encodes
  relative position — this is load-bearing, not optional.
- `wo_a` is stored FP8 in the checkpoint but the reference computes it in BF16.
- Selection, not masking: attention is over an explicit `topk_idxs` gather list
  (`-1` = padded/invalid slot → treated as −inf score). Per layer the list is
  `[window idxs (≤128)] ++ [compressed idxs]`.

### 2.2 The three cache streams per layer

1. **Window ring** (all layers): the raw per-token `kv` goes into a 128-slot ring
   (`kv_cache[pos % 128]`). Window gather indices are ring positions ordered
   oldest→newest.
2. **Compressed cache** (CSA and HCA layers): a learned `Compressor` pools each
   consecutive group of `ratio` tokens into ONE 512-dim compressed KV entry, appended
   at `pos // ratio` (offset after the 128 window slots in the same buffer).
3. **Indexer cache** (CSA layers only): a second, independent `Compressor`
   (`head_dim = 128`, with Hadamard rotation) producing FP4-fake-quantized entries the
   indexer scores against.

Compressor (`ratio r`, gated pooling, computed in FP32):

```
kv_r    = wkv(x)      # 4096 -> coff*512  (coff = 2 when r == 4 [overlap], else 1)
score_r = wgate(x)    # 4096 -> coff*512
score  += ape[pos % r]                    # learned intra-group position embedding
entry   = Σ_group softmax(score) * kv_r   # per-dim gated average over the r tokens
entry   = rms_norm(entry); rope(entry[..., -64:], freqs[group_start])
fp8_fake_quant(entry[..., :-64], block=64)          # attention compressor
# indexer compressor instead: hadamard_rotate(entry); fp4_fake_quant(entry, block=32)
```

- **Overlap mode (r = 4 only):** each entry pools 8 tokens — the current group of 4
  (second half of the projection dims) plus the previous group of 4 (first half),
  softmaxed jointly over the 8 rows. HCA's r = 128 is non-overlapping.
- **Decode incremental state:** per session, per compressor: `kv_state`/`score_state`
  rings of `coff·r` rows; an entry is emitted every `r`-th token
  (`(pos+1) % r == 0`). Prefill computes all groups batch-wise; the remainder tokens
  (< r) seed the state. Prefill and incremental decode must produce bit-identical
  entries — this is a mandatory unit test.
- Compressed entries use the *group-start* position's frequencies for RoPE
  (`freqs_cis[pos + 1 - r]`; prefill: `freqs_cis[0:cutoff:r]`).

### 2.3 CSA — Compressed Sparse Attention (ratio 4, 21 layers)

The indexer selects which compressed entries a query may attend to:

```
qi = wq_b_idx(qr).unflatten(64, 128)       # from the SHARED q_lora qr, 64 idx heads
rope(qi[..., -64:]); qi = hadamard_rotate(qi); fp4_fake_quant(qi, 32)
w  = weights_proj(x) * (128**-0.5 * 64**-0.5)      # per-head gate, bf16
score[t] = Σ_h w_h · relu(qi_h · idx_cache[t])      # over all T/4 compressed entries
topk_idxs = topk(score, 512)                        # + causality mask in prefill
```

Per-token CSA attention set = 128 window + ≤512 compressed entries (each summarizing
4 tokens → ~2048-token effective reach chosen from the *whole* context). Both q and
the indexer cache are FP4 (QAT'd) — on Blackwell this scoring maps directly onto FP4
tensor cores (vLLM: `use_fp4_indexer_cache`); the flat scan is the model's built-in
analogue of our binary directional provenance scan.

### 2.4 HCA — Heavily Compressed Attention (ratio 128, 20 layers)

No indexer: attends to **all** compressed entries (T/128 of them; 7,813 at 1M
context) plus the window. Gather list is `arange(pos // 128) + offset`.

### 2.5 mHC — Manifold-Constrained Hyper-Connections

The residual stream is **4 copies** of the hidden state: `h : [B, S, 4, 4096]`
(embedding output repeated ×4). Around each sub-block (attention and FFN):

```
mixes = linear(flatten(h), hc_fn) * rsqrt(mean(flatten(h)², -1) + eps)   # -> 24 dims
pre  = sigmoid(mixes[:4]  * s0 + base[:4]) + eps          # 4 weights
post = 2*sigmoid(mixes[4:8] * s1 + base[4:8])             # 4 weights
comb = sinkhorn(softmax(mixes[8:24] * s2 + base[8:24]).view(4,4), 20 iters)
x_in  = Σ_c pre[c] * h[c]                                  # 4 -> 1 (then RMSNorm -> block)
h_out = post ⊗ block_out + comb @ h                        # 1 -> 4 + doubly-stochastic remix
```

All in FP32 (`mix_hc = (2+4)·4 = 24`; params: `hc_{attn,ffn}_{fn,base,scale}` per layer
+ `hc_head_{fn,base,scale}` before the LM head, where a sigmoid-weighted 4→1
reduction feeds `output_norm` → `lm_head`). Sinkhorn = alternating row/column
normalization of the 4×4 matrix, 20 iterations, `eps 1e-6`.
Cost is trivial (4×4 per token) but it changes the *shape of the residual stream*,
which is the main reason DSv4 cannot reuse `forward_layer_batched_mixed` as-is (§6.4).

### 2.6 MoE

Every layer: 256 routed experts (top-6) + 1 shared expert, `moe_inter = 2048`.

- **Scoring:** `scores = sqrt(softplus(gate(x_f32)))`.
- **Selection:** `noaux_tc` — a learned per-expert bias (`e_score_correction_bias`) is
  added *for top-k selection only*; routing weights use the unbiased scores.
- **Normalization:** selected weights are renormalized to sum 1, then scaled by
  `routed_scaling_factor = 1.5`.
- **Hash routing (layers 0–2):** expert indices come from a `tid2eid : [vocab, 6]`
  i32 lookup by **token id** — no top-k. Weights are still gathered from the scores at
  those indices (gate weights exist on hash layers too). Requires token ids to be
  plumbed into the layer forward. Corollary: expert needs for layers 0–2 are known the
  moment a token is sampled → zero-latency prefetch.
- **Expert FFN:** SwiGLU with clamp: `up = clamp(up, ±10)`, `gate = clamp(gate, max=10)`
  (one-sided!), `silu(gate) * up`, computed in FP32 in the reference, routing weight
  applied *before* `w2` (inside the FP32 section).
- **Shared expert:** always-on, same shape, BF16/FP8 (not FP4).

### 2.7 RoPE / YaRN

Standard YaRN (`factor 16`, `beta_fast 32`, `beta_slow 1`, `original 65536`,
non-interleaved pairs over the 64 rope dims, complex-multiply formulation) — but **only
on compression layers**, with `compress_rope_theta = 160000`. SWA-only layers (L0, L1,
DSpark) use plain `rope_theta = 10000` with **no** YaRN (a 128-token window never
extrapolates). No mscale/attention-scale correction is applied (unlike deepseek2).

### 2.8 DSpark speculative decoding (later phase)

3 draft layers (`mtp.*` namespace, SWA-only attention, own MoE) predict a block of 5
tokens per accepted token:

- Inputs: concat of main-model hidden states from layers 40–42 (`main_proj` 3·4096→4096
  + norm) and a draft block `[sampled_token, noise ×4]` (`dspark_noise_token_id`).
- The final draft layer samples token-by-token through the LM head **plus a Markov
  bias head** (`markov_w1/w2`, rank 256 — a learned bigram model over the vocab) and
  emits a per-position `confidence` score (small head over `[hidden ‖ markov_embed]`).
- vLLM runs it with `num_speculative_tokens: 7`.

Synergy note: a confident 5-token draft block gives the expert-streaming layer the
complete routed-expert set for the whole block *before* the verify forward — combined
with hash layers this converts expert prefetch from prediction into lookup. The drafter is
built as a separate `dflash` GGUF (`deepseek_v4_speculative_decode.md`), because community GGUFs
strip the `mtp.*` tensors from the main file (bartowski: "MTP: no").

### 2.9 Chat encoding

No Jinja template; the repo ships an `encoding/` folder. Format essentials:
`<｜begin▁of▁sentence｜>{system}`, turns as `<｜User｜>…<｜Assistant｜>…<｜end▁of▁sentence｜>`,
reasoning wrapped in `<think>…</think>` (dropped from history unless tools active),
`reasoning_effort ∈ {low, high, max}` implemented as a plain-text prefix before the
system message, tools via DSML markup. We need a `Dialect` for this in `batch_test`
(the harness currently uses `Dialect::chat_ml()`); tokenizer.json comes from the HF
repo as usual.

---

## 3. Quantization & memory

### 3.1 Weight formats in the GGUF

Chosen artifact: **bartowski/DeepSeek-V4-Flash-0731-GGUF, MXFP4, 156.38 GB** (split
files), converted at llama.cpp b10173. This is the only quant that preserves the FP4
QAT expert weights ("Original quality") — routed experts as `MXFP4`, dense/attention/
shared-expert tensors as Q8_0/BF16 (verified against the file, §4.2).

GGML MXFP4 (type id **39**): 32 elements/block, 17 bytes:
`{ uint8 e; uint8 qs[16] }` — `e` is an E8M0 power-of-two scale, nibbles index the
E2M1 value table `{0, .5, 1, 1.5, 2, 3, 4, 6}` × sign (ggml stores the ×2 integer
table and folds ×0.5 into the scale). Exact layout to be locked in with raw-byte
tests against `ggml_quantize`/reference vectors (per AGENTS.md: byte assertions, no
tolerance thresholds). NVFP4 (type id 40) exists in ggml but is not needed for this
model; note the id so the loader errors informatively.

Not chosen: unsloth UD-* quants (82–162 GB) mostly use IQ formats our GGUF code table
does not map (`from_gguf_file_code` bails on stock ids ≥ 16 except k-quants/BF16);
they also re-quantize the QAT FP4 experts, sacrificing the "trained format" property.

### 3.2 Runtime memory budget (dev machine: 72 GB VRAM, 189 GB RAM, NVMe)

_Status: the sizing below is the first-pass (two-tier) arithmetic. The engine's `ExpertCache` is now
three-tier (VRAM hot pool → pinned host RAM → the NVMe pack, `docs/expert_cache_design.md`), and the
routed experts are served as the `MXFP4_KO` int8 twin — the exact 4-bit byte-reorder of the MXFP4 expert
(`latent_moe/engine.rs`, `candle-core/src/quantized/ko_quant/mxfp4_ko.rs`), so the 147 GB does not
grow._ The engine's `ExpertCache` was, in the first pass, a **two-tier** pool (see §5.1): a fixed VRAM slot pool (the hot
active set) plus a **pinned host-RAM** overflow pool. So the sizing constraint is **experts
fit in VRAM + host RAM**, which they do with margin:

| What | Size | Placement |
|---|---|---|
| Routed experts (43×256, MXFP4, 4-bit) | ≈ 147 GB | hot set in VRAM slots + remainder in **pinned host RAM**; startup-repacked from the merged GGUF mmap |
| Dense + shared experts + embed/head (Q8_0/BF16) | ≈ 8–9 GB | VRAM, always resident |
| Expert hot pool (`num_slots`) | budget ≈ 40–50 GB | VRAM (≈ 30–35 % of experts); sized by the VRAM governor |
| KV state @ 1M ctx (per session) | ≈ 3.5 GB | VRAM: window 43×128×512 ≈ 3 MB; CSA compressed 21 × T/4 × (448 FP8 + 7 scale + 64 BF16 ≈ 583 B) ≈ 3.1 GB; HCA 20 × T/128 ≈ 90 MB; indexer FP4 21 × T/4 × 68 B ≈ 360 MB |
| Activations/workspaces | governed by existing VRAM reserve | VRAM |

_KV row, as built:_ the window is the two-region latent (nope `[0,448)` FP8 E4M3 at unit scale, the 64 rope
dims BF16: 576 B/token/layer); the compressed corpus is the position-free int8+BF16 two-region cache
(448 B int8 + 14 f32 band scales + 128 B BF16 per entry, `deepseek_int8_hot_corpus.md`), not 583 B of
FP8 per-64 blocks; the Indexer keys are f32 `[G,128]` (not FP4 nibbles), spilling with the corpus cache to
host RAM past `HOT_ENTRY_CAP` while the sign index stays on the GPU. The 3.5 GB figure is the first-pass
FP8/FP4 target, not the built footprint.

**Fits:** 147 GB experts + ~9 GB dense ≈ 156 GB total ≤ 72 GB VRAM + 189 GB RAM = **261 GB**.
Hot set (~40–50 GB) in VRAM, ~100 GB in pinned RAM. Crucially, keeping experts **4-bit**
(MXFP4 payload, repacked only for tensor-core *layout*, not width — §3.3/§6.1) is what makes
this fit: requantizing to the Qwen3 int8 KO twins (8-bit, `Q6_KO`/`Q8_KO`) would balloon 147 GB → ~280 GB and
overflow RAM. DeepSeek uses `MXFP4_KO`, the exponent-collapse int8 twin that is an exact 4-bit
byte-reorder of the MXFP4 expert (no requant), read directly by the int8 grouped GEMM.

Streaming ceiling: 6 experts × 43 layers × 13.4 MB (MXFP4 expert) ≈ 3.4 GB/token if every
expert missed and had to stream pinned→VRAM — PCIe 5 x16 (~63 GB/s) caps that at ~18 t/s.
Residency (~30 %), routing locality, Markov prefetch, hash-layer lookup, and cross-session
wave batching multiply the effective rate; DSpark multiplies it again by the accepted block
length. Numbers to be measured. (If a *smaller* machine ever needs experts > VRAM+RAM, add an
NVMe tier below pinned — the non-CUDA `load_from_mmap` miss path is the template; not needed
here.)

### 3.3 Activation quantization: what we replicate, what we don't

The reference fake-quantizes (quant→dequant, storage stays BF16) at specific points to
match QAT training. Two different roles:

1. **Cache contents** (KV non-rope dims per-64 FP8; compressed entries per-64 FP8;
   indexer entries + indexer q per-32 FP4 after Hadamard): these define the *values
   attention reads* — we implement them exactly, but store the **real** quantized
   bytes instead of fake-quant BF16 (E4M3 × power-of-two ue8m0 scale re-expands
   bit-exactly into BF16, so real storage ≡ reference fake-quant; that's the entire
   FP8-KV story and it is why FP8-first is the right call).
2. **GEMM input quantization** (`act_quant` per-128 FP8 before every FP8/FP4 linear):
   affects only matmul arithmetic precision. The production expert GEMM runs the `MXFP4_KO`
   lane-major twin on the **int8 tensor-core grouped GEMM**: activations are quantized per
   128-K (q8a128), each 32-K sub of the weight is one int32 MMA whose result is folded with
   that sub's own E8M0 scale in floating point (§6.1). Native FP4 MMA is unsupported on sm_120,
   so this is the path (not an FP8/FP4 MMA). Weights stay 4-bit in memory; the MXFP4→int8-twin
   repack is an exact byte permutation, and the codec is validated against the CPU MXFP4
   dequant oracle (`cuda_mxfp4_dequant_matches_cpu`). The MoE is memory-bound on the expert
   stream, so this is not meaningfully slower than a native FP4 path would be here.

Hadamard rotation (`fast_hadamard_transform`, scale `d^-1/2`, d = 128) is required on
the indexer path (q and cache symmetrically — scores are rotation-invariant only if
both sides rotate). Size-128 Hadamard is a 7-stage butterfly; trivial CUDA kernel.

---

## 4. GGUF layout (llama.cpp `deepseek4`)

### 4.1 Metadata keys (verified against the real GGUF, `{arch}` = `deepseek4`)

```
deepseek4.block_count (43), .embedding_length (4096)
deepseek4.expert_count (256), .expert_used_count (6), .expert_shared_count (1)
deepseek4.expert_feed_forward_length (2048)
deepseek4.expert_weights_scale (1.5 = routed_scaling_factor)
deepseek4.expert_weights_norm (true), .expert_gating_func (4 = sqrtsoftplus)
deepseek4.hash_layer_count (3)
deepseek4.swiglu_clamp_exp / .swiglu_clamp_shexp   ← PER-LAYER F32 arrays (all 10.0); take [0]
deepseek4.attention.head_count (64), .head_count_kv (1), .key_length (512), .value_length (512)
deepseek4.attention.q_lora_rank (1024)   ← no kv_lora_rank; wkv is direct 4096→512
deepseek4.attention.output_group_count (8)   ← o_groups
deepseek4.attention.output_lora_rank (1024)  ← o_lora_rank
deepseek4.attention.sliding_window (128)
deepseek4.attention.compress_ratios          ← per-layer i32 array (len 46; first 43 used)
deepseek4.attention.compress_rope_freq_base (160000)
deepseek4.attention.indexer.head_count (64) / .key_length (128) / .top_k (512)
deepseek4.hyper_connection.count (4) / .sinkhorn_iterations (20) / .epsilon (1e-6)
deepseek4.rope.freq_base (10000), .dimension_count (64 = rope_head_dim)
deepseek4.context_length (1048576)
```

**Corrections from the download:** `o_groups`/`o_lora_rank` use the `output_group_count`/
`output_lora_rank` keys (not `o_*`); `rope_head_dim` is `rope.dimension_count` (not an
attention key); `swiglu_clamp_exp` is a per-layer array. **YaRN parameters
(`factor 16`, `beta_fast 32`, `beta_slow 1`, `original_max_position 65536`) are NOT stored**
— llama.cpp's `deepseek4` arch bakes them in — so the loader hardcodes them. No
`nextn_predict_layers` (MTP tensors stripped by bartowski).

### 4.2 Tensor names (verbatim)

Global: `token_embd.weight`, `output_norm.weight`, `output.weight`,
`output_hc_fn`, `output_hc_base`, `output_hc_scale`.

Per layer `blk.{i}.`:

| GGUF name | Reference param | Shape (logical) |
|---|---|---|
| `attn_norm.weight` | `attn_norm` | 4096 |
| `attn_sinks` | `attn_sink` | 64 |
| `attn_q_a.weight` | `wq_a` | 1024×4096 |
| `attn_q_a_norm.weight` | `q_norm` | 1024 |
| `attn_q_b.weight` | `wq_b` | 32768×1024 |
| `attn_kv.weight` | `wkv` | 512×4096 |
| `attn_kv_a_norm.weight` | `kv_norm` | 512 |
| `attn_output_a.weight` | `wo_a` | 8192×4096 (8 groups × 1024, per-group in 4096) |
| `attn_output_b.weight` | `wo_b` | 4096×8192 |
| `hc_attn_fn` / `hc_attn_base` / `hc_attn_scale` | `hc_attn_*` | 24×16384 / 24 / 3 |
| `hc_ffn_fn` / `hc_ffn_base` / `hc_ffn_scale` | `hc_ffn_*` | 24×16384 / 24 / 3 |
| `attn_compressor_kv.weight` | compressor `wkv` | coff·512×4096 |
| `attn_compressor_gate.weight` | compressor `wgate` | coff·512×4096 |
| `attn_compressor_ape` | compressor `ape` | ratio×coff·512 |
| `attn_compressor_norm.weight` | compressor `norm` | 512 |
| `indexer.proj.weight` | indexer `weights_proj` | 64×4096 |
| `indexer.attn_q_b.weight` | indexer `wq_b` | 8192×1024 |
| `indexer_compressor_kv.weight` / `_gate.weight` / `_ape` / `_norm.weight` | indexer compressor | 2·128-based, head_dim 128 |
| `ffn_norm.weight` | `ffn_norm` | 4096 |
| `ffn_gate_inp.weight` | `gate.weight` (router) | 256×4096 |
| `ffn_gate_tid2eid` | `gate.tid2eid` (hash layers) | 129280×6 i32 |
| `exp_probs_b.bias` (`FFN_EXP_PROBS_B`) | `gate.bias` (noaux_tc) | 256 |
| `ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight` | routed experts (3D merged) | 256×2048×4096 / 256×2048×4096 / 256×4096×2048, **MXFP4** |
| `ffn_gate_shexp.weight` / `ffn_up_shexp.weight` / `ffn_down_shexp.weight` | shared expert | 2048×4096 / 4096×2048 |
| `blk.{i}.nextn.*` (eh_proj, embed_tokens, enorm, hnorm, shared_head_*) | MTP | absent in bartowski |

Compressor/indexer tensors exist only on layers with `compress_ratios[i] > 0`
(indexer only where ratio == 4). Names/dtypes/shapes above are verified against the real
GGUF (§0); any llama.cpp pre-transposition (e.g. of `attn_output_a`) is pinned by the loader
tests at first green.

---

## 5. Current engine survey (what we build on)

The batched engine (`quantized_qwen3_moe/` as the reference model, `batched_model/` /
`batched_layer/` / `batched_inference/` / `expert_lre/` as the engine) is what DeepSeek
plugs into. The Qwen3 load + run seam is `ModelWeights::from_pack` (`quantized_qwen3_moe/loader.rs`) wrapped by
`BatchedInference::new_with_schedule` (`batched_model/inference_wrapper.rs`); DeepSeek's is
`Engine::load(path, &DEEPSEEK_V4, device)` + `BatchedEngine::new(engine)` (`load_arch.rs`).

### 5.1 The engine already streams a MoE larger than VRAM

This is the crux the first pass missed. The Qwen3 loader `mmap`s the (single) GGUF, page-locks it with
`register_mmap_cuda` (`candle-core/src/quantized/cuda/host_memory.rs`, `cuMemHostRegister ... DEVICEMAP|READ_ONLY`), and builds a **two-tier**
`ExpertCache` (`expert_lre/`):

- **Startup** (`expert_lre/startup.rs`): every `(layer, expert)` is GPU-repacked
  from the GGUF bytes into its target format; it lands in a **VRAM slot** if one is free, else
  a **pinned host-RAM slot** (`cuMemAllocHost`). After startup the mmap is unused for experts.
- **Per layer** (the first-pass `classify_and_load`; replaced by the live dispatch below): resident experts hit; misses
  `allocate_slot` (score-based LRU, the first `PINNED_LAYERS` = 2 MoE layers never evicted, `expert_lre/cache/policy.rs`), evict the victim
  VRAM→pinned, and stream the miss **pinned→VRAM** on a copy stream — one fence per batch.
  Markov `TransitionMatrix` (`expert_lre/transition.rs`) prefetch fills free slots with predicted next-layer experts.
- **VRAM governor**: a balloon measures real capacity, `expert_budget()` leaves the
  KV floor free, and `num_slots = expert_budget / max_expert_size` is the hot active set; the
  rest lives in pinned RAM.
- **Cross-session wave** (`forward_wave` / `forward_wave_contexts`): co-batches decode + prefill
  rows from many sessions into one sweep, so each MoE layer runs its expert dispatch **once per
  layer per wave** over the *union* of routed experts — the load is amortized across sessions.

> **Since then — the live MoE dispatch (`docs/moe_live_dispatch_design.md`).** The per-layer
> `classify_and_load` / `submit_moe_work` host path above is gone, and so is demand eviction.
> Every layer runs `ExpertCache::forward_routed`: `moe_bucketize` classifies each routed expert
> VRAM / pinned / cold against a live per-expert pointer table, and the grouped GEMMs run at once
> — VRAM experts in place, pinned misses copied into VRAM scratch by the expert kernel's own
> worker blocks, cold experts (on the NVMe pack only) waited on in-kernel until the stager thread
> publishes them into a pinned pad. VRAM residency changes only by promotion and eviction on the
> pipeline thread, off the critical path. The startup and governor bullets above predate the
> three-tier cache (`docs/expert_cache_design.md`).

Net: the sizing constraint is **experts fit in VRAM + host RAM + the NVMe pack**, and DeepSeek's
147 GB fit (§3.2). The MoE pipeline itself is `moe_route → moe_bucketize → gather → grouped
GEMM → silu_mul → scatter`, all on the GPU, with zero routing readback.

### 5.2 The seam map — what plugs in vs. what is model-owned

Two traits define the Qwen-lineage model↔engine contract: `BatchedModelCore` (`batched_model/model_core.rs` — leaf
accessors: layers, embed, `final_norm`, `output_proj`) and `BatchedAttentionLayer`
(`batched_layer/mod.rs`); above them sits `ManagedBatchedModel` (`batched_inference/managed_model.rs`),
the trait the scheduler binds. Verified pluggability (first-pass survey):

| Concern | Hook today | DeepSeek needs |
|---|---|---|
| Q/K/V projection | `project_qkv` | override (low-rank Q, single latent KV) |
| ln1 / ln2 | `attention_norm`/`ffn_norm` | override |
| o_proj | `output_projection`/`o_proj` | fold DSv4's grouped 8-way de-rotated o-proj here |
| MoE / FFN | `ffn_residual` → `SparseMoeBlock` + `ExpertCache` | **reuse**; new `SparseMoeBlock`-analog, 4 kernel deltas (§6.3) |
| **Attention kernel** | *hard-coded* to paged-GQA in `forward_attn_batched_single/_multi` | **new seam** — the wrongly-hardcoded bit (§6.4) |
| **Residual / layer body** | *hard-coded* `[tokens,hidden]` add in `forward_layer_batched_mixed` | **new seam** — mHC 4-copy stream (§6.4) |
| **Session KV creation** | *hard-coded* GQA `ChunkedKvBacking` | **new seam** — session-owned window/compressed/indexer KV (§6.2) |

The three "hard-coded" rows are the ones the plan fixed — see §6.4 for the *minimally-invasive*
strategy (extract each into a defaulted trait method; existing models keep the default;
DeepSeek overrides). _Built differently:_ DeepSeek implements `ManagedBatchedModel` directly
(`BatchedEngine`, `latent_moe/wave/`), so its attention kernel, mHC layer body and session KV
creation (`create_batched_session`) are owned by the wave and no seam in the Qwen layer
traits was extracted. The expert-streaming / scheduler machinery is reused.

**Quant infra status:** MXFP4 codec (CPU+GPU dequant) is **done** (§0). Native `MXFP4` has no `QType`
slot in the matmul dispatch (`dtype_to_qtype` in `candle-core/src/quantized/cuda/mod.rs` maps
`MXFP4_KO` but not `MXFP4`; `QTYPE_MXFP4` is declared with "no matmul kernel of its own"), so the float
grouped path `grouped_matmul_gemx` cannot take an MXFP4 weight. The production expert path is the
`MXFP4_KO` int8 twin on the int8 grouped GEMM (`grouped_matmul_gemx_q8a128`,
`grouped_qmatmul_dev_q8a128`; the per-sub E8M0 fold lives in
`candle-kernels/src/quantized/loader/mxfp4.cuh`, §6.1). `moe_bucketize` is
compiled `MAX_EXPERTS 512` (`candle-kernels/src/simple/moe_bucketize_common.cuh`), above the 256 routed experts. Paged
attention (`paged-decode/`, `paged-prefill/`) is GQA head_dim ∈ {64,128}, kernel-internal
RoPE — unsuited to DSv4's 512-dim single-KV gather, hence the new attention kernel family `paged-latent/`.

---

## 6. Gap analysis → what we build

### 6.1 MXFP4 weight format + FP4 tensor-core GEMM

_Status: the production expert path is `MXFP4_KO` — the MXFP4 expert repacked offline
(`prepare_ko_gguf`, an exact byte permutation: nibbles and per-sub E8M0 scales copied, no requant)
into the lane-major per-sub twin (`QTYPE_MXFP4_KO`, `candle-kernels/src/quantized/loader/mxfp4.cuh`),
read by the int8 grouped GEMM (`grouped_matmul_gemx_q8a128`, `grouped_qmatmul_dev_q8a128`) with q8a128
activations._

**Codec — DONE (§0).** `GgmlDType::MXFP4` + `BlockMXFP4 {e:u8, qs:[u8;16]}`, GGUF code 39,
CPU quantize/dequantize with raw-byte tests, standalone CUDA dequant kernel bit-exact vs CPU
(`cuda_mxfp4_dequant_matches_cpu`), GGUF load path. The plain `QMatMul` route for a native MXFP4
weight dequantizes the whole weight with that standalone kernel and multiplies in float
(`cuda_mxfp4_qmatmul_dequant_path`); it is the reference route, not the expert path.

**Expert GEMM — the int8 twin with a per-sub fold (BUILT).** The "layout, not density" idea assumed
native FP4 tensor-core MMA. **That instruction is not available on our hardware:** ptxas on sm_120
(RTX PRO 5000 Blackwell) rejects the block-scaled FP4 MMA — `Instruction 'mma with block scale' not
supported on .target 'sm_120'`, `Feature '.kind::mxf4' not supported` (llama.cpp #19662, recorded
2026-08-05 on CUDA 13.1; we build on 12.4, older still). So native MXFP4 MMA and its SF-layout repack
are **off the table** here.

The built path: each 32-K sub of an MXFP4 weight is stored as its E2M1 codebook indices expanded to int8
(`mxfp4_codebook_x4`) beside its E8M0 scale; the kernel issues one int32 MMA per sub against the q8a128
activation and folds the result with that sub's own scale `2^(e_sub-128)` in FP32 (the
`is_mxfp4_persub` branch of `kernel.cuh`). This:
- keeps experts **4-bit in memory** (147 GB pool preserved) — the codebook expansion is in-register;
- is exact in the weight (no requantization, unlike the Q8_KO twins' per-128 scale granularity);
- is not the bottleneck: the MoE is memory-bound on the pinned→VRAM expert stream.

_Replaced design: this section planned to add `block_c_mxfp4` to the float `grouped_matmul_gemx`
(`gemx_dequant.cuh`) so the weight would be dequantized per tile in registers to FP16 and fed to the FP16
tensor-core MMA, with no repack. That path does not exist: `MXFP4` has no `QType` slot in the matmul
dispatch (`dtype_to_qtype` does not map it, `QTYPE_MXFP4` carries no matmul kernel), and the expert
weights are repacked offline into `MXFP4_KO` instead, read on the int8 MMA._

(DeepSeek's own reference `fp4_gemm` corroborates avoiding native FP4: it upcasts FP4→FP8 and
runs an FP8 MMA. If a future sm_100-class board with working `kind::mxf4` is targeted, the
native path becomes the optimization; on sm_120 it is unavailable, so it is out of scope.)

### 6.2 Attention pipeline (new kernel family + cache layout)

_Status: built as `candle-kernels/src/paged-latent/` (not `sparse-attn/`): `latent_decode_kernel.cuh`,
`latent_prefill_kernel.cuh`, `latent_common.cuh`, with the Rust wrappers in `latent_moe/paged/`
(`deepseek_batched_paged_attention_plan.md` Part III). The session KV is `ChunkedKvBacking` (single-latent
mode), not flat session-owned tensors, and the window format is the two-region latent (nope FP8 at unit scale,
rope BF16). Hadamard, the Indexer and the compressor are `latent_moe/indexer.rs` and
`latent_moe/compressor/`; the Indexer cache is f32 keys, not FP4 nibbles._

New candle-kernels subdir `sparse-attn/` (as planned):
- `sparse_gather_attn`: q `[heads=64, d=512]` × gathered KV rows (`topk_idxs`,
  `-1` = skip), online softmax with per-head sink, BF16 in/out, one query block per
  CTA (decode: 1 token; prefill: tiled over positions). Mirrors the reference
  tilelang kernel including the `idx == -1 → −inf` semantics and the
  single-buffer `score/output` use of the same KV rows. FP8-aware KV reads (E4M3
  bytes + per-64 ue8m0 scale + BF16 rope tail) in the same kernel via format tag.
- Small kernels: Hadamard-128, per-64 FP8 encode (exists: `fp8_e4m3_utils.cuh`),
  per-32 FP4 encode for the indexer cache, indexer scoring
  (`relu(q·k) weighted-sum over heads` + top-k — top-512 of ≤ T/4 via existing
  radix/partial-sort or a two-pass threshold approach), compressor gated-pooling
  (prefill batch + decode incremental).

Cache state, as planned: new `Dsv4LayerKv` per session per layer — window ring (128×512),
compressed append buffer, indexer append buffer (CSA), compressor ring states, as flat
session-owned CUDA tensors with `ChunkedKvBacking` bypassed and the three-tier arenas and
substrate persistence integrated later (compressed entries append-only at 583 B each).
_Replaced design (built, see the status line above): there is no `Dsv4LayerKv`. The window ring is a
`ChunkedKvBacking` in single-latent mode from the start, so it rides the arena pool and the persistence
thread's hot→warm drain; the corpus is the per-sequence `FloatGallery` (int8+BF16 cache, f32 keys, sign
index); the compressor ring states are `IncrementalCompressor` state held by the wave._

### 6.3 MoE deltas

_Status: all four are in (the router in `latent_moe/moe.rs`, `Gate::route`; `moe_bucketize` now compiles
`MAX_EXPERTS 512`)._

- `moe_bucketize.cu`: `MAX_EXPERTS 128 → 256` (+ Rust re-export consumers).
- Router: `sqrtsoftplus` scoring, noaux_tc bias-for-selection-only, sum-normalize,
  ×1.5 scale — extend the routing kernel (`moe_route` currently fuses softmax+topk;
  add the scoring-func variant) or compute scores in a small pre-kernel.
- Hash routing: `tid2eid` gather replaces top-k on layers 0–2; token ids must flow
  into the layer forward (available in the wave step). Prefetch hook: hash-layer
  expert sets are pushed to the `ExpertCache` at token-sample time.
- `silu_mul` fused kernel: add the `swiglu_limit` clamp variant (one-sided gate
  clamp, symmetric up clamp).
- Shared expert: reuse dense-MLP machinery (Q8_0 weights).

### 6.4 Engine integration — `BatchedEngine` implements `ManagedBatchedModel` directly

_Status: built as described here. DeepSeek is a `ManagedBatchedModel` implemented directly by
`BatchedEngine` (`latent_moe/wave/managed.rs`), with its own wave sweep (`latent_moe/wave/sweep_wave.rs`);
the scheduler binds it as `Box<dyn ManagedBatchedModel>` (`candle-conversation/src/models/builder/load_arch.rs`:
`Engine::load` then `BatchedEngine::new`). `deepseek_batched_paged_attention_plan.md` §K step 5 records
the choice._

The model-to-engine contract is the scheduler-facing trait, and `BatchedEngine` fills it in:
- **Geometry and admission:** `num_layers`, `n_kv_head` (1), `head_dim` (`HEAD_DIM` = 512), `wave_geometry`,
  `prefill_width_cap` (bounded by compute saturation and what the KV side can admit, because the forward
  takes its transients from the CUDA pool rather than the wave span), `reclaimable_kv_bytes` (the weight
  zone's cedeable span), `maybe_change_dtype` (a no-op: dtypes are fixed at load).
- **Session KV creation:** `create_batched_session` builds the single-latent `ChunkedKvBacking` and pins
  the writer format to the two-region latent (nope `[0,448)` FP8 E4M3 at unit scale, rope `[448,512)` BF16).
- **The wave:** `forward_wave` calls the shared driver `wave_driver::drive_wave`, with `BatchedEngine`
  implementing its `WaveSweep` trait (`residual_row` is the mHC stream, `hc_mult·dim` F32, so the
  engine's carried hidden is flat as the plan wanted); `latent_moe/wave/sweep_wave.rs` owns the layer loop — the mHC
  `hc_pre`/`hc_post`/`hc_head` blocks (one fused Sinkhorn kernel each: 24-dim linear + sigmoid/softmax +
  20 Sinkhorn iterations on a 4×4, one warp per token, FP32 as in the reference), the paged-latent
  attention launches (§6.2), and the MoE through `ExpertCache::forward_routed`. Cross-session wave
  batching and the MoE seam are reused as they are for the Qwen models.
- **Speculative decode:** `speculative_draft`, `begin_verify`, `end_verify` and `truncate_sequence`
  (`deepseek_v4_speculative_decode.md`).

_Replaced design: the plan said not to implement a parallel `ManagedBatchedModel`, and instead to
extract three hard-coded spots — the paged-GQA attention step in `forward_attn_batched_single/_multi`, the
`norm→attn→add→norm→ffn→add` layer body of `forward_layer_batched_mixed` (with the mHC 4-copy stream hidden
inside a flat `hc_mult·4096` buffer), and the GQA `ChunkedKvBacking` construction — into defaulted trait
methods that only DeepSeek overrides, so the Qwen models stayed byte-for-byte unchanged. The code took the
other branch: the DeepSeek sweep owns the attention kernel, the mHC layer body and the session KV creation,
so no seam in the Qwen layer traits was extracted and no existing model was touched. The flat
`hc_mult·dim` residual carry survived the change._

### 6.5 Loader & config — single merged GGUF

The engine's streaming load is built around **one contiguous mmap** (`register_mmap_cuda`
page-locks a single mapping; expert byte-offsets index into it). So **merge the 4 bartowski
splits into one `DeepSeek-V4-Flash-0731-MXFP4.gguf`** (llama.cpp `gguf-split --merge`, or a
byte-merger) and load it through the same single-mmap path as every other model. The
first-pass multi-split `GgufModel` is superseded — only its `config_from_gguf` metadata logic
(§4.1) ports over. _Built: the engine loads the offline-prepared `DeepSeek-V4-Flash-0731-MXFP4_KO.gguf`
(`prepare_ko_gguf` over the merged file) through `latent_moe/loader.rs`, naming every tensor through the
`Arch` trait (`models/deepseek4.rs`)._

`ModelWeights::from_gguf_by_path_with_int8` analog: read the `deepseek4.*` keys, per-layer
kind from `compress_ratios`, and build attention weights, compressor(s), indexer, hc params
(FP32), router (+bias, +`tid2eid`), `ExpertCache` refs (MXFP4), shared expert, embed/head. YaRN
inv-freq per layer kind (`compress_rope_theta`+YaRN vs plain `rope_theta`). RoPE is applied
**in-model** (paged kernels' internal-RoPE contract doesn't apply; the sparse kernel takes
pre-rotated q and cached pre-rotated kv, matching the reference).

### 6.6 KV quantization roadmap (user directive: FP8 first)

1. **Phase 1 — BF16 caches** storing values that have *already passed through* the
   reference fake-quant points (bit-equal to reference; simplest kernels).
2. **Phase 2 — native FP8 storage**: E4M3 bytes + per-64 ue8m0 scales + BF16 rope
   tail (583 B/entry), kernel reads via format tag; bit-identical outputs to phase 1
   by construction (§3.3) — assert that in tests. Indexer cache to true FP4 nibbles
   (68 B/entry).
3. **Later — adaptive compression**: our C0–C9 policy over *compressed* entries is
   compression-on-compression; revisit only with perplexity evidence, sink-block
   protection concepts do not transfer 1:1 (each entry already summarizes 4–128
   tokens). Single-latent KV is not adaptively compressed in the code
   (`quantize_sealed_in_place` refuses a single-latent backing).

_Status (as built):_ the window is stored as the two-region latent (nope FP8 E4M3 at unit scale — a
per-64 ue8m0 scale is not representable in the per-chunk scalar record the window ring writes — and rope BF16);
the compressed corpus is the position-free int8+BF16 two-region cache (`deepseek_int8_hot_corpus.md`); the Indexer
keys stay f32. The 583 B/entry FP8+scale layout and the FP4 Indexer cache (68 B/entry) are not built.

---

## 7. Implementation plan (test-first, in order)

Each phase lands with its tests; no phase leaves stubs behind. **DONE** marks first-pass work.

- **DONE — MXFP4 codec (CPU + GPU) + reference math.** `GgmlDType::MXFP4`, raw-byte codec,
  standalone GPU dequant (bit-exact vs CPU), GGUF read path. The `latent_moe/`
  reference modules (rope/hyper/compressor/indexer/attention/moe/transformer) parity-tested
  against scalar transcriptions of `model.py`. These are the **correctness oracle** the engine
  kernels are tested against; the reference forward is not the shipping path.
- **DONE — real config + real MXFP4 decode.** `config_from_gguf` verified against the file;
  a real expert tensor decodes correctly on the GPU.

Remaining, test-first against the engine _(first-pass list; status as built: P-A done —
`merge_split_ggufs` + `prepare_ko_gguf`; P-C done, with the production expert path being the
`MXFP4_KO` int8 twin (§6.1); P-D not followed (§6.4); P-E done as `latent_moe/` + `models/deepseek4.rs`;
P-F done — `candle-conversation/tests/deepseek_rung4.rs` and the `latent_moe/wave/tests_*.rs` gates;
P-G partly — the live MoE dispatch and the paged-latent kernels, FP4 MMA is out of scope on sm_120;
P-H partly — see §6.6; P-I — the scheduler binds DeepSeek; DSpark speculative decode is built and gated,
but the scheduler's load path (`load_arch.rs`) attaches no drafter (see Open items); the conversation side
of the turn seal is specified but **Not built** (`deepseek_turn_seal_persistence.md` §9.2))_:

- **P-A — Merge to a single GGUF.** No `gguf-split`/`gguf-py`/llama.cpp on the box (verified),
  so write a small Rust merger: read the 4 splits, emit one `DeepSeek-V4-Flash-0731-MXFP4.gguf`
  on D: (merge tensor_infos with recomputed offsets, concatenate aligned tensor data, drop the
  `split.*` metadata). D: has 4.8 TB free. Confirm it loads via the single-mmap path (§6.5).
- **P-B — Copy the test, run it red.** Copy `test_parallel_batched_forwarding` into the DeepSeek model
  file (new DeepSeek `Dialect`, §2.9). It fails at load — that failing trail drives the rest (the step the
  first pass skipped). _Built as `deepseek4::tests::test_parallel_batched_forwarding`
  (`candle-transformers/src/models/deepseek4.rs`, `#[ignore]`d: the StoryRewrite harness at widths
  `[1,4,8,16,1]`, DeepSeek dialect, thinking blocks stripped before validation, speculative decode on)._
- **P-C — MXFP4 in the grouped expert GEMM (§6.1).** _As planned:_ add `block_c_mxfp4` +
  `block_type_traits` + the E2M1×E8M0 dequant to `gemx_dequant.cuh` so `grouped_matmul_gemx` dequants
  FP4→FP16 per-tile and runs the existing FP16 MMA; `dtype_to_qtype` maps MXFP4. _Replaced design:_
  `dtype_to_qtype` maps `MXFP4_KO` and not `MXFP4`; the experts are repacked offline to `MXFP4_KO` and run
  on the int8 grouped GEMM with a per-sub E8M0 fold (§6.1). No native FP4 (unsupported on sm_120).
- **P-D — Engine seams (§6.4).** _Not followed:_ the three hard-coded spots were not extracted into
  defaulted trait methods; DeepSeek implements `ManagedBatchedModel` directly, so there is no Qwen3
  regression gate to hold for this step.
- **P-E — DeepSeek overrides + loader.** Single-file loader (§6.5); the three overrides:
  sparse-gather attention kernel (§6.2, window ring + compressed + indexer, sinks, de-rotation)
  ported from the oracle; mHC `layer_forward` (hidden=16384 trick); session-owned KV. MoE
  deltas (§6.3): `moe_bucketize` 256, sqrtsoftplus/noaux router variant, one-sided swiglu clamp,
  hash-layer `tid2eid` plumbing. Experts via `ExpertCache` (streaming reused).
- **P-F — Green.** Drive the copied test to pass on the merged real weights: BF16-baseline
  single-session first, then the multi-context sweep + `wave_equivalence` co-batching gate.
  Criterion: coherent output, streaming stable, then measure t/s.
- **P-G — Perf.** FP4 MMA in the hot loop, expert-residency tuning (hot-pool sizing, hash +
  Markov prefetch across 43 layers), indexer top-k kernel, CUDA-graph capture for decode.
- **P-H — FP8 KV storage.** Native E4M3+ue8m0 window/compressed cache + FP4 indexer cache;
  bit-parity vs the BF16 baseline.
- **P-I — Products & DSpark.** zend integration (DeepSeek dialect, reasoning_effort, KV in the
  session lifecycle), long-context validation (RULER 64K–1M), then DSpark MTP (obtain `mtp.*`
  weights; verified-block decode with expert-prefetch lookahead).

## 8. Risks & open questions

1. **Minimally-invasive refactor (P-D) — retired.** The plan extracted three hard-coded seams into
   defaulted trait methods that had to leave Qwen3 byte-for-byte identical. DeepSeek implements
   `ManagedBatchedModel` directly (§6.4), so the risk did not arise; the Qwen models are untouched.
2. **FP4 MMA on sm_120 — RESOLVED (2026-08-05):** native block-scaled FP4 MMA is *not*
   available (ptxas rejects `kind::mxf4`/`block_scale` on sm_120, llama.cpp #19662). So the
   experts run as the `MXFP4_KO` int8 twin on the int8 tensor-core grouped GEMM with a per-sub E8M0
   fold (§6.1) — weights stay 4-bit in memory. Native FP4 is out of scope on this hardware.
3. **GGUF fidelity** — metadata is verified (§4); llama.cpp's DSv4 runtime reportedly has
   long-context corruption + KV-quant issues, so tensor-value fidelity is trusted only once the
   copied test produces coherent output on real weights (P-F), not before.
4. **Numeric cliffs** — compressor/gates/hc are FP32 in the reference "for convenience"; treat
   FP32 as required there (softmax over gated pools + 20 Sinkhorn iters are where BF16 drifts).
5. **Indexer top-k cost** — 64×128 q against T/4 keys dominates long context; a BF16 matmul+topk
   baseline is slow at ≥ 256K. Acceptable for bring-up; P-G owns the FP4-tensor-core scan (the
   VNNI provenance-scan experience applies).
6. **256-expert dispatch** (`MAX_EXPERTS` is 512 now) — audit every shared-memory / occupancy
   assumption in `moe_bucketize`/gather/scatter at 256, and `expert_base`/device-table sizing
   across 43 layers (11,008 experts total).
7. **Harness assumptions** — batch_test/`TestParams` `InferenceMode` K/V sweeps are GQA-era and
   don't map 1:1 onto DSv4 caches; the copied test starts with BF16/FP8 modes only.
8. **Iteration cost** — every e2e touches the merged ~156 GB mmap; the tiny-config reference
   oracle carries the fast TDD loop, real-weight runs are the gate, not the inner loop.
9. **`num_speculative_tokens` 7 vs `dspark_block_size` 5** — the built drafter uses the trained
   block size from the GGUF (`dflash.block_size`, 5); see `deepseek_v4_speculative_decode.md`.

---

## Open items

1. **The scheduler-bound engine has no speculative drafter.** `load_arch.rs` builds `Engine::load` and
   `BatchedEngine::new` (`Engine::load` is `Engine::load_with_drafter` with no drafter path); only the
   gates (`deepseek4::tests::test_parallel_batched_forwarding`, `latent_moe/wave/tests_speculative.rs`)
   call `Engine::load_with_drafter` and `BatchedEngine::with_drafter`. DSpark is built and gated but not
   reachable through the conversation engine.
2. **The drafter needs more than 64 GiB of VRAM.** `DsparkDrafter::load` refuses a smaller device
   (its routed experts are a permanently resident `Q2_KO` set of about 5 GiB), so speculative decode is
   unavailable on the 16 GB and 24 GB dev machines; the target itself still runs there through the
   expert cache.
3. **`latent_moe/footprint.rs` accounts the wrong corpus store** (f32 pair, `(m,l,acc)` tails) — see
   `deepseek_batched_paged_attention_plan.md` Open items.
4. **First-pass sizing in §3.2** (the 583 B/entry FP8 corpus, the 3.5 GB/session KV figure, the FP4
   Indexer cache) is the planned target; the built layout is recorded in the "KV row, as built" note.

---

*Doc status: rewritten 2026-08-05 after the course correction (§0) — the streaming engine is
the integration target, not a blocker; plan is test-first against it (§7). §0–§4 are verified
against the real GGUF; §5–§7 reflect the corrected engine-integration approach. Design docs are
authoritative — keep this in sync as the integration lands.*
