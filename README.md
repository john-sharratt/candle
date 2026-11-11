# One Card, One Stack

**Constraint-Driven Architecture for Asymptotically Stable Inference over Unbounded Agent Memory**

> Research fork of [Hugging Face Candle](https://github.com/huggingface/candle).
>
> - Technical report, **v2**: [docs/unbounded_agents.md](docs/unbounded_agents.md) — including its direct tests: output fidelity against depth (§9.6.8), recall past the working set (§9.11.2) and the codebase dependency battery (§9.12)
> - Technical report, **v1 (published 13 May 2026)**: [docs/unbounded_agents_v1.md](docs/unbounded_agents_v1.md) · [10.5281/zenodo.20156060](https://zenodo.org/records/20156060)
> - **Every measured number, and every outside figure it is compared with**: [docs/performance.md](docs/performance.md), with the raw rows in [docs/results/](docs/results/)

---

## The problem

Persistent agentic systems require context that grows without bound. Under standard full attention every cached token takes part in every subsequent step, so the stored error a step is exposed to — and the number of perturbed scores competing with the relevant tokens for attention — grows with context depth. This is not a compression problem; it is an architectural one. More VRAM defers the point at which tokens must be compressed or evicted; it does not change that coupling.

## The fix

**Theorem (§11.2 — Asymptotic Numerical Stability):** Under provenance-selected attention over a tiered context, the summed stored error of the tokens entering a generation step is bounded by a constant **independent of context depth N**, where the same bound under full attention grows linearly in N. It bounds exposure, not output error: it does not show that full-attention output degrades, and §11.2 says what it does and does not establish.

The fix is architectural: decouple the set of tokens participating in any generation step from the total number of tokens in context. That decoupling is *sufficient* for the bound at any depth; a full-attention system cannot offer it, because the bound's terms are the tokens it attends.

---

## What it does

Measured on three machines — an **RTX 4090 Laptop GPU (16 GB, 32 GB host RAM)**, an **RTX 3090 (24 GB, PCIe 3.0, no native FP8)** and an **RTX PRO 5000 Blackwell (72 GB)** — across twelve models from 0.5B to 284B parameters. Each result below is produced by a test in this repository, is set against the best figure published for the same model on the same class of card, and carries its reference in [docs/performance.md](docs/performance.md) (**[I*n*]** our result, **[E*n*]** an outside one).

1. **A 180B-parameter model on a 16 GB laptop GPU with 32 GB of RAM.** Qwen3.8-Flash-Next (180B total, 6B active) streams its experts VRAM → RAM → NVMe and serves eight concurrent sessions at **108.5 t/s aggregate decode**, 8/8 sessions correct at every rung, with 5.79–5.82× KV compression at C10. Every published single-GPU run of the model that states its host uses 64–128 GB of RAM, and the fastest, an RTX 5090 with 128 GB, decodes at 48.0 t/s. On a 24 GB RTX 3090 with 64 GB of RAM the same artifact decodes **138.1 t/s at one session and 543.8 t/s aggregate at ×16**, every session validated. **[I1 · I2 · I18 · E49 · E51 · E52]**

2. **Context length is free.** From 32K to 128K tokens, Flash-Next keeps **99% of its prefill and 111% of its decode** — decode is faster at 128K — and its speculative head accepts exactly 4.85 tokens a step on every row from 8K to 128K. The bound here is the model's own trained sparse-attention indexer, which caps each token's attended set; the engine adds no depth-dependent cost on top of it, where published llama.cpp runs of the same model lose a third of their decode between 6K and 110K. It is the bounded-working-set principle the theorem rests on, not a measurement of provenance. **[I3 · I4 · E49]**

3. **Up to 7.6× KV-cache compression, written inline, every output validated — no calibration data.** Each 32-token block is compressed in the forward that produced it, choosing its own format. The top rung reaches **4.1×–7.6×** across the fleet, and with it on, prefill stays within 5% of uncompressed at every depth and runs faster at 128K. llama.cpp's q8_0 cache gives 1.9× and vLLM's FP8 2×. **[I5 · I6 · I7 · E64 · E72]**

4. **Nearly 4× the compression of llama.cpp's q8_0 cache — while slowing decode less.** At 8K the top rung costs 0–8% of decode for 3.3×–6.3×. At 32K it costs 27–31% on the largest hybrids for 6.3–7.5×, where q8_0 costs 35% for 1.9×. **[I5 · I7 · E64 · E73]**

5. **One card out-decodes llama.cpp's best published rate by up to 37.9×.** Serving concurrent conversations, aggregate decode beats the best published llama.cpp single-stream figure for the same model on the same class of card by **2.6–36.3× on the RTX 3090**, **1.4–3.7× on 16 GB cards**, and **3.2–37.9× on Blackwell** (Qwen3.5-35B-A3B: 2,869.9 t/s against 194.0; Flash-Next 1,819.3 at ×16 on the TCC build against 48.02 on an RTX 5090). One case goes the other way and is stated as such: on 16 GB, a 3-bit Qwen3.6-35B that fits wholly in VRAM decodes a single stream at 183–249 t/s, where our streamed Q6_K aggregate is 336.7 — more work in total, less per stream. **[I8 · I16 · I18 · E1 · E17 · E19 · E21 · E27 · E45 · E46 · E51]**

6. **Sixty-four conversations on one card.** Qwen3.5-35B-A3B and Qwen3.6-35B-A3B on one 72 GB card: **2,869.9 and 2,761.0 t/s aggregate at ×64** under C10, from 230.2 and 476.6 t/s at one session — 12.5× and 5.8× — with every session validated and prefill holding across the range. Qwen3.5-0.8B serves 256 concurrent sessions at 10,805.3 t/s aggregate decode. The published single-card serving runs of these models stop at 5–10 concurrent requests. **[I8 · E47 · E48]**

7. **A 284B model at sixteen-way concurrency on one GPU.** DeepSeek-V4-Flash on one 72 GB card: **1,177.1 t/s prefill** — above the best published single-GPU figure of 748 — and **114.4 t/s aggregate decode**, 4.1× the best published single-GPU decode. **[I12 · E59 · E63]**

8. **Workstation work from a laptop.** On the 16 GB laptop, Qwen3.5-0.8B prefills 32 concurrent sessions at **46,056.8 t/s** — above the 24 GB RTX 3090's 42,982.2 t/s on the same rung — and decodes them at 3,810.7 t/s aggregate under C8. Qwen3-30B-A3B prefills at **6,139.1 t/s with its experts streaming**. A 35B MoE with ~28 GB of weights serves sixteen users at **330.8 and 336.7 t/s aggregate with 7.01× and 6.42× compression**, every session validated. **[I9 · I10 · I11]**

9. **One engine, every card.** The same gates pass on all three machines — 213 ladder rows on the RTX 3090, 192 on the laptop and 212 on the 72 GB card, no session failing on any of them — with the engine sizing its own memory partition to each. On the current builds C10 compresses the Qwen3.5-35B to **7.03× on the RTX 3090 and 7.01× on the laptop**, Qwen3.6-35B to 6.45× and 6.42×: the same selector, the same thresholds, making the same decisions on different silicon. **[I13 · I14]**

10. **The whole engine, proven under load.** The engine probe drives admission, per-turn context projection, the persistence thread, KV compaction and all three memory tiers at once. Its story gate is green on every probe run on every card — 8/8 on Flash-Next, 20/20 on Qwen3-30B-A3B, 16/16 on Qwen3.6-35B-A3B with speculative decode — and Flash-Next, which carries recurrent state outside the paged KV, holds 100% VRAM efficiency on the laptop and 99% on the 72 GB card. The VRAM-efficiency gate is open on the other rows: the 3090's 30B probe reads 59% and Flash-Next's 79%, and the Blackwell 35B probe reads 66% where the same probe read 92% on a neighbouring sweep, so the gate flips run to run rather than failing deterministically. **[I15 · I18]**

11. **The memory is live, and it is shared.** The retrieval figures in this repository are not from a corpus assembled for a benchmark: the provenance gates run against the running daemon's resident gallery over HTTP — 2,206 files, ~25.8M signature tokens — while it is being written to, and the scored held-out probes give R@1 0.310 / R@10 0.554 / MRR 0.397 for a single scan and R@1 0.392 / R@10 0.708 / MRR 0.504 once the working set's momentum is included. The same store is written by the coding agents and chat assistants connected to it, this engine's workers and the game's bots, and read by all of them. [`docs/results/provenance_gates_2026-10-26.md`]

The figures are aggregates where they say so. Aggregate against single-stream is the comparison the published record allows, and each row in [§6 of the performance document](docs/performance.md) gives its source's card, quantization, context length and engine.

---

## Reproducing the results

Every row is a test. Run one `cargo` process per model, with the card to itself: the gates size themselves from free VRAM.

**The width ladder** — throughput, concurrency and the compression ladder (claims 1, 3, 5, 6, 7, 8, 9):

```bash
cargo test --release --features cuda -p candle-transformers --lib \
  models::quantized_qwen38_moe::tests::test_parallel_batched_forwarding \
  -- --exact --ignored --nocapture --test-threads=1
```

| Model | Gate |
|---|---|
| Qwen3.8-Flash-Next (180B) | `models::quantized_qwen38_moe::tests::test_parallel_batched_forwarding` |
| DeepSeek-V4-Flash (284B) | `models::deepseek4::tests::test_parallel_batched_forwarding` |
| Qwen3.6-35B-A3B | `models::quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b` |
| Qwen3.5-35B-A3B | `models::quantized_qwen35_moe::tests::test_parallel_batched_forwarding_35b` |
| Qwen3-30B-A3B | `models::quantized_qwen3_moe::tests::test_parallel_batched_forwarding` |
| Qwen3.8-27B | `models::quantized_qwen38::tests::test_parallel_batched_forwarding_27b` |
| Qwen3.5-9B / 0.8B | `models::quantized_qwen35::tests::test_parallel_batched_forwarding_9b` / `_0_8b` |
| Qwen3-8B | `models::quantized_qwen3::tests::test_parallel_batched_forwarding` |
| Llama-2-7B / Llama-3.2-3B | `models::quantized_llama::tests::test_parallel_batched_forwarding_llama2` / `_llama3` |
| Qwen2-0.5B | `models::quantized_qwen2::tests::test_parallel_batched_forwarding` |

Each gate prints a `Performance Comparison` table. Its `t/s (bulk)` column is **prefill** and `t/s (single)` is **decode**, summed across the batch ([§2.3](docs/performance.md)). Each row validates every session's rewritten story against its expected text; across the 213 rows of the RTX 3090 sweep, the 192 of the laptop sweep and the 212 of the Blackwell sweep, not one session failed.

**The depth gate** — context length (claims 2 and 4):

```bash
cargo test --release --features cuda -p candle-transformers --lib \
  models::quantized_qwen38_moe::tests::profile_story_rewrite_vs_depth \
  -- --exact --ignored --nocapture --test-threads=1
```

`profile_decode_vs_depth` is the `Coherence` variant. The `long_context_*` tests in each model file run the same depth sweep for the rest of the fleet.

**The engine probe** — the whole engine under load (claim 10):

```bash
cargo test --release -p candle-conversation --features hub --test kv_fragmentation \
  qwen38_flash_next -- --exact --ignored --nocapture --test-threads=1
```

---

## System contributions

### 1. Online Markov expert prediction with wave-batched MoE

A self-learning transition matrix predicts expert routing from actual inference observations, with no offline calibration. A wave-batched grouped GEMM coalesces expert work across concurrent requests, so a weight crossing PCIe serves every session that routed to it. That is why aggregate decode keeps climbing with concurrency on a card that cannot hold the model (claims 1, 5, 6).

- Three-tier expert cache: VRAM → pinned RAM → mmap, so model size is bounded by storage, not VRAM
- Prefill and decode rows share one forward: a new conversation arriving does not stall the others
- Measured 69% prediction hit rate on Qwen3-30B-A3B routing

### 2. Three-tier paged context

```
GPU VRAM (hot)  →  CPU RAM (warm)  →  NVMe (cold)
```

Blocks are managed at 32-token granularity. Adaptive per-block quantization selects each block's K and V format from eleven compression levels (C0–C10) as the block is sealed. Two-phase prefill refresh at turn boundaries replaces each turn's decode-drifted K/V with K/V re-prefilled from the finished turn.

The Asymptotic Numerical Stability theorem is proven over this architecture. Under provenance-selected attention with hot-tier blocks originating from prefill-refreshed activations, the error per step is bounded by two terms: the hot-tier rounding constant, plus the fixed retrieval budget times the per-token error of a retrieved block. Neither depends on how many blocks reside in the warm or cold tiers.

### 3. Attentional provenance indexing with mid-decode reprojection

**Provenance indexing:** Q vectors are captured live during decode as persistent cognitive-state fingerprints. A Binary Directional Provenance scan (sign agreement between folded Q signatures) ranks the whole corpus on the GPU over a VRAM-resident paged gallery. It uses a 1-bit tensor-core backend on Ada (~9.5 ms steady-state on the laptop) and an INT8 tensor-core backend on Blackwell, both bit-identical to the scalar reference.

**Mid-decode reprojection:** every 64 decoded tokens, or at a line break once more than 16 have passed, the signatures of the last 256 decoded tokens drive a scan, and its result assembles the working set for the next forward, so the scan never blocks the current one. A probe session that would decode ahead of the real one (Speculative Context Decode, paper §6.5) is specified and not built.

### 4. Native quantized inference stack

Standard GEMM libraries dequantize weight matrices to full precision before computation, which runs out of memory during prefill on 16 GB. This stack's quantized matmul kernels never materialise a full-precision weight copy.

- Paged decode and prefill kernels at 32-token chunk granularity, including an INT8 prefill attention path
- Fused sampling kernel covering all common sampling modifiers in one CUDA launch
- Greedy decomposition for smooth 1–500 token throughput without remainder handling

---

## Applications

### Zen Code

A persistent AI coding assistant with genuine long-term memory across sessions:

- `zend` — background daemon hosting the conversation engine
- `zen-vscode` — VS Code extension (Continue fork) consuming the API
- Shared KV prefix across developer workspace forks
- Institutional memory: facts, decisions and code relationships accumulate across all sessions

### Battle Cities

An NPC narrative game where each agent maintains unbounded memory across story branches. The conversation tree supports branching, summarisation and task nodes with continuous KV projection.

---

## Crate architecture

```
candle-conversation        (conversation engine)
       ↓
candle-transformers        (batched inference, MoE and hybrid models)
       ↓
candle-nn                  (layers, VarBuilder, KV cache system)
       ↓
candle-core                (Tensor, Device, DType, ops)
       ↓
candle-kernels             (AOT CUDA kernels)
```

The map from the theory to the code is [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Build

```bash
# Basic check
cargo check --workspace

# Release build with CUDA
cargo build --workspace --release --features cuda

# Run tests
cargo test --workspace
cargo test --features cuda          # GPU tests

# Linting
cargo fmt --all -- --check
cargo clippy --workspace --tests --examples -- -D warnings
```

**Windows:** `cuda.dll`, `cublas.dll`, `curand.dll` must be on PATH.

---

## Hardware

| Machine | VRAM | Host | Link | Driver |
|---|---|---|---|---|
| RTX 4090 Laptop GPU | 16 GB | Core Ultra 9 185H, 32 GB | PCIe 4.0, trained ×8, ~12 GB/s | WDDM |
| RTX 3090 | 24 GB | i7-10700K, 64 GB | PCIe 3.0 ×16, ~12 GB/s; no native FP8 | WDDM |
| RTX PRO 5000 Blackwell | 72 GB | Ryzen 9 9950X3D, 189 GB | PCIe 5.0 ×16 | WDDM to 2026-10; TCC since |

**Model size is not bounded by VRAM.** The expert cache streams VRAM → pinned RAM → mmap, so a mixture-of-experts model's resident footprint is its dense weights plus whatever expert working set fits. A bigger card buys speed, not feasibility.

---

## Models

| Model | Size | Notes |
|---|---|---|
| DeepSeek-V4-Flash-0731 | 284B / 13B active | native-sparse latent attention, MXFP4 experts |
| Qwen3.8-Flash-Next | 180B / 6B active | hybrid + MoE with MTP drafter; Q3_KO experts under 64 GB |
| Qwen3.5-35B-A3B, Qwen3.6-35B-A3B | 35B / 3B active | DeltaNet hybrid + MoE |
| Qwen3-30B-A3B | 30B / 3B active | MoE |
| Qwen3.8-27B | 27B dense | hybrid |
| Qwen3.5-9B, Qwen3-8B, Llama-2-7B, Llama-3.2-3B, Qwen3.5-0.8B, Qwen2-0.5B | 0.5–9B dense | |

---

## License

This repository is a fork of [Hugging Face Candle](https://github.com/huggingface/candle), which is dual-licensed under [MIT](LICENSE-MIT) and [Apache 2.0](LICENSE-APACHE). All modifications in this fork are released under the same dual license.
