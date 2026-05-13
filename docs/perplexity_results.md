# WikiText-2 Perplexity Evaluation Results

## Methodology

- **Dataset**: WikiText-2 test split (`wiki.test.raw`, 4358 lines, ~1.29 MB)
- **Metric**: Perplexity = exp(average negative log-likelihood)
- **Evaluation**: Non-overlapping chunks with stride = context size
- **Loss function**: `candle_nn::loss::cross_entropy` per chunk, averaged over all scored tokens
- **Device**: CUDA (DeviceId(1))
- **Tool**: `candle-examples/examples/perplexity-eval`

## Results at Context Size 512

All models evaluated with identical parameters (context=512, stride=512, non-overlapping).

| Model | Quant | Parameters | PPL ↓ | Tokens Scored | Avg NLL |
|-------|-------|-----------|-------|---------------|---------|
| Qwen3-30B-A3B (MoE) | Q4_K_M | 30B (3B active) | **9.21** | 298,355 | 2.2204 |
| Qwen2-7B | Q4_0 | 7B | 10.20 | 298,355 | 2.3224 |
| Qwen3-14B | Q4_K_M | 14B | 11.77 | 298,355 | 2.4654 |
| Qwen3-8B | Q4_K_M | 8B | 13.12 | 298,355 | 2.5738 |
| Qwen2-0.5B | Q4_0 | 0.5B | 21.11 | 298,355 | 3.0498 |
| Llama-3.2-3B† | Q4_K_M | 3B | 21.73 | 288,372 | 3.0785 |

## Results at Context Size 2048

Larger context generally improves perplexity. Not all models fit in VRAM at this context size.

| Model | Quant | Parameters | PPL ↓ | Tokens Scored | Avg NLL |
|-------|-------|-----------|-------|---------------|---------|
| Qwen2-7B | Q4_0 | 7B | **7.80** | 298,793 | 2.0541 |
| Qwen3-8B | Q4_K_M | 8B | 9.85 | 298,793 | 2.2875 |
| Qwen2-0.5B | Q4_0 | 0.5B | 15.61 | 298,793 | 2.7479 |
| Llama-3.2-3B† | Q4_K_M | 3B | 15.89 | 288,795 | 2.7659 |

Qwen3-14B and Qwen3-30B-A3B exceeded GPU VRAM at context size 2048.

## Notes

- **†Llama-3.2-3B**: Uses VibeStudio/Nidum-Llama-3.2-3B-Uncensored fine-tune GGUF, not the base model (base model is gated and requires HF authentication).
- **Llama-3.1-8B**: Not evaluated — gated model requiring HuggingFace token. No public GGUF available in cache.
- **Token count difference**: Llama models use a different tokenizer than Qwen models, resulting in ~288K vs ~298K tokens for the same text.
- **Qwen3 vs Qwen2 at ctx=512**: Qwen2-7B (Q4_0) outperforms Qwen3-8B (Q4_K_M) and Qwen3-14B (Q4_K_M). This may reflect quantization format differences (Q4_0 vs Q4_K_M), Qwen3's thinking-mode overhead, or instruction-tuning effects on perplexity benchmarks.
- **MoE advantage**: Qwen3-30B-A3B achieves the best PPL at ctx=512 despite only 3B active parameters, demonstrating MoE routing efficiency.

## GGUF Sources

| Model | HuggingFace Repo | File |
|-------|------------------|------|
| Qwen3-30B-A3B | `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` | `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` |
| Qwen3-14B | `unsloth/Qwen3-14B-GGUF` | `Qwen3-14B-Q4_K_M.gguf` |
| Qwen3-8B | `unsloth/Qwen3-8B-GGUF` | `Qwen3-8B-Q4_K_M.gguf` |
| Qwen2-7B | `Qwen/Qwen2-7B-Instruct-GGUF` | `qwen2-7b-instruct-q4_0.gguf` |
| Qwen2-0.5B | `Qwen/Qwen2-0.5B-Instruct-GGUF` | `qwen2-0_5b-instruct-q4_0.gguf` |
| Llama-3.2-3B | `VibeStudio/Nidum-Llama-3.2-3B-Uncensored-GGUF` | `model-Q4_K_M.gguf` |
