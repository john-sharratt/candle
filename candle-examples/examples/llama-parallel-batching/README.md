# Llama Parallel Continuous Batching Example

This example demonstrates parallel continuous batching for the Llama model, showing how to process multiple sequences simultaneously even when they're at different lengths.

## What This Example Shows

1. **Prefill Phase with Windowing**: Process two prompts of different lengths using batching windows
   - Sequence 1: "What is 2+2? Answer:" (shorter)
   - Sequence 2: "Explain photosynthesis:" (longer)
2. **Batching Window Concept**: Dynamically finds common token counts for each batch window
3. **Parallel Generation**: Generate 30 tokens for both sequences using batched forwarding
4. **Performance Comparison**: Compare parallel vs sequential processing

## Key Features

- **Variable-Length Handling**: Demonstrates how to batch sequences with different lengths
- **Batching Windows**: Processes prompts by finding min(remaining_tokens) for each window
- **Independent KV Caches**: Each sequence maintains its own cache state
- **True GPU Parallelism**: Process sequences in a single forward pass
- **Uniform Batch Constraint**: All sequences in a batch must have the same input_len

## Running the Example

```bash
cd candle-examples
cargo run --example llama-parallel-batching --release --features cuda
```

## Expected Output

The example will:
1. Download the Llama-3.2-1B-Instruct model (~700MB)
2. Prefill two sequences using batching windows
3. Generate 30 tokens per sequence in parallel
4. Compare performance with sequential processing
5. Show speedup achieved through batching

## Technical Details

### Handling Variable-Length Prompts

The key to handling variable-length sequences is the **batching window** concept:

```rust
// Determine tokens remaining in each sequence
let seq1_remaining = seq1_prompt.len() - seq1_pos;
let seq2_remaining = seq2_prompt.len() - seq2_pos;

// Find common window size
let batch_len = std::cmp::min(seq1_remaining, seq2_remaining);

// Extract tokens for this window
let seq1_batch = &seq1_prompt[seq1_pos..seq1_pos + batch_len];
let seq2_batch = &seq2_prompt[seq2_pos..seq2_pos + batch_len];
```

This ensures all sequences in a batch have exactly the same `input_len`, which is required by the `forward_batch` API.

### API Usage

```rust
let mut contexts = vec![
    SequenceContext {
        kv_caches: &mut seq1_caches,
        offset: seq1_caches.current_seq_len(),
        input_ids: &seq1_input,
        input_len: batch_size,  // All sequences must have same length
    },
    SequenceContext {
        kv_caches: &mut seq2_caches,
        offset: seq2_caches.current_seq_len(),
        input_ids: &seq2_input,
        input_len: batch_size,  // Must match seq1
    },
];

let outputs = model.forward_batch(&mut contexts)?;
```

## Performance Benefits

Typical speedup for 2-sequence batch: **1.1-1.3x**

The speedup scales better with:
- Larger batch sizes (4-8 sequences)
- Longer sequences
- More complex models (7B, 13B+)

## Use Cases

- **Multi-user serving**: Handle multiple user requests simultaneously
- **Beam search**: Process multiple candidates in parallel
- **Speculative decoding**: Run draft and verification models together
- **Multi-task inference**: Process different prompts concurrently

## Model Compatibility

This example uses Llama-3.2-1B-Instruct, but the same API works with:
- Llama 2 (7B, 13B, 70B)
- Llama 3 (8B, 70B)
- Llama 3.1 (8B, 70B, 405B)
- Llama 3.2 (1B, 3B)

All quantization formats (Q4_0, Q4_K_M, Q8_0, etc.) are supported.
