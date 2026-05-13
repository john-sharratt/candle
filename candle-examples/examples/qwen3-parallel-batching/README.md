# Qwen3 Parallel Continuous Batching

This example demonstrates **parallel continuous batching** with Qwen3, showing how to handle multiple sequences with different lengths by using batching windows.

## Overview

This example demonstrates the correct way to handle variable-length sequences in batched inference:

- **Batching Windows**: Sequences are processed in windows with equal token counts
- **Independent KV caches**: Each sequence maintains its own cache state
- **Uniform Batch Constraint**: All sequences in a batch must have the same `input_len`
- **Parallel execution**: All sequences processed together in a single forward pass

## Key Features

✅ Handles sequences with different prompt lengths  
✅ Uses batching windows to maintain uniform input_len  
✅ True GPU batch parallelism  
✅ Per-sequence RoPE with independent position embeddings  
✅ Performance comparison with sequential processing  

## Running the Example

```bash
cargo run --example qwen3-parallel-batching --release --features cuda
```

### Example Output

The example will:
1. Download Qwen3-0.6B model (~400MB)
2. Create two sequences with different prompt lengths
   - Sequence 1: "What is 2+2? Answer:" (4-5 tokens)
   - Sequence 2: "Explain photosynthesis:" (3-4 tokens)
3. Process both sequences in parallel using batching windows for 30 generation steps
4. Compare parallel batching performance vs sequential processing
5. Display the speedup and generated token sequences

### Expected Results

```
Phase 1: Prefill (processing prompts with batching windows)

  Seq 1: "What is 2+2? Answer:" (5 tokens)
  Seq 2: "Explain photosynthesis:" (3 tokens)

  Batch: seq1[00..03] (03 tokens) | seq2[00..03] (03 tokens)
  Batch: seq1[03..05] (02 tokens) | seq2[03..03] (00 tokens)

  Seq 1 prefilled: 5 total tokens
  Seq 2 prefilled: 3 total tokens

⚡ Generation phase - parallel batched forwarding:
   (30 steps with batched processing)
```

## Technical Details

### Handling Variable-Length Prompts with Batching Windows

The key to correct batching is the **window concept**:

```rust
// Determine remaining tokens
let seq1_remaining = seq1_prompt.len() - seq1_pos;
let seq2_remaining = seq2_prompt.len() - seq2_pos;

// Find common window size (minimum remaining)
let batch_len = std::cmp::min(seq1_remaining, seq2_remaining);

// Both sequences get exactly batch_len tokens this iteration
// This ensures all sequences have uniform input_len
```

### SequenceContext API

The example uses `SequenceContext` for type-safe batch processing:

```rust
let mut contexts = vec![
    SequenceContext {
        kv_caches: &mut seq1_caches,
        offset: seq1_offset,        // Current KV cache length
        input_ids: &seq1_input,     // Exactly batch_len tokens
        input_len: batch_len,       // All sequences must match
    },
    SequenceContext {
        kv_caches: &mut seq2_caches,
        offset: seq2_offset,
        input_ids: &seq2_input,
        input_len: batch_len,       // MUST equal seq1's input_len
    },
];

let outputs = model.forward_batch(&mut contexts)?;
```

### Batched RoPE (Rotary Position Embeddings)

Each sequence in the batch gets its own rotation based on its current position offset:

```rust
// Different offset for each sequence
let offsets = vec![seq1_offset, seq2_offset];
let (q, k) = self.apply_rotary_emb_batched(&q, &k, &offsets)?;
```

This allows sequences at different generation stages to be processed together.

## Performance Benefits

Typical speedup for 2-sequence batch: **1.1-1.3x**

The speedup scales better with:
- Larger batch sizes (4-8 sequences)
- Longer sequences
- More complex models (7B+)

## Use Cases

- **Multi-user serving**: Handle multiple user requests simultaneously
- **Beam search**: Process multiple candidates in parallel
- **Speculative decoding**: Run draft and verification models together
- **Multi-task inference**: Process different prompts concurrently
        input_len: 1,           // All sequences must have same input_len
    },
    SequenceContext {
        kv_caches: &mut seq2_caches,
        offset: 15,             // Sequence 2 at position 15
        input_ids: &seq2_input,
        input_len: 1,           // All sequences must have same input_len
    },
];

let outputs = model.forward_batch(&mut contexts)?;
```

## Performance Benefits

- **GPU Utilization**: Amortizes kernel launch overhead across multiple sequences
- **Memory Bandwidth**: Better memory coalescing with batched operations
- **Throughput**: Higher tokens/second compared to sequential processing
- **Latency**: Each sequence completes faster than waiting in a sequential queue

## Use Cases

This pattern is ideal for:
- **Serving multiple users**: Process requests from different users in parallel
- **Speculative decoding**: Generate multiple candidate continuations simultaneously
- **Beam search**: Explore multiple hypotheses in parallel
- **Multi-turn conversations**: Handle multiple chat sessions concurrently
