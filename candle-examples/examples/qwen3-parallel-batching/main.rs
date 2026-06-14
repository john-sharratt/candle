/// Demonstrates parallel continuous batching with Qwen3.
///
/// This example shows how to process multiple sequences at different generation stages
/// simultaneously, achieving true GPU batch parallelism despite different offsets.
///
/// Run with:
/// ```bash
/// cargo run --example qwen3-parallel-batching --release --features cuda
/// ```

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::quantized::pinned_staging::PinnedStager;
use candle::{Device, Result, Tensor};
use candle_transformers::models::batched_layer::{BatchedPrefillMeta, DecodeHeaders};
use candle_transformers::models::batched_model::BatchedInference;
use candle_transformers::models::kv_cache_utils::SequenceContext;
use candle_transformers::models::quantized_qwen3::ModelWeights;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    println!("\n=== Qwen3 Parallel Continuous Batching Example ===\n");

    // Download model
    println!("Downloading Qwen3-0.6B model...");
    let api = Api::new()
        .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

    let repo = api.repo(Repo::with_revision(
        "unsloth/Qwen3-0.6B-GGUF".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let model_path = repo
        .get("Qwen3-0.6B-Q4_K_M.gguf")
        .map_err(|e| candle::Error::Msg(format!("Failed to download model: {}", e)))?;

    println!("   Model path: {:?}", model_path);

    // Initialize device and load model
    let device = Device::cuda_if_available(0)?;
    let stager = PinnedStager::new_from_device(&device);
    println!("   Device: {:?}\n", device);

    println!("Loading model weights...");
    let start = std::time::Instant::now();
    let raw_model = ModelWeights::from_gguf_by_path(&model_path, &device)?;

    // Wrap in BatchedInference with the model's RoPE configuration
    let inv_freq = raw_model
        .rope_inv_freq()
        .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
    let model = BatchedInference::new_with_inv_freq(raw_model, inv_freq, 4096, &device)?;
    println!("   Loaded in {:.2}s\n", start.elapsed().as_secs_f64());

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer_repo = api.repo(Repo::with_revision(
        "Qwen/Qwen2-0.5B-Instruct".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));
    let tokenizer_path = tokenizer_repo
        .get("tokenizer.json")
        .map_err(|e| candle::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| candle::Error::Msg(format!("Failed to load tokenizer: {}", e)))?;

    // Create two sequences with different prompt lengths
    println!("Setting up two sequences with real questions:\n");

    // Sequence 1: Longer, more complex question
    let prompt1 = "<|im_start|>user\nPlease provide a detailed explanation of photosynthesis. Include the light-dependent and light-independent reactions. Explain how chlorophyll captures light energy. What is the role of ATP and NADPH in this process?<|im_end|>\n<|im_start|>assistant\n";
    let seq1_prompt = tokenizer
        .encode(prompt1, true)
        .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    // Sequence 2: Another longer question with different complexity
    let prompt2 = "<|im_start|>user\nExplain artificial intelligence and machine learning. What is the difference between supervised and unsupervised learning? How do deep neural networks learn? What are some current applications and limitations of AI?<|im_end|>\n<|im_start|>assistant\n";
    let seq2_prompt = tokenizer
        .encode(prompt2, true)
        .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    println!("  Seq 1: \"{}\" ({} tokens)", prompt1, seq1_prompt.len());
    println!("  Seq 2: \"{}\" ({} tokens)\n", prompt2, seq2_prompt.len());

    // Create KV caches for both sequences
    let mut seq1_caches = model.model().create_kv_caches(512);
    let mut seq2_caches = model.model().create_kv_caches(512);

    // Phase 1: Prefill - process each prompt with full input_len
    println!("Prefill phase - processing with variable input_len:\n");

    // Process seq 1 entirely
    let seq1_input = Tensor::new(&seq1_prompt[..], &device)?
        .unsqueeze(0)?
        .contiguous()?;
    let seq1_offset = seq1_caches.current_seq_len();
    let mut contexts = vec![SequenceContext {
        kv_caches: &mut seq1_caches,
        offset: seq1_offset,
        input_ids: &seq1_input,
        input_len: seq1_prompt.len(),
        write_offset_shift: 0,
    }];
    let offsets: Vec<usize> = contexts.iter().map(|c| c.offset).collect();
    let seq_len = contexts[0].input_len;
    let meta = BatchedPrefillMeta::new(&offsets, seq_len, &device)?;
    let generation = stager.begin_generation();
    let _outputs = model.forward_batch(&mut contexts, &generation, DecodeHeaders::Prefill(meta))?;
    println!("  Seq 1: cached with input_len={}", seq1_prompt.len());

    // Process seq 2 entirely
    let seq2_input = Tensor::new(&seq2_prompt[..], &device)?
        .unsqueeze(0)?
        .contiguous()?;
    let seq2_offset = seq2_caches.current_seq_len();
    let mut contexts = vec![SequenceContext {
        kv_caches: &mut seq2_caches,
        offset: seq2_offset,
        input_ids: &seq2_input,
        input_len: seq2_prompt.len(),
        write_offset_shift: 0,
    }];
    let offsets2: Vec<usize> = contexts.iter().map(|c| c.offset).collect();
    let seq_len2 = contexts[0].input_len;
    let meta2 = BatchedPrefillMeta::new(&offsets2, seq_len2, &device)?;
    let generation = stager.begin_generation();
    let _outputs =
        model.forward_batch(&mut contexts, &generation, DecodeHeaders::Prefill(meta2))?;
    println!("  Seq 2: cached with input_len={}\n", seq2_prompt.len());

    // Generation phase
    let seq1_len = seq1_caches.current_seq_len();
    let seq2_len = seq2_caches.current_seq_len();

    let seq1_logits = model.model().forward(
        &mut seq1_caches,
        &Tensor::new(&[seq1_prompt[seq1_prompt.len() - 1]], &device)?
            .unsqueeze(0)?
            .contiguous()?,
        seq1_len - 1,
    )?;
    let seq1_next = sample_token(&seq1_logits)?;

    let seq2_logits = model.model().forward(
        &mut seq2_caches,
        &Tensor::new(&[seq2_prompt[seq2_prompt.len() - 1]], &device)?
            .unsqueeze(0)?
            .contiguous()?,
        seq2_len - 1,
    )?;
    let seq2_next = sample_token(&seq2_logits)?;

    // Keep prompt refs for later
    let seq1_prompt_len = seq1_prompt.len();
    let seq2_prompt_len = seq2_prompt.len();

    // Generation loop - 100 tokens for better measurement
    let num_steps = 100;

    // Now demonstrate parallel batched generation
    println!("TEST 1: Position misalignment validation");
    println!("  Offsetting Seq 2 by 1 position for misaligned batch testing");

    let mut seq1_tokens = seq1_prompt.clone();
    seq1_tokens.push(seq1_next);
    let mut seq2_tokens = seq2_prompt.clone();
    seq2_tokens.push(seq2_next);

    let seq2_offset_input = Tensor::new(&[seq2_tokens[seq2_prompt_len]], &device)?.unsqueeze(0)?;
    {
        let seq2_offset_val = seq2_caches.current_seq_len();
        let mut context = vec![SequenceContext {
            kv_caches: &mut seq2_caches,
            offset: seq2_offset_val,
            input_ids: &seq2_offset_input,
            input_len: 1,
            write_offset_shift: 0,
        }];
        let generation = stager.begin_generation();
        let _ = model.forward_batch(
            &mut context,
            &generation,
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            },
        )?;
    }
    println!(
        "  Seq 1 position: {} | Seq 2 position: {} (misaligned)\n",
        seq1_caches.current_seq_len(),
        seq2_caches.current_seq_len()
    );

    // Phase 2: Parallel generation with BATCHED single-token processing
    println!("Phase 2: Parallel generation with BATCHED single-token processing (100 steps)");
    println!("  This phase uses forward_batch for each step instead of individual forwards\n");

    let mut phase2_tokens1 = vec![seq1_next];
    let mut phase2_tokens2 = vec![seq2_next];

    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_offset = seq1_caches.current_seq_len();
        let seq2_offset = seq2_caches.current_seq_len();

        let seq1_input = Tensor::new(&[phase2_tokens1[step]], &device)?.unsqueeze(0)?;
        let seq2_input = Tensor::new(&[phase2_tokens2[step]], &device)?.unsqueeze(0)?;

        let mut contexts = vec![
            SequenceContext {
                kv_caches: &mut seq1_caches,
                offset: seq1_offset,
                input_ids: &seq1_input,
                input_len: 1,
                write_offset_shift: 0,
            },
            SequenceContext {
                kv_caches: &mut seq2_caches,
                offset: seq2_offset,
                input_ids: &seq2_input,
                input_len: 1,
                write_offset_shift: 0,
            },
        ];

        let generation = stager.begin_generation();
        let outputs = model.forward_batch(
            &mut contexts,
            &generation,
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            },
        )?;

        // Sample next tokens
        let seq1_next = sample_token(&outputs.get(0)?)?;
        let seq2_next = sample_token(&outputs.get(1)?)?;

        phase2_tokens1.push(seq1_next);
        phase2_tokens2.push(seq2_next);
        seq1_tokens.push(seq1_next);
        seq2_tokens.push(seq2_next);

        if (step + 1) % 25 == 0 {
            println!(
                "  Step {:3}: Seq1 offset={:3} token={:5} | Seq2 offset={:3} token={:5}",
                step + 1,
                seq1_offset,
                seq1_next,
                seq2_offset,
                seq2_next
            );
        }
    }
    let batched_time = start.elapsed();
    println!(
        "  Batched processing time: {:.2}ms\n",
        batched_time.as_secs_f64() * 1000.0
    );

    // Keep copies of generated tokens for Phase 3 replay
    let generated_seq1 = seq1_tokens.clone();
    let generated_seq2 = seq2_tokens.clone();

    // Phase 3: Performance comparison
    println!("Phase 3: Performance comparison");
    println!("  Sequential: individual forward() calls for each sequence");
    println!("  Parallel:   forward_batch() for both sequences together");
    println!("  Both use misaligned offsets (Seq 2 ahead by 1 position)\n");

    // Reset caches and tokens for fair comparison
    let mut seq1_caches_seq = model.model().create_kv_caches(512);
    let mut seq2_caches_seq = model.model().create_kv_caches(512);

    // Prefill both sequences again
    let seq1_prompt_tensor = Tensor::new(&seq1_prompt[..], &device)?
        .unsqueeze(0)?
        .contiguous()?;
    let seq2_prompt_tensor = Tensor::new(&seq2_prompt[..], &device)?
        .unsqueeze(0)?
        .contiguous()?;
    let _ = model
        .model()
        .forward(&mut seq1_caches_seq, &seq1_prompt_tensor, 0)?;
    let _ = model
        .model()
        .forward(&mut seq2_caches_seq, &seq2_prompt_tensor, 0)?;

    // Apply the offset to seq2_seq just like in the test
    let seq2_offset_for_seq =
        Tensor::new(&[generated_seq2[seq2_prompt_len]], &device)?.unsqueeze(0)?;
    let seq2_offset_for_seq_val = seq2_caches_seq.current_seq_len();
    let _ = model.model().forward(
        &mut seq2_caches_seq,
        &seq2_offset_for_seq,
        seq2_offset_for_seq_val,
    )?;

    // Sequential generation
    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_input =
            Tensor::new(&[generated_seq1[seq1_prompt_len + step]], &device)?.unsqueeze(0)?;
        let seq2_input =
            Tensor::new(&[generated_seq2[seq2_prompt_len + 1 + step]], &device)?.unsqueeze(0)?;

        let seq1_offset = seq1_caches_seq.current_seq_len();
        let seq2_offset = seq2_caches_seq.current_seq_len();

        let _ = model
            .model()
            .forward(&mut seq1_caches_seq, &seq1_input, seq1_offset)?;
        let _ = model
            .model()
            .forward(&mut seq2_caches_seq, &seq2_input, seq2_offset)?;
    }
    let total_sequential_time = start.elapsed().as_secs_f64() * 1000.0;

    // Parallel implementation
    let mut seq1_caches_par = model.model().create_kv_caches(512);
    let mut seq2_caches_par = model.model().create_kv_caches(512);
    let _ = model
        .model()
        .forward(&mut seq1_caches_par, &seq1_prompt_tensor, 0)?;
    let _ = model
        .model()
        .forward(&mut seq2_caches_par, &seq2_prompt_tensor, 0)?;

    // Apply the offset to seq2_par just like in the test
    let seq2_offset_for_par =
        Tensor::new(&[generated_seq2[seq2_prompt_len]], &device)?.unsqueeze(0)?;
    {
        let seq2_offset_val = seq2_caches_par.current_seq_len();
        let mut context = vec![SequenceContext {
            kv_caches: &mut seq2_caches_par,
            offset: seq2_offset_val,
            input_ids: &seq2_offset_for_par,
            input_len: 1,
            write_offset_shift: 0,
        }];
        let generation = stager.begin_generation();
        let _ = model.forward_batch(
            &mut context,
            &generation,
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            },
        )?;
    }

    // Parallel generation
    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_input =
            Tensor::new(&[generated_seq1[seq1_prompt_len + step]], &device)?.unsqueeze(0)?;
        let seq2_input =
            Tensor::new(&[generated_seq2[seq2_prompt_len + 1 + step]], &device)?.unsqueeze(0)?;

        let seq1_offset = seq1_caches_par.current_seq_len();
        let seq2_offset = seq2_caches_par.current_seq_len();

        let mut contexts = vec![
            SequenceContext {
                kv_caches: &mut seq1_caches_par,
                offset: seq1_offset,
                input_ids: &seq1_input,
                input_len: 1,
                write_offset_shift: 0,
            },
            SequenceContext {
                kv_caches: &mut seq2_caches_par,
                offset: seq2_offset,
                input_ids: &seq2_input,
                input_len: 1,
                write_offset_shift: 0,
            },
        ];

        let generation = stager.begin_generation();
        let _ = model.forward_batch(
            &mut contexts,
            &generation,
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            },
        )?;
    }
    let total_parallel_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("  Sequential: {:.2}ms", total_sequential_time);
    println!("  Parallel:   {:.2}ms", total_parallel_time);
    println!(
        "  Speedup:    {:.2}x\n",
        total_sequential_time / total_parallel_time
    );

    // Display generated text
    println!("=== Generated Output ===\n");

    // Get the full decoded text
    let full_response1 = tokenizer
        .decode(&seq1_tokens, false)
        .map_err(|e| candle::Error::Msg(format!("Decode failed: {}", e)))?;
    let full_response2 = tokenizer
        .decode(&seq2_tokens, false)
        .map_err(|e| candle::Error::Msg(format!("Decode failed: {}", e)))?;

    // Extract just the response part (after "<|im_start|>assistant\n")
    let resp1 = if let Some(pos) = full_response1.rfind("<|im_start|>assistant") {
        &full_response1[pos + 21..]
    } else {
        &full_response1
    };

    let resp2 = if let Some(pos) = full_response2.rfind("<|im_start|>assistant") {
        &full_response2[pos + 21..]
    } else {
        &full_response2
    };

    println!("Sequence 1 ({} tokens):", seq1_tokens.len());
    println!("{}", resp1.trim());

    println!("Sequence 2 ({} tokens):", seq2_tokens.len());
    println!("{}", resp2.trim());

    println!("\n=== Summary ===");
    println!(
        "Processed 2 sequences in parallel for {} generation steps",
        num_steps
    );
    println!("Sequence 1: {} total tokens", seq1_tokens.len());
    println!("Sequence 2: {} total tokens", seq2_tokens.len());
    println!(
        "Speedup: {:.2}x (parallel vs sequential)",
        total_sequential_time / total_parallel_time
    );

    Ok(())
}

/// Simple greedy sampling - picks token with highest logit
fn sample_token(logits: &Tensor) -> Result<u32> {
    let logits = logits.squeeze(0)?;
    let logits_vec = logits.to_vec1::<f32>()?;

    let max_idx = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    Ok(max_idx as u32)
}
