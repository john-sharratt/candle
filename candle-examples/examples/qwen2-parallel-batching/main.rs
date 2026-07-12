/// Parallel Continuous Batching Example for Qwen2
use candle::quantized::pinned_staging::PinnedStager;
use candle::{Device, Result, Tensor};
use candle_transformers::models::batched_layer::{BatchedPrefillMeta, DecodeHeaders};
use candle_transformers::models::batched_model::BatchedInference;
use candle_transformers::models::kv_cache_utils::SequenceContext;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    println!("\n=== Qwen2 Parallel Continuous Batching Example ===\n");

    // Download model
    let api = Api::new().map_err(|e| candle::Error::Msg(format!("HF API error: {}", e)))?;
    let repo = api.repo(Repo::with_revision(
        "Qwen/Qwen2-0.5B-Instruct-GGUF".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    println!("Downloading Qwen2-0.5B-Instruct model...");
    let model_path = repo
        .get("qwen2-0_5b-instruct-q4_0.gguf")
        .map_err(|e| candle::Error::Msg(format!("Download failed: {}", e)))?;

    let device = Device::cuda_if_available(0)?;
    let stager = PinnedStager::new_from_device(&device);
    println!("Using device: {:?}\n", device);

    // Load model
    println!("Loading model...");
    let start = std::time::Instant::now();
    let inner_model = ModelWeights::from_gguf_by_path(&model_path, &device)?;

    // Wrap in BatchedInference for parallel batching support
    let inv_freq = inner_model
        .rope_inv_freq()
        .ok_or_else(|| candle::Error::Msg("Model has no RoPE inv_freq".into()))?;
    let model = BatchedInference::new_with_inv_freq(inner_model, inv_freq, 4096, &device)?;
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f64());

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

    // Create KV caches for two independent sequences using the inner model
    let mut seq1_caches = model.model().create_kv_caches(512);
    let mut seq2_caches = model.model().create_kv_caches(512);

    // Phase 1: Prefill with real questions (different lengths)
    println!("\nPhase 1: Prefill (processing longer prompts for better performance metrics)");

    // Qwen2 chat format with longer, more complex prompts
    let prompt1 = "<|im_start|>user\nYou are a helpful AI assistant. Please provide a comprehensive explanation of quantum computing. What are quantum bits? How do quantum gates work? What are the potential applications?<|im_end|>\n<|im_start|>assistant\n";
    let seq1_prompt = tokenizer
        .encode(prompt1, true)
        .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    let prompt2 = "<|im_start|>user\nExplain the theory of evolution in detail. Include information about natural selection, adaptation, and the evidence for evolution. How has this theory shaped our understanding of biology?<|im_end|>\n<|im_start|>assistant\n";
    let seq2_prompt = tokenizer
        .encode(prompt2, true)
        .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    println!("  Seq 1: \"{}\" ({} tokens)", prompt1, seq1_prompt.len());
    println!("  Seq 2: \"{}\" ({} tokens)\n", prompt2, seq2_prompt.len());

    // Prefill: Pad both sequences to the same length and process together in ONE batch
    // This ensures they share the same ChunkedKvBacking for efficient batched decode later
    let max_prompt_len = seq1_prompt.len().max(seq2_prompt.len());

    // Pad seq1 to max length (pad with zeros on the left, but we'll use the actual lengths)
    let seq1_padded: Vec<u32> = {
        let mut v = vec![0u32; max_prompt_len - seq1_prompt.len()];
        v.extend(seq1_prompt.iter().copied());
        v
    };
    let seq2_padded: Vec<u32> = {
        let mut v = vec![0u32; max_prompt_len - seq2_prompt.len()];
        v.extend(seq2_prompt.iter().copied());
        v
    };

    // Create input tensors
    let seq1_input = Tensor::new(&seq1_padded[..], &device)?
        .unsqueeze(0)?
        .contiguous()?;
    let seq2_input = Tensor::new(&seq2_padded[..], &device)?
        .unsqueeze(0)?
        .contiguous()?;

    // Process BOTH sequences together in a single batched prefill
    // This creates a shared ChunkedKvBacking for both caches
    let (seq1_first_token, seq2_first_token) = {
        let seq1_offset = seq1_caches.current_seq_len();
        let seq2_offset = seq2_caches.current_seq_len();
        let mut contexts = vec![
            SequenceContext {
                kv_caches: &mut seq1_caches,
                offset: seq1_offset,
                input_ids: &seq1_input,
                input_len: max_prompt_len,
            },
            SequenceContext {
                kv_caches: &mut seq2_caches,
                offset: seq2_offset,
                input_ids: &seq2_input,
                input_len: max_prompt_len,
            },
        ];
        let offsets: Vec<usize> = contexts.iter().map(|c| c.offset).collect();
        let meta = BatchedPrefillMeta::new(&offsets, max_prompt_len, &device)?;
        let generation = stager.begin_generation();
        let outputs =
            model.forward_batch(&mut contexts, &generation, DecodeHeaders::Prefill(meta))?;
        // Sample from prefill output (last position gives next token prediction)
        let token1 = sample_token(&outputs.get(0)?)?;
        let token2 = sample_token(&outputs.get(1)?)?;
        (token1, token2)
    };
    println!("  Seq 1: cached with input_len={}", seq1_prompt.len());
    println!("  Seq 2: cached with input_len={}\n", seq2_prompt.len());

    // Track generated tokens
    let seq1_prompt_len = seq1_prompt.len();
    let seq2_prompt_len = seq2_prompt.len();

    let mut seq1_tokens = seq1_prompt.clone();
    seq1_tokens.push(seq1_first_token);
    let mut seq2_tokens = seq2_prompt.clone();
    seq2_tokens.push(seq2_first_token);

    // Phase 2: Parallel generation with BATCHED single-token processing
    println!("Phase 2: Parallel generation with BATCHED single-token processing (100 steps)");
    println!("  This phase uses forward_batch for each step instead of individual forwards\n");

    let num_steps = 100;
    let mut phase2_tokens1 = vec![seq1_first_token];
    let mut phase2_tokens2 = vec![seq2_first_token];

    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_offset = seq1_caches.current_seq_len();
        let seq2_offset = seq2_caches.current_seq_len();

        let seq1_input = Tensor::new(&[phase2_tokens1[step]], &device)?.unsqueeze(0)?;
        let seq2_input = Tensor::new(&[phase2_tokens2[step]], &device)?.unsqueeze(0)?;

        // Use SequenceContext for batched forward
        let mut contexts = vec![
            SequenceContext {
                kv_caches: &mut seq1_caches,
                offset: seq1_offset,
                input_ids: &seq1_input,
                input_len: 1,
            },
            SequenceContext {
                kv_caches: &mut seq2_caches,
                offset: seq2_offset,
                input_ids: &seq2_input,
                input_len: 1,
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

        let next_token1 = sample_token(&outputs.get(0)?)?;
        let next_token2 = sample_token(&outputs.get(1)?)?;

        phase2_tokens1.push(next_token1);
        phase2_tokens2.push(next_token2);
        seq1_tokens.push(next_token1);
        seq2_tokens.push(next_token2);

        if (step + 1) % 25 == 0 {
            println!(
                "  Step {:3}: Seq1 offset={:3} token={:5} | Seq2 offset={:3} token={:5}",
                step + 1,
                seq1_offset,
                next_token1,
                seq2_offset,
                next_token2
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
    println!("  Parallel:   forward_batch() for both sequences together\n");

    let mut seq1_caches_seq = model.model().create_kv_caches(512);
    let mut seq2_caches_seq = model.model().create_kv_caches(512);

    let seq1_input = Tensor::new(&seq1_prompt[..], &device)?.unsqueeze(0)?;
    let seq2_input = Tensor::new(&seq2_prompt[..], &device)?.unsqueeze(0)?;
    // Use inner model for sequential single-sequence calls
    let _ = model
        .model()
        .forward(&mut seq1_caches_seq, &seq1_input, 0)?;
    let _ = model
        .model()
        .forward(&mut seq2_caches_seq, &seq2_input, 0)?;

    // Sequential single-token generation for num_steps
    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_offset = seq1_caches_seq.current_seq_len();
        let seq2_offset = seq2_caches_seq.current_seq_len();

        let seq1_input =
            Tensor::new(&[generated_seq1[seq1_prompt_len + step]], &device)?.unsqueeze(0)?;
        let seq2_input =
            Tensor::new(&[generated_seq2[seq2_prompt_len + step]], &device)?.unsqueeze(0)?;

        let _ = model
            .model()
            .forward(&mut seq1_caches_seq, &seq1_input, seq1_offset)?;
        let _ = model
            .model()
            .forward(&mut seq2_caches_seq, &seq2_input, seq2_offset)?;
    }
    let sequential_time = start.elapsed();

    // Parallel implementation
    let mut seq1_caches_par = model.model().create_kv_caches(512);
    let mut seq2_caches_par = model.model().create_kv_caches(512);
    let _ = model
        .model()
        .forward(&mut seq1_caches_par, &seq1_input, 0)?;
    let _ = model
        .model()
        .forward(&mut seq2_caches_par, &seq2_input, 0)?;

    // Parallel single-token batched generation for num_steps
    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_offset = seq1_caches_par.current_seq_len();
        let seq2_offset = seq2_caches_par.current_seq_len();

        let seq1_input =
            Tensor::new(&[generated_seq1[seq1_prompt_len + step]], &device)?.unsqueeze(0)?;
        let seq2_input =
            Tensor::new(&[generated_seq2[seq2_prompt_len + step]], &device)?.unsqueeze(0)?;

        let mut contexts = vec![
            SequenceContext {
                kv_caches: &mut seq1_caches_par,
                offset: seq1_offset,
                input_ids: &seq1_input,
                input_len: 1,
            },
            SequenceContext {
                kv_caches: &mut seq2_caches_par,
                offset: seq2_offset,
                input_ids: &seq2_input,
                input_len: 1,
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
    let parallel_time = start.elapsed();

    println!(
        "  Sequential: {:.2}ms",
        sequential_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Parallel:   {:.2}ms",
        parallel_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Speedup:    {:.2}x\n",
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    // Summary with actual generated text
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

    println!("\nSequence 2 ({} tokens):", seq2_tokens.len());
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
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    Ok(())
}

/// Simple greedy sampling - picks token with highest logit
fn sample_token(logits: &Tensor) -> Result<u32> {
    // Squeeze the batch dimension (assuming shape is [1, vocab_size])
    let squeezed = if logits.rank() == 2 {
        logits.squeeze(0)?
    } else {
        logits.clone()
    };

    let logits_vec = squeezed.to_vec1::<f32>()?;

    if logits_vec.is_empty() {
        return Ok(0);
    }

    let max_idx = logits_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    Ok(max_idx as u32)
}
