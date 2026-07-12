/// Parallel Continuous Batching Example for Llama
use candle::quantized::pinned_staging::PinnedStager;
use candle::{Device, Result, Tensor};
use candle_transformers::models::batched_layer::{BatchedPrefillMeta, DecodeHeaders};
use candle_transformers::models::batched_model::BatchedInference;
use candle_transformers::models::kv_cache_utils::SequenceContext;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    println!("\n=== Llama Parallel Continuous Batching Example ===\n");

    // Download model
    let api = Api::new().map_err(|e| candle::Error::Msg(format!("HF API error: {}", e)))?;
    let repo = api.repo(Repo::with_revision(
        "bartowski/Llama-3.2-1B-Instruct-GGUF".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    println!("Downloading Llama-3.2-1B-Instruct model...");
    let model_path = repo
        .get("Llama-3.2-1B-Instruct-Q4_K_M.gguf")
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

    // Load tokenizer (using NousResearch tokenizer which is publicly accessible)
    println!("Loading tokenizer...");
    let tokenizer_repo = api.model("NousResearch/Hermes-3-Llama-3.1-8B".to_string());
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

    // Llama chat format with MUCH longer, complex prompts for better GPU measurement
    let prompt1 = "<|start_header_id|>user<|end_header_id|>\n\nYou are a world-class computer science instructor. Please provide an exhaustive, comprehensive explanation of machine learning, covering all fundamental concepts in detail. Discuss supervised learning, unsupervised learning, and reinforcement learning. Explain how decision trees work, including information gain and entropy calculations. Describe neural networks in depth: what are neurons, weights, biases, activation functions, backpropagation, gradient descent, learning rates, and convergence. What are convolutional neural networks and how do they apply to image processing? Explain recurrent neural networks, LSTMs, transformers, and attention mechanisms. Cover practical applications in healthcare, finance, autonomous vehicles, natural language processing, computer vision, and recommendation systems. What are the current challenges in machine learning including data quality, overfitting, underfitting, regularization techniques, hyperparameter tuning, cross-validation, and model evaluation metrics? Discuss ethical considerations in AI and machine learning.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let seq1_prompt = tokenizer
        .encode(prompt1, true)
        .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    let prompt2 = "<|start_header_id|>user<|end_header_id|>\n\nExplain the complete water cycle and climate system in great detail. Start with the basic concept of the water cycle and its importance to life on Earth. Describe evaporation: the process by which water transforms from liquid to gas, including factors that affect evaporation rates such as temperature, humidity, wind, and solar radiation. Explain transpiration from plants and how it contributes to the total water movement. Discuss condensation: how water vapor transforms back into liquid form, the role of condensation nuclei, cloud formation, and the different types of clouds (cumulus, stratus, cirrus). Describe precipitation in all its forms: rain, snow, sleet, and hail. Explain collection and infiltration: how water is stored in oceans, lakes, rivers, groundwater, and ice caps. Discuss how disruptions to the water cycle affect weather patterns, climate, and living organisms. Include information about groundwater aquifers, the hydrological balance, and human impacts on the water cycle through pollution and climate change.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let seq2_prompt = tokenizer
        .encode(prompt2, true)
        .map_err(|e| candle::Error::Msg(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    println!("  Seq 1: \"{}\" ({} tokens)", prompt1, seq1_prompt.len());
    println!("  Seq 2: \"{}\" ({} tokens)\n", prompt2, seq2_prompt.len());

    // Prefill: process each prompt independently with variable input_len
    let seq1_input = Tensor::new(&seq1_prompt[..], &device)?
        .unsqueeze(0)?
        .contiguous()?;
    let seq1_offset = seq1_caches.current_seq_len();
    let mut contexts = vec![SequenceContext {
        kv_caches: &mut seq1_caches,
        offset: seq1_offset,
        input_ids: &seq1_input,
        input_len: seq1_prompt.len(),
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
    }];
    let offsets2: Vec<usize> = contexts.iter().map(|c| c.offset).collect();
    let seq_len2 = contexts[0].input_len;
    let meta2 = BatchedPrefillMeta::new(&offsets2, seq_len2, &device)?;
    let generation = stager.begin_generation();
    let _outputs =
        model.forward_batch(&mut contexts, &generation, DecodeHeaders::Prefill(meta2))?;
    println!("  Seq 2: cached with input_len={}\n", seq2_prompt.len());

    // Get initial tokens for generation
    let seq1_len = seq1_caches.current_seq_len();
    let seq2_len = seq2_caches.current_seq_len();

    // Use inner model for single-sequence forward calls
    let seq1_logits = model.model().forward(
        &mut seq1_caches,
        &Tensor::new(&[seq1_prompt[seq1_prompt.len() - 1]], &device)?
            .unsqueeze(0)?
            .contiguous()?,
        seq1_len - 1,
    )?;
    let seq1_next = {
        let idx = seq1_logits.argmax(seq1_logits.rank() - 1)?;
        if idx.rank() == 1 {
            idx.squeeze(0)?.to_scalar::<u32>()?
        } else {
            idx.to_scalar::<u32>()?
        }
    };

    let seq2_logits = model.model().forward(
        &mut seq2_caches,
        &Tensor::new(&[seq2_prompt[seq2_prompt.len() - 1]], &device)?
            .unsqueeze(0)?
            .contiguous()?,
        seq2_len - 1,
    )?;
    let seq2_next = {
        let idx = seq2_logits.argmax(seq2_logits.rank() - 1)?;
        if idx.rank() == 1 {
            idx.squeeze(0)?.to_scalar::<u32>()?
        } else {
            idx.to_scalar::<u32>()?
        }
    };

    // TEST 1: Offset seq2 by one position to validate misaligned batching
    println!("\nTEST 1: Position misalignment validation");
    println!("  Offsetting Seq 2 by 1 position for misaligned batch testing");

    let seq2_offset_input = Tensor::new(&[seq2_next], &device)?.unsqueeze(0)?;
    {
        let seq2_offset_val = seq2_caches.current_seq_len();
        let mut context = vec![SequenceContext {
            kv_caches: &mut seq2_caches,
            offset: seq2_offset_val,
            input_ids: &seq2_offset_input,
            input_len: 1,
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
        "  Seq 1 position: {} | Seq 2 position: {} (misaligned)",
        seq1_caches.current_seq_len(),
        seq2_caches.current_seq_len()
    );

    // Phase 2: Parallel Generation with Batched Single-Token Processing
    println!("\nPhase 2: Parallel generation with BATCHED single-token processing (100 steps)");
    println!("  This phase uses forward_batch for each step instead of individual forwards\n");

    let num_steps = 100;
    let mut seq1_tokens = seq1_prompt.clone();
    seq1_tokens.push(seq1_next);
    let mut seq2_tokens = seq2_prompt.clone();
    seq2_tokens.push(seq2_next);

    let start_batch = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_offset = seq1_caches.current_seq_len();
        let seq2_offset = seq2_caches.current_seq_len();

        let seq1_input =
            Tensor::new(&[seq1_tokens[seq1_tokens.len() - 1]], &device)?.unsqueeze(0)?;
        let seq2_input =
            Tensor::new(&[seq2_tokens[seq2_tokens.len() - 1]], &device)?.unsqueeze(0)?;

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

        let next_token1 = outputs.get(0)?.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
        let next_token2 = outputs.get(1)?.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;

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
    let batched_time = start_batch.elapsed();
    println!(
        "  Batched processing time: {:.2}ms\n",
        batched_time.as_secs_f64() * 1000.0
    );

    // Phase 3: Performance Comparison
    println!("\nPhase 3: Performance comparison");
    println!("  Sequential: individual forward() calls for each sequence");
    println!("  Parallel:   forward_batch() for both sequences together");
    println!("  Both use misaligned offsets (Seq 2 ahead by 1 position)\n");

    // Replay with fresh caches, same token sequences
    let generated_seq1 = seq1_tokens.clone();
    let generated_seq2 = seq2_tokens.clone();

    // Sequential implementation
    let mut seq1_caches_seq = model.model().create_kv_caches(512);
    let mut seq2_caches_seq = model.model().create_kv_caches(512);

    let seq1_input = Tensor::new(&seq1_prompt[..], &device)?.unsqueeze(0)?;
    let seq2_input = Tensor::new(&seq2_prompt[..], &device)?.unsqueeze(0)?;
    let _ = model
        .model()
        .forward(&mut seq1_caches_seq, &seq1_input, 0)?;
    let _ = model
        .model()
        .forward(&mut seq2_caches_seq, &seq2_input, 0)?;

    // Apply the offset to seq2_seq just like in the test
    let seq2_offset_for_seq =
        Tensor::new(&[generated_seq2[seq1_prompt.len()]], &device)?.unsqueeze(0)?;
    let seq2_offset_for_seq_val = seq2_caches_seq.current_seq_len();
    let _ = model.model().forward(
        &mut seq2_caches_seq,
        &seq2_offset_for_seq,
        seq2_offset_for_seq_val,
    )?;

    // Sequential single-token generation for num_steps
    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_offset = seq1_caches_seq.current_seq_len();
        let seq2_offset = seq2_caches_seq.current_seq_len();

        let seq1_input =
            Tensor::new(&[generated_seq1[seq1_prompt.len() + step]], &device)?.unsqueeze(0)?;
        let seq2_input =
            Tensor::new(&[generated_seq2[seq2_prompt.len() + 1 + step]], &device)?.unsqueeze(0)?;

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

    // Apply the offset to seq2_par just like in the test
    let seq2_offset_for_par =
        Tensor::new(&[generated_seq2[seq2_prompt.len()]], &device)?.unsqueeze(0)?;
    {
        let seq2_offset_val = seq2_caches_par.current_seq_len();
        let mut context = vec![SequenceContext {
            kv_caches: &mut seq2_caches_par,
            offset: seq2_offset_val,
            input_ids: &seq2_offset_for_par,
            input_len: 1,
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

    // Parallel single-token batched generation for num_steps
    let start = std::time::Instant::now();
    for step in 0..num_steps {
        let seq1_offset = seq1_caches_par.current_seq_len();
        let seq2_offset = seq2_caches_par.current_seq_len();

        let seq1_input =
            Tensor::new(&[generated_seq1[seq1_prompt.len() + step]], &device)?.unsqueeze(0)?;
        let seq2_input =
            Tensor::new(&[generated_seq2[seq2_prompt.len() + 1 + step]], &device)?.unsqueeze(0)?;

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
        "  Speedup:    {:.2}x",
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    // Summary with actual generated text
    println!("\n=== Generated Output ===\n");

    // Get the full decoded text
    let full_response1 = tokenizer
        .decode(&seq1_tokens, false)
        .map_err(|e| candle::Error::Msg(format!("Decode failed: {}", e)))?;
    let full_response2 = tokenizer
        .decode(&seq2_tokens, false)
        .map_err(|e| candle::Error::Msg(format!("Decode failed: {}", e)))?;

    // Extract just the response part (after "assistant<|end_header_id|>\n\n")
    let resp1 = if let Some(pos) = full_response1.rfind("assistant<|end_header_id|>") {
        &full_response1[pos + 26..]
    } else {
        &full_response1
    };

    let resp2 = if let Some(pos) = full_response2.rfind("assistant<|end_header_id|>") {
        &full_response2[pos + 26..]
    } else {
        &full_response2
    };

    println!("Sequence 1 ({} tokens):", seq1_tokens.len());
    println!("{}\n", resp1.trim());

    println!("Sequence 2 ({} tokens):", seq2_tokens.len());
    println!("{}\n", resp2.trim());

    println!("=== Summary ===");
    println!(
        "Processed 2 sequences in parallel for {} generation steps",
        num_steps
    );
    println!("Sequence 1: {} total tokens", seq1_caches.current_seq_len());
    println!("Sequence 2: {} total tokens", seq2_caches.current_seq_len());
    println!(
        "Speedup: {:.2}x (parallel vs sequential)",
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    Ok(())
}
