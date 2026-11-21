//! KV-cache dump tooling for offline selection analysis.
//!
//! `test_dump_kv_cache_data` runs one R16 session over a Llama-3.2-3B chat prompt
//! followed by 40 decode steps, so the kernels fill the Q-capture space, and writes
//! K, V and Q for every layer in the v4 binary format. The output lands in the
//! chunked KV cache's test data folder. `test_r16_vs_f16_logits_comparison` compares
//! R16 and F16 logits at each step to find where the two first diverge, and
//! `print_top_k` prints the top logits of a `[1, vocab]` tensor.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    #![allow(
        clippy::cloned_ref_to_slice_refs,
        clippy::type_complexity,
        clippy::useless_vec
    )]

    use super::*;

    use crate::models::batch_test::utils::TestParams;

    use crate::models::batched_inference::InferenceMode;

    use crate::models::dialect::Dialect;

    #[allow(unused_imports)]
    use candle_nn::kv_cache::CacheIntegrityResult;

    use super::super::tests_gates::tests::{batched, llama3_2_3b_path, LLAMA3_CONTEXT};

    /// KV-cache dump infrastructure for offline selection analysis.
    ///
    /// Run tests in this module with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_llama::tests::kv_dump -- --ignored --nocapture
    pub(in super::super) mod kv_dump {
        use super::*;
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use std::io::Write;

        /// Dump real KV cache data (K, V, Q) from Llama-3.2-3B for offline analysis.
        ///
        /// Runs a single R16 session using the Llama-3 chat-template system+user
        /// prompt (matching what the gated test sends for context index 0), followed
        /// by 40 decode steps.  R16 mode keeps K in raw F16 with Q-capture space, so
        /// the prefill/decode kernels populate `block_r16->q[]` with real Q
        /// projections; the dump captures K + V + Q in v4 binary format.
        ///
        /// The output is written to:
        ///   `candle-nn/src/kv_cache/chunked/tests/data/llama-kv-data.bin`
        ///
        /// Run with:
        ///   cargo test --release --features cuda --lib --package candle-transformers \
        ///     quantized_llama::tests::kv_dump::test_dump_kv_cache_data -- --ignored --nocapture
        #[test]
        #[ignore]
        fn test_dump_kv_cache_data() -> Result<()> {
            let device = Device::new_cuda(0).map_err(|e| {
                candle::Error::Msg(format!(
                    "CUDA required for this test: {}. Use --features cuda",
                    e
                ))
            })?;
            println!("Using device: {:?}", device);

            // Tokenizer + chat-ML prompts (matching the gated test).
            let tokenizer_json = include_str!("../quantized_llama_tokenizer.json");
            let dialect = Dialect::llama3();
            let params = TestParams::new(40, tokenizer_json, dialect)
                .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?;
            let system_tokens = params.system_prompt_tokens(0);
            let user_tokens = params.user_prompt_tokens(0);
            let mut all_tokens: Vec<u32> = Vec::new();
            all_tokens.extend_from_slice(&system_tokens);
            all_tokens.extend_from_slice(&user_tokens);
            let prefill_len = all_tokens.len();
            println!(
                "Prefill: {} system + {} user = {} tokens",
                system_tokens.len(),
                user_tokens.len(),
                prefill_len
            );

            // Download model (same as the gated test — should already be cached).
            let model_path = llama3_2_3b_path()?;
            println!("Model path: {:?}", model_path);

            let raw = ModelWeights::from_gguf_by_path_v3(&model_path, &device)?;
            let model = batched(raw, LLAMA3_CONTEXT, &device)?;

            let n_kv_head = model.n_kv_head();
            let head_dim = model.head_dim();
            let num_layers = model.num_layers();
            println!(
                "Model: {} layers, {} kv-heads, {} head-dim",
                num_layers, n_kv_head, head_dim
            );

            // R16 session — K stored as R16 so Q values get captured during prefill / decode.
            let mode = InferenceMode::R16;
            let batch_config = BatchedConfig {
                k_format: mode.k_format(),
                v_format: mode.v_format(),
                compression_level: mode.compression_level(),
                ..Default::default()
            };
            let mut session = model.create_batched_session(batch_config)?;
            let seq_idx = session.create_sequence()?;

            // Prefill with actual prompt tokens.
            let prefill_input = Tensor::from_vec(
                all_tokens[..prefill_len].to_vec(),
                (1, prefill_len),
                &device,
            )?;
            let nl = model.num_layers();
            let logits_vec = model
                .forward_wave(
                    &mut session,
                    &[],
                    &[],
                    &[seq_idx],
                    &[prefill_input],
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits_owned()?;
            session.advance_sequence(seq_idx, prefill_len)?;
            println!("Prefill done ({} tokens)", prefill_len);

            // Decode: 40 more tokens — capture each generated token.
            let num_decode = 40usize;
            let mut last_logits = logits_vec
                .into_iter()
                .next()
                .ok_or_else(|| candle::Error::Msg("no logits from prefill".into()))?;
            for step in 0..num_decode {
                let next_token = last_logits
                    .squeeze(0)?
                    .argmax(candle::D::Minus1)?
                    .to_scalar::<u32>()?;
                all_tokens.push(next_token);
                let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                let nl = model.num_layers();
                let out = model
                    .forward_wave(
                        &mut session,
                        &[seq_idx],
                        &[input],
                        &[],
                        &[],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?
                    .logits_owned()?;
                session.advance_sequence(seq_idx, 1)?;
                last_logits = out
                    .into_iter()
                    .next()
                    .ok_or_else(|| candle::Error::Msg("no logits from decode".into()))?;
                if (step + 1) % 10 == 0 {
                    println!("  Decoded {}/{} tokens", step + 1, num_decode);
                }
            }

            let total_tokens = prefill_len + num_decode;
            println!(
                "Generated tokens (first 16): {:?}",
                &all_tokens[..all_tokens.len().min(16)]
            );
            println!("Session complete: {} tokens", total_tokens);

            // Dump all layers — R16 path returns (block_idx, k, v, q).
            let backings = session.backings();
            let mut layer_dumps: Vec<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> =
                Vec::with_capacity(num_layers);
            for (layer_idx, backing) in backings.iter().enumerate() {
                let chunks = backing.dump_sequence_r16_kv_chunks(seq_idx, None)?;
                println!("  Layer {:2}: {} R16 chunks", layer_idx, chunks.len());
                layer_dumps.push(chunks);
            }

            // Compute output path relative to workspace root.
            let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("candle-nn/src/kv_cache/chunked/tests/data");
            std::fs::create_dir_all(&out_dir).map_err(|e| {
                candle::Error::Msg(format!("Failed to create output dir {:?}: {}", out_dir, e))
            })?;
            let bin_path = out_dir.join("llama-kv-data.bin");

            // Binary format v4:
            //   header: magic[8] + version:u32 + num_layers:u32 + n_kv_head:u32
            //           + chunk_size:u32 + head_dim:u32
            //           + num_tokens:u32 + tokens:[u32; num_tokens]
            //   per layer: num_chunks:u32,
            //     per chunk: block_idx:u32 + token_start:u32
            //                + k_data:[f32; elems] + v_data:[f32; elems] + q_data:[f32; elems]
            let mut file = std::fs::File::create(&bin_path).map_err(|e| {
                candle::Error::Msg(format!("Failed to create {:?}: {}", bin_path, e))
            })?;
            file.write_all(b"KVDUMP\0\0")?;
            file.write_all(&4u32.to_le_bytes())?;
            file.write_all(&(num_layers as u32).to_le_bytes())?;
            file.write_all(&(n_kv_head as u32).to_le_bytes())?;
            file.write_all(&(candle_nn::CHUNK_SIZE as u32).to_le_bytes())?;
            file.write_all(&(head_dim as u32).to_le_bytes())?;
            file.write_all(&(all_tokens.len() as u32).to_le_bytes())?;
            for &t in &all_tokens {
                file.write_all(&t.to_le_bytes())?;
            }

            let chunk_size = candle_nn::CHUNK_SIZE;
            let mut total_chunks = 0usize;
            for chunks in &layer_dumps {
                file.write_all(&(chunks.len() as u32).to_le_bytes())?;
                for (block_idx, k_data, v_data, q_data) in chunks {
                    let token_start = block_idx * chunk_size;
                    file.write_all(&(*block_idx as u32).to_le_bytes())?;
                    file.write_all(&(token_start as u32).to_le_bytes())?;
                    for &v in k_data {
                        file.write_all(&v.to_le_bytes())?;
                    }
                    for &v in v_data {
                        file.write_all(&v.to_le_bytes())?;
                    }
                    for &v in q_data {
                        file.write_all(&v.to_le_bytes())?;
                    }
                    total_chunks += 1;
                }
            }

            println!(
                "\nDumped {} total chunks (K+V+Q v4) across {} layers to {:?}",
                total_chunks, num_layers, bin_path
            );
            let meta = std::fs::metadata(&bin_path)?;
            println!("File size: {:.1} MB", meta.len() as f64 / (1024.0 * 1024.0));

            Ok(())
        }
    }

    // mod kv_dump

    /// R16 vs F16 diagnostic: compare logits at every step to isolate where divergence starts.
    ///
    /// Run with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_llama::tests::test_r16_vs_f16_logits_comparison -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_r16_vs_f16_logits_comparison() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};

        let device =
            Device::new_cuda(0).map_err(|e| candle::Error::Msg(format!("CUDA required: {}", e)))?;

        // Download model
        let model_path = llama3_2_3b_path()?;

        let raw = ModelWeights::from_gguf_by_path_v3(&model_path, &device)?;
        let model = batched(raw, LLAMA3_CONTEXT, &device)?;
        println!(
            "Model loaded: {} layers, {} kv-heads, hdim={}",
            model.num_layers(),
            model.n_kv_head(),
            model.head_dim()
        );

        // F16 session
        let f16_config = BatchedConfig::default().with_dtype(DType::F16);
        let mut f16_session = model.create_batched_session(f16_config)?;
        let f16_seq = f16_session.create_sequence()?;

        // R16 session
        let r16_config = BatchedConfig {
            k_format: InferenceMode::R16.k_format(),
            v_format: InferenceMode::R16.v_format(),
            compression_level: None,
            ..Default::default()
        };
        let mut r16_session = model.create_batched_session(r16_config)?;
        let r16_seq = r16_session.create_sequence()?;

        // Fixed prefill tokens (same as the gated test StoryRewrite prompt-ish)
        // Use a short recognisable prefix so we can trace outputs.
        let prefill_tokens: Vec<u32> = vec![
            128000, 128006, 9125, 128007,
            271, // <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
            2675, 527, 264, 11190, 18328, 13, // You are a helpful assistant.
            128009, 128006, 882, 128007,
            271, // <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
            36227, 757, 264, 2875, 3446, 922, // Tell me a short story about
            264, 33671, 889, 33095, 279, 18566, // a dragon who guards the castle
            128009, 128006, 78191, 128007,
            271, // <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        ];
        let prefill_len = prefill_tokens.len();
        println!("\n=== Prefill: {} tokens ===", prefill_len);

        let prefill_input = Tensor::from_vec(prefill_tokens.clone(), (1, prefill_len), &device)?;

        // Prefill both
        let nl = model.num_layers();
        let f16_logits = model
            .forward_wave(
                &mut f16_session,
                &[],
                &[],
                &[f16_seq],
                &[prefill_input.clone()],
                &[],
                &[],
                0,
                nl,
                None,
            )?
            .logits_owned()?;
        f16_session.advance_sequence(f16_seq, prefill_len)?;

        let r16_logits = model
            .forward_wave(
                &mut r16_session,
                &[],
                &[],
                &[r16_seq],
                &[prefill_input],
                &[],
                &[],
                0,
                nl,
                None,
            )?
            .logits_owned()?;
        r16_session.advance_sequence(r16_seq, prefill_len)?;

        let f16_l = &f16_logits[0]; // [1, vocab]
        let r16_l = &r16_logits[0];

        // Compare prefill logits
        let diff = (f16_l - r16_l)?.abs()?.to_dtype(DType::F32)?;
        let max_diff = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let mean_diff = diff.flatten_all()?.mean_all()?.to_scalar::<f32>()?;

        let f16_argmax = f16_l
            .squeeze(0)?
            .argmax(candle::D::Minus1)?
            .to_scalar::<u32>()?;
        let r16_argmax = r16_l
            .squeeze(0)?
            .argmax(candle::D::Minus1)?
            .to_scalar::<u32>()?;

        let prefill_match = f16_argmax == r16_argmax;
        println!(
            "  Prefill logits: max_diff={:.6e}  mean_diff={:.6e}  f16_argmax={}  r16_argmax={}  {}",
            max_diff,
            mean_diff,
            f16_argmax,
            r16_argmax,
            if prefill_match { "MATCH" } else { "DIVERGE" }
        );

        // Top-5 comparison for prefill
        print_top_k("  F16 prefill top-5", f16_l, 5)?;
        print_top_k("  R16 prefill top-5", r16_l, 5)?;

        // Decode steps: always use F16 argmax as the next token (so both see same input)
        let num_decode = 15;
        let mut all_match = prefill_match;
        let mut first_diverge_step: Option<usize> = if !prefill_match { Some(0) } else { None };
        let mut next_token = f16_argmax;

        println!(
            "\n=== Decode: {} steps (using F16 argmax as input for both) ===",
            num_decode
        );
        for step in 0..num_decode {
            let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;

            let nl = model.num_layers();
            let f16_out = model
                .forward_wave(
                    &mut f16_session,
                    &[f16_seq],
                    &[input.clone()],
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits_owned()?;
            f16_session.advance_sequence(f16_seq, 1)?;

            let r16_out = model
                .forward_wave(
                    &mut r16_session,
                    &[r16_seq],
                    &[input],
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )?
                .logits_owned()?;
            r16_session.advance_sequence(r16_seq, 1)?;

            let f16_l = &f16_out[0];
            let r16_l = &r16_out[0];

            let diff = (f16_l - r16_l)?.abs()?.to_dtype(DType::F32)?;
            let max_d = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            let mean_d = diff.flatten_all()?.mean_all()?.to_scalar::<f32>()?;

            let f16_am = f16_l
                .squeeze(0)?
                .argmax(candle::D::Minus1)?
                .to_scalar::<u32>()?;
            let r16_am = r16_l
                .squeeze(0)?
                .argmax(candle::D::Minus1)?
                .to_scalar::<u32>()?;

            let step_match = f16_am == r16_am;
            if !step_match && first_diverge_step.is_none() {
                first_diverge_step = Some(step + 1);
            }
            if !step_match {
                all_match = false;
            }

            println!(
                "  step {:2}: max_diff={:.6e}  mean_diff={:.6e}  f16={:6}  r16={:6}  {}",
                step + 1,
                max_d,
                mean_d,
                f16_am,
                r16_am,
                if step_match { "MATCH" } else { "DIVERGE <<<" }
            );

            // If argmax diverged, also show top-5 for this step
            if !step_match {
                print_top_k("    F16 top-5", f16_l, 5)?;
                print_top_k("    R16 top-5", r16_l, 5)?;
            }

            next_token = f16_am; // always use F16's choice
        }

        println!("\n=== Summary ===");
        println!("  Prefill match: {}", prefill_match);
        if let Some(s) = first_diverge_step {
            if s == 0 {
                println!("  First divergence: PREFILL (before any decode)");
                println!("  Diagnosis: BUG is in prefill K-write to R16 or prefill attention read from R16");
            } else {
                println!("  First divergence: decode step {}", s);
                println!("  Diagnosis: prefill path is OK; BUG is in decode scatter-write or decode attention read from R16");
            }
        } else {
            println!(
                "  All {} decode steps match! R16 and F16 are equivalent.",
                num_decode
            );
        }

        if !all_match {
            candle::bail!("R16 diverged from F16 — see above for diagnosis");
        }

        Ok(())
    }

    /// Print top-K token IDs and logit values from a [1, vocab] tensor.
    pub(in super::super) fn print_top_k(label: &str, logits: &Tensor, k: usize) -> Result<()> {
        let flat = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let vals: Vec<f32> = flat.to_vec1()?;
        let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = indexed[..k.min(indexed.len())]
            .iter()
            .map(|(i, v)| format!("{}={:.4}", i, v))
            .collect();
        println!("{}: [{}]", label, top.join(", "));
        Ok(())
    }
}
