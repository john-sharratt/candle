//! Ignored test that dumps the real K, V and Q cache data of Qwen3-30B-A3B for
//! offline selection analysis.
//!
//! It runs one R16 session on the same ChatML prompt as the gated test, followed
//! by 40 greedy decode steps, and writes every layer's K, V and Q in the v4
//! binary format. R16 keeps K in raw F16 with room to capture Q, so the Q
//! projections are recorded alongside the cache.

use super::*;

// Tests

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_question_mark, clippy::type_complexity)]

    use super::*;

    use crate::models::batch_test::utils::TestParams;

    use crate::models::batched_inference::InferenceMode;

    use crate::models::dialect::Dialect;

    use super::super::tests_gates::tests::{batched, instruct_2507_pack, INSTRUCT_2507_CONTEXT};

    /// KV-cache dump for offline selection analysis (Qwen3-MoE).
    ///
    /// Run with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_qwen3_moe::tests::kv_dump::test_dump_kv_cache_data -- --ignored --nocapture
    pub(in super::super) mod kv_dump {
        use super::*;
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use std::io::Write;

        /// Dump real KV cache data (K, V, Q) from Qwen3-30B-A3B MoE.
        ///
        /// Runs a single R16 session using the same ChatML system+user prompt as the
        /// gated test (StoryRewrite), followed by 40 decode steps.  R16 mode keeps K
        /// in raw F16 with Q-capture space, so the prefill/decode kernels populate
        /// `block_r16->q[]` with real Q projections; the dump captures K + V + Q in
        /// v4 binary format for offline selection analysis.
        ///
        /// Output: `candle-nn/src/kv_cache/chunked/tests/data/qwen3-kv-data.bin`
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

            // Download tokenizer
            let api = crate::models::batch_test::test_helpers::api()
                .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;
            let tok_repo = api.model("Qwen/Qwen3-30B-A3B-Instruct-2507".to_string());
            let tokenizer_path = tok_repo
                .get("tokenizer.json")
                .map_err(|e| candle::Error::Msg(format!("Failed to download tokenizer: {}", e)))?;
            let tokenizer_json = std::fs::read_to_string(&tokenizer_path)
                .map_err(|e| candle::Error::Msg(format!("Failed to read tokenizer: {}", e)))?;

            let dialect = Dialect::chat_ml();
            let params = TestParams::new(10, &tokenizer_json, dialect)
                .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
                .with_suppress_thinking(true);

            // Build the full ChatML prompt: system + user + assistant header
            // This matches what the gated test sends for context index 0.
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

            let pack = instruct_2507_pack(&device, None)?;
            println!("Model pack: {:?}", pack);

            let raw = ModelWeights::from_pack(&pack, &device, None)?;
            let model = batched(raw, INSTRUCT_2507_CONTEXT, 4096, &device)?;

            let n_kv_head = model.n_kv_head();
            let head_dim = model.head_dim();
            let num_layers = model.num_layers();
            println!(
                "Model: {} layers, {} kv-heads, {} head-dim",
                num_layers, n_kv_head, head_dim
            );

            // Create an R16 session — K stored as R16 so Q values get captured in
            // each block_r16->q[] field during prefill / decode.
            let mode = InferenceMode::R16;
            let batch_config = BatchedConfig {
                k_format: mode.k_format(),
                v_format: mode.v_format(),
                compression_level: mode.compression_level(),
                ..Default::default()
            };
            let mut session = model.create_batched_session(batch_config)?;
            let seq_idx = session.create_sequence()?;

            // Prefill with actual prompt tokens
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

            // Decode: 40 tokens (greedy argmax) — matching production generate_token_count
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
            println!("Session complete: {} total tokens", total_tokens);

            // Dump all layers — R16 path returns (block_idx, k, v, q).
            let backings = session.backings();
            let mut layer_dumps: Vec<Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>> =
                Vec::with_capacity(num_layers);
            for (layer_idx, backing) in backings.iter().enumerate() {
                let chunks = backing.dump_sequence_r16_kv_chunks(seq_idx, None)?;
                println!("  Layer {:2}: {} R16 chunks", layer_idx, chunks.len());
                layer_dumps.push(chunks);
            }

            // Output path
            let out_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("candle-nn/src/kv_cache/chunked/tests/data");
            std::fs::create_dir_all(&out_dir).map_err(|e| {
                candle::Error::Msg(format!("Failed to create output dir {:?}: {}", out_dir, e))
            })?;
            let bin_path = out_dir.join("qwen3-kv-data.bin");

            // Binary format v4 — header + tokens + per-chunk (block_idx, token_start, k, v, q).
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
}
