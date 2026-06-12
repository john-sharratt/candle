//! Shared test infrastructure for batched sampling kernel tests.
//!
//! Provides CPU reference implementation, GPU test harness, `SamplingParams`,
//! multi-dtype helpers, and utility functions used by all sampling test files.

#![allow(dead_code)]

use candle_core::cuda_backend::cudarc;
use candle_kernels::sampling::run_batched_sampling;
use core::ffi::c_void;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::Arc;

pub use candle_kernels::sampling::DType;
pub use float8::F8E4M3;
pub use half::{bf16, f16};

// ============================================================================
// CPU Reference Implementation
// ============================================================================
// Mirrors the kernel's logic exactly so we can validate GPU output.

/// Apply penalties to a logit value, matching the kernel's branchless logic.
pub fn cpu_apply_penalties(
    logit: f32,
    token_id: usize,
    batch_idx: usize,
    vocab_size: usize,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    eos_boost: f32,
    eos_token_id: i32,
    eos_ramp_start: i32,
    eos_ramp_len: i32,
    eos_boost_max_multiplier: f32,
    current_len: i32,
    eot_boost: f32,
    eot_token_id: i32,
    eot_ramp_start: i32,
    eot_ramp_len: i32,
    eot_boost_max_multiplier: f32,
    thinking_len: i32,
    cross_turn_penalty: f32,
    cross_turn_counts: &[i32],
    token_counts: &[i32],
    recent_tokens: &[i32],
    recent_lens: &[i32],
    max_recent_len: usize,
    banned_tokens: &[i32],
    banned_per_seq: i32,
) -> f32 {
    let mut v = logit;

    // Protected tokens: EOS and EOT (</think>) must never be penalised.
    // Penalties would suppress the model's ability to stop generating.
    let is_eos = token_id as i32 == eos_token_id;
    let is_eot = token_id as i32 == eot_token_id;
    let is_protected = is_eos || is_eot;
    let saved_logit = v;

    // Repeat penalty via recent token bitset
    if repeat_penalty != 1.0 && !recent_tokens.is_empty() && !recent_lens.is_empty() {
        let rlen = recent_lens[batch_idx] as usize;
        let base = batch_idx * max_recent_len;
        let mut is_recent = false;
        for i in 0..rlen {
            if recent_tokens[base + i] as usize == token_id {
                is_recent = true;
                break;
            }
        }
        if is_recent {
            if v >= 0.0 {
                v /= repeat_penalty;
            } else {
                v *= repeat_penalty;
            }
        }
    }

    // Frequency and presence penalties via token_counts
    if !token_counts.is_empty() {
        let count = token_counts[batch_idx * vocab_size + token_id];
        if count > 0 {
            v -= frequency_penalty * count as f32;
            v -= presence_penalty;
        }
    }

    // Cross-turn penalty (flat per token seen in prior turns)
    if cross_turn_penalty != 0.0 && !cross_turn_counts.is_empty() {
        let count = cross_turn_counts[batch_idx * vocab_size + token_id];
        if count > 0 {
            v -= cross_turn_penalty;
        }
    }

    // Restore logit for protected tokens (EOS, EOT) — undo all penalties above
    if is_protected {
        v = saved_logit;
    }

    // EOS boost (static or dynamic ramp)
    if token_id as i32 == eos_token_id && eos_boost != 0.0 {
        let effective_boost = if eos_ramp_len > 0 && eos_boost_max_multiplier > 0.0 {
            let ramp_span = (eos_ramp_len - eos_ramp_start).max(1) as f32;
            let t = ((current_len - eos_ramp_start).max(0) as f32 / ramp_span).min(1.0);
            eos_boost * t * eos_boost_max_multiplier
        } else {
            eos_boost
        };
        v += effective_boost;
    }

    // EOT boost (end-of-thinking ramp) — same formula as EOS but uses thinking_len
    if token_id as i32 == eot_token_id && eot_boost != 0.0 && thinking_len > 0 {
        let effective_eot = if eot_ramp_len > 0 && eot_boost_max_multiplier > 0.0 {
            let ramp_span = (eot_ramp_len - eot_ramp_start).max(1) as f32;
            let t = ((thinking_len - eot_ramp_start).max(0) as f32 / ramp_span).min(1.0);
            eot_boost * t * eot_boost_max_multiplier
        } else {
            eot_boost
        };
        v += effective_eot;
    }

    // Banned tokens
    if banned_per_seq > 0 {
        let base = batch_idx * banned_per_seq as usize;
        for i in 0..banned_per_seq as usize {
            let tok = banned_tokens[base + i];
            if tok == -1 {
                break;
            }
            if tok as usize == token_id {
                v = f32::NEG_INFINITY;
                break;
            }
        }
    } else if !banned_tokens.is_empty() {
        for &tok in banned_tokens.iter() {
            if tok == -1 {
                break;
            }
            if tok as usize == token_id {
                v = f32::NEG_INFINITY;
                break;
            }
        }
    }

    v
}

/// CPU DRY penalty computation — matches the kernel's precompute + lookup approach.
/// Returns a map of (token_id → penalty_value) for tokens that have DRY penalties.
pub fn cpu_compute_dry_penalties(
    recent_tokens: &[i32],
    recent_len: usize,
    batch_idx: usize,
    max_recent_len: usize,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_range: i32,
) -> Vec<(usize, f32)> {
    if dry_multiplier == 0.0 || recent_len < 2 {
        return vec![];
    }

    let base = batch_idx * max_recent_len;
    let range = if dry_range > 0 {
        dry_range.min(recent_len as i32) as usize
    } else {
        recent_len
    };

    let mut penalties: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();

    penalties.clear();

    for p in 0..range.saturating_sub(1) {
        let end_pos = recent_len - 1;
        let cmp_pos = recent_len - 2 - p;

        let mut match_len: i32 = 0;
        let mut a = cmp_pos as isize;
        let mut b = end_pos as isize;

        loop {
            if a < 0 || b < 0 {
                break;
            }
            if recent_tokens[base + a as usize] != recent_tokens[base + b as usize] {
                break;
            }
            match_len += 1;
            a -= 1;
            b -= 1;
            if b <= a {
                break;
            }
        }

        if match_len > dry_allowed_length {
            let next_pos = cmp_pos + 1;
            if next_pos < recent_len {
                let penalty_token = recent_tokens[base + next_pos] as usize;
                let penalty = dry_multiplier * dry_base.powi(match_len - dry_allowed_length);
                let entry = penalties.entry(penalty_token).or_insert(0.0f32);
                *entry = entry.max(penalty);
            }
        }
    }

    penalties.into_iter().collect()
}

/// Full CPU reference sampler for one sequence.
pub fn cpu_sample_argmax(
    logits: &[f32],
    vocab_size: usize,
    batch_idx: usize,
    temperature: f32,
    top_k: i32,
    _top_p: f32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: i32,
    dry_range: i32,
    eos_boost: f32,
    eos_token_id: i32,
    eos_ramp_start: i32,
    eos_ramp_len: i32,
    eos_boost_max_multiplier: f32,
    current_len: i32,
    eot_boost: f32,
    eot_token_id: i32,
    eot_ramp_start: i32,
    eot_ramp_len: i32,
    eot_boost_max_multiplier: f32,
    thinking_len: i32,
    cross_turn_penalty: f32,
    cross_turn_counts: &[i32],
    token_counts: &[i32],
    banned_tokens: &[i32],
    _num_banned: i32,
    banned_per_seq: i32,
    recent_tokens: &[i32],
    recent_lens: &[i32],
    max_recent_len: usize,
    stencil: &[i32],
) -> u32 {
    // Stencil path
    if !stencil.is_empty() {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_tok: u32 = 0;

        for &tok_id in stencil.iter() {
            let tid = tok_id as usize;
            if tid >= vocab_size {
                continue;
            }
            let mut v = logits[batch_idx * vocab_size + tid];

            v = cpu_apply_penalties(
                v,
                tid,
                batch_idx,
                vocab_size,
                repeat_penalty,
                frequency_penalty,
                presence_penalty,
                eos_boost,
                eos_token_id,
                eos_ramp_start,
                eos_ramp_len,
                eos_boost_max_multiplier,
                current_len,
                eot_boost,
                eot_token_id,
                eot_ramp_start,
                eot_ramp_len,
                eot_boost_max_multiplier,
                thinking_len,
                cross_turn_penalty,
                cross_turn_counts,
                token_counts,
                recent_tokens,
                recent_lens,
                max_recent_len,
                banned_tokens,
                banned_per_seq,
            );

            if v > best_val {
                best_val = v;
                best_tok = tok_id as u32;
            }
        }
        return best_tok;
    }

    // Compute DRY penalties
    let dry_penalties =
        if dry_multiplier != 0.0 && !recent_tokens.is_empty() && !recent_lens.is_empty() {
            let rlen = recent_lens[batch_idx] as usize;
            cpu_compute_dry_penalties(
                recent_tokens,
                rlen,
                batch_idx,
                max_recent_len,
                dry_multiplier,
                dry_base,
                dry_allowed_length,
                dry_range,
            )
        } else {
            vec![]
        };

    // Apply all penalties to get effective logits
    let mut penalized: Vec<f32> = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let mut v = logits[batch_idx * vocab_size + i];

        v = cpu_apply_penalties(
            v,
            i,
            batch_idx,
            vocab_size,
            repeat_penalty,
            frequency_penalty,
            presence_penalty,
            eos_boost,
            eos_token_id,
            eos_ramp_start,
            eos_ramp_len,
            eos_boost_max_multiplier,
            current_len,
            eot_boost,
            eot_token_id,
            eot_ramp_start,
            eot_ramp_len,
            eot_boost_max_multiplier,
            thinking_len,
            cross_turn_penalty,
            cross_turn_counts,
            token_counts,
            recent_tokens,
            recent_lens,
            max_recent_len,
            banned_tokens,
            banned_per_seq,
        );

        // Apply DRY penalty
        for &(tok, penalty) in &dry_penalties {
            if tok == i {
                v -= penalty;
            }
        }

        penalized.push(v);
    }

    // Argmax (temperature <= 0)
    if !(temperature > 0.0) {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx: u32 = 0;
        for (i, &v) in penalized.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        return best_idx;
    }

    // Temperature scaling + softmax
    let inv_temp = 1.0 / temperature;
    let max_val = penalized.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = penalized
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_val) * inv_temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for p in probs.iter_mut() {
        p.1 /= sum;
    }

    // Top-k filtering
    if top_k > 0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs.truncate(top_k as usize);
    }

    // Return the highest probability token (deterministic for testing)
    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| *i as u32)
        .unwrap_or(0)
}

// ============================================================================
// GPU Test Harness
// ============================================================================

/// Parameters for a single test invocation of the batched sampling kernel.
#[derive(Clone)]
pub struct SamplingParams {
    pub logits_f32: Vec<f32>,
    pub batch_size: i32,
    pub vocab_size: i32,
    pub dtype: i32,

    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,

    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,

    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_range: i32,

    pub eos_boost: f32,
    pub eos_token_id: i32,
    pub eos_ramp_start: i32,
    pub eos_ramp_len: i32,
    pub eos_boost_max_multiplier: f32,

    pub cross_turn_penalty: f32,
    pub cross_turn_counts: Vec<i32>,

    pub current_lens: Vec<i32>,

    pub eot_boost: f32,
    pub eot_token_id: i32,
    pub eot_ramp_start: i32,
    pub eot_ramp_len: i32,
    pub eot_boost_max_multiplier: f32,
    pub thinking_lens: Vec<i32>,

    pub token_counts: Vec<i32>,
    pub banned_tokens: Vec<i32>,
    pub num_banned_tokens: i32,
    pub banned_tokens_per_seq: i32,

    pub recent_tokens: Vec<i32>,
    pub recent_lens: Vec<i32>,
    pub max_recent_len: i32,

    pub stencil: Vec<i32>,
    pub stencil_size: i32,

    pub seed: u64,
    pub rng_offsets: Vec<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            logits_f32: vec![],
            batch_size: 1,
            vocab_size: 0,
            dtype: DType::F32 as i32,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_range: 0,
            eos_boost: 0.0,
            eos_token_id: -1,
            eos_ramp_start: 0,
            eos_ramp_len: 0,
            eos_boost_max_multiplier: 0.0,
            cross_turn_penalty: 0.0,
            cross_turn_counts: vec![],
            current_lens: vec![],
            eot_boost: 0.0,
            eot_token_id: -1,
            eot_ramp_start: 0,
            eot_ramp_len: 0,
            eot_boost_max_multiplier: 0.0,
            thinking_lens: vec![],
            token_counts: vec![],
            banned_tokens: vec![],
            num_banned_tokens: 0,
            banned_tokens_per_seq: 0,
            recent_tokens: vec![],
            recent_lens: vec![],
            max_recent_len: 0,
            stencil: vec![],
            stencil_size: 0,
            seed: 42,
            rng_offsets: vec![0],
        }
    }
}

/// Upload a Vec<T> to GPU. Returns None if empty.
pub fn upload<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    data: &[T],
) -> Option<CudaSlice<T>> {
    if data.is_empty() {
        None
    } else {
        Some(stream.memcpy_stod(data).expect("memcpy_stod failed"))
    }
}

/// Allocate zeroed GPU memory.
pub fn alloc_zeroed<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    count: usize,
) -> CudaSlice<T> {
    stream.alloc_zeros(count).expect("alloc_zeros failed")
}

/// Run the GPU kernel and return sampled tokens.
pub fn run_gpu(stream: &Arc<CudaStream>, p: &SamplingParams) -> Vec<u32> {
    let logits_gpu = stream
        .memcpy_stod(&p.logits_f32)
        .expect("upload logits failed");

    let token_counts_gpu = upload(stream, &p.token_counts);
    let cross_turn_gpu = upload(stream, &p.cross_turn_counts);
    let current_lens_gpu = upload(stream, &p.current_lens);
    let thinking_lens_gpu = upload(stream, &p.thinking_lens);
    let banned_gpu = upload(stream, &p.banned_tokens);
    let recent_gpu = upload(stream, &p.recent_tokens);
    let recent_lens_gpu = upload(stream, &p.recent_lens);
    let stencil_gpu = upload(stream, &p.stencil);

    let mut output_gpu: CudaSlice<u32> = alloc_zeroed(stream, p.batch_size as usize);

    let mut rng_offsets_gpu: CudaSlice<u64> = if p.rng_offsets.is_empty() {
        alloc_zeroed(stream, p.batch_size as usize)
    } else {
        stream
            .memcpy_stod(&p.rng_offsets)
            .expect("upload rng_offsets")
    };

    {
        let (logits_ptr, _g1) = logits_gpu.device_ptr(stream);
        let (output_ptr, _g2) = output_gpu.device_ptr_mut(stream);
        let (rng_ptr, _g3) = rng_offsets_gpu.device_ptr_mut(stream);

        let tc_ptr = token_counts_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let cross_ptr = cross_turn_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let cur_lens_ptr = current_lens_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let think_lens_ptr = thinking_lens_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let ban_ptr = banned_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let recent_ptr = recent_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let recent_lens_ptr = recent_lens_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let stencil_ptr = stencil_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());

        unsafe {
            run_batched_sampling(
                logits_ptr as *const c_void,
                p.batch_size,
                p.vocab_size,
                p.dtype,
                p.temperature,
                p.top_k,
                p.top_p,
                p.repeat_penalty,
                p.frequency_penalty,
                p.presence_penalty,
                p.dry_multiplier,
                p.dry_base,
                p.dry_allowed_length,
                p.dry_range,
                p.eos_boost,
                p.eos_token_id,
                p.eos_ramp_start,
                p.eos_ramp_len,
                p.eos_boost_max_multiplier,
                p.cross_turn_penalty,
                cross_ptr,
                cur_lens_ptr,
                p.eot_boost,
                p.eot_token_id,
                p.eot_ramp_start,
                p.eot_ramp_len,
                p.eot_boost_max_multiplier,
                think_lens_ptr,
                tc_ptr,
                ban_ptr,
                p.num_banned_tokens,
                p.banned_tokens_per_seq,
                recent_ptr,
                recent_lens_ptr,
                p.max_recent_len,
                stencil_ptr,
                p.stencil_size,
                output_ptr as *mut u32,
                p.seed,
                rng_ptr as *mut u64,
            );
        }
    }

    stream.synchronize().expect("sync failed");

    stream
        .memcpy_dtov(&output_gpu)
        .expect("download output failed")
}

/// Run CPU reference for all sequences in the batch.
pub fn run_cpu(p: &SamplingParams) -> Vec<u32> {
    (0..p.batch_size as usize)
        .map(|batch_idx| {
            let current_len = if batch_idx < p.current_lens.len() {
                p.current_lens[batch_idx]
            } else {
                0
            };
            cpu_sample_argmax(
                &p.logits_f32,
                p.vocab_size as usize,
                batch_idx,
                p.temperature,
                p.top_k,
                p.top_p,
                p.repeat_penalty,
                p.frequency_penalty,
                p.presence_penalty,
                p.dry_multiplier,
                p.dry_base,
                p.dry_allowed_length,
                p.dry_range,
                p.eos_boost,
                p.eos_token_id,
                p.eos_ramp_start,
                p.eos_ramp_len,
                p.eos_boost_max_multiplier,
                current_len,
                p.eot_boost,
                p.eot_token_id,
                p.eot_ramp_start,
                p.eot_ramp_len,
                p.eot_boost_max_multiplier,
                if batch_idx < p.thinking_lens.len() {
                    p.thinking_lens[batch_idx]
                } else {
                    0
                },
                p.cross_turn_penalty,
                &p.cross_turn_counts,
                &p.token_counts,
                &p.banned_tokens,
                p.num_banned_tokens,
                p.banned_tokens_per_seq,
                &p.recent_tokens,
                &p.recent_lens,
                p.max_recent_len as usize,
                &p.stencil,
            )
        })
        .collect()
}

/// Compare GPU and CPU results, printing details on mismatch.
pub fn assert_gpu_cpu_match(stream: &Arc<CudaStream>, p: &SamplingParams, test_name: &str) {
    let gpu = run_gpu(stream, p);
    let cpu = run_cpu(p);
    assert_eq!(gpu.len(), cpu.len(), "{test_name}: batch size mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(
            g, c,
            "{test_name}: seq {i} mismatch: GPU={g}, CPU={c}  \
             (vocab={}, temp={}, top_k={}, top_p={})",
            p.vocab_size, p.temperature, p.top_k, p.top_p
        );
    }
}

/// For stochastic tests: verify the GPU result is a valid token.
pub fn assert_valid_token(
    stream: &Arc<CudaStream>,
    p: &SamplingParams,
    test_name: &str,
    allowed: Option<&[u32]>,
) {
    let gpu = run_gpu(stream, p);
    for (i, &tok) in gpu.iter().enumerate() {
        assert!(
            (tok as i32) < p.vocab_size,
            "{test_name}: seq {i} token {tok} >= vocab_size {}",
            p.vocab_size
        );
        if let Some(allowed) = allowed {
            assert!(
                allowed.contains(&tok),
                "{test_name}: seq {i} token {tok} not in allowed set {allowed:?}"
            );
        }
    }
}

/// Helper to build linearly spaced logits where token `peak` has the max value.
pub fn make_peaked_logits(vocab_size: usize, peak: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab_size];
    for (i, v) in logits.iter_mut().enumerate() {
        *v = -(((i as i64 - peak as i64).abs()) as f32);
    }
    logits[peak] = 10.0;
    logits
}

/// Helper to create uniform logits.
pub fn make_uniform_logits(vocab_size: usize, value: f32) -> Vec<f32> {
    vec![value; vocab_size]
}

/// Create a CUDA stream on device 0 for testing.
pub fn test_stream() -> Arc<CudaStream> {
    let ctx = CudaContext::new(0).expect("no CUDA device");
    ctx.default_stream()
}

// ============================================================================
// Multi-Dtype Infrastructure
// ============================================================================

/// Run the GPU kernel with logits stored in the target dtype `T`.
pub fn run_gpu_typed<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    typed_logits: &[T],
    p: &SamplingParams,
    dtype_enum: i32,
) -> Vec<u32> {
    let logits_gpu = stream
        .memcpy_stod(typed_logits)
        .expect("upload typed logits failed");

    let token_counts_gpu = upload(stream, &p.token_counts);
    let cross_turn_gpu = upload(stream, &p.cross_turn_counts);
    let current_lens_gpu = upload(stream, &p.current_lens);
    let thinking_lens_gpu = upload(stream, &p.thinking_lens);
    let banned_gpu = upload(stream, &p.banned_tokens);
    let recent_gpu = upload(stream, &p.recent_tokens);
    let recent_lens_gpu = upload(stream, &p.recent_lens);
    let stencil_gpu = upload(stream, &p.stencil);

    let mut output_gpu: CudaSlice<u32> = alloc_zeroed(stream, p.batch_size as usize);

    let mut rng_offsets_gpu: CudaSlice<u64> = if p.rng_offsets.is_empty() {
        alloc_zeroed(stream, p.batch_size as usize)
    } else {
        stream
            .memcpy_stod(&p.rng_offsets)
            .expect("upload rng_offsets")
    };

    {
        let (logits_ptr, _g1) = logits_gpu.device_ptr(stream);
        let (output_ptr, _g2) = output_gpu.device_ptr_mut(stream);
        let (rng_ptr, _g3) = rng_offsets_gpu.device_ptr_mut(stream);

        let tc_ptr = token_counts_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let cross_ptr = cross_turn_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let cur_lens_ptr = current_lens_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let think_lens_ptr = thinking_lens_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let ban_ptr = banned_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let recent_ptr = recent_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let recent_lens_ptr = recent_lens_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());
        let stencil_ptr = stencil_gpu
            .as_ref()
            .map(|s| {
                let (p, _) = s.device_ptr(stream);
                p as *const i32
            })
            .unwrap_or(std::ptr::null());

        unsafe {
            run_batched_sampling(
                logits_ptr as *const c_void,
                p.batch_size,
                p.vocab_size,
                dtype_enum,
                p.temperature,
                p.top_k,
                p.top_p,
                p.repeat_penalty,
                p.frequency_penalty,
                p.presence_penalty,
                p.dry_multiplier,
                p.dry_base,
                p.dry_allowed_length,
                p.dry_range,
                p.eos_boost,
                p.eos_token_id,
                p.eos_ramp_start,
                p.eos_ramp_len,
                p.eos_boost_max_multiplier,
                p.cross_turn_penalty,
                cross_ptr,
                cur_lens_ptr,
                p.eot_boost,
                p.eot_token_id,
                p.eot_ramp_start,
                p.eot_ramp_len,
                p.eot_boost_max_multiplier,
                think_lens_ptr,
                tc_ptr,
                ban_ptr,
                p.num_banned_tokens,
                p.banned_tokens_per_seq,
                recent_ptr,
                recent_lens_ptr,
                p.max_recent_len,
                stencil_ptr,
                p.stencil_size,
                output_ptr as *mut u32,
                p.seed,
                rng_ptr as *mut u64,
            );
        }
    }

    stream.synchronize().expect("sync failed");
    stream
        .memcpy_dtov(&output_gpu)
        .expect("download output failed")
}

/// Convert f32 logits to f16 and return both the typed buffer and the
/// "round-tripped" f32 values the kernel will actually see.
pub fn f32_to_f16(logits: &[f32]) -> (Vec<f16>, Vec<f32>) {
    let typed: Vec<f16> = logits.iter().map(|&v| f16::from_f32(v)).collect();
    let quantised: Vec<f32> = typed.iter().map(|v| v.to_f32()).collect();
    (typed, quantised)
}

/// Convert f32 logits to bf16.
pub fn f32_to_bf16(logits: &[f32]) -> (Vec<bf16>, Vec<f32>) {
    let typed: Vec<bf16> = logits.iter().map(|&v| bf16::from_f32(v)).collect();
    let quantised: Vec<f32> = typed.iter().map(|v| v.to_f32()).collect();
    (typed, quantised)
}

/// Convert f32 logits to fp8 e4m3.
pub fn f32_to_fp8(logits: &[f32]) -> (Vec<F8E4M3>, Vec<f32>) {
    let typed: Vec<F8E4M3> = logits.iter().map(|&v| F8E4M3::from_f32(v)).collect();
    let quantised: Vec<f32> = typed.iter().map(|v| v.to_f32()).collect();
    (typed, quantised)
}

/// Run GPU with typed logits and compare against CPU reference that uses the
/// quantised-back-to-f32 logits (so both sides see the same precision).
pub fn assert_typed_gpu_cpu_match<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    typed_logits: &[T],
    quantised_f32: &[f32],
    p: &SamplingParams,
    dtype_enum: i32,
    test_name: &str,
) {
    let gpu = run_gpu_typed(stream, typed_logits, p, dtype_enum);
    let mut p_cpu = p.clone();
    p_cpu.logits_f32 = quantised_f32.to_vec();
    let cpu = run_cpu(&p_cpu);
    assert_eq!(gpu.len(), cpu.len(), "{test_name}: batch size mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(
            g, c,
            "{test_name}: seq {i} mismatch: GPU={g}, CPU={c}  \
             (dtype_enum={dtype_enum}, vocab={}, temp={})",
            p.vocab_size, p.temperature,
        );
    }
}
