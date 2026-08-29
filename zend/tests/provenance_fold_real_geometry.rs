//! Tier 2 — the provenance fold against real checkpoints (T5d.2, T5e.4, T5e.5).
//!
//! Tier 1 can fold synthetic signatures under any parameters it likes. What it
//! cannot do is read a geometry off a real GGUF, and every claim here is about
//! the geometry: that the derivation is an identity on the model being replaced,
//! that it fills all three groups on the model replacing it, and that the packing
//! path a 256-wide head is supposed to take is the one it takes.
//!
//! ```text
//! cargo test -p zend --release --features cuda --test provenance_fold_real_geometry \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## Why a stamp comparison is a bit-identity check
//!
//! T5e.4 asks whether the derived parameters produce byte-identical folded
//! signatures to the locked constants over a real turn. The fold is a pure
//! function of `(raw signature, FoldParams)`, and the capture path stamps each
//! record with the very parameters it folded under — one value, held on the
//! scheduler, used for both. So over one raw turn, equal parameters *are* equal
//! bytes, and asserting `stamp == FoldParams::locked()` on a record written by
//! the real Qwen3-30B is the whole claim: it reads the geometry out of the
//! checkpoint, derives from it, and compares against the constants that geometry
//! used to be hardcoded as.
//!
//! Re-folding the raw signature a second way would be the more literal test and a
//! weaker one — it would compare two expressions of the same `FoldParams` rather
//! than checking which `FoldParams` the model's own geometry yields.

use std::path::PathBuf;

use candle::Device;
use candle_conversation::models::Model;
use candle_conversation::projection;
use candle_conversation::projection::{TimelineId, TurnIndex};
use candle_conversation::provenance::{decode_wide_sigs, wide_sig_fold_params, FoldParams};
use candle_conversation::{ConversationEngine, SamplingConfig, Sequence};

const PROJECTION_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// The outgoing production model — 48 layers, 4 KV heads, head_dim 128.
const OUTGOING: Model = Model::Qwen3_30B_A3B_Q4;

/// The incoming hybrid — 10 attention layers, 2 KV heads, head_dim 256.
const HYBRID: Model = Model::Qwen36_35B_A3B_Q4;

// ── helpers ──────────────────────────────────────────────────────────────────

fn build_engine(
    model: Model,
    workspace: PathBuf,
    device: &Device,
) -> (ConversationEngine, Sequence) {
    let dialect = model.clone().spec().dialect.clone();
    let workspace_str = workspace.display().to_string();
    let mut proj_builder = projection::Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_YAML,
        &[("workspace", workspace_str.as_str())],
        Some(&dialect),
    )
    .expect("parse projection.yaml");
    let dialogue_layer = proj_builder
        .id_for_layer("dialogue")
        .expect("dialogue layer");
    let primary_group = proj_builder
        .id_for_group("primary_conversation")
        .expect("primary group");

    let mut builder = model
        .builder()
        .workspace_path(workspace)
        .sampling(SamplingConfig::argmax())
        .seed(0)
        .max_response_tokens(24)
        .thinking(false);
    let conv_config = builder.conversation_config();
    let engine = builder.engine(device).expect("engine load");

    let tokenizer = engine.tokenizer().clone();
    proj_builder
        .tokenize_templates::<anyhow::Error, _>(|s| {
            let encoded = tokenizer
                .encode(s, false)
                .map_err(|e| anyhow::anyhow!("template tokenise: {e}"))?;
            Ok(encoded.get_ids().to_vec())
        })
        .expect("tokenize templates");

    let formatted_prompt = builder.format_system_prompt();
    let conv = engine
        .new_conversation_with_projection(
            &formatted_prompt,
            proj_builder,
            dialogue_layer,
            primary_group,
            conv_config,
        )
        .expect("new conv");
    (engine, conv)
}

/// The stored wide-Q record for the first sealed turn on `timeline`.
///
/// Polls: the seal enqueues onto the off-thread persistence writer, so a single
/// read that found nothing would be reporting its own timing rather than a
/// missing capture.
fn first_turn_sig_blob(engine: &ConversationEngine, timeline: TimelineId) -> Vec<u8> {
    let conv = engine.conversation();
    for _ in 0..100 {
        if let Some(b) = conv.read().wide_q_sigs_blob(timeline, TurnIndex(0)) {
            if !b.is_empty() {
                return b.to_vec();
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!(
        "no wide-Q signature was stored for the first turn — provenance retrieval \
         has nothing to match against, and the conversation would fall back to \
         recency without saying so"
    );
}

/// Run one turn and hand back its stored signature record.
fn seal_one_turn(model: Model, device: &Device) -> (Vec<u8>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let label = format!("{model:?}");
    let (engine, mut conv) = build_engine(model, tmp.path().to_path_buf(), device);
    let timeline = conv.timeline_id();
    let reply = conv
        .send_turn("Name the capital of France in one word.")
        .expect("send_turn");
    // The factual one-liner from §6.5: it separates a mis-read checkpoint from a
    // model that simply did not follow the instruction, and those are
    // indistinguishable in a signature diff.
    eprintln!("[{label}] reply: {}", reply.text.trim());
    let blob = first_turn_sig_blob(&engine, timeline);
    drop(engine);
    (blob, tmp)
}

/// Bits set across a folded signature's group `g`, summed over its heads.
fn group_popcount(words: &[u64], n_heads: usize, g: usize) -> u32 {
    let wph = words.len() / n_heads.max(1);
    let heads_per_group = n_heads / 3;
    let start = g * heads_per_group * wph;
    let end = start + heads_per_group * wph;
    words[start..end].iter().map(|w| w.count_ones()).sum()
}

// ── T5e.4 — no regression on the outgoing model ──────────────────────────────

/// **The derivation is an identity on Qwen3-30B.**
///
/// This is what licensed changing all five hardcoded fold parameters at once:
/// the model still in production must fold exactly as it did. Read the geometry
/// from the real checkpoint, derive from it, and compare against the constants.
#[test]
#[ignore = "Tier 2: loads the pinned Qwen3-30B-A3B GGUF (18.5 GB) and seals one \
            turn (~2 min). Run with: cargo test -p zend --release --features cuda \
            --test provenance_fold_real_geometry -- --ignored --nocapture \
            --test-threads=1"]
fn qwen3_30b_folds_under_the_locked_constants() {
    let device = Device::new_cuda(0).expect("cuda");
    let (blob, _tmp) = seal_one_turn(OUTGOING, &device);

    let stamped = wide_sig_fold_params(&blob).expect("the record states its fold");
    assert_eq!(
        stamped,
        FoldParams::locked(),
        "the outgoing model's derived fold is no longer the locked one. Every \
         signature already on disk was folded with the constants, so this is a \
         silent retrieval regression on the production corpus, not a new-model \
         concern."
    );

    let sigs = decode_wide_sigs(&blob).expect("decode");
    assert!(!sigs.is_empty(), "record decoded to zero tokens");
    let s = &sigs[0];
    assert_eq!(s.n_heads, 12, "3 groups x 4 KV heads");
    assert_eq!(s.words.len(), 12 * 2, "head_dim 128 = 2 u64 per head");
}

// ── T5e.5 — real-geometry fold on the hybrid ─────────────────────────────────

/// **The hybrid fills all three groups, at the shape §4.8 predicts.**
///
/// 10 attention layers, 2 KV heads, head_dim 256 → 3 groups x 2 heads x 256 bits
/// = 1536 bits, the same budget the 30B spends on 12 narrower heads.
///
/// The load-bearing assertion is the per-group popcount. Under the locked
/// `[46, 1, 1]` a 10-layer stack puts every layer in group 0 and leaves groups 1
/// and 2 **all zero** — and an all-zero group is not a weak signature, it is a
/// scorer input that agrees with everything it is compared to. It produces
/// confident matches against unrelated turns, which is invisible in every
/// output.
#[test]
#[ignore = "Tier 2: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and seals one \
            turn (~2 min). Run with: cargo test -p zend --release --features cuda \
            --test provenance_fold_real_geometry -- --ignored --nocapture \
            --test-threads=1"]
fn hybrid_fold_fills_every_group_at_head_dim_256() {
    let device = Device::new_cuda(0).expect("cuda");
    let (blob, _tmp) = seal_one_turn(HYBRID, &device);

    let stamped = wide_sig_fold_params(&blob).expect("the record states its fold");
    assert_eq!(
        stamped,
        FoldParams::derive(2, 10, 256),
        "the hybrid's stored fold is not the one its geometry derives"
    );
    assert_eq!(stamped.group_sizes, [8, 1, 1], "[n-2, 1, 1] over 10 layers");
    assert_eq!(stamped.shift, 64, "head_dim / 4");

    let sigs = decode_wide_sigs(&blob).expect("decode");
    assert!(!sigs.is_empty(), "record decoded to zero tokens");
    let s = &sigs[0];
    assert_eq!(s.n_heads, 6, "3 groups x 2 KV heads");
    assert_eq!(s.words.len(), 6 * 4, "head_dim 256 = 4 u64 per head");

    for g in 0..3 {
        let bits = group_popcount(&s.words, s.n_heads as usize, g);
        assert!(
            bits > 0,
            "group {g} of the hybrid's signature is entirely zero. It will match \
             every turn it is scored against with full confidence."
        );
    }
}

// ── T5d.2 — the GPU sign-pack path is taken ──────────────────────────────────

/// **At head_dim 256 the capture uses the GPU sign-pack path.**
///
/// Asserts the path, never a duration. The CPU fallback is *correct* — it
/// produces bit-identical signatures — so a demotion costs a device-to-host copy
/// of every layer's R16 Q on every seal and changes nothing observable about the
/// output. Only a path assertion can see it.
#[test]
#[ignore = "Tier 2: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and seals one \
            turn (~2 min). Run with: cargo test -p zend --release --features cuda \
            --test provenance_fold_real_geometry -- --ignored --nocapture \
            --test-threads=1"]
fn hybrid_capture_takes_the_gpu_sign_pack_path() {
    let device = Device::new_cuda(0).expect("cuda");
    let (_blob, _tmp) = seal_one_turn(HYBRID, &device);

    let (gpu, cpu) = candle_conversation::provenance_capture_path_counts();
    assert!(
        gpu > 0,
        "no provenance capture went through the GPU sign-pack path ({gpu} gpu, \
         {cpu} cpu). At head_dim 256 `prov_sub_head_dim` returning 0 demotes \
         every seal to a full R16 device-to-host copy, and the signatures it \
         produces are correct, so nothing else reports it."
    );
    assert_eq!(
        cpu, 0,
        "{cpu} captures fell back to the CPU R16 gather while the GPU path was \
         available"
    );
}
