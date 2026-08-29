//! Tier 2 — the hybrid's recurrent-state hooks against real weights.
//!
//! Between the tier-1 `RecurrentStateStore` tests, which run on synthetic dims
//! and never reach a kernel, and the tier-3 zend tests, which exercise a whole
//! daemon and localise nothing. These drive [`HybridBatched`]'s
//! `ManagedBatchedModel` hooks directly, so a failure names one hook.
//!
//! ## Everything here asserts on exported bytes
//!
//! `export_recurrent` is the only view of the state that is not itself part of
//! what is under test, and byte equality is the only comparison worth making:
//! these operations are copies, moves and discards, not arithmetic, so a
//! tolerance would be a way of not noticing a layout bug. A state that is subtly
//! wrong produces text that is entirely fluent — which is the failure signature
//! this whole area is built around (`docs/deltanet_state_persistence.md` §6).

use candle::{DType, Device, Result, Tensor};

use crate::models::batch_test::test_helpers::hf_get;
use crate::models::batch_test::utils::{decode_replay_probe, TestParams};
use crate::models::batched_inference::{
    BatchedConfig, BatchedInferenceSession, ManagedBatchedModel,
};
use crate::models::delta_net::ExportedLayerState;
use crate::models::dialect::Dialect;
use crate::models::quantized_qwen36_moe::from_gguf_path;
use crate::models::qwen35::{HybridBatched, Qwen35LoadOptions};

const MODEL_REPO: &str = "unsloth/Qwen3.6-35B-A3B-GGUF";
const MODEL_REV: &str = "a483e9e6cbd595906af30beda3187c2663a1118c";
const MODEL_FILE: &str = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
const TOK_REPO: &str = "Qwen/Qwen3.6-35B-A3B";
const TOK_REV: &str = "995ad96eacd98c81ed38be0c5b274b04031597b0";

/// Recurrent layers on this stack — 40 layers at 3:1.
const RECURRENT_LAYERS: usize = 30;

// ── harness ──────────────────────────────────────────────────────────────────

fn load() -> Result<(HybridBatched, Device)> {
    let path = hf_get(MODEL_REPO, hf_hub::RepoType::Model, MODEL_REV, MODEL_FILE)?;
    let device = Device::new_cuda(0)?;
    let model = from_gguf_path(&path, &device, Qwen35LoadOptions::default())?;
    Ok((model, device))
}

/// A prompt with the family's real turn structure, long enough to cross the
/// 32-token chunk boundary the recurrent scan carries state across — a probe
/// that stayed inside one chunk would never exercise the carry.
fn prompt_tokens() -> Result<Vec<u32>> {
    let tok_path = hf_get(TOK_REPO, hf_hub::RepoType::Model, TOK_REV, "tokenizer.json")?;
    let params = TestParams::new(4, &std::fs::read_to_string(&tok_path)?, Dialect::qwen35())
        .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
        .with_suppress_thinking(true);
    let mut tokens = params.system_prompt_tokens(0);
    tokens.extend(params.user_prompt_tokens(0));
    tokens.truncate(160);
    Ok(tokens)
}

/// Prefill `ids` onto `seq` in one wave, returning the next-token argmax.
fn feed(
    model: &HybridBatched,
    session: &mut BatchedInferenceSession,
    seq: usize,
    ids: &[u32],
    device: &Device,
) -> Result<u32> {
    let t = Tensor::from_vec(ids.to_vec(), (1, ids.len()), device)?;
    let step = model.forward_wave(
        session,
        &[],
        &[],
        &[seq],
        &[t],
        &[],
        &[],
        0,
        ManagedBatchedModel::num_layers(model),
        None,
    )?;
    let row = step
        .logits
        .as_ref()
        .ok_or_else(|| candle::Error::Msg("no logits".into()))?[0]
        .flatten_all()?
        .to_dtype(DType::F32)?;
    let next = row.argmax(0)?.to_scalar::<u32>()?;
    drop(step);
    Ok(next)
}

/// Greedy-continue `seq` for `n` steps, returning the emitted ids.
fn decode_n(
    model: &HybridBatched,
    session: &mut BatchedInferenceSession,
    seq: usize,
    first: u32,
    n: usize,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    let mut tok = first;
    for _ in 0..n {
        out.push(tok);
        tok = feed(model, session, seq, &[tok], device)?;
    }
    Ok(out)
}

/// Carve a view of `parent` that borrows **all** of its K/V, and copy the
/// recurrent state across — the production turn carve, in one call.
///
/// The explicit full range is load-bearing. `create_view_sequence` treats an
/// empty `visible_block_ranges` as *borrow nothing* and returns a zero-block
/// view without complaint; the "empty means every block" default belongs to the
/// scheduler's wrapper, which builds `vec![(0, total_blocks)]` before calling
/// down. A view carved with `&[]` decodes from an empty context and produces
/// fluent, plausible, entirely unrelated text — which is how a first draft of
/// the continuation test below "found" a fork bug that was its own.
fn carve_view(
    model: &HybridBatched,
    session: &mut BatchedInferenceSession,
    parent: usize,
) -> Result<usize> {
    let blocks = session
        .sequence_block_count(parent)
        .ok_or_else(|| candle::Error::Msg("parent has no slot".into()))?;
    let child = session.create_view_sequence(parent, &[(0, blocks)])?;
    assert_eq!(
        child.borrowed_block_count, blocks,
        "the view borrowed {} of the parent's {blocks} blocks",
        child.borrowed_block_count
    );
    ManagedBatchedModel::fork_recurrent(model, parent, child.view_idx)?;
    Ok(child.view_idx)
}

fn export(model: &HybridBatched, seq: usize) -> Result<Vec<ExportedLayerState>> {
    let (_hash, layers) = ManagedBatchedModel::export_recurrent(model, seq)?
        .ok_or_else(|| candle::Error::Msg("the hybrid reported no recurrent state".into()))?;
    Ok(layers)
}

/// Layer indices whose exported bytes differ.
fn diff(a: &[ExportedLayerState], b: &[ExportedLayerState]) -> Vec<u32> {
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.state != y.state || x.conv_tail != y.conv_tail)
        .map(|(x, _)| x.layer_index)
        .collect()
}

fn any_non_zero(s: &[ExportedLayerState]) -> bool {
    s.iter()
        .any(|l| l.state.iter().any(|&b| b != 0) || l.conv_tail.iter().any(|&b| b != 0))
}

const RUN: &str = "cargo test --release --features cuda --lib -p candle-transformers \
                   qwen35::recurrent_gates -- --ignored --nocapture --test-threads=1";

// ── T3.8 — the harness control is honest again ───────────────────────────────

/// **`decode_replay_probe` reports zero divergences on the hybrid.**
///
/// The probe exists to separate harness drift from model non-determinism, and on
/// this lineage it used to manufacture the drift it is there to rule out: it
/// rewound one sequence's K/V between replays, which restores everything a
/// sequence remembers *only* on a model whose memory is entirely K/V. Each
/// replay therefore entered from a state one token further along, and the
/// divergence it reported was its own.
///
/// It now allocates a fresh sequence per replay. A non-zero count here means
/// either the hybrid really is non-deterministic or the reset is incomplete
/// again — and both would otherwise be read off this instrument as fact.
#[test]
#[ignore = "Tier 2: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and needs a GPU (~3 min)."]
fn decode_replay_probe_is_clean_on_the_hybrid() -> Result<()> {
    eprintln!("run with: {RUN}");
    let (model, device) = load()?;
    let ids = prompt_tokens()?;
    let dirty = decode_replay_probe(&model, &device, &ids, 8, "qwen36-hybrid")?;
    assert_eq!(
        dirty, 0,
        "{dirty} of 8 replays diverged. Before the per-replay fresh sequence this \
         was guaranteed on any recurrent model and read as 'the hybrid is \
         non-deterministic'; if it is back, no per-layer instrument built on this \
         probe means anything until it clears."
    );
    Ok(())
}

// ── T1.6 — the fork copy stays on the device ─────────────────────────────────

/// **`fork_recurrent` gives the child its own copy of the parent's state.**
///
/// Two claims, one asserted and one structural. Asserted: the child equals the
/// parent at the fork and then moves independently — advancing the child leaves
/// the parent byte-identical, which is what distinguishes a copy from an alias.
/// An alias is the bug that makes every turn's view write through to the
/// conversation it forked from.
///
/// Structural: that the copy never lands in host memory.
/// `DeltaNetState::snapshot` is `Tensor::copy()` on both buffers, which is a
/// device operation on a device tensor — there is no host round-trip to observe,
/// and asserting device residency of the *result* would not distinguish one that
/// had taken one. This test pins the semantics; the residency is a property of
/// the operation it is built from.
#[test]
#[ignore = "Tier 2: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and needs a GPU (~2 min)."]
fn fork_recurrent_is_an_independent_copy() -> Result<()> {
    eprintln!("run with: {RUN}");
    let (model, device) = load()?;
    let ids = prompt_tokens()?;
    let mut session = model.create_batched_session(BatchedConfig::default())?;

    let parent = session.create_sequence()?;
    let next = feed(&model, &mut session, parent, &ids, &device)?;
    let at_fork = export(&model, parent)?;
    assert_eq!(at_fork.len(), RECURRENT_LAYERS, "layer count");
    assert!(
        any_non_zero(&at_fork),
        "the parent's state is all zeros after a {}-token prefill — the recurrent \
         layers absorbed nothing and the model is answering from its ten \
         attention layers",
        ids.len()
    );

    let child = carve_view(&model, &mut session, parent)?;
    let child_at_fork = export(&model, child)?;
    assert!(
        diff(&at_fork, &child_at_fork).is_empty(),
        "the fork did not reproduce the parent's state"
    );

    decode_n(&model, &mut session, child, next, 4, &device)?;
    let parent_after = export(&model, parent)?;
    let child_after = export(&model, child)?;

    assert!(
        diff(&at_fork, &parent_after).is_empty(),
        "decoding on the child moved the parent's state: the two share a buffer, \
         so every turn's view has been writing through to its parent"
    );
    assert!(
        !diff(&child_at_fork, &child_after).is_empty(),
        "four decode steps left the child's state byte-identical — nothing \
         advanced, so the comparison above is vacuous"
    );
    Ok(())
}

// ── T7.6 — parent and fork continue identically ──────────────────────────────

/// **A fork continues exactly as its parent would have.**
///
/// Under argmax there is no sampling noise to hide behind: the two must emit the
/// same tokens, and their states must stay byte-identical. A fork that copies
/// the K/V and not the state produces a fluent, plausible, *different*
/// continuation — indistinguishable from the model having said something else.
///
/// **Separate sessions, so neither run can disturb the other.** Production
/// freezes the parent for the life of its view — the view *is* the turn — so
/// decoding both on one session would be asking a question the engine never
/// asks. Two independent runs from the same prompt ask the intended one.
///
/// See [`carve_view`] for the trap this test found the hard way: the first two
/// attempts failed identically, with the child emitting fluent text unrelated to
/// the prompt, because the view had borrowed no K/V at all.
#[test]
#[ignore = "Tier 2: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and needs a GPU (~3 min)."]
fn a_fork_and_its_parent_produce_identical_continuations() -> Result<()> {
    eprintln!("run with: {RUN}");
    let (model, device) = load()?;
    let ids = prompt_tokens()?;

    // The parent's own continuation, with no view ever taken.
    let mut solo = model.create_batched_session(BatchedConfig::default())?;
    let lone = solo.create_sequence()?;
    let next = feed(&model, &mut solo, lone, &ids, &device)?;
    let from_parent = decode_n(&model, &mut solo, lone, next, 12, &device)?;
    let parent_state = export(&model, lone)?;

    // The same prompt, continued on a fork instead.
    let mut forked = model.create_batched_session(BatchedConfig::default())?;
    let parent = forked.create_sequence()?;
    let next2 = feed(&model, &mut forked, parent, &ids, &device)?;
    assert_eq!(next2, next, "the two prefills disagreed before any fork");
    let child = carve_view(&model, &mut forked, parent)?;
    let from_child = decode_n(&model, &mut forked, child, next, 12, &device)?;

    assert_eq!(
        from_parent, from_child,
        "the fork diverged from its parent under argmax. Both continuations read \
         fluently; only this comparison says one of them is running on a state it \
         never inherited."
    );
    assert!(
        diff(&parent_state, &export(&model, child)?).is_empty(),
        "identical tokens but different state — the continuations agreed by \
         accident of the attention layers, not because the recurrence carried"
    );
    Ok(())
}

// ── T3.4 — the `<think>` oracle ──────────────────────────────────────────────

/// **A discarded thinking pass leaves no trace in the state that gets sealed.**
///
/// Site 8 as an oracle. The old shape decoded the response — advancing `S`
/// through every DeltaNet layer, thinking tokens included — then truncated the
/// K/V back to the turn boundary and re-prefilled the turn reasoning-free,
/// advancing `S` a *second* time on a state that had already absorbed the
/// thinking. The state ended up having seen `[prefix][thinking][clean]` while
/// the K/V held `[prefix][clean]`, on every thinking turn of every conversation.
///
/// The fix is structural: the turn decodes on a child, and sealing clean
/// discards the child instead of rewinding the parent. So these must land on the
/// same bytes —
///
/// - **A**: fork a child, decode thinking tokens on it, release it, then feed
///   the clean response to the parent.
/// - **B**: feed the clean response to a parent that never forked.
///
/// Byte equality, not tolerance: the same arithmetic over the same tokens, so
/// any difference is state that leaked across the discard.
#[test]
#[ignore = "Tier 2: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and needs a GPU (~3 min)."]
fn a_discarded_thinking_pass_does_not_reach_the_sealed_state() -> Result<()> {
    eprintln!("run with: {RUN}");
    let (model, device) = load()?;
    let ids = prompt_tokens()?;
    // Stands in for the reasoning-free response: any fixed token run works, as
    // long as both paths feed the identical one.
    let clean: Vec<u32> = ids.iter().rev().take(24).copied().collect();

    // Path A — a thinking pass on a child, discarded, then the clean feed.
    let mut session = model.create_batched_session(BatchedConfig::default())?;
    let parent = session.create_sequence()?;
    let next = feed(&model, &mut session, parent, &ids, &device)?;
    let boundary = export(&model, parent)?;

    let child = carve_view(&model, &mut session, parent)?;
    decode_n(&model, &mut session, child, next, 8, &device)?;
    ManagedBatchedModel::release_sequence(&model, child)?;
    assert!(
        diff(&boundary, &export(&model, parent)?).is_empty(),
        "the discarded child's decoding moved the parent's state before the clean \
         re-feed even ran — the turn boundary did not hold"
    );
    feed(&model, &mut session, parent, &clean, &device)?;
    let with_thinking = export(&model, parent)?;

    // Path B — the same clean feed, on a sequence that never thought.
    let mut plain_session = model.create_batched_session(BatchedConfig::default())?;
    let plain = plain_session.create_sequence()?;
    feed(&model, &mut plain_session, plain, &ids, &device)?;
    feed(&model, &mut plain_session, plain, &clean, &device)?;
    let without_thinking = export(&model, plain)?;

    let moved = diff(&with_thinking, &without_thinking);
    assert!(
        moved.is_empty(),
        "{} recurrent layers ended in a different state after a discarded \
         thinking pass (first: {:?}). The state has seen [prefix][thinking][clean] \
         while the K/V holds [prefix][clean], and every reply from here on reads \
         perfectly.",
        moved.len(),
        moved.first()
    );
    Ok(())
}
