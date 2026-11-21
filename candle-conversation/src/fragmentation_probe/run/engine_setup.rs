//! Stands up the engine over a scratch substrate and runs the probe's phases,
//! evaluating every gate before any of them fails the run.
//!
//! The scratch substrate is never the workspace's own, because a default
//! working directory would append probe conversations to the live log and
//! reload the real history. The fixture is the forward gate's own story, so the
//! probe's prompts match the gate's rows token for token. Tracing is left to
//! the caller, since a second subscriber install silently does nothing. Every
//! gate is reported before any one of them fails, so a failing threshold cannot
//! hide the evidence for its sibling.

use super::*;

/// A prefix of the forward gate's **own** story fixture, `fraction`/8ths long.
///
/// The gate's fixture rather than a fixture of this example's, for two reasons.
/// Phase B is meant to be read against the gate's clean rows, and a comparison
/// between two different prompts is not a comparison — same text, same tokenizer,
/// same token count. And `story_prompt`'s own doc records why a second copy is
/// worse than it looks: the file is checked in with CRLF, so an independent
/// `include_str!` of it "would be a different prompt that merely looked identical
/// in the source".
///
/// Phase A varies the fraction so conversations differ in KV size, which is what
/// spreads occupancy across size classes instead of pouring it all into one.
pub(super) fn story_slice(story: &str, fraction: usize) -> &str {
    let want = story.len() * fraction.clamp(1, 8) / 8;
    // Cut on a char boundary — the story is not ASCII.
    let mut end = want.min(story.len());
    while end > 0 && !story.is_char_boundary(end) {
        end -= 1;
    }
    &story[..end]
}

/// Run one probe, and return what it measured.
///
/// Loads the model, stands up an engine on a scratch substrate, runs the three phases,
/// prints the tables as it goes, and evaluates the gates. Every gate is evaluated and
/// reported before any of them fails the run — see [`ProbeOutcome`].
///
/// The caller owns the decision about tracing. Nothing here initialises a subscriber,
/// because a test process has usually already installed one and a second `try_init`
/// would silently do nothing; a driver that wants the compaction pass and the weight
/// side's reclaim visible installs INFO before calling.
pub fn run(probe: &Probe) -> anyhow::Result<ProbeOutcome> {
    let device = Device::new_cuda(probe.device)?;
    let builder = probe.builder();
    let resolved = builder.resolve_model(&device)?;
    let tokenizer = resolved.tokenizer()?;
    println!("Loading {:?} …", resolved.pack);
    let model = builder
        .load_model(&builder.weight_files(&resolved.pack)?, &device, None)?
        .model;
    run_on_model(probe, &device, &resolved.pack, tokenizer, model)
}

/// Stand up an engine on `model` over a scratch substrate.
///
/// **A scratch substrate, never the workspace's own.** A builder with no
/// `workspace_path` opens the substrate under the *process working directory* — so run
/// from the repo root this harness would append its conversations to the live
/// `substrate` and reload the user's real history on the way in. The returned store
/// must be held for the engine's whole life; dropping it removes the store, and a
/// previous run's corpse is swept on creation. See `scratch_substrate` for what that
/// does and does not promise. The scratch store is bound to the probe's model like any
/// other.
///
/// `pack` is the model pack `model` was loaded from — the authority on its logits
/// width, which the engine's sampling buffers are sized to, and on its identity.
pub(in super::super) fn start_engine(
    probe: &Probe,
    device: &Device,
    pack: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model: Box<dyn crate::ManagedBatchedModel + Send>,
) -> anyhow::Result<(ScratchSubstrate, Arc<ConversationEngine>)> {
    let scratch = ScratchSubstrate::new()?;
    if scratch.swept() > 0 {
        println!(
            "swept {} scratch substrate(s) left by a previous run",
            scratch.swept(),
        );
    }
    println!("substrate: {:?}", scratch.path());
    let builder = probe
        .builder()
        .model_path(pack)
        .workspace_path(scratch.path());
    let identity = builder.model_identity(&builder.weight_files(pack)?)?;
    let substrate = builder.bind_substrate(&identity)?;
    let engine_config = builder.engine_config(tokenizer, device, substrate);

    let engine = Arc::new(ConversationEngine::new(
        model,
        tokenizer.clone(),
        engine_config,
        GuestRegistry::new(),
    )?);
    println!("Engine up.\n");
    Ok((scratch, engine))
}
