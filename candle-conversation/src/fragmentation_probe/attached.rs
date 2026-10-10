//! Two substrates on one engine — the shared-model host's arrangement
//! (`docs/shared_model_host.md` §3.2), driven through the real engine.
//!
//! One model loads and one scheduler runs; a first substrate is the engine's
//! own, and a second is attached to it, as `tokerad` attaches npcd's beside
//! zend's. Each runs the forward gate's story batch at the same moment, so
//! their sequences share every wave, and each substrate's persistence thread
//! drains its own.
//!
//! The probe answers three questions, and fails on any of them:
//!
//! 1. **Are both batches' answers right?** Two substrates mint timelines from
//!    one process-wide allocator and hold content-addressed sections alike; a
//!    scheduler map that confused them would answer one substrate's sequence
//!    from the other's KV, which only a content check sees.
//! 2. **Did each substrate persist its own conversations?** Each attachment
//!    has its own persistence thread and writer; reopened after shutdown, each
//!    store declares a sealed turn of every conversation its batch opened, and
//!    of none the other's did.
//! 3. **Does the second attachment detach without stopping the first?** The
//!    core stops only when its last attachment detaches, so the first serves
//!    one more turn after the detach.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::thread;

use candle::Device;

use super::batch::run_story_batch;
use super::probe::Probe;
use super::run::start_engine;
use crate::cancel::CancelToken;
use crate::conversation::Sequence;
use crate::persistence::streams::StreamDecl;
use crate::persistence::SubstratePersistence;
use crate::scratch_substrate::ScratchSubstrate;
use crate::substrate::Substrate;
use crate::tenant_rank::TenantRank;

/// What one substrate's half of the run produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentOutcome {
    /// Sessions whose rewrite matched the story.
    pub story_pass: usize,
    pub width: usize,
    /// Story mismatches and turn errors, worded for the failure report.
    pub failures: Vec<String>,
    /// Of the conversations this half opened, how many the substrate holds a
    /// sealed turn of when reopened after shutdown.
    pub persisted: usize,
    /// Of the conversations the *other* half opened, how many this substrate
    /// holds — anything but zero is one tenant's turn written to another's
    /// store.
    pub foreign: usize,
}

/// Both halves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachedOutcome {
    pub own: AttachmentOutcome,
    pub attached: AttachmentOutcome,
    /// Whether the engine's own attachment still served a turn after the
    /// second one detached.
    pub own_survived_detach: bool,
}

impl AttachedOutcome {
    /// Every gate's verdict, failures first.
    pub fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (name, half) in [("own", &self.own), ("attached", &self.attached)] {
            if half.story_pass != half.width {
                out.push(format!(
                    "{name}: story {}/{} — {}",
                    half.story_pass,
                    half.width,
                    half.failures.join("; ")
                ));
            }
            if half.persisted < half.width {
                out.push(format!(
                    "{name}: {} conversations persisted, {} opened",
                    half.persisted, half.width
                ));
            }
            if half.foreign > 0 {
                out.push(format!(
                    "{name}: holds {} of the other substrate's conversations",
                    half.foreign
                ));
            }
        }
        if !self.own_survived_detach {
            out.push("detaching the second substrate stopped the first".into());
        }
        out
    }

    pub fn assert_passed(&self) {
        let failures = self.failures();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

/// The timelines the substrate at `workspace` holds a sealed turn of, read
/// back from its redo log's turn declarations — read-only, after its writer
/// has gone.
fn persisted_timelines(workspace: &Path) -> anyhow::Result<HashSet<u64>> {
    let mut substrate = Substrate::new();
    SubstratePersistence::open_in_with_substrate_read_only(workspace, &mut substrate)?;
    Ok(substrate
        .all_streams()
        .filter_map(|(_, s)| match &s.decl {
            Some(StreamDecl::Turn(t)) => Some(t.timeline_id),
            _ => None,
        })
        .collect())
}

/// The raw timeline ids of `conversations`.
fn timelines_of(conversations: &[Sequence]) -> HashSet<u64> {
    conversations
        .iter()
        .map(|c| c.timeline_id().raw())
        .collect()
}

/// Run `width` sessions of the story on each of two substrates sharing one
/// engine, at once.
pub fn run_attached(probe: &Probe, width: usize) -> anyhow::Result<AttachedOutcome> {
    let device = Device::new_cuda(probe.device)?;
    let builder = probe.builder();
    let (model_path, tokenizer_path) = builder.resolve_paths_pub()?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    println!("Loading {model_path:?} …");
    let model = builder
        .load_model(&builder.weight_files(&model_path)?, &device, None)?
        .model;
    let (own_scratch, own) = start_engine(probe, &device, &tokenizer, model)?;

    // The second substrate, bound to the same model and attached ranked after
    // the first, as a second tenant's is.
    let other_scratch = ScratchSubstrate::new()?;
    println!("attached substrate: {:?}", other_scratch.path());
    let other_builder = probe.builder().workspace_path(other_scratch.path());
    let identity =
        other_builder.model_identity(&other_builder.weight_files(&model_path)?, &tokenizer_path)?;
    let bound = other_builder.bind_substrate(&identity)?;
    let rank = TenantRank::nth(1).expect("a second tenant has a rank");
    let attached = Arc::new(own.attach(bound, CancelToken::new(), rank, HashMap::new())?);
    assert_eq!(own.attached_count(), 2, "two substrates on one engine");

    // Both batches at once, so their sequences share every wave.
    let (own_batch, other_batch) = thread::scope(|s| {
        let a = s.spawn(|| {
            run_story_batch(
                &own,
                &builder,
                width,
                probe.baseline_decode,
                probe.story_gate(),
            )
        });
        let b = s.spawn(|| {
            run_story_batch(
                &attached,
                &other_builder,
                width,
                probe.baseline_decode,
                probe.story_gate(),
            )
        });
        (a.join(), b.join())
    });
    let own_batch = own_batch.map_err(|_| anyhow::anyhow!("the own batch panicked"))??;
    let other_batch = other_batch.map_err(|_| anyhow::anyhow!("the attached batch panicked"))??;

    let own_timelines = timelines_of(&own_batch.conversations);
    let other_timelines = timelines_of(&other_batch.conversations);

    // The second detaches; the first must still serve — a turn through the
    // scheduler, not just its own persistence.
    drop(other_batch.conversations);
    attached.shutdown()?;
    let after = run_story_batch(&own, &builder, 1, probe.baseline_decode, probe.story_gate())?;
    let own_survived_detach = own.attached_count() == 1 && after.story_pass == 1;
    drop(after.conversations);
    drop(own_batch.conversations);
    own.shutdown()?;
    drop(attached);
    drop(own);

    let own_store = persisted_timelines(own_scratch.path())?;
    let other_store = persisted_timelines(other_scratch.path())?;
    Ok(AttachedOutcome {
        own: AttachmentOutcome {
            story_pass: own_batch.story_pass,
            width,
            failures: own_batch
                .story_fail
                .into_iter()
                .chain(own_batch.errors)
                .collect(),
            persisted: own_timelines.intersection(&own_store).count(),
            foreign: other_timelines.intersection(&own_store).count(),
        },
        attached: AttachmentOutcome {
            story_pass: other_batch.story_pass,
            width,
            failures: other_batch
                .story_fail
                .into_iter()
                .chain(other_batch.errors)
                .collect(),
            persisted: other_timelines.intersection(&other_store).count(),
            foreign: own_timelines.intersection(&other_store).count(),
        },
        own_survived_detach,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(story_pass: usize, persisted: usize, foreign: usize) -> AttachmentOutcome {
        AttachmentOutcome {
            story_pass,
            width: 4,
            failures: vec!["s2 diverged".into()],
            persisted,
            foreign,
        }
    }

    /// Every gate is reported, each naming its half.
    #[test]
    fn every_failure_names_its_half() {
        let ok = AttachedOutcome {
            own: half(4, 4, 0),
            attached: half(4, 4, 0),
            own_survived_detach: true,
        };
        assert!(ok.failures().is_empty());
        let bad = AttachedOutcome {
            own: half(3, 4, 1),
            attached: half(4, 2, 0),
            own_survived_detach: false,
        };
        assert_eq!(
            bad.failures(),
            [
                "own: story 3/4 — s2 diverged",
                "own: holds 1 of the other substrate's conversations",
                "attached: 2 conversations persisted, 4 opened",
                "detaching the second substrate stopped the first",
            ]
        );
    }
}
