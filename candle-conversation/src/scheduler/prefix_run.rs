//! A run of prefix sections between two holes, joined per layer and injected
//! onto a section ingest's scratch slot as one piece
//! (`docs/placed_section_layout.md` §3.1b).
//!
//! The sections of a prefix are contiguous in position until a hole parts
//! them, and a sealed sequence carries no hole, so a prefix goes in as one run
//! per stretch between holes: the run is injected at the tail, the hole is left
//! after it, and the next run is injected past the hole.

use candle_nn::kv_cache::{ArenaLocation, SealedChunk, SealedSequence};

use super::*;

/// The sections of one run, joined per layer.
pub(super) struct PrefixRun {
    chunks: Vec<Vec<SealedChunk>>,
    tokens: Vec<usize>,
    sections: usize,
}

impl PrefixRun {
    /// An empty run over `layers` layers.
    pub(super) fn new(layers: usize) -> Self {
        Self {
            chunks: (0..layers).map(|_| Vec::new()).collect(),
            tokens: vec![0; layers],
            sections: 0,
        }
    }

    /// Join one section, already cut to its layers' common prefix, onto the
    /// run.
    pub(super) fn push(&mut self, per_layer: &[SealedSequence]) {
        for (layer, seq) in per_layer.iter().enumerate() {
            self.chunks[layer].extend(seq.chunks.iter().cloned());
            self.tokens[layer] += seq.token_count;
        }
        self.sections += 1;
    }

    /// The run as one sealed sequence per layer, leaving the run empty, or
    /// `None` when it holds no token.
    fn take(&mut self, chunk_size: usize) -> Option<(Vec<SealedSequence>, usize)> {
        if self.tokens.first().copied().unwrap_or(0) == 0 {
            self.sections = 0;
            return None;
        }
        let sections = std::mem::take(&mut self.sections);
        let sealed = self
            .chunks
            .iter_mut()
            .zip(self.tokens.iter_mut())
            .map(|(chunks, tokens)| SealedSequence {
                chunks: std::mem::take(chunks),
                token_count: std::mem::take(tokens),
                chunk_size,
                location: ArenaLocation::Gpu,
            })
            .collect();
        Some((sealed, sections))
    }
}

impl Scheduler {
    /// Inject `run` at `sequence_id`'s tail and empty it. Every section was
    /// cut to its own layers' common prefix before it was joined, so the run
    /// can only be cut if the join itself misaligned them, which is refused.
    pub(super) fn inject_prefix_run(
        &mut self,
        sequence_id: SequenceId,
        run: &mut PrefixRun,
    ) -> Result<(), ConversationError> {
        let Some((sealed, sections)) = run.take(self.chunk_size) else {
            return Ok(());
        };
        let injected = self
            .session
            .inject_sealed_at_tail(sequence_id.0, &sealed)
            .map_err(ConversationError::Model)?;
        if injected.was_cut() {
            return Err(ConversationError::Other(format!(
                "prepare_section_ingest: a joined run of {sections} prefix section(s) was cut \
                 by {} chunk(s) although each section's layers agreed — the join misaligned them",
                injected.cut_chunks
            )));
        }
        Ok(())
    }
}
