//! Divides a model's KV layers into the ones a wave steps together and the ones
//! a draft head steps on its own.
//!
//! Stream layers advance on every wave and hold the same token prefix at a wave
//! boundary, so every aggregate across layers is taken over them alone. A draft
//! layer is allocated so the head has somewhere to write, and it is excluded from
//! those aggregates. A layer nothing has written reads a zero minimum, which the
//! wave-entry reconciler would take for an offset running ahead and clamp the
//! sequence to zero. The intended arrangement is a head whose layer is a stream
//! layer like any other.

/// How a model's KV layers divide into the ones a wave steps together and the
/// ones stepped by a pass of their own.
///
/// A **stream** layer advances on every wave: the trunk sweep writes it, so all
/// stream layers hold the same token prefix at a wave boundary, and every
/// stream-wide aggregate — the token prefix `sequence_backing_tokens` reports,
/// the block structure `build_decode_metadata` reconciles a group to — is taken
/// over exactly this set.
///
/// A **draft** layer is one a NextN/MTP head owns but the wave does not step:
/// it is allocated, so the head has somewhere to write, and excluded from every
/// stream-wide aggregate, so a layer holding nothing cannot speak for the ones
/// that do.
///
/// A head is only meant to sit here while it is being brought up. The finished
/// arrangement is the one `qwen35::draft` documents and `qwen35::engine::kv_layers`
/// declares: the head's pass runs over the same rows at the same positions in
/// the same wave, so its layer stands at the same length as its siblings and is
/// a **stream** layer like any other — covered by the wave's own metadata, and
/// invisible to fork, view, prefix injection, turn sealing and truncation
/// precisely because it is not a special case. A model moves from `draft` to
/// `stream_only` when its head starts stepping in lockstep, and `draft` should
/// be 0 everywhere once it does.
///
/// Folding a draft layer into a stream aggregate is silent and destructive. A
/// minimum over token prefixes reads 0 for a layer nothing has written, and the
/// wave-entry reconciler treats that as "offset ran ahead of the backing" and
/// clamps the sequence to 0 — every wave, with only a `tracing::warn!` to show
/// for it. The trunk then re-attends from position 0 and the model answers
/// fluently and wrongly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayers {
    /// Layers the wave steps together.
    pub stream: usize,
    /// Layers past the stream, each stepped by a draft head's own pass.
    pub draft: usize,
}

impl KvLayers {
    /// A model whose every KV layer advances on every wave.
    pub fn stream_only(stream: usize) -> Self {
        Self { stream, draft: 0 }
    }

    /// Backings the session allocates.
    pub fn total(&self) -> usize {
        self.stream + self.draft
    }
}
