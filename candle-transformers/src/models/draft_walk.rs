//! The cohort draft walk — the loop every NextN/MTP drafter runs, once.
//!
//! A drafter proposes `max_len` tokens for a whole cohort by walking positions
//! forward: step `j` needs `embed(argmax)` of step `j-1`, so it is serial
//! *within* a sequence, but step `j` of one sequence is independent of step `j`
//! of every other. So the walk is one batched pass per position, and what
//! differs between models is only the arithmetic of a single step — the head's
//! own block. Everything around it is the same for every drafter, and three
//! parts of it are load-bearing in ways that are invisible when they are wrong:
//!
//! * **Every drafted position's write chunk is allocated BEFORE the loop.**
//!   The loop builds its slot headers with an empty `snapshot_seqs`, so each
//!   row carries a zero-copy LIVE pointer into its `GpuChunks` buffer — sound
//!   only under the precondition `build_decode_metadata_at` states for it, "a
//!   plain decode row, whose write chunk is pre-ensured so it never reallocs".
//!   A draft walk is not that: it advances `max_len` positions, so a step that
//!   crossed a `CHUNK_SIZE` boundary would allocate mid-walk and REBUILD the
//!   buffer the previous step's header still points at. The walk deliberately
//!   never synchronises, so that step's kernel is still in flight reading the
//!   freed block, and the address is reissued almost immediately by a pool
//!   being churned tens of thousands of times per wave. It is a device-side
//!   out-of-range access, not an error return — it poisons the context and
//!   every later CUDA call fails with it. It is also invisible to
//!   `CUDA_LAUNCH_BLOCKING=1` and to compute-sanitizer, because both serialise
//!   each step before the next builds metadata, which is exactly what removes
//!   the overlap. Ensuring the whole range up front adds no allocation; it only
//!   moves it to a point where no kernel is reading.
//! * **The rope table's depth is read from the head's own layer, and refused
//!   rather than defaulted.** A zero-block table is not a small table, it is one
//!   the paged kernel indexes straight past: silent wrong RoPE on every drafted
//!   position. Speculation is lossless, so it could only ever surface as
//!   acceptance quietly collapsing. It must also be read AFTER the ensure
//!   above, which can grow the backing's block count.
//! * **The walk's positions are rolled back whatever happens.** A proposal is
//!   written into the head's KV as if real and then truncated away; the tokens
//!   the target accepts are written again, properly, by the next wave. A walk
//!   that failed mid-flight must not leave the head's layer longer than the
//!   trunk's — that is a length skew the next wave would "heal" by truncating a
//!   token the caller was already given.

use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::{DType, Result, Tensor};
use candle_nn::kv_cache::{ChunkedKvBacking, KvCache};

use super::batched_inference::BatchedInferenceSession;
use super::operand_guard::expect_dtype;

/// One position of the walk, for the whole cohort.
///
/// Given the tokens each sequence is following and the carried hidden, the
/// implementation writes the head's K/V at `at` and returns the hidden the next
/// step reads together with this step's logits. The carried hidden is opaque:
/// a head whose block runs on a plain residual passes that, one whose block
/// runs on a hyper-connection stream passes the wide residual, and the walk
/// never looks inside it.
///
/// `headers` is the head layer's slot-header buffer and its stride, already
/// built for `at`. The rope tables belong to the step, not the walk — they are
/// the one part of a position that is genuinely the model's own.
pub type DraftStep<'f> = dyn FnMut(
        &Tensor,
        &Tensor,
        &mut [&mut KvCache],
        &[usize],
        (&GpuBuf, u64),
        &Generation,
    ) -> Result<(Tensor, Tensor)>
    + 'f;

/// Walk `max_len` drafted positions for `seqs`, returning one proposal list per
/// sequence in order.
///
/// `committed` is the token each sequence is following into its first drafted
/// position, and `seeds` the carried hidden to start from — one row per
/// sequence, already stacked.
///
/// The head's KV is left exactly as it was found.
/// `prepare` runs **after** the walk has reserved its storage and **before**
/// the first step, and whatever it returns is held for the walk's lifetime.
///
/// That ordering is the whole reason it is a hook rather than something the
/// caller does around the call. A model that needs a transient arena tier for
/// the head's attention opens a forward here — and a forward that owns the
/// partition refuses any arena created inside it, because the wave's storage is
/// claimed before the forward opens. Reserve first, then open: the other order
/// works only for as long as every arena the walk touches happens to exist
/// already, which is true at low compression and false at C8.
// Eight operands, none of which groups with another: the session, the cohort it
// walks, where that cohort's KV lives, what it has committed, what it seeds
// from, how far to walk, and the two callbacks. A params struct would name the
// bundle without making any of them optional or related.
#[allow(clippy::too_many_arguments)]
pub fn draft_walk<G>(
    session: &mut BatchedInferenceSession,
    seqs: &[usize],
    kv_layer: usize,
    committed: &[u32],
    seeds: &Tensor,
    max_len: usize,
    prepare: impl FnOnce() -> Result<G>,
    step: &mut DraftStep<'_>,
) -> Result<Vec<Vec<u32>>> {
    let n = seqs.len();
    if committed.len() != n {
        candle::bail!(
            "draft walk: {n} sequences against {} committed tokens",
            committed.len()
        );
    }
    if n == 0 || max_len == 0 {
        return Ok(vec![Vec::new(); n]);
    }
    let dev = seeds.device().clone();

    // The head ropes on ABSOLUTE sequence positions, like every trunk layer:
    // its history is the sequence's, one row per token, so a drafted position
    // is `offset + step` and the verify wave that replaces it ropes the same
    // token at the same place.
    let base: Vec<usize> = seqs
        .iter()
        .map(|&s| session.sequence_offset(s).unwrap_or(0))
        .collect();

    // Pre-ensure the whole walk's write chunks — see the module docs.
    //
    // **Both allocators, not just the obvious one.** `ensure_for_offset` covers
    // the span the walk writes, but the per-step `build_decode_metadata_at`
    // below has an allocator of its own — `ensure_for_batch_entries_all`, which
    // sizes each step's write chunk as it builds that step's headers. Running
    // inside the loop it is normally a no-op, because the chunks are already
    // there; when it is not, it is an arena created inside a forward that owns
    // the partition, which is refused outright. That is invisible until a
    // configuration turns up where the arena genuinely does not exist yet —
    // compressed KV at width, where the formats in play are not the ones an
    // uncompressed run created.
    //
    // So every step's entries are ensured here, where no forward is open and a
    // refusal cannot happen. Inside the loop they are then no-ops by
    // construction rather than by luck.
    if let Some(backing) = session.backing(kv_layer) {
        for (i, &s) in seqs.iter().enumerate() {
            backing.ensure_for_offset(s, base[i], max_len)?;
        }
        let group = std::slice::from_ref(backing);
        for j in 0..max_len {
            let entries: Vec<(usize, usize)> =
                seqs.iter().zip(&base).map(|(&s, &b)| (s, b + j)).collect();
            ChunkedKvBacking::ensure_for_batch_entries_all(group, &entries, 1)?;
        }
    }

    // Everything that allocates has now run; from here the walk only computes.
    let _prepared = prepare()?;

    let generation = session.begin_stager_generation();
    let mut ids = Tensor::from_vec(committed.to_vec(), n, &dev)?;
    let mut h = seeds.clone();
    let mut steps: Vec<Tensor> = Vec::with_capacity(max_len);

    let drafted = (|| -> Result<()> {
        for j in 0..max_len {
            let at: Vec<usize> = base.iter().map(|&b| b + j).collect();
            let overrides: Vec<(usize, usize)> =
                seqs.iter().copied().zip(at.iter().copied()).collect();
            // The head's layer alone: the trunk's layers are not being written
            // by a draft, and naming them would reconcile a group against
            // positions that do not exist on them.
            let (headers, stride) = session.build_decode_metadata_at(
                kv_layer..kv_layer + 1,
                seqs,
                &generation,
                &overrides,
                &[],
                &[],
            )?;
            // The decode path does NOT build metadata per call — it dereferences
            // whatever pointer the headers resolve to, and `None` resolves to
            // literal 0. That is an illegal address on device, not an error
            // return, so it is worth one branch here.
            let headers = headers.ok_or_else(|| {
                candle::Error::Msg(format!(
                    "draft walk: no slot headers for {n} sequences at step {j} — the decode \
                     kernel would dereference a null table"
                ))
            })?;

            let (h_next, logits) = {
                let mut data = session.caches_for_sequences_mut(seqs);
                if data.len() != n {
                    candle::bail!(
                        "draft walk: {} of {n} sequences still have live slots",
                        data.len()
                    );
                }
                let mut caches: Vec<&mut KvCache> = data
                    .iter_mut()
                    .map(|(_, _, c)| &mut c.caches[kv_layer])
                    .collect();
                let out = step(&ids, &h, &mut caches, &at, (&headers, stride), &generation)?;
                // The decode kernel commits its write on the device; the host
                // block table is advanced here, the way the wave driver advances
                // every layer of a decode row. The next step's metadata is built
                // against this length.
                for (c, &p) in caches.iter_mut().zip(&at) {
                    c.set_current_seq_len(p + 1)?;
                }
                out
            };

            // `argmax_keepdim`, not `argmax`: the latter drops the axis, and a
            // one-row cohort would come back rank-0 rather than `[1, 1]`.
            //
            // The reduction already emits U32 on both backends, and the next
            // step's embedding gather requires it — so it is checked rather than
            // cast. A cast would be a full pass over the ids on any backend that
            // ever stopped emitting U32, silently, once per drafted token.
            let next = logits.argmax_keepdim(candle::D::Minus1)?;
            expect_dtype(&next, DType::U32, "draft walk argmax")?;
            ids = next.flatten_all()?;
            steps.push(ids.clone());
            h = h_next;
        }
        Ok(())
    })();

    // Roll the walk's positions away, error or not — see the module docs.
    let mut rolled_back = Ok(());
    for (i, &seq) in seqs.iter().enumerate() {
        if let Some(caches) = session.sequence_caches_mut(seq) {
            if let Some(c) = caches.caches.get_mut(kv_layer) {
                let r = c.truncate_to_offset(base[i]);
                if rolled_back.is_ok() {
                    rolled_back = r;
                }
            }
        }
    }
    drafted?;
    rolled_back?;

    // The one readback: `[n, max_len]`, so the whole cohort's whole block
    // crosses the bus in a single transfer.
    let refs: Vec<&Tensor> = steps.iter().collect();
    Tensor::stack(&refs, 1)?.to_vec2::<u32>()
}

/// The arena depth a drafter's rope tables must cover, read from the head's own
/// KV layer.
///
/// Refused rather than defaulted, for the reason the module docs give: a
/// zero-block table is one the paged kernel indexes straight past. Call it
/// AFTER [`draft_walk`]'s pre-ensure has run — which is why it is exposed
/// separately rather than folded into the walk, since the step closure needs
/// the answer before the walk begins.
pub fn draft_rope_depth(
    session: &BatchedInferenceSession,
    seqs: &[usize],
    kv_layer: usize,
) -> Result<usize> {
    let seq = *seqs
        .first()
        .ok_or_else(|| candle::Error::msg("draft rope depth: empty cohort"))?;
    session
        .sequence_caches(seq)
        .and_then(|c| c.caches.get(kv_layer))
        .map(|k| k.k_cache().chunked_max_blocks())
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "draft walk: sequence {seq} has no live KV layer {kv_layer} to size the rope \
                 table from — it was released after the cohort was formed"
            ))
        })
}

/// Pre-allocate the walk's write chunks without walking.
///
/// [`draft_walk`] does this itself, but a caller that needs
/// [`draft_rope_depth`] before building its step closure has to force the
/// allocation first — the ensure can grow the backing's block count, and a rope
/// table sized before it is a table the walk indexes past.
pub fn draft_reserve(
    session: &BatchedInferenceSession,
    seqs: &[usize],
    kv_layer: usize,
    max_len: usize,
) -> Result<()> {
    if let Some(backing) = session.backing(kv_layer) {
        for &s in seqs {
            let base = session.sequence_offset(s).unwrap_or(0);
            backing.ensure_for_offset(s, base, max_len)?;
        }
    }
    Ok(())
}
