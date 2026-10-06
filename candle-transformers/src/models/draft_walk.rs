//! The cohort draft walk — the loop every NextN/MTP drafter runs, once.
//!
//! A drafter proposes `max_len` tokens for a whole cohort by walking positions
//! forward: step `j` needs `embed(argmax)` of step `j-1`, so it is serial
//! *within* a sequence, but step `j` of one sequence is independent of step `j`
//! of every other. So the walk is one batched pass per position, and what
//! differs between models is only the arithmetic of a single step — the head's
//! own block. Everything around it is the same for every drafter, and two
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
//! * **The walk's positions are rolled back whatever happens.** A proposal is
//!   written into the head's KV as if real and then truncated away; the tokens
//!   the target accepts are written again, properly, by the next wave. A walk
//!   that failed mid-flight must not leave the head's layer longer than the
//!   trunk's — that is a length skew the next wave would "heal" by truncating a
//!   token the caller was already given.

use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::{DType, Result, Tensor, D};
use candle_nn::kv_cache::{ChunkedKvBacking, KvCache};

use super::batched_inference::BatchedInferenceSession;
use super::operand_guard::{expect_dense_view, expect_dtype};
#[cfg(feature = "cuda")]
use super::wave_buffers::upload_into;

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
/// built for `at`; each header names its sequence's RoPE rung. The model-side
/// rotation belongs to the step, not the walk — it is the one part of a
/// position that is genuinely the model's own.
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
///
/// **A walk opens no forward of its own**, and there is no hook for one. A
/// forward's transient tier stands until the *next* forward's first phase hands
/// it back, so a caller arriving between forwards — which is what a walk is —
/// already has ground for the head's attention to lay its spans in. Opening one
/// here is worse than unnecessary: a forward that owns the partition refuses
/// every arena created inside it, and a walk cannot avoid creating them, because
/// a filling chunk gets sealed into a policy-chosen format and the next step
/// then asks for a chunk of a key that did not exist when the walk began. That
/// key is not knowable in advance, so no amount of pre-ensuring reaches it.
/// Uncompressed runs survive because every key they touch already exists; C8
/// does not. See `qwen4exp::draft::mtp_draft`, which records the measurement.
// Seven operands, none of which groups with another: the session, the cohort it
// walks, where that cohort's KV lives, what it has committed, what it seeds
// from, how far to walk, and the step callback. A params struct would name the
// bundle without making any of them optional or related.
#[allow(clippy::too_many_arguments)]
pub fn draft_walk(
    session: &mut BatchedInferenceSession,
    seqs: &[usize],
    kv_layer: usize,
    committed: &[u32],
    seeds: &Tensor,
    max_len: usize,
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
    let generation = session.begin_stager_generation();
    // The walk's tokens, one row each: the committed ones the first step
    // follows, then every step's pick. One buffer the session holds, so no step
    // allocates; the picks feed the next step from where they were written, and
    // the whole block crosses the bus once at the end.
    let tokens = session.walk_tokens(max_len + 1, n)?;
    upload_tokens(&tokens.get(0)?, committed)?;
    let mut ids = tokens.get(0)?;
    let mut h = seeds.clone();

    // **The walk's launches run as a chain of graphs**, as a forward's do
    // (`docs/decode_graphs.md`): each step's block — dozens of small launches
    // over a handful of rows — is recorded and handed to the driver a segment
    // at a time, so the GPU runs a step while the host records the next. Issued
    // eagerly, the steps were launch-bound: on Flash-Next's single-session
    // decode the eager stream sat idle ~3.9 ms of every 30 ms step between its
    // own back-to-back launches. A failed walk drops the capture, discarding
    // the segment it was recording; the rollback below runs eagerly.
    let device = seeds.device();
    #[cfg(feature = "cuda")]
    let capture = match device {
        candle::Device::Cuda(cuda) => Some(cuda.begin_wave_capture()?),
        _ => None,
    };
    device.record_launches()?;

    let drafted = (|| -> Result<()> {
        for j in 0..max_len {
            let at: Vec<usize> = base.iter().map(|&b| b + j).collect();
            let overrides: Vec<(usize, usize)> =
                seqs.iter().copied().zip(at.iter().copied()).collect();
            // The head's layer alone: the trunk's layers are not being written
            // by a draft, and naming them would reconcile a group against
            // positions that do not exist on them.
            //
            // Built inside the recording: the slot-state upload is recorded
            // into the segment, and whatever must meet the device eagerly
            // pauses the recording itself.
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

            // One launch of the fused batched sampler's greedy path over the
            // cohort's `[n, vocab]` rows — the kernel the scheduler's sampler
            // runs at temperature zero — not the generic `argmax` reduction,
            // whose half-precision path addresses every element through the
            // strided-index walk. The rows are the head's dense output, so the
            // reshape is a view, and the sampler reads them from their own first
            // element; it yields `[n]` for any cohort, one row included.
            //
            // The sampler emits U32, and the next step's embedding gather
            // requires it — so it is checked rather than cast. A cast would be a
            // full pass over the ids on any backend that ever stopped emitting
            // U32, silently, once per drafted token.
            //
            // The head proposes over its whole row. A padded column can win
            // here, and costs nothing: verification gives it no probability,
            // so it is never committed.
            expect_dense_view(&logits, "draft walk logits")?;
            let vocab = logits.dim(D::Minus1)?;
            let next = tokens.get(j + 1)?;
            logits
                .reshape((n, vocab))?
                .batched_sample_argmax_into(vocab, &next)?;
            ids = next;
            h = h_next;
        }
        Ok(())
    })();
    #[cfg(feature = "cuda")]
    let drafted = match (drafted, capture) {
        (Ok(()), Some(capture)) => capture.finish(),
        (drafted, _) => drafted,
    };

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

    // The one readback: the `[max_len, n]` picks, so the whole cohort's whole
    // block crosses the bus in a single transfer, transposed on the host.
    let picks = tokens.narrow(0, 1, max_len)?.to_vec2::<u32>()?;
    Ok((0..n)
        .map(|i| picks.iter().map(|row| row[i]).collect())
        .collect())
}

/// The committed tokens into the walk buffer's first row: an upload into the
/// buffer itself on CUDA, a write of the host buffer otherwise.
fn upload_tokens(row: &Tensor, committed: &[u32]) -> Result<()> {
    expect_dtype(row, DType::U32, "draft walk tokens")?;
    #[cfg(feature = "cuda")]
    if row.device().is_cuda() {
        return upload_into(row, committed);
    }
    row.slice_set(&Tensor::new(committed, row.device())?, 0, 0)
}

/// Pre-allocate the walk's write chunks without walking.
///
/// [`draft_walk`] does this itself; a caller that reads the head layer's block
/// structure before building its step closure forces the allocation first, so
/// what it reads is the structure the walk will write into.
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
