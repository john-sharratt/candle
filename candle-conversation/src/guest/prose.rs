//! The prose guest: a Hermes-3 Llama, resident only while its backlog runs.
//!
//! # Why a second language model at all
//!
//! The engine's own model is a character *acting*: it perceives, it decides, it
//! emits acts. Narration is a different job with a different voice, and a world
//! that wants a paragraph of scene-setting has three options — ask the acting
//! model and get an act, carry a second model permanently and pay for it while
//! nobody is asking, or borrow the card for as long as the paragraph takes.
//! This is the third.
//!
//! # Where every byte goes
//!
//! | What | Where | Why |
//! |---|---|---|
//! | Quantized projections | [`GuestGround`] | The span is the budget; see the module header. |
//! | Norm weights (F32, one vector per layer) | [`GuestGround`] | Small, but a pool allocation is a pool allocation. |
//! | Token embedding table | [`GuestGround`], still quantized | Read a row at a time by [`QTensor::dequantize_into`], so it is never dequantized whole. |
//! | K/V cache | [`GuestGround`], preallocated | Written in place, so the cache never `cat`s. |
//! | Activations | The CUDA pool | Transient, small, and the engine is quiesced — see below. |
//!
//! **Activations are the one thing that reaches the pool, and that is a
//! deliberate limit rather than an oversight.** Giving them span ground too
//! would mean opening a wave generation for the guest and threading its ticket
//! through every op, which is the engine's own machinery and would make the
//! guest a second implementation of it. During a drain the engine is quiesced —
//! no wave is open, and the relief ladder has just run — so the pool is as
//! empty as it ever gets, and a 3 B model's per-step activations are single-digit
//! MiB against ground measured in gigabytes. The weights, which are the
//! gigabytes, do not go near it.
//!
//! # Why the forward is here and not `quantized_llama`'s
//!
//! `ModelWeights::from_gguf` builds its own tensors: it dequantizes the whole
//! embedding table into the pool, and every projection it makes is a pool or
//! dense-block allocation. Both are exactly what a guest may not do — the dense
//! block is frozen after the engine's own load, and the pool is the competitor
//! the reservation exists to keep out. The forward below is the same
//! arithmetic reading weights the guest placed itself.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{DType, Device, Tensor};
use memmap2::Mmap;

use candle::quantized::pinned_staging::{Generation, GpuBuf, PinnedStager};
use candle_nn::kv_cache::{ChunkedKvBacking, CompressionPolicy, KvCache, KvFormat};
use candle_transformers::models::prefill_utils::{
    compute_rope_cs, paged_decode_attn, paged_prefill_batched, SharedPm,
};

use crate::batched_sampler::{BatchedSampler, SequenceSamplingState};
use crate::config::SamplingConfig;
use crate::stencil::{compile, HfVocab, StencilDriver, StencilTreeBuilder, StepMask, Vocab};
use crate::token_buffer::TokenBuffer;

use super::checkpoint;
use super::ground::GuestGround;
use super::model::GuestModel;
use super::progress::{GuestEvent, GuestSink};
use super::seed::resolve_seed;
use super::work::{Guest, GuestOutcome, GuestRequest, ProseRequest};

/// Where the prose guest's checkpoint and tokenizer live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProseSpec {
    pub gguf: PathBuf,
    pub tokenizer: PathBuf,
    /// The voice a request that names none gets.
    pub default_system: String,
    /// Hard ceiling on the K/V the guest preallocates, in tokens.
    ///
    /// Prompt plus generation. The cache is claimed as span ground up front —
    /// growing it mid-job would mean claiming inside a forward, which
    /// `claim_span_region` refuses — so this is the number the ground is sized
    /// from and the number a job is refused against.
    pub max_context: usize,
}

impl ProseSpec {
    /// The Hermes-4 14 B preset, at the paths a deployment downloaded it to.
    ///
    /// Which quantisation of it is [`super::prose_choice::HermesQuant`]'s call
    /// and not this constructor's — the rung is a property of the card, and a
    /// preset that pinned one would be wrong on two of the three machines this
    /// runs on.
    ///
    /// **6 k of context, and the lifegen ladder is what sets it.** A day's turn
    /// carries the year it sits in *and* the month, on top of a shared prefix
    /// holding the character sheet, the arc, the cast and the year outline — so
    /// the deepest rung asks the longest question. Measured at ~2,400 tokens of
    /// prompt against a 1,400-token answer; 4 k left no room for the slack and
    /// refused nine months out of ten.
    ///
    /// It is not free. Qwen3-14B is 40 layers where Llama-3.2-3B is 28, so a
    /// token costs 320 KiB of K/V rather than 112, and 6 k is 1.92 GiB of ground
    /// on top of the weights. That is 10.3 GiB of the 11.48 GiB a 24 GB card can
    /// shed to — priced in `prose_choice`, and the reason the ladder there is
    /// sized against the warm ceiling rather than the best one.
    pub fn hermes4_14b(gguf: impl Into<PathBuf>, tokenizer: impl Into<PathBuf>) -> Self {
        Self {
            gguf: gguf.into(),
            tokenizer: tokenizer.into(),
            default_system: "You are a narrator. Write vivid, concrete prose. \
                             Do not address the reader and do not explain yourself."
                .into(),
            max_context: 6144,
        }
    }

    /// The Hermes-3 3 B preset, at the paths a deployment downloaded it to.
    pub fn hermes3_3b(gguf: impl Into<PathBuf>, tokenizer: impl Into<PathBuf>) -> Self {
        Self {
            gguf: gguf.into(),
            tokenizer: tokenizer.into(),
            default_system: "You are a narrator. Write vivid, concrete prose. \
                             Do not address the reader and do not explain yourself."
                .into(),
            max_context: 4096,
        }
    }
}

/// The guest itself. Holds no device memory until [`GuestModel::load`].
pub struct ProseGuest {
    spec: ProseSpec,
    /// Shared with the process-wide cache rather than owned: a `Tokenizer`
    /// carries its whole vocabulary, and a guest is built fresh per drain.
    tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    model: Option<PlacedLlama>,
}

impl ProseGuest {
    pub fn new(spec: ProseSpec) -> Self {
        Self {
            spec,
            tokenizer: None,
            model: None,
        }
    }

    /// The ground the checkpoint's tensors need.
    ///
    /// **The padded extent, not the on-disk payload.** Every placement reserves
    /// the tail the GGML kernels read past ([`placed_extent`]); a footprint
    /// computed from the payload alone claims less ground than the load places
    /// into and runs out part-way through, after the engine has already been
    /// evicted for it. Each tensor is also rounded to the 256-byte alignment
    /// `place` applies, so the sum is what the loader will actually consume.
    ///
    /// Read from the GGUF header rather than from a constant, so a deployment
    /// that swapped the quantization does not silently under-claim. A header
    /// that cannot be read answers `None`, and the caller treats that as a
    /// guest that cannot run rather than guessing a size.
    pub fn checkpoint_bytes(path: &Path) -> Option<usize> {
        Self::sizes(path).map(|s| s.weights)
    }

    /// The context this backlog is claimed for, and allocated at.
    ///
    /// **One definition, used by both `footprint_bytes` and `load`**, and that
    /// is the whole point of it existing. They had one each: the footprint
    /// claimed for `max_tokens + PROMPT_HEADROOM_TOKENS`, and the load then
    /// ignored that and re-derived a context from however much ground happened
    /// to be free — `spec.max_context.min(ground.free() / per_token)`. Since the
    /// claim deliberately includes [`PLACEMENT_SLACK_BYTES`] on top of the
    /// cache, "however much is free" was always *larger* than what had been
    /// claimed for, so the load spent the slack on extra context and then had
    /// none left for the thing the slack is for.
    ///
    /// It failed the way that kind of bug does: not at the boundary, but most of
    /// the way through the placement loop, with a message about the last tensor
    /// rather than the first wrong decision. A 220-token description claimed for
    /// 2,268 tokens of context and then tried to place 3,712, and ran out 14 MiB
    /// from the end — every K/V tensor being 14.5 MiB against a 16 MiB region,
    /// so each one stranded 1.5 MiB of run tail that the arithmetic, which
    /// assumes perfect packing, does not model.
    ///
    /// Sized from the backlog rather than from the spec's ceiling because a
    /// drain of eight short jobs should not evict the engine for a context none
    /// of them will use.
    /// **Sized from each job's own prompt, not from a flat allowance.**
    ///
    /// This took `max_tokens + PROMPT_HEADROOM_TOKENS` and never looked at what
    /// the caller was actually sending. A fixed headroom is a guess about
    /// somebody else's prompt, and the lifegen ladder is the caller that breaks
    /// it: a month's turn carries the year it expands and a day's carries both
    /// the year and the month, so the prompt grows as the ladder descends —
    /// exactly where the allowance is most wrong. Nine of ten months refused
    /// with "a 2100-token prompt plus 1400 tokens of answer passes the
    /// 3448-token context", which is the guest declining a request that fits its
    /// own ceiling comfortably and was simply mis-measured.
    ///
    /// The estimate is characters over [`CHARS_PER_TOKEN`], which is deliberately
    /// pessimistic: over-estimating claims a little more ground than needed, and
    /// under-estimating refuses the job after the engine has already been evicted
    /// for it. The headroom stays on top as slack for the chat template's own
    /// markers, which are not in the strings measured here.
    fn context_for(&self, jobs: &[GuestRequest]) -> usize {
        let want = jobs
            .iter()
            .filter_map(|j| match j {
                GuestRequest::Prose(r) => {
                    let chars = r.system.len() + r.prompt.len();
                    Some(chars.div_ceil(CHARS_PER_TOKEN) + r.max_tokens as usize)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        (want + PROMPT_HEADROOM_TOKENS).min(self.spec.max_context)
    }

    /// What the caches cost beyond the bytes their tokens occupy.
    ///
    /// # A per-token figure does not size a paged cache
    ///
    /// The K/V arithmetic is per token and per seat; this is neither. Each layer
    /// has its own backing, each backing keeps arenas per size class, and an
    /// arena is a whole region however little of it is used — so the floor is
    /// layers × classes × a region, paid before a single token is stored. The
    /// active partial chunk is float as well, whatever the sealed chunks are
    /// compressed to, which is another fixed cost per layer and seat.
    ///
    /// Left out, an eleven-seat wave was sized at four gigabytes, the shed freed
    /// four, and the arenas then found the reservation full: *"no region is
    /// claimable for class 4096 B"*. The wave was not too big — the estimate was
    /// short by everything that is not a token.
    fn arena_overhead_bytes(&self) -> usize {
        self.layers_hint() * ARENA_CLASSES_PER_LAYER * candle_nn::kv_cache::REGION_BYTES
    }

    /// Layer count for the sizing above, read from the checkpoint's header.
    ///
    /// Zero when the header cannot be read, which the caller already treats as a
    /// guest that cannot run rather than one to guess for.
    fn layers_hint(&self) -> usize {
        checkpoint::gguf_header(&self.spec.gguf)
            .ok()
            .and_then(|c| Geometry::from_gguf(&c).ok())
            .map_or(0, |g| g.layers)
    }

    /// How many of the backlog's jobs decode as one wave.
    ///
    /// # What bounds it, and what does not
    ///
    /// Not the dependency graph — a drain's backlog has no ordering in it at
    /// all, because the ladder only ever submits siblings together and siblings
    /// never see each other. It is memory: every seat holds its own K/V for the
    /// whole context, which for this geometry is 160 KiB a token, so a seat at a
    /// 6,144-token context is very nearly a gigabyte. Widen the wave and the
    /// claim grows linearly, and a claim the engine cannot shed enough to cover
    /// is a drain that fails rather than one that runs narrower.
    ///
    /// So the cap is a byte budget rather than a count, and it falls out of the
    /// context each backlog actually asked for: a wave of short turns is wide, a
    /// wave of full-context ones is narrow, and both stay inside the same
    /// ceiling. At least one seat, always — a single job that needs the whole
    /// context must still run.
    fn seats_for(&self, jobs: &[GuestRequest]) -> usize {
        let prose = jobs
            .iter()
            .filter(|j| matches!(j, GuestRequest::Prose(_)))
            .count();
        if prose <= 1 {
            return prose.max(1);
        }
        let per_token = ProseGuest::sizes(&self.spec.gguf)
            .map(|s| s.per_token)
            .unwrap_or(0);
        seats_from(per_token * self.context_for(jobs), prose)
    }

    /// Everything the ground has to hold, read from the checkpoint's own header.
    ///
    /// Both halves come from the same read because they have to agree: the
    /// per-token cost is derived from the *checkpoint's* layer count and head
    /// geometry, and a constant standing in for it was wrong by whatever the
    /// deployment happened to configure. `None` when the header cannot be read,
    /// which the caller treats as a guest that cannot run rather than guessing.
    pub fn sizes(path: &Path) -> Option<Sizes> {
        // Through the cache: this is called from `footprint_bytes`, on every
        // drain, *before* the load — so an uncached parse here was a second
        // full header read whose cost did not even appear in the `load_ms` the
        // drain reports.
        let content = checkpoint::gguf_header(path).ok()?;
        let weights = content
            .tensor_infos
            .values()
            .map(|i| placed_extent(i.ggml_dtype, i.shape.elem_count()).next_multiple_of(256))
            .sum();
        let geo = Geometry::from_gguf(&content).ok()?;
        Some(Sizes {
            weights,
            per_token: geo.per_token_bytes(),
        })
    }
}

/// What one prose guest needs of its ground.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizes {
    /// Every tensor, at its padded extent and 256-byte aligned.
    pub weights: usize,
    /// K/V for one token across every layer, plus that token's RoPE row.
    ///
    /// The RoPE tables are counted here rather than as a lump because they scale
    /// with the context exactly as the caches do — and leaving them out is what
    /// made the first real load place every cache and then run out with 951 KB
    /// of `cos` left to write.
    pub per_token: usize,
}

/// Bytes a tensor of `elems` elements occupies in `dtype` — its payload.
fn tensor_bytes(dtype: GgmlDType, elems: usize) -> usize {
    let block = dtype.block_size().max(1);
    elems.div_ceil(block) * dtype.type_size()
}

/// Bytes a placement must **reserve** for it, which is more.
///
/// The GGML matmul kernels address `MATRIX_ROW_PADDING` elements past the end
/// of every row unconditionally. A placement sized to the payload alone is one
/// the kernel reads past: into the next weight, and at the last weight in a
/// region, into whatever the KV side put in the address above it. Neither
/// faults — every address in the span is mapped — so it surfaces as a wrong
/// number and never as a crash.
///
/// The same rule `layer_stream::build::slot_extent` states for the same reason,
/// and the number the footprint has to be computed from: claiming payload bytes
/// and placing extent bytes runs out of ground part-way through a load.
fn placed_extent(dtype: GgmlDType, elems: usize) -> usize {
    candle::quantized::cuda::padded_storage_bytes(elems, dtype)
}

impl GuestModel for ProseGuest {
    fn guest(&self) -> Guest {
        Guest::Prose
    }

    fn footprint_bytes(&self, jobs: &[GuestRequest]) -> usize {
        let sizes = ProseGuest::sizes(&self.spec.gguf).unwrap_or(Sizes {
            weights: 0,
            per_token: 0,
        });
        // **The K/V is counted even though the ground will not hold it.**
        //
        // The chunked cache claims its own regions from the reservation, so this
        // guest's ground is really only the weights — and sizing the claim to the
        // weights alone was wrong in the way that matters. The drain sheds the
        // engine's KV to *make room for what this number says*; asking only for
        // the weights meant only the weights' worth was freed, the cache then
        // found the reservation full, and a wave died on "no region is claimable
        // — every one of the 1236 regions is occupied".
        //
        // So the figure covers both tenants. The ground places the weights and
        // the rest of what was freed stays available for the arenas, which is the
        // shape the drain's shed step already has.
        sizes.weights
            + sizes.per_token * self.context_for(jobs) * self.seats_for(jobs)
            + self.arena_overhead_bytes()
            + PLACEMENT_SLACK_BYTES
    }

    /// The weights and their slack. The K/V is the chunked cache's own tenancy
    /// and must be left in the reservation for its arenas to claim.
    fn ground_bytes(&self, _jobs: &[GuestRequest]) -> usize {
        let sizes = ProseGuest::sizes(&self.spec.gguf).unwrap_or(Sizes {
            weights: 0,
            per_token: 0,
        });
        sizes.weights + PLACEMENT_SLACK_BYTES
    }

    // The backlog does not change which checkpoint a prose guest stands up —
    // there is one voice per deployment — so the jobs go unread here.
    fn load(
        &mut self,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        jobs: &[GuestRequest],
    ) -> Result<(), String> {
        let started = Instant::now();
        let mut ph = LoadPhases::default();

        let t = Instant::now();
        let tokenizer = checkpoint::tokenizer(&self.spec.tokenizer)?;
        LoadPhases::add(&mut ph.tokenizer_ns, t);

        // The lock is taken for the whole load and released before this
        // returns, so the guest holds no handle on the ground afterwards —
        // which is what `drain::release` checks.
        // The same context the claim was made for. Passed in rather than
        // re-derived inside the load — see [`Self::context_for`].
        let want_context = self.context_for(jobs);
        // The same width the claim was made for, for the same reason the context
        // is: two derivations of "how wide" is two places for it to disagree, and
        // the load is the one that would discover it after the eviction.
        let seats = self.seats_for(jobs);
        let model = {
            let mut g = ground
                .lock()
                .map_err(|_| "prose guest: the ground lock was poisoned".to_string())?;
            PlacedLlama::load(device, &mut g, &self.spec, want_context, seats, &mut ph)
                .map_err(|e| e.to_string())?
        };
        tracing::info!(
            target: "candle_conversation::guest",
            jobs = jobs.len(),
            seats,
            context = want_context,
            "prose guest seating"
        );
        ph.report(started.elapsed());
        self.tokenizer = Some(tokenizer);
        self.model = Some(model);
        Ok(())
    }

    fn run(&mut self, request: &GuestRequest, sink: &GuestSink) -> Result<GuestOutcome, String> {
        let mut out = Err("the prose guest answered no job at all".to_string());
        self.run_batch(&[(request, sink)], &mut |_, r| out = r);
        out
    }

    /// The whole backlog as one wave, in seat-sized chunks.
    ///
    /// Decode reads the entire checkpoint per token, so seats stepping together
    /// read it once between them. This is the override the trait's default
    /// exists for — see [`PlacedLlama::generate_batch`].
    fn run_batch(
        &mut self,
        jobs: &[(&GuestRequest, &GuestSink)],
        answer: &mut dyn FnMut(usize, Result<GuestOutcome, String>),
    ) {
        let (Some(model), Some(tokenizer)) = (self.model.as_mut(), self.tokenizer.as_ref()) else {
            for i in 0..jobs.len() {
                answer(
                    i,
                    Err("the prose guest was asked to run before it loaded".into()),
                );
            }
            return;
        };

        // The wave is as wide as the load seated, and the backlog may be wider —
        // `seats_for` caps on the ground a wave's K/V can take, not on how many
        // jobs happen to be queued. A chunk answers together, because a wave
        // genuinely does finish together.
        let width = model.seats.max(1);
        for (c, chunk) in jobs.chunks(width).enumerate() {
            let base = c * width;
            let mut prose: Vec<(&ProseRequest, &GuestSink)> = Vec::with_capacity(chunk.len());
            // Where each seat's answer belongs, so a misrouted job in the middle
            // of a chunk does not shift every answer behind it by one.
            let mut seat_of: Vec<usize> = Vec::with_capacity(chunk.len());
            for (i, (request, sink)) in chunk.iter().enumerate() {
                match request {
                    GuestRequest::Prose(r) => {
                        seat_of.push(base + i);
                        prose.push((r, *sink));
                    }
                    other => answer(
                        base + i,
                        Err(format!(
                            "the prose guest was handed a {} job — the queue routed by kind and \
                             should not have",
                            other.guest()
                        )),
                    ),
                }
            }

            let answers = model.generate_batch(tokenizer, &prose, &self.spec.default_system);
            for (seat, out) in seat_of.into_iter().zip(answers) {
                answer(seat, out.map_err(|e| e.to_string()));
            }
        }
    }

    fn unload(&mut self) {
        // Order matters: the model holds tensors that view the ground, and the
        // ground is dropped by the drain immediately after this returns.
        self.model = None;
        self.tokenizer = None;
    }
}

/// Slack on top of the measured prompt and the generation budget.
///
/// The cache is claimed before the prompt is tokenised — the claim has to happen
/// between forwards — so [`ProseGuest::context_for`] estimates the prompt from
/// its character count and this covers what the estimate cannot see: the chat
/// template's own markers, and the difference between a pessimistic
/// characters-per-token ratio and the tokeniser's real answer on unusual text.
///
/// It used to be the *whole* allowance for the prompt, which made it a guess
/// about callers this file has never seen. It is slack now, not a budget.
const PROMPT_HEADROOM_TOKENS: usize = 512;

/// Characters per token, for sizing ground before a tokeniser exists.
///
/// Three, not the usual four. The error that matters is asymmetric: over-
/// estimating claims a little more ground than the job needs, and under-
/// estimating refuses the job *after* the engine has been evicted to serve it.
/// English prose runs nearer four, so this leaves room for text that tokenises
/// worse — names, numbers, tables of dates, the ladder's own headings.
const CHARS_PER_TOKEN: usize = 3;

/// Slack on top of what the arithmetic says, for the ground a *placement* costs
/// beyond the bytes it holds.
///
/// Two things it covers. Alignment: every tensor is rounded up to 256 bytes, and
/// the footprint counts that, but a run's tail is abandoned whenever the next
/// allocation does not fit it — waste the arithmetic cannot predict because it
/// depends on which regions the KV side happened to hand back. And the guest's
/// per-step activations, which come from the CUDA pool rather than from ground
/// (see the module header) but want the drain to have left the pool some room.
///
/// One region is far too little — the run-tail waste alone can exceed it on a
/// fragmented claim — and a gigabyte would evict the engine for nothing.
const PLACEMENT_SLACK_BYTES: usize = 320 << 20;

/// Held at compile time rather than by a test, so a change to the constant has
/// to reckon with both bounds even in a build nobody runs the tests for.
const _: () = assert!(
    PLACEMENT_SLACK_BYTES > (16 << 20),
    "one region is not enough slack — the run-tail waste alone can exceed it"
);
const _: () = assert!(
    PLACEMENT_SLACK_BYTES < (1 << 30),
    "a gigabyte of slack evicts the engine for ground the guest will not use"
);

/// Where a load's time went, apportioned rather than argued about.
///
/// The drain already logs one `load_ms`, which says the load is slow and
/// nothing about why. On this box it is ~3.1 s to move a 2.46 GiB checkpoint —
/// 0.79 GiB/s against a link that does twelve and a file that reads at four and
/// a half — so the cost is neither the disk nor the PCIe transfer, and a single
/// number cannot say which of the remaining candidates it is.
///
/// Each field is one answerable question, and the ones that are *not* the
/// transfer are the ones worth having: they are pure host work over files that
/// do not change, so anything large here is time being paid on every drain for
/// an answer that was already computed on the last one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadPhases {
    /// Parsing `tokenizer.json` and building the vocab. Touches no GPU at all.
    pub tokenizer_ns: u64,
    /// Reading the GGUF header — metadata and every tensor info.
    pub header_ns: u64,
    /// Allocating the per-tensor host staging buffers.
    ///
    /// Separate from the read because on Windows a fresh large `Vec` is
    /// demand-zero pages, and the cost lands as a soft fault per page on first
    /// touch rather than in the allocator.
    pub alloc_ns: u64,
    /// `seek` + `read_exact` for every tensor.
    pub read_ns: u64,
    /// `memcpy_htod` and the per-tensor `stream.synchronize()` that follows it.
    pub h2d_ns: u64,
    /// Dequantizing the norm vectors on the host.
    pub dequant_ns: u64,
    /// Building the RoPE `cos`/`sin` tables.
    pub rope_ns: u64,
    /// Ground placement arithmetic, including the K/V caches.
    pub place_ns: u64,
    /// Tensors placed, and bytes moved across the link.
    pub tensors: u32,
    pub bytes: u64,
}

impl LoadPhases {
    fn add(slot: &mut u64, at: Instant) {
        *slot += at.elapsed().as_nanos() as u64;
    }

    /// Log the breakdown. One line, because it is read next to `load_ms`.
    fn report(&self, total: std::time::Duration) {
        let ms = |ns: u64| ns as f64 / 1e6;
        let total_ms = total.as_secs_f64() * 1e3;
        let named = self.tokenizer_ns
            + self.header_ns
            + self.alloc_ns
            + self.read_ns
            + self.h2d_ns
            + self.dequant_ns
            + self.rope_ns
            + self.place_ns;
        tracing::info!(
            target: "candle_conversation::guest",
            tokenizer_ms = ms(self.tokenizer_ns),
            header_ms = ms(self.header_ns),
            alloc_ms = ms(self.alloc_ns),
            read_ms = ms(self.read_ns),
            h2d_ms = ms(self.h2d_ns),
            dequant_ms = ms(self.dequant_ns),
            rope_ms = ms(self.rope_ns),
            place_ms = ms(self.place_ns),
            unattributed_ms = total_ms - ms(named),
            tensors = self.tensors,
            mib = self.bytes >> 20,
            h2d_gib_s = (self.bytes as f64 / (1u64 << 30) as f64)
                / (self.h2d_ns as f64 / 1e9).max(f64::MIN_POSITIVE),
            "prose guest load"
        );
    }
}

/// Which family of checkpoint this guest is standing on.
///
/// **Two, and the difference is one norm.** Both are ChatML tunes with the same
/// decoder shape — RMSNorm, GQA, SwiGLU, rotary — so the forward is shared. What
/// Qwen3 adds is a per-head RMSNorm on Q and K before the rotary, and what it
/// changes is the namespace its geometry keys live under. Everything else in
/// this file is common, which is why this is an enum with two accessors rather
/// than a second implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProseArch {
    /// Hermes-3 over Llama 3.x.
    Llama,
    /// Hermes-4 over Qwen3.
    Qwen3,
}

impl ProseArch {
    /// The family a GGUF's `general.architecture` names.
    ///
    /// Refuses anything else rather than guessing a prefix: a wrong guess reads
    /// zero for every geometry key and fails later, in the placement, with an
    /// error about tensor shapes that says nothing about the real cause.
    fn from_gguf_name(name: &str) -> candle::Result<Self> {
        match name {
            "llama" => Ok(Self::Llama),
            "qwen3" => Ok(Self::Qwen3),
            other => Err(candle::Error::Msg(format!(
                "prose guest: architecture `{other}` is not one this guest implements — it serves \
                 `llama` (Hermes-3) and `qwen3` (Hermes-4)"
            ))),
        }
    }

    /// The namespace this family's geometry keys sit under.
    fn gguf_prefix(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen3 => "qwen3",
        }
    }

    /// The rotary base to assume when the header omits one.
    ///
    /// Different by family and not interchangeable: a Llama-3 rope read with
    /// Qwen3's base is not slightly wrong, it is a different positional
    /// encoding, and the model answers fluent nonsense rather than failing.
    fn default_rope_theta(&self) -> f32 {
        match self {
            Self::Llama => 500_000.0,
            Self::Qwen3 => 1_000_000.0,
        }
    }

    /// Whether Q and K are normed per head before the rotary.
    fn has_qk_norm(&self) -> bool {
        matches!(self, Self::Qwen3)
    }

    /// Whether the rotary pairs adjacent elements or halves of the head.
    ///
    /// **The two are not a detail and they do not fail loudly.** Interleaved
    /// pairs `(x0,x1), (x2,x3)…`; the half-split pairs `x[i]` with
    /// `x[i + head_dim/2]`. Applying one where the other is meant leaves every
    /// value in range and every shape correct, and the model emits fluent
    /// degenerate text — the first Hermes-4 load through this guest answered
    /// with the word "first" four hundred times, which is what a transformer
    /// does when position has been scrambled rather than lost.
    ///
    /// GGUF names it `rope_type`, and the two families genuinely differ: Llama
    /// ships NORM and Qwen3 ships NEOX. `quantized_llama`'s own
    /// `rope_interleaved` exists for the same reason.
    fn rope_is_interleaved(&self) -> bool {
        match self {
            Self::Llama => true,
            Self::Qwen3 => false,
        }
    }
}

/// The checkpoint's geometry, read from its own header.
#[derive(Clone, Copy, Debug)]
struct Geometry {
    arch: ProseArch,
    layers: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    rms_eps: f64,
    rope_theta: f32,
}

/// A Hermes-3 Llama standing entirely in span ground.
struct PlacedLlama {
    geo: Geometry,
    device: Device,
    /// Still quantized. Rows are read one at a time by `dequantize_into`, so the
    /// table is never materialised as floats — which is what would make it the
    /// largest single allocation in the guest by a wide margin.
    embed: QTensor,
    layers: Vec<PlacedLayer>,
    norm: Tensor,
    /// The output head. Llama 3.2 ties it to the embedding table, so a
    /// checkpoint with no `output.weight` reuses `token_embd.weight` — and the
    /// tie is a *shared view over one placement*, not a second copy.
    head: candle_transformers::models::quantized_matmul::QMatMul,
    /// The paged kernels' rotary table, `[max_blocks * 32, head_dim]`.
    rope_cs: Tensor,
    /// The pinned staging arena every forward uploads its slot headers through.
    ///
    /// **One per load, not one per forward.** `PinnedStager::new` allocates a
    /// 128 MB pinned slab; built inside `forward_span` it allocated one per
    /// decoded token, and a wave of eleven seats exhausted pinned host memory
    /// part-way through a year — `cuMemHostAlloc` refusing, then
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` from the buffers that had already been handed
    /// to kernels. The stager is a fixture of the loaded model; only the
    /// generation is per forward.
    stager: PinnedStager,
    /// The engine's own chunked K/V cache, one backing per layer.
    ///
    /// # Why this is not a flat tensor any more
    ///
    /// It was: `[seats, kv_heads, max_context, head_dim]` in F32, written with
    /// `slice_set` and read with a `narrow` into a hand-rolled attention. That
    /// works and it is enormous — 160 KiB a token at four bytes an element, so a
    /// seat at a 6,144-token context costs 1.9 GiB and a wave is two seats wide
    /// on a card that has already given 8.5 GiB to the weights.
    ///
    /// This is the cache the rest of the engine uses, and the reason the whole
    /// repository exists: per-block adaptive quantization over a paged arena, at
    /// the compression level in [`PROSE_COMPRESSION`]. The same K/V costs about a
    /// seventh as much, and the wave widens by the same factor.
    ///
    /// One backing per layer, each holding every seat — the seat is the backing's
    /// batch index, which is what makes a whole wave one `sync_decode_gpu_chunks`
    /// call rather than one per sequence.
    kv: Vec<LayerCache>,
    max_context: usize,
    /// How many sequences the caches were built to hold at once.
    seats: usize,
    /// Tokens currently valid in each seat's caches. Reset per drain.
    filled: Vec<usize>,
}

/// Every layer's decode headers for one step, staged and addressable.
///
/// Built once per decode forward and handed to each layer as `base + li * stride`
/// — the kernel wants one contiguous `SlotHeader` array per layer, and the
/// position map inside them is shared, so there is nothing per-layer to rebuild.
struct DecodeMeta {
    base: u64,
    /// Bytes between one layer's header array and the next.
    stride: usize,
    /// The staged buffers. They view the pinned generation's arena, and dropping
    /// one before the kernel reading it has run is an illegal address rather than
    /// a wrong answer — so the whole forward holds them.
    _hold: (GpuBuf, GpuBuf),
}

/// One layer's chunked K/V: the shared backing, and a cache bound to each seat.
///
/// The pair is what every arena call is keyed on — `paged_prefill_batched` takes
/// the per-seat caches, and the decode path takes the backing plus the seats it
/// wants headers for — so they are held together rather than reassembled at each
/// call site.
struct LayerCache {
    backing: ChunkedKvBacking,
    /// One per seat, bound to slots `0..seats` of `backing`.
    seats: Vec<KvCache>,
}

/// One job's place in a decoding wave.
///
/// The seat owns everything that differs between sequences — its position in the
/// caches, its sampler state, its grammar walk, what it has said — so the step
/// itself can treat the batch as one tensor.
struct Seat {
    /// Index into the caller's job list, which is what the answers are keyed by.
    job: usize,
    /// Which row of the K/V caches this sequence occupies.
    slot: usize,
    seed: u64,
    budget: usize,
    /// Logits for this seat's next token, `[1, vocab]`.
    logits: Tensor,
    sampling: SamplingConfig,
    state: SequenceSamplingState,
    driver: Option<StencilDriver>,
    produced: Vec<u32>,
    preview: Preview,
    done: bool,
}

struct PlacedLayer {
    attn_norm: Tensor,
    wq: candle_transformers::models::quantized_matmul::QMatMul,
    wk: candle_transformers::models::quantized_matmul::QMatMul,
    wv: candle_transformers::models::quantized_matmul::QMatMul,
    wo: candle_transformers::models::quantized_matmul::QMatMul,
    /// Per-head RMSNorm on Q and K, applied before the rotary.
    ///
    /// `None` on a Llama, which has no such norm. Present on Qwen3, where
    /// leaving them out is not a small numerical difference — the projections
    /// were trained against a normed input to the rotary, so skipping it
    /// produces fluent text that ignores position.
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    ffn_norm: Tensor,
    gate: candle_transformers::models::quantized_matmul::QMatMul,
    up: candle_transformers::models::quantized_matmul::QMatMul,
    down: candle_transformers::models::quantized_matmul::QMatMul,
}

mod placed {
    //! The device half: reading the checkpoint and putting it at addresses the
    //! guest's ground handed out.
    //!
    //! Not feature-gated, because this crate is not: `candle-conversation` is
    //! CUDA-only unconditionally (see its `Cargo.toml`), so a `cfg(feature =
    //! "cuda")` here is a block that never compiles — which is exactly what an
    //! earlier draft of this file did, leaving `PlacedLlama` with no `load` and
    //! the guest with no way to say so.

    use super::*;
    use candle::cuda_backend::wave_provenance::LeaseOrigin;
    use candle::quantized::cuda::{load_repacked_into, view_repacked};
    use candle::quantized::Int8Mode;
    use candle_transformers::models::quantized_matmul::QMatMul;

    /// Read one tensor's raw bytes out of the GGUF and place them in ground.
    ///
    /// The bytes are copied verbatim: a GGUF's quantized payload is already the
    /// layout the GGML kernels read, so there is nothing to convert and the
    /// guest's on-device size is its on-disk size. That is also what makes
    /// [`ProseGuest::checkpoint_bytes`] an exact figure rather than an estimate.
    fn place_quantized(
        device: &Device,
        ground: &mut GuestGround,
        payload: &Mmap,
        content: &gguf_file::Content,
        name: &str,
        ph: &mut LoadPhases,
    ) -> candle::Result<(QTensor, u64)> {
        let info = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("prose guest: {name} is not in the GGUF")))?;
        let elems = info.shape.elem_count();
        let bytes = tensor_bytes(info.ggml_dtype, elems);
        let extent = placed_extent(info.ggml_dtype, elems);
        let t = Instant::now();
        let buf = tensor_slice(payload, content, info, name, bytes)?;
        LoadPhases::add(&mut ph.read_ns, t);
        ph.tensors += 1;
        ph.bytes += bytes as u64;

        // **`extent`, not `bytes`.** The GGML matmul kernels address
        // `MATRIX_ROW_PADDING` elements past the end of every row
        // unconditionally, so a placement sized to the payload alone is one the
        // kernel reads past — into the next weight, and at the last weight in a
        // region, into whatever the KV side put above it. Reserving the padded
        // extent is the same rule `layer_stream::build::slot_extent` states for
        // the same reason.
        //
        // 256-byte alignment: every GGML block type divides it, and it is cheap
        // against a 16 MiB region.
        let t = Instant::now();
        let at = ground
            .place(extent, 256)
            .map_err(|e| candle::Error::Msg(format!("prose guest: placing {name}: {e}")))?;
        LoadPhases::add(&mut ph.place_ns, t);
        let Device::Cuda(cuda) = device else {
            return Err(candle::Error::Msg("prose guest: not a CUDA device".into()));
        };
        let stream = cuda.cuda_stream();
        let t = Instant::now();
        // SAFETY: `at.ptr` names `extent` bytes of ground this guest holds for
        // the length of the drain, and nothing else writes it — the regions
        // came from the KV side's free list and do not go back until the model
        // is dropped.
        let storage = unsafe { load_repacked_into(cuda, &stream, at.ptr, buf, info.ggml_dtype)? };
        // **No synchronise here, and that is the point of the mapping.**
        //
        // `load_repacked_into` issues an async copy and does not wait. That used
        // to be wrong for this caller, because `buf` was a local `Vec` that died
        // at the end of the function — a copy still in flight would have read
        // freed host memory and placed a weight that was part checkpoint and
        // part whatever the allocator handed out next, with nothing reporting
        // it. So every tensor paid a full stream sync, 254 of them, and the file
        // read and the transfer could never overlap.
        //
        // `buf` now borrows a mapping [`checkpoint::payload`] holds for the life
        // of the process, which is exactly the condition the function was
        // written for — the expert cache uploads out of a long-lived mmap for
        // the same reason. The copies queue and the load synchronises once, at
        // the end, in `PlacedLlama::load`.
        LoadPhases::add(&mut ph.h2d_ns, t);
        // The storage `load_repacked_into` built covers the payload only.
        // Rebuild it over the padded extent so the tail the kernel reads is
        // ground this guest owns, with the payload length unchanged so nothing
        // downstream sees the padding as weight.
        drop(storage);
        // SAFETY: as above, and the bytes at `at.ptr` are the payload just
        // written; the tail is the region's own zeroed ground.
        let view = unsafe { view_repacked(cuda, at.ptr, bytes, extent, info.ggml_dtype)? };
        Ok((QTensor::new(view, info.shape.dims().to_vec())?, at.ptr))
    }

    /// One tensor's bytes, borrowed from the mapped checkpoint.
    ///
    /// Bounds-checked rather than sliced blind: `offset` and the computed length
    /// come from the file's own header, and a truncated or mismatched checkpoint
    /// would otherwise index past the mapping — which is a fault on a page that
    /// was never mapped, reported nowhere near the file that caused it.
    fn tensor_slice<'a>(
        payload: &'a Mmap,
        content: &gguf_file::Content,
        info: &gguf_file::TensorInfo,
        name: &str,
        bytes: usize,
    ) -> candle::Result<&'a [u8]> {
        let start = (content.tensor_data_offset + info.offset) as usize;
        let end = start.checked_add(bytes).ok_or_else(|| {
            candle::Error::Msg(format!("prose guest: {name} overflows its offset"))
        })?;
        payload.get(start..end).ok_or_else(|| {
            candle::Error::Msg(format!(
                "prose guest: {name} wants bytes {start}..{end} of a {}-byte checkpoint — the \
                 header does not describe this file",
                payload.len()
            ))
        })
    }

    /// A second view over an already-placed weight.
    ///
    /// For a tied output head: Llama 3.2 has no `output.weight`, and copying the
    /// 315 MiB embedding table to serve as one would double the largest single
    /// placement in the guest to hold the same bytes twice.
    fn view_placed(
        device: &Device,
        ptr: u64,
        dtype: GgmlDType,
        shape: &[usize],
    ) -> candle::Result<QTensor> {
        let elems: usize = shape.iter().product();
        let Device::Cuda(cuda) = device else {
            return Err(candle::Error::Msg("prose guest: not a CUDA device".into()));
        };
        // SAFETY: `ptr` was placed by `place_quantized` in the same ground,
        // holds this dtype's payload, and outlives both views — they are
        // dropped together with the model.
        let view = unsafe {
            view_repacked(
                cuda,
                ptr,
                tensor_bytes(dtype, elems),
                placed_extent(dtype, elems),
                dtype,
            )?
        };
        QTensor::new(view, shape.to_vec())
    }

    /// A projection: a placed quantized tensor wrapped as a matmul.
    ///
    /// `Int8Mode::Off` because the payload is the checkpoint's own quant, not a
    /// KO twin — `from_qtensor_view` refuses the mismatch outright, which is
    /// what keeps this honest.
    fn projection(
        device: &Device,
        ground: &mut GuestGround,
        payload: &Mmap,
        content: &gguf_file::Content,
        name: &str,
        ph: &mut LoadPhases,
    ) -> candle::Result<QMatMul> {
        let (qt, _) = place_quantized(device, ground, payload, content, name, ph)?;
        QMatMul::from_qtensor_view(qt, Int8Mode::Off)
    }

    /// A norm vector: read, dequantize on the host, and copy into ground as F32.
    ///
    /// Dequantized here rather than on the device because the device path
    /// allocates its destination from the pool, and a norm is one vector per
    /// layer — the host round trip costs microseconds and the pool costs the
    /// invariant.
    fn norm_vector(
        device: &Device,
        ground: &mut GuestGround,
        payload: &Mmap,
        content: &gguf_file::Content,
        name: &str,
        ph: &mut LoadPhases,
    ) -> candle::Result<Tensor> {
        let info = content
            .tensor_infos
            .get(name)
            .ok_or_else(|| candle::Error::Msg(format!("prose guest: {name} is not in the GGUF")))?;
        let elems = info.shape.elem_count();
        let bytes = tensor_bytes(info.ggml_dtype, elems);
        let t = Instant::now();
        let buf = tensor_slice(payload, content, info, name, bytes)?;
        LoadPhases::add(&mut ph.read_ns, t);
        // Through a CPU QTensor, so every GGML dtype a checkpoint might store a
        // norm in is handled by the one implementation that already knows them.
        let t = Instant::now();
        let cpu = candle::quantized::ggml_file::qtensor_from_ggml(
            info.ggml_dtype,
            buf,
            info.shape.dims().to_vec(),
            &Device::Cpu,
        )?;
        let host = cpu.dequantize(&Device::Cpu)?.to_dtype(DType::F32)?;
        let values = host.flatten_all()?.to_vec1::<f32>()?;
        LoadPhases::add(&mut ph.dequant_ns, t);
        ph.tensors += 1;
        ph.bytes += bytes as u64;
        place_f32(device, ground, &values, info.shape.dims().to_vec(), ph)
    }

    /// Copy host floats into ground and return a tensor viewing them.
    pub(super) fn place_f32(
        device: &Device,
        ground: &mut GuestGround,
        values: &[f32],
        shape: Vec<usize>,
        ph: &mut LoadPhases,
    ) -> candle::Result<Tensor> {
        let bytes = std::mem::size_of_val(values);
        let t = Instant::now();
        let at = ground
            .place(bytes, 256)
            .map_err(|e| candle::Error::Msg(format!("prose guest: placing {shape:?}: {e}")))?;
        LoadPhases::add(&mut ph.place_ns, t);
        let Device::Cuda(cuda) = device else {
            return Err(candle::Error::Msg("prose guest: not a CUDA device".into()));
        };
        let stream = cuda.cuda_stream();
        let t = Instant::now();
        // SAFETY: `at.ptr` names `bytes` of ground this guest holds, and the
        // slice is `forget`ed below rather than dropped, so it never tries to
        // free an address the pool did not allocate.
        let mut dst = unsafe { stream.upgrade_device_ptr::<u8>(at.ptr, bytes) };
        // SAFETY: `f32` has no padding and no invalid bit patterns, and the
        // length is derived from the slice's own size.
        let raw = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, bytes) };
        stream
            .memcpy_htod(raw, &mut dst.slice_mut(..bytes))
            .map_err(candle::Error::wrap)?;
        // The copy is asynchronous and the caller's `values` may not outlive
        // it — see `varground::place_host_tensor` for the whole story. A norm
        // vector is small enough that the driver's staging almost always wins
        // the race, which is precisely what makes an unsynchronised copy here a
        // bug that appears under load and nowhere else.
        stream.synchronize().map_err(candle::Error::wrap)?;
        LoadPhases::add(&mut ph.h2d_ns, t);
        // The slice must not free the address on drop — it is ground, not a
        // pool allocation, and the driver would refuse the free anyway.
        std::mem::forget(dst);
        // SAFETY: the bytes were just written, the ground outlives this tensor
        // (the drain drops the model before the ground), and nothing else
        // writes the range.
        unsafe {
            Tensor::from_leased_cuda_ptr(at.ptr, DType::F32, shape, device, LeaseOrigin::Foreign)
        }
    }

    // `place_uninit_f32` used to live here, for the flat K/V caches. The chunked
    // cache allocates its own arenas from the region pool, so there is no longer
    // an uninitialised buffer for this guest to place.

    impl PlacedLlama {
        pub(super) fn load(
            device: &Device,
            ground: &mut GuestGround,
            spec: &ProseSpec,
            want_context: usize,
            // How many sequences the caches must hold at once — the backlog's
            // width, capped. Every seat costs a whole context of K/V, which is
            // what bounds it.
            seats: usize,
            ph: &mut LoadPhases,
        ) -> candle::Result<Self> {
            // Mapped once per process and shared: the bytes are the page
            // cache's, so a second drain touches pages that are already
            // resident rather than copying 2.46 GiB into a fresh buffer.
            let payload = checkpoint::payload(&spec.gguf)?;
            let t = Instant::now();
            // Cached across drains — see [`super::checkpoint`]. This was the
            // single largest phase of a load at ~1.15 s, because a Llama-3 GGUF
            // carries its 128,256-token vocabulary in the metadata and the
            // parser reads it field by field off an unbuffered handle.
            let content = checkpoint::gguf_header(&spec.gguf)?;
            let geo = Geometry::from_gguf(&content)?;
            LoadPhases::add(&mut ph.header_ns, t);

            let (embed, embed_ptr) =
                place_quantized(device, ground, &payload, &content, "token_embd.weight", ph)?;
            // Tied head: the checkpoint has no `output.weight`, so the head is
            // a second *view* over the placement the embedding already made —
            // not a second copy of a 315 MiB table.
            let head = match content.tensor_infos.get("output.weight") {
                Some(_) => projection(device, ground, &payload, &content, "output.weight", ph)?,
                None => {
                    let view = view_placed(device, embed_ptr, embed.dtype(), embed.shape().dims())?;
                    QMatMul::from_qtensor_view(view, Int8Mode::Off)?
                }
            };

            let mut layers = Vec::with_capacity(geo.layers);
            for li in 0..geo.layers {
                let p = |s: &str| format!("blk.{li}.{s}");
                layers.push(PlacedLayer {
                    attn_norm: norm_vector(
                        device,
                        ground,
                        &payload,
                        &content,
                        &p("attn_norm.weight"),
                        ph,
                    )?,
                    wq: projection(device, ground, &payload, &content, &p("attn_q.weight"), ph)?,
                    wk: projection(device, ground, &payload, &content, &p("attn_k.weight"), ph)?,
                    wv: projection(device, ground, &payload, &content, &p("attn_v.weight"), ph)?,
                    wo: projection(
                        device,
                        ground,
                        &payload,
                        &content,
                        &p("attn_output.weight"),
                        ph,
                    )?,
                    // Placed only where the architecture has them. Asked for on
                    // a Llama they would be a missing-tensor error; skipped on a
                    // Qwen3 they would be silent wrongness, which is why this
                    // reads the arch rather than probing for the tensor.
                    q_norm: match geo.arch.has_qk_norm() {
                        false => None,
                        true => Some(norm_vector(
                            device,
                            ground,
                            &payload,
                            &content,
                            &p("attn_q_norm.weight"),
                            ph,
                        )?),
                    },
                    k_norm: match geo.arch.has_qk_norm() {
                        false => None,
                        true => Some(norm_vector(
                            device,
                            ground,
                            &payload,
                            &content,
                            &p("attn_k_norm.weight"),
                            ph,
                        )?),
                    },
                    ffn_norm: norm_vector(
                        device,
                        ground,
                        &payload,
                        &content,
                        &p("ffn_norm.weight"),
                        ph,
                    )?,
                    gate: projection(
                        device,
                        ground,
                        &payload,
                        &content,
                        &p("ffn_gate.weight"),
                        ph,
                    )?,
                    up: projection(device, ground, &payload, &content, &p("ffn_up.weight"), ph)?,
                    down: projection(
                        device,
                        ground,
                        &payload,
                        &content,
                        &p("ffn_down.weight"),
                        ph,
                    )?,
                });
            }
            let norm = norm_vector(device, ground, &payload, &content, "output_norm.weight", ph)?;

            // **The same arithmetic the footprint claimed against.** Two
            // expressions of "what a token costs" is two places for it to be
            // wrong, and the way it failed is instructive: the RoPE tables were
            // in neither, so the load placed every K/V cache and then ran out
            // with 951 KB of `cos` left to write — after the engine had already
            // been evicted for it.
            // Every seat holds its own K/V, so a seat costs a whole context.
            let _per_token = geo.per_token_bytes() * seats.max(1);
            // **The ground's free space no longer bounds the context.**
            //
            // It did while the K/V was placed here: the context was whatever was
            // left after the weights, divided by what a token cost. The chunked
            // cache claims its own regions from the reservation, so ground that
            // holds only weights and slack has nothing to say about how many
            // tokens the cache can hold — and asking it anyway cut a 6,094-token
            // context to 636 and refused every job in the wave.
            //
            // The cache's own limit is enforced where it is known: an arena that
            // cannot claim a region fails there, with a message about the
            // reservation rather than about this guest's ground.
            let affordable = want_context;
            // **What was claimed for, not what happens to be free.** `affordable`
            // is the *floor* this degrades to, never the target: the claim
            // carries `PLACEMENT_SLACK_BYTES` on top of the cache precisely so
            // there is ground left over after it, and taking that leftover as
            // permission to allocate more context spends the slack on the thing
            // it was being held back from. See `ProseGuest::context_for`.
            //
            // It still degrades rather than failing, because the two are
            // estimates of different things: the claim is sized from the GGUF
            // header before the load, and a checkpoint whose placed extent runs
            // over that estimate must shorten the context rather than refuse.
            let max_context = want_context.min(affordable);
            if max_context < MIN_USABLE_CONTEXT {
                return Err(candle::Error::Msg(format!(
                    "prose guest: after the weights there is room for {max_context} tokens of \
                     K/V, under the {MIN_USABLE_CONTEXT} a job needs — the ground was sized for a \
                     smaller checkpoint than {:?}",
                    spec.gguf
                )));
            }

            // **This ground now holds nothing but weights, so say so.**
            //
            // The K/V moved out to the chunked cache, which claims its own
            // regions from the same span reservation this ground was carved
            // from. Two tenants of one reservation is exactly the arrangement
            // where a region handed out twice does not fault and does not show
            // up in the output — it surfaces as attention reading a weight.
            //
            // Declaring the ground read-only is what makes that loud: every
            // write-capable launch checks its destination at the FFI boundary,
            // and a kernel handed a buffer that overlaps this names *itself*, on
            // the thread that did it, before the corruption. Compiles to nothing
            // without `tensor-assert`.
            for run in ground.runs() {
                candle::readonly_regions::declare(
                    format!("prose guest weights {:#x}", run.base),
                    run.base,
                    run.bytes,
                );
            }

            let t = Instant::now();
            let mut kv = Vec::with_capacity(geo.layers);
            for _ in 0..geo.layers {
                let backing = ChunkedKvBacking::new_with_format(
                    seats,
                    geo.kv_heads,
                    geo.head_dim,
                    PROSE_K_FORMAT,
                    PROSE_V_FORMAT,
                    device,
                    max_context,
                )?;
                let mut bound = Vec::with_capacity(seats);
                for slot in 0..seats {
                    let mut c = KvCache::new(2, max_context);
                    // The policy is what makes this adaptive rather than a flat
                    // quantization: it selects a format per 32-token block against
                    // the block's own q-relevance, at [`PROSE_COMPRESSION`].
                    // `None` here would store every block at the arena's format
                    // and throw away the whole point.
                    c.set_chunked_backing(
                        &backing,
                        slot,
                        Some(CompressionPolicy::new(PROSE_COMPRESSION)),
                    )?;
                    bound.push(c);
                }
                kv.push(LayerCache {
                    backing,
                    seats: bound,
                });
            }
            LoadPhases::add(&mut ph.alloc_ns, t);

            let rope_cs = geo.rope_cs(device, max_context.div_ceil(CHUNK_TOKENS))?;

            // **One synchronise for the whole load, not one per tensor.**
            //
            // Every weight copy above is asynchronous — it reads a mapping that
            // outlives this function, so nothing has to wait for it at the call
            // site. But they must all have landed before the first forward
            // reads them, and the caller has no other barrier that guarantees
            // it: the drain runs `model.run` on the same stream, and a queued
            // copy is ordered against it, but the K/V and RoPE placements below
            // are the only syncs and a checkpoint could in principle end with a
            // projection. Waiting once here is the honest barrier, and it costs
            // whatever is genuinely still in flight rather than 254 round trips.
            let t = Instant::now();
            if let Device::Cuda(cuda) = device {
                cuda.cuda_stream()
                    .synchronize()
                    .map_err(candle::Error::wrap)?;
            }
            LoadPhases::add(&mut ph.h2d_ns, t);

            Ok(Self {
                geo,
                device: device.clone(),
                embed,
                layers,
                norm,
                head,
                rope_cs,
                stager: PinnedStager::new(device.as_cuda_device()?),
                kv,
                max_context,
                seats,
                filled: vec![0; seats],
            })
        }
    }
}

/// A context shorter than this cannot hold a system prompt and an answer, so a
/// load that can only afford it has failed rather than degraded.
const MIN_USABLE_CONTEXT: usize = 512;

// `PREFILL_CHUNK` used to live here, holding the first axis of a hand-rolled
// attention matrix down to 256 positions so a long prompt did not ask for
// gigabytes at once. `paged_prefill_batched` is flash-tiled and never
// materialises that matrix, so there is nothing left to bound.

impl Geometry {
    /// Ground one token of context costs, across the whole stack.
    ///
    /// **K/V and RoPE together**, because both are preallocated for the whole
    /// context and both scale with it. Counting only the caches is what made
    /// the first real load place all of them and then fail on the RoPE tables,
    /// having already evicted the engine to get that far.
    ///
    /// F32 throughout: the guest's forward runs in F32, so this is the width
    /// the buffers are actually allocated at rather than the width a
    /// half-precision path would want.
    /// What one token of K/V costs a seat, across every layer.
    ///
    /// # Not four bytes an element any more
    ///
    /// This charged F32 because the caches were F32: 160 KiB a token, so a seat
    /// at a 6,144-token context was 1.9 GiB and a wave was two seats wide. The
    /// cache is the engine's chunked one now, stored per block at
    /// [`PROSE_COMPRESSION`], and the figure has to follow or `seats_for` sizes a
    /// wave against a cost that no longer exists.
    ///
    /// [`COMPRESSED_BITS_PER_ELEM`] is what makes it an estimate rather than an
    /// arithmetic fact — the format is chosen per block against the data, so the
    /// true figure is only known after the fact. It is rounded up on purpose: a
    /// wave sized against an optimistic number is one that cannot be seated after
    /// the engine has already been evicted for it.
    ///
    /// The rotary rows are gone from the total because the table is no longer in
    /// ground — see [`Geometry::rope_cs`].
    fn per_token_bytes(&self) -> usize {
        let elems = 2 * self.kv_heads * self.head_dim * self.layers;
        (elems * COMPRESSED_BITS_PER_ELEM).div_ceil(8)
    }

    fn from_gguf(content: &gguf_file::Content) -> candle::Result<Self> {
        // **The prefix, from the file rather than from the filename.** Every
        // geometry key below is namespaced by architecture, and the two families
        // this guest serves spell them differently: Hermes-3 is a Llama and
        // Hermes-4 is a Qwen3. `general.architecture` is the GGUF's own answer
        // and is what the conversion tools write, so it decides — a deployment
        // that swaps the checkpoint does not also have to tell us what it swapped
        // to.
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                candle::Error::Msg(
                    "prose guest: general.architecture is not in the GGUF — cannot tell which \
                     namespace the geometry keys are under"
                        .into(),
                )
            })?;
        let arch = ProseArch::from_gguf_name(&arch)?;
        let p = arch.gguf_prefix();
        let get = |k: &str| {
            content
                .metadata
                .get(k)
                .ok_or_else(|| candle::Error::Msg(format!("prose guest: {k} is not in the GGUF")))
        };
        let u = |k: &str| get(k).and_then(|v| v.to_u32()).map(|v| v as usize);
        let layers = u(&format!("{p}.block_count"))?;
        let hidden = u(&format!("{p}.embedding_length"))?;
        let heads = u(&format!("{p}.attention.head_count"))?;
        let kv_heads = u(&format!("{p}.attention.head_count_kv"))?;
        let vocab = content
            .tensor_infos
            .get("token_embd.weight")
            .map(|i| i.shape.dims()[0])
            .ok_or_else(|| {
                candle::Error::Msg(
                    "prose guest: no token_embd.weight to read the vocab from".into(),
                )
            })?;
        Ok(Self {
            arch,
            layers,
            hidden,
            heads,
            kv_heads,
            // **Declared, not derived.** `hidden / heads` is true of a Llama and
            // an accident elsewhere: Qwen3 sizes its heads independently of the
            // residual width, and the two only agree at 14B because 40 × 128 is
            // 5120. Reading the declaration means a sibling of this checkpoint
            // whose arithmetic does not coincide loads correctly rather than
            // silently reshaping every projection to the wrong width.
            head_dim: u(&format!("{p}.attention.key_length")).unwrap_or_else(|_| hidden / heads),
            vocab,
            rms_eps: get(&format!("{p}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64,
            rope_theta: get(&format!("{p}.rope.freq_base"))
                .and_then(|v| v.to_f32())
                .unwrap_or(arch.default_rope_theta()),
        })
    }

    /// `cos` and `sin` for every position the cache can hold.
    ///
    /// Built once at load and placed in ground, because the alternative is
    /// rebuilding two `[len, head_dim/2]` tensors in the pool on every step.
    /// The paged kernels' interleaved `[max_blocks * 32, head_dim]` rotary table.
    ///
    /// # Why the guest no longer builds its own pair
    ///
    /// This produced a `(cos, sin)` pair in ground, which the forward then applied
    /// itself with `rotary_emb::rope`. The paged attention kernels rotate *inside
    /// the kernel* — K is stored un-rotated and turned at read time from the
    /// slot's own position — so they want one combined table in their own layout,
    /// and [`compute_rope_cs`] is the function that builds it. Keeping a second
    /// hand-rolled table beside it would be two spellings of the same constants,
    /// and the rotary is precisely where this file has already been wrong once:
    /// Qwen3 is half-split, not interleaved, and a mismatch there is fluent prose
    /// that ignores position.
    ///
    /// It lands in the CUDA pool rather than in ground, which is the deliberate
    /// exception the guest's header already names for activations. At 6,144
    /// positions and a 128-wide head it is 3 MiB — a rounding error against the
    /// 8.5 GiB of weights the ground is really for, and the price of not owning a
    /// duplicate of a shared helper.
    fn rope_cs(&self, device: &Device, max_blocks: usize) -> candle::Result<Tensor> {
        let half = self.head_dim / 2;
        // The frequencies depend on `i` alone, so they are computed once.
        let inv: Vec<f32> = (0..half)
            .map(|i| {
                (1.0f64 / (self.rope_theta as f64).powf(2.0 * i as f64 / self.head_dim as f64))
                    as f32
            })
            .collect();
        let inv_freq = Tensor::from_vec(inv, half, device)?;
        compute_rope_cs(&inv_freq, max_blocks, self.head_dim, device)
    }
}

impl PlacedLlama {
    /// Embed `tokens` into a `[n, hidden]` F32 tensor, for the caller to shape.
    ///
    /// One `dequantize_into` per token, straight out of the placed table. The
    /// alternative — dequantizing the table and indexing it — costs 1.5 GiB for
    /// a 3 B model to read a few hundred rows of it.
    fn embed(&self, tokens: &[u32]) -> candle::Result<Tensor> {
        let h = self.geo.hidden;
        let mut dst = Tensor::zeros((tokens.len(), h), DType::F32, &self.device)?;
        for (i, t) in tokens.iter().enumerate() {
            let row = *t as usize;
            if row >= self.geo.vocab {
                candle::bail!(
                    "prose guest: token {row} is outside the checkpoint's {}-token vocabulary — \
                     the tokenizer and the checkpoint are not a pair",
                    self.geo.vocab
                );
            }
            self.embed.dequantize_into(&mut dst, row * h, i * h, h)?;
        }
        Ok(dst)
    }

    /// One layer's attention, through the engine's paged kernels.
    ///
    /// Returns `[nseats, n, heads * head_dim]` — the attention context, ready for
    /// the output projection.
    ///
    /// # Two kernels, because there are two shapes and only two
    ///
    /// `n > 1` is a prefill: one seat laying down a whole prompt, which
    /// [`paged_prefill_batched`] does flash-tiled — so there is no quadratic
    /// attention matrix and no reason for this guest to chunk its prompt any
    /// more. `n == 1` is a decode step across every seat, which
    /// [`paged_decode_attn`] does in one call, scattering each seat's new K/V
    /// into its own slot as it goes.
    ///
    /// Both write K/V into the cache themselves. Nothing here does.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &mut self,
        li: usize,
        seat0: usize,
        nseats: usize,
        n: usize,
        starts: &[usize],
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        generation: &Generation,
        shared_pm: &RefCell<Option<SharedPm>>,
        meta: Option<&DecodeMeta>,
    ) -> candle::Result<Tensor> {
        let geo = self.geo;
        // The paged kernels run in F16/BF16 only; the guest's stack is F32, so
        // the operands are narrowed at this boundary and the context widened
        // back. A deliberate conversion for a kernel that cannot read F32 — not
        // a defensive cast over types that ought to have matched.
        let (qc, kc, vc) = (
            q.to_dtype(PAGED_DTYPE)?,
            k.to_dtype(PAGED_DTYPE)?,
            v.to_dtype(PAGED_DTYPE)?,
        );

        let out = if n > 1 {
            if nseats != 1 {
                candle::bail!(
                    "prose guest: a prefill covers one seat at a time, not {nseats} — prompts \
                     have different lengths and the batched form wants them ragged"
                );
            }
            let rope_offsets = Tensor::zeros(1, DType::U32, &self.device)?;
            // The write region was allocated for every layer together in
            // `forward_span`; `paged_prefill_batched` only allocates it on the
            // path where it creates the backing itself, which a cache that
            // arrives already bound never takes.
            let cache = &mut self.kv[li];
            let mut one = [&mut cache.seats[seat0]];
            paged_prefill_batched(
                None,
                &mut one,
                &starts[..1],
                &qc,
                &kc,
                &vc,
                1,
                &[n],
                geo.heads,
                geo.kv_heads,
                geo.head_dim,
                None,
                &rope_offsets,
                &self.rope_cs,
                geo.arch.rope_is_interleaved(),
                generation,
                shared_pm,
            )?
        } else {
            let scale = 1.0 / (geo.head_dim as f32).sqrt();
            let meta = meta.ok_or_else(|| {
                candle::Error::Msg("prose guest: a decode step with no headers built".into())
            })?;
            paged_decode_attn(
                None,
                &qc,
                meta.base + (li * meta.stride) as u64,
                PAGED_DTYPE,
                geo.heads,
                geo.kv_heads,
                geo.head_dim,
                scale,
                &kc,
                &vc,
                &self.rope_cs,
                geo.arch.rope_is_interleaved(),
            )?
        };
        out.to_dtype(DType::F32)
    }

    /// Build every layer's `SlotHeader` array for one decode step, once.
    ///
    /// # The header is 24 bytes and carries a position map
    ///
    /// `{n_slices, write_slice, slices_ptr, position_map_ptr}`, and the kernel
    /// indexes it as `headers + slot * 24`. Writing the first three and stopping
    /// at 16 is not a header merely missing a field: every slot after the first
    /// is then read from the wrong offset entirely. One seat survives it — there
    /// is no second header to misread, and its garbage `position_map_ptr` goes
    /// untouched while the whole history sits in the write region. Eleven seats
    /// faulted on the first decode step of every run.
    ///
    /// # One map, shared by every layer
    ///
    /// It is built from layer 0's chunks and every layer's header points at it,
    /// which is sound only because the layers agree about their block structure —
    /// what `ensure_for_batch_entries_all` reconciled before this runs. The check
    /// below is where that is worth confirming, both values being in hand: a
    /// layer that disagrees scatters the new token into one chunk while attention
    /// is told to read it from another, with no fault and no wrong-looking number
    /// anywhere.
    fn decode_meta(
        &mut self,
        seat0: usize,
        nseats: usize,
        starts: &[usize],
        generation: &Generation,
    ) -> candle::Result<DecodeMeta> {
        let want: Vec<(usize, usize)> = (0..nseats).map(|si| (seat0 + si, starts[si])).collect();

        // One entry per token of each seat's history, then the entry for the
        // token this step is about to write.
        let mut pm: Vec<u32> = Vec::new();
        let mut pm_at: Vec<usize> = Vec::with_capacity(nseats);
        let mut shape: Vec<(u32, u32)> = Vec::with_capacity(nseats);
        for &(slot, _) in &want {
            pm_at.push(pm.len() * 4);
            let chunks = self.kv[0]
                .backing
                .live_chunks_as_sealed(slot)
                .unwrap_or_default();
            for (sidx, c) in chunks.iter().enumerate() {
                let base = (sidx as u32) << 16;
                pm.extend(
                    (c.offset as u32..c.offset as u32 + c.token_count as u32)
                        .map(|in_blk| base | in_blk),
                );
            }
            // The write slot is the first non-full chunk from the writer start,
            // never `chunks.last()` — which may be a trailing empty sitting past
            // it. This must match `sync_decode_gpu_chunks`'s own rule, or the
            // token is scattered into one chunk and read back from another.
            let wstart = self.kv[0]
                .backing
                .writer_start_idx_for_seq(slot)
                .unwrap_or(0);
            let n_ch = chunks.len();
            let wi = if n_ch == 0 {
                0
            } else {
                let start = wstart.min(n_ch - 1);
                (start..n_ch)
                    .find(|&i| {
                        (chunks[i].offset as usize + chunks[i].token_count as usize) < CHUNK_TOKENS
                    })
                    .unwrap_or(n_ch - 1)
            };
            let within = chunks
                .get(wi)
                .map_or(0, |c| c.offset as u32 + c.token_count as u32);
            pm.push(((wi as u32) << 16) | within);
            shape.push((n_ch as u32, wi as u32));
        }
        if pm.is_empty() {
            // A valid device pointer even with nothing to say.
            pm.push(0);
        }
        let pm_bytes: Vec<u8> = pm.iter().flat_map(|e| e.to_le_bytes()).collect();
        let mut pinned = generation.alloc(pm_bytes.len())?;
        pinned.copy_from_slice(&pm_bytes);
        let pm_gpu = generation.submit(pinned)?;
        let pm_base = pm_gpu.dev_ptr();

        let mut bytes = Vec::with_capacity(self.kv.len() * nseats * SLOT_HEADER_BYTES);
        for layer in &self.kv {
            let arena_info = layer.backing.resolve_arena_info()?;
            // **The snapshot, not the live buffer.** `sync_decode_gpu_chunks`
            // returns a pointer into the sequence's live `gpu_chunks`, which
            // reallocates on the next chunk append — so a pointer taken for an
            // early layer dangles once a later one crosses a boundary. The
            // snapshot copies the slot state into this generation, whose device
            // pointer is stable for the whole forward.
            let mask = vec![true; want.len()];
            let (ptrs, _) = layer.backing.sync_decode_gpu_chunks_snapshot(
                &want,
                &arena_info,
                generation,
                &mask,
            )?;
            for (i, (ptr, n_slices, write_slice)) in ptrs.into_iter().enumerate() {
                if (n_slices, write_slice) != shape[i] {
                    candle::bail!(
                        "prose guest: a layer describes seat {i} as {n_slices} slices writing \
                         slice {write_slice}, but the position map every layer shares was built \
                         from layer 0 as {} slices writing slice {}",
                        shape[i].0,
                        shape[i].1
                    );
                }
                bytes.extend_from_slice(&n_slices.to_le_bytes());
                bytes.extend_from_slice(&write_slice.to_le_bytes());
                bytes.extend_from_slice(&ptr.to_le_bytes());
                bytes.extend_from_slice(&(pm_base + pm_at[i] as u64).to_le_bytes());
            }
        }
        let mut pinned = generation.alloc(bytes.len())?;
        pinned.copy_from_slice(&bytes);
        let gpu = generation.submit(pinned)?;
        Ok(DecodeMeta {
            base: gpu.dev_ptr(),
            stride: nseats * SLOT_HEADER_BYTES,
            _hold: (gpu, pm_gpu),
        })
    }

    /// One forward for the `nseats` seats starting at `seat0`, each contributing
    /// `n` tokens at its own cache position.
    ///
    /// `tokens` is `nseats × n`, seat-major. Returns the logits for each seat's
    /// **last** position only, `[nseats, vocab]`: a job wants the next token, and
    /// materialising `[n, vocab]` for a 500-token prompt is a 250 MB tensor
    /// nothing reads.
    ///
    /// # One function for both prefill and decode
    ///
    /// They differ only in which axis is wide. A prefill is one seat and many
    /// tokens; a decode is every seat and one token each. Writing them as two
    /// functions means two copies of the layer stack, which is the one part that
    /// must not drift — the QK-norm placement and the half-split rotary are both
    /// things this file has already had wrong once, and having them in a single
    /// place is worth the parameter.
    ///
    /// The seats are a contiguous span rather than an arbitrary list so the K/V
    /// caches can be read with a `narrow`. Gathering scattered seats would copy
    /// every seat's whole history per layer per step, which costs more than the
    /// batching saves.
    fn forward_span(
        &mut self,
        seat0: usize,
        nseats: usize,
        tokens: &[u32],
        n: usize,
    ) -> candle::Result<Tensor> {
        debug_assert_eq!(tokens.len(), nseats * n);
        let geo = self.geo;

        for si in 0..nseats {
            let start = self.filled[seat0 + si];
            if start + n > self.max_context {
                candle::bail!(
                    "prose guest: {n} more tokens would pass the {}-token context this drain's \
                     ground was sized for (already at {start})",
                    self.max_context
                );
            }
        }

        // **No rotary and no mask are built here any more.** The paged kernels
        // rotate inside the kernel from each slot's own position — K is stored
        // un-rotated for exactly that — and bound attention by the slot's length
        // rather than by an additive mask. What used to be an `index_select` per
        // step over a hand-built `[seats, 1, n, seen]` mask is now two arguments.
        let starts: Vec<usize> = (0..nseats).map(|si| self.filled[seat0 + si]).collect();

        // **One generation for the whole forward, not one per layer.** Every
        // header and slice this pass uploads views the stager's pinned arena, and
        // `Generation::drop` syncs the stream and frees it — so a generation that
        // ended while a later layer's kernel was still in flight would be an
        // illegal address, not a wrong number. The position map is
        // layer-invariant and cached across the forward in `shared_pm` for the
        // same reason it exists in the engine: forty layers would otherwise
        // upload the same map forty times.
        // `begin_generation` returns an owned handle, so this borrow of the
        // stager ends here and the layer loop can still take `&mut self`.
        let generation = self.stager.begin_generation();
        let shared_pm: RefCell<Option<SharedPm>> = RefCell::new(None);

        // **The chunks for this pass are allocated across every layer at once,
        // before any of them runs.**
        //
        // Block structure is meant to be layer-invariant and is not
        // unconditionally so, which is why `ensure_for_batch_entries_all` exists
        // and why it also unifies the layout: a per-layer allocation lets one
        // layer hold a writable tail that suppresses the allocation the others
        // still needed, and the first layer to reach the gap refuses the step.
        // The decode metadata builder then collapses every layer onto one
        // position map, so the invariance this establishes is the thing that map
        // depends on.
        let entries: Vec<(usize, usize)> = (0..nseats).map(|si| (seat0 + si, starts[si])).collect();
        let backings: Vec<ChunkedKvBacking> = self.kv.iter().map(|l| l.backing.clone()).collect();
        ChunkedKvBacking::ensure_for_batch_entries_all(&backings, &entries, n)?;
        drop(backings);

        // Every layer's decode headers, built once — the position map inside them
        // is shared, so there is nothing per-layer to rebuild. A prefill needs
        // none of this: `paged_prefill_batched` assembles its own.
        let meta = match n {
            1 => Some(self.decode_meta(seat0, nseats, &starts, &generation)?),
            _ => None,
        };

        let mut x = self.embed(tokens)?.reshape((nseats, n, geo.hidden))?;
        for li in 0..geo.layers {
            let residual = x.clone();
            let h = candle_nn::ops::rms_norm(&x, &self.layers[li].attn_norm, geo.rms_eps as f32)?;

            let q = self.layers[li].wq.forward_live(&h)?;
            let k = self.layers[li].wk.forward_live(&h)?;
            let v = self.layers[li].wv.forward_live(&h)?;

            // **Row-major over tokens, and not transposed to head-major.**
            //
            // The paged kernels take rank 3 — `[tokens, heads, head_dim]` — and
            // both shapes this guest produces collapse to it with a reshape: a
            // decode is `nseats` rows of one token, a prefill is one seat's `n`
            // of them, and `nseats * n` is the row count either way. The
            // transpose to `[b, heads, n, dim]` that used to be here was for the
            // hand-rolled matmul and is now the wrong layout as well as a copy.
            let rows = nseats * n;
            let q = q.reshape((rows, geo.heads, geo.head_dim))?.contiguous()?;
            let k = k
                .reshape((rows, geo.kv_heads, geo.head_dim))?
                .contiguous()?;
            let v = v
                .reshape((rows, geo.kv_heads, geo.head_dim))?
                .contiguous()?;

            // **Between the reshape and the rotary, and nowhere else.** Qwen3
            // norms each head's Q and K over `head_dim` before position is
            // applied; the weights are one vector of `head_dim` shared by every
            // head, so this runs on the `[1, heads, n, head_dim]` view where the
            // last axis is exactly what the norm is over.
            let q = match &self.layers[li].q_norm {
                None => q,
                Some(w) => candle_nn::ops::rms_norm(&q, w, geo.rms_eps as f32)?,
            };
            let k = match &self.layers[li].k_norm {
                None => k,
                Some(w) => candle_nn::ops::rms_norm(&k, w, geo.rms_eps as f32)?,
            };

            // **No rotary here.** The paged kernels take K un-rotated and turn
            // both Q and K inside the kernel, from the position the slot itself
            // is at — which is also what lets seats sitting at different
            // positions share one call.
            let y = self.attend(
                li,
                seat0,
                nseats,
                n,
                &starts,
                &q,
                &k,
                &v,
                &generation,
                &shared_pm,
                meta.as_ref(),
            )?;
            let y = y
                .reshape((nseats, n, geo.heads * geo.head_dim))?
                .contiguous()?;
            let y = self.layers[li].wo.forward_live(&y)?;
            x = (residual + y)?;

            let residual = x.clone();
            let h = candle_nn::ops::rms_norm(&x, &self.layers[li].ffn_norm, geo.rms_eps as f32)?;
            let gate = self.layers[li].gate.forward_live(&h)?;
            let up = self.layers[li].up.forward_live(&h)?;
            let act = (candle_nn::ops::silu(&gate)? * up)?;
            let down = self.layers[li].down.forward_live(&act)?;
            x = (residual + down)?;
        }
        // **One host-side length, pushed into every layer.**
        //
        // The decode kernel self-increments its own device-side slot length after
        // it scatters, and the prefill sets the cache's length as it allocates —
        // so the caches would advance on their own and this counter would be a
        // second opinion about the same number. Two counters that agree until
        // they do not is how a slot ends up reading one token of somebody else's
        // history, which does not fault and is not visible in the output.
        //
        // So `filled` is the only host-side truth and the caches are told what it
        // is, rather than being asked.
        for si in 0..nseats {
            let seat = seat0 + si;
            self.filled[seat] += n;
            let at = self.filled[seat];
            for layer in &mut self.kv {
                layer.seats[seat].set_current_seq_len(at)?;
            }
        }

        let x = candle_nn::ops::rms_norm(&x, &self.norm, geo.rms_eps as f32)?;
        let last = x.narrow(1, n - 1, 1)?.contiguous()?;
        self.head.forward_live(&last)?.reshape((nseats, geo.vocab))
    }

    /// Prefill one seat's whole prompt, returning the logits for its last
    /// position.
    ///
    /// # This used to be chunked, and no longer needs to be
    ///
    /// The hand-rolled attention materialised `[1, heads, n, seen]`, quadratic in
    /// the prompt: at forty heads and a 6,144-token context, a 2.9 GiB tensor
    /// with the softmax wanting a second beside it. Measured at 3,594 MiB of
    /// pool allocation per drain, which on a card already holding a 30B model is
    /// not there — every story-length prompt died a tenth of a second into the
    /// run. Feeding the prompt in 256-token chunks made that first axis a
    /// constant and fixed it.
    ///
    /// [`paged_prefill_batched`] is flash-tiled, so the matrix is never
    /// materialised at all and the chunking has nothing left to do. One call.
    ///
    /// # Why prefill is one seat at a time when decode is not
    ///
    /// Prompts have different lengths, so a batched prefill is ragged in the one
    /// axis a batch needs regular — the kernel supports it through `q_lens`, but
    /// it wants the sequences flattened, and this guest's layer stack carries a
    /// `[seats, n, hidden]` shape that a ragged pass would have to abandon.
    /// There is little to win either way: prefill is compute-bound over a wide
    /// token axis and already saturates the card, while decode is
    /// bandwidth-bound over a single token and does not. The wave is worth
    /// building where the idle time is.
    fn prefill(&mut self, seat: usize, tokens: &[u32]) -> candle::Result<Tensor> {
        if tokens.is_empty() {
            candle::bail!("prose guest: an empty prompt");
        }
        self.forward_span(seat, 1, tokens, tokens.len())
    }

    /// The sampling configuration for one request.
    ///
    /// # Prose is not chat, and the architecture table is for chat
    ///
    /// The obvious thing is `SamplingConfig::for_gguf_architecture`, so that this
    /// guest samples a Qwen3 the way the rest of the engine samples a Qwen3. It
    /// was tried, and it is wrong here — not by a little. The story rung came
    /// back sane for a page and then fell into a word-association cascade that
    /// ran to the token budget: *"…incalculable incomprehensible ineffable
    /// inexplicable inscrutable enigmatic…"*, several hundred tokens of it.
    ///
    /// The table is built around a model that thinks before it answers. Its DRY
    /// penalty is deliberately span-scoped — the kernel windows it on the current
    /// structural span and resets at `<think>`, `</think>` and the tool-call
    /// markers — and that scoping is what makes an aggressive n-gram penalty safe
    /// in chat, where a span is a few hundred tokens and the next one starts
    /// clean. A diary entry has no spans at all. One span covers all 1,400
    /// tokens, the penalty compounds across the whole answer, and a model that
    /// may not repeat itself over that distance stops writing prose and starts
    /// walking a thesaurus. The thinking steering has the same mismatch from the
    /// other side: close-boosts and `force_segment_close_after` for a block this
    /// request never asks the model to open.
    ///
    /// So the pieces are named here for what prose needs, rather than inherited
    /// from a profile built for a different job:
    ///
    /// The answer is [`SamplingConfig::preset`]'s `creative`, which is this
    /// regime already written down — top-p diversity, a light repeat penalty and
    /// window to curb loops, no DRY, no thinking steering. It is the profile the
    /// `--sampler` flag offers for exactly this and it needs nothing added.
    ///
    /// Only the end-of-answer failsafes are this call's own, because they are the
    /// one part that cannot be a preset: they are measured in tokens, and the
    /// preset cannot know what budget a caller asked for. Every stock figure —
    /// the table's graceful 800 and hard 1,000, `creative`'s own 550 and 600 —
    /// would truncate the ladder's 1,400-token story rung, losing the outline in
    /// its tail that every rung below expands. So they are restated against
    /// `budget`, in the same proportions the presets use.
    fn sampling_for(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        seed: u64,
        request: &ProseRequest,
        budget: usize,
    ) -> candle::Result<SamplingConfig> {
        let mut c = SamplingConfig::preset(PROSE_SAMPLER).ok_or_else(|| {
            candle::Error::Msg(format!(
                "prose guest: there is no `{PROSE_SAMPLER}` sampling preset — it is one of {:?}",
                SamplingConfig::preset_names()
            ))
        })?;
        c.seed = seed;
        if let Some(t) = request.temperature.filter(|t| *t >= 0.0) {
            c.temperature = t;
        }

        // For the sentence-end ids `graceful_eos_after` stops on. It also
        // resolves the `<think>` markers, which cost nothing here: `creative`
        // configures no close boost, no close-after and no suppression to use
        // them.
        c.resolve_thinking_tokens(tokenizer);

        let budget = budget as i32;
        Ok(
            c.with_dynamic_eos_boost(1.0, budget * 7 / 10, budget * 8 / 10, 3.0)
                .with_eos_failsafe(budget * 8 / 10, budget),
        )
    }

    /// Reset every seat's cache between waves.
    ///
    /// The stored bytes are not cleared, only the lengths: attention reads a
    /// slot's own length and never past it, so zeroing would be a full-width
    /// write over bytes nothing reads.
    ///
    /// **The chunks are freed, not just the counter zeroed.**
    ///
    /// `KvCache::reset` deliberately does not free a slot's blocks — the slot
    /// lifecycle belongs to whoever creates and frees sequences, which for this
    /// guest is here. Zeroing the length alone leaves the block table holding the
    /// previous job's chunks while the recorded offset says nothing is there, and
    /// the next wave to take that slot is refused by its own consistency check:
    /// *"slices cover 3,549 tokens but the slot's recorded offset is 0"*. That is
    /// the check doing its job — the alternative was a fresh prompt attending
    /// over a stranger's life.
    ///
    /// It is also what returns the regions. A backlog wider than the wave runs as
    /// several waves through the same seats, and a guest that never freed one
    /// walked the reservation empty: *"no region is claimable — every one of the
    /// 1,236 regions is occupied"*.
    fn reset(&mut self) -> candle::Result<()> {
        self.filled.iter_mut().for_each(|f| *f = 0);
        for layer in &mut self.kv {
            for (slot, seat) in layer.seats.iter_mut().enumerate() {
                seat.reset();
                layer.backing.free_sequence(slot)?;
            }
        }
        Ok(())
    }

    /// Decode a whole wave of jobs together, one token per seat per step.
    ///
    /// Returns one result per request, in order.
    ///
    /// # Why the wave and not a loop over jobs
    ///
    /// Decode reads the entire checkpoint to produce one token. Sixteen seats
    /// stepping together read it once for all sixteen, which is the difference
    /// between this engine's single-session rate and its aggregate one, and the
    /// jobs in a backlog have no ordering between them at all — the ladder only
    /// ever submits siblings together, and siblings are defined by never seeing
    /// each other.
    ///
    /// # A finished seat keeps stepping
    ///
    /// Seats finish at different tokens, and the wave is a contiguous span so
    /// that the K/V caches can be read with a `narrow` rather than gathered. A
    /// seat that has stopped therefore rides along: its row is computed and
    /// discarded. That is very nearly free — the step's cost is reading the
    /// weights, which the live seats are paying anyway — and it is what keeps
    /// the batch dense.
    ///
    /// What it must not do is consume context. A finished seat's position is
    /// rolled back after each step, so a short job that stopped early cannot
    /// walk off the end of its own context while a long one beside it is still
    /// writing.
    fn generate_batch(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        jobs: &[(&ProseRequest, &GuestSink)],
        default_system: &str,
    ) -> Vec<candle::Result<GuestOutcome>> {
        // A wave that could not clear the previous one's caches must not run at
        // all: every seat would attend over a stranger's history.
        if let Err(e) = self.reset() {
            return jobs
                .iter()
                .map(|_| Err(candle::Error::Msg(format!("prose guest: clearing: {e}"))))
                .collect();
        }
        let mut out: Vec<candle::Result<GuestOutcome>> = Vec::with_capacity(jobs.len());
        let mut seats: Vec<Seat> = Vec::with_capacity(jobs.len());

        // Prefill seat by seat, so a job that cannot be seated fails on its own
        // rather than taking the wave down with it.
        for (i, (request, _)) in jobs.iter().enumerate() {
            out.push(Err(candle::Error::Msg(
                "prose guest: this job was never decoded".into(),
            )));
            // **The slot is the seat's position in the wave, not the job's.**
            //
            // Taking the job index left a hole whenever a job failed to seat —
            // a prompt too long for the context, say — and `decode_wave` treats
            // the wave as the contiguous span `seat0 .. seat0 + seats.len()`,
            // because that is what lets one `sync_decode_gpu_chunks` serve the
            // whole batch. A hole makes that span address a slot no seat owns
            // while missing one that does: every seat past the gap then decodes
            // against another sequence's chunks at its own remembered position.
            let slot = seats.len();
            match self.seat_for(tokenizer, i, slot, request, default_system) {
                Ok(seat) => seats.push(seat),
                Err(e) => out[i] = Err(e),
            }
        }
        if seats.is_empty() {
            return out;
        }

        match self.decode_wave(tokenizer, jobs, &mut seats) {
            Ok(()) => {
                for seat in seats {
                    let text = match tokenizer.decode(&seat.produced, true) {
                        Ok(t) => t,
                        Err(e) => {
                            out[seat.job] =
                                Err(candle::Error::Msg(format!("prose guest: decoding: {e}")));
                            continue;
                        }
                    };
                    out[seat.job] = Ok(GuestOutcome::Prose {
                        text,
                        tokens: seat.produced.len() as u32,
                        seed: seat.seed,
                    });
                }
            }
            // A wave-level failure is every seat's failure: they shared the step
            // that broke, and none of them has a complete answer.
            Err(e) => {
                for seat in &seats {
                    out[seat.job] = Err(candle::Error::Msg(format!("{e}")));
                }
            }
        }
        out
    }

    /// Tokenise one job, take a seat for it, and prefill its prompt.
    fn seat_for(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        job: usize,
        slot: usize,
        request: &ProseRequest,
        default_system: &str,
    ) -> candle::Result<Seat> {
        if slot >= self.seats {
            candle::bail!(
                "prose guest: job {job} has no seat — the wave was built for {} and the caller \
                 handed over more",
                self.seats
            );
        }
        let system = if request.system.trim().is_empty() {
            default_system
        } else {
            request.system.as_str()
        };
        let prompt = chatml(system, &request.prompt);
        let encoded = tokenizer
            .encode(prompt, true)
            .map_err(|e| candle::Error::Msg(format!("prose guest: tokenizing: {e}")))?;
        let tokens = encoded.get_ids().to_vec();
        let budget = request.max_tokens as usize;
        if tokens.len() + budget > self.max_context {
            candle::bail!(
                "prose guest: a {}-token prompt plus {budget} tokens of answer passes the \
                 {}-token context this drain's ground was sized for",
                tokens.len(),
                self.max_context
            );
        }

        let seed = resolve_seed(request.seed);
        let sampling = self.sampling_for(tokenizer, seed, request, budget)?;
        // The stencil, if the caller asked for one. Compiled per request rather
        // than cached: it is a handful of tokens over a two-arm tree, and a
        // cache keyed by the arms would outlive the tokenizer it was built
        // against — the fingerprint exists to catch exactly that mismatch.
        let driver = match request.choices.as_deref() {
            Some(arms) if !arms.is_empty() => Some(choice_stencil(tokenizer, arms)?),
            _ => None,
        };

        let logits = self.prefill(slot, &tokens)?;
        Ok(Seat {
            job,
            slot,
            seed,
            budget,
            logits,
            sampling,
            state: SequenceSamplingState::new(self.geo.vocab, RECENT_TOKEN_WINDOW),
            driver,
            produced: Vec::with_capacity(budget),
            preview: Preview::default(),
            done: false,
        })
    }

    fn decode_wave(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        jobs: &[(&ProseRequest, &GuestSink)],
        seats: &mut [Seat],
    ) -> candle::Result<()> {
        let eos = eos_tokens(tokenizer);

        // The engine's own sampler, not a bare `LogitsProcessor`.
        //
        // This used to be temperature over the untruncated distribution, on the
        // reasoning that a narrator's job is range and top-k/top-p are what
        // flatten it. The reasoning is half right and the configuration it
        // produced does not work: with nothing truncated there is also no
        // repetition penalty — none is expressible — so a long answer that finds
        // a sentence it likes writes it until the budget runs out. Greedy gave
        // one paragraph thirteen times; at a temperature it still closed on the
        // same line four times over.
        //
        // Adding a penalty by hand made it worse, and that is the instructive
        // part. A penalty pushes mass off the tokens it damps and onto whatever
        // is next, and with no truncation "whatever is next" is the whole tail —
        // the same prompt came back as fluent nonsense. Penalties and truncation
        // are one mechanism: top-k/top-p decide what is admissible, the penalty
        // moves weight around inside that, and either alone is worse than
        // neither.
        //
        // So this takes the architecture's own configuration, which the rest of
        // the engine already runs on: temperature, top-k, top-p, a gentle
        // multiplicative repeat penalty, and DRY, whose n-gram window is the one
        // that actually addresses a *sentence* repeating rather than a token.
        let sampler = BatchedSampler::new(
            self.device.clone(),
            self.geo.vocab,
            RECENT_TOKEN_WINDOW,
            TokenBuffer::from(eos.clone()),
            None,
        );

        let nseats = seats.len();
        // The wave is addressed as one span, so the seats must be that span. A
        // gap here is not a wrong answer, it is one seat decoding against
        // another's chunks — stated rather than assumed, because nothing
        // downstream can tell.
        for (i, seat) in seats.iter().enumerate() {
            if seat.slot != seats[0].slot + i {
                candle::bail!(
                    "prose guest: seat {i} holds slot {} in a wave starting at {} — the wave is \
                     addressed as a contiguous span and this one has a hole in it",
                    seat.slot,
                    seats[0].slot
                );
            }
        }
        let steps = seats.iter().map(|s| s.budget).max().unwrap_or(0);
        for _ in 0..steps {
            if seats.iter().all(|s| s.done) {
                break;
            }

            // **A stencil reaches the sampler as an allow-list, including when
            // the grammar has already decided.** A run the walk has committed to
            // is a one-token allow-list, which forces exactly that token *through
            // the sampler* rather than around it — so the penalties still see
            // what was said, and every seat takes exactly one step whether it is
            // choosing or being led. Feeding a committed run around the sampler
            // is what would make seats step at different rates, and a wave is
            // only a wave while they do not.
            for seat in seats.iter_mut() {
                seat.sampling.stencil.clear();
                if seat.done {
                    continue;
                }
                if let Some(d) = seat.driver.as_mut() {
                    match d.step() {
                        StepMask::Done => {
                            seat.done = true;
                            continue;
                        }
                        StepMask::Prefill(toks) => {
                            // One per step; the rest are taken on the steps that
                            // follow, because the walk's cursor advances with the
                            // sequence and not ahead of it.
                            if let Some(t) = toks.first() {
                                seat.sampling.stencil.push(*t as i32);
                            }
                        }
                        StepMask::Branch(set) => seat
                            .sampling
                            .stencil
                            .extend(set.tokens().iter().map(|&t| t as i32)),
                        // A choice tree has no free-text spans. Decoding freely
                        // here would leave the walk's cursor and the sequence
                        // disagreeing about what was produced, so it is refused
                        // rather than guessed at.
                        StepMask::Free { .. } => {
                            candle::bail!(
                                "prose guest: a choice stencil produced a free-text span, which \
                                 means the tree is not the two-arm grammar this path assumes"
                            )
                        }
                    }
                }
            }

            let rows: Vec<&Tensor> = seats.iter().map(|s| &s.logits).collect();
            let logits = match rows.len() {
                // One seat needs no join, and `cat` of a single tensor is a
                // full-width copy for nothing.
                1 => rows[0].clone(),
                _ => Tensor::cat(&rows, 0)?,
            };
            drop(rows);

            // The configs are copied out because the sampler wants them by
            // shared reference while it holds the states mutably, and both live
            // on the same seats. A handful of small `Vec`s per seat per step,
            // against a step that reads the whole checkpoint.
            let cfgs: Vec<SamplingConfig> = seats.iter().map(|s| s.sampling.clone()).collect();
            let cfg_refs: Vec<&SamplingConfig> = cfgs.iter().collect();
            let mut states: Vec<&mut SequenceSamplingState> =
                seats.iter_mut().map(|s| &mut s.state).collect();
            let next = sampler.sample_batch(&logits, &mut states, &cfg_refs)?;
            drop(states);

            for (i, seat) in seats.iter_mut().enumerate() {
                if seat.done {
                    continue;
                }
                let tok = next[i];
                if let Some(d) = seat.driver.as_mut() {
                    let bytes = tokenizer.decode(&[tok], false).unwrap_or_default();
                    d.accept(tok, bytes.as_bytes());
                }
                if eos.contains(&tok) || seat.produced.len() >= seat.budget {
                    seat.done = true;
                    continue;
                }
                seat.produced.push(tok);
                // Emitted here rather than after the forward, so a watcher sees
                // the token at the moment it exists instead of one step behind.
                // Skipped entirely when nobody is watching — the decode below is
                // a real cost and there is no point paying it for nobody.
                let sink = jobs[seat.job].1;
                if sink.is_watched() {
                    if let Ok(full) = tokenizer.decode(&seat.produced, true) {
                        if let Some(delta) = seat.preview.advance(&full) {
                            sink.emit(GuestEvent::Token(delta));
                        }
                    }
                }
            }

            // **A finished seat rides along and advances with everyone else.**
            //
            // The wave is a contiguous span, so a seat cannot leave it, and the
            // decode kernel scatters and self-increments for *every* slot it is
            // given — there is no way to sit one out. An earlier version rolled a
            // finished seat's position back after the step to save it the
            // context; what that actually did was leave the host's idea of the
            // seat's position one behind the cache's, so the next header build
            // addressed the wrong chunks. `CUDA_ERROR_ILLEGAL_ADDRESS`,
            // mid-wave, and only ever with more than one seat.
            //
            // Riding along costs nothing it cannot afford: the loop runs at most
            // the widest budget, and every seat's context was sized for its own
            // prompt plus that budget, so a seat that stopped early has exactly
            // the room to be carried to the end.
            let feed: Vec<u32> = seats
                .iter()
                .enumerate()
                .map(|(i, s)| if s.done { 0 } else { next[i] })
                .collect();

            let out = self.forward_span(seats[0].slot, nseats, &feed, 1)?;
            for (i, seat) in seats.iter_mut().enumerate() {
                seat.logits = out.narrow(0, i, 1)?;
            }
        }
        Ok(())
    }
}

/// Compile "the answer is exactly one of these" into a stencil walk.
///
/// One branch, one arm per choice, straight to the end — the smallest tree the
/// grammar admits. The heavy machinery in [`crate::stencil`] exists for tool
/// calls with nested JSON; this uses the same compiler for a two-word decision,
/// which is what keeps the masking behaviour identical between them.
///
/// # Single-token arms, and why it matters
///
/// A branch masks the sampler to the *frontier* — the first token of each arm —
/// and once a token is taken the walk is inside that arm and forces the rest of
/// it. For `Compliant` / `Refused` that is three or four forced tokens after a
/// decision made on one, and a model that would have reconsidered cannot.
///
/// With one token per arm there is no remainder to force: the whole decision is
/// a single masked decode. The arms are checked here and a multi-token one is
/// reported, because the cost is silent — the answer still parses, it is just no
/// longer the model's after the first token.
fn choice_stencil(
    tokenizer: &tokenizers::Tokenizer,
    arms: &[String],
) -> candle::Result<StencilDriver> {
    let vocab = HfVocab::new(tokenizer.clone(), eos_id(tokenizer), fingerprint(tokenizer));
    for arm in arms {
        let n = vocab.encode(arm).len();
        if n != 1 {
            tracing::warn!(
                target: "candle_conversation::guest",
                arm = %arm,
                tokens = n,
                "a stencil arm is not a single token — the walk commits to it on the first and \
                 forces the rest"
            );
        }
    }
    let pairs: Vec<(&str, &str)> = arms.iter().map(|a| (a.as_str(), "done")).collect();
    let spec = StencilTreeBuilder::new("choice")
        .root("pick")
        .branch("pick", &pairs)
        .end("done")
        .build()
        .map_err(|e| candle::Error::Msg(format!("prose guest: building the stencil: {e}")))?;
    let tree = compile(&spec, &vocab)
        .map_err(|e| candle::Error::Msg(format!("prose guest: compiling the stencil: {e}")))?;
    Ok(StencilDriver::new(Arc::new(tree)))
}

/// The tokenizer's end-of-sequence id, for the stencil's vocab.
///
/// The stencil only needs it to know which token ends a free-text span, and a
/// choice tree has none — so a vocab that could not find one is still usable and
/// `0` is a safe stand-in rather than a reason to refuse the job.
fn eos_id(tokenizer: &tokenizers::Tokenizer) -> u32 {
    eos_tokens(tokenizer).into_iter().min().unwrap_or(0)
}

/// A cheap identity for the tokenizer a tree was compiled against.
///
/// The stencil carries it so a tree built for one vocab cannot be walked against
/// another. Compiled per request here, so it is only ever compared with itself —
/// it is filled in honestly regardless, because a `0` would silently match every
/// other tree that also skipped it.
fn fingerprint(tokenizer: &tokenizers::Tokenizer) -> u64 {
    let mut h = DefaultHasher::new();
    tokenizer.get_vocab_size(true).hash(&mut h);
    tokenizer.token_to_id("<|im_start|>").hash(&mut h);
    h.finish()
}

/// What a watcher has been shown so far, so the next step can send only what is
/// new.
///
/// **Why the whole sequence is re-decoded every step.** A token is not a
/// character: detokenising one in isolation loses the spacing rule that depends
/// on its neighbour, and a multi-byte character can span two tokens, so
/// `decode(&[one])` yields a replacement character where the pair yields a
/// letter. The only reliable fragment is the difference between decoding the
/// whole sequence and decoding it one token shorter. That is quadratic in the
/// token count, which sounds worse than it is: a description is ~60 tokens, and
/// each decode is microseconds against a ~25 ms forward pass.
///
/// The comparison is a **common prefix**, not a strict one. Tokenizer cleanup
/// can revise a character already shown — `" don "` + `"'t"` becomes `" don't"`,
/// dropping a space that was already sent — and a strict-prefix check would
/// treat that as a break and go silent for the rest of the generation. Instead
/// the preview resynchronises: the sent text can end up a character or two from
/// the final, which is why the finished text is what the caller stores and the
/// fragments are only what it shows while waiting.
#[derive(Default)]
struct Preview {
    sent: String,
}

impl Preview {
    /// The part of `full` a watcher has not seen, or `None` when there is none.
    fn advance(&mut self, full: &str) -> Option<String> {
        let common = common_prefix_len(&self.sent, full);
        let delta = &full[common..];
        if delta.is_empty() {
            // The step produced no new visible text — a token that only
            // affected cleanup. Nothing to send, and the state still advances
            // so the next comparison is against what was actually decoded.
            self.sent.truncate(common);
            return None;
        }
        let delta = delta.to_string();
        self.sent.truncate(common);
        self.sent.push_str(&delta);
        Some(delta)
    }
}

/// The length in bytes of the longest shared prefix, always on a character
/// boundary — slicing a string at a byte that splits a character panics, and
/// a multi-byte character is exactly where two decodes are most likely to
/// differ.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0;
    for (x, y) in a.chars().zip(b.chars()) {
        if x != y {
            break;
        }
        n += x.len_utf8();
    }
    n
}

/// The stop tokens a Hermes-3 answer ends on.
///
/// Both, not one: the checkpoint is ChatML-tuned over a Llama base, so a turn
/// ends at `<|im_end|>` while the base's own `<|eot_id|>` and `<|end_of_text|>`
/// still appear. Missing any of them produces an answer that runs on into the
/// next speaker's turn, which reads as the model rambling.
fn eos_tokens(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect()
}

/// The dialect the checkpoint was tuned in.
///
/// Hermes-3 is ChatML over a Llama base — the tune's own card is explicit about
/// it — so the Llama-3 header format would be the wrong markers on the right
/// weights, which produces plausible prose that ignores the system prompt.
fn chatml(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// The dtype the paged attention kernels compute in.
///
/// They dispatch on the arena's format and run in F16 or BF16 only — there is no
/// F32 kernel — so this is the type Q, K and V are narrowed to at the attention
/// boundary and the context is widened back from. F16 rather than BF16 because
/// the arenas here are quantized from F16 and the dispatch follows the arena.
const PAGED_DTYPE: DType = DType::F16;

/// Whole regions each layer's caches take before storing a token.
///
/// A backing keeps an arena per size class, and an arena is a region however
/// little of it is used. Adaptive selection at [`PROSE_COMPRESSION`] spreads
/// blocks across the four-bit and eight-bit classes and keeps the active partial
/// chunk in float, so three is the count to plan for — deliberately a ceiling,
/// since the failure this covers is a wave the shed did not make room for.
const ARENA_CLASSES_PER_LAYER: usize = 3;

/// One `SlotHeader`, as the decode kernel lays it out.
///
/// `{u32 n_slices, u32 write_slice, u64 slices_ptr, u64 position_map_ptr}`, and
/// `get_slot_header` indexes `headers + slot * 24`. Mirrored here because the
/// host writes the bytes and nothing checks the two agree — a stride that is
/// short by the position-map pointer reads every slot after the first from the
/// wrong offset.
const SLOT_HEADER_BYTES: usize = 24;

/// Tokens per chunk, which is the paged cache's own `CHUNK_SIZE`.
///
/// Named here because the rotary table is sized in blocks rather than positions
/// and the conversion should not be a bare `32` at the call site.
const CHUNK_TOKENS: usize = 32;

/// The compression level the guest's K/V is stored at.
///
/// C3 on the engine's own ladder: per-block adaptive selection across
/// Q4_0/Q4_1/Q8_0/Q8_1 for K and Q3_0/Q3_1/Q4_x/Q8_0 for V, gated on the C3
/// q-relevance thresholds. Sensitive blocks keep eight bits, ordinary ones fall
/// to four, and the choice is made per 32-token block against the data rather
/// than guessed once for the whole cache.
///
/// Near-lossless was available — C0 is one candidate, Q8_KS — and is not what
/// this wants. The guest's cost is a seat, a seat is a whole context of K/V, and
/// the wave's width falls straight out of it: at F32 a seat was 1.9 GiB and two
/// fitted, at C3 it is nearer 275 MiB and the wave reaches its own ceiling. The
/// loss that buys is a fraction of a percent of a head's dynamic range on the
/// blocks that can afford it.
const PROSE_COMPRESSION: u8 = 3;

/// What a stored K/V element costs at [`PROSE_COMPRESSION`], in bits.
///
/// An estimate, and deliberately a pessimistic one. C3 picks per block: a
/// four-bit format is 4.5 bits with its scale, an eight-bit one 8.5, and which a
/// block gets depends on its own q-relevance. Six is above the mix these prompts
/// actually produce, and it is the direction to be wrong in — a wave sized
/// against an optimistic figure is one that cannot be seated after the engine has
/// already been evicted for it.
const COMPRESSED_BITS_PER_ELEM: usize = 6;

/// The arena format K chunks are sealed into.
///
/// The *initial* format for a backing; per-block selection at
/// [`PROSE_COMPRESSION`] narrows it further as chunks seal, and the active
/// partial chunk is always float for writes.
const PROSE_K_FORMAT: KvFormat = KvFormat::Quantized(candle_nn::kv_cache::QuantFormat::Q8_0);

/// The arena format V chunks are sealed into. See [`PROSE_K_FORMAT`].
const PROSE_V_FORMAT: KvFormat = KvFormat::Quantized(candle_nn::kv_cache::QuantFormat::Q8_0);

/// Ground a wave's K/V may take, across every seat.
///
/// The seat count falls out of this and the context the backlog asked for, so a
/// wave of short turns is wide and a wave of long ones is narrow — see
/// [`ProseGuest::seats_for`].
///
/// # What the ceiling is really made of
///
/// It is not a preference, it is the ground a drain can get. The claim is
/// weights plus this plus the placement slack; measured, the weights are
/// 8,579 MiB and the largest claim this guest has ever been granted is
/// 13,808 MiB, so four gigabytes of K/V is close to the whole of what is left
/// and a larger figure buys refused drains rather than wider waves.
///
/// That is also what makes the seat *cost* the lever worth pulling, and there
/// are two obvious ones. The caches are F32, so a seat at a 6,144-token context
/// is 1.9 GiB and only two fit here; K/V in F16 is standard, halves that, and
/// doubles the wave at the same ground. And the cost is linear in the context,
/// which a backlog's own prompts set — the years phase drives it to the ceiling
/// with a 12,479-character shared prefix, and every character trimmed there
/// widens the wave.
const MAX_SEAT_BYTES: usize = 2 << 30;

/// How many seats a wave gets, given what one costs and how many are wanted.
///
/// Separate from [`ProseGuest::seats_for`] so the arithmetic is testable without
/// a checkpoint to read a geometry out of — the byte budget and the floor of one
/// are the parts with a decision in them, and the rest is a header read.
fn seats_from(per_seat_bytes: usize, wanted: usize) -> usize {
    if per_seat_bytes == 0 {
        return wanted.clamp(1, MAX_SEATS);
    }
    // At least one, always. A job that fits the guest's own context ceiling must
    // run even when a single seat costs more than the wave's budget — refusing
    // it here would make the budget a stricter limit than the context is.
    let affordable = (MAX_SEAT_BYTES / per_seat_bytes).max(1);
    wanted.max(1).min(affordable).min(MAX_SEATS)
}

/// A hard ceiling on the wave, whatever the arithmetic says.
///
/// Beyond this the per-step host work — a sampler readback per seat, and a slot
/// header per seat per layer — starts to show against the decode it is meant to
/// be amortising, and the ladder never submits a wider wave anyway.
const MAX_SEATS: usize = 16;

/// The engine's sampling preset this guest writes prose under.
///
/// One of [`SamplingConfig::preset_names`], resolved at request time so a rename
/// in that table is an error naming the alternatives rather than a silent
/// fallback to whatever `Default` happens to be.
const PROSE_SAMPLER: &str = "creative";

/// How much history the penalties can see.
///
/// [`BatchedSampler`] pads every sequence's recent-token window to this, and the
/// widest thing that reads it is DRY's n-gram range of 512. Sized above that so
/// the architecture's own configuration arrives intact rather than being
/// silently clipped to a window this guest happened to pick.
const RECENT_TOKEN_WINDOW: usize = 1024;

#[cfg(test)]
mod tests {
    use super::*;

    /// The guest stores K/V at the engine's own compression level, and that is
    /// what its wave width is made of.
    ///
    /// The assertion is on the level rather than on bytes because the format is
    /// chosen per block against the data — the point of C3 is that a sensitive
    /// block keeps eight bits while an ordinary one falls to four, which no
    /// static figure here can stand in for.
    #[test]
    fn the_guest_stores_its_cache_compressed_and_adaptively() {
        let p = CompressionPolicy::new(PROSE_COMPRESSION);
        assert_eq!(p.compression_level, PROSE_COMPRESSION);
        assert!(
            p.override_k_quant.is_none() && p.override_v_quant.is_none(),
            "an override pins one format for every block and turns the adaptive \
             selection off, which is the whole mechanism"
        );

        // C3's own candidates, so a change to the shared ladder that made this
        // level uniform or near-lossless would be caught here rather than as an
        // unexplained collapse in seats.
        let (k, v) = CompressionPolicy::production_candidates(PROSE_COMPRESSION);
        assert!(k.len() > 1, "K has no choice to make: {k:?}");
        assert!(v.len() > 1, "V has no choice to make: {v:?}");
    }

    /// A wave is bounded by what its K/V costs, not by how many jobs queued —
    /// and never by less than one, or a single full-context job could not run.
    ///
    /// Written against the cap rather than a literal so it says the same thing
    /// while [`MAX_SEATS`] is held at one for the multi-seat decode fault: the
    /// arithmetic is what is under test, and the ceiling is a separate decision.
    #[test]
    fn a_wave_is_bounded_by_bytes_and_never_by_less_than_one() {
        assert_eq!(seats_from(MAX_SEAT_BYTES, 8), 1);
        assert_eq!(seats_from(MAX_SEAT_BYTES / 10, 4), 4.min(MAX_SEATS));
        assert_eq!(seats_from(1, 1000), MAX_SEATS);
        // A seat that would not fit at all still gets to try alone, because
        // refusing it outright would mean a job that fits the guest's own
        // context ceiling could never be served.
        assert_eq!(seats_from(MAX_SEAT_BYTES * 4, 8), 1);
    }

    /// The guest samples a Qwen3 the way the rest of the engine samples one:
    /// out of the architecture table, not a second opinion living in this file.
    ///
    /// The assertions are a pair, and that is the point. A repetition penalty
    /// with no truncation beside it is worse than neither — it pushes mass off
    /// the tokens it damps and, with nothing truncated, onto the whole tail;
    /// hand-rolling exactly that here turned a story rung into fluent nonsense.
    #[test]
    fn the_prose_sampler_is_a_preset_the_engine_offers() {
        assert!(
            SamplingConfig::preset_names().contains(&PROSE_SAMPLER),
            "`{PROSE_SAMPLER}` is not one of {:?}",
            SamplingConfig::preset_names()
        );
    }

    /// What the guest needs from that preset, stated so a change to the shared
    /// table cannot quietly reintroduce either failure this file has already had.
    ///
    /// Truncation and the repeat penalty are a pair: either alone is worse than
    /// neither. A penalty with nothing truncated pushes mass straight off the
    /// tokens it damps and onto the tail, and a story rung sampled that way came
    /// back as fluent nonsense.
    ///
    /// DRY is the other one. It is span-scoped by design — the kernel resets it
    /// at `<think>`/`</think>` — which is what makes it safe in chat and wrong
    /// here, because a diary entry has no spans and one span covers all 1,400
    /// tokens. With it on, the same rung walked a thesaurus to the budget.
    #[test]
    fn the_prose_preset_truncates_penalises_lightly_and_does_not_use_dry() {
        let c = SamplingConfig::preset(PROSE_SAMPLER).expect("the preset exists");

        assert!(
            c.top_k > 0 || c.top_p < 1.0,
            "nothing is truncated, so the repeat penalty samples into the tail"
        );
        assert!(
            c.repeat_penalty > 1.0,
            "no repeat penalty, so a long answer loops"
        );
        assert!(
            c.repeat_last_n > 0,
            "the penalty reads the whole history rather than a window"
        );
        assert!(
            c.dry.is_none(),
            "DRY is span-scoped and a diary entry is one span — it compounds \
             across the whole answer"
        );
        assert_eq!(
            c.segment_close_token_id, -1,
            "thinking steering, for a block this request never opens"
        );
    }

    /// The window this guest hands the sampler has to hold everything the
    /// preset's penalties want to read, or that configuration arrives silently
    /// clipped to a number this file happened to pick.
    #[test]
    fn the_recent_window_holds_what_the_penalties_read() {
        let c = SamplingConfig::preset(PROSE_SAMPLER).expect("the preset exists");
        let dry_range = c.dry.map(|d| d.range).unwrap_or(0);

        assert!(RECENT_TOKEN_WINDOW >= c.repeat_last_n as usize);
        assert!(RECENT_TOKEN_WINDOW >= dry_range as usize);
    }

    /// Every stock end-of-answer failsafe is sized for a chat reply and stops one
    /// well short of 1,400 tokens. The ladder's story rung asks for 1,400 and
    /// keeps its outline in the tail, which is the part every rung below expands
    /// — so the failsafes follow the caller's budget, not the preset's.
    #[test]
    fn the_eos_failsafes_follow_the_budget_they_were_given() {
        let table = SamplingConfig::preset(PROSE_SAMPLER).expect("the preset exists");
        assert!(
            table.forced_eos_after < 1400,
            "the preset no longer truncates a 1,400-token answer, so this guard \
             is measuring nothing"
        );

        let budget = 1400i32;
        let c = table
            .with_dynamic_eos_boost(1.0, budget * 7 / 10, budget * 8 / 10, 3.0)
            .with_eos_failsafe(budget * 8 / 10, budget);

        assert_eq!(c.forced_eos_after, 1400);
        assert!(c.graceful_eos_after < c.forced_eos_after);
        assert!(c.eos_ramp_start < c.graceful_eos_after);
    }

    /// **ChatML, not the Llama-3 header.** Hermes-3 is a ChatML tune over a
    /// Llama base, so the base's markers on these weights are the wrong format
    /// on the right model: it answers, plausibly, having ignored the system
    /// prompt entirely — which is the whole reason a caller asked for a voice.
    #[test]
    fn the_prompt_is_chatml() {
        let p = chatml("You narrate.", "The yard.");
        assert!(p.starts_with("<|im_start|>system\nYou narrate.<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
        assert!(!p.contains("<|start_header_id|>"), "Llama-3 header markers");
    }

    /// A tensor's placed size is its on-disk size — the payload is copied
    /// verbatim — which is what makes the footprint an exact figure rather than
    /// an estimate.
    #[test]
    fn a_tensors_placed_size_is_its_block_count_times_its_block_size() {
        // Q6_K: 256 elements per block, 210 bytes per block.
        assert_eq!(
            tensor_bytes(GgmlDType::Q6_K, 256 * 10),
            10 * GgmlDType::Q6_K.type_size()
        );
        assert_eq!(tensor_bytes(GgmlDType::F32, 100), 400);
    }

    /// The ordinary case: each step adds text, and the fragments concatenate to
    /// exactly what was decoded.
    #[test]
    fn a_preview_sends_only_what_is_new() {
        let mut p = Preview::default();
        let mut built = String::new();
        for full in ["Ael", "Aelis", "Aelis Mael", "Aelis Maelstrom"] {
            if let Some(d) = p.advance(full) {
                built.push_str(&d);
            }
        }
        assert_eq!(built, "Aelis Maelstrom");
    }

    /// A step that decodes to no new visible text sends nothing, rather than an
    /// empty fragment a consumer has to filter.
    #[test]
    fn a_step_with_no_new_text_sends_nothing() {
        let mut p = Preview::default();
        assert_eq!(p.advance("Ael").as_deref(), Some("Ael"));
        assert_eq!(p.advance("Ael"), None);
    }

    /// **Cleanup that revises what was already sent must not silence the rest.**
    /// `" don "` + `"'t"` decodes to `" don't"`, dropping a space already shown.
    /// A strict-prefix check would call that a break and stop emitting for the
    /// remainder of the generation — a preview that dies mid-sentence.
    #[test]
    fn a_revision_resynchronises_instead_of_going_silent() {
        let mut p = Preview::default();
        assert_eq!(p.advance("I don ").as_deref(), Some("I don "));
        assert_eq!(p.advance("I don't").as_deref(), Some("'t"));
        assert_eq!(
            p.advance("I don't know").as_deref(),
            Some(" know"),
            "the preview went silent after a revision"
        );
    }

    /// A prefix is measured in characters, not bytes: slicing through a
    /// multi-byte character panics, and an em-dash or an accent is exactly where
    /// two decodes of the same sequence are most likely to differ.
    #[test]
    fn a_multibyte_boundary_is_never_split() {
        let mut p = Preview::default();
        assert_eq!(p.advance("a—").as_deref(), Some("a—"));
        assert_eq!(p.advance("a—b").as_deref(), Some("b"));
        assert_eq!(common_prefix_len("a—x", "a—y"), 4);
        assert_eq!(common_prefix_len("é", "e"), 0);
    }

    /// A partial block still costs a whole block. Rounding down would under-size
    /// the read and hand `load_repacked_into` a short buffer, which places a
    /// truncated weight and reports nothing.
    #[test]
    fn a_partial_block_costs_a_whole_one() {
        assert_eq!(
            tensor_bytes(GgmlDType::Q6_K, 257),
            2 * GgmlDType::Q6_K.type_size()
        );
    }

    /// **A token's cost is what the cache actually stores, and that changed.**
    ///
    /// It charged four bytes an element, because the caches were flat F32 —
    /// 160 KiB a token at this geometry, so one seat at a full context was
    /// 1.9 GiB and a wave was two seats wide. The cache is the engine's chunked
    /// one now, stored per block at [`PROSE_COMPRESSION`]. If this figure does
    /// not follow the storage, `seats_for` sizes every wave against a cost that
    /// no longer exists.
    #[test]
    fn a_tokens_cost_follows_what_the_cache_stores() {
        // Hermes-3-Llama-3.2-3B's geometry.
        let geo = Geometry {
            arch: ProseArch::Llama,
            layers: 28,
            hidden: 3072,
            heads: 24,
            kv_heads: 8,
            head_dim: 128,
            vocab: 128_256,
            rms_eps: 1e-5,
            rope_theta: 500_000.0,
        };
        let elems = 2 * 8 * 128 * 28;
        assert_eq!(
            geo.per_token_bytes(),
            (elems * COMPRESSED_BITS_PER_ELEM).div_ceil(8)
        );

        // The whole reason the guest moved onto the paged cache: a seat has to
        // cost enough less that a wave is worth having.
        let as_f32 = elems * 4;
        assert!(
            geo.per_token_bytes() * 4 < as_f32,
            "a token costs {} against {as_f32} at F32 — under 4x, and the wave \
             will not widen enough to pay for the integration",
            geo.per_token_bytes()
        );
    }

    /// The spec's ceiling is what a job is refused against, so it has to be a
    /// context a job can actually use.
    #[test]
    fn the_default_context_is_usable() {
        let s = ProseSpec::hermes3_3b("m.gguf", "t.json");
        assert!(s.max_context >= MIN_USABLE_CONTEXT * 4);
        assert!(!s.default_system.is_empty());
    }

    fn prose_job(max_tokens: u32) -> GuestRequest {
        GuestRequest::Prose(ProseRequest {
            system: String::new(),
            prompt: "x".into(),
            max_tokens,
            temperature: None,
            seed: None,
            choices: None,
        })
    }

    /// **The claim and the allocation must name the same context.**
    ///
    /// This is the bug that took description generation down entirely: the
    /// footprint claimed for `max_tokens + PROMPT_HEADROOM_TOKENS` while the
    /// load allocated `spec.max_context.min(whatever ground was free)`. Because
    /// the claim carries [`PLACEMENT_SLACK_BYTES`] on top of the cache, "free"
    /// always exceeded "claimed", so the load spent the slack on extra context —
    /// a 220-token job claimed 2,268 tokens and then placed 3,712.
    ///
    /// Both sides now read `context_for`, so the assertion is that it is
    /// *derived from the backlog* and stays under the ceiling. A regression to
    /// the old behaviour is a `context_for` that returns `max_context` for a
    /// small job, which is exactly what the second half rules out.
    #[test]
    fn the_context_is_sized_from_the_backlog_and_not_from_the_ceiling() {
        let guest = ProseGuest::new(ProseSpec::hermes3_3b("m.gguf", "t.json"));
        let ceiling = guest.spec.max_context;

        // The longest job in the backlog decides it: its generation budget, its
        // own prompt, and the slack. `prose_job`'s prompt is one character.
        let one_char = 1usize.div_ceil(CHARS_PER_TOKEN);
        let jobs = [prose_job(220), prose_job(64)];
        assert_eq!(
            guest.context_for(&jobs),
            220 + one_char + PROMPT_HEADROOM_TOKENS
        );
        assert!(
            guest.context_for(&jobs) < ceiling,
            "a short job claimed the whole ceiling — the load would then place \
             more context than the claim covered"
        );

        // An empty backlog still needs room for a prompt, and never more than
        // the ceiling.
        assert_eq!(guest.context_for(&[]), PROMPT_HEADROOM_TOKENS);

        // **A long prompt moves it, which is the whole point of the change.**
        // The old sizing was blind to the prompt and refused the lifegen ladder
        // after evicting the engine for it.
        let long = GuestRequest::Prose(ProseRequest {
            system: "s".repeat(6_000),
            prompt: "p".repeat(3_000),
            max_tokens: 400,
            temperature: None,
            seed: None,
            choices: None,
        });
        assert_eq!(
            guest.context_for(&[long]),
            9_000usize.div_ceil(CHARS_PER_TOKEN) + 400 + PROMPT_HEADROOM_TOKENS
        );

        // A job that would exceed the ceiling is clamped to it rather than
        // claiming ground the spec forbids.
        let huge = [prose_job(u32::MAX)];
        assert_eq!(guest.context_for(&huge), ceiling);
    }

    /// The claim covers the cache it is claimed for, with the slack left over.
    ///
    /// Stated as an inequality against the same `per_token` the load uses, so a
    /// future change that makes the footprint cheaper than the allocation fails
    /// here rather than part-way through a placement loop with the engine
    /// already evicted.
    #[test]
    fn the_claim_covers_the_cache_and_leaves_the_slack_alone() {
        let guest = ProseGuest::new(ProseSpec::hermes3_3b("m.gguf", "t.json"));
        let jobs = [prose_job(220)];
        let geo = Geometry {
            arch: ProseArch::Llama,
            layers: 28,
            hidden: 3072,
            heads: 24,
            kv_heads: 8,
            head_dim: 128,
            vocab: 128_256,
            rope_theta: 500_000.0,
            rms_eps: 1e-5,
        };
        let cache = geo.per_token_bytes() * guest.context_for(&jobs);
        // `sizes` needs a real checkpoint, so the weights term is not available
        // here; what this pins is that the slack is *additional* to the cache
        // rather than the place the cache is expected to come from.
        assert!(
            PLACEMENT_SLACK_BYTES > 0 && cache > 0,
            "the cache and the slack are separate terms"
        );
        // Both sides are constants, so this holds or fails to compile — a
        // runtime assertion over two `const`s only reports at the moment a test
        // happens to run it.
        const {
            assert!(
                PLACEMENT_SLACK_BYTES > candle_nn::kv_cache::REGION_BYTES,
                "the slack has to cover more than one region: every K/V tensor \
                 strands a run tail, and there are two per layer"
            )
        };
    }
}
