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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use memmap2::Mmap;

use crate::stencil::{compile, HfVocab, StencilDriver, StencilTreeBuilder, StepMask, Vocab};

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
        // The K/V cache and RoPE tables, at the longest context any job in this
        // backlog can reach. Sized from the backlog rather than from the ceiling
        // because a drain of eight short jobs should not evict the engine for a
        // context none of them will use.
        let longest = jobs
            .iter()
            .filter_map(|j| match j {
                GuestRequest::Prose(r) => Some(r.max_tokens as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let ctx = (longest + PROMPT_HEADROOM_TOKENS).min(self.spec.max_context);
        sizes.weights + sizes.per_token * ctx + PLACEMENT_SLACK_BYTES
    }

    // The backlog does not change which checkpoint a prose guest stands up —
    // there is one voice per deployment — so the jobs go unread here.
    fn load(
        &mut self,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        _jobs: &[GuestRequest],
    ) -> Result<(), String> {
        let started = Instant::now();
        let mut ph = LoadPhases::default();

        let t = Instant::now();
        let tokenizer = checkpoint::tokenizer(&self.spec.tokenizer)?;
        LoadPhases::add(&mut ph.tokenizer_ns, t);

        // The lock is taken for the whole load and released before this
        // returns, so the guest holds no handle on the ground afterwards —
        // which is what `drain::release` checks.
        let model = {
            let mut g = ground
                .lock()
                .map_err(|_| "prose guest: the ground lock was poisoned".to_string())?;
            PlacedLlama::load(device, &mut g, &self.spec, &mut ph).map_err(|e| e.to_string())?
        };
        ph.report(started.elapsed());
        self.tokenizer = Some(tokenizer);
        self.model = Some(model);
        Ok(())
    }

    fn run(&mut self, request: &GuestRequest, sink: &GuestSink) -> Result<GuestOutcome, String> {
        let GuestRequest::Prose(r) = request else {
            return Err(format!(
                "the prose guest was handed a {} job — the queue routed by kind and should not \
                 have",
                request.guest()
            ));
        };
        let (Some(model), Some(tokenizer)) = (self.model.as_mut(), self.tokenizer.as_ref()) else {
            return Err("the prose guest was asked to run before it loaded".into());
        };
        model
            .generate(tokenizer, r, &self.spec.default_system, sink)
            .map_err(|e| e.to_string())
    }

    fn unload(&mut self) {
        // Order matters: the model holds tensors that view the ground, and the
        // ground is dropped by the drain immediately after this returns.
        self.model = None;
        self.tokenizer = None;
    }
}

/// Tokens of room left for a prompt on top of a job's generation budget.
///
/// The cache is claimed before the prompt is tokenised — the claim has to
/// happen between forwards, and tokenising is cheap but happens per job — so
/// the prompt's length is not yet known when the ground is sized. This is the
/// allowance; a prompt past it is refused with its own length in the message
/// rather than overrunning the cache.
const PROMPT_HEADROOM_TOKENS: usize = 2048;

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

/// The checkpoint's geometry, read from its own header.
#[derive(Clone, Copy, Debug)]
struct Geometry {
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
    /// `[cos, sin]` for every position the cache can hold, built once at load.
    rope: (Tensor, Tensor),
    /// Preallocated `[kv_heads, max_context, head_dim]` per layer, written in
    /// place. Growing a cache by `cat` would allocate and copy the whole
    /// history once per token.
    k_cache: Vec<Tensor>,
    v_cache: Vec<Tensor>,
    max_context: usize,
    /// Tokens currently valid in the caches. Reset per job.
    filled: usize,
}

struct PlacedLayer {
    attn_norm: Tensor,
    wq: candle_transformers::models::quantized_matmul::QMatMul,
    wk: candle_transformers::models::quantized_matmul::QMatMul,
    wv: candle_transformers::models::quantized_matmul::QMatMul,
    wo: candle_transformers::models::quantized_matmul::QMatMul,
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

    /// An uninitialised F32 buffer in ground, for a cache the kernels fill.
    pub(super) fn place_uninit_f32(
        device: &Device,
        ground: &mut GuestGround,
        shape: Vec<usize>,
        ph: &mut LoadPhases,
    ) -> candle::Result<Tensor> {
        let elems: usize = shape.iter().product();
        let t = Instant::now();
        let at = ground
            .place(elems * 4, 256)
            .map_err(|e| candle::Error::Msg(format!("prose guest: placing {shape:?}: {e}")))?;
        LoadPhases::add(&mut ph.place_ns, t);
        // SAFETY: as `place_f32`, except the bytes are whatever the region
        // recycle left — which is zero, because the pool zeroes a region before
        // handing it out. Every element is written by `slice_set` before the
        // attention reads it, bounded by `filled`.
        unsafe {
            Tensor::from_leased_cuda_ptr(at.ptr, DType::F32, shape, device, LeaseOrigin::Foreign)
        }
    }

    impl PlacedLlama {
        pub(super) fn load(
            device: &Device,
            ground: &mut GuestGround,
            spec: &ProseSpec,
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

            // The cache is sized from what is *left* rather than from the spec's
            // ceiling: the ground was claimed against an estimate, and a
            // checkpoint slightly larger than the estimate must shorten the
            // context rather than fail the load.
            // **The same arithmetic the footprint claimed against.** Two
            // expressions of "what a token costs" is two places for it to be
            // wrong, and the way it failed is instructive: the RoPE tables were
            // in neither, so the load placed every K/V cache and then ran out
            // with 951 KB of `cos` left to write — after the engine had already
            // been evicted for it.
            let per_token = geo.per_token_bytes();
            // A whole region of slack, because a placement costs more ground
            // than it holds: a run's tail is abandoned whenever the next
            // allocation does not fit it, and how often that happens depends on
            // which regions the KV side handed back.
            let reserve = candle_nn::kv_cache::REGION_BYTES;
            let affordable = ground.free().saturating_sub(reserve) / per_token.max(1);
            let max_context = spec.max_context.min(affordable);
            if max_context < MIN_USABLE_CONTEXT {
                return Err(candle::Error::Msg(format!(
                    "prose guest: after the weights there is room for {max_context} tokens of \
                     K/V, under the {MIN_USABLE_CONTEXT} a job needs — the ground was sized for a \
                     smaller checkpoint than {:?}",
                    spec.gguf
                )));
            }

            let mut k_cache = Vec::with_capacity(geo.layers);
            let mut v_cache = Vec::with_capacity(geo.layers);
            for _ in 0..geo.layers {
                let shape = vec![1, geo.kv_heads, max_context, geo.head_dim];
                k_cache.push(place_uninit_f32(device, ground, shape.clone(), ph)?);
                v_cache.push(place_uninit_f32(device, ground, shape, ph)?);
            }

            let rope = geo.rope_tables(device, max_context, ground, ph)?;

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
                rope,
                k_cache,
                v_cache,
                max_context,
                filled: 0,
            })
        }
    }
}

/// A context shorter than this cannot hold a system prompt and an answer, so a
/// load that can only afford it has failed rather than degraded.
const MIN_USABLE_CONTEXT: usize = 512;

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
    fn per_token_bytes(&self) -> usize {
        // K and V, per layer.
        let kv = 2 * self.kv_heads * self.head_dim * 4 * self.layers;
        // One row each of `cos` and `sin`, shared by every layer.
        let rope = 2 * (self.head_dim / 2) * 4;
        kv + rope
    }

    fn from_gguf(content: &gguf_file::Content) -> candle::Result<Self> {
        let get = |k: &str| {
            content
                .metadata
                .get(k)
                .ok_or_else(|| candle::Error::Msg(format!("prose guest: {k} is not in the GGUF")))
        };
        let u = |k: &str| get(k).and_then(|v| v.to_u32()).map(|v| v as usize);
        let layers = u("llama.block_count")?;
        let hidden = u("llama.embedding_length")?;
        let heads = u("llama.attention.head_count")?;
        let kv_heads = u("llama.attention.head_count_kv")?;
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
            layers,
            hidden,
            heads,
            kv_heads,
            head_dim: hidden / heads,
            vocab,
            rms_eps: get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
            rope_theta: get("llama.rope.freq_base")
                .and_then(|v| v.to_f32())
                .unwrap_or(500_000.0),
        })
    }

    /// `cos` and `sin` for every position the cache can hold.
    ///
    /// Built once at load and placed in ground, because the alternative is
    /// rebuilding two `[len, head_dim/2]` tensors in the pool on every step.
    fn rope_tables(
        &self,
        device: &Device,
        len: usize,
        ground: &mut GuestGround,
        ph: &mut LoadPhases,
    ) -> candle::Result<(Tensor, Tensor)> {
        let t = Instant::now();
        let half = self.head_dim / 2;
        // **The frequencies depend on `i` alone, so they are computed once.**
        // Inside the position loop this was `len` × `half` calls to `powf` —
        // 262,144 of them for a 4096-token context where 64 are enough, every
        // one recomputing a value that had not changed.
        let freqs: Vec<f64> = (0..half)
            .map(|i| 1.0f64 / (self.rope_theta as f64).powf(2.0 * i as f64 / self.head_dim as f64))
            .collect();
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for pos in 0..len {
            for &freq in &freqs {
                let theta = pos as f64 * freq;
                cos.push(theta.cos() as f32);
                sin.push(theta.sin() as f32);
            }
        }
        LoadPhases::add(&mut ph.rope_ns, t);
        Ok((
            placed::place_f32(device, ground, &cos, vec![len, half], ph)?,
            placed::place_f32(device, ground, &sin, vec![len, half], ph)?,
        ))
    }
}

impl PlacedLlama {
    /// Embed `tokens` into a `[1, n, hidden]` F32 tensor.
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
        dst.reshape((1, tokens.len(), h))
    }

    /// One forward over `tokens`, which start at cache position `self.filled`.
    ///
    /// Returns the logits for the **last** position only: a guest job wants the
    /// next token, and materialising `[n, vocab]` for a 500-token prompt is a
    /// 250 MB tensor nothing reads.
    fn forward(&mut self, tokens: &[u32]) -> candle::Result<Tensor> {
        let n = tokens.len();
        let start = self.filled;
        if start + n > self.max_context {
            candle::bail!(
                "prose guest: {n} more tokens would pass the {}-token context this drain's \
                 ground was sized for (already at {start})",
                self.max_context
            );
        }
        let geo = self.geo;
        let (cos, sin) = (
            self.rope.0.narrow(0, start, n)?,
            self.rope.1.narrow(0, start, n)?,
        );
        let mask = causal_mask(n, start, &self.device)?;

        let mut x = self.embed(tokens)?;
        for li in 0..geo.layers {
            let residual = x.clone();
            let h = candle_nn::ops::rms_norm(&x, &self.layers[li].attn_norm, geo.rms_eps as f32)?;

            let q = self.layers[li].wq.forward_live(&h)?;
            let k = self.layers[li].wk.forward_live(&h)?;
            let v = self.layers[li].wv.forward_live(&h)?;

            let q = q
                .reshape((1, n, geo.heads, geo.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let k = k
                .reshape((1, n, geo.kv_heads, geo.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v
                .reshape((1, n, geo.kv_heads, geo.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            let q = candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?;

            // Written in place. A cache grown by `cat` reallocates and copies
            // the whole history once per decoded token, which is quadratic in
            // the answer's length for no reason.
            self.k_cache[li].slice_set(&k, 2, start)?;
            self.v_cache[li].slice_set(&v, 2, start)?;
            let seen = start + n;
            let k_all = self.k_cache[li].narrow(2, 0, seen)?;
            let v_all = self.v_cache[li].narrow(2, 0, seen)?;

            let repeat = geo.heads / geo.kv_heads;
            let k_all = repeat_kv(&k_all, repeat)?;
            let v_all = repeat_kv(&v_all, repeat)?;

            let scale = 1.0 / (geo.head_dim as f64).sqrt();
            let att = (q.matmul(&k_all.transpose(2, 3)?.contiguous()?)? * scale)?;
            let att = att.broadcast_add(&mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let y = att.matmul(&v_all)?;
            let y = y
                .transpose(1, 2)?
                .reshape((1, n, geo.heads * geo.head_dim))?
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
        self.filled = start + n;

        let x = candle_nn::ops::rms_norm(&x, &self.norm, geo.rms_eps as f32)?;
        let last = x.narrow(1, n - 1, 1)?.contiguous()?;
        self.head.forward_live(&last)?.reshape((1, geo.vocab))
    }

    /// Reset the cache between jobs.
    ///
    /// The buffers are not cleared, only the count: attention reads `filled`
    /// positions and never the ones past it, so zeroing would be a full-width
    /// write over bytes nothing reads.
    fn reset(&mut self) {
        self.filled = 0;
    }

    fn generate(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        request: &ProseRequest,
        default_system: &str,
        sink: &GuestSink,
    ) -> candle::Result<GuestOutcome> {
        self.reset();
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

        // No top-p and no top-k: a narrator's job is range, and the two
        // truncations are what make a model that has range write like one that
        // does not. Temperature alone, and `None` for it is greedy — which a
        // caller asking for reproducible prose is entitled to.
        let mut sampler = LogitsProcessor::new(
            seed,
            request.temperature.map(|t| t as f64).filter(|t| *t > 0.0),
            None,
            None,
        );
        let eos = eos_tokens(tokenizer);

        // The stencil, if the caller asked for one. Compiled per request rather
        // than cached: it is a handful of tokens over a two-arm tree, and a
        // cache keyed by the arms would outlive the tokenizer it was built
        // against — the fingerprint exists to catch exactly that mismatch.
        let mut driver = match request.choices.as_deref() {
            Some(arms) if !arms.is_empty() => Some(choice_stencil(tokenizer, arms)?),
            _ => None,
        };

        let mut logits = self.forward(&tokens)?;
        let mut produced: Vec<u32> = Vec::with_capacity(budget);
        let mut live = Preview::default();
        for _ in 0..budget {
            // Under a stencil the sampler chooses only among the tokens the
            // grammar allows. `Prefill` is a run the grammar has already
            // decided — fed through the model without sampling, because there
            // is nothing to choose.
            let mut allowed = None;
            if let Some(d) = driver.as_mut() {
                match d.step() {
                    StepMask::Done => break,
                    StepMask::Prefill(toks) => {
                        for t in toks {
                            produced.push(t);
                            logits = self.forward(&[t])?;
                        }
                        continue;
                    }
                    StepMask::Branch(set) => allowed = Some(set),
                    // A choice tree has no free-text spans. Decoding freely here
                    // would leave the walk's cursor and the sequence disagreeing
                    // about what was produced, so it is refused rather than
                    // guessed at.
                    StepMask::Free { .. } => {
                        candle::bail!(
                            "prose guest: a choice stencil produced a free-text span, which \
                             means the tree is not the two-arm grammar this path assumes"
                        )
                    }
                }
            }
            let step_logits = match &allowed {
                None => logits.squeeze(0)?,
                Some(set) => {
                    let mut row = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                    set.apply(&mut row);
                    let n = row.len();
                    Tensor::from_vec(row, n, logits.device())?
                }
            };
            let next = sampler.sample(&step_logits)?;
            if let Some(d) = driver.as_mut() {
                let bytes = tokenizer.decode(&[next], false).unwrap_or_default();
                d.accept(next, bytes.as_bytes());
            }
            if eos.contains(&next) {
                break;
            }
            produced.push(next);
            // Emitted here rather than after the forward, so a watcher sees the
            // token at the moment it exists instead of one step behind. Skipped
            // entirely when nobody is watching — the decode below is a real cost
            // and there is no point paying it for nobody.
            if sink.is_watched() {
                if let Ok(full) = tokenizer.decode(&produced, true) {
                    if let Some(delta) = live.advance(&full) {
                        sink.emit(GuestEvent::Token(delta));
                    }
                }
            }
            logits = self.forward(&[next])?;
        }

        let text = tokenizer
            .decode(&produced, true)
            .map_err(|e| candle::Error::Msg(format!("prose guest: decoding: {e}")))?;
        Ok(GuestOutcome::Prose {
            text,
            tokens: produced.len() as u32,
            seed,
        })
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

/// A `[1, 1, n, start + n]` additive mask: zero where a query may attend, and
/// `-inf` where it may not.
///
/// The `start` columns are the cache, which every query in this pass may see;
/// the last `n` are this pass's own tokens, which are causal among themselves.
fn causal_mask(n: usize, start: usize, device: &Device) -> candle::Result<Tensor> {
    let total = start + n;
    let mut m = vec![0f32; n * total];
    for row in 0..n {
        for col in 0..total {
            if col > start + row {
                m[row * total + col] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(m, (1, 1, n, total), device)
}

/// Expand `kv_heads` to `heads` by repeating each head `repeat` times.
///
/// Grouped-query attention: eight K/V heads serve twenty-four query heads on
/// the 3 B. A `repeat` of one is the multi-head case and returns the input
/// untouched rather than paying for a copy that changes nothing.
fn repeat_kv(x: &Tensor, repeat: usize) -> candle::Result<Tensor> {
    if repeat == 1 {
        return Ok(x.clone());
    }
    let (b, kv_heads, seq, dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv_heads, repeat, seq, dim))?
        .reshape((b, kv_heads * repeat, seq, dim))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// **A token's cost counts everything that scales with the context.**
    ///
    /// It counted the K/V caches and not the RoPE tables, so the first real
    /// load placed every cache and then ran out with 951 KB of `cos` left to
    /// write — after the engine had already been evicted for it. Both are
    /// preallocated for the whole context; both belong here.
    #[test]
    fn a_tokens_cost_covers_the_caches_and_the_rope_tables() {
        // Hermes-3-Llama-3.2-3B's geometry.
        let geo = Geometry {
            layers: 28,
            hidden: 3072,
            heads: 24,
            kv_heads: 8,
            head_dim: 128,
            vocab: 128_256,
            rms_eps: 1e-5,
            rope_theta: 500_000.0,
        };
        let kv = 2 * 8 * 128 * 4 * 28;
        let rope = 2 * 64 * 4;
        assert_eq!(geo.per_token_bytes(), kv + rope);
        assert!(
            geo.per_token_bytes() > kv,
            "the RoPE tables are not counted, so a load will run out on them"
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
}
