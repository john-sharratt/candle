//! The image guest: Z-Image-Turbo, co-resident with the engine for one drain.
//!
//! # What the pipeline is
//!
//! Three models in sequence, in two stages over the same ground. The encoder
//! stands first and conditions every prompt in the backlog; it is dropped, and
//! the transformer and the autoencoder are placed over the bytes it held. So
//! the claim is the larger stage, not the sum — on the 3090 beside a resident
//! engine the sum (11.8 GB) does not fit and the larger stage (8.4 GB) does:
//!
//! | Piece | What it is |
//! |---|---|
//! | Text encoder | Qwen3-4B at Q8_0, read at `hidden_states[-2]` |
//! | Transformer | `ZImageTransformer2DModel`, a 6B NextDiT, at the rung the card calls for |
//! | Autoencoder | The FLUX VAE under diffusers tensor names |
//!
//! See [`candle_transformers::models::z_image`] for the architecture. Two things
//! about it matter here rather than there.
//!
//! **It is guidance-distilled.** The model card runs Turbo at
//! `guidance_scale=0.0`: no negative prompt, and *one* forward per step instead
//! of a two-wide guidance batch. That is half the work of an undistilled model
//! and the reason there is no `guidance` setting to tune.
//!
//! **It is a dozen steps.** Not fifty. A portrait is a couple of seconds of
//! drain rather than a minute of it, which is what makes a guest that blocks
//! every character's thinking an acceptable thing to run at all. The schedule is
//! trained at eight; the callers ask for twelve, which buys detail.
//!
//! **What the seed does not do.** Distillation costs variety, and on this model
//! it costs almost all of it: the same prompt at three seeds returns the same
//! person, at eight steps and at twelve alike. The face is decided by the
//! conditioning, and the same seed with a described subject returns someone
//! else entirely. Nothing in this file is a lever on that — a caller who wants
//! a different face writes a different prompt.
//!
//! # Where the weights live
//!
//! In guest ground — regions the KV side hands back for the length of the drain
//! — placed by [`super::groundgguf`], which repacks every projection the int8
//! matmul can tile into its KO twin on the way in. That is not an optimisation
//! at the margin: measured on the 3090, the same weights on the dequantising
//! path take 2.25 s a step against 0.33 s, so int8 is most of why this fits in a
//! drain.
//!
//! The embedding table is the exception and does not reach the card at all: it
//! stays quantised on the host and only the couple of dozen rows a prompt names
//! are decoded, a 300 KiB transfer ([`EmbedTable`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle::quantized::Int8Mode;
use candle::{DType, Device, Tensor};
use candle_transformers::models::qwen3;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};
use candle_transformers::models::z_image::text_encoder::{EmbedTable, EMBED_TOKENS};
use candle_transformers::models::z_image::{
    model as zi, quantized_model as zq, sampling, TextEncoder,
};

use super::ground::GuestGround;
use super::groundgguf::{ground_bytes, place_gguf};
use super::model::GuestModel;
use super::progress::{GuestEvent, GuestSink};
use super::seed::resolve_seed;
use super::tiled::{decode_tiled, encode_tiled, fit_tile, SCALE, TILE};
use super::varground::GroundVars;
use super::work::{
    Guest, GuestImage, GuestOutcome, GuestRequest, ImageLora, ImageReference, ImageRequest,
};

/// The VAE's latent scaling, from `vae/config.json`.
///
/// FLUX's numbers, because this is FLUX's autoencoder — the config still says
/// `"_name_or_path": "flux-dev"`. `latent = (x − shift) · scale` on the way in
/// and the inverse on the way out; it is not a tuning knob, and a wrong value
/// decodes a differently-scaled latent into saturated mush.
const VAE_SCALE: f64 = 0.3611;
const VAE_SHIFT: f64 = 0.1159;

/// Z-Image conditions on `hidden_states[-2]` — every layer of the encoder but
/// the last, un-normed. The final layer is specialised toward predicting a
/// token, which is not what conditioning wants.
const ENCODER_SKIP_LAST: usize = 1;

/// The width the autoencoder holds its weights and runs its decode at.
///
/// **bf16, and here that is a fitting decision rather than a speed one.** The
/// decoder is the pass with the largest working set in the pipeline — it
/// upsamples to 128 channels at the image's full resolution, so a single f32
/// tensor is 134 MiB at 512×512 and a residual block holds several — and unlike
/// the weights, those activations come from the CUDA pool. A co-resident guest
/// does not get the pool: the engine's span holds nearly the whole card and a
/// drain frees *span regions*, which the pool cannot touch. Measured on the
/// 3090 with a 9 GB engine resident, the guest had 1,875 MiB of pool to work in
/// and an f32 decode did not fit.
///
/// bf16 halves it. f16 would too and must not be used: this is the same
/// autoencoder as FLUX's, its activations reach past f16's 65,504, and the
/// failure is a saturated image rather than an error. bf16 keeps f32's exponent.
const VAE_DTYPE: DType = DType::BF16;

/// The prompt is wrapped in Qwen3's chat template before encoding, because the
/// encoder is an instruct model and was conditioned that way during training.
fn chat_wrap(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// Where the image guest's checkpoints live.
///
/// The two transformers are single GGUF files; the autoencoder is a directory
/// because a safetensors checkpoint is routinely sharded and a deployment that
/// re-downloads it should not have to re-list the shards here.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageSpec {
    /// The Z-Image transformer, at whichever rung this deployment downloaded.
    ///
    /// This is [`ImageLora::Diversity`]'s checkpoint — the standing one, with
    /// the diversity adapter already fused in. A "LoRA" here is a whole file
    /// rather than a runtime patch, because the guest reloads its weights every
    /// drain anyway: fusing offline and swapping the path costs nothing at
    /// serve time, where a live merge would be a per-drain tensor pass.
    pub transformer: PathBuf,
    /// [`ImageLora::Restricted`]'s checkpoint, when the deployment fused one.
    /// What is fused into it is the deployment's business; this only routes.
    ///
    /// Genuinely optional — a deployment that never asks for the variant should
    /// not have to download and fuse a file nothing uses. A request naming it
    /// anyway is refused by name at load, before any weights move.
    pub transformer_restricted: Option<PathBuf>,
    /// The Qwen3-4B text encoder.
    pub text_encoder: PathBuf,
    /// The encoder's `config.json`, from the Z-Image release — the GGUF carries
    /// tensor data and nothing that says how wide the model is.
    pub encoder_config: PathBuf,
    /// The autoencoder's safetensors directory.
    pub vae: PathBuf,
    /// The Qwen3 tokenizer's `tokenizer.json`.
    pub tokenizer: PathBuf,
    /// The schedule's shift, from `scheduler/scheduler_config.json`. 3.0 is
    /// Z-Image's own, and what its step count is distilled against.
    pub shift: f64,
}

impl ImageSpec {
    pub fn z_image(
        transformer: impl Into<PathBuf>,
        text_encoder: impl Into<PathBuf>,
        encoder_config: impl Into<PathBuf>,
        vae: impl Into<PathBuf>,
        tokenizer: impl Into<PathBuf>,
    ) -> Self {
        Self {
            transformer: transformer.into(),
            transformer_restricted: None,
            text_encoder: text_encoder.into(),
            encoder_config: encoder_config.into(),
            vae: vae.into(),
            tokenizer: tokenizer.into(),
            shift: 3.0,
        }
    }

    /// The transformer checkpoint `lora` names, or a refusal a caller can act
    /// on when the deployment never configured that variant.
    fn transformer_for(&self, lora: ImageLora) -> Result<&PathBuf, String> {
        match lora {
            ImageLora::Diversity => Ok(&self.transformer),
            ImageLora::Restricted => self.transformer_restricted.as_ref().ok_or_else(|| {
                format!(
                    "the image guest has no checkpoint for the '{lora}' lora — this deployment \
                     did not configure one"
                )
            }),
        }
    }
}

/// Which lora this backlog loads: the oldest image job's, since that is the job
/// the drain exists to serve. Jobs asking for a different variant are refused
/// by [`GuestModel::run`] and resubmitted, landing in a drain of their own.
fn drain_lora(jobs: &[GuestRequest]) -> ImageLora {
    jobs.iter()
        .find_map(|j| match j {
            GuestRequest::Image(r) => Some(r.lora),
            _ => None,
        })
        .unwrap_or_default()
}

/// The guest. Holds no device memory until [`GuestModel::load`].
pub struct ImageGuest {
    spec: ImageSpec,
    loaded: Option<Loaded>,
}

/// Ground left unarena'd, so a placement after the arena still has somewhere.
///
/// The arena takes the largest free run and a bump never gives any back, so
/// anything the load does afterwards would find nothing. Small, because nothing
/// large is placed after the weights.
const ARENA_HEADROOM_BYTES: usize = 64 << 20;

/// Below this an arena is not worth opening.
///
/// A tile decode's working set is hundreds of megabytes; an arena that cannot
/// hold one would refuse the first carve that overran it, having first taken
/// the ground the pool might have used.
const MIN_ARENA_BYTES: usize = 512 << 20;

struct Loaded {
    device: Device,
    /// Every backlog prompt's conditioning, by prompt, **on the host**.
    ///
    /// Made at load by the encoder, which is gone by the time a draw runs. On
    /// the host rather than the card because the encoder's activations had no
    /// arena to carve from and the transformer was then placed over the ground
    /// the encoder stood in — a conditioning left on the card could share its
    /// bytes with nothing, but one on the host cannot share them by construction.
    /// A few hundred rows of 2,560 floats; the upload per draw is a megabyte.
    conditioning: HashMap<String, Tensor>,
    transformer: zq::ZImageTransformer,
    vae: AutoEncoderKL,
    shift: f64,
    /// Which fused checkpoint is standing, so a job that queued behind this
    /// drain asking for a different one is refused instead of drawn with the
    /// wrong weights.
    lora: ImageLora,
}

impl ImageGuest {
    pub fn new(spec: ImageSpec) -> Self {
        Self { spec, loaded: None }
    }

    /// Ground the weights will take, at `mode`: the larger of the two stages.
    ///
    /// Read from the headers rather than the file lengths, because a KO twin is
    /// not the size of the bytes it was built from — see
    /// [`super::groundgguf::ground_bytes`]. A claim that is short fails at the
    /// last tensor, after the engine has already been evicted for it.
    fn weight_bytes(&self, mode: Int8Mode, transformer: &Path) -> usize {
        let t = ground_bytes(transformer, mode, &[]).unwrap_or(0);
        let e = ground_bytes(&self.spec.text_encoder, mode, &[EMBED_TOKENS]).unwrap_or(0);
        staged(e, t + vae_bytes(&self.spec.vae))
    }
}

/// Ground for two stages that stand on the same bytes one after the other.
fn staged(encode: usize, draw: usize) -> usize {
    encode.max(draw)
}

/// Ground for the latents, the attention bands and the decode, at one size —
/// the activation arena the claim leaves after the weights.
///
/// Two terms, the larger of which is the arena. The area term is the
/// transformer's: what scales with the image is its activations, and this is
/// the rate for those. The floor is the decoder's: since [`decode_tiled`] its
/// working set is one tile whatever the output is, and the tile is sized to
/// the arena ([`autoencoder_tile`]) — so the floor is what holds a
/// [`CLAIMED_TILE`] tile, below which a draw decodes in many small ones.
///
/// Over-claiming costs the engine eviction and nothing else; under-claiming
/// is a draw refused at its first overrun, after the engine was evicted for it.
fn activation_headroom(width: u32, height: u32) -> usize {
    let pixels = width as usize * height as usize;
    let tile_side = CLAIMED_TILE * SCALE;
    let decode = tile_side * tile_side * AUTOENCODER_BYTES_PER_PIXEL;
    (pixels * ACTIVATION_BYTES_PER_PIXEL).max(decode)
}

/// What one draw's activations cost, per pixel of output.
///
/// # Measured
///
/// A denoise step is one block's intermediates, the block chain's two carried
/// hidden states (`kv_cache::guest_chain`) and the step's few tensors between
/// them. On the 3090 a 512×512 step's block peaked at 331 MiB — 1.3 KiB a
/// pixel — and the step at about 1.5 KiB. Two leaves the margin a 1024×1024
/// draw needs without pricing it out of a card with an engine resident: at the
/// old three, 1024² claimed 10.1 GB against the 9.7 the 3090 could free, and
/// every such draw was refused before it started.
const ACTIVATION_BYTES_PER_PIXEL: usize = 2 * 1024;

impl GuestModel for ImageGuest {
    fn guest(&self) -> Guest {
        Guest::Image
    }

    fn footprint_bytes(&self, jobs: &[GuestRequest]) -> usize {
        let largest = jobs
            .iter()
            .filter_map(|j| match j {
                GuestRequest::Image(r) => Some(activation_headroom(r.width, r.height)),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        // The largest job in the backlog, not the sum: the jobs run one after
        // another and each releases its activations before the next starts, so
        // summing would evict the engine for a peak that never happens.
        //
        // The weights are the larger stage, not the sum: `load` conditions the
        // whole backlog before the transformer is placed, so the encoder is
        // loaded once per drain either way and never stands beside it.
        let mode = Int8Mode::auto(&Device::Cpu);
        // Sized against the checkpoint this backlog will actually load. When
        // the backlog names an unconfigured variant the standing checkpoint
        // stands in — the claim only has to be sane, because `load` refuses
        // the same backlog by name before any weights move.
        let transformer = self
            .spec
            .transformer_for(drain_lora(jobs))
            .unwrap_or(&self.spec.transformer);
        self.weight_bytes(mode, transformer) + largest
    }

    fn load(
        &mut self,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        jobs: &[GuestRequest],
    ) -> Result<(), String> {
        // One decision for the whole pipeline: there is no reason for the
        // encoder and the transformer to run at different numeric modes, and a
        // card without the int8 MMA answers `Off` for both.
        let mode = Int8Mode::auto(device);

        // Which fused checkpoint this backlog asked for — decided before any
        // weights move, so a variant the deployment never configured is
        // refused by name rather than discovered as a missing file halfway
        // through placing the pipeline.
        let lora = drain_lora(jobs);
        let transformer_path = self.spec.transformer_for(lora)?.clone();

        let tokenizer = tokenizers::Tokenizer::from_file(&self.spec.tokenizer)
            .map_err(|e| format!("image guest tokenizer {:?}: {e}", self.spec.tokenizer))?;
        let cfg: qwen3::Config = serde_json::from_slice(
            &std::fs::read(&self.spec.encoder_config)
                .map_err(|e| format!("image guest {:?}: {e}", self.spec.encoder_config))?,
        )
        .map_err(|e| format!("image guest {:?}: {e}", self.spec.encoder_config))?;

        // Read on the host and never placed, so the 1.55 GiB of embeddings is
        // neither ground the KV side lost nor VRAM the transformer wanted.
        // **A load stage per line.** A guest load evicts the engine's working
        // set before it starts, so when one dies the question is always *which
        // checkpoint* — and the answer has to survive a hard fault, which a
        // return value does not.
        // **What the POOL has, which is not what the drain freed.**
        //
        // A drain evicts KV regions, and those are span ground — the weights go
        // there and the accounting works. The activations do not: they come from
        // the CUDA pool, which cannot touch the reservation, so the room a guest
        // has to *work* in is whatever the span left over. That number is
        // invisible everywhere else and is exactly what an out-of-memory here
        // means, so the load says it out loud.
        //
        // At info, with the time since the load began: the load is most of a
        // draw's own cost, and which stage of it is the question a slow one
        // always turns out to be.
        let began = std::time::Instant::now();
        let stage = |what: &str| {
            let free = match device {
                Device::Cuda(_) => device.mem_get_info().map(|(f, _)| f).unwrap_or(0),
                _ => 0,
            };
            tracing::info!(
                target: "candle_conversation::guest",
                at_ms = began.elapsed().as_millis() as u64,
                pool_free_mib = free / (1 << 20),
                no_ticket_mib = no_ticket_mib(),
                "image guest: {what}"
            );
        };
        // **Stage one: the encoder, for the whole backlog.** Scoped so the
        // encoder and every tensor it placed are dropped before the ground is
        // rewound below — the scope is the guarantee, not a comment.
        let conditioning = {
            stage("reading the embedding table on the host");
            let table = EmbedTable::from_gguf(&self.spec.text_encoder, &cfg)
                .map_err(|e| format!("image guest: the embedding table: {e}"))?;
            stage("placing the text encoder");
            let vb = place_gguf(
                &self.spec.text_encoder,
                device,
                ground,
                mode,
                &[EMBED_TOKENS],
            )?;
            stage("building the text encoder");
            let encoder = TextEncoder::new(&cfg, ENCODER_SKIP_LAST, mode, vb)
                .map_err(|e| format!("image guest: the text encoder: {e}"))?;
            stage("conditioning the backlog");
            condition(&tokenizer, &table, &encoder, device, jobs, lora)
                .map_err(|e| format!("image guest: conditioning: {e}"))?
        };

        // **Stage two stands where stage one stood.** Every encoder kernel has
        // to have finished reading its weights before the transformer's copies
        // land on them; the host readback in `condition` waited for the last
        // forward, and this barrier states it rather than relying on it.
        device
            .synchronize()
            .map_err(|e| format!("image guest: the stage barrier: {e}"))?;
        if let Some((before, after)) = candle::vram::trim_pool_after_load(device) {
            tracing::debug!(
                target: "candle_conversation::guest",
                released_mib = before.saturating_sub(after) / (1 << 20),
                "image guest: released the encoder's pool blocks"
            );
        }
        // **The activation arena is placed first, on the widest run.** It must
        // be one contiguous stretch, and the bump serves the widest run first:
        // placed after the weights, it got whatever run they left — 1,347 MiB
        // of a 1,536 MiB claim on the 3090, and a 1024-class draw then
        // overran it. The weights are tensors of tens of megabytes and fit in
        // the runs it leaves. Opened once they are in, below.
        let wanted = jobs
            .iter()
            .filter_map(|j| match j {
                GuestRequest::Image(r) => Some(activation_headroom(r.width, r.height)),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let arena = {
            let mut g = ground
                .lock()
                .map_err(|_| "image guest: the ground lock was poisoned".to_string())?;
            g.rewind();
            let len = wanted.min(g.largest_free_run().saturating_sub(ARENA_HEADROOM_BYTES));
            if len >= MIN_ARENA_BYTES {
                g.place(len, 256).ok().map(|at| (at, len))
            } else {
                None
            }
        };

        stage("placing the transformer");
        tracing::debug!(
            target: "candle_conversation::guest",
            lora = %lora,
            path = ?transformer_path,
            "image guest: transformer variant"
        );
        let vb = place_gguf(&transformer_path, device, ground, mode, &[])?;
        stage("building the transformer");
        let transformer = zq::ZImageTransformer::new(zi::Config::turbo(), mode, vb)
            .map_err(|e| format!("image guest: the transformer: {e}"))?;
        stage("placing the autoencoder");

        let vae = {
            let files = super::varground::safetensors_in(&self.spec.vae)?;
            GroundVars::with(&files, VAE_DTYPE, device, ground, |vb| {
                AutoEncoderKL::new(vb, 3, 3, vae_config())
                    .map_err(|e| format!("image guest: the decoder: {e}"))
            })?
        };
        stage("loaded");

        // **The arena the guest's own activations carve from.**
        //
        // `varground` places the weights in ground and says plainly that it does
        // not place activations. This closes that gap: without it a draw
        // allocates its every intermediate from the CUDA pool, which on a
        // co-resident guest is whatever the engine's span left over — and that,
        // not the size of any one tensor, is how a draw runs out of memory with
        // gigabytes of span standing idle.
        //
        // It stands on the run placed for it above, which is the claim
        // `footprint_bytes` makes for activations. Two things have to hold for
        // it to be correct, and both are elsewhere:
        //
        // - **Routing is inheritance-only**, so a chain that starts owned stays
        //   owned however well the arena is sized. The weights carry a routing
        //   seed (`varground::guest_origin`) and each stage seeds its input
        //   (`kv_cache::guest_stage`); that is what a ticket descends from.
        // - **A bump holds a generation's sum, not its peak.** Nothing here
        //   bounds that — the stages inside the model forwards do, per
        //   transformer block and per decoder resnet, and they nest.
        if let Some((at, len)) = arena {
            candle_nn::kv_cache::open_guest_arena(
                &device
                    .as_cuda_device()
                    .map_err(|e| e.to_string())?
                    .cuda_stream(),
                at.ptr,
                len,
            );
            tracing::info!(
                target: "candle_conversation::guest",
                arena_mib = len >> 20,
                wanted_mib = wanted >> 20,
                "image guest: activations arena open"
            );
        }

        self.loaded = Some(Loaded {
            device: device.clone(),
            conditioning,
            transformer,
            vae,
            shift: self.spec.shift,
            lora,
        });
        Ok(())
    }

    /// An image has nothing to *show* until the decoder runs, so what reaches a
    /// watcher here is a count rather than a preview: the drain's `Loading` for
    /// the model crossing the link, then [`GuestEvent::Step`] per denoise step
    /// and once more for the decode.
    fn run(&mut self, request: &GuestRequest, sink: &GuestSink) -> Result<GuestOutcome, String> {
        let GuestRequest::Image(r) = request else {
            return Err(format!(
                "the image guest was handed a {} job — the queue routed by kind and should not \
                 have",
                request.guest()
            ));
        };
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| "the image guest was asked to run before it loaded".to_string())?;
        // A backlog can hold jobs for two different fused checkpoints; only
        // the first's is standing. Refusing the others is correct rather than
        // convenient: drawing them here would silently use the wrong weights,
        // and a resubmitted job lands in a drain that loads its own.
        if r.lora != loaded.lora {
            return Err(format!(
                "the '{}' checkpoint is loaded for this drain and the job asked for '{}' — \
                 resubmit and it will get a drain of its own",
                loaded.lora, r.lora
            ));
        }
        loaded.draw(r, sink).map_err(|e| e.to_string())
    }

    fn unload(&mut self) {
        // **Before the ground goes back, never after.** A ticket that outlived
        // the arena would carve from regions the KV side has taken back, and
        // hand a diffusion model's scratch the same addresses as attention
        // state. Taken first for that reason, and unconditionally — closing an
        // arena that was never opened is a no-op.
        if let Some(l) = self.loaded.as_ref() {
            if let Ok(cuda) = l.device.as_cuda_device() {
                candle_nn::kv_cache::close_guest_arena(&cuda.cuda_stream());
            }
        }
        // Every tensor in here views ground, and the drain hands the ground
        // back the moment this returns.
        self.loaded = None;
    }
}

/// The autoencoder's architecture.
///
/// Diffusers naming, so this loads through the stable-diffusion VAE rather than
/// the flux one — the architecture is FLUX's, but the tensor names are not.
fn vae_config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: zi::Config::turbo().in_channels,
        norm_num_groups: 32,
        use_quant_conv: false,
        use_post_quant_conv: false,
    }
}

/// Bytes the autoencoder's safetensors occupy, from the files themselves — so a
/// deployment that swapped it for a differently-sized checkpoint does not
/// under-claim, which would be discovered after the engine was already evicted.
fn vae_bytes(dir: &std::path::Path) -> usize {
    super::varground::safetensors_in(dir)
        .into_iter()
        .flatten()
        .filter_map(|p| std::fs::metadata(&p).ok().map(|m| m.len() as usize))
        .sum()
}

/// What one autoencoder tile costs the arena, per pixel of output.
///
/// **Measured, with a margin for what the stages now carry.** On the 3090 a
/// 256×256 tile's decode came to a high-water of 1,248 MiB — 19.5 KiB a pixel,
/// most of it the full-resolution resnets' im2col buffers (576 MiB each at a
/// 512 tile) — while every stage's result still went to the pool. Each stage
/// now carries its result in the tile, on top of that. Tighten it from the
/// high-water [`arena_reading`] logs.
const AUTOENCODER_BYTES_PER_PIXEL: usize = 24 << 10;

/// The smallest tile the activation claim is sized to hold, in latent cells —
/// 32, a 256×256 tile.
///
/// The arena is what the claim leaves after the weights, so a claim that only
/// priced the denoise left an arena the decode then had to fit itself to: on
/// the 3090 that came to 192×192 tiles, sixteen of them for a 512×512 draw.
const CLAIMED_TILE: usize = 32;

/// The autoencoder's tile side, sized to the arena this drain opened: the
/// largest whose stages it holds ([`fit_tile`]). The full tile when no arena is
/// open — a guest without one allocates from the pool, which has no cursor to
/// overrun.
fn autoencoder_tile(device: &Device) -> usize {
    let capacity = device.as_cuda_device().ok().and_then(|cuda| {
        candle_nn::kv_cache::guest_domain_stats(cuda.cuda_stream().context().ordinal())
            .map(|(_, _, capacity)| capacity)
    });
    match capacity {
        Some(capacity) => fit_tile(capacity, AUTOENCODER_BYTES_PER_PIXEL),
        None => TILE,
    }
}

/// Where the activation arena stands after `what`: its cursor, its high-water
/// and its capacity.
///
/// At info, because an overrun is an error and this is the number the claim
/// has to be sized from — the high-water of the largest stage. A cursor above
/// zero between stages is carves no stage owned.
fn arena_reading(device: &Device, what: &str) {
    let Ok(cuda) = device.as_cuda_device() else {
        return;
    };
    if let Some((cursor, peak, capacity)) =
        candle_nn::kv_cache::guest_domain_stats(cuda.cuda_stream().context().ordinal())
    {
        tracing::info!(
            target: "candle_conversation::guest",
            cursor_mib = cursor >> 20,
            peak_mib = peak >> 20,
            capacity_mib = capacity >> 20,
            no_ticket_mib = no_ticket_mib(),
            "image guest: activation arena after {what}"
        );
    }
}

/// Bytes the process has allocated from the pool for an operand that had no
/// arena ticket, since it started, in MiB. Process-wide, so the difference
/// between two readings is what happened between them — on any thread.
fn no_ticket_mib() -> u64 {
    use candle::cuda_backend::wave_provenance::{arena_declines, ArenaDecline};
    arena_declines(ArenaDecline::NoTicket).1 >> 20
}

/// Every distinct prompt among the backlog's `lora` jobs, conditioned, on the
/// host.
///
/// A job for another checkpoint is refused at run and is not conditioned here,
/// and a prompt asked twice is encoded once — the encoding depends on nothing
/// else.
fn condition(
    tokenizer: &tokenizers::Tokenizer,
    table: &EmbedTable,
    encoder: &TextEncoder,
    device: &Device,
    jobs: &[GuestRequest],
    lora: ImageLora,
) -> candle::Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for job in jobs {
        let GuestRequest::Image(r) = job else {
            continue;
        };
        if r.lora != lora || out.contains_key(&r.prompt) {
            continue;
        }
        let ids = tokenizer
            .encode(chat_wrap(&r.prompt), true)
            .map_err(|e| candle::Error::Msg(format!("tokenizing: {e}")))?
            .get_ids()
            .to_vec();
        let embeds = table.rows(&ids, device)?;
        // No padding was added, so every position is a real token and the
        // reference's attention-mask filter is the identity here.
        let cap = encoder
            .encode(&embeds)?
            .contiguous()?
            .to_device(&Device::Cpu)?;
        tracing::debug!(
            target: "candle_conversation::guest",
            tokens = ids.len(),
            caption_spread = %spread(&cap),
            "image guest conditioned"
        );
        out.insert(r.prompt.clone(), cap);
    }
    Ok(out)
}

impl Loaded {
    /// Encode a reference and mix the noise the walk will start under.
    ///
    /// Returns the latent to denoise and the schedule to denoise it with, so
    /// the two cannot disagree about how much noise is in the sample — the
    /// mixing sigma is read off the schedule's own first entry rather than
    /// recomputed here.
    ///
    /// # The arithmetic, which is the whole of it
    ///
    /// The flow's parameterisation is `x_σ = (1 − σ)·x₀ + σ·ε`: at σ = 1 the
    /// sample is pure noise, which is why an ordinary draw starts from
    /// `randn`. So joining the trajectory at σ means mixing exactly that, and
    /// the walk from there is the same Euler loop with no other change. Getting
    /// the mix wrong does not fail — it hands the model a sample carrying more
    /// or less noise than the time it is told, and the picture comes out
    /// washed or muddy rather than wrong.
    fn start_from(
        &self,
        reference: &ImageReference,
        noise: &Tensor,
        request: &ImageRequest,
        shift: f64,
        sink: &GuestSink,
    ) -> candle::Result<(Tensor, Vec<f64>)> {
        // **The hold is a fraction of the SCHEDULE, not of sigma.** It names
        // how much of the walk the reference is allowed to skip, and the sigma
        // that corresponds to is whatever the shift says — see
        // [`sampling::sigma_at`], which also records the measurement that
        // settled it. Reading the hold as a sigma directly is the obvious
        // mistake and it crushes the dial's entire useful range into its
        // bottom eighth.
        let remaining = 1.0 - reference.hold.clamp(0.0, 1.0) as f64;
        let start = sampling::sigma_at(remaining, shift);

        // A hold of zero is an ordinary draw, and is answered as one rather
        // than by encoding a picture that will contribute nothing: the dial is
        // continuous at its bottom end, so the UI needs no special case, and
        // the pipeline should not pay for the position either.
        if start >= 1.0 {
            return Ok((
                noise.clone(),
                sampling::sigmas(request.steps as usize, shift),
            ));
        }

        // **The schedule first, before anything is announced.** A reference
        // draw runs the fraction of the budget it covers, so the total is not
        // the count the caller asked for — and a first event carrying the
        // asked-for total would be corrected downward by the second, which a
        // watcher sees as a bar jumping backwards.
        let n = sampling::steps_from(request.steps as usize, shift, start);
        let sigmas = sampling::sigmas_from(n, shift, start);

        // Announced as its own unit of the bar. It is one autoencoder pass, so
        // it is comparable to the decode at the far end — long enough that a
        // watcher left on "Waiting for the engine…" would think the draw had
        // stalled before it started.
        sink.emit(GuestEvent::Step {
            done: 0,
            total: n as u32 + 1,
            what: "reading the reference",
        });

        let (h, w) = (request.height as usize, request.width as usize);
        // `[H, W, 3]` on the host — the layout the caller's RGB8 already is —
        // then permuted to the `[1, 3, H, W]` every convolution wants.
        let px: Vec<f32> = reference.pixels.iter().map(|b| *b as f32).collect();
        let px = Tensor::from_vec(px, (h, w, 3), &self.device)?
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?;
        // The autoencoder's own range. It emits [-1, 1] at the far end and
        // takes [-1, 1] here; feeding it 0..255 encodes a picture that is
        // entirely out of distribution and reads as noise to the transformer.
        let px = super::varground::into_arena(&((px / 127.5)? - 1.0)?, &self.device)?;

        let vae = &self.vae;
        let z = encode_tiled(&px, autoencoder_tile(&self.device), |tile| {
            // A stage per tile, as the decode takes. The mode, not a sample:
            // the noise this latent will carry is chosen by the hold a line
            // below, and a second undeclared helping from the encoder would sit
            // on top of it.
            candle_nn::kv_cache::guest_stage(&tile.to_dtype(VAE_DTYPE)?, |t| {
                Ok(vae.encode(t)?.mode())
            })
        })?;
        // The inverse of the scaling the decode applies on the way out.
        let z = ((z.to_dtype(DType::F32)? - VAE_SHIFT)? * VAE_SCALE)?.squeeze(0)?;
        if z.dims() != noise.dims() {
            candle::bail!(
                "the reference encoded to {:?} and the draw's latent is {:?}",
                z.dims(),
                noise.dims()
            );
        }

        // **The encoder's pool blocks, returned before the denoise asks for
        // its own.** Same trade as the trim between the denoise and the decode
        // — the two passes want differently shaped memory, and freeing one only
        // returns its blocks to the pool in the wrong shapes.
        if let Some((before, after)) = candle::vram::trim_pool_after_load(&self.device) {
            tracing::debug!(
                target: "candle_conversation::guest",
                released_mib = before.saturating_sub(after) / (1 << 20),
                "image guest: released the reference encode's pool blocks"
            );
        }

        // Read off the schedule, so the sample's noise and the time the model
        // is given cannot drift apart.
        let s = sigmas[0];
        let latent = ((z * (1.0 - s))? + (noise * s)?)?;
        tracing::debug!(
            target: "candle_conversation::guest",
            hold = reference.hold,
            start_sigma = s,
            steps = n,
            of = request.steps,
            latent_spread = %spread(&latent),
            "image guest: started from a reference"
        );
        Ok((latent, sigmas))
    }

    /// # What `sink` is told
    ///
    /// One unit per denoise step, and one more for the decode — so `total` is
    /// `steps + 1`. The extra unit is not padding: the decoder is a full-
    /// resolution upsample and is the single longest operation in the draw at
    /// any size worth asking for, so a bar that ended at the last denoise step
    /// would reach 100% and then stall for the part of the wait a watcher most
    /// wants accounted for.
    fn draw(&mut self, request: &ImageRequest, sink: &GuestSink) -> candle::Result<GuestOutcome> {
        // **Every allocation in a draw is supposed to come from ground.**
        //
        // The guest claims what it needs before the engine is evicted for it, so
        // a draw that reaches the CUDA driver for fresh memory is asking for
        // something nobody budgeted — and that, not the size of any one tensor,
        // is the only way a draw can run out of memory. Armed here so the report
        // names the call sites with the stacks that caused them; compiles to
        // nothing without `forbidden_allocations`.
        let _forbidden = candle::forbidden_alloc::armed();
        let out = self.draw_inner(request, sink);
        drop(_forbidden);
        // **What the claim should have been, measured.** A bump does not free
        // within a generation, so a stage's cost is the *sum* of its
        // intermediates rather than their peak — and the moment a carve fails,
        // that output is owned, nothing downstream can inherit, and the rest of
        // the forward is pool. So the high-water here is not a curiosity: it is
        // the number the activation claim has to cover for the arena to hold a
        // whole stage, and the only alternative to measuring it is the guessing
        // this replaced.
        arena_reading(&self.device, "the draw");
        let report = candle::forbidden_alloc::take_report();
        if !report.is_clean() {
            tracing::warn!(
                target: "candle_conversation::guest",
                "image guest: allocations outside ground during a draw:\n{report}"
            );
        }
        out
    }

    fn draw_inner(
        &mut self,
        request: &ImageRequest,
        sink: &GuestSink,
    ) -> candle::Result<GuestOutcome> {
        // A drawn seed is reported back, so an operator who liked a draw can ask
        // for it again. A seed nobody can recover makes every good image a
        // one-off.
        let seed = resolve_seed(request.seed);
        self.device.set_seed(seed)?;

        // **One generation for the whole draw, so every stage inside it is
        // enclosed.** A stage's result is carried in its enclosing generation
        // (`kv_cache::guest_stage`) — a block's in the step, a step's and a
        // tile's here — and the ops between stages carve where their operands
        // live. Without it each step and each tile was outermost, and its
        // result, and everything read from it, went to the pool. Dropped at the
        // end of this function, after the PNG has been read off the card.
        let stream = self.device.as_cuda_device()?.cuda_stream();
        let _draw = candle_nn::kv_cache::begin_guest(&stream).ok();

        let cap = self
            .conditioning
            .get(&request.prompt)
            .ok_or_else(|| {
                candle::Error::Msg(
                    "image guest: this prompt was not in the backlog the drain conditioned".into(),
                )
            })?
            .to_device(&self.device)?;
        // Seeded, so the caption's projection in every step carves here too.
        let cap = super::varground::into_arena(&cap, &self.device)?;

        let (lh, lw) = (request.height as usize / 8, request.width as usize / 8);
        let channels = zi::Config::turbo().in_channels;
        let noise = super::varground::into_arena(
            &Tensor::randn(0f32, 1f32, (channels, lh, lw), &self.device)?,
            &self.device,
        )?;

        // The caller may move the schedule's shift; absent one, the
        // deployment's. It is the only sampler dial this model has — there is
        // no guidance branch to weigh against, because Turbo is distilled.
        let shift = request.shift.unwrap_or(self.shift);

        // **Where the walk begins.** Without a reference it begins at pure
        // noise, which is every ordinary draw. With one, the picture is encoded
        // and the walk joins the schedule partway down it — see
        // [`ImageReference`] for what that does and does not buy.
        let (latent, sigmas) = match &request.reference {
            None => (noise, sampling::sigmas(request.steps as usize, shift)),
            Some(reference) => self.start_from(reference, &noise, request, shift, sink)?,
        };

        // The decode is the `+ 1`; see this function's own note. Taken from the
        // schedule rather than from the request, because a reference draw runs
        // the fraction of the budget it covers and a bar sized off the ask
        // would stop short of full.
        let steps = sigmas.len().saturating_sub(1) as u32;
        let total = steps + 1;
        let mut step_no = 0usize;
        let latent = sampling::denoise(&latent, &sigmas, |x, t| {
            // **A step is a stage, and the blocks inside it are stages too** —
            // see `z_image::quantized_model`'s block loops. The blocks bound
            // the arena: each releases to its mark and carries its hidden state
            // in the step. The step bounds the carry: what its blocks carried
            // goes when it closes, and only the velocity is carried on, here.
            //
            // Turbo is guidance-distilled, so this is the whole step: one
            // forward, no negative prompt, no extrapolation between two halves
            // of a batch.
            let out =
                candle_nn::kv_cache::guest_stage(x, |x| self.transformer.forward(x, &cap, t))?;
            step_no += 1;
            if step_no == 1 {
                arena_reading(&self.device, "the first denoise step");
            }
            tracing::debug!(
                target: "candle_conversation::guest",
                step = step_no,
                of = steps,
                t,
                velocity_spread = %spread(&out),
                "image guest denoised"
            );
            // After the forward rather than before it, so the count is work
            // finished. A step reported on entry would show one that has not
            // happened, and at eight steps that is a whole eighth of the bar.
            sink.emit(GuestEvent::Step {
                done: step_no as u32,
                total,
                what: "denoising",
            });
            Ok(out)
        })?;
        // Between the two passes, so a draw that overruns says which one did:
        // a cursor standing high here is the denoise leaving carves behind its
        // stages, not the decode needing more than the arena.
        arena_reading(&self.device, "the denoise");

        // **Give the denoise's pool blocks back to the driver before the decode
        // asks for its own.**
        //
        // The two passes want very differently shaped memory: thirty-four blocks
        // of transformer activations, then a decoder that wants half-gigabyte
        // runs at full resolution. Freeing the first only returns it to the
        // async pool, which keeps the blocks — so the decode asks the driver for
        // memory the process is already holding in the wrong shapes. That is an
        // out-of-memory with the pool's own accounting saying there is room, and
        // it is the same fault the standalone example hit at 1024×1024.
        //
        // Legal here for the reason `trim_pool_after_load` documents: it
        // synchronises before it unmaps, and the denoise is finished and its
        // result read.
        if let Some((before, after)) = candle::vram::trim_pool_after_load(&self.device) {
            tracing::debug!(
                target: "candle_conversation::guest",
                released_mib = before.saturating_sub(after) / (1 << 20),
                "image guest: released the denoise's pool blocks before decoding"
            );
        }

        // Announced before it runs, not after: this is the one unit long enough
        // that a watcher needs to know it has *started*, and the event that says
        // it finished is the finished image itself.
        sink.emit(GuestEvent::Step {
            done: steps,
            total,
            what: "decoding",
        });
        let latent = ((latent.to_dtype(DType::F32)? / VAE_SCALE)? + VAE_SHIFT)?;
        // **Tiled, so the decoder's peak does not scale with the image.** A
        // latent that fits one tile — every 512×512 draw — is decoded in a
        // single call and this costs nothing; a larger one is decoded in
        // overlapping tiles and cross-faded. See [`tiled_decode`] for the
        // out-of-memory this removes and why the guest's own ground cannot back
        // a decode.
        let vae = &self.vae;
        let pixels = decode_tiled(
            &latent.unsqueeze(0)?,
            autoencoder_tile(&self.device),
            |tile| {
                // A tile is a stage, for the reason a step is: the arena holds
                // one tile's peak rather than every tile's sum, and the decoded
                // pixels — under a megabyte at a 256 tile — are carried on in
                // the draw.
                candle_nn::kv_cache::guest_stage(&tile.to_dtype(VAE_DTYPE)?, |t| vae.decode(t))
            },
        )?;
        tracing::debug!(
            target: "candle_conversation::guest",
            pixel_spread = %spread(&pixels),
            "image guest decoded"
        );
        // The VAE emits [-1, 1]; the encoder wants [0, 255]. Widened here, on
        // the one tensor the host is about to read anyway.
        let pixels = ((pixels.to_dtype(DType::F32)?.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
        let png = encode_png(&pixels)?;
        Ok(GuestOutcome::Image(GuestImage {
            width: request.width,
            height: request.height,
            png,
            seed,
        }))
    }
}

/// `min…max, mean` for a tensor, or why it could not be read.
///
/// A diagnostic, at `debug`, and the smallest thing that tells the three
/// failures apart: all-zero (the weights did not load), all-NaN (the arithmetic
/// went wrong), and a real spread (the pipeline is working and the picture is
/// merely bad). Without it a flat image looks the same in all three cases —
/// which is exactly where the first real draw left off.
fn spread(t: &Tensor) -> String {
    let flat = match t.to_dtype(DType::F32).and_then(|t| t.flatten_all()) {
        Ok(f) => f,
        Err(e) => return format!("unreadable: {e}"),
    };
    let Ok(v) = flat.to_vec1::<f32>() else {
        return "unreadable".into();
    };
    if v.is_empty() {
        return "empty".into();
    }
    let nan = v.iter().filter(|x| !x.is_finite()).count();
    let finite: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return format!("all {} values non-finite", v.len());
    }
    let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
    let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = finite.iter().sum::<f32>() / finite.len() as f32;
    format!(
        "{min:.4}…{max:.4} mean {mean:.4}{}",
        if nan > 0 {
            format!(" ({nan} non-finite)")
        } else {
            String::new()
        }
    )
}

/// Turn a `[1, 3, h, w]` F32 tensor of 0–255 values into PNG bytes.
///
/// **Encoded here, before the drain hands the ground back.** The caller gets
/// bytes rather than a tensor because a tensor viewing guest ground is a view
/// of memory the KV side owns again the moment the drain ends — it would read
/// as an image right up until something else wrote those regions.
fn encode_png(pixels: &Tensor) -> candle::Result<Vec<u8>> {
    let (_, channels, height, width) = pixels.dims4()?;
    if channels != 3 {
        candle::bail!("image guest: the decoder produced {channels} channels, expected 3");
    }
    // CHW on the device to HWC on the host, which is what every encoder wants.
    let hwc = pixels.squeeze(0)?.permute((1, 2, 0))?.contiguous()?;
    let flat: Vec<u8> = hwc
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(|v| v as u8)
        .collect();
    let buf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(width as u32, height as u32, flat).ok_or_else(|| {
            candle::Error::Msg("image guest: the pixel buffer is not w*h*3 bytes".into())
        })?;
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(buf)
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| candle::Error::Msg(format!("image guest: encoding the PNG: {e}")))?;
    Ok(out.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The decode is what the headroom is for.** The figure exists because the
    /// autoencoder, not the transformer, is the peak: the denoise bands its
    /// attention and the decoder upsamples to full resolution. So the headroom
    /// must scale with the *image*, and must not fall below what a small one
    /// still needs.
    /// **The two stages share ground, so the claim is the larger, not the sum.**
    /// On the 3090 beside a resident engine the sum is 2 GB more than the span
    /// can give, and the larger stage fits.
    #[test]
    fn the_weights_claim_the_larger_stage() {
        let (encoder, transformer, vae) = (3_800usize << 20, 7_230usize << 20, 160usize << 20);
        assert_eq!(staged(encoder, transformer + vae), transformer + vae);
        assert_eq!(staged(9 << 30, 1 << 30), 9 << 30);
    }

    #[test]
    fn headroom_scales_with_the_image_and_has_a_floor() {
        let small = activation_headroom(256, 256);
        let one_k = activation_headroom(1024, 1024);
        assert_eq!(
            small,
            1536 << 20,
            "a small image still gets a 256² tile's decode"
        );
        assert_eq!(
            fit_tile(small, AUTOENCODER_BYTES_PER_PIXEL),
            CLAIMED_TILE,
            "the floor holds the tile it is sized for"
        );
        assert_eq!(one_k, 2 << 30, "1024×1024 wants 2 GiB");
        // Four times the pixels, four times the room — above the floor.
        assert_eq!(activation_headroom(2048, 1024), 2 * one_k);
    }

    /// The chat wrapper is what the encoder was conditioned with, and a prompt
    /// that reaches it bare encodes to something the transformer was never
    /// trained against — an image that is not wrong so much as unrelated.
    #[test]
    fn the_prompt_is_wrapped_in_the_instruct_template() {
        let w = chat_wrap("a rusty robot");
        assert!(w.starts_with("<|im_start|>user\n"), "{w}");
        assert!(w.ends_with("<|im_end|>\n<|im_start|>assistant\n"), "{w}");
        assert!(w.contains("a rusty robot"));
    }

    fn image_job(lora: ImageLora) -> GuestRequest {
        GuestRequest::Image(ImageRequest {
            prompt: "a lantern".into(),
            width: 512,
            height: 512,
            steps: 4,
            seed: None,
            lora,
            reference: None,
            shift: None,
        })
    }

    fn spec() -> ImageSpec {
        ImageSpec::z_image(
            "transformer.gguf",
            "encoder.gguf",
            "config.json",
            "vae",
            "tokenizer.json",
        )
    }

    /// **The oldest image job picks the checkpoint.** It is the job the drain
    /// was triggered for, so it must be servable; anything queued behind it
    /// asking for a different variant is refused at `run` and resubmitted.
    #[test]
    fn the_oldest_image_job_decides_the_drains_lora() {
        assert_eq!(drain_lora(&[]), ImageLora::Diversity);
        assert_eq!(
            drain_lora(&[
                image_job(ImageLora::Restricted),
                image_job(ImageLora::Diversity)
            ]),
            ImageLora::Restricted
        );
        assert_eq!(
            drain_lora(&[
                image_job(ImageLora::Diversity),
                image_job(ImageLora::Restricted)
            ]),
            ImageLora::Diversity
        );
    }

    /// The default lora always resolves; a variant resolves only when the
    /// deployment configured a checkpoint for it, and the refusal names it.
    #[test]
    fn a_variant_without_a_checkpoint_is_refused_by_name() {
        let mut s = spec();
        assert_eq!(
            s.transformer_for(ImageLora::Diversity).unwrap(),
            &PathBuf::from("transformer.gguf")
        );
        let err = s.transformer_for(ImageLora::Restricted).unwrap_err();
        assert!(err.contains("restricted"), "{err}");

        s.transformer_restricted = Some("r.gguf".into());
        assert_eq!(
            s.transformer_for(ImageLora::Restricted).unwrap(),
            &PathBuf::from("r.gguf")
        );
    }
}
