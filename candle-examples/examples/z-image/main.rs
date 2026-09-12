//! Z-Image-Turbo text-to-image.
//!
//! Four models in sequence: Qwen3-4B encodes the prompt (taken at
//! `hidden_states[-2]`), a NextDiT transformer denoises a latent along a
//! rectified flow, and the FLUX autoencoder decodes it. See
//! [`candle_transformers::models::z_image`] for what each piece is and which of
//! them candle already carried.
//!
//! Both the encoder and the transformer are GGUF, and on a card with the int8
//! MMA every projection in them runs as a KO twin on the tensor cores — see
//! [`candle_transformers::models::z_image::quant_choice`] for why that is the
//! default rather than a concession.
//!
//! ```bash
//! cargo run --release --features cuda --example z-image -- \
//!     --prompt "a rusty robot walking on a sandy beach, golden hour"
//! ```

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{Error as E, Result};
use candle::quantized::Int8Mode;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;
use candle_transformers::models::stable_diffusion::vae;
use candle_transformers::models::z_image::text_encoder::EmbedTable;
use candle_transformers::models::z_image::{
    model as zi, quant_choice, quantized_model as zq, sampling, TextEncoder, ZQuant,
};
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use clap::Parser;
use tokenizers::Tokenizer;

/// The release, for the pieces that are not weights: the tokenizer, the encoder's
/// config, and the autoencoder (which is 0.3 GB and has no quantised form worth
/// having).
const REPO: &str = "Tongyi-MAI/Z-Image-Turbo";

/// Z-Image conditions on `hidden_states[-2]` — every layer of the encoder but
/// the last, un-normed. The final layer is specialised toward predicting a
/// token, which is not what conditioning wants.
const ENCODER_SKIP_LAST: usize = 1;

/// The VAE's latent scaling, from `vae/config.json`.
///
/// `latent = (x − shift) · scale` on the way in and the inverse on the way out.
/// These are FLUX's numbers because this *is* FLUX's autoencoder — the config
/// still says `"_name_or_path": "flux-dev"`.
const VAE_SCALE: f64 = 0.3611;
const VAE_SHIFT: f64 = 0.1159;

/// The prompt is wrapped in Qwen3's chat template before encoding, because the
/// encoder is an instruct model and was conditioned that way during training.
/// `enable_thinking=True` in the reference, which for this template is the
/// plain generation prompt with an empty think block.
fn chat_wrap(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

/// An explicit rung, for overriding the card's own verdict.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum QuantArg {
    Q6k,
    Q8,
}

impl From<QuantArg> for ZQuant {
    fn from(q: QuantArg) -> Self {
        match q {
            QuantArg::Q6k => ZQuant::Q6K,
            QuantArg::Q8 => ZQuant::Q8_0,
        }
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(
        long,
        default_value = "a rusty robot walking on a sandy beach, golden hour"
    )]
    prompt: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 1024)]
    height: usize,

    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Turbo is distilled for very few steps; the model card runs 8.
    #[arg(long, default_value_t = 8)]
    steps: usize,

    /// The schedule's shift, from `scheduler/scheduler_config.json`.
    #[arg(long, default_value_t = 3.0)]
    shift: f64,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long, default_value = "z-image.png")]
    out: String,

    /// Force a rung rather than letting the card decide. For comparing what a
    /// quantisation costs on hardware that could run more.
    #[arg(long, value_enum)]
    quant: Option<QuantArg>,

    /// A transformer GGUF to load instead of the one the card's rung names.
    ///
    /// What this is for is a *fused* checkpoint — see the `z-image-fuse`
    /// example, which writes the SDA diversity adapter into the weights. The
    /// result is an ordinary GGUF that nothing else needs to know about, so
    /// running one is a path rather than a mode: `--quant` still reports which
    /// rung the card would have chosen, and this overrides where it reads from.
    #[arg(long)]
    transformer: Option<std::path::PathBuf>,

    /// Run the same GGUFs on the standard dequantising matmul instead of the
    /// int8 tensor-core one. The reference the int8 result is judged against —
    /// same weights, same schedule, only the kernel differs.
    #[arg(long)]
    no_int8: bool,

    #[arg(long)]
    tracing: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome, guard) = tracing_chrome::ChromeLayerBuilder::new().build();
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry().with(chrome).init();
        Some(guard)
    } else {
        None
    };

    // The latent is the image over 8, and the transformer patches it by 2 —
    // so a side that is not a multiple of 16 is silently rounded somewhere in
    // between and the output is not the size that was asked for.
    for (name, v) in [("height", args.height), ("width", args.width)] {
        anyhow::ensure!(v % 16 == 0, "{name} must be a multiple of 16, got {v}");
    }

    let device = candle_examples::device(args.cpu)?;
    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
    }

    // The numeric mode both models are loaded for: `Precision` on a card with
    // the int8 MMA (Ampere and up), `Off` elsewhere, in which case the same GGUFs
    // run the standard dequantising path. One decision for the whole pipeline —
    // there is no reason for the encoder and the transformer to disagree.
    let mode = if args.no_int8 {
        Int8Mode::Off
    } else {
        Int8Mode::auto(&device)
    };
    println!("int8 mode {mode:?}");

    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::model(REPO.to_string()));

    // ── the prompt, through Qwen3-4B ─────────────────────────────────────
    //
    // Encoded first and dropped before the transformer is built. They are the
    // two largest tenants, and holding both at once is what would put a 16 GB
    // card out of reach for no reason.
    //
    // The weights are the GGUF; the tokenizer and the config still come from the
    // release, because the GGUF carries neither.
    let cap = {
        println!("loading the text encoder");
        let tok = Tokenizer::from_file(repo.get("tokenizer/tokenizer.json")?).map_err(E::msg)?;
        let cfg: qwen3::Config =
            serde_json::from_slice(&std::fs::read(repo.get("text_encoder/config.json")?)?)?;
        let path = api
            .repo(hf_hub::Repo::model(
                quant_choice::TEXT_ENCODER_REPO.to_string(),
            ))
            .get(quant_choice::TEXT_ENCODER_FILE)?;
        // The table is read on the host and the model without it, so the 1.55 GiB
        // of embeddings never reaches the card — see [`z_image::text_encoder`].
        let table = EmbedTable::from_gguf(&path, &cfg)?;
        let vb = QVarBuilder::from_gguf(path, &device)?;
        let model = TextEncoder::new(&cfg, ENCODER_SKIP_LAST, mode, vb)?;

        let ids = tok
            .encode(chat_wrap(&args.prompt), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        println!("prompt is {} tokens", ids.len());
        // No padding was added, so every position is a real token and the
        // reference's attention-mask filter is the identity here.
        let embeds = table.rows(&ids, &device)?;
        model.encode(&embeds)?.contiguous()?
    };
    // The magnitude, not just the shape: a conditioning signal that has gone to
    // NaN or to zero still has the right dims, and the only place it shows is
    // sixty seconds later in a black image.
    println!(
        "caption embedding {:?}, rms {:.4}",
        cap.dims(),
        cap.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?
    );

    // ── the transformer ──────────────────────────────────────────────────
    //
    // Which checkpoint is the card's decision, not a flag: the transformer is
    // dense, so every parameter is read on every step and the file size is the
    // requirement. See [`z_image::quant_choice`].
    let cfg = zi::Config::turbo();
    let total_vram = match &device {
        Device::Cuda(_) => device.mem_get_info()?.1 as u64,
        // Nothing to size against, so take the smallest rung — it runs
        // anywhere a larger one would.
        _ => 0,
    };
    let quant = match args.quant {
        Some(q) => q.into(),
        None => ZQuant::for_vram(total_vram),
    };
    println!(
        "{:.1} GiB VRAM → {:?} ({:.1} GiB)",
        total_vram as f64 / (1u64 << 30) as f64,
        quant,
        quant.bytes() as f64 / (1u64 << 30) as f64,
    );

    // Scoped, so the transformer is dropped before the autoencoder is built.
    //
    // The three models run in sequence and never need each other, so the card
    // only ever has to hold the largest — not the sum. Holding the transformer
    // across the decode is what put a Q6 run out of memory on a 24 GB card:
    // 5.5 GiB of weights nothing was going to read again, beside a 1024×1024
    // decode that upsamples to 128 channels at full resolution.
    let latent = {
        let path = match &args.transformer {
            Some(p) => {
                println!("loading the transformer from {p:?}");
                p.clone()
            }
            None => {
                let file = quant.filename();
                println!("loading the transformer from {}/{file}", quant.repo());
                api.repo(hf_hub::Repo::model(quant.repo().to_string()))
                    .get(file)?
            }
        };
        let vb = QVarBuilder::from_gguf(path, &device)?;
        let transformer = zq::ZImageTransformer::new(cfg.clone(), mode, vb)?;

        let (lh, lw) = (args.height / 8, args.width / 8);
        let latent = Tensor::randn(0f32, 1f32, (cfg.in_channels, lh, lw), &device)?;

        let sigmas = sampling::sigmas(args.steps, args.shift);
        println!("{} steps, sigmas {:.4?}", args.steps, sigmas);
        let start = std::time::Instant::now();
        let mut step_no = 0usize;
        let latent = sampling::denoise(&latent, &sigmas, |x, t| {
            let step_start = std::time::Instant::now();
            // Turbo is guidance-distilled: the model card runs it at
            // `guidance_scale=0.0`, so there is no negative prompt and no second
            // forward per step. That is half the work of an undistilled model.
            let out = transformer.forward(x, &cap, t)?;
            step_no += 1;
            // The velocity's magnitude, per step. A rectified flow's steps are
            // all the same size, so this is a flat-ish line for a healthy run
            // and the one number that says *when* a run went wrong rather than
            // only that it did.
            println!(
                "  step {step_no}/{} t={t:.4} rms {:.4} {:.2}s",
                args.steps,
                out.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?,
                step_start.elapsed().as_secs_f32()
            );
            Ok(out)
        })?;
        println!("denoised in {:.2}s", start.elapsed().as_secs_f32());
        latent
    };

    // ── decode ───────────────────────────────────────────────────────────
    //
    // **Give the transformer's memory back to the driver, not just to the pool.**
    // Dropping it frees ~7 GiB into the async pool, which keeps the blocks — and
    // they are the transformer's shapes, not the decoder's. A 1024×1024 decode
    // upsamples to 128 channels at full resolution, so it wants half-gigabyte
    // runs the pool has no matching block for, and asks the driver for memory it
    // is already holding. That is an out-of-memory on a 24 GiB card with 17 GiB
    // free by its own accounting.
    //
    // `trim_pool_after_load` synchronises before it unmaps, which is what makes
    // this call site legal: `cuMemPoolTrimTo` is not stream-ordered, so it needs
    // a moment when no kernel holds a pointer into the freed blocks. The denoise
    // is finished and its result has been read, so this is one.
    if let Some((before, after)) = candle::vram::trim_pool_after_load(&device) {
        println!(
            "released {:.1} GiB of the transformer's pool blocks",
            (before.saturating_sub(after)) as f64 / (1u64 << 30) as f64
        );
    }
    println!("loading the autoencoder");
    // Diffusers naming, so this is the stable-diffusion VAE rather than the
    // flux one — the architecture is FLUX's, but the tensor names are not.
    let ae_cfg = vae::AutoEncoderKLConfig {
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        latent_channels: cfg.in_channels,
        norm_num_groups: 32,
        use_quant_conv: false,
        use_post_quant_conv: false,
    };
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[repo.get("vae/diffusion_pytorch_model.safetensors")?],
            DType::F32,
            &device,
        )?
    };
    let ae = vae::AutoEncoderKL::new(vb, 3, 3, ae_cfg)?;

    let latent = ((latent.to_dtype(DType::F32)? / VAE_SCALE)? + VAE_SHIFT)?;
    let img = ae.decode(&latent.unsqueeze(0)?)?;
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?
        .i(0)?;
    candle_examples::save_image(&img, &args.out)?;
    println!("wrote {}", args.out);
    Ok(())
}
