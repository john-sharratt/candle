//! Fuse a weight adapter into a Z-Image-Turbo checkpoint.
//!
//! Z-Image-Turbo is guidance-distilled, and distillation costs variety: the same
//! prompt returns the same face whatever the seed, because the identity is
//! decided by the conditioning and not by the noise. The SDA adapter — the one
//! fetched when `--adapter` names nothing — is a LoKr trained to recover that
//! diversity; any ai-toolkit LoKr or LoRA over the same modules fuses the same
//! way. See [`candle_transformers::models::z_image::adapter`] for the formats
//! and why fusing into the weights is better than applying at every step.
//!
//! The output is an ordinary GGUF. Nothing needs to know it was fused — a
//! deployment points at it instead of the stock file and the denoise loop is
//! unchanged.
//!
//! ```bash
//! cargo run --release --example z-image-fuse -- \
//!     --checkpoint ~/.cache/huggingface/.../Z-Image-Turbo-Q6_K.gguf \
//!     --out        Z-Image-Turbo-Q6_K-sda.gguf
//! ```
//!
//! The adapter is downloaded from the hub unless `--adapter` names a local file.
//! `--strength` scales it; 1.0 is what it was trained at, and the card asks for
//! less only when stacking it with other adapters.
//!
//! **This runs on the CPU and wants RAM, not VRAM.** The whole checkpoint is
//! held quantised while the file is written — around 6 GB for a Q6_K 6B — plus
//! one dequantised tensor at a time. It is minutes of work, run once.

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::path::PathBuf;

use anyhow::{Context, Result};
use candle::quantized::gguf_file;
use candle::Device;
use candle_transformers::models::z_image::adapter::{self, Fusion, SDA_FILE, SDA_REPO};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Fuse a LoKr or LoRA adapter into a Z-Image GGUF")]
struct Args {
    /// The Z-Image transformer GGUF to read.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Where to write the fused checkpoint.
    #[arg(long)]
    out: PathBuf,

    /// The adapter safetensors. Downloaded from the hub when absent.
    #[arg(long)]
    adapter: Option<PathBuf>,

    /// How much of the adapter to fuse. 1.0 is the trained strength.
    #[arg(long, default_value_t = adapter::DEFAULT_STRENGTH)]
    strength: f64,

    /// Print the per-tensor `‖Δ‖ / ‖W‖` rather than only the summary.
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !(0.0..=2.0).contains(&args.strength) {
        anyhow::bail!(
            "a strength of {} is outside 0..2 — 1.0 is the trained value and the adapter's card \
             asks for 0.5–0.7 only when stacking",
            args.strength
        );
    }
    if args.out == args.checkpoint {
        anyhow::bail!("--out is the checkpoint itself; fusing in place would destroy the original");
    }

    // The CPU, deliberately. The arithmetic is one pass over the weights and the
    // result goes to a file, so a GPU would buy nothing and would cap the job at
    // whatever VRAM is free.
    let device = Device::Cpu;

    let adapter = match args.adapter {
        Some(p) => p,
        None => {
            println!("fetching {SDA_REPO}/{SDA_FILE}");
            hf_hub::api::sync::Api::new()?
                .model(SDA_REPO.to_string())
                .get(SDA_FILE)?
        }
    };
    let fusion = Fusion::load(&adapter, &device)
        .with_context(|| format!("reading the adapter at {adapter:?}"))?;
    println!("adapter: {} modules from {:?}", fusion.len(), adapter);

    let mut reader = std::fs::File::open(&args.checkpoint)
        .with_context(|| format!("opening {:?}", args.checkpoint))?;
    let content = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("reading {:?} as gguf", args.checkpoint))?;
    println!(
        "checkpoint: {} tensors, {} metadata keys",
        content.tensor_infos.len(),
        content.metadata.len()
    );

    let started = std::time::Instant::now();
    let mut touched = 0usize;
    let mut worst: Option<(String, f64)> = None;
    let mut total = 0f64;
    let tensors = adapter::fuse(
        &content,
        &mut reader,
        &fusion,
        args.strength,
        &device,
        |f| {
            touched += 1;
            total += f.relative;
            if args.verbose {
                println!("  {:<44} {:>8.4}  {:?}", f.name, f.relative, f.dtype);
            }
            if worst.as_ref().is_none_or(|(_, w)| f.relative > *w) {
                worst = Some((f.name.clone(), f.relative));
            }
        },
    )?;

    // **The check that the scale was right.** A LoKr's `alpha` is a sentinel for
    // a full-rank adapter, and reading it as a multiplier would scale every
    // delta by 10¹⁰ — which no shape check would catch. A delta that is a small
    // fraction of the weight it modifies is what a trained adapter looks like;
    // anything near or above the weight itself is arithmetic gone wrong, and it
    // is better to refuse than to write a checkpoint that is quietly ruined.
    let mean = if touched > 0 {
        total / touched as f64
    } else {
        0.0
    };
    if let Some((name, w)) = &worst {
        println!("fused {touched} tensors — mean ‖Δ‖/‖W‖ {mean:.4}, worst {w:.4} at {name}");
        if *w > 1.0 {
            anyhow::bail!(
                "`{name}`'s delta is {w:.3}× the weight it modifies. A trained adapter is a small \
                 correction; this is the signature of a misread scale, and the checkpoint has not \
                 been written."
            );
        }
    }
    if touched == 0 {
        anyhow::bail!("no tensor was adapted — the adapter and the checkpoint do not match");
    }

    let refs: Vec<(&str, &candle::quantized::QTensor)> =
        tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let metadata: Vec<(&str, &gguf_file::Value)> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    let mut out =
        std::fs::File::create(&args.out).with_context(|| format!("creating {:?}", args.out))?;
    gguf_file::write(&mut out, &metadata, &refs)?;
    let bytes = out.metadata()?.len();
    println!(
        "wrote {:?} — {} tensors, {:.2} GB, in {:.1}s",
        args.out,
        refs.len(),
        bytes as f64 / 1e9,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
