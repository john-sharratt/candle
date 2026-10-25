//! The provenance gates over a real substrate, offline
//! (`docs/provenance_scoped_signatures.md` §3.2–§3.4, §4).
//!
//! Opens a workspace's substrate **read-only** — beside a running daemon, which
//! keeps writing it — and runs on the CPU, so a GPU serving decode is left
//! alone. Probes are cut by the projection's own boundary rule and window shape
//! (`provenance::boundaries`, `provenance::probe_windows`) and scored by late
//! fusion's votes (`provenance::window_score`): summed per file, which is
//! today's scorer, and tallied per window under each file aggregate.
//!
//! - **Gate 2** (`corpus::coverage`): every `code_reading` turn's signature
//!   tokens against its tokens.
//! - **Gate 3** (`retrieval`): recall of a file from probes that are not its
//!   own tokens, and the size bias of each rule.
//! - **Gate 4** (`paper`): the paper conversation's projections against
//!   written and mechanical expectations.
//!
//! Run, from the repository root:
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> files [path filter]
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> retrieval [probes per source]
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> sweep [probes per source] [background probes] [only this source]
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> paper <dialogue timeline, hex> <expectations.yaml>

mod corpus;
mod gpu_gallery;
mod paper;
mod ranking;
mod retrieval;
mod stats;
mod sweep;

use std::path::PathBuf;
use std::time::Instant;

use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::substrate::Substrate;
use candle_conversation::{REPROJECT_EVERY_N_TOKENS, REPROJECT_TRIGGER_TEXTS};
use tokenizers::Tokenizer;

use corpus::{Gallery, ANALYSIS, CODE_READING, DIALOGUE};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().map(String::as_str).unwrap_or(".."));
    let command = args.get(1).map(String::as_str).unwrap_or("retrieval");

    let t = Instant::now();
    let mut sub = Substrate::new();
    let mut sp = SubstratePersistence::open_in_with_substrate_read_only(&dir, &mut sub)?;
    println!(
        "opened {} read-only in {:.1}s",
        dir.display(),
        t.elapsed().as_secs_f64()
    );
    let tokenizer = match sp.read_tokenizer_bytes()? {
        Some(bytes) => Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("{e}"))?,
        None => anyhow::bail!("the substrate holds no tokenizer"),
    };
    let triggers: Vec<String> = REPROJECT_TRIGGER_TEXTS.map(String::from).to_vec();
    let rule = BoundaryRule::from_tokenizer(&tokenizer, &triggers, REPROJECT_EVERY_N_TOKENS);

    let layers: Vec<u32> = [CODE_READING, DIALOGUE]
        .into_iter()
        .chain(ANALYSIS)
        .collect();
    let mut by_layer = corpus::load(&sub, &mut sp, &layers)?;
    let files = corpus::files(&mut by_layer, &tokenizer);
    corpus::coverage(&files);
    if files.len() < 2 {
        println!("fewer than two code_reading files — nothing to rank yet");
        return Ok(());
    }
    let Some(gallery) = Gallery::build(&files, &rule) else {
        anyhow::bail!("the gallery holds no signature, or signatures of more than one width");
    };
    println!(
        "gallery: {} files, {} tokens, {} windows",
        files.len(),
        gallery.flat.len(),
        gallery.windows.len()
    );
    match command {
        "files" => {
            let filter = args.get(2).map(String::as_str).unwrap_or("");
            let shown = files
                .iter()
                .filter(|f| f.path.as_deref().unwrap_or("").contains(filter));
            for f in shown {
                println!(
                    "{:016x} {} turns {}",
                    f.timeline,
                    f.turns.len(),
                    f.path.as_deref().unwrap_or("<no path>")
                );
            }
        }
        "retrieval" => {
            let per_source = args.get(2).map(|a| a.parse()).transpose()?.unwrap_or(40);
            retrieval::run(&sub, &files, &gallery, &mut by_layer, &rule, per_source);
        }
        "sweep" => {
            let per_source = args.get(2).map(|a| a.parse()).transpose()?.unwrap_or(40);
            let background_probes = args.get(3).map(|a| a.parse()).transpose()?.unwrap_or(64);
            let only = args.get(4).map(String::as_str);
            sweep::run(
                &sub,
                &files,
                &mut by_layer,
                &rule,
                per_source,
                background_probes,
                only,
            )?;
        }
        "paper" => {
            let timeline = u64::from_str_radix(
                args.get(2)
                    .ok_or_else(|| anyhow::anyhow!("paper needs the dialogue's timeline"))?,
                16,
            )?;
            let expectations = PathBuf::from(
                args.get(3)
                    .ok_or_else(|| anyhow::anyhow!("paper needs the expectations file"))?,
            );
            paper::run(
                &files,
                &gallery,
                &mut by_layer,
                &rule,
                &tokenizer,
                timeline,
                &expectations,
            )?;
        }
        other => anyhow::bail!("unknown command `{other}`: files, retrieval, sweep or paper"),
    }
    Ok(())
}
