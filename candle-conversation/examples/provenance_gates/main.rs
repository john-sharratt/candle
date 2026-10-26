//! The provenance gates over a real substrate, offline
//! (`docs/provenance_scoped_signatures.md` §3.2–§3.4, §4).
//!
//! Opens a workspace's substrate **read-only** — beside a running daemon, which
//! keeps writing it — and never touches the GPU. Probes are cut by the
//! projection's own boundary rule and window shape (`provenance::boundaries`,
//! `provenance::probe_windows`) and scored by late fusion's votes
//! (`provenance::window_score`): summed per file, which is today's scorer, and
//! tallied per window under each file aggregate.
//!
//! - **Gate 2** (`corpus::coverage`): every `code_reading` turn's signature
//!   tokens against its tokens.
//! - **Gate 3** (`retrieval`): recall of a file from probes that are not its
//!   own tokens, and the size bias of each rule.
//! - **Gate 4** (`paper`): the paper conversation's projections against
//!   written and mechanical expectations.
//! - **Questions** (`questions`): each file's calibration questions probing
//!   for it, their own turn left out, with and without the questions in the
//!   gallery.
//! - **Normalization** (`normalization`): every question scored raw and as
//!   production normalizes it, and against each file's background.
//! - **Chunk factor** (`chunk-factor`): a calibration factor per stored
//!   signature chunk, from the votes it wins on other files' questions.
//! - **Density** (`density`): each file's question 0 against galleries holding
//!   more and more of every file's other questions.
//! - **Question cut** (`question-cut`): the anomalous-signal exclusion over a
//!   gallery that holds calibration questions.
//! - **Competitions** (`competitions`): calibration questions scored in a
//!   competition of their own, combined with the content's.
//! - **Thinking** (`thinking`): question and thinking probes against the
//!   calibration competition whole, cut to its questions, or to its thinking.
//! - **Calibration cut** (`calibration-cut`): the calibration competition's
//!   cut as production learns it, over every calibrated file, judged on a
//!   question held out of both the learning and the gallery.
//! - **Momentum** (`momentum`): each file's held-out questions asked in
//!   sequence, their scans fed through the working set's momentum, staying on
//!   one file and switching to another.
//! - **Layers** (`layers`): fold-group weightings for content and for the
//!   calibration competition apart, from one scan each.
//!
//! The sweep, gate 4, the questions gate, the normalization probe and the
//! chunk factor scan on a running daemon's gallery
//! (`POST /v1/substrate/zend/scan`, `remote_gallery`), named by
//! `--daemon <url> --node <member hex>`; the rest scans on the CPU.
//!
//! Run, from the repository root:
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> files [path filter]
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> retrieval [probes per source]
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> sweep [probes per source] [background probes] [only this source] \
//!     --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> paper <dialogue timeline, hex> <expectations.yaml> \
//!     --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> questions [probes per file] --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> normalization [path filter …] --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> chunk-factor --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> density --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> question-cut --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> competitions --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> thinking --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> calibration-cut --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> momentum --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> layers --daemon <url> --node <hex>
//!   cargo run -p candle-conversation --release --example provenance_gates -- \
//!     <workspace dir> scan-loop [passes] --daemon <url> --node <hex>

mod anomaly;
mod attribution;
mod background;
mod boilerplate;
mod calibration_cut;
mod chunk_factor;
mod competitions;
mod corpus;
mod density;
mod judge;
mod layers_gate;
mod momentum_gate;
mod norm_probe;
mod paper;
mod question_cut;
mod questions_gate;
mod ranking;
mod remote_gallery;
mod retrieval;
mod rules;
mod scan_loop;
mod stats;
mod sweep;
mod thinking_gate;

use std::path::PathBuf;
use std::time::Instant;

use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::provenance::boundaries::BoundaryRule;
use candle_conversation::substrate::Substrate;
use candle_conversation::{REPROJECT_EVERY_N_TOKENS, REPROJECT_TRIGGER_TEXTS};
use tokenizers::Tokenizer;

use corpus::{Gallery, ANALYSIS, CODE_READING, DIALOGUE};
use remote_gallery::Daemon;

/// `args` less `--daemon <url>` and `--node <hex>`, and the daemon they name
/// when both are given.
fn split_daemon(args: Vec<String>) -> anyhow::Result<(Vec<String>, Option<Daemon>)> {
    let (mut url, mut node, mut rest) = (None, None, Vec::new());
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--daemon" => {
                url = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--daemon needs a url"))?,
                )
            }
            "--node" => {
                node = Some(
                    it.next()
                        .ok_or_else(|| anyhow::anyhow!("--node needs a member"))?,
                )
            }
            _ => rest.push(a),
        }
    }
    let daemon = match (url, node) {
        (Some(url), Some(node)) => Some(Daemon {
            url: url.trim_end_matches('/').to_string(),
            node,
        }),
        (None, None) => None,
        _ => anyhow::bail!("--daemon and --node go together"),
    };
    Ok((rest, daemon))
}

fn main() -> anyhow::Result<()> {
    let (args, daemon) = split_daemon(std::env::args().skip(1).collect())?;
    let dir = PathBuf::from(args.first().map(String::as_str).unwrap_or(".."));
    let command = args.get(1).map(String::as_str).unwrap_or("retrieval");
    let daemon = || {
        daemon.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "`{command}` scans on a running daemon: give --daemon <url> --node <hex>"
            )
        })
    };

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
    let files = corpus::files(&mut by_layer, &tokenizer, corpus::questions(&sub));
    println!(
        "calibration: {} files, {} questions",
        files.iter().filter(|f| !f.questions.is_empty()).count(),
        files.iter().map(|f| f.questions.len()).sum::<usize>()
    );
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
                daemon()?,
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
                &mut by_layer,
                &rule,
                &tokenizer,
                timeline,
                &expectations,
                daemon()?,
            )?;
        }
        "questions" => {
            let per_file = args.get(2).map(|a| a.parse()).transpose()?.unwrap_or(4);
            questions_gate::run(&files, per_file, daemon()?)?;
        }
        "normalization" => {
            norm_probe::run(&files, &args[2..], daemon()?)?;
        }
        "chunk-factor" => {
            chunk_factor::run(&files, &tokenizer, daemon()?)?;
        }
        "density" => {
            density::run(&files, daemon()?)?;
        }
        "question-cut" => {
            question_cut::run(&files, daemon()?)?;
        }
        "competitions" => {
            competitions::run(&files, daemon()?)?;
        }
        "thinking" => {
            thinking_gate::run(&files, daemon()?)?;
        }
        "calibration-cut" => {
            calibration_cut::run(&files, daemon()?)?;
        }
        "momentum" => {
            momentum_gate::run(&files, daemon()?)?;
        }
        "layers" => {
            layers_gate::run(&files, daemon()?)?;
        }
        "scan-loop" => {
            let passes = args.get(2).map(|a| a.parse()).transpose()?.unwrap_or(5);
            scan_loop::run(&files, passes, daemon()?)?;
        }
        other => {
            anyhow::bail!(
                "unknown command `{other}`: files, retrieval, sweep, paper, questions, \
                 normalization, chunk-factor, density, question-cut, competitions, thinking, \
                 calibration-cut, momentum, layers or scan-loop"
            )
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_daemon_flags_come_out_of_the_arguments_anywhere() {
        let args = |s: &str| s.split(' ').map(String::from).collect::<Vec<_>>();
        let (rest, daemon) =
            split_daemon(args(".. --daemon http://h:8081/ questions 1 --node 00ab")).unwrap();
        assert_eq!(rest, args(".. questions 1"));
        let daemon = daemon.unwrap();
        assert_eq!(
            (daemon.url.as_str(), daemon.node.as_str()),
            ("http://h:8081", "00ab")
        );
        assert!(split_daemon(args(".. questions")).unwrap().1.is_none());
        assert!(split_daemon(args(".. --node 00ab")).is_err());
    }
}
