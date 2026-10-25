//! The daemon's scan alone, looped: every calibration question in batches
//! against the files' own turns, pass after pass, each batch named — what a
//! gate's scanning costs, apart from everything the harness does with it.

use std::time::Instant;

use anyhow::Context;
use candle_conversation::provenance::WideQSig;

use crate::background::question_probes;
use crate::corpus::File;
use crate::remote_gallery::{Daemon, Questions, RemoteGallery};

/// Probes per request, as the background pass sends them.
const BATCH: usize = 256;

pub fn run(files: &[File], passes: usize, daemon: &Daemon) -> anyhow::Result<()> {
    let gallery = RemoteGallery::new(files, daemon);
    let probes = question_probes(files);
    let batches: Vec<Vec<&[WideQSig]>> = probes
        .chunks(BATCH)
        .map(|c| c.iter().map(|(_, s)| *s).collect())
        .collect();
    for pass in 0..passes {
        let t = Instant::now();
        for (b, sigs) in batches.iter().enumerate() {
            let tokens: usize = sigs.iter().map(|s| s.len()).sum();
            gallery
                .votes(sigs, &|_, _| false, Questions::Without, &|_, _| false)
                .with_context(|| {
                    format!(
                        "pass {pass} batch {b} of {} ({} probes, {tokens} probe tokens)",
                        batches.len(),
                        sigs.len()
                    )
                })?;
        }
        println!(
            "pass {pass}: {} batches in {:.1}s",
            batches.len(),
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
