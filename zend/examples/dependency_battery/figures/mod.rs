//! The paper's ten data figures (`docs/unbounded_agents.md` §9.12, §9.6) as
//! inline-SVG snippets, drawn from the CSVs `analyse` writes: each a
//! `<figure class="fig">` coloured only by the site's chart classes, written
//! to `<dir>/figures/<name>.svg.html` for the paper's markdown to carry as is.
//! The three figures over time also write the points they plot as a CSV
//! beside the snippet, so a plot is reproduced from its own data.

mod by_arm;
mod context;
mod curve;
mod effects;
mod forest;
mod mix;
mod needle;
mod placed;
mod questions;
mod samples;
mod svg;
mod table;
mod text;
mod timeline;
mod vram;

use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context};

use samples::{MIX, VRAM};
use table::Table;
use timeline::Drawn;

/// The most points a figure over time plots.
pub const MAX_POINTS: usize = 96;

/// Write every figure of `dir`'s CSVs into `dir/figures`, a gallery of
/// `gallery_files` files ranked for the needle figure and a token taking
/// `bf16_per_token` bytes at BF16 for the context figure; the paths written.
pub fn write(dir: &Path, gallery_files: u32, bf16_per_token: f64) -> anyhow::Result<Vec<PathBuf>> {
    let out = dir.join("figures");
    fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let read = |name: &str| Table::read(&dir.join(name));
    let mut snippets: Vec<(&str, String)> = Vec::new();

    snippets.push((
        "fig_9_12_1_recall_by_arm",
        by_arm::svg(&by_arm::points(&read("fig_9_12_1_recall_by_arm.csv")?)?),
    ));
    let contrasts = read("fig_9_12_2_effects.csv")?;
    let paired = contrasts
        .records()
        .next()
        .ok_or_else(|| anyhow!("no contrasts"))?
        .num("questions")? as u32;
    snippets.push((
        "fig_9_12_2_effects",
        effects::contrasts_svg(&effects::contrast_rows(&contrasts)?, paired),
    ));
    snippets.push((
        "fig_9_12_3_questions",
        questions::svg(&questions::points(&read("fig_9_12_3_questions.csv")?)?),
    ));
    snippets.push((
        "fig_9_12_4_gain_by_naming",
        effects::naming_svg(&effects::naming_rows(&read(
            "fig_9_12_4_gain_by_naming.csv",
        )?)?),
    ));
    snippets.push((
        "fig_9_12_5_retrieval_curve",
        curve::svg(&curve::curves(&read("fig_9_12_5_retrieval_curve.csv")?)?),
    ));
    snippets.push((
        "fig_9_12_6_recall_by_placement",
        placed::svg(&placed::halves(&read(
            "fig_9_12_6_recall_by_placement.csv",
        )?)?),
    ));
    let needles = needle::needles(&read("fig_9_12_7_needle_ranks.csv")?)?;
    snippets.push((
        "fig_9_12_7_needle_ranks",
        needle::svg(
            &needle::analysed(&needles, gallery_files),
            needles.len(),
            gallery_files,
        ),
    ));

    let telemetry = dir.join("fig_9_6_telemetry.csv");
    let raw = samples::read(
        BufReader::new(
            File::open(&telemetry).with_context(|| format!("opening {}", telemetry.display()))?,
        ),
        |h| h.starts_with(VRAM) || h.starts_with(MIX),
    )
    .with_context(|| format!("reading {}", telemetry.display()))?;
    let points = raw.downsample(MAX_POINTS)?;
    let drawn: Vec<(&str, Drawn)> = vec![
        ("fig_9_6_1_vram_by_tenant", vram::figure(&points, &raw)?),
        ("fig_9_6_2_format_mix", mix::figure(&points, &raw)?),
        (
            "fig_9_6_3_context_vs_bf16",
            context::figure(&points, &raw, bf16_per_token)?,
        ),
    ];

    let mut written = Vec::new();
    for (name, d) in drawn {
        let path = out.join(format!("{name}.csv"));
        fs::write(&path, d.csv).with_context(|| format!("writing {}", path.display()))?;
        written.push(path);
        snippets.push((name, d.svg));
    }
    for (name, svg) in snippets {
        if svg.contains("\n\n") {
            return Err(anyhow!(
                "{name} has a blank line, which would end its HTML block"
            ));
        }
        let path = out.join(format!("{name}.svg.html"));
        fs::write(&path, svg).with_context(|| format!("writing {}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}
