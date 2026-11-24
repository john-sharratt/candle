//! The analysis written out: `summary.md`, the tables the paper quotes, and a
//! CSV per figure (`docs/unbounded_agents.md` §9.12, §6.5, §9.6.9), each a
//! plain table a plotting script reads without knowing the battery.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use anyhow::Context;
use serde_json::Value;

use crate::analyse::{
    arm_rows, contrasts, gain_by_naming, reach, recall_by_placement, retrieval_curve, scores,
    telemetry_table, ContrastRow,
};
use crate::arms::{Arm, ARMS};
use crate::load;
use crate::report_baselines;
use crate::run::Line;
use crate::seen;

/// The token positions Figure 9.12-5 samples: every 64 tokens to 4,096.
pub fn curve_grid() -> Vec<u32> {
    (0..=64).map(|i| i * 64).collect()
}

pub fn pct(x: f64) -> String {
    if x.is_nan() {
        "—".into()
    } else {
        format!("{:.1}", x * 100.0)
    }
}

pub fn csv(header: &[&str], rows: impl IntoIterator<Item = Vec<String>>) -> String {
    let mut out = header.join(",");
    out.push('\n');
    for r in rows {
        out.push_str(&r.join(","));
        out.push('\n');
    }
    out
}

/// A family of contrasts as `summary.md`'s table.
pub fn contrast_table(md: &mut String, rows: &[ContrastRow]) -> anyhow::Result<()> {
    writeln!(
        md,
        "| Contrast | Arms | Questions | Effect (points) | 95% interval | p | p (Holm) |"
    )?;
    writeln!(md, "|---|---|---:|---:|---|---:|---:|")?;
    for c in rows {
        writeln!(
            md,
            "| {} | {:?} − {:?} | {} | {} | {} to {} | {:.2e} | {:.2e} |",
            c.name,
            c.arm,
            c.against,
            c.questions,
            pct(c.effect),
            pct(c.lo),
            pct(c.hi),
            c.p,
            c.p_holm
        )?;
    }
    Ok(())
}

/// A family of contrasts as its figure's CSV.
pub fn contrast_csv(rows: &[ContrastRow]) -> String {
    csv(
        &[
            "contrast",
            "arm",
            "against",
            "questions",
            "effect",
            "lo",
            "hi",
            "p",
            "p_holm",
        ],
        rows.iter().map(|c| {
            vec![
                format!("\"{}\"", c.name),
                format!("{:?}", c.arm),
                format!("{:?}", c.against),
                c.questions.to_string(),
                c.effect.to_string(),
                c.lo.to_string(),
                c.hi.to_string(),
                c.p.to_string(),
                c.p_holm.to_string(),
            ]
        }),
    )
}

/// Write every table and figure of `lines` (and of `samples`, the telemetry,
/// whose tokens take `bf16_per_token` bytes each at BF16) into `dir`.
pub fn write(
    lines: &[Line],
    samples: &[Value],
    bf16_per_token: Option<f64>,
    dir: &Path,
) -> anyhow::Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let s = scores(lines);
    let mut md = String::new();
    let failed = lines.iter().filter(|l| l.error.is_some()).count();
    writeln!(
        md,
        "# Battery-128 analysis\n\n{} answers, {failed} failed after every try.\n",
        lines.len()
    )?;

    let arms = arm_rows(lines, &s);
    writeln!(md, "## Figure 9.12-1 — recall by arm\n")?;
    writeln!(
        md,
        "| Arm | Questions | Mean recall | 95% interval | Run means | Full marks |"
    )?;
    writeln!(md, "|---|---:|---:|---|---|---:|")?;
    for r in &arms {
        writeln!(
            md,
            "| {:?} | {} | {}% | {}–{}% | {} | {} |",
            r.arm,
            r.questions,
            pct(r.mean),
            pct(r.lo),
            pct(r.hi),
            r.run_means
                .iter()
                .map(|m| pct(*m))
                .collect::<Vec<_>>()
                .join(", "),
            r.full_marks
        )?;
    }
    fs::write(
        dir.join("fig_9_12_1_recall_by_arm.csv"),
        csv(
            &[
                "arm",
                "questions",
                "mean",
                "lo",
                "hi",
                "run_means",
                "full_marks",
            ],
            arms.iter().map(|r| {
                vec![
                    format!("{:?}", r.arm),
                    r.questions.to_string(),
                    r.mean.to_string(),
                    r.lo.to_string(),
                    r.hi.to_string(),
                    r.run_means
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(";"),
                    r.full_marks.to_string(),
                ]
            }),
        ),
    )?;

    let cs = contrasts(&s);
    writeln!(
        md,
        "\n## Figure 9.12-2 — effect sizes (the pre-registered family)\n"
    )?;
    contrast_table(&mut md, &cs)?;
    fs::write(dir.join("fig_9_12_2_effects.csv"), contrast_csv(&cs))?;
    report_baselines::write(&mut md, lines, &s, dir)?;
    seen::write(&mut md, lines, dir)?;

    // Figure 9.12-3: each question, held (B1) against reprojecting (C1).
    let categories: BTreeMap<&str, (&str, bool)> = lines
        .iter()
        .map(|l| (l.id.as_str(), (l.category.as_str(), l.names_file)))
        .collect();
    let (b1, c1) = (s.get(&Arm::B1), s.get(&Arm::C1));
    let mut above = 0;
    let mut below = 0;
    let mut scatter = Vec::new();
    if let (Some(b1), Some(c1)) = (b1, c1) {
        for (id, x) in b1 {
            if let Some(y) = c1.get(id) {
                above += (y > x) as usize;
                below += (y < x) as usize;
                let (cat, _) = categories.get(id.as_str()).copied().unwrap_or(("", false));
                scatter.push(vec![
                    id.clone(),
                    cat.to_string(),
                    x.to_string(),
                    y.to_string(),
                ]);
            }
        }
    }
    writeln!(
        md,
        "\n## Figure 9.12-3 — question by question\n\n{above} questions above the diagonal \
         (C1 > B1), {below} below.\n"
    )?;
    fs::write(
        dir.join("fig_9_12_3_questions.csv"),
        csv(&["id", "category", "b1", "c1"], scatter),
    )?;

    let names: BTreeMap<String, bool> = categories
        .iter()
        .map(|(id, (_, n))| (id.to_string(), *n))
        .collect();
    let split = gain_by_naming(&s, Arm::C1, Arm::B1, &names);
    writeln!(md, "## Figure 9.12-4 — where the gain falls (C1 − B1)\n")?;
    writeln!(
        md,
        "| Question names its file | Questions | Gain (points) | 95% interval |"
    )?;
    writeln!(md, "|---|---:|---:|---|")?;
    for (named, n, g, lo, hi) in split {
        writeln!(
            md,
            "| {named} | {n} | {} | {} to {} |",
            pct(g),
            pct(lo),
            pct(hi)
        )?;
    }
    fs::write(
        dir.join("fig_9_12_4_gain_by_naming.csv"),
        csv(
            &["names_file", "questions", "gain", "lo", "hi"],
            split.iter().map(|(named, n, g, lo, hi)| {
                vec![
                    named.to_string(),
                    n.to_string(),
                    g.to_string(),
                    lo.to_string(),
                    hi.to_string(),
                ]
            }),
        ),
    )?;

    let grid = curve_grid();
    let mut curve_rows = Vec::new();
    for arm in [Arm::C1, Arm::B1, Arm::D1] {
        for (t, share, lo, hi) in retrieval_curve(lines, arm, &grid) {
            curve_rows.push(vec![
                format!("{arm:?}"),
                t.to_string(),
                share.to_string(),
                lo.to_string(),
                hi.to_string(),
            ]);
        }
    }
    fs::write(
        dir.join("fig_9_12_5_retrieval_curve.csv"),
        csv(&["arm", "token", "share", "lo", "hi"], curve_rows),
    )?;

    let by_placement = recall_by_placement(lines);
    writeln!(
        md,
        "\n## Figure 9.12-6 — recall when the answer file is placed\n"
    )?;
    writeln!(
        md,
        "| Answer file placed for the first answer token | Answers | Recall | 95% interval |"
    )?;
    writeln!(md, "|---|---:|---:|---|")?;
    for (placed, n, r, lo, hi) in by_placement {
        writeln!(
            md,
            "| {placed} | {n} | {}% | {}–{}% |",
            pct(r),
            pct(lo),
            pct(hi)
        )?;
    }
    fs::write(
        dir.join("fig_9_12_6_recall_by_placement.csv"),
        csv(
            &["placed", "answers", "recall", "lo", "hi"],
            by_placement.iter().map(|(p, n, r, lo, hi)| {
                vec![
                    p.to_string(),
                    n.to_string(),
                    r.to_string(),
                    lo.to_string(),
                    hi.to_string(),
                ]
            }),
        ),
    )?;

    writeln!(md, "\n## §6.5 — how the reasoning steers retrieval\n")?;
    writeln!(
        md,
        "| Arm | Answers | Placed at opening | At reasoning's end | For first answer token | \
         Reprojections in thinking | Brought in, of those the opening missed |"
    )?;
    writeln!(md, "|---|---:|---:|---:|---:|---:|---:|")?;
    let mut reach_rows = Vec::new();
    for arm in ARMS.into_iter().filter(|a| a.projects()) {
        let r = reach(lines, arm);
        writeln!(
            md,
            "| {:?} | {} | {}% | {}% | {}% | {:.1} | {}% |",
            r.arm,
            r.answers,
            pct(r.opening),
            pct(r.think_end),
            pct(r.first_answer),
            r.in_thinking,
            pct(r.brought_in)
        )?;
        reach_rows.push(vec![
            format!("{:?}", r.arm),
            r.answers.to_string(),
            r.opening.to_string(),
            r.think_end.to_string(),
            r.first_answer.to_string(),
            r.in_thinking.to_string(),
            r.brought_in.to_string(),
        ]);
    }
    fs::write(
        dir.join("table_6_5_reach.csv"),
        csv(
            &[
                "arm",
                "answers",
                "opening",
                "think_end",
                "first_answer",
                "in_thinking",
                "brought_in",
            ],
            reach_rows,
        ),
    )?;

    let (header, rows) = telemetry_table(samples);
    let header: Vec<&str> = header.iter().map(String::as_str).collect();
    fs::write(dir.join("fig_9_6_telemetry.csv"), csv(&header, rows))?;
    writeln!(
        md,
        "\n{} telemetry samples in `fig_9_6_telemetry.csv` (Figures 9.6-1 to 9.6-3).",
        samples.len()
    )?;

    match bf16_per_token {
        Some(per_token) => {
            let rows = load::table(samples, per_token);
            writeln!(
                md,
                "\n## §9.6.9 — the card under load\n\nBF16 bytes per token: {per_token:.1}.\n"
            )?;
            writeln!(md, "| Under the battery's load | Mean | Peak |")?;
            writeln!(md, "|---|---:|---:|")?;
            for r in &rows {
                writeln!(md, "| {} | {:.2} | {:.2} |", r.label, r.mean, r.peak)?;
            }
            fs::write(
                dir.join("table_9_6_load.csv"),
                csv(
                    &["row", "mean", "peak"],
                    rows.iter()
                        .map(|r| vec![r.label.to_string(), r.mean.to_string(), r.peak.to_string()]),
                ),
            )?;

            let bands = load::bands(samples, per_token);
            let formats: Vec<String> = bands
                .iter()
                .flat_map(|b| b.shares.keys().cloned())
                .collect::<BTreeSet<String>>()
                .into_iter()
                .collect();
            writeln!(
                md,
                "\n| In flight | Samples | Tokens resident | Compression | {} |",
                formats.join(" | ")
            )?;
            writeln!(md, "|---|---:|---:|---:|{}", "---:|".repeat(formats.len()))?;
            let mut band_rows = Vec::new();
            for b in &bands {
                let shares: Vec<f64> = formats
                    .iter()
                    .map(|f| b.shares.get(f).copied().unwrap_or(0.0))
                    .collect();
                writeln!(
                    md,
                    "| {}–{} | {} | {:.0} | {:.2}× | {} |",
                    b.lo,
                    b.hi,
                    b.samples,
                    b.tokens,
                    b.compression,
                    shares
                        .iter()
                        .map(|s| format!("{:.1}%", 100.0 * s))
                        .collect::<Vec<_>>()
                        .join(" | ")
                )?;
                let mut row = vec![
                    b.lo.to_string(),
                    b.hi.to_string(),
                    b.samples.to_string(),
                    b.tokens.to_string(),
                    b.compression.to_string(),
                ];
                row.extend(shares.iter().map(f64::to_string));
                band_rows.push(row);
            }
            let mut header = vec!["lo", "hi", "samples", "tokens", "compression"];
            header.extend(formats.iter().map(String::as_str));
            fs::write(
                dir.join("table_9_6_load_bands.csv"),
                csv(&header, band_rows),
            )?;
        }
        None => writeln!(
            md,
            "\nNo K/V census beside the answers, so no load table (§9.6.9)."
        )?,
    }

    fs::write(dir.join("summary.md"), md)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_curve_is_sampled_every_64_tokens_to_4096() {
        let g = curve_grid();
        assert_eq!((g.len(), g[0], g[1], g[64]), (65, 0, 64, 4_096));
    }

    #[test]
    fn a_csv_is_a_header_and_its_rows() {
        assert_eq!(
            csv(&["a", "b"], vec![vec!["1".into(), "2".into()]]),
            "a,b\n1,2\n"
        );
        assert_eq!(pct(0.1234), "12.3");
        assert_eq!(pct(f64::NAN), "—");
    }
}
