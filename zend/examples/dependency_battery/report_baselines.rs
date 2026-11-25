//! The analysis of the two baselines written out: the secondary family of
//! contrasts (`fig_9_12_2b_effects_secondary.csv`), R1's and P1's retrieval hit rates
//! (`table_9_12_retrieval.csv`), and every arm's grounding, completion tokens
//! and tool calls (`table_9_12_grounding.csv`), each beside its section of
//! `summary.md`. A file without R1 or T1 answers writes the tables empty.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use crate::analyse::{secondary, Scores};
use crate::arms::Arm;
use crate::grounding::grounding_rows;
use crate::hits::retrieval_hits;
use crate::report::{contrast_csv, contrast_table, csv, pct};
use crate::run::Line;

/// A mean that may have no sample, to one decimal.
fn one_place(x: f64) -> String {
    if x.is_nan() {
        "—".into()
    } else {
        format!("{x:.1}")
    }
}

/// Write the baselines' sections into `md` and their CSVs into `dir`.
pub fn write(md: &mut String, lines: &[Line], scores: &Scores, dir: &Path) -> anyhow::Result<()> {
    let family = secondary(scores);
    writeln!(
        md,
        "\n## Figure 9.12-2b — the memory against the baselines (secondary family)\n\n\
         Holm-corrected across this family alone; the pre-registered family above is \
         corrected without it.\n"
    )?;
    if family.is_empty() {
        writeln!(md, "No R1 or T1 answers, so no contrast to test.")?;
    } else {
        contrast_table(md, &family)?;
    }
    fs::write(
        dir.join("fig_9_12_2b_effects_secondary.csv"),
        contrast_csv(&family),
    )?;

    writeln!(
        md,
        "\n## The files handed as text — an answer file among them (R1 by BM25, P1 by provenance)\n"
    )?;
    let hits: Vec<_> = [Arm::R1, Arm::P1]
        .into_iter()
        .filter_map(|a| retrieval_hits(lines, a))
        .collect();
    if hits.is_empty() {
        writeln!(md, "No R1 or P1 answers.")?;
    } else {
        writeln!(
            md,
            "| Arm | Questions | Answers | Answer file retrieved (hit@k) | Top file an answer \
             file (hit@1) | Files per prompt | Files cut at the budget |"
        )?;
        writeln!(md, "|---|---:|---:|---:|---:|---:|---:|")?;
        for h in &hits {
            writeln!(
                md,
                "| {:?} | {} | {} | {}% | {}% | {:.1} | {}% |",
                h.arm,
                h.questions,
                h.answers,
                pct(h.hit),
                pct(h.hit_at_1),
                h.files,
                pct(h.cut)
            )?;
        }
    }
    fs::write(
        dir.join("table_9_12_retrieval.csv"),
        csv(
            &[
                "arm",
                "questions",
                "answers",
                "hit",
                "hit_at_1",
                "files",
                "cut",
            ],
            hits.iter().map(|h| {
                vec![
                    format!("{:?}", h.arm),
                    h.questions.to_string(),
                    h.answers.to_string(),
                    h.hit.to_string(),
                    h.hit_at_1.to_string(),
                    h.files.to_string(),
                    h.cut.to_string(),
                ]
            }),
        ),
    )?;

    let rows = grounding_rows(lines);
    writeln!(md, "\n## Table 9.12 — grounding and cost by arm\n")?;
    writeln!(
        md,
        "| Arm | Answers | Citing a name | Mean grounding | Mean completion tokens | Tool calls \
         read | Mean tool calls |"
    )?;
    writeln!(md, "|---|---:|---:|---:|---:|---:|---:|")?;
    for r in &rows {
        writeln!(
            md,
            "| {:?} | {} | {} | {}% | {} | {} | {} |",
            r.arm,
            r.answers,
            r.cited,
            pct(r.grounding),
            one_place(r.completion_tokens),
            r.counted,
            one_place(r.tool_calls)
        )?;
    }
    fs::write(
        dir.join("table_9_12_grounding.csv"),
        csv(
            &[
                "arm",
                "answers",
                "cited",
                "grounding",
                "completion_tokens",
                "tool_calls_read",
                "tool_calls",
            ],
            rows.iter().map(|r| {
                vec![
                    format!("{:?}", r.arm),
                    r.answers.to_string(),
                    r.cited.to_string(),
                    r.grounding.to_string(),
                    r.completion_tokens.to_string(),
                    r.counted.to_string(),
                    r.tool_calls.to_string(),
                ]
            }),
        ),
    )?;
    if let Some(t1) = rows.iter().find(|r| r.arm == Arm::T1) {
        writeln!(
            md,
            "\nT1 called {} tools an answer on average, over the {} answers whose calls were \
             read back.",
            one_place(t1.tool_calls),
            t1.counted
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyse::scores;
    use crate::placement::Placement;

    fn line(arm: Arm, id: &str, recall: f64) -> Line {
        Line {
            arm: arm.id().into(),
            run: 1,
            id: id.into(),
            category: "direct".into(),
            names_file: false,
            recall,
            missed: Vec::new(),
            grounding: Some(1.0),
            ungrounded: Vec::new(),
            completion_tokens: 10,
            think: None,
            placement: Placement::default(),
            secs: 0.0,
            answer: String::new(),
            error: None,
            retrieved: Vec::new(),
            retrieval_hit: None,
            tool_calls: None,
            view: None,
        }
    }

    /// **A file of the seven memory arms alone still writes every table**,
    /// the baselines' empty.
    #[test]
    fn a_file_without_the_baselines_writes_them_empty() {
        let dir = tempfile::tempdir().unwrap();
        let lines = vec![line(Arm::C1, "q", 1.0), line(Arm::A1, "q", 0.0)];
        let mut md = String::new();
        write(&mut md, &lines, &scores(&lines), dir.path()).unwrap();
        assert!(md.contains("No R1 or T1 answers, so no contrast to test."));
        assert!(md.contains("No R1 or P1 answers."));
        assert!(md.contains("| C1 | 1 | 1 | 100.0% | 10.0 | 0 | — |"));
        assert!(!md.contains("T1 called"));
        assert_eq!(
            fs::read_to_string(dir.path().join("fig_9_12_2b_effects_secondary.csv")).unwrap(),
            "contrast,arm,against,questions,effect,lo,hi,p,p_holm\n"
        );
        assert_eq!(
            fs::read_to_string(dir.path().join("table_9_12_grounding.csv")).unwrap(),
            "arm,answers,cited,grounding,completion_tokens,tool_calls_read,tool_calls\n\
             A1,1,1,1,10,0,NaN\n\
             C1,1,1,1,10,0,NaN\n"
        );
    }

    #[test]
    fn the_baselines_are_tabled_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let mut t1 = line(Arm::T1, "q", 0.5);
        t1.tool_calls = Some(3);
        let lines = vec![line(Arm::C1, "q", 1.0), line(Arm::R1, "q", 0.0), t1];
        let mut md = String::new();
        write(&mut md, &lines, &scores(&lines), dir.path()).unwrap();
        assert!(md.contains(
            "| Provenance memory against retrieval-augmented generation | C1 − R1 | 1 | 100.0 |"
        ));
        assert!(md.contains("| Provenance memory against tool use | C1 − T1 | 1 | 50.0 |"));
        assert!(md.contains("T1 called 3.0 tools an answer on average, over the 1 answers"));
    }
}
