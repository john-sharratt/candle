//! Figures 9.12-2 and 9.12-4 as forest plots: each contrast's paired effect
//! on recall with its Holm-corrected p, and the reprojection gain (C1 − B1)
//! split by whether the question names its answer file.

use crate::figures::forest::{self, Frame, Row};
use crate::figures::table::Table;
use crate::figures::text::{p_value, signed};

/// `fig_9_12_2_effects.csv`'s contrasts as rows, strong where the Holm
/// p-value is below 0.05.
pub fn contrast_rows(t: &Table) -> anyhow::Result<Vec<Row>> {
    t.records()
        .map(|r| {
            let p = r.num("p_holm")?;
            Ok(Row {
                label: r.get("contrast")?.to_string(),
                sub: format!(
                    "{} \u{2212} {} · Holm p = {}",
                    r.get("arm")?,
                    r.get("against")?,
                    p_value(p)
                ),
                effect: r.num("effect")?,
                lo: r.num("lo")?,
                hi: r.num("hi")?,
                strong: p < 0.05,
            })
        })
        .collect()
}

fn interval(r: &Row) -> String {
    format!(
        "{} points ({} to {})",
        signed(r.effect),
        signed(r.lo),
        signed(r.hi)
    )
}

/// A family of contrasts as a forest titled `title`.
fn family_svg(title: &str, rows: &[Row], questions: u32) -> String {
    let described: Vec<String> = rows
        .iter()
        .map(|r| format!("{} ({}), {}", r.label, r.sub, interval(r)))
        .collect();
    let aria = format!(
        "The effect of each contrast on mean recall over {questions} paired questions, with its 95% \
         bootstrap interval and Holm-corrected Wilcoxon p-value: {}.",
        described.join("; ")
    );
    forest::svg(
        &Frame {
            title,
            subtitle: &format!(
                "paired difference in mean recall, points · {questions} questions · 95% interval · faded: Holm p ≥ 0.05"
            ),
            aria: &aria,
            step: 0.1,
        },
        rows,
    )
}

/// Figure 9.12-2: the pre-registered family.
pub fn contrasts_svg(rows: &[Row], questions: u32) -> String {
    family_svg("What each part of the memory buys", rows, questions)
}

/// Figure 9.12-2b: the secondary family, the memory against the baselines.
pub fn secondary_svg(rows: &[Row], questions: u32) -> String {
    family_svg("The memory against the baselines", rows, questions)
}

/// `fig_9_12_4_gain_by_naming.csv`'s two halves as rows, strong where the
/// interval leaves out zero.
pub fn naming_rows(t: &Table) -> anyhow::Result<Vec<Row>> {
    t.records()
        .map(|r| {
            let names = r.flag("names_file")?;
            let n = r.num("questions")? as u32;
            let (lo, hi) = (r.num("lo")?, r.num("hi")?);
            Ok(Row {
                label: if names {
                    format!("Names its file (n={n})")
                } else {
                    format!("Does not (n={n})")
                },
                sub: if names {
                    "the question names its answer file".into()
                } else {
                    "the question leaves its answer file unnamed".into()
                },
                effect: r.num("gain")?,
                lo,
                hi,
                strong: lo > 0.0 || hi < 0.0,
            })
        })
        .collect()
}

pub fn naming_svg(rows: &[Row]) -> String {
    let described: Vec<String> = rows
        .iter()
        .map(|r| format!("{}, {}", r.label, interval(r)))
        .collect();
    let aria = format!(
        "The gain in mean recall of reprojecting during the reasoning (C1) over the held projection \
         (B1), split by whether the question names its answer file, with 95% bootstrap intervals: {}.",
        described.join("; ")
    );
    forest::svg(
        &Frame {
            title: "Where the gain falls",
            subtitle:
                "C1 \u{2212} B1 in mean recall, points · 95% bootstrap interval over questions",
            aria: &aria,
            step: 0.05,
        },
        rows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contrasts_read_as_rows_strong_below_holm_five_percent() {
        let t = Table::parse(
            "contrast,arm,against,questions,effect,lo,hi,p,p_holm\n\
             \"The memory\",C1,A1,128,0.36,0.3,0.42,1e-16,1.83e-15\n\
             \"The seed files\",C1,C0,128,0.004,-0.027,0.035,0.67,0.67\n",
        )
        .unwrap();
        let rows = contrast_rows(&t).unwrap();
        assert_eq!(rows[0].sub, "C1 \u{2212} A1 · Holm p = 1.8e-15");
        assert_eq!((rows[0].strong, rows[1].strong), (true, false));
        assert_eq!(rows[1].label, "The seed files");
        let out = contrasts_svg(&rows, 128);
        assert!(out.contains(
            "aria-label=\"The effect of each contrast on mean recall over 128 paired questions, with its 95% \
             bootstrap interval and Holm-corrected Wilcoxon p-value: The memory (C1 \u{2212} A1 · Holm p = 1.8e-15), \
             +36.0 points (+30.0 to +42.0); The seed files (C1 \u{2212} C0 · Holm p = 0.670), +0.4 points (\u{2212}2.7 to +3.5).\""
        ));
    }

    /// **The secondary family is drawn as its own forest**, under its own
    /// title, from a CSV of the same columns.
    #[test]
    fn the_secondary_family_is_its_own_forest() {
        let t = Table::parse(
            "contrast,arm,against,questions,effect,lo,hi,p,p_holm\n\
             \"Provenance memory against tool use\",C1,T1,128,0.1,0.05,0.15,0.001,0.002\n",
        )
        .unwrap();
        let rows = contrast_rows(&t).unwrap();
        assert_eq!(rows[0].sub, "C1 \u{2212} T1 · Holm p = 0.002");
        let out = secondary_svg(&rows, 128);
        assert!(out.contains(">The memory against the baselines</text>"));
        assert!(out.contains(">Provenance memory against tool use</text>"));
        assert!(!out.contains("What each part of the memory buys"));
        assert!(!out.contains("\n\n"));
    }

    #[test]
    fn the_naming_split_reads_as_two_labelled_rows() {
        let t = Table::parse(
            "names_file,questions,gain,lo,hi\n\
             true,17,-0.03,-0.11,0.04\n\
             false,111,0.058,0.022,0.097\n",
        )
        .unwrap();
        let rows = naming_rows(&t).unwrap();
        assert_eq!(
            rows[0],
            Row {
                label: "Names its file (n=17)".into(),
                sub: "the question names its answer file".into(),
                effect: -0.03,
                lo: -0.11,
                hi: 0.04,
                strong: false,
            }
        );
        assert_eq!(
            (rows[1].label.as_str(), rows[1].strong),
            ("Does not (n=111)", true)
        );
        let out = naming_svg(&rows);
        assert!(out.contains("<text class=\"tick mid\" x=\"372\" y=\"164\">\u{2212}15</text>"));
        assert!(out.contains("<text class=\"tick mid\" x=\"604\" y=\"164\">+10</text>"));
    }
}
