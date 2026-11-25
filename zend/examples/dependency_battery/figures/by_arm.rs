//! Figure 9.12-1, recall by arm: each arm's mean recall over the questions as
//! a dot on its 95% bootstrap interval, each of its runs' means as a small dot
//! beside it. The reprojecting arms (C) are drawn in the post's colour, the
//! arms they are measured against — the baselines R1 and T1 among them, when
//! the answers hold them — in the comparison colour.

use crate::figures::svg::{Anchor, Linear, Svg, WIDTH};
use crate::figures::table::Table;
use crate::figures::text::{ceil_to, pct, pct0, ticks};

/// One arm's row of `fig_9_12_1_recall_by_arm.csv`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmPoint {
    pub arm: String,
    pub questions: u32,
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    pub runs: Vec<f64>,
}

pub fn points(t: &Table) -> anyhow::Result<Vec<ArmPoint>> {
    t.records()
        .map(|r| {
            Ok(ArmPoint {
                arm: r.get("arm")?.to_string(),
                questions: r.num("questions")? as u32,
                mean: r.num("mean")?,
                lo: r.num("lo")?,
                hi: r.num("hi")?,
                runs: r.nums("run_means")?,
            })
        })
        .collect()
}

/// The arm's reprojection and its seed, as its two label lines; a baseline's
/// in place of the reprojection, and no seed.
pub fn dials(arm: &str) -> (&'static str, &'static str) {
    let reprojection = match arm.chars().next() {
        Some('A') => "none",
        Some('D') => "once",
        Some('B') => "held",
        Some('C') => "full",
        Some('R') => "BM25",
        Some('P') => "selected",
        Some('T') => "tools",
        _ => "",
    };
    let baseline = matches!(arm.chars().next(), Some('R' | 'P' | 'T'));
    let seed = if arm.ends_with('1') && !baseline {
        "seed"
    } else {
        "no seed"
    };
    (reprojection, seed)
}

/// The arms at the highest mean, joined for the description.
pub fn best(points: &[ArmPoint]) -> String {
    let top = points
        .iter()
        .map(|p| p.mean)
        .fold(f64::NEG_INFINITY, f64::max);
    let arms: Vec<&str> = points
        .iter()
        .filter(|p| p.mean == top)
        .map(|p| p.arm.as_str())
        .collect();
    match arms.as_slice() {
        [] => String::new(),
        [one] => format!(" {one} recalls the most."),
        many => format!(" {} recall the most.", many.join(" and ")),
    }
}

const X0: f64 = 70.0;
const X1: f64 = WIDTH - 16.0;
const Y0: f64 = 250.0;
const Y1: f64 = 62.0;

pub fn svg(points: &[ArmPoint]) -> String {
    let questions = points.first().map(|p| p.questions).unwrap_or(0);
    let described: Vec<String> = points
        .iter()
        .map(|p| format!("{} {} ({} to {})", p.arm, pct(p.mean), pct(p.lo), pct(p.hi)))
        .collect();
    let aria = format!(
        "Mean recall by arm over {questions} questions, each with its 95% bootstrap interval: {}.{}",
        described.join("; "),
        best(points)
    );
    let mut svg = Svg::new(
        306.0,
        &aria,
        "Recall by arm",
        &format!(
            "mean recall over {questions} questions · bar: 95% bootstrap interval · small dots: each run's mean"
        ),
    );
    let top = ceil_to(points.iter().map(|p| p.hi).fold(0.1, f64::max), 0.1);
    let y = Linear::new((0.0, top), (Y0, Y1));
    svg.y_grid(&y, &ticks(0.0, top, 0.1, pct0), (X0, X1), 0.0);
    let slot = (X1 - X0) / points.len().max(1) as f64;
    for (i, p) in points.iter().enumerate() {
        let cx = X0 + (i as f64 + 0.5) * slot;
        let ours = p.arm.starts_with('C');
        let (band, band_opacity, dot, value) = if ours {
            ("us-soft", None, "dot-us", "v-us")
        } else {
            ("them", Some(0.35), "dot-them", "v-them")
        };
        svg.rect(
            band,
            cx - 7.0,
            y.at(p.hi),
            14.0,
            y.at(p.lo) - y.at(p.hi),
            band_opacity,
        );
        for run in &p.runs {
            svg.circle("q1", cx + 16.0, y.at(*run), 2.5, None);
        }
        svg.circle(dot, cx, y.at(p.mean), 6.0, None);
        svg.text(value, cx, y.at(p.hi) - 8.0, Anchor::Middle, &pct(p.mean));
        let (reprojection, seed) = dials(&p.arm);
        svg.text("cat", cx, 268.0, Anchor::Middle, &p.arm);
        svg.text("cat-sub", cx, 282.0, Anchor::Middle, reprojection);
        svg.text("cat-sub", cx, 295.0, Anchor::Middle, seed);
    }
    svg.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Table {
        Table::parse(
            "arm,questions,mean,lo,hi,run_means,full_marks\n\
             B1,128,0.25,0.2,0.3,0.25;0.2,4\n\
             C1,128,0.375,0.3,0.4,0.4,6\n",
        )
        .unwrap()
    }

    #[test]
    fn the_rows_read_back_from_the_csv() {
        assert_eq!(
            points(&table()).unwrap()[1],
            ArmPoint {
                arm: "C1".into(),
                questions: 128,
                mean: 0.375,
                lo: 0.3,
                hi: 0.4,
                runs: vec![0.4],
            }
        );
    }

    /// Two arms share 554 px: slots 277 wide, centres at 208.5 and 485.5;
    /// the axis tops out at 40%, so 0.25 sits at 250 − 0.25/0.4·188 = 132.5.
    #[test]
    fn each_arm_is_its_interval_its_runs_and_its_mean() {
        let out = svg(&points(&table()).unwrap());
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        for want in [
            "<rect class=\"them\" opacity=\"0.35\" x=\"201.5\" y=\"109\" width=\"14\" height=\"47\"/>",
            "<circle class=\"q1\" cx=\"224.5\" cy=\"132.5\" r=\"2.5\"/>",
            "<circle class=\"q1\" cx=\"224.5\" cy=\"156\" r=\"2.5\"/>",
            "<circle class=\"dot-them\" cx=\"208.5\" cy=\"132.5\" r=\"6\"/>",
            "<text class=\"v-them mid\" x=\"208.5\" y=\"101\">25.0%</text>",
            "<text class=\"cat-sub mid\" x=\"208.5\" y=\"282\">held</text>",
            "<rect class=\"us-soft\" x=\"478.5\" y=\"62\" width=\"14\" height=\"47\"/>",
            "<circle class=\"dot-us\" cx=\"485.5\" cy=\"73.8\" r=\"6\"/>",
            "<text class=\"cat-sub mid\" x=\"485.5\" y=\"295\">seed</text>",
            "<text class=\"tick\" x=\"62\" y=\"65.5\" text-anchor=\"end\">40%</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{out}");
        }
        assert!(out.contains("aria-label=\"Mean recall by arm over 128 questions, each with its 95% bootstrap interval: B1 25.0% (20.0% to 30.0%); C1 37.5% (30.0% to 40.0%). C1 recalls the most.\""));
        assert!(!out.contains("\n\n"));
    }

    /// **A baseline is labelled by what it is handed, and has no seed.**
    #[test]
    fn the_baselines_are_labelled_by_what_they_are_given() {
        assert_eq!(dials("C1"), ("full", "seed"));
        assert_eq!(dials("A0"), ("none", "no seed"));
        assert_eq!(dials("R1"), ("BM25", "no seed"));
        assert_eq!(dials("P1"), ("selected", "no seed"));
        assert_eq!(dials("T1"), ("tools", "no seed"));
    }

    /// **The description names whichever arm recalls the most**, not the one
    /// expected to.
    #[test]
    fn the_description_names_the_best_arm_from_the_data() {
        let mut ps = points(&table()).unwrap();
        ps.push(ArmPoint {
            arm: "R1".into(),
            questions: 128,
            mean: 0.375,
            lo: 0.3,
            hi: 0.45,
            runs: vec![0.375],
        });
        assert_eq!(best(&ps), " C1 and R1 recall the most.");
        ps[2].mean = 0.5;
        assert_eq!(best(&ps), " R1 recalls the most.");
        assert_eq!(best(&[]), "");
    }
}
