//! Figure 9.12-6, recall by placement: the recall of the answers whose
//! projection held an answer file for their first answer token, against the
//! answers whose did not, each a bar with its 95% bootstrap interval.

use crate::figures::svg::{Anchor, Linear, Svg};
use crate::figures::table::Table;
use crate::figures::text::{ceil_to, n, pct, pct0, thousands, ticks};

/// One half of `fig_9_12_6_recall_by_placement.csv`.
#[derive(Debug, Clone, PartialEq)]
pub struct Half {
    pub placed: bool,
    pub answers: u64,
    pub recall: f64,
    pub lo: f64,
    pub hi: f64,
}

pub fn halves(t: &Table) -> anyhow::Result<Vec<Half>> {
    t.records()
        .map(|r| {
            Ok(Half {
                placed: r.flag("placed")?,
                answers: r.num("answers")? as u64,
                recall: r.num("recall")?,
                lo: r.num("lo")?,
                hi: r.num("hi")?,
            })
        })
        .collect()
}

const X0: f64 = 250.0;
const X1: f64 = 600.0;
const TOP: f64 = 70.0;
const ROW: f64 = 46.0;

fn label(h: &Half) -> &'static str {
    if h.placed {
        "Answer file placed"
    } else {
        "Not placed"
    }
}

pub fn svg(halves: &[Half]) -> String {
    let bottom = TOP + halves.len() as f64 * ROW;
    let described: Vec<String> = halves
        .iter()
        .map(|h| {
            format!(
                "{} ({} answers): {} ({} to {})",
                label(h).to_lowercase(),
                thousands(h.answers),
                pct(h.recall),
                pct(h.lo),
                pct(h.hi)
            )
        })
        .collect();
    let aria = format!(
        "Mean recall of the answers by whether their projection held an answer file for the first \
         answer token, with 95% bootstrap intervals: {}.",
        described.join("; ")
    );
    let mut svg = Svg::new(
        bottom + 26.0,
        &aria,
        "Recall when the answer file is in view",
        "mean recall by whether an answer file was placed for the first answer token · 95% interval",
    );
    let top = ceil_to(halves.iter().map(|h| h.hi).fold(0.1, f64::max), 0.1);
    let x = Linear::new((0.0, top), (X0, X1));
    let columns: Vec<(f64, String)> = ticks(0.0, top, 0.1, pct0)
        .into_iter()
        .map(|(v, l)| (x.at(v), l))
        .collect();
    svg.x_grid(&columns, (TOP - 4.0, bottom), bottom + 14.0);
    for (i, h) in halves.iter().enumerate() {
        let cy = TOP + i as f64 * ROW + 16.0;
        let bar = if h.placed { "us" } else { "us-soft" };
        svg.text("cat", 16.0, cy - 2.0, Anchor::Start, label(h));
        svg.text(
            "cat-sub",
            16.0,
            cy + 12.0,
            Anchor::Start,
            &format!("{} answers", thousands(h.answers)),
        );
        svg.rect(bar, X0, cy - 11.0, x.at(h.recall) - X0, 22.0, None);
        let (lo, hi) = (n(x.at(h.lo)), n(x.at(h.hi)));
        let (up, down, mid) = (n(cy - 6.0), n(cy + 6.0), n(cy));
        svg.path(
            "grid base",
            &format!("M{lo} {up} V{down} M{lo} {mid} H{hi} M{hi} {up} V{down}"),
            None,
        );
        svg.text(
            "v-us",
            x.at(h.hi) + 8.0,
            cy + 4.0,
            Anchor::Start,
            &pct(h.recall),
        );
    }
    svg.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The axis tops out at 60% over 350 px, so 50% ends its bar at 541.7.
    #[test]
    fn each_half_is_a_bar_with_its_interval() {
        let t = Table::parse(
            "placed,answers,recall,lo,hi\ntrue,1025,0.5,0.48,0.54\nfalse,895,0.12,0.12,0.18\n",
        )
        .unwrap();
        let h = halves(&t).unwrap();
        assert_eq!(
            h[1],
            Half {
                placed: false,
                answers: 895,
                recall: 0.12,
                lo: 0.12,
                hi: 0.18,
            }
        );
        let out = svg(&h);
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        for want in [
            "<svg viewBox=\"0 0 640 188\" role=\"img\" aria-label=\"Mean recall of the answers by whether their projection held an answer file for the first answer token, with 95% bootstrap intervals: answer file placed (1,025 answers): 50.0% (48.0% to 54.0%); not placed (895 answers): 12.0% (12.0% to 18.0%).\">",
            "<rect class=\"us\" x=\"250\" y=\"75\" width=\"291.7\" height=\"22\"/>",
            "<path class=\"grid base\" d=\"M530 80 V92 M530 86 H565 M565 80 V92\"/>",
            "<text class=\"v-us\" x=\"573\" y=\"90\">50.0%</text>",
            "<rect class=\"us-soft\" x=\"250\" y=\"121\" width=\"70\" height=\"22\"/>",
            "<text class=\"cat-sub\" x=\"16\" y=\"144\">895 answers</text>",
            "<text class=\"tick mid\" x=\"600\" y=\"176\">60%</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{out}");
        }
    }
}
