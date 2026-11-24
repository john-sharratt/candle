//! A forest plot: one row per effect, its point on its interval against a
//! zero line, labelled at the left. Figures 9.12-2 and 9.12-4 are drawn as
//! one. Effects are differences of shares, printed as points.

use crate::figures::svg::{Anchor, Linear, Svg};
use crate::figures::text::{ceil_to, floor_to, n, signed, ticks};

/// One effect and its 95% interval, as shares.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub label: String,
    pub sub: String,
    pub effect: f64,
    pub lo: f64,
    pub hi: f64,
    /// Drawn at full strength; a row that is not is faded.
    pub strong: bool,
}

/// A forest plot's title, subtitle, description and tick step (a share).
pub struct Frame<'a> {
    pub title: &'a str,
    pub subtitle: &'a str,
    pub aria: &'a str,
    pub step: f64,
}

const X0: f64 = 372.0;
const X1: f64 = 604.0;
const TOP: f64 = 70.0;
const ROW: f64 = 40.0;

/// A share's difference as a whole-point tick label.
fn points(v: f64) -> String {
    let p = (v * 100.0).round();
    if p == 0.0 {
        "0".into()
    } else {
        signed(p / 100.0).trim_end_matches(".0").to_string()
    }
}

pub fn svg(frame: &Frame, rows: &[Row]) -> String {
    let bottom = TOP + rows.len() as f64 * ROW;
    let mut svg = Svg::new(bottom + 24.0, frame.aria, frame.title, frame.subtitle);
    let lo = floor_to(rows.iter().map(|r| r.lo).fold(0.0, f64::min), frame.step);
    let hi = ceil_to(rows.iter().map(|r| r.hi).fold(0.0, f64::max), frame.step);
    let x = Linear::new((lo, hi), (X0, X1));
    let at: Vec<(f64, String)> = ticks(lo, hi, frame.step, points)
        .into_iter()
        .map(|(v, label)| (x.at(v), label))
        .collect();
    svg.x_grid(&at, (TOP - 4.0, bottom), bottom + 14.0);
    svg.path(
        "parity",
        &format!("M{} {} V{}", n(x.at(0.0)), n(TOP - 4.0), n(bottom)),
        None,
    );
    for (i, r) in rows.iter().enumerate() {
        let cy = TOP + i as f64 * ROW + 14.0;
        let (opacity, value) = if r.strong {
            (None, "v-us")
        } else {
            (Some(0.45), "v-them")
        };
        svg.text("cat", 16.0, cy - 2.0, Anchor::Start, &r.label);
        svg.text("cat-sub", 16.0, cy + 12.0, Anchor::Start, &r.sub);
        svg.path(
            "line-us",
            &format!("M{} {} H{}", n(x.at(r.lo)), n(cy), n(x.at(r.hi))),
            opacity,
        );
        svg.circle("dot-us", x.at(r.effect), cy, 6.0, opacity);
        svg.text(
            value,
            x.at(r.effect),
            cy - 10.0,
            Anchor::Middle,
            &signed(r.effect),
        );
    }
    svg.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The axis runs −10 to +20 points over 232 px, so zero sits at
    /// 372 + 232/3 = 449.3 and +10 at 526.7.
    #[test]
    fn each_row_is_its_interval_and_its_point_against_zero() {
        let rows = [
            Row {
                label: "Strong".into(),
                sub: "C1 − B1".into(),
                effect: 0.1,
                lo: 0.05,
                hi: 0.15,
                strong: true,
            },
            Row {
                label: "Weak".into(),
                sub: "C1 − C0".into(),
                effect: 0.0,
                lo: -0.03,
                hi: 0.03,
                strong: false,
            },
        ];
        let frame = Frame {
            title: "T",
            subtitle: "S",
            aria: "A",
            step: 0.1,
        };
        let out = svg(&frame, &rows);
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        for want in [
            "<svg viewBox=\"0 0 640 174\" role=\"img\" aria-label=\"A\">",
            "<path class=\"grid\" d=\"M372 66 V150 M449.3 66 V150 M526.7 66 V150 M604 66 V150\"/>",
            "<text class=\"tick mid\" x=\"372\" y=\"164\">\u{2212}10</text>",
            "<text class=\"tick mid\" x=\"604\" y=\"164\">+20</text>",
            "<path class=\"parity\" d=\"M449.3 66 V150\"/>",
            "<text class=\"cat\" x=\"16\" y=\"82\">Strong</text>",
            "<path class=\"line-us\" d=\"M488 84 H565.3\"/>",
            "<circle class=\"dot-us\" cx=\"526.7\" cy=\"84\" r=\"6\"/>",
            "<text class=\"v-us mid\" x=\"526.7\" y=\"74\">+10.0</text>",
            "<path class=\"line-us\" opacity=\"0.45\" d=\"M426.1 124 H472.5\"/>",
            "<circle class=\"dot-us\" opacity=\"0.45\" cx=\"449.3\" cy=\"124\" r=\"6\"/>",
            "<text class=\"v-them mid\" x=\"449.3\" y=\"114\">+0.0</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{out}");
        }
    }
}
