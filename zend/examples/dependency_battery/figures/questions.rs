//! Figure 9.12-3, question by question: each question's mean recall under
//! reprojection (C1) against the held projection (B1), with the diagonal of
//! no difference and the count on each side of it.

use crate::figures::svg::{Anchor, Linear, Svg};
use crate::figures::table::Table;
use crate::figures::text::{n, pct0, ticks};

/// Each question's `(b1, c1)` from `fig_9_12_3_questions.csv`.
pub fn points(t: &Table) -> anyhow::Result<Vec<(f64, f64)>> {
    t.records()
        .map(|r| Ok((r.num("b1")?, r.num("c1")?)))
        .collect()
}

/// The questions above the diagonal (C1 > B1), below it, and on it, compared
/// exactly as the analysis's summary counts them.
pub fn sides(points: &[(f64, f64)]) -> (usize, usize, usize) {
    let above = points.iter().filter(|(b, c)| c > b).count();
    let below = points.iter().filter(|(b, c)| c < b).count();
    (above, below, points.len() - above - below)
}

const X0: f64 = 70.0;
const X1: f64 = 350.0;
const Y0: f64 = 340.0;
const Y1: f64 = 60.0;

pub fn svg(points: &[(f64, f64)]) -> String {
    let (above, below, on) = sides(points);
    let total = points.len();
    let aria = format!(
        "Each of the {total} questions' mean recall under reprojection during the reasoning (C1) \
         against the held projection (B1): {above} questions lie above the diagonal, where C1 recalls \
         more, {below} below it, where B1 does, and {on} on it."
    );
    let mut svg = Svg::new(
        384.0,
        &aria,
        "Question by question",
        &format!(
            "each question's mean recall over its runs · {above} above the diagonal, {below} below, {on} on it"
        ),
    );
    let x = Linear::new((0.0, 1.0), (X0, X1));
    let y = Linear::new((0.0, 1.0), (Y0, Y1));
    let grid = ticks(0.0, 1.0, 0.25, pct0);
    svg.y_grid(&y, &grid, (X0, X1), 0.0);
    let columns: Vec<(f64, String)> = grid.iter().map(|(v, l)| (x.at(*v), l.clone())).collect();
    svg.x_grid(&columns, (Y1, Y0), Y0 + 16.0);
    svg.path(
        "parity",
        &format!("M{} {} L{} {}", n(X0), n(Y0), n(X1), n(Y1)),
        None,
    );
    for (b, c) in points {
        svg.circle("us", x.at(*b), y.at(*c), 4.0, Some(0.3));
    }
    svg.text("t-dim", X0 + 12.0, Y1 + 16.0, Anchor::Start, "C1 higher");
    svg.text("t-dim", X1 - 12.0, Y0 - 10.0, Anchor::End, "B1 higher");
    svg.text(
        "t-dim",
        (X0 + X1) / 2.0,
        Y0 + 32.0,
        Anchor::Middle,
        "B1 recall · projection held",
    );
    svg.text(
        "t-dim",
        X0,
        Y1 - 8.0,
        Anchor::Start,
        "C1 recall · reprojecting",
    );
    for (i, (count, value, what)) in [
        (above, "v-us", "above the diagonal: C1 recalls more"),
        (below, "v-them", "below it: B1 recalls more"),
        (on, "v-them", "on it: the same recall"),
    ]
    .into_iter()
    .enumerate()
    {
        let top = 110.0 + i as f64 * 56.0;
        svg.text(value, 396.0, top, Anchor::Start, &count.to_string());
        svg.text("cat-sub", 396.0, top + 16.0, Anchor::Start, what);
    }
    svg.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sides_of_the_diagonal_are_counted_exactly() {
        let t = Table::parse("id,category,b1,c1\na,x,0.5,1\nb,x,1,0.5\nc,x,0.25,0.25\nd,x,0,0.5\n")
            .unwrap();
        let p = points(&t).unwrap();
        assert_eq!(sides(&p), (2, 1, 1));
        let out = svg(&p);
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        for want in [
            "<path class=\"parity\" d=\"M70 340 L350 60\"/>",
            "<circle class=\"us\" opacity=\"0.3\" cx=\"210\" cy=\"60\" r=\"4\"/>",
            "<circle class=\"us\" opacity=\"0.3\" cx=\"140\" cy=\"270\" r=\"4\"/>",
            "<text class=\"tick mid\" x=\"280\" y=\"356\">75%</text>",
            "<text class=\"v-us\" x=\"396\" y=\"110\">2</text>",
            "<text class=\"v-them\" x=\"396\" y=\"222\">1</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{out}");
        }
        assert!(out.contains("2 above the diagonal, 1 below, 1 on it"));
    }
}
