//! What Figures 9.6-1 to 9.6-3 share: a plot over the battery's hours, its
//! time axis, and layers stacked bottom up over it.

use crate::figures::svg::Precision::Whole;
use crate::figures::svg::{area_path, Anchor, Linear, Svg};
use crate::figures::text::{ceil_to, ticks};

pub const X0: f64 = 56.0;
pub const X1: f64 = 584.0;
pub const Y0: f64 = 280.0;
pub const Y1: f64 = 72.0;
pub const HEIGHT: f64 = 324.0;

/// A figure over time: its snippet, and the points it plots as a CSV.
#[derive(Debug, Clone, PartialEq)]
pub struct Drawn {
    pub svg: String,
    pub csv: String,
}

/// A tick step for an axis `span` hours long: the first of the usual steps
/// that leaves at most eight intervals.
pub fn hour_step(span: f64) -> f64 {
    [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 12.0, 24.0]
        .into_iter()
        .find(|s| span / s <= 8.0)
        .unwrap_or(48.0)
}

/// An hour tick's label: to two decimals, trailing zeros dropped.
pub fn hours_label(h: f64) -> String {
    let s = format!("{h:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// The time axis over `hours` (each point's hour), drawn under the plot;
/// returns its scale.
pub fn time_axis(svg: &mut Svg, hours: &[f64]) -> Linear {
    let last = hours.iter().copied().fold(0.0, f64::max);
    let step = hour_step(last);
    let end = ceil_to(last.max(step), step);
    let x = Linear::new((0.0, end), (X0, X1));
    let columns: Vec<(f64, String)> = ticks(0.0, end, step, hours_label)
        .into_iter()
        .map(|(h, l)| (x.at(h), l))
        .collect();
    svg.x_grid(&columns, (Y1, Y0), Y0 + 16.0);
    svg.text(
        "t-dim",
        (X0 + X1) / 2.0,
        Y0 + 34.0,
        Anchor::Middle,
        "hours into the battery",
    );
    x
}

/// `layers`, each a class and a value per point, stacked bottom up over the
/// points' `xs`.
pub fn stack(svg: &mut Svg, xs: &[f64], y: &Linear, layers: &[(&str, Vec<f64>)]) {
    let mut base = vec![0.0; xs.len()];
    for (class, values) in layers {
        let top: Vec<f64> = base.iter().zip(values).map(|(b, v)| b + v).collect();
        let upper: Vec<(f64, f64)> = xs.iter().zip(&top).map(|(x, v)| (*x, y.at(*v))).collect();
        let lower: Vec<(f64, f64)> = xs.iter().zip(&base).map(|(x, v)| (*x, y.at(*v))).collect();
        svg.path(class, &area_path(&upper, &lower, Whole), None);
        base = top;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hour_step_leaves_at_most_eight_intervals() {
        assert_eq!(hour_step(1.5), 0.25);
        assert_eq!(hour_step(7.0), 1.0);
        assert_eq!(hour_step(9.3), 2.0);
        assert_eq!(hour_step(30.0), 4.0);
        assert_eq!(
            [0.0, 0.25, 0.5, 2.0, 10.0].map(hours_label),
            ["0", "0.25", "0.5", "2", "10"].map(String::from)
        );
    }

    #[test]
    fn layers_stack_on_the_ones_below() {
        let mut svg = Svg::new(100.0, "", "", "");
        let y = Linear::new((0.0, 10.0), (100.0, 0.0));
        stack(
            &mut svg,
            &[0.0, 50.0],
            &y,
            &[("q1", vec![2.0, 4.0]), ("b0", vec![1.0, 1.0])],
        );
        let x = time_axis(&mut svg, &[0.0, 9.3]);
        assert_eq!(x.at(10.0), X1);
        let out = svg.finish();
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        for want in [
            "<path class=\"q1\" d=\"M0 80 50 60 50 100 0 100 Z\"/>",
            "<path class=\"b0\" d=\"M0 70 50 50 50 60 0 80 Z\"/>",
            "<text class=\"tick mid\" x=\"584\" y=\"296\">10</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{out}");
        }
    }
}
