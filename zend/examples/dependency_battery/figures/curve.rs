//! Figure 9.12-5, the retrieval curve: for each arm, the share of answers
//! whose projection held an answer file at each token of the answer, with
//! its 95% bootstrap band.

use crate::figures::by_arm::dials;
use crate::figures::svg::Precision::Tenth;
use crate::figures::svg::{area_path, line_path, Anchor, Linear, Svg, Swatch};
use crate::figures::table::Table;
use crate::figures::text::{ceil_to, floor_to, pct, pct0, thousands, ticks};

/// One arm's curve: `(token, share, lo, hi)` per sampled token.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve {
    pub arm: String,
    pub points: Vec<(f64, f64, f64, f64)>,
}

/// The curves of `fig_9_12_5_retrieval_curve.csv`, arms in file order.
pub fn curves(t: &Table) -> anyhow::Result<Vec<Curve>> {
    let mut out: Vec<Curve> = Vec::new();
    for r in t.records() {
        let arm = r.get("arm")?;
        let point = (r.num("token")?, r.num("share")?, r.num("lo")?, r.num("hi")?);
        match out.last_mut() {
            Some(c) if c.arm == arm => c.points.push(point),
            _ => out.push(Curve {
                arm: arm.to_string(),
                points: vec![point],
            }),
        }
    }
    Ok(out)
}

/// An arm's line class and its band's class and opacity.
fn style(arm: &str) -> (&'static str, Option<(&'static str, Option<f64>)>) {
    match arm.chars().next() {
        Some('C') => ("line-us", Some(("area-us", None))),
        Some('B') => ("line-them", Some(("them", Some(0.12)))),
        _ => ("line-them dash", None),
    }
}

const X0: f64 = 62.0;
const X1: f64 = 616.0;
const Y0: f64 = 280.0;
const Y1: f64 = 70.0;

pub fn svg(curves: &[Curve]) -> String {
    let all = curves.iter().flat_map(|c| c.points.iter());
    let last = all.clone().map(|p| p.0).fold(0.0, f64::max);
    let lo = floor_to(all.clone().map(|p| p.2).fold(1.0, f64::min), 0.1);
    let hi = ceil_to(all.map(|p| p.3).fold(0.0, f64::max), 0.1);
    let described: Vec<String> = curves
        .iter()
        .filter_map(|c| {
            let (first, end) = (c.points.first()?, c.points.last()?);
            Some(format!(
                "{} from {} at the first token to {} at token {}",
                c.arm,
                pct(first.1),
                pct(end.1),
                thousands(end.0 as u64)
            ))
        })
        .collect();
    let aria = format!(
        "The share of answers whose projection held an answer file, by token position in the answer, \
         with 95% bootstrap bands: {}.",
        described.join("; ")
    );
    let mut svg = Svg::new(
        324.0,
        &aria,
        "When the answer file is in view",
        "answers with an answer file placed, by token of the answer · band: 95% bootstrap interval",
    );
    let x = Linear::new((0.0, last.max(1.0)), (X0, X1));
    let y = Linear::new((lo, hi), (Y0, Y1));
    svg.y_grid(&y, &ticks(lo, hi, 0.1, pct0), (X0, X1), lo);
    let columns: Vec<(f64, String)> = ticks(0.0, last, (last / 4.0).max(1.0), |t| {
        thousands(t.round() as u64)
    })
    .into_iter()
    .map(|(t, l)| (x.at(t), l))
    .collect();
    svg.x_grid(&columns, (Y1, Y0), Y0 + 16.0);
    svg.text(
        "t-dim",
        (X0 + X1) / 2.0,
        Y0 + 34.0,
        Anchor::Middle,
        "token position in the answer",
    );
    for c in curves {
        if let (_, Some((band, opacity))) = style(&c.arm) {
            let top: Vec<(f64, f64)> = c.points.iter().map(|p| (x.at(p.0), y.at(p.3))).collect();
            let bottom: Vec<(f64, f64)> = c.points.iter().map(|p| (x.at(p.0), y.at(p.2))).collect();
            svg.path(band, &area_path(&top, &bottom, Tenth), opacity);
        }
    }
    for c in curves.iter().rev() {
        let line: Vec<(f64, f64)> = c.points.iter().map(|p| (x.at(p.0), y.at(p.1))).collect();
        svg.path(style(&c.arm).0, &line_path(&line, Tenth), None);
    }
    let labels: Vec<String> = curves
        .iter()
        .map(|c| format!("{} {}", c.arm, dials(&c.arm).0))
        .collect();
    let items: Vec<(Swatch, &str)> = curves
        .iter()
        .zip(&labels)
        .map(|(c, l)| (Swatch::Line(style(&c.arm).0), l.as_str()))
        .collect();
    svg.legend(330.0, 56.0, &items);
    svg.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Table {
        Table::parse(
            "arm,token,share,lo,hi\n\
             C1,0,0.5,0.46,0.54\n\
             C1,64,0.6,0.56,0.64\n\
             D1,0,0.4,0.36,0.44\n\
             D1,64,0.5,0.46,0.54\n",
        )
        .unwrap()
    }

    #[test]
    fn rows_group_into_one_curve_per_arm_in_file_order() {
        let c = curves(&table()).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].arm, "C1");
        assert_eq!(
            c[1].points,
            vec![(0.0, 0.4, 0.36, 0.44), (64.0, 0.5, 0.46, 0.54)]
        );
    }

    /// The axis runs 30% to 70% over 210 px; C1's 50% at token 0 sits at
    /// 280 − 0.2/0.4·210 = 175, its 60% at token 64 at 122.5.
    #[test]
    fn each_arm_is_its_band_and_its_line() {
        let out = svg(&curves(&table()).unwrap());
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        for want in [
            "<path class=\"area-us\" d=\"M62 154 616 101.5 616 143.5 62 196 Z\"/>",
            "<path class=\"line-them dash\" d=\"M62 227.5 616 175\"/>",
            "<path class=\"line-us\" d=\"M62 175 616 122.5\"/>",
            "<text class=\"tick mid\" x=\"616\" y=\"296\">64</text>",
            "<text class=\"tick\" x=\"54\" y=\"283.5\" text-anchor=\"end\">30%</text>",
            "<text x=\"348\" y=\"56\">C1 reprojecting</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{out}");
        }
        assert!(
            out.contains("C1 from 50.0% at the first token to 60.0% at token 64; D1 from 40.0%")
        );
        assert!(!out.contains("class=\"them\""));
    }
}
