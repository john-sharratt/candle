//! Figure 9.6-2, the format mix: the share of the resident tokens each K/V
//! format held over the battery, stacked to 100%.

use crate::figures::samples::{csv, Samples, HOUR, MIX, TIME};
use crate::figures::svg::{Linear, Svg, Swatch};
use crate::figures::text::{pct, pct0, ticks};
use crate::figures::timeline::{stack, time_axis, Drawn, HEIGHT, X0, X1, Y0, Y1};

/// The formats' classes in census order, alternating hue and the post's own
/// colour so neighbouring layers stay apart.
const CLASSES: [&str; 7] = ["q4", "q3", "b0", "q1", "b2", "q2", "b3"];

/// Each point's share per format, from its tokens per format.
fn shares(s: &Samples, formats: &[String]) -> anyhow::Result<Vec<Vec<f64>>> {
    let tokens: Vec<Vec<f64>> = formats
        .iter()
        .map(|f| s.series(&format!("{MIX}{f}")))
        .collect::<anyhow::Result<_>>()?;
    let totals: Vec<f64> = (0..s.rows.len())
        .map(|i| tokens.iter().map(|t| t[i]).sum())
        .collect();
    Ok(tokens
        .iter()
        .map(|t| {
            t.iter()
                .zip(&totals)
                .map(|(v, total)| if *total > 0.0 { v / total } else { 0.0 })
                .collect()
        })
        .collect())
}

/// The figure from `points`, the downsampled samples, described from `raw`,
/// every loaded sample: each format's share of all the tokens resident over
/// the run.
pub fn figure(points: &Samples, raw: &Samples) -> anyhow::Result<Drawn> {
    let formats = raw.under(MIX);
    let hours: Vec<f64> = points.series(TIME)?.iter().map(|t| t / HOUR).collect();
    let layers = shares(points, &formats)?;
    let raw_tokens: Vec<f64> = formats
        .iter()
        .map(|f| Ok(raw.series(&format!("{MIX}{f}"))?.iter().sum()))
        .collect::<anyhow::Result<_>>()?;
    let all: f64 = raw_tokens.iter().sum();
    let described: Vec<String> = formats
        .iter()
        .zip(&raw_tokens)
        .map(|(f, t)| format!("{f} {}", pct(if all > 0.0 { t / all } else { 0.0 })))
        .collect();
    let aria = format!(
        "The share of the resident K/V tokens each format held over the battery, stacked to 100%; \
         over the whole run: {}.",
        described.join(", ")
    );
    let mut svg = Svg::new(
        HEIGHT,
        &aria,
        "The format mix",
        "share of the resident tokens held in each K/V format, per 32-token block",
    );
    let y = Linear::new((0.0, 1.0), (Y0, Y1));
    svg.y_grid(&y, &ticks(0.0, 1.0, 0.25, pct0), (X0, X1), 0.0);
    let x = time_axis(&mut svg, &hours);
    let xs: Vec<f64> = hours.iter().map(|h| x.at(*h)).collect();
    let classed: Vec<(&str, Vec<f64>)> = layers
        .iter()
        .enumerate()
        .map(|(i, l)| (CLASSES[i % CLASSES.len()], l.clone()))
        .collect();
    stack(&mut svg, &xs, &y, &classed);
    let items: Vec<(Swatch, &str)> = formats
        .iter()
        .enumerate()
        .map(|(i, f)| (Swatch::Fill(CLASSES[i % CLASSES.len()]), f.as_str()))
        .collect();
    svg.legend(X0, 58.0, &items);
    let mut header = vec!["hours".to_string()];
    header.extend(formats.iter().map(|f| format!("{f}_share")));
    let header: Vec<&str> = header.iter().map(String::as_str).collect();
    let rows: Vec<Vec<f64>> = (0..hours.len())
        .map(|i| {
            let mut r = vec![hours[i]];
            r.extend(layers.iter().map(|l| l[i]));
            r
        })
        .collect();
    Ok(Drawn {
        svg: svg.finish(),
        csv: csv(&header, &rows),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Samples {
        Samples {
            names: vec![
                "t_s".into(),
                "in_flight".into(),
                format!("{MIX}Q4_0"),
                format!("{MIX}Q8_0"),
            ],
            rows: vec![
                vec![0.0, 2.0, 300.0, 100.0],
                vec![3600.0, 4.0, 100.0, 300.0],
            ],
        }
    }

    /// A quarter of the tokens at hour 0 are Q8_0, three quarters at hour 1:
    /// Q4_0's top falls from 75% (124) to 25% (228).
    #[test]
    fn each_format_is_its_share_stacked_to_one_hundred_percent() {
        let s = samples();
        let d = figure(&s, &s).unwrap();
        let lines: Vec<&str> = d.svg.lines().map(str::trim).collect();
        for want in [
            "<path class=\"q4\" d=\"M56 124 584 228 584 280 56 280 Z\"/>",
            "<path class=\"q3\" d=\"M56 72 584 72 584 228 56 124 Z\"/>",
            "<text x=\"74\" y=\"58\">Q4_0</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{}", d.svg);
        }
        assert!(d
            .svg
            .contains("over the whole run: Q4_0 50.0%, Q8_0 50.0%."));
        assert_eq!(
            d.csv,
            "hours,Q4_0_share,Q8_0_share\n0,0.75,0.25\n1,0.25,0.75\n"
        );
    }
}
