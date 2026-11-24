//! Figure 9.6-1, the card by tenant: the GiB each of the span's tenants held
//! over the battery, stacked, with the answers in flight over them on a scale
//! of their own at the right.

use crate::figures::samples::{csv, mean_peak, Samples, GIB_MIB, HOUR, IN_FLIGHT, TIME, VRAM};
use crate::figures::svg::Precision::Whole;
use crate::figures::svg::{line_path, Anchor, Linear, Svg, Swatch};
use crate::figures::text::{ceil_to, n, ticks};
use crate::figures::timeline::{stack, time_axis, Drawn, HEIGHT, X0, X1, Y0, Y1};

/// The tenants stacked, bottom up: column, class, legend label.
pub const TENANTS: [(&str, &str, &str); 5] = [
    ("dense", "q4", "dense"),
    ("experts", "q1", "experts"),
    ("kv", "b0", "KV"),
    ("transient", "q2", "activations"),
    ("state", "b3", "state"),
];

fn gib(s: &Samples, tenant: &str) -> anyhow::Result<Vec<f64>> {
    Ok(s.series(&format!("{VRAM}{tenant}"))?
        .into_iter()
        .map(|m| m / GIB_MIB)
        .collect())
}

/// The figure from `points`, the downsampled samples, described from `raw`,
/// every loaded sample.
pub fn figure(points: &Samples, raw: &Samples) -> anyhow::Result<Drawn> {
    let hours: Vec<f64> = points.series(TIME)?.iter().map(|t| t / HOUR).collect();
    let flight = points.series(IN_FLIGHT)?;
    let layers: Vec<(&str, Vec<f64>)> = TENANTS
        .iter()
        .map(|(col, class, _)| Ok((*class, gib(points, col)?)))
        .collect::<anyhow::Result<_>>()?;
    let mut described = Vec::new();
    for (col, _, label) in TENANTS {
        let (m, p) = mean_peak(&gib(raw, col)?);
        described.push(format!("{label} {m:.1} GiB on average, {p:.1} at peak"));
    }
    let (fm, fp) = mean_peak(&raw.series(IN_FLIGHT)?);
    let span = raw.series(TIME)?.iter().copied().fold(0.0, f64::max) / HOUR;
    let aria = format!(
        "The card's memory by tenant over the battery's {span:.1} hours, stacked, in GiB: {}; with \
         {fm:.1} answers in flight on average and {fp:.0} at most.",
        described.join(", ")
    );
    let mut svg = Svg::new(
        HEIGHT,
        &aria,
        "The card under the battery",
        "GiB each tenant of the span held, stacked · answers in flight on the right scale",
    );
    let totals: Vec<f64> = (0..hours.len())
        .map(|i| layers.iter().map(|(_, v)| v[i]).sum())
        .collect();
    let peak = totals.iter().copied().fold(1.0, f64::max);
    let step = if peak > 80.0 { 20.0 } else { 10.0 };
    let top = ceil_to(peak, step);
    let y = Linear::new((0.0, top), (Y0, Y1));
    svg.y_grid(&y, &ticks(0.0, top, step, n), (X0, X1), 0.0);
    let x = time_axis(&mut svg, &hours);
    let xs: Vec<f64> = hours.iter().map(|h| x.at(*h)).collect();
    stack(&mut svg, &xs, &y, &layers);
    let flight_top = ceil_to(flight.iter().copied().fold(1.0, f64::max), 4.0);
    let y2 = Linear::new((0.0, flight_top), (Y0, Y1));
    for (v, label) in ticks(0.0, flight_top, flight_top / 4.0, n) {
        svg.text(
            "tick t-them",
            X1 + 8.0,
            y2.at(v) + 3.5,
            Anchor::Start,
            &label,
        );
    }
    let line: Vec<(f64, f64)> = xs
        .iter()
        .zip(&flight)
        .map(|(x, f)| (*x, y2.at(*f)))
        .collect();
    svg.path("line-them", &line_path(&line, Whole), None);
    svg.text("tick", X0 - 8.0, Y1 - 12.0, Anchor::End, "GiB");
    svg.text(
        "tick t-them",
        X1 + 8.0,
        Y1 - 12.0,
        Anchor::Start,
        "in flight",
    );
    let mut items: Vec<(Swatch, &str)> = TENANTS
        .iter()
        .map(|(_, class, label)| (Swatch::Fill(class), *label))
        .collect();
    items.push((Swatch::Line("line-them"), "answers in flight"));
    svg.legend(X0, 58.0, &items);
    let rows: Vec<Vec<f64>> = (0..hours.len())
        .map(|i| {
            let mut r = vec![hours[i], flight[i]];
            r.extend(layers.iter().map(|(_, v)| v[i]));
            r
        })
        .collect();
    Ok(Drawn {
        svg: svg.finish(),
        csv: csv(
            &[
                "hours",
                "in_flight",
                "dense_gib",
                "experts_gib",
                "kv_gib",
                "transient_gib",
                "state_gib",
            ],
            &rows,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples() -> Samples {
        let names = [
            "t_s",
            "in_flight",
            "dense",
            "experts",
            "kv",
            "transient",
            "state",
        ];
        Samples {
            names: names
                .iter()
                .map(|n| {
                    if n.starts_with("t_") || n.starts_with("in_") {
                        n.to_string()
                    } else {
                        format!("{VRAM}{n}")
                    }
                })
                .collect(),
            rows: vec![
                vec![0.0, 4.0, 10240.0, 20480.0, 5120.0, 1024.0, 2048.0],
                vec![7200.0, 16.0, 10240.0, 30720.0, 10240.0, 1024.0, 2048.0],
            ],
        }
    }

    /// Two hours, peaking at 10 + 30 + 10 + 1 + 2 = 53 GiB on an axis to 60:
    /// the experts' top at hour 2 sits at 280 − 40/60·208 = 141.3.
    #[test]
    fn the_tenants_stack_in_gib_under_the_answers_in_flight() {
        let s = samples();
        let d = figure(&s, &s).unwrap();
        let lines: Vec<&str> = d.svg.lines().map(str::trim).collect();
        for want in [
            "<path class=\"q4\" d=\"M56 245 584 245 584 280 56 280 Z\"/>",
            "<text class=\"tick mid\" x=\"122\" y=\"296\">0.25</text>",
            "<path class=\"q1\" d=\"M56 176 584 141 584 245 56 245 Z\"/>",
            "<path class=\"line-them\" d=\"M56 228 584 72\"/>",
            "<text class=\"tick t-them\" x=\"592\" y=\"75.5\">16</text>",
            "<text class=\"tick\" x=\"48\" y=\"75.5\" text-anchor=\"end\">60</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{}", d.svg);
        }
        assert!(d.svg.contains("experts 25.0 GiB on average, 30.0 at peak"));
        assert!(d
            .svg
            .contains("with 10.0 answers in flight on average and 16 at most."));
        assert_eq!(
            d.csv,
            "hours,in_flight,dense_gib,experts_gib,kv_gib,transient_gib,state_gib\n\
             0,4,10,20,5,1,2\n2,16,10,30,10,1,2\n"
        );
    }
}
