//! Figure 9.6-3, the context against BF16: over the battery, the size the
//! resident tokens would take at BF16 (their count times the BF16 bytes a
//! token takes over every K/V layer), the KV arenas the span actually held for
//! them, and the KV ground the span had — held arenas and free regions both.

use crate::figures::samples::{csv, mean_peak, Samples, GIB_MIB, HOUR, MIX, TIME, VRAM};
use crate::figures::svg::Precision::Whole;
use crate::figures::svg::{area_path, line_path, Linear, Svg, Swatch};
use crate::figures::text::{ceil_to, n, ticks};
use crate::figures::timeline::{time_axis, Drawn, HEIGHT, X0, X1, Y0, Y1};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Per point: the tokens resident, their BF16 GiB, the KV arenas held and the
/// KV ground, GiB.
fn series(s: &Samples, bf16_per_token: f64) -> anyhow::Result<[Vec<f64>; 4]> {
    let formats = s.under(MIX);
    let mut tokens = vec![0.0; s.rows.len()];
    for f in &formats {
        for (t, v) in tokens.iter_mut().zip(s.series(&format!("{MIX}{f}"))?) {
            *t += v;
        }
    }
    let bf16: Vec<f64> = tokens.iter().map(|t| t * bf16_per_token / GIB).collect();
    let kv: Vec<f64> = s
        .series(&format!("{VRAM}kv"))?
        .iter()
        .map(|m| m / GIB_MIB)
        .collect();
    let ground: Vec<f64> = kv
        .iter()
        .zip(s.series(&format!("{VRAM}kv_free"))?)
        .map(|(k, f)| k + f / GIB_MIB)
        .collect();
    Ok([tokens, bf16, kv, ground])
}

/// The figure from `points`, the downsampled samples, described from `raw`,
/// every loaded sample, a token taking `bf16_per_token` bytes at BF16.
pub fn figure(points: &Samples, raw: &Samples, bf16_per_token: f64) -> anyhow::Result<Drawn> {
    let hours: Vec<f64> = points.series(TIME)?.iter().map(|t| t / HOUR).collect();
    let [tokens, bf16, kv, ground] = series(points, bf16_per_token)?;
    let [_, raw_bf16, raw_kv, raw_ground] = series(raw, bf16_per_token)?;
    let (bm, bp) = mean_peak(&raw_bf16);
    let (km, kp) = mean_peak(&raw_kv);
    let (gm, _) = mean_peak(&raw_ground);
    let ratio: Vec<f64> = raw_bf16
        .iter()
        .zip(&raw_kv)
        .filter(|(_, k)| **k > 0.0)
        .map(|(b, k)| b / k)
        .collect();
    let (rm, _) = mean_peak(&ratio);
    let aria = format!(
        "Over the battery, the resident tokens would take {bm:.1} GiB at BF16 on average and {bp:.1} \
         at peak; the KV arenas the span held for them took {km:.1} GiB on average and {kp:.1} at \
         peak, {rm:.2} times smaller, out of {gm:.1} GiB of KV ground on average."
    );
    let mut svg = Svg::new(
        HEIGHT,
        &aria,
        "The context against BF16",
        &format!(
            "GiB · resident tokens at BF16, {} B each, against the KV arenas that hold them",
            n(bf16_per_token)
        ),
    );
    let peak = bf16
        .iter()
        .chain(&kv)
        .chain(&ground)
        .copied()
        .fold(1.0, f64::max);
    let step = if peak > 40.0 { 10.0 } else { 5.0 };
    let top = ceil_to(peak, step);
    let y = Linear::new((0.0, top), (Y0, Y1));
    svg.y_grid(&y, &ticks(0.0, top, step, n), (X0, X1), 0.0);
    let x = time_axis(&mut svg, &hours);
    let at = |v: &[f64]| -> Vec<(f64, f64)> {
        hours
            .iter()
            .zip(v)
            .map(|(h, v)| (x.at(*h), y.at(*v)))
            .collect()
    };
    let floor: Vec<(f64, f64)> = hours.iter().map(|h| (x.at(*h), y.at(0.0))).collect();
    svg.path("area-us", &area_path(&at(&kv), &floor, Whole), None);
    svg.path("line-us dash", &line_path(&at(&ground), Whole), None);
    svg.path("line-them", &line_path(&at(&bf16), Whole), None);
    svg.path("line-us", &line_path(&at(&kv), Whole), None);
    svg.legend(
        X0,
        58.0,
        &[
            (Swatch::Line("line-them"), "tokens at BF16"),
            (Swatch::Line("line-us"), "KV arenas held"),
            (Swatch::Line("line-us dash"), "KV ground"),
        ],
    );
    let rows: Vec<Vec<f64>> = (0..hours.len())
        .map(|i| vec![hours[i], tokens[i], bf16[i], kv[i], ground[i]])
        .collect();
    Ok(Drawn {
        svg: svg.finish(),
        csv: csv(
            &["hours", "tokens", "bf16_gib", "kv_gib", "kv_ground_gib"],
            &rows,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2²⁰ tokens at 2 KiB each are 2 GiB at BF16, held in 1 GiB of arenas
    /// out of 3 GiB of ground.
    #[test]
    fn the_tokens_at_bf16_stand_beside_the_arenas_that_hold_them() {
        let s = Samples {
            names: vec![
                "t_s".into(),
                "in_flight".into(),
                format!("{MIX}Q4_0"),
                format!("{MIX}Q8_0"),
                format!("{VRAM}kv"),
                format!("{VRAM}kv_free"),
            ],
            rows: vec![
                vec![0.0, 2.0, 524_288.0, 524_288.0, 1024.0, 2048.0],
                vec![3600.0, 4.0, 1_048_576.0, 1_048_576.0, 2048.0, 2048.0],
            ],
        };
        let d = figure(&s, &s, 2048.0).unwrap();
        let lines: Vec<&str> = d.svg.lines().map(str::trim).collect();
        for want in [
            "<path class=\"line-them\" d=\"M56 197 584 114\"/>",
            "<path class=\"line-us\" d=\"M56 238 584 197\"/>",
            "<path class=\"line-us dash\" d=\"M56 155 584 114\"/>",
            "<path class=\"area-us\" d=\"M56 238 584 197 584 280 56 280 Z\"/>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{}", d.svg);
        }
        assert!(d.svg.contains(
            "would take 3.0 GiB at BF16 on average and 4.0 at peak; the KV arenas the span held for \
             them took 1.5 GiB on average and 2.0 at peak, 2.00 times smaller, out of 3.5 GiB"
        ));
        assert_eq!(
            d.csv,
            "hours,tokens,bf16_gib,kv_gib,kv_ground_gib\n0,1048576,2,1,3\n1,2097152,4,2,4\n"
        );
    }
}
