//! Figure 9.12-7, the needle in the haystack: the share of questions whose
//! best-ranked answer file lies within the top `k` of the gallery, for `k`
//! from 1 to 100 on a log axis, with its seeded 95% bootstrap band over
//! questions, against what a random ranking of the gallery's files would
//! give — for a question with `m` answer files among `N`, `min(1, k·m/N)`.

use crate::analyse::{RESAMPLES, SEED};
use crate::figures::svg::Precision::Tenth;
use crate::figures::svg::{area_path, line_path, Anchor, Linear, Log, Svg, Swatch};
use crate::figures::table::Table;
use crate::figures::text::{pct0, thousands, ticks};
use crate::stats::{bootstrap_mean_interval, mean};

/// One question's best answer-file rank (`None` when no answer file was
/// ranked) and how many of its answer files the gallery holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Needle {
    pub best_rank: Option<u32>,
    pub answer_files: u32,
}

pub fn needles(t: &Table) -> anyhow::Result<Vec<Needle>> {
    t.records()
        .map(|r| {
            let rank = r.get("best_rank")?;
            Ok(Needle {
                best_rank: if rank.is_empty() {
                    None
                } else {
                    Some(r.num("best_rank")? as u32)
                },
                answer_files: r.num("answer_files_ranked")? as u32,
            })
        })
        .collect()
}

/// The curve at one `k`: the share of questions hit, its band, and the
/// random ranking's expected share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub k: u32,
    pub hit: f64,
    pub lo: f64,
    pub hi: f64,
    pub random: f64,
}

/// The curve for `k` in `1..=k_max` over a gallery of `files` files, each
/// `k`'s band from `resamples` resamples of `seed`'s stream — the same
/// questions drawn at every `k`, since every `k` replays one stream.
pub fn curve(
    needles: &[Needle],
    files: u32,
    k_max: u32,
    resamples: usize,
    seed: u64,
) -> Vec<Point> {
    (1..=k_max)
        .map(|k| {
            let hits: Vec<f64> = needles
                .iter()
                .map(|q| f64::from(q.best_rank.is_some_and(|r| r <= k)))
                .collect();
            let (lo, hi) = bootstrap_mean_interval(&hits, resamples, seed);
            let random: Vec<f64> = needles
                .iter()
                .map(|q| (f64::from(k) * f64::from(q.answer_files) / f64::from(files)).min(1.0))
                .collect();
            Point {
                k,
                hit: mean(&hits),
                lo,
                hi,
                random: mean(&random),
            }
        })
        .collect()
}

/// The curve over `k` = 1 to 100 at the analysis's seed and resamples.
pub fn analysed(needles: &[Needle], files: u32) -> Vec<Point> {
    curve(needles, files, 100, RESAMPLES, SEED)
}

const X0: f64 = 62.0;
const X1: f64 = 616.0;
const Y0: f64 = 290.0;
const Y1: f64 = 70.0;
const MARKED: [u32; 2] = [1, 5];

pub fn svg(points: &[Point], questions: usize, files: u32) -> String {
    let at = |k: u32| points.iter().find(|p| p.k == k);
    let k_max = points.last().map(|p| p.k).unwrap_or(1);
    let mut said_at = vec![1, 5, 10, k_max];
    said_at.dedup();
    let said: Vec<String> = said_at
        .into_iter()
        .filter_map(|k| at(k).map(|p| format!("{} at k = {k}", pct0(p.hit))))
        .collect();
    let random = at(k_max).map(|p| pct0(p.random)).unwrap_or_default();
    let aria = format!(
        "The share of the {questions} questions whose best-ranked answer file lies within the top k of \
         the gallery's {} files, with a 95% bootstrap band over questions: {}; a random ranking of the \
         gallery would reach {random} at k = {k_max}.",
        thousands(u64::from(files)),
        said.join(", ")
    );
    let mut svg = Svg::new(
        334.0,
        &aria,
        "The needle in the haystack",
        &format!(
            "share of {questions} questions with an answer file in the top k of {} files · band: 95% bootstrap",
            thousands(u64::from(files))
        ),
    );
    let x = Log::new((1.0, f64::from(k_max.max(2))), (X0, X1));
    let y = Linear::new((0.0, 1.0), (Y0, Y1));
    svg.y_grid(&y, &ticks(0.0, 1.0, 0.25, pct0), (X0, X1), 0.0);
    let columns: Vec<(f64, String)> = [1, 2, 5, 10, 20, 50, 100]
        .into_iter()
        .filter(|k| *k <= k_max)
        .map(|k| (x.at(f64::from(k)), k.to_string()))
        .collect();
    svg.x_grid(&columns, (Y1, Y0), Y0 + 16.0);
    svg.text(
        "t-dim",
        (X0 + X1) / 2.0,
        Y0 + 34.0,
        Anchor::Middle,
        "k, the rank cut-off (log scale)",
    );
    let xy = |k: u32, v: f64| (x.at(f64::from(k)), y.at(v));
    let top: Vec<(f64, f64)> = points.iter().map(|p| xy(p.k, p.hi)).collect();
    let bottom: Vec<(f64, f64)> = points.iter().map(|p| xy(p.k, p.lo)).collect();
    svg.path("area-us", &area_path(&top, &bottom, Tenth), None);
    let random: Vec<(f64, f64)> = points.iter().map(|p| xy(p.k, p.random)).collect();
    svg.path("line-them dash", &line_path(&random, Tenth), None);
    let hit: Vec<(f64, f64)> = points.iter().map(|p| xy(p.k, p.hit)).collect();
    svg.path("line-us", &line_path(&hit, Tenth), None);
    for k in MARKED {
        if let Some(p) = at(k) {
            let (cx, cy) = xy(k, p.hit);
            svg.circle("dot-us", cx, cy, 5.5, None);
            svg.text(
                "v-us",
                cx + 10.0,
                cy - 8.0,
                Anchor::Start,
                &format!("{} at {k}", pct0(p.hit)),
            );
        }
    }
    svg.legend(
        330.0,
        56.0,
        &[
            (Swatch::Line("line-us"), "best answer-file rank"),
            (Swatch::Line("line-them dash"), "random ranking"),
        ],
    );
    svg.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four() -> Vec<Needle> {
        let t = Table::parse(
            "id,category,best_rank,answer_files_ranked\n\
             a,x,1,1\nb,x,3,2\nc,x,,1\nd,x,10,1\n",
        )
        .unwrap();
        needles(&t).unwrap()
    }

    #[test]
    fn an_unranked_question_reads_as_never_hit() {
        assert_eq!(
            four()[2],
            Needle {
                best_rank: None,
                answer_files: 1
            }
        );
    }

    /// Hits at 1, 3 and 10; random at k = 1 over 8 files is the mean of
    /// 1/8, 2/8, 1/8, 1/8, and at k = 10 every question is certain.
    #[test]
    fn the_curve_counts_hits_and_the_random_ranking_per_k() {
        let c = curve(&four(), 8, 10, 200, 5);
        let got: Vec<(u32, f64)> = c.iter().map(|p| (p.k, p.hit)).collect();
        assert_eq!(got[..4], [(1, 0.25), (2, 0.25), (3, 0.5), (4, 0.5)]);
        assert_eq!(c[9].hit, 0.75);
        assert_eq!((c[0].random, c[9].random), (0.15625, 1.0));
        let hits_at_3 = [1.0, 1.0, 0.0, 0.0];
        assert_eq!(
            (c[2].lo, c[2].hi),
            bootstrap_mean_interval(&hits_at_3, 200, 5)
        );
    }

    /// k runs 1 to 10 on a log axis over 554 px; k = 5 sits at
    /// 62 + log10(5)·554 = 449.2, and 50% at 290 − 110 = 180.
    #[test]
    fn the_marked_ranks_are_dotted_and_labelled() {
        let pts = curve(&four(), 8, 10, 200, 5);
        let out = svg(&pts, 4, 8);
        let lines: Vec<&str> = out.lines().map(str::trim).collect();
        for want in [
            "<circle class=\"dot-us\" cx=\"62\" cy=\"235\" r=\"5.5\"/>",
            "<text class=\"v-us\" x=\"72\" y=\"227\">25% at 1</text>",
            "<circle class=\"dot-us\" cx=\"449.2\" cy=\"180\" r=\"5.5\"/>",
            "<text class=\"v-us\" x=\"459.2\" y=\"172\">50% at 5</text>",
            "<text class=\"tick mid\" x=\"616\" y=\"306\">10</text>",
        ] {
            assert!(lines.contains(&want), "missing {want}\n{out}");
        }
        assert!(out.contains("25% at k = 1, 50% at k = 5, 75% at k = 10; a random ranking of the gallery would reach 100% at k = 10."));
    }
}
