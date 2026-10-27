//! The resident KV by storage format over time, for the arena panel: the few
//! formats that held the most, each its own series, and everything else as
//! one `other` series — so a census naming twenty formats draws as a handful
//! of bands rather than twenty.
//!
//! The formats are chosen by their peak over the whole window, not by the
//! latest census, so a format that held most of the KV an hour ago keeps its
//! band and its colour while it drains rather than vanishing into `other`.
//!
//! Each row's tokens are keyed by the series' name, so a cluster's rows add up
//! format by format whatever order each member ranks its formats in
//! (`cluster::dispatch::fuse`).

use std::collections::BTreeMap;

use serde::Serialize;

/// The series' name for the formats outside the top.
pub const OTHER: &str = "other";

/// One census: its slot (unix ms), and `(format, tokens)` rows.
pub struct Census {
    pub at: u64,
    pub rows: Vec<(String, f64)>,
}

#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct MixSeries {
    /// The series, in stacking order: the top formats, largest peak first,
    /// then [`OTHER`] when anything fell outside them.
    pub formats: Vec<String>,
    pub rows: Vec<MixRow>,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct MixRow {
    pub at: u64,
    /// Tokens per series, by the series' name.
    pub tok: BTreeMap<String, f64>,
}

/// Fold `censuses` into at most `top` named series and one [`OTHER`].
pub fn mix_series(censuses: &[Census], top: usize) -> MixSeries {
    let mut peaks: Vec<(String, f64)> = Vec::new();
    for c in censuses {
        for (name, tokens) in &c.rows {
            match peaks.iter_mut().find(|(n, _)| n == name) {
                Some((_, p)) => *p = p.max(*tokens),
                None => peaks.push((name.clone(), *tokens)),
            }
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let has_other = peaks.len() > top;
    let mut formats: Vec<String> = peaks.into_iter().take(top).map(|(n, _)| n).collect();
    let named = formats.len();
    if has_other {
        formats.push(OTHER.to_string());
    }
    let rows = censuses
        .iter()
        .map(|c| {
            let mut tok: BTreeMap<String, f64> = BTreeMap::new();
            for (name, tokens) in &c.rows {
                let series = if formats[..named].contains(name) {
                    name.as_str()
                } else {
                    OTHER
                };
                *tok.entry(series.to_owned()).or_default() += tokens;
            }
            MixRow { at: c.at, tok }
        })
        .collect();
    MixSeries { formats, rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn census(at: u64, rows: &[(&str, f64)]) -> Census {
        Census {
            at,
            rows: rows.iter().map(|(n, t)| (n.to_string(), *t)).collect(),
        }
    }

    fn tok(rows: &[(&str, f64)]) -> BTreeMap<String, f64> {
        rows.iter().map(|(n, t)| (n.to_string(), *t)).collect()
    }

    /// The top formats by peak keep their own series, in peak order; the rest
    /// add up in `other`, row by row.
    #[test]
    fn the_top_formats_by_peak_keep_their_series_and_the_rest_fold_into_other() {
        let s = mix_series(
            &[
                census(5_000, &[("BF16", 900.0), ("Q4_0", 100.0), ("Q2_0", 5.0)]),
                census(10_000, &[("Q4_0", 600.0), ("Q8_0", 50.0), ("Q2_0", 7.0)]),
            ],
            2,
        );
        assert_eq!(s.formats, vec!["BF16", "Q4_0", "other"]);
        assert_eq!(
            s.rows,
            vec![
                MixRow {
                    at: 5_000,
                    tok: tok(&[("BF16", 900.0), ("Q4_0", 100.0), ("other", 5.0)])
                },
                MixRow {
                    at: 10_000,
                    tok: tok(&[("Q4_0", 600.0), ("other", 57.0)])
                },
            ]
        );
    }

    /// No more formats than the top: no `other` series at all.
    #[test]
    fn few_formats_have_no_other() {
        let s = mix_series(&[census(5_000, &[("Q4_0", 3.0)])], 6);
        assert_eq!(s.formats, vec!["Q4_0"]);
        assert_eq!(
            s.rows,
            vec![MixRow {
                at: 5_000,
                tok: tok(&[("Q4_0", 3.0)])
            }]
        );
    }

    /// No census, no series.
    #[test]
    fn no_census_is_empty() {
        assert_eq!(mix_series(&[], 6), MixSeries::default());
    }
}
