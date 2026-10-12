//! The resident KV by storage format over time, for the arena panel: the few
//! formats that held the most, each its own series, and everything else as
//! one `other` series — so a census naming twenty formats draws as a handful
//! of bands rather than twenty.
//!
//! The formats are chosen by their peak over the whole window, not by the
//! latest census, so a format that held most of the KV an hour ago keeps its
//! band and its colour while it drains rather than vanishing into `other`.

use serde::Serialize;

/// The series' name for the formats outside the top.
pub const OTHER: &str = "other";

/// One census: seconds before the answer, and `(format, tokens)` rows.
pub struct Census {
    pub age: f64,
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
    pub age: f64,
    /// Tokens per series, in [`MixSeries::formats`] order.
    pub tok: Vec<f64>,
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
            let mut tok = vec![0.0; formats.len()];
            for (name, tokens) in &c.rows {
                match formats[..named].iter().position(|f| f == name) {
                    Some(i) => tok[i] += tokens,
                    None => tok[named] += tokens,
                }
            }
            MixRow {
                age: (c.age * 100.0).round() / 100.0,
                tok,
            }
        })
        .collect();
    MixSeries { formats, rows }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn census(age: f64, rows: &[(&str, f64)]) -> Census {
        Census {
            age,
            rows: rows.iter().map(|(n, t)| (n.to_string(), *t)).collect(),
        }
    }

    /// The top formats by peak keep their own series, in peak order; the rest
    /// add up in `other`, row by row.
    #[test]
    fn the_top_formats_by_peak_keep_their_series_and_the_rest_fold_into_other() {
        let s = mix_series(
            &[
                census(8.0, &[("BF16", 900.0), ("Q4_0", 100.0), ("Q2_0", 5.0)]),
                census(4.0, &[("Q4_0", 600.0), ("Q8_0", 50.0), ("Q2_0", 7.0)]),
            ],
            2,
        );
        assert_eq!(s.formats, vec!["BF16", "Q4_0", "other"]);
        assert_eq!(
            s.rows,
            vec![
                MixRow {
                    age: 8.0,
                    tok: vec![900.0, 100.0, 5.0]
                },
                MixRow {
                    age: 4.0,
                    tok: vec![0.0, 600.0, 57.0]
                },
            ]
        );
    }

    /// No more formats than the top: no `other` series at all.
    #[test]
    fn few_formats_have_no_other() {
        let s = mix_series(&[census(1.0, &[("Q4_0", 3.0)])], 6);
        assert_eq!(s.formats, vec!["Q4_0"]);
        assert_eq!(
            s.rows,
            vec![MixRow {
                age: 1.0,
                tok: vec![3.0]
            }]
        );
    }

    /// No census, no series.
    #[test]
    fn no_census_is_empty() {
        assert_eq!(mix_series(&[], 6), MixSeries::default());
    }
}
