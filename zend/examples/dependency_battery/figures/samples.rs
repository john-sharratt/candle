//! The telemetry table (`fig_9_6_telemetry.csv`), streamed: one row per
//! sample, hundreds of columns wide, of which a figure needs a dozen. Only the
//! columns asked for are kept, only the samples with answers in flight and
//! every kept column present, and the result is bucketed down to a few
//! hundred points of equal spans of time.

use std::io::BufRead;

use anyhow::{anyhow, Context};

use crate::stats::mean;

/// The time column, seconds from the first sample.
pub const TIME: &str = "t_s";
/// The answers in flight when the sample was taken.
pub const IN_FLIGHT: &str = "in_flight";
/// The newest wave's card by tenant, MiB.
pub const VRAM: &str = "telemetry/waves/vram/";
/// The newest format census: the tokens resident in each format.
pub const MIX: &str = "telemetry/mix/tok/";
/// Seconds in an hour, the figures' time unit.
pub const HOUR: f64 = 3600.0;
/// MiB in a GiB, the figures' memory unit.
pub const GIB_MIB: f64 = 1024.0;

/// The mean and the peak of `xs`.
pub fn mean_peak(xs: &[f64]) -> (f64, f64) {
    (
        mean(xs),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// The kept columns and the kept samples' values in them.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    pub names: Vec<String>,
    pub rows: Vec<Vec<f64>>,
}

/// Read `reader`'s table, keeping [`TIME`], [`IN_FLIGHT`] and every column
/// `keep` accepts, in header order.
pub fn read(mut reader: impl BufRead, keep: impl Fn(&str) -> bool) -> anyhow::Result<Samples> {
    let mut line = String::new();
    reader.read_line(&mut line).context("reading the header")?;
    let mut names = Vec::new();
    let slots: Vec<Option<usize>> = line
        .trim_end()
        .split(',')
        .map(|h| {
            (h == TIME || h == IN_FLIGHT || keep(h)).then(|| {
                names.push(h.to_string());
                names.len() - 1
            })
        })
        .collect();
    let in_flight = names
        .iter()
        .position(|n| n == IN_FLIGHT)
        .ok_or_else(|| anyhow!("no {IN_FLIGHT} column"))?;
    if !names.iter().any(|n| n == TIME) {
        return Err(anyhow!("no {TIME} column"));
    }
    let mut rows = Vec::new();
    let mut number = 1;
    loop {
        line.clear();
        if reader
            .read_line(&mut line)
            .with_context(|| format!("reading line {}", number + 1))?
            == 0
        {
            break;
        }
        number += 1;
        let mut row = vec![f64::NAN; names.len()];
        for (field, slot) in line.trim_end().split(',').zip(&slots) {
            if let Some(i) = slot {
                row[*i] = field.parse().unwrap_or(f64::NAN);
            }
        }
        if row.iter().all(|v| v.is_finite()) && row[in_flight] > 0.0 {
            rows.push(row);
        }
    }
    Ok(Samples { names, rows })
}

impl Samples {
    pub fn column(&self, name: &str) -> anyhow::Result<usize> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| anyhow!("no telemetry column {name}"))
    }

    /// The names kept that start with `prefix`, the prefix taken off.
    pub fn under(&self, prefix: &str) -> Vec<String> {
        self.names
            .iter()
            .filter_map(|n| n.strip_prefix(prefix).map(str::to_string))
            .collect()
    }

    /// The column `name`'s values, sample by sample.
    pub fn series(&self, name: &str) -> anyhow::Result<Vec<f64>> {
        let i = self.column(name)?;
        Ok(self.rows.iter().map(|r| r[i]).collect())
    }

    /// The samples bucketed into at most `max` equal spans of time, each
    /// bucket the mean of its samples in every column, empty buckets left
    /// out.
    pub fn downsample(&self, max: usize) -> anyhow::Result<Samples> {
        let t = self.column(TIME)?;
        let (t0, t1) = self
            .rows
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), r| {
                (a.min(r[t]), b.max(r[t]))
            });
        let width = self.names.len();
        let mut sums = vec![(vec![0.0; width], 0usize); max];
        for r in &self.rows {
            let at = if t1 > t0 {
                (((r[t] - t0) / (t1 - t0)) * max as f64).floor() as usize
            } else {
                0
            };
            let (sum, count) = &mut sums[at.min(max - 1)];
            sum.iter_mut().zip(r).for_each(|(s, v)| *s += v);
            *count += 1;
        }
        Ok(Samples {
            names: self.names.clone(),
            rows: sums
                .into_iter()
                .filter(|(_, c)| *c > 0)
                .map(|(s, c)| s.into_iter().map(|v| v / c as f64).collect())
                .collect(),
        })
    }
}

/// A plotted series as a CSV: `header`, then a line per point.
pub fn csv(header: &[&str], rows: &[Vec<f64>]) -> String {
    let mut out = header.join(",");
    out.push('\n');
    for r in rows {
        out.push_str(&r.iter().map(f64::to_string).collect::<Vec<_>>().join(","));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    const TABLE: &str = "t_s,in_flight,a,b/x,b/y\r\n\
                         0.0,0,1,1,1\r\n\
                         5.0,2,9,2,4\r\n\
                         10.0,3,9,,4\r\n\
                         15.0,4,9,6,8\r\n";

    #[test]
    fn only_kept_columns_and_loaded_complete_samples_are_read() {
        let s = read(Cursor::new(TABLE), |h| h.starts_with("b/")).unwrap();
        assert_eq!(s.names, vec!["t_s", "in_flight", "b/x", "b/y"]);
        assert_eq!(
            s.rows,
            vec![vec![5.0, 2.0, 2.0, 4.0], vec![15.0, 4.0, 6.0, 8.0]]
        );
        assert_eq!(s.under("b/"), vec!["x", "y"]);
        assert_eq!(s.series("b/y").unwrap(), vec![4.0, 8.0]);
        assert!(s.column("a").is_err());
        assert!(read(Cursor::new("t_s,a\n1,2\n"), |_| true).is_err());
    }

    /// Four samples over 0–30 s into two buckets: the first three fall
    /// before 15 s, the last alone at its end.
    #[test]
    fn downsampling_takes_bucket_means_over_equal_spans_of_time() {
        let s = Samples {
            names: vec!["t_s".into(), "in_flight".into()],
            rows: vec![
                vec![0.0, 1.0],
                vec![5.0, 2.0],
                vec![10.0, 6.0],
                vec![30.0, 8.0],
            ],
        };
        assert_eq!(
            s.downsample(2).unwrap().rows,
            vec![vec![5.0, 3.0], vec![30.0, 8.0]]
        );
        assert_eq!(s.downsample(96).unwrap().rows.len(), 4);
        assert_eq!(csv(&["a", "b"], &[vec![1.0, 2.5]]), "a,b\n1,2.5\n");
        assert_eq!(mean_peak(&[1.0, 4.0, 1.0]), (2.0, 4.0));
    }
}
