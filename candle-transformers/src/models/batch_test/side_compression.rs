//! How a quantized rung splits its bits between K and V.
//!
//! The comparison table's `Compress` column is one number over both sides, so a
//! rung that trades K precision for V compression looks unchanged there while
//! the two sides move in opposite directions. This measures each side's ratio
//! and format mix per run and prints them as one table, rung by rung, so a
//! ladder can be checked for smooth transitions on both axes.

use crate::models::batched_inference::BatchedInferenceSession;

/// One side's compression for a run: its ratio against F16 and the share of
/// its bands at each format, most compressed first.
#[derive(Debug, Clone, Default)]
pub struct SideCompression {
    pub ratio: Option<f64>,
    /// `(format, bands)`, sorted by bytes per element ascending.
    pub mix: Vec<(String, usize)>,
}

impl SideCompression {
    /// Measure one side — K when `is_value` is false — over `sequences`.
    pub fn measure(session: &BatchedInferenceSession, sequences: &[usize], is_value: bool) -> Self {
        let mut mix: Vec<_> = session
            .compression_dist_by_side(sequences, is_value)
            .into_iter()
            .collect();
        mix.sort_by(|a, b| a.0.type_size().cmp(&b.0.type_size()).then(a.1.cmp(&b.1)));
        Self {
            ratio: session.compression_ratio_by_side(sequences, is_value),
            mix: mix
                .into_iter()
                .map(|(f, n)| (format!("{f:?}"), n))
                .collect(),
        }
    }

    /// The mix as `Q3_0 41% Q4_0 30% …`, shares of this side's bands.
    pub fn mix_line(&self) -> String {
        let total: usize = self.mix.iter().map(|(_, n)| n).sum();
        if total == 0 {
            return "-".to_string();
        }
        self.mix
            .iter()
            .map(|(f, n)| format!("{f} {:.0}%", 100.0 * *n as f64 / total as f64))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// A ratio as the table shows it, `-` where there is none.
fn ratio_cell(r: Option<f64>) -> String {
    r.map_or_else(|| "-".to_string(), |r| format!("{r:.2}x"))
}

/// Print every quantized row's K and V ratios and format mixes, in run order.
pub fn print_side_table(rows: &[(String, Option<f64>, &SideCompression, &SideCompression)]) {
    if rows.is_empty() {
        return;
    }
    println!("\n=== KV compression by side ===");
    println!(
        "{:>8}  {:>7}  {:>7}  {:>7}  K formats | V formats",
        "KvMode", "total", "K", "V"
    );
    for (mode, total, k, v) in rows {
        println!(
            "{:>8}  {:>7}  {:>7}  {:>7}  {} | {}",
            mode,
            ratio_cell(*total),
            ratio_cell(k.ratio),
            ratio_cell(v.ratio),
            k.mix_line(),
            v.mix_line(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mix_reads_as_shares_of_the_side() {
        let side = SideCompression {
            ratio: Some(3.2),
            mix: vec![("Q3_0".into(), 1), ("Q4_0".into(), 3)],
        };
        assert_eq!(side.mix_line(), "Q3_0 25% Q4_0 75%");
        assert_eq!(SideCompression::default().mix_line(), "-");
        assert_eq!(ratio_cell(Some(2.756)), "2.76x");
        assert_eq!(ratio_cell(None), "-");
    }
}
