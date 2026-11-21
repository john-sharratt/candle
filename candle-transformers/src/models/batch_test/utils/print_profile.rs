//! Prints the profile tables for a finished row and the span table that compares
//! configurations.
//!
//! Every span accumulator is reset at the row's phase boundaries, so a row's
//! tables hold that row alone, and nothing prints without the profile feature.
//! The comparison table takes its rows as the union of span names, sorted so
//! the layout stays stable as spans are added. Each cell is a total with its
//! count.

use super::*;

impl TestParams {
    /// Print one finished row's four profile tables — prompt and generate, model
    /// and pipeline spans — straight after the row, headed by the row's name.
    ///
    /// Every span accumulator is reset at the row's phase boundaries, so each
    /// table holds that row alone: a span that grows with the rows before it
    /// shows up as the same span read down successive dumps. Prints nothing
    /// without the `profile` feature, which leaves every snapshot empty.
    pub(super) fn print_row_profiles(&self, result: &TestResults, label: Option<&str>) {
        let row = std::slice::from_ref(result);
        let name = match label {
            Some(label) => label.to_string(),
            None => format!("{:?}×{}", result.config.mode, result.config.num_contexts),
        };
        self.print_profile_table(row, &format!("{name} — Bulk (Prompt) Profile"), |r| {
            &r.bulk_profile
        });
        self.print_profile_table(row, &format!("{name} — Single (Generate) Profile"), |r| {
            &r.single_profile
        });
        self.print_profile_table(
            row,
            &format!("{name} — Pipeline Profile, PREFILL (bulk)"),
            |r| &r.pipeline_bulk_profile,
        );
        self.print_profile_table(
            row,
            &format!("{name} — Pipeline Profile, DECODE (single)"),
            |r| &r.pipeline_profile,
        );
    }

    /// Print a transposed profile-timing table.
    ///
    /// Rows = span names (union across configs), Columns = test configs.
    /// `extract` pulls the relevant [`ProfileSnapshot`] from each result
    /// (bulk or single).  Skipped when no config has any profile entries.
    pub(super) fn print_profile_table(
        &self,
        results: &[TestResults],
        title: &str,
        extract: impl Fn(&TestResults) -> &ProfileSnapshot,
    ) {
        // Collect snapshots.
        let snaps: Vec<&ProfileSnapshot> = results.iter().map(|r| extract(r)).collect();

        // Build the union of all span names and sort alphabetically so the
        // table stays stable as new rows are added.
        let mut span_names: Vec<String> = Vec::new();
        for snap in &snaps {
            for (name, _, _) in &snap.entries {
                if !span_names.contains(name) {
                    span_names.push(name.clone());
                }
            }
        }
        span_names.sort();

        // Skip if nothing to show.
        if span_names.is_empty() {
            return;
        }

        // Column headers.
        let headers: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("#{} {:?}×{}", i + 1, r.config.mode, r.config.num_contexts))
            .collect();

        // Column width: each column shows "total_ms (×count)" => needs space.
        let col_w = headers.iter().map(|h| h.len()).max().unwrap_or(10).max(18);
        let label_w = span_names
            .iter()
            .map(|n| n.len())
            .max()
            .unwrap_or(16)
            .max(20);

        // ── Header ──
        println!("\n=== {} ===", title);
        print!("┌{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┬{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┐");

        print!("│ {:label_w$} ", "Span");
        for h in &headers {
            print!("│ {:>col_w$} ", h);
        }
        println!("│");

        print!("├{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┼{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┤");

        // ── Data rows ──
        for span in &span_names {
            print!("│ {:label_w$} ", span);
            for snap in &snaps {
                let cell =
                    if let Some((_, ms, count)) = snap.entries.iter().find(|(n, _, _)| n == span) {
                        format!("{:.1}ms (×{})", ms, count)
                    } else {
                        "-".to_string()
                    };
                print!("│ {:>col_w$} ", cell);
            }
            println!("│");
        }

        // ── Footer ──
        print!("└{:─<lw$}", "", lw = label_w + 2);
        for _ in &headers {
            print!("┴{:─<cw$}", "", cw = col_w + 2);
        }
        println!("┘");
    }
}
