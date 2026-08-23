use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct ProfileEntry {
    pub name: &'static str,
    pub total: Duration,
    pub count: u64,
    first_seen: usize,
}

#[derive(Debug, Clone, Default)]
pub struct SampledSelectionBenchmarkResult {
    entries: Vec<ProfileEntry>,
}

impl SampledSelectionBenchmarkResult {
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn record(&mut self, name: &'static str, start: Instant) {
        self.record_duration(name, start.elapsed(), 1);
    }

    pub fn record_duration(&mut self, name: &'static str, elapsed: Duration, count: u64) {
        if count == 0 {
            return;
        }
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.name == name) {
            entry.total += elapsed;
            entry.count += count;
        } else {
            let first_seen = self.entries.len();
            self.entries.push(ProfileEntry {
                name,
                total: elapsed,
                count,
                first_seen,
            });
        }
    }

    pub fn report(&self, header: &str) -> String {
        if self.entries.is_empty() {
            return format!("{header}\n(no sampled-selection profiling data recorded)");
        }

        let mut entries = self.entries.clone();
        entries.sort_by_key(|e| e.first_seen);
        let total_ms_by_name = entries
            .iter()
            .map(|entry| (entry.name, entry.total.as_secs_f64() * 1000.0))
            .collect::<HashMap<_, _>>();

        let mut rows = Vec::with_capacity(entries.len());

        for entry in entries {
            let total_ms = entry.total.as_secs_f64() * 1000.0;
            let avg_ms = if entry.count > 0 {
                total_ms / entry.count as f64
            } else {
                0.0
            };
            let share = profile_share_percent(entry.name, total_ms, &total_ms_by_name);
            let short = entry
                .name
                .strip_prefix("benchmark.")
                .or_else(|| entry.name.strip_prefix("quantization."))
                .unwrap_or(entry.name);
            let depth = short.matches('.').count();
            rows.push((
                profile_root(entry.name).to_string(),
                short.to_string(),
                depth,
                total_ms,
                entry.count,
                avg_ms,
                share,
            ));
        }

        // Use inverted indentation: in execution-order output (inner scopes first, totals last)
        // the deepest leaf entries appear first and summarising totals appear after. Inverting
        // the indent (leaves at column 0, outer totals more indented) means each summary line
        // visually "closes" the block above it rather than appearing shallower than its children.
        let max_depth = rows.iter().map(|(_, _, d, ..)| *d).max().unwrap_or(0);
        let display_names: Vec<String> = rows
            .iter()
            .map(|(_, short, depth, ..)| {
                let indent = "  ".repeat(max_depth.saturating_sub(*depth));
                format!("{indent}{short}")
            })
            .collect();
        let scope_width = display_names
            .iter()
            .map(|s| s.chars().count())
            .chain(std::iter::once("Scope".len()))
            .chain(std::iter::once(28))
            .max()
            .unwrap_or(28);

        let mut rendered_rows = Vec::with_capacity(rows.len() + 1);
        rendered_rows.push(format!(
            "{:<scope_width$} {:>11} {:>8} {:>11} {:>10}",
            "Scope",
            "Total",
            "Count",
            "Avg",
            "Parent %",
            scope_width = scope_width
        ));
        for ((_, _, _, total_ms, count, avg_ms, share), display_name) in
            rows.iter().zip(display_names.iter())
        {
            rendered_rows.push(format!(
                "{:<scope_width$} {:>9.2}ms {:>8} {:>9.2}ms {:>8.1}%",
                display_name,
                total_ms,
                count,
                avg_ms,
                share,
                scope_width = scope_width
            ));
        }

        let title = format!("  {header}");
        let legend =
            "  Ordered by execution flow; percentages are relative to each row's parent scope";
        let inner_width = rendered_rows
            .iter()
            .map(|row| row.chars().count())
            .chain(std::iter::once(title.chars().count()))
            .chain(std::iter::once(legend.chars().count()))
            .max()
            .unwrap_or(title.chars().count())
            .max(40);

        let mut out = String::new();
        out.push('\n');
        out.push_str(&format!("╔{}╗\n", "═".repeat(inner_width + 2)));
        out.push_str(&format!(
            "║ {:<inner_width$} ║\n",
            title,
            inner_width = inner_width
        ));
        out.push_str(&format!(
            "║ {:<inner_width$} ║\n",
            legend,
            inner_width = inner_width
        ));
        out.push_str(&format!("╠{}╣\n", "═".repeat(inner_width + 2)));
        out.push_str(&format!(
            "║ {:<inner_width$} ║\n",
            rendered_rows[0],
            inner_width = inner_width
        ));
        out.push_str(&format!("╟{}╢\n", "─".repeat(inner_width + 2)));
        let mut last_root: Option<&str> = None;
        for ((root, _, _, _, _, _, _), row) in rows.iter().zip(rendered_rows.iter().skip(1)) {
            if last_root != Some(root.as_str()) {
                if let Some(label) = profile_section_label(root) {
                    let section = format!("-- {label} --");
                    out.push_str(&format!(
                        "║ {:<inner_width$} ║\n",
                        section,
                        inner_width = inner_width
                    ));
                }
                last_root = Some(root.as_str());
            }
            out.push_str(&format!(
                "║ {:<inner_width$} ║\n",
                row,
                inner_width = inner_width
            ));
        }
        out.push_str(&format!("╚{}╝\n", "═".repeat(inner_width + 2)));
        out
    }
}

pub(crate) fn sampled_profile_record(
    profile: Option<&mut SampledSelectionBenchmarkResult>,
    name: &'static str,
    start: Instant,
) {
    sampled_profile_record_duration(profile, name, start.elapsed(), 1);
}

pub(crate) fn sampled_profile_record_duration(
    profile: Option<&mut SampledSelectionBenchmarkResult>,
    name: &'static str,
    elapsed: Duration,
    count: u64,
) {
    if let Some(profile) = profile {
        profile.record_duration(name, elapsed, count);
    }
}

fn profile_root(name: &str) -> &str {
    name.split('.').next().unwrap_or(name)
}

fn profile_share_percent(
    name: &str,
    total_ms: f64,
    total_ms_by_name: &HashMap<&'static str, f64>,
) -> f64 {
    let denominator = profile_parent_total_ms(name, total_ms_by_name)
        .or_else(|| profile_section_total_ms(name, total_ms_by_name))
        .unwrap_or(total_ms);
    if denominator > 0.0 {
        total_ms / denominator * 100.0
    } else {
        0.0
    }
}

fn profile_parent_total_ms(
    name: &str,
    total_ms_by_name: &HashMap<&'static str, f64>,
) -> Option<f64> {
    let mut parts = name.split('.').collect::<Vec<_>>();
    parts.pop()?;
    while parts.len() > 1 {
        let candidate = format!("{}.total", parts.join("."));
        if candidate != name {
            if let Some(total_ms) = total_ms_by_name.get(candidate.as_str()) {
                return Some(*total_ms);
            }
        }
        parts.pop();
    }
    None
}

fn profile_section_total_ms(
    name: &str,
    total_ms_by_name: &HashMap<&'static str, f64>,
) -> Option<f64> {
    let root = profile_root(name);
    let explicit_total = format!("{root}.total");
    if let Some(total_ms) = total_ms_by_name.get(explicit_total.as_str()) {
        return Some(*total_ms);
    }

    let section_total = total_ms_by_name
        .iter()
        .filter(|(entry_name, _)| {
            profile_root(entry_name) == root
                && entry_name.ends_with(".total")
                && entry_name[root.len() + 1..].split('.').count() == 2
        })
        .map(|(_, total_ms)| *total_ms)
        .sum::<f64>();

    (section_total > 0.0).then_some(section_total)
}

fn profile_section_label(root: &str) -> Option<&'static str> {
    match root {
        "benchmark" => Some("Benchmark Harness"),
        "quantization" => Some("Production sample_quantization"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_profile_report_rows_align() {
        let mut result = SampledSelectionBenchmarkResult::default();
        result.record_duration(
            "quantization.key.select.total",
            Duration::from_millis(17),
            616,
        );
        result.record_duration("benchmark.total", Duration::from_millis(336), 1);

        let report = result.report("Sampled-Selection Full Workflow Profile");
        let widths = report
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| line.chars().count())
            .collect::<Vec<_>>();

        assert!(!widths.is_empty());
        assert!(
            widths.windows(2).all(|pair| pair[0] == pair[1]),
            "misaligned report:\n{report}"
        );
    }

    #[test]
    fn sampled_profile_report_follows_execution_order() {
        let mut result = SampledSelectionBenchmarkResult::default();
        result.record_duration("benchmark.io.load_dump", Duration::from_millis(3), 1);
        result.record_duration("benchmark.batch.k.flatten", Duration::from_millis(2), 1);
        result.record_duration(
            "quantization.key.surface.dispatch.probe.cuda",
            Duration::from_millis(1),
            1,
        );
        result.record_duration(
            "quantization.key.surface.gpu.prepare_inputs",
            Duration::from_millis(1),
            1,
        );
        result.record_duration(
            "quantization.key.surface.gpu.total",
            Duration::from_millis(4),
            1,
        );
        result.record_duration(
            "quantization.key.surface.total",
            Duration::from_millis(5),
            1,
        );
        result.record_duration("benchmark.batch.total", Duration::from_millis(6), 1);
        result.record_duration("benchmark.total", Duration::from_millis(7), 1);

        let report = result.report("Sampled-Selection Full Workflow Profile");
        let positions = [
            report.find("io.load_dump").unwrap(),
            report.find("batch.k.flatten").unwrap(),
            report.find("key.surface.dispatch.probe.cuda").unwrap(),
            report.find("key.surface.gpu.prepare_inputs").unwrap(),
            report.find("key.surface.gpu.total").unwrap(),
            report.find("key.surface.total").unwrap(),
            report.find("batch.total").unwrap(),
            // rfind " total " (space-before-total) uniquely finds standalone `total`;
            // other .total entries have a dot before `total`, never a space.
            report.rfind(" total ").unwrap(),
        ];

        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "wrong order:\n{report}"
        );
    }

    #[test]
    fn sampled_profile_report_uses_parent_level_shares() {
        let mut result = SampledSelectionBenchmarkResult::default();
        result.record_duration(
            "quantization.key.surface.gpu.prepare_inputs",
            Duration::from_millis(25),
            1,
        );
        result.record_duration(
            "quantization.key.surface.gpu.kernel",
            Duration::from_millis(75),
            1,
        );
        result.record_duration(
            "quantization.key.surface.gpu.total",
            Duration::from_millis(100),
            1,
        );
        result.record_duration(
            "quantization.key.surface.total",
            Duration::from_millis(100),
            1,
        );
        result.record_duration("quantization.key.total", Duration::from_millis(100), 1);

        let report = result.report("Sampled-Selection Full Workflow Profile");

        let prepare_line = report
            .lines()
            .find(|line| line.contains("prepare_inputs"))
            .unwrap();
        let kernel_line = report
            .lines()
            .find(|line| line.contains("gpu.kernel"))
            .unwrap();
        let total_line = report
            .lines()
            .find(|line| line.contains("gpu.total"))
            .unwrap();

        assert!(prepare_line.contains("25.0%"), "wrong share:\n{report}");
        assert!(kernel_line.contains("75.0%"), "wrong share:\n{report}");
        assert!(total_line.contains("100.0%"), "wrong share:\n{report}");
    }
}
