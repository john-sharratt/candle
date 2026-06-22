//! Tests for the dialogue layer's `tools` collection:
//! per-tool BDP discrimination and projection top-3 selection.

use candle_conversation::projection::{Reserved, SectionId};

use crate::corpus::{load_fixtures, TOOLS};
use crate::harness::Harness;

/// For each of the 8 tools, probe with `_pos_1` (kept out of corpus) and
/// verify the intra-tool section score is strictly above the mean inter-tool score.
#[test]
fn intra_tool_score_exceeds_inter_tool_mean() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let resolvers = h.scan_all_pos1(&pf, &manifest);
    let mut failures: Vec<String> = Vec::new();

    for (probe_tool, resolver) in &resolvers {
        let intra = h.section_score(resolver, probe_tool);
        let inter_scores: Vec<f32> = TOOLS
            .iter()
            .filter(|&&t| t != *probe_tool)
            .map(|&t| h.section_score(resolver, t))
            .collect();
        let inter_mean = inter_scores.iter().sum::<f32>() / inter_scores.len() as f32;

        if intra <= inter_mean {
            failures.push(format!(
                "{probe_tool}: intra={intra:.2}  inter_mean={inter_mean:.2}  ratio={:.2}",
                if inter_mean > 0.0 {
                    intra / inter_mean
                } else {
                    f32::INFINITY
                }
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "intra-tool BDP score ≤ inter-tool mean for:\n  {}",
        failures.join("\n  ")
    );
}

/// For each tool, the production projection pipeline selects that tool's
/// section inside the dialogue layer's `top_k=3 tools` collection.
#[test]
fn projection_top3_includes_probe_tool() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let target = h.target();
    let resolvers = h.scan_all_pos1(&pf, &manifest);
    let mut failures: Vec<String> = Vec::new();

    for (probe_tool, resolver) in &resolvers {
        let projection = h.builder.project(target, resolver);
        let emitted = h.emitted_tools(&projection);

        if !emitted.contains(probe_tool) {
            failures.push(format!(
                "{probe_tool}: not in emitted top-3; got {emitted:?}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "probe tool not selected by projection:\n  {}",
        failures.join("\n  ")
    );
}

/// With 8 tools and a top-k smaller than 8, every projection drops at least one
/// tool — so the catalog summary section must be emitted, and emitted *before*
/// any tool section (it is a compact overview of everything, including what was
/// dropped).
#[test]
fn partial_selection_emits_summary_before_tools() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let target = h.target();
    let summary_id = SectionId::reserved(Reserved::ToolSummary);
    let tool_ids: std::collections::HashSet<SectionId> =
        h.tool_section_ids.values().copied().collect();

    let mut failures: Vec<String> = Vec::new();
    for (probe_tool, resolver) in &h.scan_all_pos1(&pf, &manifest) {
        let projection = h.builder.project(target, resolver);
        let ids: Vec<SectionId> = projection.sealed_sections().map(|rs| rs.id).collect();

        // Sanity: the selection is a proper subset of the 8 tools.
        let n_tools = ids.iter().filter(|i| tool_ids.contains(i)).count();
        assert!(
            n_tools < TOOLS.len(),
            "{probe_tool}: expected partial selection, got all {n_tools}",
        );

        let summary_pos = ids.iter().position(|&i| i == summary_id);
        let first_tool_pos = ids.iter().position(|i| tool_ids.contains(i));
        match (summary_pos, first_tool_pos) {
            (Some(sp), Some(tp)) if sp < tp => {}
            (Some(sp), Some(tp)) => {
                failures.push(format!("{probe_tool}: summary at {sp}, first tool at {tp}"))
            }
            (None, _) => failures.push(format!("{probe_tool}: summary section not emitted")),
            (_, None) => failures.push(format!("{probe_tool}: no tools emitted")),
        }
    }
    assert!(
        failures.is_empty(),
        "summary-before-tools failed:\n  {}",
        failures.join("\n  ")
    );
}
