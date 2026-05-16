//! Probe quality tests: verifies that intra-tool BDP scores degrade as probe
//! certainty decreases from positive → boundary → negative.
//!
//! Positive probes are unambiguous tool-use queries.
//! Boundary probes are edge cases where the tool may or may not be appropriate.
//! Negative probes are queries where the tool is mentioned but explicitly rejected.
//!
//! These tests use mean-agreement (not max-agreement) because the max formula
//! saturates at 128 across case types, masking genuine signal differences.

use crate::corpus::load_fixtures;
use crate::harness::Harness;

/// A positive probe yields a higher intra-tool mean score than a boundary probe
/// for the same tool, confirming that BDP captures intent certainty.
#[test]
fn positive_probe_scores_above_boundary() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();

    let cases = [
        ("weather",    "weather_pos_2",    "weather_bnd_0"),
        ("calculator", "calculator_pos_2", "calculator_bnd_0"),
        ("code_run",   "code_run_pos_2",   "code_run_bnd_0"),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (tool, pos_id, bnd_id) in &cases {
        let pos_resolver = h.scan(&pf, &manifest, pos_id);
        let bnd_resolver = h.scan(&pf, &manifest, bnd_id);

        let pos_score = h.section_score_formula(&pos_resolver, tool, false);
        let bnd_score = h.section_score_formula(&bnd_resolver, tool, false);

        if pos_score <= bnd_score {
            failures.push(format!(
                "{tool}: positive({pos_id})={pos_score:.2}  boundary({bnd_id})={bnd_score:.2}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "positive probe (mean) did not outscore boundary probe:\n  {}",
        failures.join("\n  ")
    );
}

/// A positive probe yields a higher intra-tool mean score than a negative probe
/// for the same tool.  Negative probes mention the tool but reject it, so
/// they should carry the weakest BDP alignment with the tool corpus.
#[test]
fn positive_probe_scores_above_negative() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();

    let cases = [
        ("weather",    "weather_pos_2",    "weather_neg_0"),
        ("calculator", "calculator_pos_2", "calculator_neg_0"),
        ("code_run",   "code_run_pos_2",   "code_run_neg_0"),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (tool, pos_id, neg_id) in &cases {
        let pos_resolver = h.scan(&pf, &manifest, pos_id);
        let neg_resolver = h.scan(&pf, &manifest, neg_id);

        let pos_score = h.section_score_formula(&pos_resolver, tool, false);
        let neg_score = h.section_score_formula(&neg_resolver, tool, false);

        if pos_score <= neg_score {
            failures.push(format!(
                "{tool}: positive({pos_id})={pos_score:.2}  negative({neg_id})={neg_score:.2}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "positive probe (mean) did not outscore negative probe:\n  {}",
        failures.join("\n  ")
    );
}
