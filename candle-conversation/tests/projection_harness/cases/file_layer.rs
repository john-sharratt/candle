//! Fine-discrimination tests for the two file tools (file_read vs file_write).
//!
//! These tools are semantically similar — both reference file operations — so
//! correctly separating them is a harder test of BDP signal quality than the
//! eight-way discrimination in tools_layer.

use crate::corpus::load_fixtures;
use crate::harness::Harness;

/// A `file_read` probe scores higher on the `file_read` corpus than on
/// the `file_write` corpus, despite the semantic proximity of the two tools.
#[test]
fn file_read_probe_prefers_file_read_over_file_write() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let resolver = h.scan(&pf, &manifest, "file_read_pos_1");

    let read_score = h.section_score(&resolver, "file_read");
    let write_score = h.section_score(&resolver, "file_write");

    assert!(
        read_score > write_score,
        "file_read probe: read={read_score:.2}  write={write_score:.2}  \
         expected read > write"
    );
}

/// A `file_write` probe scores higher on the `file_write` corpus than on
/// the `file_read` corpus.
#[test]
fn file_write_probe_prefers_file_write_over_file_read() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();
    let resolver = h.scan(&pf, &manifest, "file_write_pos_1");

    let write_score = h.section_score(&resolver, "file_write");
    let read_score = h.section_score(&resolver, "file_read");

    assert!(
        write_score > read_score,
        "file_write probe: write={write_score:.2}  read={read_score:.2}  \
         expected write > read"
    );
}

/// Both file tools outperform the mean score of the six non-file tools when
/// probed with their respective `_pos_1` scenarios.
#[test]
fn file_tools_score_above_non_file_mean() {
    let (manifest, pf) = load_fixtures();
    let h = Harness::build();

    const NON_FILE: &[&str] = &["weather", "web_search", "code_run", "datetime", "calculator", "random"];

    for tool in &["file_read", "file_write"] {
        let probe_id = format!("{tool}_pos_1");
        let resolver = h.scan(&pf, &manifest, &probe_id);

        let intra = h.section_score(&resolver, tool);
        let non_file_scores: Vec<f32> = NON_FILE
            .iter()
            .map(|&t| h.section_score(&resolver, t))
            .collect();
        let non_file_mean = non_file_scores.iter().sum::<f32>() / non_file_scores.len() as f32;

        assert!(
            intra > non_file_mean,
            "{tool} probe: intra={intra:.2}  non_file_mean={non_file_mean:.2}  \
             expected intra > non_file_mean"
        );
    }
}
