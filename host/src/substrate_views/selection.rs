//! A group's selection rule, as the inspector names it.

use candle_conversation::projection::{SelectionRule, WorkingSetShare};

/// One-line description of a group's selection rule.
pub fn fmt_selection(rule: &SelectionRule) -> String {
    match rule {
        SelectionRule::AlwaysVisible => "all".to_string(),
        SelectionRule::TopK { k } => format!("top {k}"),
        SelectionRule::Single => "single".to_string(),
        SelectionRule::Named { selector } => format!("named({selector})"),
        SelectionRule::Offered { selector, k } => format!("offered({selector}, k={k})"),
        SelectionRule::Sequence {
            recent,
            historical_top_k,
            lineage_base,
        } => format!("recent {recent} + top {historical_top_k}, lineage from {lineage_base}"),
        SelectionRule::Birth { window } => format!("by birth, window {window}"),
        SelectionRule::WorkingSet { share } => match share {
            WorkingSetShare::Readings => "working set (readings)".to_string(),
            WorkingSetShare::Folders => "working set (folders)".to_string(),
            WorkingSetShare::Guides => "working set (guides)".to_string(),
            WorkingSetShare::Journals => "working set (journals)".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every selection rule renders as the inspector's one-liner.
    #[test]
    fn selection_rules_render_for_the_viewer() {
        assert_eq!(fmt_selection(&SelectionRule::AlwaysVisible), "all");
        assert_eq!(fmt_selection(&SelectionRule::TopK { k: 3 }), "top 3");
        assert_eq!(fmt_selection(&SelectionRule::Single), "single");
        assert_eq!(
            fmt_selection(&SelectionRule::Named {
                selector: "effort".into()
            }),
            "named(effort)"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::Offered {
                selector: "tools".into(),
                k: 4
            }),
            "offered(tools, k=4)"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::Sequence {
                recent: 16,
                historical_top_k: 8,
                lineage_base: 1000.0,
            }),
            "recent 16 + top 8, lineage from 1000"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::Birth { window: 73_728 }),
            "by birth, window 73728"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::WorkingSet {
                share: WorkingSetShare::Folders
            }),
            "working set (folders)"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::WorkingSet {
                share: WorkingSetShare::Readings
            }),
            "working set (readings)"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::WorkingSet {
                share: WorkingSetShare::Journals
            }),
            "working set (journals)"
        );
    }
}
