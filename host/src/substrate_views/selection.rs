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
        } => format!("recent {recent} + top {historical_top_k}"),
        SelectionRule::WorkingSet { share } => match share {
            WorkingSetShare::Folders => "working set (folders)".to_string(),
            WorkingSetShare::Remainder => "working set (files)".to_string(),
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
            }),
            "recent 16 + top 8"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::WorkingSet {
                share: WorkingSetShare::Folders
            }),
            "working set (folders)"
        );
        assert_eq!(
            fmt_selection(&SelectionRule::WorkingSet {
                share: WorkingSetShare::Remainder
            }),
            "working set (files)"
        );
    }
}
