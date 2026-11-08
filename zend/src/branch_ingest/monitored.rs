//! The branches the ingest reads: a whitelist by name, in
//! `prompts/ingest_branches.yaml` (`docs/zend_branch_ingest.md` §3.1).
//!
//! A repository's record branches are narrowed to these before its trees are
//! walked, so a branch off the list contributes no unit — and every reading
//! only such a branch held is planned as gone and tombstoned.

use std::sync::OnceLock;

use serde::Deserialize;
use zend_vfs::RecordBranch;

/// The whitelist, as shipped.
const LIST: &str = include_str!("../prompts/ingest_branches.yaml");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BranchList {
    branches: Vec<String>,
}

/// The names of the branches the ingest reads.
pub fn monitored() -> &'static [String] {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| parse(LIST))
}

fn parse(text: &str) -> Vec<String> {
    serde_yaml::from_str::<BranchList>(text)
        .expect("ingest_branches.yaml parses")
        .branches
}

/// `tips` less every branch `names` does not name, in their order.
pub fn keep_monitored(tips: &mut Vec<RecordBranch>, names: &[String]) {
    tips.retain(|t| names.iter().any(|n| n == t.name.as_str()));
}

#[cfg(test)]
mod tests {
    use zend_vfs::{BranchName, Oid};

    use super::*;

    fn tip(name: &str) -> RecordBranch {
        RecordBranch {
            name: BranchName::parse(name).unwrap(),
            tip: Oid::parse("1234567890123456789012345678901234567890").unwrap(),
        }
    }

    fn names(tips: &[RecordBranch]) -> Vec<&str> {
        tips.iter().map(|t| t.name.as_str()).collect()
    }

    /// **The shipped list is the default branches alone.**
    #[test]
    fn the_shipped_list_names_main_and_master() {
        assert_eq!(monitored(), ["main", "master"]);
    }

    /// **Only a listed branch is kept, matched by its whole name** — a branch
    /// whose name only starts with a listed one, or nests under it, is not.
    #[test]
    fn only_a_listed_branch_is_kept() {
        let mut tips = vec![
            tip("main"),
            tip("main-old"),
            tip("main/topic"),
            tip("zen/challenge-r3-value-of"),
            tip("master"),
        ];
        keep_monitored(&mut tips, &["main".to_string(), "master".to_string()]);
        assert_eq!(names(&tips), ["main", "master"]);
    }

    /// **An empty list keeps nothing**, so a repository holding none of the
    /// listed branches has nothing read.
    #[test]
    fn an_empty_list_keeps_no_branch() {
        let mut tips = vec![tip("main"), tip("topic")];
        keep_monitored(&mut tips, &[]);
        assert!(tips.is_empty());
    }

    /// **A key the list does not know is refused**, so a misspelt key cannot
    /// leave the whitelist silently empty.
    #[test]
    fn an_unknown_key_is_refused() {
        let parsed = serde_yaml::from_str::<BranchList>("branch:\n  - main\n");
        assert!(parsed.is_err());
        assert_eq!(
            parse("branches:\n  - main\n  - zen/memory-and-guides\n"),
            ["main", "zen/memory-and-guides"]
        );
    }
}
