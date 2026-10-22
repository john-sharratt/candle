//! What a layer unit's conversation carries, and the key it is committed
//! under.
//!
//! Every unit records its layer ([`LAYER_KEY`]) and subject ([`SUBJECT_KEY`])
//! when its conversation is minted, and the rest — the key it commits under,
//! the unit it built on, the files it depends on — once it has answered. The
//! content key ([`CONTENT_KEY`]) is the commit, as it is for `code_reading`'s
//! files: a conversation without one never finished.
//!
//! **A unit's key names its layer and the unit it was built on**, so no two
//! layers ever share one — the content keys, the leases and the retrieval index
//! are all keyed by it, and a layer above `code_reading` keyed by the file
//! would read as already done.

use sha2::{Digest, Sha256};

pub use crate::branch_ingest::keys::{BRANCHES_KEY, COMMIT_KEY, CONTENT_KEY, QUANT_KEY};

/// The layer a unit belongs to.
pub const LAYER_KEY: &str = "layer";

/// What the unit is of: a file's workspace-relative path, or a day.
pub const SUBJECT_KEY: &str = "subject";

/// The key of the unit it was built on — the unit one layer down for the same
/// file or day, or the digest of the day's dialogue.
pub const AFTER_KEY: &str = "after";

/// The `code_reading` key of the file a file unit is of — what retrieval
/// scopes it by: a branch whose tree holds that file version holds this unit.
pub const FILE_KEY: &str = "file_key";

/// The files the unit's own file depends on, as `path\tkey` lines: the
/// version of each it was built against. A unit whose dependency has moved on
/// is stale.
pub const DEPS_KEY: &str = "deps";

/// A digest of what the layer asks its units — its question, its tools, its
/// thinking, whether it names edges. A unit built under another is stale: a
/// layer whose question changes is asked again.
pub const SPEC_KEY: &str = "spec";

/// The key the unit was leased under. An edges layer commits under a key of
/// what it depended on, known only once it has answered, so a member that
/// waited on the lease finds the unit committed by this one.
pub const JOB_KEY: &str = "job";

/// Bumped when what a unit of every layer means changes, so each is built
/// again.
pub const UNIT_KEY_VERSION: u32 = 1;

/// The content key of `layer`'s unit built on the unit `after`.
pub fn unit_key(layer: &str, after: &str) -> String {
    let digest = Sha256::digest(after.as_bytes());
    let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
    format!("u{UNIT_KEY_VERSION}:{layer}:{hex}")
}

/// [`SPEC_KEY`]'s value for `spec`.
pub fn spec_digest(spec: &super::spec::LayerSpec) -> String {
    let mut h = Sha256::new();
    h.update(
        format!(
            "{UNIT_KEY_VERSION}\n{:?}\n{}\n{}\n",
            spec.unit, spec.effort, spec.edges
        )
        .as_bytes(),
    );
    h.update(spec.tools.join(",").as_bytes());
    h.update(b"\n");
    h.update(spec.prompt.as_bytes());
    h.finalize()[..12]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// [`DEPS_KEY`]'s value for `deps`, sorted so the same set is written alike.
pub fn deps_value(deps: &[(String, String)]) -> String {
    let mut lines: Vec<String> = deps.iter().map(|(p, k)| format!("{p}\t{k}")).collect();
    lines.sort();
    lines.dedup();
    lines.join("\n")
}

/// The `(path, key)` pairs [`deps_value`] wrote.
pub fn parse_deps(value: &str) -> Vec<(String, String)> {
    value
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(p, k)| (p.to_string(), k.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_units_key_names_its_layer_and_differs_by_what_it_built_on() {
        let a = unit_key("static_analysis", "v2:candle/a.rs@1");
        assert!(a.starts_with("u1:static_analysis:"));
        assert_ne!(a, unit_key("static_analysis", "v2:candle/a.rs@2"));
        assert_ne!(a, unit_key("critical_analysis", "v2:candle/a.rs@1"));
        assert_eq!(a, unit_key("static_analysis", "v2:candle/a.rs@1"));
    }

    #[test]
    fn deps_round_trip_sorted_and_deduplicated() {
        let deps = vec![
            ("candle/b.rs".to_string(), "k2".to_string()),
            ("candle/a.rs".to_string(), "k1".to_string()),
            ("candle/b.rs".to_string(), "k2".to_string()),
        ];
        let v = deps_value(&deps);
        assert_eq!(v, "candle/a.rs\tk1\ncandle/b.rs\tk2");
        assert_eq!(
            parse_deps(&v),
            vec![
                ("candle/a.rs".to_string(), "k1".to_string()),
                ("candle/b.rs".to_string(), "k2".to_string())
            ]
        );
        assert!(parse_deps("").is_empty());
    }
}
