//! The knowledge base's mandatory files (`docs/placed_section_layout.md`
//! §3.5b): named by workspace path — each repository's root listing and its
//! anchor documents — and placed whole in every free knowledge base the seed
//! dial leaves on, inside the range and outside the competition.
//!
//! A path can have a record per version the branches hold; the one placed is
//! the version most branches hold, ties to the lower content key, so every
//! projection picks the same one.

use super::record::FileRecord;

/// The record placed for each of `paths` that `records` holds one of, in
/// `paths`' order.
pub fn mandatory_records<'a>(records: &'a [FileRecord], paths: &[String]) -> Vec<&'a FileRecord> {
    paths
        .iter()
        .filter_map(|path| {
            records.iter().filter(|r| &r.path == path).min_by(|a, b| {
                b.branches
                    .len()
                    .cmp(&a.branches.len())
                    .then_with(|| a.content_key.cmp(&b.content_key))
            })
        })
        .collect()
}

/// Whether `mandatory` — the mandatory files' records — leaves room for the
/// rest in a range of `positions`: at most half of it, so provenance always
/// has the other half to place what the question reaches.
pub fn leaves_room(mandatory: &[&FileRecord], positions: usize) -> bool {
    let tokens: usize = mandatory.iter().map(|r| r.tokens() as usize).sum();
    tokens * 2 <= positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{ChunkRecord, FileUnit, LineSpan, MemberRef};
    use crate::persistence::content_hash::ContentHash;
    use crate::persistence::streams::ContentAddress;

    fn record(path: &str, blob: &str, branches: &[&str], tokens: u32) -> FileRecord {
        FileRecord {
            content_key: format!("v1:{path}@{blob}"),
            path: path.into(),
            blob: blob.into(),
            unit: FileUnit::File,
            lines: 1,
            lang: "md".into(),
            branches: branches.iter().map(|b| b.to_string()).collect(),
            chunks: vec![ChunkRecord {
                lines: LineSpan { first: 1, last: 1 },
                member: MemberRef {
                    address: ContentAddress {
                        prefix_hash: ContentHash::ZERO,
                        section_hash: ContentHash::ZERO,
                    },
                    tokens,
                },
            }],
        }
    }

    /// **Each path's record is the version most branches hold**, ties to the
    /// lower content key, in the paths' order; a path with no record is left
    /// out.
    #[test]
    fn each_path_places_the_version_most_branches_hold() {
        let records = [
            record("r/README.md", "b", &["dev"], 10),
            record("r/README.md", "a", &["main", "dev"], 10),
            record("r/", "c", &["main"], 5),
            record("r/", "d", &["dev"], 5),
            record("r/src/x.rs", "e", &["main"], 99),
        ];
        let paths = vec![
            "r/".to_string(),
            "r/README.md".to_string(),
            "r/AGENTS.md".to_string(),
        ];
        let keys: Vec<&str> = mandatory_records(&records, &paths)
            .iter()
            .map(|r| r.content_key.as_str())
            .collect();
        assert_eq!(keys, ["v1:r/@c", "v1:r/README.md@a"]);
    }

    /// **The mandatory files may take half the range, no more.**
    #[test]
    fn the_mandatory_files_take_at_most_half_the_range() {
        let a = record("r/a.md", "a", &[], 300);
        let b = record("r/b.md", "b", &[], 200);
        assert!(leaves_room(&[&a, &b], 1_000));
        assert!(!leaves_room(&[&a, &b], 999));
        assert!(leaves_room(&[], 0));
    }
}
