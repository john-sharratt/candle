//! The layers whose units are the knowledge base's records, as the substrate
//! inspector shows them: `repo_map`'s folders and `code_reading`'s files, one
//! folder at a time (`host::substrate_views::path_level`).
//!
//! Both layers seal their units as members of the system prompt's `files`
//! collection, not as conversations, so a layer's own timelines are none. What
//! the layer holds is the records of its kind. The overview counts them from
//! the keys alone; a folder's level reads the records under it and no others.

use candle_conversation::files::names::chunk_name;
use candle_conversation::files::{files_owner, FileRecord, FileUnit};
use candle_conversation::projection::Conversation;
use host::substrate_views::{path_level, PathLevel, PathUnit, PathUnitView};

use super::records;

/// The kind of record `layer` holds: the folders' listings for `repo_map`, the
/// files for `code_reading`, none for any other layer.
pub fn unit_of_layer(layer: &str) -> Option<FileUnit> {
    match layer {
        "repo_map" => Some(FileUnit::Folder),
        "code_reading" => Some(FileUnit::File),
        _ => None,
    }
}

/// How many records of `unit` the knowledge base holds, from their keys.
pub fn count(conversation: &Conversation, unit: FileUnit) -> usize {
    records::count(conversation, unit == FileUnit::Folder)
}

/// The folder `at` of the records of `unit` (`""` for the repositories): its
/// own listing, its folders with what lies below each, its files with their
/// versions.
pub fn level(conversation: &Conversation, unit: FileUnit, at: &str) -> PathLevel {
    let below = records::below(conversation, at);
    let read = conversation.read();
    let units: Vec<PathUnit> = below
        .into_iter()
        .filter(|r| r.unit == unit)
        .map(|r| {
            // Placed once this process holds its last section, as
            // `records::placeable` asks — under the one read.
            let placed = r.chunks.last().is_some_and(|last| {
                read.owned_section_named(files_owner(), &chunk_name(&r.path, &r.blob, last.lines))
                    .is_some()
            });
            PathUnit {
                unit: PathUnitView {
                    blob: r.blob.clone(),
                    branches: r.branches.clone(),
                    lines: r.lines,
                    sections: r.chunks.len(),
                    tokens: record_tokens(&r),
                    placed,
                },
                path: r.path,
            }
        })
        .collect();
    path_level(at, units)
}

fn record_tokens(r: &FileRecord) -> usize {
    r.chunks.iter().map(|c| c.member.tokens as usize).sum()
}

#[cfg(test)]
mod tests {
    use candle_conversation::files::{ChunkRecord, LineSpan, MemberRef};
    use candle_conversation::persistence::content_hash::ContentHash;
    use candle_conversation::persistence::streams::ContentAddress;

    use super::*;

    fn record(path: &str, unit: FileUnit, tokens: &[u32]) -> FileRecord {
        let chunks = tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| ChunkRecord {
                lines: LineSpan {
                    first: i as u32 * 10 + 1,
                    last: i as u32 * 10 + 10,
                },
                member: MemberRef {
                    address: ContentAddress {
                        prefix_hash: ContentHash { lo: 1, hi: 0 },
                        section_hash: ContentHash {
                            lo: i as u64 + 2,
                            hi: 0,
                        },
                    },
                    tokens: t,
                },
            })
            .collect();
        FileRecord {
            content_key: format!("v1:{path}@b"),
            path: path.into(),
            blob: "b".into(),
            unit,
            lines: 10 * tokens.len() as u32,
            lang: "rs".into(),
            branches: vec!["main".into()],
            chunks,
        }
    }

    /// The two layers read their own kind of record, and no other layer reads
    /// any.
    #[test]
    fn each_layer_holds_its_own_kind() {
        assert_eq!(unit_of_layer("repo_map"), Some(FileUnit::Folder));
        assert_eq!(unit_of_layer("code_reading"), Some(FileUnit::File));
        assert_eq!(unit_of_layer("dialogue"), None);
    }

    /// **A layer's level holds its kind of record alone**: a file's tokens the
    /// sum of its sections', placed once its last section is named here, and
    /// the folder's own listing left to `repo_map`.
    #[test]
    fn a_layers_level_holds_its_own_records() {
        let conversation = Conversation::new();
        records::put(
            &conversation,
            &record("candle/src/a.rs", FileUnit::File, &[30, 12]),
        )
        .unwrap();
        records::put(
            &conversation,
            &record("candle/src/", FileUnit::Folder, &[5]),
        )
        .unwrap();
        records::put(&conversation, &record("candle/b.rs", FileUnit::File, &[3])).unwrap();
        let section = conversation.allocate_owned_section().expect("an owned id");
        conversation.register_owned_section(files_owner(), "candle/src/a.rs@b#11-20", section);

        assert_eq!(count(&conversation, FileUnit::File), 2);
        assert_eq!(count(&conversation, FileUnit::Folder), 1);

        let top = level(&conversation, FileUnit::File, "");
        assert_eq!(
            (
                top.children[0].path.as_str(),
                top.children[0].unit_count,
                top.children[0].tokens
            ),
            ("candle/", 2, 45)
        );

        let src = level(&conversation, FileUnit::File, "candle/src/");
        assert!(src.units.is_empty(), "the folder's listing is repo_map's");
        let a = &src.children[0];
        assert_eq!(
            (
                a.path.as_str(),
                a.units[0].sections,
                a.units[0].tokens,
                a.units[0].placed
            ),
            ("candle/src/a.rs", 2, 42, true)
        );

        let listing = level(&conversation, FileUnit::Folder, "candle/src/");
        assert_eq!((listing.units.len(), listing.units[0].placed), (1, false));
        assert!(listing.children.is_empty());
    }
}
