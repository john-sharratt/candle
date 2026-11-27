//! Which files of the knowledge base each kind holds
//! (`docs/provenance_system_collection.md` §4).
//!
//! The kinds are the working-set groups that once held readings — `scopes`
//! (readings), `guidance` (guides), `journals` (journal notes) and
//! `structure` (folders) — whose policies, shares and path rules are the
//! kinds' own. A folder kind holds the folder records; every other kind holds
//! file records, a drawing kind those its `draws` rule takes, and the kind
//! they draw from the rest: a file is placed through one kind or the other,
//! never both.

use super::schema::{GroupSchema, SelectionRule, WorkingSetShare};
use crate::files::{FileRecord, FileUnit};

/// The share a working-set group's members take, when it is one.
pub(crate) fn kind_share(group: &GroupSchema) -> Option<WorkingSetShare> {
    match group.selection {
        SelectionRule::WorkingSet { share } => Some(share),
        _ => None,
    }
}

/// The records of `records` that `kind` holds, `siblings` the other kinds of
/// its layer (any that draw from it take their files out of it).
pub(crate) fn kind_records<'a>(
    kind: &GroupSchema,
    siblings: &[&GroupSchema],
    records: &'a [FileRecord],
) -> Vec<&'a FileRecord> {
    let folders = kind_share(kind) == Some(WorkingSetShare::Folders);
    records
        .iter()
        .filter(|r| (r.unit == FileUnit::Folder) == folders)
        .filter(|r| match &kind.draws {
            Some(draws) => draws.takes(&r.path),
            None => !siblings.iter().any(|s| {
                s.id != kind.id
                    && s.draws
                        .as_ref()
                        .is_some_and(|d| d.from == kind.id && d.takes(&r.path))
            }),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::LineSpan;
    use crate::files::{ChunkRecord, MemberRef};
    use crate::persistence::content_hash::ContentHash;
    use crate::persistence::streams::ContentAddress;
    use crate::projection::schema::Draws;
    use crate::projection::Builder;

    fn record(path: &str, unit: FileUnit) -> FileRecord {
        let member = MemberRef {
            address: ContentAddress {
                prefix_hash: ContentHash::ZERO,
                section_hash: ContentHash::ZERO,
            },
            tokens: 1,
        };
        FileRecord {
            content_key: format!("v1:{path}@b"),
            path: path.into(),
            blob: "b".into(),
            unit,
            lines: 1,
            lang: "rs".into(),
            branches: vec!["main".into()],
            chunks: vec![ChunkRecord {
                lines: LineSpan { first: 1, last: 1 },
                member,
            }],
        }
    }

    fn group(builder: &Builder, name: &str) -> GroupSchema {
        let id = builder.id_for_group(name).unwrap();
        builder
            .schema()
            .layers
            .iter()
            .flat_map(|l| l.groups.iter())
            .find(|g| g.id == id)
            .unwrap()
            .clone()
    }

    /// **A file is placed through one kind or the other**: a guide under the
    /// guides' path goes to the drawing kind and leaves the readings, the
    /// hub file the rule excepts stays a reading, and a folder record goes
    /// only to the folder kind.
    #[test]
    fn a_file_is_held_by_one_kind_and_a_folder_by_the_folder_kind() {
        let records = vec![
            record("candle/zend/src/lib.rs", FileUnit::File),
            record("candle/mcp/src/guides/memory.md", FileUnit::File),
            record("candle/mcp/src/guides/README.md", FileUnit::File),
            record("candle/zend/src/", FileUnit::Folder),
        ];
        let builder = Builder::from_yaml(KINDS_YAML).unwrap();
        let scopes = group(&builder, "scopes");
        let guidance = group(&builder, "guidance");
        let structure = group(&builder, "structure");
        assert_eq!(
            guidance.draws,
            Some(Draws {
                from: scopes.id,
                paths: vec!["candle/mcp/src/guides/".into()],
                except: vec!["README.md".into()],
            })
        );
        let siblings = [&scopes, &guidance];
        let paths = |rs: Vec<&FileRecord>| -> Vec<String> {
            rs.into_iter().map(|r| r.path.clone()).collect()
        };
        assert_eq!(
            paths(kind_records(&scopes, &siblings, &records)),
            ["candle/zend/src/lib.rs", "candle/mcp/src/guides/README.md"]
        );
        assert_eq!(
            paths(kind_records(&guidance, &siblings, &records)),
            ["candle/mcp/src/guides/memory.md"]
        );
        assert_eq!(
            paths(kind_records(&structure, &[&structure], &records)),
            ["candle/zend/src/"]
        );
    }

    const KINDS_YAML: &str = r#"
system_prompt:
  sections:
    - id: alpha
      content: "alpha"
layers:
  - name: repo_map
    window: 4000
    score_formula: max
    budget:
      priority: 10
    summary: &summary
      turns:
        max_tokens: 256
        user: { system_prompt: c, user_prompt: c }
        assistant: { system_prompt: c, user_prompt: c }
    groups:
      - id: structure
        selection: { kind: working_set, share: folders }
  - name: code_reading
    window: 4000
    score_formula: max
    budget:
      priority: 20
    summary: *summary
    groups:
      - id: scopes
        selection: { kind: working_set, share: readings }
      - id: guidance
        draws:
          from: scopes
          paths: [candle/mcp/src/guides/]
          except: [README.md]
        selection: { kind: working_set, share: guides }
  - name: dialogue
    window: 4000
    score_formula: max
    budget:
      priority: 100
    summary: *summary
    groups:
      - id: convo
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 4
"#;
}
