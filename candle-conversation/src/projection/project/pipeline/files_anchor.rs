//! Where the knowledge base's files are placed in the system prompt: at the
//! `files` collection's declared position (`docs/provenance_system_collection.md`
//! §3) — after every section the schema declares ahead of it — and whether a
//! seed declared after it follows them.
//!
//! The files stand where the schema puts the collection, so moving an item
//! across it in the schema moves the files with it: a seed declared before the
//! collection is placed before the files, and its sealed K/V, computed with no
//! file ahead of it, is never what the prompt reads after them.

use std::collections::HashSet;

use super::*;

/// The collection the knowledge base's files are placed for.
pub(crate) const FILES_COLLECTION: &str = "files";

/// The `files` collection's place in a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FilesPosition {
    /// Every section id declared ahead of the collection: its sections and
    /// every variant of its trees.
    before: HashSet<SectionId>,
    /// Whether a seed tree is declared after the collection.
    pub seed_follows: bool,
}

/// The `files` collection's place in `schema`; `None` when it declares none.
pub(crate) fn files_position(schema: &Schema) -> Option<FilesPosition> {
    let items = &schema.system_prompt.items;
    let at = items.iter().position(
        |it| matches!(it, SystemPromptItem::Collection(c) if c.name == FILES_COLLECTION),
    )?;
    let mut before = HashSet::new();
    for item in &items[..at] {
        match item {
            SystemPromptItem::Section(s) => {
                before.insert(s.id);
            }
            SystemPromptItem::SectionTree(t) => before.extend(
                t.nodes
                    .iter()
                    .flat_map(|n| n.options.iter())
                    .flat_map(|o| o.variants.iter())
                    .map(|v| v.id),
            ),
            SystemPromptItem::Collection(c) => before.extend(c.sections.iter().map(|s| s.id)),
        }
    }
    let seed_follows = items[at + 1..].iter().any(|it| {
        matches!(it, SystemPromptItem::SectionTree(t)
            if t.nodes.iter().any(|n| n.name == SEED_SELECTOR))
    });
    Some(FilesPosition {
        before,
        seed_follows,
    })
}

impl FilesPosition {
    /// Where in `segments` the files are inserted: just after the last
    /// segment declared ahead of the collection. `None` when none of them is
    /// placed.
    pub(crate) fn insert_at(&self, segments: &[ProjectionSegment]) -> Option<usize> {
        segments
            .iter()
            .rposition(|seg| {
                matches!(seg, ProjectionSegment::Sealed(SealedKind::Section(s))
                    if self.before.contains(&s.id))
            })
            .map(|p| p + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::Builder;

    const LAYERS: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: c, user_prompt: c }
        assistant: { system_prompt: c, user_prompt: c }
    groups:
      - id: convo
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 4
"#;

    const OPEN: &str = "    - kind: section\n      id: files_open\n      content: \"open\"\n";
    const SEED: &str = "    - kind: section_tree\n      nodes:\n        - kind: optional\n          id: seed\n          content: \"seed\"\n          default: present\n";
    const FILES: &str = "    - kind: collection\n      name: files\n      selection: { kind: always_visible }\n      sections: []\n";
    const CLOSE: &str = "    - kind: section\n      id: close\n      content: \"close\"\n";

    fn builder(items: &[&str]) -> Builder {
        let yaml = format!("system_prompt:\n  items:\n{}{LAYERS}", items.concat());
        Builder::from_yaml(&yaml).unwrap()
    }

    fn section_id(b: &Builder, name: &str) -> SectionId {
        b.schema()
            .system_prompt
            .items
            .iter()
            .find_map(|it| match it {
                SystemPromptItem::Section(s) if s.name == name => Some(s.id),
                _ => None,
            })
            .unwrap()
    }

    fn seed_id(b: &Builder) -> SectionId {
        b.schema()
            .system_prompt
            .section_trees()
            .next()
            .unwrap()
            .default_present_ids[0]
    }

    fn segment(id: SectionId) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id)))
    }

    /// **A seed declared before the collection is placed before the files**:
    /// they go after it, and no seed follows them.
    #[test]
    fn files_go_after_a_seed_declared_before_them() {
        let b = builder(&[OPEN, SEED, FILES, CLOSE]);
        let at = files_position(b.schema()).unwrap();
        assert!(!at.seed_follows);
        let segments = [
            segment(section_id(&b, "files_open")),
            segment(seed_id(&b)),
            segment(section_id(&b, "close")),
        ];
        assert_eq!(at.insert_at(&segments), Some(2));
    }

    /// **A seed declared after the collection follows the files**: they go
    /// straight after the heading.
    #[test]
    fn files_go_before_a_seed_declared_after_them() {
        let b = builder(&[OPEN, FILES, SEED, CLOSE]);
        let at = files_position(b.schema()).unwrap();
        assert!(at.seed_follows);
        let segments = [
            segment(section_id(&b, "files_open")),
            segment(seed_id(&b)),
            segment(section_id(&b, "close")),
        ];
        assert_eq!(at.insert_at(&segments), Some(1));
    }

    /// **A schema without the collection places no files.**
    #[test]
    fn a_schema_without_the_collection_has_no_position() {
        assert!(files_position(builder(&[OPEN, CLOSE]).schema()).is_none());
    }
}
