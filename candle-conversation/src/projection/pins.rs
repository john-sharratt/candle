//! Which system-prompt items every selecting query attends whole
//! (`docs/hit_driven_projection.md` §5).
//!
//! A schema flags its pinned items `pin: true` — a section, a section-tree
//! node (every branch's variant of it), or a collection (every member, as it
//! stands at each projection) — and bounds them with `system_prompt:
//! pin_tokens:`. The projection marks each
//! emitted section with its [`Pin`], and the assembler turns the marked
//! sections' positions into the ranges the model attends outside its budget,
//! held to `pin_tokens`.

use std::collections::HashSet;

use super::ids::{CollectionId, SectionId};
use super::schema::{SystemPromptItem, SystemPromptSchema};

/// How an emitted section is pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Pin {
    /// Selected by score like everything else.
    #[default]
    No,
    /// Pinned whole: a pinned section or section-tree node.
    Whole,
    /// A member of a pinned collection. Members are pinned newest first, as
    /// far as `pin_tokens` allows after every whole pin; the rest are
    /// selected by score.
    Member,
}

/// The pinned items of one system prompt, and how many positions they may
/// pin together.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PinSet {
    sections: HashSet<SectionId>,
    collections: HashSet<CollectionId>,
    /// The most positions the pinned ranges may cover.
    pub tokens: usize,
}

impl PinSet {
    /// Resolve the flagged items' `names` against `sp`: each names a
    /// top-level section, a section-tree node, or a collection (top-level or a
    /// tree node's). A flag with a bound of zero is refused, as is a name the
    /// built prompt does not hold.
    pub fn resolve(
        names: &[String],
        tokens: usize,
        sp: &SystemPromptSchema,
    ) -> Result<Self, String> {
        let mut set = Self {
            tokens,
            ..Self::default()
        };
        if names.is_empty() {
            return Ok(set);
        }
        if tokens == 0 {
            return Err(
                "system_prompt items are flagged `pin: true` but system_prompt.pin_tokens is 0"
                    .into(),
            );
        }
        for name in names {
            if !set.add(name, sp) {
                return Err(format!(
                    "`pin: true` on {name:?}, which the built system prompt holds no section, \
                     section-tree node or collection of"
                ));
            }
        }
        Ok(set)
    }

    fn add(&mut self, name: &str, sp: &SystemPromptSchema) -> bool {
        let mut found = false;
        for item in &sp.items {
            match item {
                SystemPromptItem::Section(s) if s.name == name => {
                    self.sections.insert(s.id);
                    found = true;
                }
                SystemPromptItem::Collection(c) if c.name == name => {
                    self.collections.insert(c.id);
                    found = true;
                }
                SystemPromptItem::SectionTree(t) => {
                    for node in &t.nodes {
                        if node.name == name {
                            self.sections.extend(
                                node.options
                                    .iter()
                                    .flat_map(|o| o.variants.iter().map(|v| v.id)),
                            );
                            found = true;
                        }
                        if let Some(tc) = &node.collection {
                            if tc.collection.name == name {
                                self.collections.insert(tc.collection.id);
                                found = true;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        found
    }

    /// Whether nothing is pinned.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty() && self.collections.is_empty()
    }

    /// How section `id` of `sp` is pinned. A pinned collection's members are
    /// read from `sp` as it stands, so a member added since the schema was
    /// built — a journal entry — is pinned too.
    pub fn pin_of(&self, sp: &SystemPromptSchema, id: SectionId) -> Pin {
        if self.sections.contains(&id) {
            return Pin::Whole;
        }
        if self.collections.is_empty() {
            return Pin::No;
        }
        let member = sp.items.iter().any(|item| match item {
            SystemPromptItem::Collection(c) => {
                self.collections.contains(&c.id)
                    && (c.sections.iter().any(|s| s.id == id) || c.summary_section == Some(id))
            }
            SystemPromptItem::SectionTree(t) => t.nodes.iter().any(|n| {
                n.collection.as_ref().is_some_and(|tc| {
                    self.collections.contains(&tc.collection.id)
                        && (tc.variants.iter().flatten().any(|v| v.id == id)
                            || tc.collection.sections.iter().any(|s| s.id == id))
                })
            }),
            _ => false,
        });
        if member {
            Pin::Member
        } else {
            Pin::No
        }
    }
}

/// Which of `pieces` — `(pin, tokens)` in emission order — are pinned within
/// `budget` positions: every whole pin first, in order, each that fits; then
/// members newest first (from the end) until one does not fit, so the pinned
/// members are always the newest run. Returns one flag per piece.
pub fn within_budget(pieces: &[(Pin, usize)], budget: usize) -> Vec<bool> {
    let mut keep = vec![false; pieces.len()];
    let mut used = 0usize;
    for (i, &(pin, tokens)) in pieces.iter().enumerate() {
        if pin == Pin::Whole && used + tokens <= budget {
            keep[i] = true;
            used += tokens;
        }
    }
    for (i, &(pin, tokens)) in pieces.iter().enumerate().rev() {
        if pin != Pin::Member {
            continue;
        }
        if used + tokens > budget {
            break;
        }
        keep[i] = true;
        used += tokens;
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::{within_budget, Pin};

    /// Whole pins come first, in order, each that fits; then members newest
    /// first until one does not fit — an older member never jumps a newer one.
    #[test]
    fn the_budget_takes_whole_pins_then_the_newest_members() {
        let pieces = [
            (Pin::Whole, 300),  // persona
            (Pin::No, 5000),    // grounding
            (Pin::Whole, 120),  // journal intro
            (Pin::Member, 200), // oldest entry
            (Pin::Member, 250),
            (Pin::Member, 400), // newest entry
            (Pin::No, 9000),    // tools
        ];
        assert_eq!(
            within_budget(&pieces, 1024),
            vec![true, false, true, false, false, true, false]
        );
        assert_eq!(
            within_budget(&pieces, 1300),
            vec![true, false, true, true, true, true, false]
        );
        assert_eq!(
            within_budget(&pieces, 1100),
            vec![true, false, true, false, true, true, false],
            "the oldest member would fit alone, but the run stops at the first miss"
        );
        assert_eq!(
            within_budget(&pieces, 100),
            vec![false, false, false, false, false, false, false]
        );
        assert_eq!(
            within_budget(&pieces, 200),
            vec![false, false, true, false, false, false, false],
            "a whole pin that does not fit leaves room for a smaller one"
        );
    }
}
