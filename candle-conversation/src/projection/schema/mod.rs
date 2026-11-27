//! The static schema of a projection: what content may exist in which layer and
//! group, declared once and frozen after construction.
//!
//! A schema is parsed and validated once. Content is appended into its groups
//! over time, but the schema itself never changes, and everything dynamic, such
//! as token counts and scores, comes from the content resolver at projection
//! time. The one system prompt is shared by every target, so its sealed K/V is
//! reused across layers, and a layer differs only by the branch its dials select.
//! Budgets are flexbox-style: priorities set each node's ideal share, and minimum
//! and maximum percentages bound it.
//!
//! - `group`: a group's gates, budget and draw rules.
//! - `layer`: a cognitive layer, its window, rank and decode priority.
//! - `reserved_range`: a placed section's fixed range of positions.
//! - `selection`: the rule that picks a group's surviving turns.
//! - `summary`: the compression prompts for the summary tree.
//! - `system_prompt`: the shared prompt, its collections and selector trees.
//! - `tests_group`: tests for the group rules.

use std::num::NonZeroUsize;
use std::ops::Range;

use super::adaptive::{AnchorConfig, BudgetAdaptive, LocalityConfig, MemberBudgetAdaptive};
use super::hits::HitsConfig;
use super::ids::{CollectionId, GroupId, LayerId, SectionId};
use super::pins::PinSet;
use super::policy::{PolicyConfig, SelectionPolicy};
use crate::summary_tree::scope::Scope;
use crate::working_set::WorkingSetConfig;

/// The complete parsed, validated schema. Immutable after construction.
///
/// The **single** [`system_prompt`](Self::system_prompt) is shared by every
/// projection target — its framing, tool catalog, and section-tree dials are
/// declared once, so the sealed system-prompt K/V is reused across all layers.
/// A layer differs only by the section-tree branch it selects via its
/// [`LayerSchema::dials`]. Each layer still carries its own `window` (per-target
/// turn budget) and turn groups.
///
/// The schema does **not** carry dialect-specific structural tokens
/// (turn-boundary markers, role openers/closers).  Those are runtime
/// concerns owned by the scheduler / projection assembler — see
/// `crate::scheduler::projection_assembler::BoundaryMarkers`.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Ordered: layer 0 first. Order is meaningful for masking and emission.
    pub layers: Vec<LayerSchema>,
    /// The one system prompt every projection emits, framed by the target
    /// layer's [`dials`](LayerSchema::dials).
    pub system_prompt: SystemPromptSchema,
    /// Turn every repetition penalty off for the duration of a tool call.
    ///
    /// # Why this is opt-in
    ///
    /// It is right for one kind of caller and badly wrong for the other, and
    /// which one you are is a property of the schema rather than of the engine.
    ///
    /// An assistant's tool arguments are *quotations*: file paths, identifiers,
    /// numbers, names lifted from the prompt or an earlier span. Penalising a
    /// token because it appeared before is exactly wrong there — the value is
    /// only correct if it repeats. That caller wants this on.
    ///
    /// A character's tool arguments are the opposite. `say` takes what the
    /// character means; the argument *is* the prose, and it is the one span in
    /// the turn where repetition control matters most. With this on, a cast
    /// decoded its every utterance with presence, frequency, repeat and DRY all
    /// switched off — and one character said the same sentence, word for word,
    /// for a hundred turns.
    ///
    /// Off by default: the penalties a caller configured are the ones it gets.
    /// A schema that needs verbatim arguments asks for the exemption by name.
    pub free_tool_calls_from_penalties: bool,
    /// How a selecting model's hits are counted and how they move a placed
    /// piece's score (`hits:`, [`HitsConfig`]).
    pub hits: HitsConfig,
    /// A projection held to the target's own conversation: the system prompt
    /// and the target group's turns. Nothing is gathered from another layer,
    /// the working set is left out, and no other layer is scanned, so the
    /// projection teaches nothing back (no momentum, no backgrounds).
    ///
    /// **Not a YAML field.** Set at runtime on a copy of a builder by
    /// [`super::Builder::focused`], for a question put beside a conversation
    /// about the conversation itself — a journal draft — where what the
    /// conversation retrieved from elsewhere is only distraction, and placed
    /// reading would be taken for its own history.
    pub focus: bool,
}

impl Schema {
    /// Every `SectionId` (as raw `u32`) the schema allocates: the unified
    /// system-prompt sections and section-tree variants, every system-prompt
    /// collection's compression-summary prompts, and each layer's own
    /// `turns`/`summaries` compression prompts.
    ///
    /// Runtime section-id allocation
    /// ([`super::Builder::add_section_to_collection`]) must stay disjoint from all
    /// of these — the compression-prompt sections never appear in
    /// `system_prompt.items`, so a max over only the *visible* sections would
    /// alias them.
    pub fn all_section_ids(&self) -> Vec<u32> {
        // `all_section_ids()` (not `all_sections()`) so section-tree variant ids
        // are counted too — `all_sections()` skips them.
        let mut ids: Vec<u32> = self
            .system_prompt
            .all_section_ids()
            .map(|id| id.raw())
            .collect();
        for layer in &self.layers {
            ids.extend(layer.all_section_ids());
        }
        ids
    }
}

mod group;
mod item_slot;
mod layer;
mod reserved_range;
mod selection;
mod summary;
mod system_prompt;
#[cfg(test)]
mod tests_group;

pub use group::*;
pub use item_slot::*;
pub use layer::*;
pub use reserved_range::*;
pub use selection::*;
pub use summary::*;
pub use system_prompt::*;
