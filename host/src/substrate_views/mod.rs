//! The substrate inspector's views, which every tenant serves over its own
//! substrate (`docs/shared_model_host.md` §3.5).
//!
//! A lazy hierarchy. The [`overview`] is the tenant's line in the host's
//! `GET /v1/substrate` — storage, counts and layer metadata, cheap however large
//! the corpus — and each level below it is a view the page fetches only when
//! that node expands: [`system_prompt`], one layer's [`layer_conversations`],
//! one conversation's [`timeline`]. The tenant serves them under
//! `/v1/substrate/<tenant>/`, which its line names as `views`.
//!
//! All read the live `Substrate` through a `Conversation` handle — one version
//! of it, held for the walk — never a rebuild from the redo log.
//!
//! | Module | View |
//! |---|---|
//! | [`overview`] | the top of the tree |
//! | [`segments`] | the redo log's files on disk |
//! | [`selection`] | a group's selection rule, named |
//! | [`layer`] | one layer's conversations |
//! | [`system_prompt`] | the shared system prompt's framing |
//! | [`timeline`] | one conversation's forest |

mod layer;
mod overview;
mod segments;
mod selection;
mod system_prompt;
mod timeline;

pub use layer::{layer_conversations, title, ConvView, LayerConversations, Unit};
pub use overview::{overview, Counts, GroupView, LayerView, Storage, SubstrateOverview};
pub use segments::{segment_files, SegmentView};
pub use selection::fmt_selection;
pub use system_prompt::{system_prompt, system_prompt_labels, SectionView, SystemPromptView};
pub use timeline::{timeline, TimelineDetail, TurnView};
