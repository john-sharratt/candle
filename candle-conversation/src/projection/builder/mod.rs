//! The public `Builder`, which turns a schema and a turn resolver into projections.
//!
//! A builder is built once from YAML or from a schema and then projects any number
//! of targets. Its turn state lives in the resolver rather than in the builder, so
//! the schema never changes once it is constructed. Validation runs at
//! construction, so a schema that builds successfully is guaranteed to project
//! without error.
//!
//! - `construct`: building and validating from YAML or a schema.
//! - `plain_prompt`: the single-layer schema for a plain prompt.
//! - `projecting`: the projection entry points.
//! - `schema_access`: read access and name lookups.
//! - `sections`: runtime section and collection mutators.
//! - `tuning`: by-name adjustments before first use.

use std::sync::Arc;

use super::adaptive::ScanPolicy;
use super::error::ConstructionError;
use super::hits::HitsConfig;
use super::ids::{CollectionId, GroupId, LayerId, Reserved, SectionId};
use super::pins::PinSet;
use super::policy::SelectionPolicy;
use super::project::{
    run, run_with_sink, PriorBelief, Projection, ProjectionMode, ProjectionTarget, SelectionState,
};
use super::schema::{
    Budget, CompressionPrompt, Content, CorruptTurnPolicy, DecodePriority, GatherScope,
    GroupSchema, LayerDials, LayerSchema, LayerSummary, Schema, SectionCollection, SectionSchema,
    SelectionRule, SystemPromptItem, SystemPromptSchema, TreeCollection, TreeVariant, TurnSummary,
};
use super::script::ProjectionScript;
use super::swap_cap::DEFAULT_MAX_SWAPS;
use super::yaml::{from_yaml, NameMaps};
use crate::substrate::ContentResolver;
use crate::summary_tree::scope::Scope;
use crate::summary_tree::SelectionDiagnostics;

// ── Builder ───────────────────────────────────────────────────────────────────

/// The one turn group of a plain-prompt schema ([`Builder::for_plain_prompt`]
/// and its reserved forms).
pub const PLAIN_PROMPT_GROUP: &str = "primary_conversation";

/// The tag no turn is written with, which is what a closed group admits
/// ([`Builder::for_closed_reserved`]).
const CLOSED_GROUP_TAG: &str = "group:closed";

/// Compiled schema for the multi-layer projection substrate.
///
/// Holds the immutable [`Schema`] and reverse-lookup maps from YAML string
/// names to opaque ids. Produces a [`Projection`] on each
/// [`Builder::project`] call.
///
/// **Turn state lives in the resolver, not here.** Pass a [`super::resolver::Substrate`]
/// (or any [`ContentResolver`] impl) to [`Builder::project`]; the schema
/// never changes once constructed.
#[derive(Debug, Clone)]
pub struct Builder {
    pub(super) schema: Schema,
    /// Reverse maps from YAML string names to ids. Empty when constructed
    /// via [`Builder::from_schema`].
    pub(super) name_maps: NameMaps,
    /// The resolved YAML this builder was parsed from, retained so the
    /// substrate can persist it as the `Template` record (the projection
    /// schema is reconstructable via [`Builder::from_yaml`]). `None` for
    /// programmatic [`Builder::from_schema`] construction.
    source_yaml: Option<String>,
    /// Recorded projections to play back instead of choosing. Only the
    /// diagnostic decode carries one ([`Self::with_script`]).
    script: Option<Arc<ProjectionScript>>,
}

mod construct;
mod plain_prompt;
mod projecting;
mod schema_access;
mod sections;
mod tuning;

use construct::*;
use sections::*;
