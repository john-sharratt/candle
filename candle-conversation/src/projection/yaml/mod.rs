//! Parses the YAML of the projection schema: the text is read into shadow types,
//! checked, and built into the schema with its name maps.
//!
//! `from_yaml` parses the document, assigns identifiers in declaration order,
//! builds the name maps, and runs the parse-time validation before it returns.
//! Malformed YAML is reported as a parse error, and a structural mistake is
//! reported as its own construction error.
//!
//! - `build`: building the schema and its system prompt.
//! - `document_shadow`: shadow types for the document and its tree nodes.
//! - `layer_group_shadow`: shadow types for layers and groups.
//! - `parse_groups`: selection, budget and anchor settings.
//! - `section_tree`: expansion of section trees.
//! - `yaml_ids`: identifier and name maps.
//! - `yaml_policy`: policy blocks.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use candle_transformers::models::dialect::{Dialect, DialectTemplate};
use serde::Deserialize;

use crate::provenance::gallery_arena::Keep;
use crate::provenance::{FusionMode, VoteLaw};

use super::adaptive::{
    AnchorConfig, AnchorMember, BudgetAdaptive, LocalityConfig, MemberBudgetAdaptive, ScanCases,
    ScanNorm, ScanPolicy,
};
use super::error::ConstructionError;
use super::hits::HitsConfig;
use super::ids::{CollectionId, GroupId, LayerId, SectionId};
use super::persisted_ids;
use super::pins::PinSet;
use super::policy::{PolicyConfig, PolicyPreset, SelectionPolicy};
use super::project::OptionalState;
use super::schema::{
    Budget, CompressionPrompt, Content, CorruptTurnPolicy, DecodePriority, Draws, GatherScope,
    GroupSchema, LayerDials, LayerSchema, LayerSummary, Recency, Schema, SectionCollection,
    SectionSchema, SectionTree, SelectionDefault, SelectionRule, SystemPromptItem,
    SystemPromptSchema, TreeCollection, TreeDim, TreeNode, TreeOption, TreeVariant, TurnSummary,
    WorkingSetShare, LINEAGE_BASE,
};
use super::swap_cap::DEFAULT_MAX_SWAPS;
use crate::summary_tree::scope::Scope;
use crate::working_set::{FillShares, TypeShareSpec, TypeShares, WorkingSetConfig};

mod build;
mod document_shadow;
mod layer_group_shadow;
mod parse_groups;
mod section_tree;
mod yaml_ids;
mod yaml_policy;

pub use build::*;
use document_shadow::*;
use layer_group_shadow::*;
use parse_groups::*;
use section_tree::*;
pub use yaml_ids::*;
use yaml_policy::*;
