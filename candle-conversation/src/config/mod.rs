//! The settings a conversation, its scheduler and its sampler are built from.
//!
//! Each kind of setting lives in its own file, so the engine-wide, per-conversation
//! and per-token knobs can be read separately. Architecture defaults come from
//! each family's published sampling. A family missing from the table is caught by
//! a test, because an absent backstop fails silently.
//!
//! - `decode_health`: the thresholds that abort a degenerate decode.
//! - `dry`: the DRY repetition penalty.
//! - `engine_config`: engine-wide and scheduler settings.
//! - `sampling`: per-token sampling and reasoning steering.
//! - `sequence_config`: per-conversation sequence settings.
//! - `tests_config`: the defaults a silent misconfiguration would hide.

use std::collections::HashMap;
use std::sync::Arc;

use candle_nn::kv_cache::{KvFormat, QuantFormat};
use candle_transformers::models::batched_inference::BatchedConfig;

use crate::cancel::CancelToken;
use crate::line_ends::line_end_token_ids;
use crate::models::Dialect;
use crate::persistence::BoundSubstrate;
use crate::projection::{CorruptTurnPolicy, LayerId};
use crate::stencil::AllowedSet;
use crate::tenant_rank::TenantRank;
use crate::token_buffer::TokenBuffer;
use crate::tree::ConversationTreeConfig;

mod decode_health;
mod dry;
mod engine_config;
mod sampling;
mod sequence_config;
#[cfg(test)]
mod tests_config;

pub use decode_health::*;
pub use dry::*;
pub use engine_config::*;
pub use sampling::*;
pub use sequence_config::*;
