//! hash_state_init tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::HashStateError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct InitRequest {
    pub id: Option<String>,
    #[validate(length(min = 1))]
    pub algorithm: String,
}

#[derive(Serialize)]
pub struct InitResponse {
    pub id: String,
    pub algorithm: String,
    pub created_at: String,
}

pub struct HashStateInit;

impl Tool for HashStateInit {
    const NAME: &'static str = "hash_state_init";
    const DESCRIPTION: &'static str =
        "Begin an incremental, streaming hash computation — data is fed in \
         chunks rather than all at once. Returns a state id for the \
         subsequent hash_state_update and hash_state_finalize calls. Use \
         when the input is too large to hold in memory at once.";

    type Request = InitRequest;
    type Response = InitResponse;
    type Error = HashStateError;

    fn run(ctx: &ToolContext, req: InitRequest) -> Result<InitResponse, HashStateError> {
        let id = req.id.unwrap_or_else(|| format!("hs_{}", Uuid::new_v4()));
        ctx.hash_states.create(&id, &req.algorithm)
            .map_err(|e| {
                if e.contains("already exists") {
                    HashStateError::IdAlreadyExists(id.clone())
                } else {
                    HashStateError::UnknownAlgorithm(req.algorithm.clone())
                }
            })?;
        let created_at = ctx.hash_states.get_created_at(&id).unwrap_or_default();
        Ok(InitResponse { id, algorithm: req.algorithm, created_at })
    }
}

pub const HASH_STATE_INIT: RegisteredTool = RegisteredTool::new::<HashStateInit>();
