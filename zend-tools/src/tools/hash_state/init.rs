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
        "Initialize an incremental hash state for streaming data. \
         Use for: hashing large data in chunks, computing checksums over streams. \
         Returns id for subsequent update/finalize calls.";

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
