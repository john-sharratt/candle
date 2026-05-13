//! hash_state_update tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::{decode_data, HashStateError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct UpdateRequest {
    #[validate(length(min = 1))]
    pub id: String,
    #[validate(length(min = 1))]
    pub data: String,
    pub data_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct UpdateResponse {
    pub id: String,
    pub total_bytes: u64,
}

pub struct HashStateUpdate;

impl Tool for HashStateUpdate {
    const NAME: &'static str = "hash_state_update";
    const DESCRIPTION: &'static str =
        "Feed data into a running hash state. Can be called multiple times.";

    type Request = UpdateRequest;
    type Response = UpdateResponse;
    type Error = HashStateError;

    fn run(ctx: &ToolContext, req: UpdateRequest) -> Result<UpdateResponse, HashStateError> {
        let enc = req.data_encoding.as_deref().unwrap_or("text");
        let data = decode_data(&req.data, enc)?;
        let total_bytes = ctx.hash_states.update(&req.id, &data)
            .ok_or_else(|| HashStateError::NotFound(req.id.clone()))?;
        Ok(UpdateResponse { id: req.id, total_bytes })
    }
}

pub const HASH_STATE_UPDATE: RegisteredTool = RegisteredTool::new::<HashStateUpdate>();
