//! hash_state_update tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_data, HashStateError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct UpdateRequest {
    /// State id returned by hash_state_init. Required.
    #[validate(length(min = 1))]
    pub id: String,
    /// The chunk of data to absorb, interpreted according to `data_encoding`. Required.
    #[validate(length(min = 1))]
    pub data: String,
    /// How `data` is encoded. One of: text, hex, base64. Defaults to text.
    #[schemars(with = "Option<super::super::DataEncoding>")]
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
        "Absorb another chunk of data into an in-progress streaming hash \
         computation. Call repeatedly to feed a large input piece by piece \
         before hash_state_finalize emits the digest.";

    type Request = UpdateRequest;
    type Response = UpdateResponse;
    type Error = HashStateError;

    fn run(ctx: &ToolContext, req: UpdateRequest) -> Result<UpdateResponse, HashStateError> {
        let enc = req.data_encoding.as_deref().unwrap_or("text");
        let data = decode_data(&req.data, enc)?;
        let total_bytes = ctx
            .hash_states
            .update(&req.id, &data)
            .ok_or_else(|| HashStateError::NotFound(req.id.clone()))?;
        Ok(UpdateResponse {
            id: req.id,
            total_bytes,
        })
    }
}

pub const HASH_STATE_UPDATE: RegisteredTool = RegisteredTool::new::<HashStateUpdate>();
