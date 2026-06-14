//! hash_state_finalize tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::HashStateError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct FinalizeRequest {
    #[validate(length(min = 1))]
    pub id: String,
    pub output_encoding: Option<String>,
    pub keep: Option<bool>,
}

#[derive(Serialize)]
pub struct FinalizeResponse {
    pub id: String,
    pub digest: String,
    pub algorithm: String,
    pub output_encoding: String,
}

pub struct HashStateFinalize;

impl Tool for HashStateFinalize {
    const NAME: &'static str = "hash_state_finalize";
    const DESCRIPTION: &'static str =
        "Conclude a streaming hash computation and emit the final digest over \
         everything absorbed so far, discarding the accumulated state by \
         default.";

    type Request = FinalizeRequest;
    type Response = FinalizeResponse;
    type Error = HashStateError;

    fn run(ctx: &ToolContext, req: FinalizeRequest) -> Result<FinalizeResponse, HashStateError> {
        let (digest_bytes, algo) = ctx
            .hash_states
            .finalize(&req.id)
            .ok_or_else(|| HashStateError::NotFound(req.id.clone()))?;

        let enc = req.output_encoding.as_deref().unwrap_or("hex");
        let digest = match enc {
            "base64" => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(&digest_bytes)
            }
            _ => hex::encode(&digest_bytes),
        };

        if req.keep != Some(true) {
            ctx.hash_states.delete(&req.id);
        }

        Ok(FinalizeResponse {
            id: req.id,
            digest,
            algorithm: algo,
            output_encoding: enc.to_string(),
        })
    }
}

pub const HASH_STATE_FINALIZE: RegisteredTool = RegisteredTool::new::<HashStateFinalize>();
