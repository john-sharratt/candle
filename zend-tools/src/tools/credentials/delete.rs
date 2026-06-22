//! credential_delete tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::CredError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct DeleteRequest {
    #[validate(length(min = 1))]
    pub name: String,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    pub name: String,
    pub deleted: bool,
}

pub struct CredentialDelete;

impl Tool for CredentialDelete {
    const NAME: &'static str = "credential_delete";
    const DESCRIPTION: &'static str =
        "Permanently delete a stored credential by name. Use when the user wants to revoke or \
         remove a saved credential, when rotating keys, when an old credential is no longer \
         needed, or when cleaning up after a workflow. Triggered by \"delete the credential \
         for\", \"remove that key\", \"forget that password\", \"rotate this credential\", \
         \"revoke\". Active sessions using the deleted credential continue running until closed; \
         only new opens fail. Returns the name and a deleted flag. Use cred_list to \
         find the exact name before deleting.";

    type Request = DeleteRequest;
    type Response = DeleteResponse;
    type Error = CredError;

    fn run(ctx: &ToolContext, req: DeleteRequest) -> Result<DeleteResponse, CredError> {
        let deleted = ctx.credentials.delete(&req.name);
        if !deleted {
            return Err(CredError::NotFound(req.name));
        }
        Ok(DeleteResponse {
            name: req.name,
            deleted: true,
        })
    }
}

pub const CREDENTIAL_DELETE: RegisteredTool = RegisteredTool::new::<CredentialDelete>();
