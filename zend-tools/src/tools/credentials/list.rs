//! cred_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::CredError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    #[serde(rename = "type")]
    pub cred_type: Option<String>,
}

#[derive(Serialize)]
pub struct CredEntry {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub cred_type: String,
    pub username: Option<String>,
    pub default_host: Option<String>,
    pub default_port: Option<u16>,
    pub created_at: String,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub credentials: Vec<CredEntry>,
}

pub struct CredentialList;

impl Tool for CredentialList {
    const NAME: &'static str = "cred_list";
    const DESCRIPTION: &'static str =
        "List all credentials available to the current user, returning metadata only — names, \
         types, usernames, default hosts, creation dates — never the secret material itself. \
         Use to discover what credentials exist before opening a session, to find a credential \
         name by its friendly name, to check what's been saved, or to verify a credential is \
         still present. Triggered by \"what credentials do I have\", \"list saved logins\", \
         \"show my keys\", \"do I have a credential for\", \"what's stored\". An optional type \
         filter narrows to a specific kind (e.g. ssh_key, http_bearer).";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = CredError;

    fn run(ctx: &ToolContext, req: ListRequest) -> Result<ListResponse, CredError> {
        let creds = ctx.credentials.list(req.cred_type.as_deref());
        let credentials = creds
            .into_iter()
            .map(|c| CredEntry {
                id: c.id,
                name: c.name,
                cred_type: c.cred_type,
                username: c.username,
                default_host: c.default_host,
                default_port: c.default_port,
                created_at: c.created_at,
            })
            .collect();
        Ok(ListResponse { credentials })
    }
}

pub const CREDENTIAL_LIST: RegisteredTool = RegisteredTool::new::<CredentialList>();
