//! http_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::HttpSessionError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

#[derive(Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub base_url: Option<String>,
    pub credential_name: Option<String>,
    pub opened_at: String,
    pub alive: bool,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub sessions: Vec<SessionInfo>,
}

pub struct HttpSessionList;

impl Tool for HttpSessionList {
    const NAME: &'static str = "http_session_list";
    const DESCRIPTION: &'static str =
        "List active HTTP sessions for the current conversation — session_id, base_url, \
         credential name, opened_at, last_activity. Use to find session_ids by base URL or \
         to see which APIs are currently configured. Triggered by \"list HTTP sessions\", \
         \"what API connections are open\", \"show my API sessions\".";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = HttpSessionError;

    fn run(ctx: &ToolContext, _req: ListRequest) -> Result<ListResponse, HttpSessionError> {
        let sessions = ctx.sessions.list_http().into_iter().map(|arc| {
            let e = arc.lock().unwrap();
            SessionInfo {
                session_id: e.meta.session_id.clone(),
                base_url: e.base_url.clone(),
                credential_name: e.credential_name.clone(),
                opened_at: e.meta.opened_at.clone(),
                alive: e.meta.alive,
            }
        }).collect();
        Ok(ListResponse { sessions })
    }
}

pub const HTTP_SESSION_LIST: RegisteredTool = RegisteredTool::new::<HttpSessionList>();
