//! tls_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::TlsError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

#[derive(Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub host: String,
    pub port: u16,
    pub local_addr: String,
    pub opened_at: String,
    pub alive: bool,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub sessions: Vec<SessionInfo>,
}

pub struct TlsSessionList;

impl Tool for TlsSessionList {
    const NAME: &'static str = "tls_session_list";
    const DESCRIPTION: &'static str = "List all open TLS sessions.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = TlsError;

    fn run(ctx: &ToolContext, _req: ListRequest) -> Result<ListResponse, TlsError> {
        let sessions = ctx.sessions.list_tls().into_iter().map(|arc| {
            let e = arc.lock().unwrap();
            SessionInfo {
                session_id: e.meta.session_id.clone(),
                host: e.host.clone(),
                port: e.port,
                local_addr: e.local_addr.clone(),
                opened_at: e.meta.opened_at.clone(),
                alive: e.meta.alive,
            }
        }).collect();
        Ok(ListResponse { sessions })
    }
}

pub const TLS_SESSION_LIST: RegisteredTool = RegisteredTool::new::<TlsSessionList>();
