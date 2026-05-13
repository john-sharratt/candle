//! telnet_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::TelnetError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

#[derive(Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub host: String,
    pub port: u16,
    pub opened_at: String,
    pub alive: bool,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub sessions: Vec<SessionInfo>,
}

pub struct TelnetSessionList;

impl Tool for TelnetSessionList {
    const NAME: &'static str = "telnet_session_list";
    const DESCRIPTION: &'static str = "List all open Telnet sessions.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = TelnetError;

    fn run(ctx: &ToolContext, _req: ListRequest) -> Result<ListResponse, TelnetError> {
        let sessions = ctx.sessions.list_telnet().into_iter().map(|arc| {
            let e = arc.lock().unwrap();
            SessionInfo {
                session_id: e.meta.session_id.clone(),
                host: e.host.clone(),
                port: e.port,
                opened_at: e.meta.opened_at.clone(),
                alive: e.meta.alive,
            }
        }).collect();
        Ok(ListResponse { sessions })
    }
}

pub const TELNET_SESSION_LIST: RegisteredTool = RegisteredTool::new::<TelnetSessionList>();
