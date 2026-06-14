//! tcp_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::TcpError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

#[derive(Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub peer_addr: String,
    pub local_addr: String,
    pub opened_at: String,
    pub alive: bool,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub sessions: Vec<SessionInfo>,
}

pub struct TcpSessionList;

impl Tool for TcpSessionList {
    const NAME: &'static str = "tcp_session_list";
    const DESCRIPTION: &'static str = "Enumerate the raw TCP stream sockets currently open in this conversation, with each peer address and idle state.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = TcpError;

    fn run(ctx: &ToolContext, _req: ListRequest) -> Result<ListResponse, TcpError> {
        let sessions = ctx
            .sessions
            .list_tcp()
            .into_iter()
            .map(|arc| {
                let e = arc.lock().unwrap();
                SessionInfo {
                    session_id: e.meta.session_id.clone(),
                    peer_addr: e.peer_addr.clone(),
                    local_addr: e.local_addr.clone(),
                    opened_at: e.meta.opened_at.clone(),
                    alive: e.meta.alive,
                }
            })
            .collect();
        Ok(ListResponse { sessions })
    }
}

pub const TCP_SESSION_LIST: RegisteredTool = RegisteredTool::new::<TcpSessionList>();
