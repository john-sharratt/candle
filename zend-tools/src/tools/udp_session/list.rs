//! udp_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::UdpError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

#[derive(Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub local_addr: String,
    pub default_peer: String,
    pub opened_at: String,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub sessions: Vec<SessionInfo>,
}

pub struct UdpSessionList;

impl Tool for UdpSessionList {
    const NAME: &'static str = "udp_session_list";
    const DESCRIPTION: &'static str = "Enumerate the UDP datagram sockets currently bound in this conversation, with each default peer.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = UdpError;

    fn run(ctx: &ToolContext, _req: ListRequest) -> Result<ListResponse, UdpError> {
        let sessions = ctx
            .sessions
            .list_udp()
            .into_iter()
            .map(|arc| {
                let e = arc.lock().unwrap();
                SessionInfo {
                    session_id: e.meta.session_id.clone(),
                    local_addr: e.local_addr.clone(),
                    default_peer: e.default_peer.clone(),
                    opened_at: e.meta.opened_at.clone(),
                }
            })
            .collect();
        Ok(ListResponse { sessions })
    }
}

pub const UDP_SESSION_LIST: RegisteredTool = RegisteredTool::new::<UdpSessionList>();
