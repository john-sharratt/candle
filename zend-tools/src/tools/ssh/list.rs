//! ssh_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::SshError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

#[derive(Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub host: String,
    pub credential_name: String,
    pub opened_at: String,
    pub last_activity: String,
    pub cwd: String,
    pub alive: bool,
}

#[derive(Serialize)]
pub struct ListResponse {
    pub sessions: Vec<SessionInfo>,
}

pub struct SshSessionList;

impl Tool for SshSessionList {
    const NAME: &'static str = "ssh_session_list";
    const DESCRIPTION: &'static str = "Enumerate the secure shell sessions currently logged in to remote servers, with each host and connection status.";

    type Request = ListRequest;
    type Response = ListResponse;
    type Error = SshError;

    fn run(ctx: &ToolContext, _req: ListRequest) -> Result<ListResponse, SshError> {
        let sessions = ctx.sessions.list_ssh();
        let infos = sessions
            .into_iter()
            .map(|arc| {
                let e = arc.lock().unwrap();
                SessionInfo {
                    session_id: e.meta.session_id.clone(),
                    host: e.host.clone(),
                    credential_name: e.credential_name.clone(),
                    opened_at: e.meta.opened_at.clone(),
                    last_activity: e.meta.last_activity.clone(),
                    cwd: e.cwd.clone(),
                    alive: e.meta.alive,
                }
            })
            .collect();
        Ok(ListResponse { sessions: infos })
    }
}

pub const SSH_SESSION_LIST: RegisteredTool = RegisteredTool::new::<SshSessionList>();
