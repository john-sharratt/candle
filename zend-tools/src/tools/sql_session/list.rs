//! sql_session_list tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::SqlError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ListRequest {}

#[derive(Serialize)]
pub struct ListResponse {
    pub sessions: Vec<serde_json::Value>,
}

pub struct SqlSessionList;

impl Tool for SqlSessionList {
    const NAME: &'static str = "sql_session_list";
    const DESCRIPTION: &'static str = "Enumerate the database sessions currently open in this conversation, with each database path and connection state.";
    type Request = ListRequest;
    type Response = ListResponse;
    type Error = SqlError;

    fn run(ctx: &ToolContext, _req: ListRequest) -> Result<ListResponse, SqlError> {
        let sessions = ctx
            .sessions
            .list_sql()
            .into_iter()
            .map(|e| {
                let g = e.lock().unwrap();
                serde_json::json!({
                    "session_id": g.meta.session_id,
                    "driver": g.driver,
                    "database": g.dsn,
                    "opened_at": g.meta.opened_at,
                    "last_activity": g.meta.last_activity,
                    "alive": g.meta.alive,
                })
            })
            .collect();
        Ok(ListResponse { sessions })
    }
}

pub const SQL_SESSION_LIST: RegisteredTool = RegisteredTool::new::<SqlSessionList>();
