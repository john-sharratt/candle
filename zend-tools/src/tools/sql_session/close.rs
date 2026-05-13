//! sql_session_close tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::SqlError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct CloseRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
}

#[derive(Serialize)]
pub struct CloseResponse {
    pub session_id: String,
    pub closed: bool,
}

pub struct SqlSessionClose;

impl Tool for SqlSessionClose {
    const NAME: &'static str = "sql_session_close";
    const DESCRIPTION: &'static str =
        "Close a SQL session, releasing the database connection.";
    type Request = CloseRequest;
    type Response = CloseResponse;
    type Error = SqlError;

    fn run(ctx: &ToolContext, req: CloseRequest) -> Result<CloseResponse, SqlError> {
        let removed = ctx.sessions.remove_sql(&req.session_id);
        Ok(CloseResponse {
            session_id: req.session_id,
            closed: removed,
        })
    }
}

pub const SQL_SESSION_CLOSE: RegisteredTool = RegisteredTool::new::<SqlSessionClose>();
