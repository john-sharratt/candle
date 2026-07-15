//! sql_session_open tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::SqlError;
use crate::state::sessions::{SessionMeta, SqlConn, SqlEntry};
use crate::{RegisteredTool, Tool, ToolContext};

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn detect_sqlite(path: &str) -> bool {
    path == ":memory:"
        || path.ends_with(".db")
        || path.ends_with(".sqlite")
        || path.ends_with(".sqlite3")
        || (!path.contains("://") && !path.is_empty())
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    /// Optional name of a stored `sql_password` credential, used for its default
    /// database. Omit to open a SQLite file or `:memory:` directly — SQLite needs
    /// no credential.
    pub credential_name: Option<String>,
    /// SQLite database path, or `:memory:` for an in-memory database. Defaults to
    /// the credential's default database if one is named, otherwise `:memory:`.
    pub database: Option<String>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub driver: String,
    pub database: String,
}

pub struct SqlSessionOpen;

impl Tool for SqlSessionOpen {
    const NAME: &'static str = "sql_session_open";
    const DESCRIPTION: &'static str =
        "Open a SQL database session. SQLite is supported natively and needs no \
         credential — pass the database path (or ':memory:' for in-memory). A \
         credential_name is optional and only supplies a default database. \
         Use sql_session_query to run statements. Close with sql_session_close.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = SqlError;

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, SqlError> {
        let cred = match &req.credential_name {
            Some(name) => {
                let cred = ctx
                    .credentials
                    .get_by_name(name)
                    .ok_or_else(|| SqlError::CredentialNotFound(name.clone()))?;
                if cred.cred_type != "sql_password" {
                    return Err(SqlError::InvalidCredentialType);
                }
                Some(cred)
            }
            None => None,
        };
        let db_path = req
            .database
            .or_else(|| cred.as_ref().and_then(|c| c.default_database.clone()))
            .unwrap_or_else(|| ":memory:".into());

        if !detect_sqlite(&db_path) {
            return Err(SqlError::NotSupported(format!(
                "{db_path}: only SQLite paths are supported; use ':memory:' or a .db/.sqlite file"
            )));
        }

        if ctx.sessions.list_sql().len() >= 5 {
            return Err(SqlError::SessionLimitExceeded);
        }

        let conn = rusqlite::Connection::open(&db_path)
            .map_err(|e| SqlError::ConnectionFailed(e.to_string()))?;

        let sid = format!("sess_{}", Uuid::new_v4().simple());
        ctx.sessions.insert_sql(SqlEntry {
            meta: SessionMeta {
                session_id: sid.clone(),
                opened_at: now(),
                last_activity: now(),
                alive: true,
            },
            dsn: db_path.clone(),
            driver: "sqlite".into(),
            conn: SqlConn::Sqlite(conn),
        });

        Ok(OpenResponse {
            session_id: sid,
            driver: "sqlite".into(),
            database: db_path,
        })
    }
}

pub const SQL_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<SqlSessionOpen>();
