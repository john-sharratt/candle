//! sql_session_query tool.

use base64::Engine as _;
use rusqlite::types::Value as SqlValue;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::SqlError;
use crate::state::sessions::SqlConn;
use crate::{RegisteredTool, Tool, ToolContext};

fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn sql_to_json(v: SqlValue) -> serde_json::Value {
    match v {
        SqlValue::Null => serde_json::Value::Null,
        SqlValue::Integer(i) => serde_json::Value::Number(i.into()),
        SqlValue::Real(f) => serde_json::json!(f),
        SqlValue::Text(s) => serde_json::Value::String(s),
        SqlValue::Blob(b) => {
            serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(&b))
        }
    }
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct QueryRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    #[validate(length(min = 1))]
    pub query: String,
    pub params: Option<Vec<serde_json::Value>>,
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub rows_affected: usize,
    pub last_insert_rowid: Option<i64>,
}

pub struct SqlSessionQuery;

impl Tool for SqlSessionQuery {
    const NAME: &'static str = "sql_session_query";
    const DESCRIPTION: &'static str =
        "Execute a SQL statement in an open session. Returns rows and columns for SELECT; \
         rows_affected and last_insert_rowid for INSERT/UPDATE/DELETE. Supports \
         positional parameters via the params array (? placeholders).";

    type Request = QueryRequest;
    type Response = QueryResponse;
    type Error = SqlError;

    fn run(ctx: &ToolContext, req: QueryRequest) -> Result<QueryResponse, SqlError> {
        let entry = ctx
            .sessions
            .get_sql(&req.session_id)
            .ok_or_else(|| SqlError::SessionNotFound(req.session_id.clone()))?;
        let mut guard = entry.lock().unwrap();
        guard.meta.last_activity = now();

        let SqlConn::Sqlite(conn) = &mut guard.conn;

        let params_json: Vec<serde_json::Value> = req.params.unwrap_or_default();
        let params_sql: Vec<Box<dyn rusqlite::ToSql>> = params_json
            .iter()
            .map(|v| {
                let b: Box<dyn rusqlite::ToSql> = match v {
                    serde_json::Value::Null => Box::new(rusqlite::types::Null),
                    serde_json::Value::Bool(b) => Box::new(*b as i64),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            Box::new(i)
                        } else {
                            Box::new(n.as_f64().unwrap_or(0.0))
                        }
                    }
                    serde_json::Value::String(s) => Box::new(s.clone()),
                    other => Box::new(other.to_string()),
                };
                b
            })
            .collect();

        let refs: Vec<&dyn rusqlite::ToSql> = params_sql.iter().map(|b| b.as_ref()).collect();

        let mut stmt = conn
            .prepare(&req.query)
            .map_err(|e| SqlError::QueryFailed(e.to_string()))?;

        let col_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();

        let is_select = req.query.trim().to_uppercase().starts_with("SELECT")
            || req.query.trim().to_uppercase().starts_with("WITH");

        if is_select {
            let rows: Vec<Vec<serde_json::Value>> = stmt
                .query_map(refs.as_slice(), |row| {
                    let mut vals = Vec::new();
                    for i in 0..col_names.len() {
                        let v: SqlValue = row.get(i)?;
                        vals.push(sql_to_json(v));
                    }
                    Ok(vals)
                })
                .map_err(|e| SqlError::QueryFailed(e.to_string()))?
                .collect::<Result<_, _>>()
                .map_err(|e: rusqlite::Error| SqlError::QueryFailed(e.to_string()))?;

            Ok(QueryResponse {
                columns: col_names,
                rows,
                rows_affected: 0,
                last_insert_rowid: None,
            })
        } else {
            let n = stmt
                .execute(refs.as_slice())
                .map_err(|e| SqlError::QueryFailed(e.to_string()))?;
            let last_id = conn.last_insert_rowid();
            Ok(QueryResponse {
                columns: vec![],
                rows: vec![],
                rows_affected: n,
                last_insert_rowid: if last_id != 0 { Some(last_id) } else { None },
            })
        }
    }
}

pub const SQL_SESSION_QUERY: RegisteredTool = RegisteredTool::new::<SqlSessionQuery>();
