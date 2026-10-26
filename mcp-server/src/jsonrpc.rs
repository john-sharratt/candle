//! JSON-RPC 2.0 as MCP carries it: one message per POST — a request, a
//! notification, or a client's response to a server request. Batches were
//! removed from MCP in 2025-06-18 and are refused.

use serde_json::{json, Value};

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;

/// One message a client sent.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// A call that wants an answer under `id`.
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    /// A call that wants none.
    Notification { method: String, params: Value },
    /// The client answering something the server asked; the server asks
    /// nothing, so it is acknowledged and dropped.
    Response,
}

/// Read one message from `body`, or the error object to answer with.
pub fn parse(body: &[u8]) -> Result<Message, Value> {
    let value: Value = serde_json::from_slice(body)
        .map_err(|e| error(Value::Null, PARSE_ERROR, &e.to_string()))?;
    let Some(obj) = value.as_object() else {
        return Err(error(
            Value::Null,
            INVALID_REQUEST,
            "a message is one JSON object; batches are not part of MCP",
        ));
    };
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(error(
            Value::Null,
            INVALID_REQUEST,
            "`jsonrpc` must be \"2.0\"",
        ));
    }
    let id = obj.get("id").cloned().filter(|id| !id.is_null());
    let params = obj.get("params").cloned().unwrap_or(json!({}));
    match (obj.get("method").and_then(Value::as_str), id) {
        (Some(method), Some(id)) => Ok(Message::Request {
            id,
            method: method.to_string(),
            params,
        }),
        (Some(method), None) => Ok(Message::Notification {
            method: method.to_string(),
            params,
        }),
        (None, Some(_)) if obj.contains_key("result") || obj.contains_key("error") => {
            Ok(Message::Response)
        }
        (None, _) => Err(error(
            Value::Null,
            INVALID_REQUEST,
            "a request names its `method`",
        )),
    }
}

/// The answer to request `id`.
pub fn result(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

/// The error answer to request `id`.
pub fn error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// A notification from the server.
pub fn notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "method": method, "params": params})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_a_notification_and_a_response_are_told_apart() {
        assert_eq!(
            parse(br#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#).unwrap(),
            Message::Request {
                id: json!(7),
                method: "ping".into(),
                params: json!({})
            }
        );
        assert_eq!(
            parse(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).unwrap(),
            Message::Notification {
                method: "notifications/initialized".into(),
                params: json!({})
            }
        );
        assert_eq!(
            parse(br#"{"jsonrpc":"2.0","id":1,"result":{}}"#).unwrap(),
            Message::Response
        );
    }

    #[test]
    fn a_batch_and_a_bad_body_are_refused_with_their_codes() {
        let code = |e: Value| e["error"]["code"].as_i64().unwrap();
        assert_eq!(code(parse(b"not json").unwrap_err()), PARSE_ERROR);
        assert_eq!(code(parse(b"[]").unwrap_err()), INVALID_REQUEST);
        assert_eq!(
            code(parse(br#"{"id":1,"method":"x"}"#).unwrap_err()),
            INVALID_REQUEST
        );
    }

    #[test]
    fn answers_are_exact() {
        assert_eq!(
            result(json!(1), json!({})).to_string(),
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#
        );
        assert_eq!(
            error(json!("a"), METHOD_NOT_FOUND, "no").to_string(),
            r#"{"jsonrpc":"2.0","id":"a","error":{"code":-32601,"message":"no"}}"#
        );
    }
}
