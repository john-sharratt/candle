//! http_session_request tool.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::tools::web_fetch::is_private_url;
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};
use super::HttpSessionError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReqRequest {
    #[validate(length(min = 1))]
    pub session_id: String,
    pub method: Option<String>,
    pub path: String,
    pub headers: Option<HashMap<String, String>>,
    pub query: Option<HashMap<String, String>>,
    pub body: Option<String>,
    pub body_json: Option<serde_json::Value>,
    pub timeout_sec: Option<u32>,
    #[validate(range(max = 1048576))]
    pub max_response_bytes: Option<usize>,
}

#[derive(Serialize)]
pub struct ReqResponse {
    pub status: u16,
    pub status_text: String,
    pub headers: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_b64: Option<String>,
    pub final_url: String,
    pub duration_ms: u64,
    pub body_truncated: bool,
}

pub struct HttpSessionRequest;

impl Tool for HttpSessionRequest {
    const NAME: &'static str = "http_session_request";
    const DESCRIPTION: &'static str =
        "Issue an HTTP request through an existing session — GET, POST, PUT, PATCH, DELETE, \
         HEAD, or OPTIONS — with optional headers, query parameters, and body. Use for: REST \
         API calls, GraphQL queries, posting JSON data, fetching authenticated resources, \
         anything needing cookies or auth state across calls. Triggered by \"call the API\", \
         \"POST to\", \"GET from\", \"send a request to\", \"hit the endpoint\", \"submit to\". \
         Returns status, headers, body (text) or body_b64 (binary), final URL, and duration. \
         GET/HEAD/OPTIONS skip confirmation; POST/PUT/PATCH/DELETE confirm. For one-shot public \
         page retrieval use web_fetch instead.";

    type Request = ReqRequest;
    type Response = ReqResponse;
    type Error = HttpSessionError;

    fn confirmation(req: &ReqRequest) -> Option<ConfirmationDetails> {
        let method = req.method.as_deref().unwrap_or("GET").to_uppercase();
        if matches!(method.as_str(), "POST" | "PUT" | "PATCH" | "DELETE") {
            Some(ConfirmationDetails::new(format!("{method} {}", req.path))
                .with_field("session_id", req.session_id.clone())
                .with_field("method", method))
        } else {
            None
        }
    }

    fn run(ctx: &ToolContext, req: ReqRequest) -> Result<ReqResponse, HttpSessionError> {
        let entry_arc = ctx.sessions.get_http(&req.session_id)
            .ok_or_else(|| HttpSessionError::SessionNotFound(req.session_id.clone()))?;
        let entry = entry_arc.lock().unwrap();

        let url = if req.path.starts_with("http://") || req.path.starts_with("https://") {
            req.path.clone()
        } else if let Some(base) = &entry.base_url {
            format!("{}/{}", base.trim_end_matches('/'), req.path.trim_start_matches('/'))
        } else {
            req.path.clone()
        };

        if is_private_url(&url) {
            return Err(HttpSessionError::UrlBlocked("private/localhost URLs blocked".to_string()));
        }

        let method = req.method.as_deref().unwrap_or("GET").to_uppercase();
        let method = reqwest::Method::from_bytes(method.as_bytes())
            .map_err(|e| HttpSessionError::ConnectionFailed(e.to_string()))?;

        let mut builder = entry.client.request(method, &url);

        if let Some(headers) = req.headers {
            for (k, v) in headers {
                if let (Ok(name), Ok(value)) = (
                    reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                    reqwest::header::HeaderValue::from_str(&v),
                ) {
                    builder = builder.header(name, value);
                }
            }
        }

        if let Some(query) = req.query {
            builder = builder.query(&query.into_iter().collect::<Vec<_>>());
        }

        if let Some(json) = req.body_json {
            builder = builder.json(&json);
        } else if let Some(body) = req.body {
            builder = builder.body(body);
        }

        let start = std::time::Instant::now();
        let resp = builder.send()
            .map_err(|e| if e.is_timeout() {
                HttpSessionError::Timeout
            } else {
                HttpSessionError::ConnectionFailed(e.to_string())
            })?;

        let duration_ms = start.elapsed().as_millis() as u64;
        let status = resp.status();
        let final_url = resp.url().to_string();
        let content_type = resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let mut resp_headers = HashMap::new();
        for (k, v) in resp.headers() {
            resp_headers.insert(k.to_string(), v.to_str().unwrap_or("").to_string());
        }

        let cap = req.max_response_bytes.unwrap_or(32768).min(1048576);
        let body_bytes = resp.bytes()
            .map_err(|e| HttpSessionError::ConnectionFailed(e.to_string()))?;
        let truncated = body_bytes.len() > cap;
        let body_slice = &body_bytes[..body_bytes.len().min(cap)];

        // Return body as text if content-type is text-ish and bytes are valid UTF-8
        let is_text_type = content_type.contains("text/")
            || content_type.contains("application/json")
            || content_type.contains("application/xml")
            || content_type.contains("application/javascript")
            || content_type.contains("+json")
            || content_type.contains("+xml");

        let (body, body_b64) = if is_text_type {
            match std::str::from_utf8(body_slice) {
                Ok(s) => (Some(s.to_string()), None),
                Err(_) => (None, Some(base64_encode(body_slice))),
            }
        } else {
            match std::str::from_utf8(body_slice) {
                Ok(s) => (Some(s.to_string()), None),
                Err(_) => (None, Some(base64_encode(body_slice))),
            }
        };

        Ok(ReqResponse {
            status: status.as_u16(),
            status_text: status.canonical_reason().unwrap_or("").to_string(),
            headers: resp_headers,
            body,
            body_b64,
            final_url,
            duration_ms,
            body_truncated: truncated,
        })
    }
}

fn base64_encode(data: &[u8]) -> String {
    use std::fmt::Write;
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
        let _ = write!(out, "{}{}{}{}",
            CHARS[b0 >> 2] as char,
            CHARS[((b0 & 3) << 4) | (b1 >> 4)] as char,
            if chunk.len() > 1 { CHARS[((b1 & 0xf) << 2) | (b2 >> 6)] as char } else { '=' },
            if chunk.len() > 2 { CHARS[b2 & 0x3f] as char } else { '=' },
        );
    }
    out
}

pub const HTTP_SESSION_REQUEST: RegisteredTool = RegisteredTool::new::<HttpSessionRequest>();
