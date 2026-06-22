//! http_session_open tool.

use std::collections::HashMap;

use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use super::HttpSessionError;
use crate::state::sessions::{HttpEntry, SessionMeta};
use crate::{ConfirmationDetails, RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OpenRequest {
    pub base_url: Option<String>,
    pub credential_name: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub timeout_sec: Option<u32>,
    pub follow_redirects: Option<bool>,
}

#[derive(Serialize)]
pub struct OpenResponse {
    pub session_id: String,
    pub base_url: Option<String>,
}

pub struct HttpSessionOpen;

impl Tool for HttpSessionOpen {
    const NAME: &'static str = "http_session_open";
    const DESCRIPTION: &'static str =
        "Open a persistent HTTP client session with optional base URL, default headers, and \
         authentication credentials applied to every request. Use before making API calls that \
         need cookies or auth across multiple requests, when the user mentions a specific API \
         or service to interact with, or when setting up a workflow hitting the same endpoint \
         multiple times. Triggered by \"connect to the API\", \"set up a session for\", \"use \
         this base URL\", \"open an HTTP client for\", \"I want to call the X API\". Returns \
         session_id and base_url. Subsequent requests use http_request. For one-shot \
         retrieval of a public page use web_fetch — no session needed.";

    type Request = OpenRequest;
    type Response = OpenResponse;
    type Error = HttpSessionError;

    fn confirmation(req: &OpenRequest) -> Option<ConfirmationDetails> {
        Some(ConfirmationDetails::new("Open HTTP session").with_field(
            "base_url",
            req.base_url.as_deref().unwrap_or("(none)").to_string(),
        ))
    }

    fn run(ctx: &ToolContext, req: OpenRequest) -> Result<OpenResponse, HttpSessionError> {
        let mut builder = reqwest::blocking::Client::builder()
            .cookie_store(true)
            .timeout(std::time::Duration::from_secs(
                req.timeout_sec.unwrap_or(30) as u64,
            ));

        if req.follow_redirects == Some(false) {
            builder = builder.redirect(reqwest::redirect::Policy::none());
        }

        let mut default_headers = reqwest::header::HeaderMap::new();
        if let Some(headers) = req.headers {
            for (k, v) in headers {
                if let (Ok(name), Ok(value)) = (
                    reqwest::header::HeaderName::from_bytes(k.as_bytes()),
                    reqwest::header::HeaderValue::from_str(&v),
                ) {
                    default_headers.insert(name, value);
                }
            }
        }

        let mut credential_name = None;
        if let Some(cred_name) = &req.credential_name {
            let cred = ctx
                .credentials
                .get_by_name(cred_name)
                .ok_or_else(|| HttpSessionError::CredentialNotFound(cred_name.clone()))?;
            credential_name = Some(cred.name.clone());
            match cred.cred_type.as_str() {
                "http_basic" => {
                    let username = cred.username.as_deref().unwrap_or("");
                    let encoded = base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        format!("{username}:{}", cred.secret),
                    );
                    default_headers.insert(
                        reqwest::header::AUTHORIZATION,
                        reqwest::header::HeaderValue::from_str(&format!("Basic {encoded}"))
                            .unwrap(),
                    );
                }
                "api_key" | "bearer_token" => {
                    default_headers.insert(
                        reqwest::header::AUTHORIZATION,
                        reqwest::header::HeaderValue::from_str(&format!("Bearer {}", cred.secret))
                            .unwrap(),
                    );
                }
                "http_header" => {
                    let hname = cred.header_name.as_deref().unwrap_or("X-Api-Key");
                    if let (Ok(name), Ok(value)) = (
                        reqwest::header::HeaderName::from_bytes(hname.as_bytes()),
                        reqwest::header::HeaderValue::from_str(&cred.secret),
                    ) {
                        default_headers.insert(name, value);
                    }
                }
                other => return Err(HttpSessionError::InvalidCredentialType(other.to_string())),
            }
        }

        builder = builder.default_headers(default_headers);
        let client = builder
            .build()
            .map_err(|e| HttpSessionError::ConnectionFailed(e.to_string()))?;

        let session_id = format!("sess_{}", Uuid::new_v4());
        let entry = HttpEntry {
            meta: SessionMeta {
                session_id: session_id.clone(),
                opened_at: Utc::now().to_rfc3339(),
                last_activity: Utc::now().to_rfc3339(),
                alive: true,
            },
            base_url: req.base_url.clone(),
            credential_name,
            client,
        };
        ctx.sessions.insert_http(entry);

        Ok(OpenResponse {
            session_id,
            base_url: req.base_url,
        })
    }
}

pub const HTTP_SESSION_OPEN: RegisteredTool = RegisteredTool::new::<HttpSessionOpen>();
