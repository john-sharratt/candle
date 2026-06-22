//! `totp` tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Debug, Error)]
pub enum TotpError {
    #[error("credential not found: {0}")]
    CredentialNotFound(String),
    #[error("invalid credential type: expected totp_secret, got {0}")]
    InvalidCredentialType(String),
    #[error("TOTP generation failed: {0}")]
    TotpFailed(String),
}

impl ToolError for TotpError {
    fn code(&self) -> &'static str {
        match self {
            TotpError::CredentialNotFound(_) => "credential_not_found",
            TotpError::InvalidCredentialType(_) => "invalid_credential_type",
            TotpError::TotpFailed(_) => "totp_failed",
        }
    }
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    #[validate(length(min = 1))]
    pub credential_name: String,
    pub digits: Option<u32>,
    pub period_sec: Option<u32>,
    pub algorithm: Option<String>,
}

#[derive(Serialize)]
pub struct Response {
    pub code: String,
    pub valid_for_sec: u32,
    pub credential_name: String,
}

pub struct TotpGenerate;

impl Tool for TotpGenerate {
    const NAME: &'static str = "totp";
    const DESCRIPTION: &'static str =
        "Generate a current TOTP (Time-based One-Time Password / authenticator app) code \
         from a stored totp_secret credential. Use for: producing a 2FA or MFA code before \
         logging in to a service, automating authentication flows that require a rotating OTP, \
         getting the current 6-digit authenticator code for a named secret. Triggered by \
         \"generate a 2FA code\", \"get the TOTP code for\", \"what's the authenticator code for \
         my <name> secret\", \"generate OTP for\". Pass the credential's name in `credential_name` \
         — this tool reads the stored totp_secret itself, so do NOT fetch or look up the \
         credential first; call this directly. Returns the current code, seconds remaining until \
         it expires, and the credential name.";

    type Request = Request;
    type Response = Response;
    type Error = TotpError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, TotpError> {
        let cred = ctx
            .credentials
            .get_by_name(&req.credential_name)
            .ok_or_else(|| TotpError::CredentialNotFound(req.credential_name.clone()))?;

        if cred.cred_type != "totp_secret" {
            return Err(TotpError::InvalidCredentialType(cred.cred_type));
        }

        let digits = req.digits.unwrap_or(6);
        let period = req.period_sec.unwrap_or(30);
        let algo_str = req.algorithm.as_deref().unwrap_or("sha1");

        let algo = match algo_str {
            "sha1" => totp_rs::Algorithm::SHA1,
            "sha256" => totp_rs::Algorithm::SHA256,
            "sha512" => totp_rs::Algorithm::SHA512,
            other => {
                return Err(TotpError::TotpFailed(format!(
                    "unknown TOTP algorithm: {other}"
                )))
            }
        };

        let secret = totp_rs::Secret::Encoded(cred.secret.clone());
        let secret_bytes = secret
            .to_bytes()
            .map_err(|e| TotpError::TotpFailed(e.to_string()))?;
        let totp = totp_rs::TOTP::new(
            algo,
            digits as usize,
            1,
            period as u64,
            secret_bytes,
            None,
            cred.name.clone(),
        )
        .map_err(|e| TotpError::TotpFailed(e.to_string()))?;

        let code = totp
            .generate_current()
            .map_err(|e| TotpError::TotpFailed(e.to_string()))?;

        // Calculate seconds remaining in current period
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let valid_for_sec = period - (now % period as u64) as u32;

        Ok(Response {
            code,
            valid_for_sec,
            credential_name: cred.name,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<TotpGenerate>();
