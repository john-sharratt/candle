//! `datetime` tool — current date/time in any timezone.

use chrono::Utc;
use chrono_tz::Tz;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    pub timezone: Option<String>,
}

#[derive(Serialize)]
pub struct Response {
    pub timezone: String,
    pub iso8601: String,
    pub unix: i64,
    pub weekday: String,
}

#[derive(Debug, Error)]
pub enum DatetimeError {
    #[error("unknown timezone: {0}")]
    InvalidTimezone(String),
}

impl ToolError for DatetimeError {
    fn code(&self) -> &'static str {
        match self {
            DatetimeError::InvalidTimezone(_) => "invalid_timezone",
        }
    }
}

pub struct DatetimeTool;

impl Tool for DatetimeTool {
    const NAME: &'static str = "datetime";
    const DESCRIPTION: &'static str =
        "Return the current date and time in a specified IANA timezone. Use for: getting \
         today's date, current time in a particular city or zone, what day of the week it is, \
         the current ISO timestamp for logging, checking the time in another part of the world. \
         Triggered by \"what time is it\", \"what's the date\", \"what day is today\", \"current \
         time in [city]\", \"time in Tokyo right now\", \"what's today's date\". Returns ISO 8601 \
         timestamp, unix epoch, weekday name, and timezone. Stateless and instant — anchors the \
         model when it would otherwise hallucinate dates.";

    type Request = Request;
    type Response = Response;
    type Error = DatetimeError;

    fn run(_ctx: &ToolContext, req: Request) -> Result<Response, DatetimeError> {
        let tz_str = req.timezone.as_deref().unwrap_or("UTC");
        let tz: Tz = tz_str
            .parse()
            .map_err(|_| DatetimeError::InvalidTimezone(tz_str.to_string()))?;
        let now_utc = Utc::now();
        let now_tz = now_utc.with_timezone(&tz);
        Ok(Response {
            timezone: tz_str.to_string(),
            iso8601: now_tz.to_rfc3339(),
            unix: now_utc.timestamp(),
            weekday: now_tz.format("%A").to_string(),
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<DatetimeTool>();
