//! `calculator` tool — evaluate a mathematical expression.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    #[validate(length(min = 1, max = 1024))]
    pub expression: String,
}

#[derive(Serialize)]
pub struct Response {
    pub expression: String,
    pub result: f64,
}

#[derive(Debug, Error)]
pub enum CalcError {
    #[error("parse error: {0}")]
    ParseError(String),
    #[error("math error: {0}")]
    MathError(String),
}

impl ToolError for CalcError {
    fn code(&self) -> &'static str {
        match self {
            CalcError::ParseError(_) => "parse_error",
            CalcError::MathError(_) => "math_error",
        }
    }
}

pub struct CalculatorTool;

impl Tool for CalculatorTool {
    const NAME: &'static str = "calculator";
    const DESCRIPTION: &'static str =
        "Evaluate an arithmetic or scientific expression and return the exact result. Use for: \
         arithmetic the model would otherwise compute mentally and get wrong, multi-digit \
         multiplication or division, percentage calculations, square roots, trigonometry, \
         evaluating formulas with parentheses and standard functions. Supports +, -, *, /, %, \
         ^, sqrt, sin, cos, tan, log, ln, exp, abs, min, max, floor, ceil. Triggered by \
         \"calculate\", \"compute\", \"what is X times Y\", \"how much is\", \"what's the \
         square root of\", or any explicit math problem. Returns the numeric result. Use \
         unit_convert for unit conversions; use random for generating random values.";

    type Request = Request;
    type Response = Response;
    type Error = CalcError;

    fn run(_ctx: &ToolContext, req: Request) -> Result<Response, CalcError> {
        let result = evalexpr::eval_number(&req.expression)
            .map_err(|e| CalcError::ParseError(e.to_string()))?;
        if result.is_nan() {
            return Err(CalcError::MathError("result is NaN".to_string()));
        }
        if result.is_infinite() {
            return Err(CalcError::MathError(
                "result is infinite (division by zero?)".to_string(),
            ));
        }
        Ok(Response {
            expression: req.expression,
            result,
        })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<CalculatorTool>();
