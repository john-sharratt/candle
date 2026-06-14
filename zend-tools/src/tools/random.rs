//! `random` tool — generate random values.

use rand::seq::SliceRandom;
use rand::Rng;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    pub kind: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub choices: Option<Vec<String>>,
    #[validate(range(min = 1, max = 1000))]
    pub count: Option<u32>,
    pub sides: Option<u32>,
}

#[derive(Serialize)]
pub struct Response {
    pub result: Value,
}

#[derive(Debug, Error)]
pub enum RandomError {
    #[error("invalid kind: {0}")]
    InvalidKind(String),
    #[error("invalid params: {0}")]
    InvalidParams(String),
}

impl ToolError for RandomError {
    fn code(&self) -> &'static str {
        match self {
            RandomError::InvalidKind(_) => "invalid_kind",
            RandomError::InvalidParams(_) => "invalid_params",
        }
    }
}

pub struct RandomTool;

impl Tool for RandomTool {
    const NAME: &'static str = "random";
    const DESCRIPTION: &'static str =
        "Generate genuinely random values when actual randomness is needed rather than the \
         model's biased pseudo-random picks (which favour 7, 37, \"blue\"). Modes: integer \
         (whole number in a range), float (real in a range), choice (pick from a list), \
         shuffle (randomise list order), dice (roll N dice with S sides). Use for: rolling \
         dice in a game, picking randomly between options, shuffling a list, flipping a coin, \
         drawing names, generating test data. Triggered by \"roll a die\", \"pick one \
         randomly\", \"shuffle these\", \"random number between\", \"flip a coin\", \"choose \
         at random\", \"give me N random\". Returns values appropriate to the chosen mode.";

    type Request = Request;
    type Response = Response;
    type Error = RandomError;

    fn run(_ctx: &ToolContext, req: Request) -> Result<Response, RandomError> {
        let count = req.count.unwrap_or(1) as usize;
        let mut rng = rand::rng();

        let result = match req.kind.as_str() {
            "integer" => {
                let min = req.min.unwrap_or(0.0) as i64;
                let max = req.max.unwrap_or(100.0) as i64;
                if min >= max {
                    return Err(RandomError::InvalidParams("min must be < max".to_string()));
                }
                if count == 1 {
                    Value::Number(rng.random_range(min..=max).into())
                } else {
                    Value::Array(
                        (0..count)
                            .map(|_| Value::Number(rng.random_range(min..=max).into()))
                            .collect(),
                    )
                }
            }
            "float" => {
                let min = req.min.unwrap_or(0.0);
                let max = req.max.unwrap_or(1.0);
                if min >= max {
                    return Err(RandomError::InvalidParams("min must be < max".to_string()));
                }
                let gen_float = || -> Value {
                    let f: f64 = rand::rng().random_range(min..max);
                    serde_json::Number::from_f64(f)
                        .map(Value::Number)
                        .unwrap_or(Value::Null)
                };
                if count == 1 {
                    gen_float()
                } else {
                    Value::Array((0..count).map(|_| gen_float()).collect())
                }
            }
            "choice" => {
                let choices = req.choices.as_ref().ok_or_else(|| {
                    RandomError::InvalidParams("choices required for kind=choice".to_string())
                })?;
                if choices.is_empty() {
                    return Err(RandomError::InvalidParams(
                        "choices must not be empty".to_string(),
                    ));
                }
                let pick = || -> Value {
                    let i = rand::rng().random_range(0..choices.len());
                    Value::String(choices[i].clone())
                };
                if count == 1 {
                    pick()
                } else {
                    Value::Array((0..count).map(|_| pick()).collect())
                }
            }
            "shuffle" => {
                let mut choices = req.choices.clone().ok_or_else(|| {
                    RandomError::InvalidParams("choices required for kind=shuffle".to_string())
                })?;
                choices.shuffle(&mut rng);
                Value::Array(choices.into_iter().map(Value::String).collect())
            }
            "dice" => {
                let sides = req.sides.unwrap_or(6);
                if sides < 2 {
                    return Err(RandomError::InvalidParams("sides must be >= 2".to_string()));
                }
                let roll =
                    || -> Value { Value::Number(rand::rng().random_range(1u32..=sides).into()) };
                if count == 1 {
                    roll()
                } else {
                    Value::Array((0..count).map(|_| roll()).collect())
                }
            }
            other => return Err(RandomError::InvalidKind(other.to_string())),
        };

        Ok(Response { result })
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<RandomTool>();
