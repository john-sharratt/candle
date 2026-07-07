//! `calculator` tool — evaluate a mathematical expression.

use evalexpr::{build_operator_tree, DefaultNumericTypes, Node, Operator, Value};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// Arithmetic/scientific expression to evaluate (e.g. "2 + 2", "sqrt(16)",
    /// "sin(0) * 3"). Supports +, -, *, /, %, ^, sqrt, sin, cos, tan, ln,
    /// log (base 10), log2, exp, abs, min, max, floor, ceil, round. Required;
    /// length 1–1024 characters.
    #[validate(length(min = 1, max = 1024))]
    pub expression: String,
}

/// The evaluated result, kept in its natural type: an integer expression
/// (`2 + 2`, `6 / 2`) returns an integer, a fractional one (`7 / 2`, `sqrt(2)`)
/// returns a float. `#[serde(untagged)]` serialises each as a bare JSON number
/// (`4` vs `3.5`) rather than forcing everything to a float with a spurious `.0`.
#[derive(Serialize)]
#[serde(untagged)]
pub enum NumericResult {
    Int(i64),
    Float(f64),
}

#[derive(Serialize)]
pub struct Response {
    pub expression: String,
    pub result: NumericResult,
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
         ^, sqrt, sin, cos, tan, ln, log (base 10), log2, exp, abs, min, max, floor, ceil, \
         round. Triggered by \"calculate\", \"compute\", \"what is X times Y\", \"how much is\", \
         \"what's the square root of\", or any explicit math problem. Returns the numeric \
         result. Use unit_convert for unit conversions; use random for generating random values.";

    type Request = Request;
    type Response = Response;
    type Error = CalcError;

    fn run(_ctx: &ToolContext, req: Request) -> Result<Response, CalcError> {
        // evalexpr parses (respecting precedence / parens); we walk the tree
        // ourselves so division can choose int vs float per-operation and
        // integer-only sub-expressions keep full `i64` precision — neither of
        // which evalexpr's own evaluator does (it floor-divides integers and its
        // math functions are namespaced `math::…`).
        let tree = build_operator_tree::<DefaultNumericTypes>(&req.expression)
            .map_err(|e| CalcError::ParseError(e.to_string()))?;
        let result = match eval_node(&tree)? {
            Num::Int(i) => NumericResult::Int(i),
            Num::Float(f) => {
                if f.is_nan() {
                    return Err(CalcError::MathError("result is NaN".to_string()));
                }
                if f.is_infinite() {
                    return Err(CalcError::MathError(
                        "result is infinite (division by zero?)".to_string(),
                    ));
                }
                NumericResult::Float(f)
            }
        };
        Ok(Response {
            expression: req.expression,
            result,
        })
    }
}

/// A numeric value that remembers whether it is an exact integer or a float.
/// Integer arithmetic stays `i64` (exact, no `2^53` precision cliff) until an
/// operation genuinely produces a fraction or overflows, at which point it
/// promotes to `f64`.
#[derive(Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }
}

/// Recursively evaluate a parsed evalexpr node into a [`Num`].
fn eval_node(node: &Node<DefaultNumericTypes>) -> Result<Num, CalcError> {
    match node.operator() {
        // `build_operator_tree` wraps the expression in a root node; an empty
        // input yields a childless root (already excluded by length validation).
        Operator::RootNode => match node.children() {
            [child] => eval_node(child),
            _ => Err(CalcError::ParseError("empty expression".to_string())),
        },
        Operator::Const { value } => match value {
            Value::Int(i) => Ok(Num::Int(*i)),
            Value::Float(f) => Ok(Num::Float(*f)),
            _ => Err(CalcError::ParseError(
                "expression is not numeric".to_string(),
            )),
        },
        Operator::Neg => match eval_node(child(node, 0)?)? {
            Num::Int(i) => Ok(i
                .checked_neg()
                .map(Num::Int)
                .unwrap_or(Num::Float(-(i as f64)))),
            Num::Float(f) => Ok(Num::Float(-f)),
        },
        Operator::Add => int_or_float(node, i64::checked_add, |a, b| a + b),
        Operator::Sub => int_or_float(node, i64::checked_sub, |a, b| a - b),
        Operator::Mul => int_or_float(node, i64::checked_mul, |a, b| a * b),
        Operator::Div => eval_div(node),
        Operator::Mod => eval_mod(node),
        Operator::Exp => eval_exp(node),
        Operator::FunctionIdentifier { identifier } => eval_function(identifier, node),
        Operator::VariableIdentifierRead { identifier } => Err(CalcError::ParseError(format!(
            "unknown identifier: {identifier}"
        ))),
        _ => Err(CalcError::ParseError(
            "unsupported operation in expression".to_string(),
        )),
    }
}

/// The `index`-th child of a node, or a parse error if it is missing.
fn child(
    node: &Node<DefaultNumericTypes>,
    index: usize,
) -> Result<&Node<DefaultNumericTypes>, CalcError> {
    node.children()
        .get(index)
        .ok_or_else(|| CalcError::ParseError("malformed expression".to_string()))
}

/// Evaluate a binary node's two operands.
fn operands(node: &Node<DefaultNumericTypes>) -> Result<(Num, Num), CalcError> {
    Ok((eval_node(child(node, 0)?)?, eval_node(child(node, 1)?)?))
}

/// `+`, `-`, `*`: stay `i64` when both operands are integers and the checked op
/// doesn't overflow; otherwise fall back to `f64`.
fn int_or_float(
    node: &Node<DefaultNumericTypes>,
    checked: fn(i64, i64) -> Option<i64>,
    float: fn(f64, f64) -> f64,
) -> Result<Num, CalcError> {
    let (a, b) = operands(node)?;
    Ok(match (a, b) {
        (Num::Int(x), Num::Int(y)) => checked(x, y)
            .map(Num::Int)
            .unwrap_or(Num::Float(float(x as f64, y as f64))),
        _ => Num::Float(float(a.as_f64(), b.as_f64())),
    })
}

/// Smart division: integer result when it divides evenly, float when there is a
/// remainder or either operand is already a float.
fn eval_div(node: &Node<DefaultNumericTypes>) -> Result<Num, CalcError> {
    let (a, b) = operands(node)?;
    match (a, b) {
        (Num::Int(x), Num::Int(y)) => {
            if y == 0 {
                return Err(CalcError::MathError("division by zero".to_string()));
            }
            if x % y == 0 {
                Ok(Num::Int(x / y))
            } else {
                Ok(Num::Float(x as f64 / y as f64))
            }
        }
        _ => {
            let divisor = b.as_f64();
            if divisor == 0.0 {
                return Err(CalcError::MathError("division by zero".to_string()));
            }
            Ok(Num::Float(a.as_f64() / divisor))
        }
    }
}

/// `%`: integer remainder for two integers, `f64` remainder otherwise.
fn eval_mod(node: &Node<DefaultNumericTypes>) -> Result<Num, CalcError> {
    let (a, b) = operands(node)?;
    match (a, b) {
        (Num::Int(x), Num::Int(y)) => {
            if y == 0 {
                return Err(CalcError::MathError("modulo by zero".to_string()));
            }
            Ok(Num::Int(x % y))
        }
        _ => Ok(Num::Float(a.as_f64() % b.as_f64())),
    }
}

/// `^`: exact integer power when the base and a non-negative exponent are
/// integers and it doesn't overflow; otherwise `f64::powf`.
fn eval_exp(node: &Node<DefaultNumericTypes>) -> Result<Num, CalcError> {
    let (a, b) = operands(node)?;
    Ok(match (a, b) {
        (Num::Int(base), Num::Int(exp)) if exp >= 0 => u32::try_from(exp)
            .ok()
            .and_then(|e| base.checked_pow(e))
            .map(Num::Int)
            .unwrap_or(Num::Float((base as f64).powf(exp as f64))),
        _ => Num::Float(a.as_f64().powf(b.as_f64())),
    })
}

/// Evaluate a function call. Scientific functions return `f64`; `abs` preserves
/// its argument's type; `floor`/`ceil`/`round` yield integers; `min`/`max`
/// return the winning argument unchanged (preserving its int/float type).
fn eval_function(name: &str, node: &Node<DefaultNumericTypes>) -> Result<Num, CalcError> {
    // A function node has a single child: the lone argument, or a `Tuple` whose
    // children are the arguments (e.g. `min(1, 2)`).
    let args: Vec<&Node<DefaultNumericTypes>> = match node.children() {
        [only] if matches!(only.operator(), Operator::Tuple) => only.children().iter().collect(),
        other => other.iter().collect(),
    };
    let expect = |n: usize| -> Result<(), CalcError> {
        if args.len() == n {
            Ok(())
        } else {
            Err(CalcError::ParseError(format!(
                "{name} expects {n} argument(s), got {}",
                args.len()
            )))
        }
    };

    match name {
        "sqrt" | "sin" | "cos" | "tan" | "ln" | "log" | "log2" | "exp" => {
            expect(1)?;
            let x = eval_node(args[0])?.as_f64();
            let r = match name {
                "sqrt" => x.sqrt(),
                "sin" => x.sin(),
                "cos" => x.cos(),
                "tan" => x.tan(),
                "ln" => x.ln(),
                "log" => x.log10(),
                "log2" => x.log2(),
                "exp" => x.exp(),
                _ => unreachable!(),
            };
            Ok(Num::Float(r))
        }
        "abs" => {
            expect(1)?;
            Ok(match eval_node(args[0])? {
                Num::Int(i) => Num::Int(i.abs()),
                Num::Float(f) => Num::Float(f.abs()),
            })
        }
        "floor" | "ceil" | "round" => {
            expect(1)?;
            Ok(match eval_node(args[0])? {
                Num::Int(i) => Num::Int(i),
                Num::Float(f) => {
                    let r = match name {
                        "floor" => f.floor(),
                        "ceil" => f.ceil(),
                        "round" => f.round(),
                        _ => unreachable!(),
                    };
                    Num::Int(r as i64)
                }
            })
        }
        "min" | "max" => {
            if args.is_empty() {
                return Err(CalcError::ParseError(format!(
                    "{name} expects at least 1 argument"
                )));
            }
            let mut best = eval_node(args[0])?;
            for arg in &args[1..] {
                let candidate = eval_node(arg)?;
                let take = if name == "min" {
                    candidate.as_f64() < best.as_f64()
                } else {
                    candidate.as_f64() > best.as_f64()
                };
                if take {
                    best = candidate;
                }
            }
            Ok(best)
        }
        _ => Err(CalcError::ParseError(format!("unknown function: {name}"))),
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<CalculatorTool>();
