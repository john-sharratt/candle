//! Error types for the narrator module.

use std::fmt;

/// Error returned by [`parse_turn`](super::parse_turn).
#[derive(Debug)]
pub enum ParseError {
    /// All lines were empty or whitespace.
    EmptyInput,
    /// A line could not be parsed as a valid RPE command.
    InvalidCommand { line: String, error: clap::Error },
    /// A valid command was parsed but its content field is empty.
    EmptyContent { line: String, field: &'static str },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::EmptyInput => write!(f, "input contains no non-empty lines"),
            ParseError::InvalidCommand { line, error } => {
                write!(f, "invalid command on line {:?}: {}", line, error)
            }
            ParseError::EmptyContent { line, field } => {
                write!(f, "line {:?} is missing required {field} content", line)
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Error returned by [`text_to_inputs`](super::text_to_inputs).
#[derive(Debug)]
pub enum ConvertError {
    /// The model never produced valid JSON within the attempt limit.
    MaxAttemptsExceeded {
        last_error: String,
        last_output: String,
    },
    /// An inference call failed.
    InferenceError(String),
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvertError::MaxAttemptsExceeded {
                last_error,
                last_output,
            } => {
                write!(
                    f,
                    "converter exceeded attempt limit: {}\n  raw output: {:?}",
                    last_error,
                    if last_output.len() > 300 {
                        format!("{}…", &last_output[..300])
                    } else {
                        last_output.clone()
                    }
                )
            }
            ConvertError::InferenceError(msg) => {
                write!(f, "inference error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ConvertError {}
