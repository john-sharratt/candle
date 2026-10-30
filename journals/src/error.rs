//! What can go wrong: a file that is broken, and an operation the rules
//! refuse. Both are answered as errors, naming where (§3.3).

use std::fmt;

use thiserror::Error;

/// Something wrong with a file in a journal repository: the file, the line
/// when one can be named, and what is wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// The file's path in its repository, `/`-separated.
    pub file: String,
    /// Its 1-based line, when the fault has one.
    pub line: Option<usize>,
    pub what: String,
}

impl Problem {
    pub fn new(file: impl Into<String>, line: Option<usize>, what: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            line,
            what: what.into(),
        }
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "{}:{}: {}", self.file, line, self.what),
            None => write!(f, "{}: {}", self.file, self.what),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// A file is broken; the journal it belongs to cannot be read or written
    /// until another commit fixes it.
    #[error("{0}")]
    Broken(Problem),
    /// The request names something that is not well formed.
    #[error("{0}")]
    Invalid(String),
    /// What it names does not exist.
    #[error("{0}")]
    NotFound(String),
    /// The caller may not do this.
    #[error("{0}")]
    Forbidden(String),
    /// The rules refuse it as things stand: the eleventh sub-journal, a cap.
    #[error("{0}")]
    Refused(String),
    /// What it was based on changed underneath it; read again and retry.
    #[error("{0}")]
    Conflict(String),
}

impl Error {
    pub fn invalid(what: impl Into<String>) -> Self {
        Self::Invalid(what.into())
    }
    pub fn not_found(what: impl Into<String>) -> Self {
        Self::NotFound(what.into())
    }
    pub fn forbidden(what: impl Into<String>) -> Self {
        Self::Forbidden(what.into())
    }
    pub fn refused(what: impl Into<String>) -> Self {
        Self::Refused(what.into())
    }
    pub fn conflict(what: impl Into<String>) -> Self {
        Self::Conflict(what.into())
    }
}

impl From<Problem> for Error {
    fn from(p: Problem) -> Self {
        Self::Broken(p)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_problem_names_its_file_and_line() {
        assert_eq!(
            Problem::new("projects/candle/lines.yaml", Some(3), "`text` is missing").to_string(),
            "projects/candle/lines.yaml:3: `text` is missing"
        );
        assert_eq!(
            Problem::new("family.yaml", None, "no member is an owner").to_string(),
            "family.yaml: no member is an owner"
        );
        assert_eq!(
            Error::from(Problem::new("a", Some(1), "b")).to_string(),
            "a:1: b"
        );
    }
}
