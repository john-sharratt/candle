//! A character's act sent to its world's authority
//! (`docs/replicated_substrate.md` §9.3): a character lent to a member that
//! does not hold the world sends each act that would change it to the
//! authority as `POST /v1/world/:wid/act`, and the authority performs it and
//! answers with the world's verdict. An act that changes only the character's
//! own mind stays where the character is.
//!
//! The act travels as the one-line call the character made — `{"tool": …,
//! args…}` — so the authority reads it with the same parser and the same
//! checks as a call decoded there; the verdict travels as `{kind, line}`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::engine::act::{parse, Act};
use crate::engine::body::Outcome;

/// The body of `POST /v1/world/:wid/act`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActRequest {
    /// The body acting, in the world the path names.
    pub body: String,
    /// The act, as the call line the character made.
    pub call: String,
}

/// `act` as the call line it was made as.
pub fn call_line(act: &Act) -> String {
    let mut call: Map<String, Value> = Map::new();
    call.insert("tool".into(), Value::String(act.tool.to_string()));
    for (k, v) in &act.args {
        call.insert(k.clone(), v.clone());
    }
    Value::Object(call).to_string()
}

/// The act a call line makes, as the parser reads it; the reason when it is
/// not one act.
pub fn act_of(call: &str) -> Result<Act, String> {
    let mut parsed = parse(call);
    match (parsed.acts.len(), parsed.rejected.first()) {
        (1, _) => Ok(parsed.acts.remove(0)),
        (_, Some(r)) => Err(r.line()),
        _ => Err("the call was not one act".into()),
    }
}

/// The world's verdict as it travels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Verdict {
    Did { line: String },
    Departed { line: String },
    Refused { line: String },
    NotOfTheBody,
}

impl From<&Outcome> for Verdict {
    fn from(o: &Outcome) -> Self {
        match o {
            Outcome::Did(line) => Verdict::Did { line: line.clone() },
            Outcome::Departed(line) => Verdict::Departed { line: line.clone() },
            Outcome::Refused(line) => Verdict::Refused { line: line.clone() },
            Outcome::NotOfTheBody => Verdict::NotOfTheBody,
        }
    }
}

impl From<Verdict> for Outcome {
    fn from(v: Verdict) -> Self {
        match v {
            Verdict::Did { line } => Outcome::Did(line),
            Verdict::Departed { line } => Outcome::Departed(line),
            Verdict::Refused { line } => Outcome::Refused(line),
            Verdict::NotOfTheBody => Outcome::NotOfTheBody,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// An act sent as its call line is the same act at the other end.
    #[test]
    fn an_act_round_trips_through_its_call_line() {
        let act = act_of(r#"{"tool": "shout", "intent": "over here"}"#).unwrap();
        let line = call_line(&act);
        assert_eq!(act_of(&line).unwrap(), act);
        assert_eq!(
            serde_json::from_str::<Value>(&line).unwrap(),
            json!({"tool": "shout", "intent": "over here"})
        );
    }

    #[test]
    fn a_line_that_is_not_one_act_says_why() {
        assert!(act_of("{not json").is_err());
        assert!(act_of(r#"{"tool": "no_such_tool"}"#).is_err());
    }

    /// The verdict keeps its kind and its line, as JSON.
    #[test]
    fn a_verdict_travels_as_kind_and_line() {
        let v = Verdict::from(&Outcome::Refused("There is no such room.".into()));
        let wire = serde_json::to_value(&v).unwrap();
        assert_eq!(
            wire,
            json!({"kind": "refused", "line": "There is no such room."})
        );
        let back: Verdict = serde_json::from_value(wire).unwrap();
        assert_eq!(
            Outcome::from(back),
            Outcome::Refused("There is no such room.".into())
        );
        assert_eq!(
            serde_json::to_value(Verdict::NotOfTheBody).unwrap(),
            json!({"kind": "not_of_the_body"})
        );
    }
}
