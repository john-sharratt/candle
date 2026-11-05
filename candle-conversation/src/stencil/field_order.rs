//! The order a call's fields are written in, and what may follow each one —
//! the one statement of it the grammar compiles (`tool_call`'s
//! `build_sequence` and `opt_gates`) and the function-block readers follow
//! (`element_close::follows_in`).
//!
//! Required fields come first, in schema order, every one of them; then the
//! optionals, in schema order, any of them skipped. A choice among optionals
//! ([`Pick`]) withholds the close until a member is written, and an
//! `ExactlyOne` member written withholds every other member.

use super::element_close::Follows;
use super::tool_call::{Param, Pick, ToolSpec};

/// Where an object stands on the choice among its optionals ([`Pick`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Owed {
    /// No choice, or one that is made and allows more.
    Nothing,
    /// No member written yet: the close is withheld.
    AMember,
    /// An `ExactlyOne` member is written: no other is offered.
    NoMore,
}

impl Owed {
    /// The standing before any of `optional` is written.
    pub(crate) fn initial(optional: &[&Param]) -> Owed {
        match optional.iter().any(|p| p.pick != Pick::None) {
            true => Owed::AMember,
            false => Owed::Nothing,
        }
    }

    /// The standing once an optional of choice `pick` is written.
    pub(crate) fn after(self, pick: Pick) -> Owed {
        match pick {
            Pick::ExactlyOne => Owed::NoMore,
            Pick::AtLeastOne => Owed::Nothing,
            Pick::None => self,
        }
    }

    /// Whether optional `j` of `optional` is offered from this standing.
    ///
    /// **A standing that owes a member can always pay it.** While one is owed,
    /// nothing past the last member is offered, so a walk cannot step over the
    /// choice and arrive at a gate with no way on.
    pub(crate) fn offers(self, optional: &[&Param], j: usize) -> bool {
        match self {
            Owed::AMember => optional
                .iter()
                .rposition(|p| p.pick != Pick::None)
                .is_some_and(|last| j <= last),
            Owed::NoMore => optional[j].pick == Pick::None,
            Owed::Nothing => true,
        }
    }

    /// Whether the object may close from this standing.
    pub(crate) fn may_close(self) -> bool {
        self != Owed::AMember
    }
}

/// What may follow the value of the last of `written`, the fields of a call
/// to `spec` in the order they were written: another field, the close, or
/// either. `None` when the order is not one this rule knows — a field the
/// schema does not declare, or a schema whose enum values decide the rest of
/// the object (`Param::requires`, `Param::shapes`).
pub(crate) fn follows_after(spec: &ToolSpec, written: &[&str]) -> Option<Follows> {
    if spec
        .params
        .iter()
        .any(|p| !p.requires.is_empty() || !p.shapes.is_empty())
    {
        return None;
    }
    let required: Vec<&Param> = spec.params.iter().filter(|p| p.required).collect();
    let optional: Vec<&Param> = spec.params.iter().filter(|p| !p.required).collect();
    let key = *written.last()?;
    let mut owed = Owed::initial(&optional);
    let from = match required.iter().position(|p| p.name == key) {
        Some(at) if at + 1 < required.len() => return Some(Follows::Element),
        Some(_) => 0,
        None => {
            for name in written {
                if let Some(p) = optional.iter().find(|p| p.name == *name) {
                    owed = owed.after(p.pick);
                }
            }
            optional.iter().position(|p| p.name == key)? + 1
        }
    };
    let element = (from..optional.len()).any(|j| owed.offers(&optional, j));
    Some(match (element, owed.may_close()) {
        (true, true) => Follows::Either,
        (true, false) => Follows::Element,
        (false, _) => Follows::Close,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::stencil::parse_tools;

    fn spec(json: &str) -> ToolSpec {
        parse_tools(json).unwrap().remove(0)
    }

    /// **Required fields are followed by the next, the last by the optionals
    /// or the close, and the last optional by the close alone.**
    #[test]
    fn each_field_is_followed_by_what_its_place_allows() {
        let write = spec(
            r#"[{"name":"file_write","params":[
                {"name":"path","type":"string","required":true},
                {"name":"mode","type":"string","required":false},
                {"name":"content","type":"string","required":true},
                {"name":"note","type":"string","required":false}]}]"#,
        );
        let after = |w: &[&str]| follows_after(&write, w);
        assert_eq!(after(&["path"]), Some(Follows::Element));
        assert_eq!(after(&["path", "content"]), Some(Follows::Either));
        assert_eq!(after(&["path", "content", "mode"]), Some(Follows::Either));
        assert_eq!(after(&["path", "content", "note"]), Some(Follows::Close));
        assert_eq!(after(&["path", "content", "mode", "note"]), Some(Follows::Close));
        assert_eq!(after(&["path", "stray"]), None);
        assert_eq!(after(&[]), None);
    }

    /// **A choice among optionals withholds the close until a member is
    /// written**, and an `ExactlyOne` member written withholds the others,
    /// where an `AtLeastOne` member leaves them offered.
    #[test]
    fn a_choice_among_optionals_is_followed_as_the_grammar_offers_it() {
        for (rule, after_a) in [("anyOf", Follows::Either), ("oneOf", Follows::Close)] {
            let mut schema = json!({
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "a": {"type": "string"},
                    "b": {"type": "string"}
                },
                "required": ["id"]
            });
            schema[rule] = json!([{"required": ["a"]}, {"required": ["b"]}]);
            let tool = ToolSpec::from_json_schema("t", &schema);
            let after = |w: &[&str]| follows_after(&tool, w);
            assert_eq!(after(&["id"]), Some(Follows::Element), "{rule}");
            assert_eq!(after(&["id", "a"]), Some(after_a), "{rule}");
            assert_eq!(after(&["id", "b"]), Some(Follows::Close), "{rule}");
        }
    }
}
