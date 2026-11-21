//! Tests that each kind's answer grammar has the same order of size, and that every length told to the writer is the one it is checked against.
//!
//! The grammar for a story is the life event's with two more fields, so the tests hold the grammars to one order of size. Each field the grammar bounds must be named in the lengths the writer is told, and each name must be a field of the call. A list must have an end that fits inside the cap the whole list is checked against. An answer that is not its kind is refused, and each kind answers only with its own call or the call that declines.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use candle_conversation::models::DialectType;

    use candle_conversation::stencil::{compile_tool_call_tree, ToolCallEnvelope};

    use super::*;

    use super::super::tests_event::tests::keeper;

    /// **Each kind's answer grammar is of one order of size.** A story's call
    /// is the life event's with two more fields and a second list; its grammar
    /// must not be a different order of thing.
    #[test]
    fn the_answer_grammars_are_of_one_order_of_size() {
        let env = ToolCallEnvelope::for_dialect(&DialectType::ChatML.dialect())
            .with_max_string_tokens(1800);
        let size = |kind| {
            compile_tool_call_tree(&specs(kind), &env)
                .unwrap()
                .nodes
                .len()
        };
        let (life, story, pair) = (
            size(Kind::LifeEvent),
            size(Kind::Gap),
            size(Kind::Contradiction),
        );
        assert!(
            story <= life * 3 && pair <= life * 3,
            "life {life} nodes, story {story}, contradiction {pair}"
        );
    }

    /// **What the writer is told is what it is held to.** Every field the
    /// grammar bounds is named in [`lengths`], every name there is a field of
    /// the call, and the line says each cap.
    #[test]
    fn the_lengths_named_are_the_fields_bounded() {
        for kind in [Kind::LifeEvent, Kind::Contradiction, Kind::Gap] {
            let call = &specs(kind)[0];
            let bounded: Vec<&str> = call
                .params
                .iter()
                .filter(|p| {
                    p.max_tokens.is_some()
                        || p.items.as_ref().is_some_and(|i| i.max_tokens.is_some())
                })
                .map(|p| p.name.as_str())
                .collect();
            let named: Vec<&str> = caps(kind).iter().map(|(n, _)| *n).collect();
            let mut b = bounded.clone();
            let mut n = named.clone();
            b.sort_unstable();
            n.sort_unstable();
            assert_eq!(b, n, "{kind:?}: bounded fields and stated lengths differ");
            // A text field's grammar bound is its stated cap's, so the length
            // told and the length the grammar closes at cannot drift apart.
            for p in call.params.iter().filter(|p| p.items.is_none()) {
                if let Some((_, words)) = caps(kind).iter().find(|(n, _)| *n == p.name) {
                    assert_eq!(
                        p.max_tokens,
                        Some(tokens_for(*words)),
                        "{kind:?}: `{}` is bounded at another length than it is told",
                        p.name
                    );
                }
            }
            let line = lengths(kind);
            for (name, words) in caps(kind) {
                assert!(
                    line.contains(&format!("`{name}` under {words} words")),
                    "{kind:?}: {line}"
                );
            }
        }
    }

    pub(in super::super) fn call(name: &str, args: serde_json::Value) -> String {
        format!(
            "<tool_call>\n{}\n</tool_call>",
            serde_json::json!({ "name": name, "arguments": args })
        )
    }

    #[test]
    fn every_list_asked_for_has_an_end() {
        for kind in [Kind::LifeEvent, Kind::Contradiction, Kind::Gap] {
            for spec in specs(kind) {
                for p in spec.params.iter().filter(|p| p.ty == ParamType::Array) {
                    let max = p
                        .max_items
                        .unwrap_or_else(|| panic!("{} has no end", p.name));
                    assert!(p.min_items <= max && max <= FACTS_MOST, "{}", p.name);
                }
            }
        }
        // A full cast fits inside what `who` is checked against.
        const _: () = assert!(PEOPLE_MOST * PERSON_WORDS <= WHO_WORDS);
        let who = specs(Kind::Gap)[0]
            .params
            .iter()
            .find(|p| p.name == "who")
            .cloned()
            .unwrap();
        assert_eq!((who.min_items, who.max_items), (1, Some(PEOPLE_MOST)));
    }

    #[test]
    fn an_answer_that_is_not_its_kind_is_refused() {
        let (_dir, c, t) = keeper();
        assert!(check("I think Keeper should…", Kind::LifeEvent, &t, &c)
            .unwrap_err()
            .contains("not a call"));
        let story = call(STORY, serde_json::json!({ "title": "x" }));
        assert!(check(&story, Kind::LifeEvent, &t, &c)
            .unwrap_err()
            .contains("no call named `story`"));
    }

    #[test]
    fn each_kind_answers_with_its_own_call_or_declines() {
        let names = |k| -> Vec<String> { specs(k).into_iter().map(|s| s.name).collect() };
        assert_eq!(names(Kind::LifeEvent), [EVENT, NOTHING]);
        assert_eq!(names(Kind::Contradiction), [CORRECTION, NOTHING]);
        assert_eq!(names(Kind::Gap), [STORY, NOTHING]);
    }
}
