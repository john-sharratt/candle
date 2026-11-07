//! The values a turn's calls may give the fields whose values are known
//! (`docs/stencil_tree.md` §9.2): a file's anchors, a part's lines. A pattern
//! holds a value to a shape; a held value holds it to what exists, so a value
//! that names nothing cannot be written.
//!
//! [`ValueHolds`] is what a caller states — plain strings, no vocabulary.
//! [`HeldValues`] is what a turn walks: each held set built into a
//! [`ValueRule`] over the literal values, owned by the turn and dropped with
//! it ([`ValueRule::owned`]). Never [`ValueRule::shared`], whose cache keeps
//! every pattern for the process's life and would grow by one per round.

use std::collections::HashMap;
use std::mem;
use std::sync::Arc;

use super::value_rule::ValueRule;
use super::vocab::Vocab;

/// One field's held values, as a caller states them: for `tools` (a tool and
/// the aliases it answers to), `field` takes one of the values its table
/// gives for what the call has already written in the fields `on`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldHold {
    pub tools: Vec<String>,
    pub field: String,
    /// The fields, written earlier in the same call, the values depend on;
    /// empty when they depend on nothing.
    pub on: Vec<String>,
    /// For each combination of `on`'s values, in `on`'s order, the values the
    /// field may take.
    pub table: HashMap<Vec<String>, Vec<String>>,
}

/// What a turn holds its calls' values to, as stated.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValueHolds {
    pub fields: Vec<FieldHold>,
}

/// One field's held rules, built.
#[derive(Debug)]
struct Held {
    on: Vec<Arc<str>>,
    rules: HashMap<Vec<String>, Arc<ValueRule>>,
}

/// What a turn holds its calls' values to, built over a vocabulary.
#[derive(Debug, Default)]
pub struct HeldValues {
    by_field: HashMap<(String, Arc<str>), Held>,
}

/// The rules the last [`HeldValues::build`] made, by value set, so a set that
/// has not changed since — a file no edit touched, a part of the same length —
/// is not built again. It holds only the sets that build used, so it never
/// grows past one turn's.
#[derive(Debug, Default)]
pub struct HeldRules {
    rules: HashMap<Vec<String>, Option<Arc<ValueRule>>>,
}

impl HeldRules {
    /// How many value sets it holds.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// A pattern matching exactly `values`: each escaped, alternated.
pub fn literal_pattern(values: &[String]) -> String {
    let arms: Vec<String> = values.iter().map(|v| escape(v)).collect();
    format!("(?:{})", arms.join("|"))
}

/// `value` with every character a pattern reads as an operator escaped.
fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if "\\.+*?()|[]{}^$#&-~".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

impl HeldValues {
    /// `holds` built over `vocab`, taking each set's rule from `kept` when
    /// the last build made it, and leaving `kept` with this build's. A set with
    /// no values, or one whose rule cannot be built (past the rule's state
    /// cap), is left out: its field keeps its own pattern.
    pub fn build(holds: &ValueHolds, vocab: &dyn Vocab, kept: &mut HeldRules) -> HeldValues {
        let mut by_field = HashMap::new();
        let mut last = mem::take(&mut kept.rules);
        // One rule per distinct set: every part of three lines has the same.
        let built = &mut kept.rules;
        for hold in &holds.fields {
            let field: Arc<str> = Arc::from(hold.field.as_str());
            let mut rules = HashMap::new();
            for (given, values) in &hold.table {
                if values.is_empty() {
                    continue;
                }
                let rule = built.entry(values.clone()).or_insert_with(|| {
                    last.remove(values).unwrap_or_else(|| {
                        ValueRule::owned(&literal_pattern(values), vocab)
                            .map_err(|e| {
                                tracing::debug!(field = %hold.field, "a held value set was left to its pattern: {e}");
                            })
                            .ok()
                            .map(Arc::new)
                    })
                });
                if let Some(rule) = rule {
                    rules.insert(given.clone(), Arc::clone(rule));
                }
            }
            if rules.is_empty() {
                continue;
            }
            let on: Vec<Arc<str>> = hold.on.iter().map(|f| Arc::from(f.as_str())).collect();
            for tool in &hold.tools {
                by_field.insert(
                    (tool.clone(), Arc::clone(&field)),
                    Held {
                        on: on.clone(),
                        rules: rules.clone(),
                    },
                );
            }
        }
        HeldValues { by_field }
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.by_field.is_empty()
    }

    /// The rule `tool`'s `field` is held to, given what the call has written
    /// so far; `None` when it is not held, or the fields it depends on name
    /// nothing held.
    pub fn rule_for(
        &self,
        tool: &str,
        field: &Arc<str>,
        written: &HashMap<Arc<str>, String>,
    ) -> Option<Arc<ValueRule>> {
        let held = self.by_field.get(&(tool.to_string(), Arc::clone(field)))?;
        let given: Option<Vec<String>> = held.on.iter().map(|f| written.get(f).cloned()).collect();
        held.rules.get(&given?).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::super::vocab::TestVocab;
    use super::*;

    /// **A literal pattern matches its values and nothing else**, whatever
    /// operators they hold — exactly these bytes.
    #[test]
    fn a_literal_pattern_escapes_its_values() {
        assert_eq!(
            literal_pattern(&["+1".into(), "+12".into(), "a-b".into()]),
            r"(?:\+1|\+12|a\-b)"
        );
        assert_eq!(literal_pattern(&["KehT".into()]), "(?:KehT)");
    }

    /// **Every operator the pattern reads is escaped**, not only the two the
    /// test above happens to use — and a value of plain letters and digits is
    /// left exactly as it stands.
    ///
    /// The escape list is `"\\.+*?()|[]{}^$#&-~"`; a value holding one of them
    /// unescaped would let that character act on the whole alternation, so a
    /// held value like `a(b` would match `ab` and hold the model to a shape it
    /// never agreed to.
    #[test]
    fn every_operator_in_a_value_is_escaped() {
        // Each operator alone, and the exact bytes it must come out as.
        for (op, expected) in [
            ("\\", r"(?:\\)"),
            (".", r"(?:\.)"),
            ("+", r"(?:\+)"),
            ("*", r"(?:\*)"),
            ("?", r"(?:\?)"),
            ("(", r"(?:\()"),
            (")", r"(?:\))"),
            ("|", r"(?:\|)"),
            ("[", r"(?:\[)"),
            ("]", r"(?:\])"),
            ("{", r"(?:\{)"),
            ("}", r"(?:\})"),
            ("^", r"(?:\^)"),
            ("$", r"(?:\$)"),
            ("#", r"(?:\#)"),
            ("&", r"(?:\&)"),
            ("-", r"(?:\-)"),
            ("~", r"(?:\~)"),
        ] {
            assert_eq!(
                literal_pattern(&[op.to_string()]),
                expected,
                "operator {op:?}"
            );
        }
        // All of them at once, in one value, in the order the escape list
        // holds: the leading backslash escapes to two, and every operator
        // after it to itself with one backslash before it.
        assert_eq!(
            literal_pattern(&[r"\.+*?()|[]{}^$#&-~".to_string()]),
            r"(?:\\\.\+\*\?\(\)\|\[\]\{\}\^\$\#\&\-\~)"
        );
        // A value of plain letters and digits is left as it is, and the two
        // arms are joined with a bare `|`.
        assert_eq!(
            literal_pattern(&["main2".to_string(), "Zen42".to_string()]),
            "(?:main2|Zen42)"
        );
    }

    fn parts(table: &[(&str, &[&str])]) -> ValueHolds {
        ValueHolds {
            fields: vec![FieldHold {
                tools: vec!["file_edit".into()],
                field: "part".into(),
                on: vec!["path".into()],
                table: table
                    .iter()
                    .map(|(path, values)| {
                        (
                            vec![path.to_string()],
                            values.iter().map(|v| v.to_string()).collect(),
                        )
                    })
                    .collect(),
            }],
        }
    }

    /// **A set unchanged since the last build is the same rule**, a changed
    /// one is built afresh, and a set the build no longer uses is let go.
    #[test]
    fn an_unchanged_set_keeps_its_rule_across_builds() {
        let v = TestVocab::new();
        let mut kept = HeldRules::default();
        let field: Arc<str> = Arc::from("part");
        let rule = |held: &HeldValues, path: &str| {
            let written = HashMap::from([(Arc::from("path"), path.to_string())]);
            held.rule_for("file_edit", &field, &written).unwrap()
        };
        let first = HeldValues::build(
            &parts(&[("a.rs", &["KehT"]), ("b.rs", &["r7Lm"])]),
            &v,
            &mut kept,
        );
        assert_eq!(kept.len(), 2);
        let second = HeldValues::build(
            &parts(&[("a.rs", &["KehT"]), ("b.rs", &["r7Lm", "Zq0p"])]),
            &v,
            &mut kept,
        );
        assert!(Arc::ptr_eq(&rule(&first, "a.rs"), &rule(&second, "a.rs")));
        assert!(!Arc::ptr_eq(&rule(&first, "b.rs"), &rule(&second, "b.rs")));
        assert_eq!(kept.len(), 2, "`r7Lm` alone is no longer used");
    }
}
