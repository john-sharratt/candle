//! The most of a working set's budget one type of document may hold
//! (`docs/zend_working_set.md` §4.10).
//!
//! A unit's type is the first of the declared types whose patterns match its
//! subject — `<repo>/<path>`, the path a `code_reading` unit is of. A unit
//! matching none is unbounded. Each type holds at most `max_share` of
//! `budget_tokens`; provenance orders everything else.

use regex::Regex;

/// The type a unit is, and the most of the budget its type may hold, in
/// thousandths — what a working-set member carries, so the cap travels with
/// it (`WorkingSet::observe`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeCap {
    /// The type's position among the declared types.
    pub id: u8,
    /// `max_share × 1000`, rounded.
    pub per_mille: u16,
}

impl TypeCap {
    /// The tokens this type may hold of a budget of `budget_tokens`.
    pub fn tokens(&self, budget_tokens: usize) -> usize {
        budget_tokens * self.per_mille as usize / 1000
    }
}

/// One declared type.
#[derive(Debug, Clone)]
struct TypeShare {
    name: String,
    files: Vec<Regex>,
    per_mille: u16,
}

/// A `working_set:` block's `type_shares`, compiled. Empty: every unit is
/// unbounded.
#[derive(Debug, Clone, Default)]
pub struct TypeShares {
    types: Vec<TypeShare>,
}

/// One type as the schema declares it: its name, its subject patterns, and
/// its maximum share of the budget.
pub type TypeShareSpec = (String, Vec<String>, f32);

impl TypeShares {
    /// Compile `specs`, refusing a share outside `(0, 1]`, a name used twice
    /// or empty, a type with no pattern, a pattern that does not compile, and
    /// more types than a member can name.
    pub fn new(specs: &[TypeShareSpec]) -> Result<Self, String> {
        if specs.len() > u8::MAX as usize {
            return Err(format!(
                "{} types — at most {} may be declared",
                specs.len(),
                u8::MAX
            ));
        }
        let mut types: Vec<TypeShare> = Vec::with_capacity(specs.len());
        for (name, files, max_share) in specs {
            if name.is_empty() {
                return Err("a type needs a name".into());
            }
            if types.iter().any(|t| t.name == *name) {
                return Err(format!("type {name:?} is declared twice"));
            }
            if !(*max_share > 0.0 && *max_share <= 1.0) {
                return Err(format!(
                    "type {name:?}: max_share {max_share} must be in (0, 1]"
                ));
            }
            if files.is_empty() {
                return Err(format!("type {name:?} names no files pattern"));
            }
            let files = files
                .iter()
                .map(|p| Regex::new(p).map_err(|e| format!("type {name:?}: pattern {p:?}: {e}")))
                .collect::<Result<Vec<_>, _>>()?;
            types.push(TypeShare {
                name: name.clone(),
                files,
                per_mille: ((max_share * 1000.0).round() as u16).max(1),
            });
        }
        Ok(Self { types })
    }

    /// The type of the unit whose subject is `subject`: the first declared
    /// type a pattern of which matches it, `None` when none does.
    pub fn classify(&self, subject: &str) -> Option<TypeCap> {
        self.types
            .iter()
            .position(|t| t.files.iter().any(|r| r.is_match(subject)))
            .map(|i| TypeCap {
                id: i as u8,
                per_mille: self.types[i].per_mille,
            })
    }

    /// The declared types' names, in order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.types.iter().map(|t| t.name.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

/// Two compilations of the same declaration are equal: same names, patterns
/// and shares, in the same order.
impl PartialEq for TypeShares {
    fn eq(&self, other: &Self) -> bool {
        self.types.len() == other.types.len()
            && self.types.iter().zip(&other.types).all(|(a, b)| {
                a.name == b.name
                    && a.per_mille == b.per_mille
                    && a.files.len() == b.files.len()
                    && a.files
                        .iter()
                        .zip(&b.files)
                        .all(|(x, y)| x.as_str() == y.as_str())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, files: &[&str], share: f32) -> TypeShareSpec {
        (
            name.to_string(),
            files.iter().map(|s| s.to_string()).collect(),
            share,
        )
    }

    fn production() -> TypeShares {
        TypeShares::new(&[
            spec("docs", &[r"\.(md|markdown|mdx|txt|rst|adoc)$"], 0.35),
            spec(
                "config",
                &[
                    r"\.(ya?ml|toml|json|jsonc|ini|cfg|conf|env|properties)$",
                    r"(^|/)(Cargo\.toml|package\.json|pyproject\.toml|go\.mod)$",
                ],
                0.15,
            ),
        ])
        .unwrap()
    }

    /// **A subject is the first type it matches, or none**: documents and
    /// configuration are typed, code is unbounded.
    #[test]
    fn a_subject_is_the_first_type_it_matches() {
        let t = production();
        let docs = Some(TypeCap {
            id: 0,
            per_mille: 350,
        });
        let config = Some(TypeCap {
            id: 1,
            per_mille: 150,
        });
        assert_eq!(t.classify("candle/README.md"), docs);
        assert_eq!(t.classify("candle/docs/zend_working_set.md"), docs);
        assert_eq!(t.classify("candle/mcp/src/guides/zend-thread.md"), docs);
        assert_eq!(
            t.classify("candle/zend/src/prompts/projection.yaml"),
            config
        );
        assert_eq!(t.classify("candle/Cargo.toml"), config);
        assert_eq!(t.classify("candle/go.mod"), config);
        assert_eq!(t.classify("candle/zend/src/session.rs"), None);
        assert_eq!(
            t.classify("candle/candle-kernels/src/provenance/bdp_scan.cu"),
            None
        );
        assert_eq!(t.classify("candle/docs/"), None, "a folder is no document");
    }

    /// **The first match wins** where two types would both take a subject.
    #[test]
    fn the_first_declared_type_wins() {
        let t =
            TypeShares::new(&[spec("a", &[r"\.md$"], 0.5), spec("b", &[r"README"], 0.2)]).unwrap();
        assert_eq!(t.classify("x/README.md").map(|c| c.id), Some(0));
        assert_eq!(t.classify("x/README").map(|c| c.id), Some(1));
    }

    /// **A share is a fraction of the budget**, in whole tokens.
    #[test]
    fn a_share_is_a_fraction_of_the_budget() {
        let docs = TypeCap {
            id: 0,
            per_mille: 350,
        };
        assert_eq!(docs.tokens(64_000), 22_400);
        assert_eq!(docs.tokens(0), 0);
        let one = TypeShares::new(&[spec("all", &["."], 1.0)]).unwrap();
        assert_eq!(one.classify("a").unwrap().tokens(64_000), 64_000);
    }

    /// **A declaration the arithmetic cannot use is refused**, saying why.
    #[test]
    fn a_bad_declaration_is_refused() {
        let err = |specs: &[TypeShareSpec]| TypeShares::new(specs).unwrap_err();
        assert_eq!(
            err(&[spec("docs", &["x"], 0.0)]),
            "type \"docs\": max_share 0 must be in (0, 1]"
        );
        assert_eq!(
            err(&[spec("docs", &["x"], 1.5)]),
            "type \"docs\": max_share 1.5 must be in (0, 1]"
        );
        assert_eq!(
            err(&[spec("docs", &["x"], 0.2), spec("docs", &["y"], 0.2)]),
            "type \"docs\" is declared twice"
        );
        assert_eq!(err(&[spec("", &["x"], 0.2)]), "a type needs a name");
        assert_eq!(
            err(&[spec("docs", &[], 0.2)]),
            "type \"docs\" names no files pattern"
        );
        assert!(err(&[spec("docs", &["("], 0.2)]).starts_with("type \"docs\": pattern \"(\": "));
    }

    /// Two compilations of one declaration are equal; another share is not.
    #[test]
    fn equal_declarations_compile_equal() {
        assert_eq!(production(), production());
        let other = TypeShares::new(&[spec("docs", &[r"\.md$"], 0.3)]).unwrap();
        assert_ne!(production(), other);
        assert_eq!(
            production().names().collect::<Vec<_>>(),
            vec!["docs", "config"]
        );
        assert!(TypeShares::default().is_empty());
    }
}
