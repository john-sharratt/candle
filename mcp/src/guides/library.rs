//! Every guide: the catalogue `guide_list` answers, one guide's body for
//! `guide_read`, and the line each tool lists its guides on.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::guide::Guide;

/// A tool listed by more guides than this names them only, without their
/// one-liners, so its description stays short; `guide_list` has the rest.
pub const NAMES_ONLY_PAST: usize = 4;

/// Every guide, by name.
#[derive(Debug, Clone)]
pub struct Library {
    guides: BTreeMap<String, Guide>,
    /// The guides each tool is listed by, in name order.
    by_tool: BTreeMap<String, Vec<String>>,
}

impl Library {
    /// The library `guides` make; a name given twice is refused.
    pub fn new(guides: Vec<Guide>) -> anyhow::Result<Self> {
        let mut by_name = BTreeMap::new();
        let mut by_tool: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for g in guides {
            for tool in &g.tools {
                by_tool
                    .entry(tool.clone())
                    .or_default()
                    .push(g.name.clone());
            }
            if let Some(twice) = by_name.insert(g.name.clone(), g) {
                anyhow::bail!("two guides are named `{}`", twice.name);
            }
        }
        for names in by_tool.values_mut() {
            names.sort();
        }
        Ok(Self {
            guides: by_name,
            by_tool,
        })
    }

    /// How many guides it holds.
    pub fn count(&self) -> usize {
        self.guides.len()
    }

    pub fn get(&self, name: &str) -> Option<&Guide> {
        self.guides.get(name)
    }

    /// Every guide's name, in order.
    pub fn names(&self) -> Vec<&str> {
        self.guides.keys().map(String::as_str).collect()
    }

    /// The catalogue: each guide's name and when to use it.
    pub fn catalogue(&self) -> String {
        let mut out = String::from(
            "Guides — the methods for doing work well with these tools. Read one with \
             guide_read {name} before starting work of its kind.\n",
        );
        for g in self.guides.values() {
            out.push_str(&format!("\n- {} — {}", g.name, g.description));
        }
        out
    }

    /// The line a tool lists its guides on, or none when no guide lists it.
    pub fn line_for(&self, tool: &str) -> Option<String> {
        let names = self.by_tool.get(tool)?;
        if names.len() > NAMES_ONLY_PAST {
            return Some(format!(
                "Guides: {} — read one with guide_read before you start; guide_list says what \
                 each is for.",
                names.join(", ")
            ));
        }
        let each: Vec<String> = names
            .iter()
            .filter_map(|n| self.guides.get(n))
            .map(|g| format!("{} — {}", g.name, g.description.trim_end_matches('.')))
            .collect();
        Some(format!(
            "Guides: {}. Read one with guide_read before you start.",
            each.join("; ")
        ))
    }

    /// Append to each of `tools` the line listing its guides.
    pub fn decorate(&self, tools: &mut [Value]) {
        for tool in tools {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            let Some(line) = self.line_for(name) else {
                continue;
            };
            let described = tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim_end()
                .to_string();
            tool["description"] = Value::String(if described.is_empty() {
                line
            } else {
                format!("{described}\n\n{line}")
            });
        }
    }

    /// The tools a guide lists that none of `served` is: a guide that names
    /// a tool no backend serves, each with the guides that name it.
    pub fn unserved(&self, served: &BTreeSet<&str>) -> Vec<(String, Vec<String>)> {
        self.by_tool
            .iter()
            .filter(|(tool, _)| !served.contains(tool.as_str()))
            .map(|(tool, names)| (tool.clone(), names.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn guide(name: &str, tools: &[&str]) -> Guide {
        Guide {
            name: name.into(),
            description: format!("{name} work."),
            tools: tools.iter().map(|t| t.to_string()).collect(),
            body: format!("# {name}"),
        }
    }

    #[test]
    fn a_name_given_twice_is_refused() {
        let twice = Library::new(vec![guide("a", &["t"]), guide("a", &["u"])]).unwrap_err();
        assert_eq!(twice.to_string(), "two guides are named `a`");
    }

    #[test]
    fn the_catalogue_lists_each_guide_with_when_to_use_it() {
        let l = Library::new(vec![guide("research", &["t"]), guide("deploy", &["u"])]).unwrap();
        assert!(l
            .catalogue()
            .ends_with("\n- deploy — deploy work.\n- research — research work."));
    }

    /// **A tool names its guides with what each is for, until it is listed by
    /// many** — then by name only, so its description stays short.
    #[test]
    fn a_tool_lists_its_guides_by_line_and_by_name_past_a_few() {
        let few = Library::new(vec![guide("b", &["t"]), guide("a", &["t"])]).unwrap();
        assert_eq!(
            few.line_for("t").unwrap(),
            "Guides: a — a work; b — b work. Read one with guide_read before you start."
        );
        assert_eq!(few.line_for("other"), None);
        let many = Library::new(
            ["e", "d", "c", "b", "a"]
                .iter()
                .map(|n| guide(n, &["t"]))
                .collect(),
        )
        .unwrap();
        assert_eq!(
            many.line_for("t").unwrap(),
            "Guides: a, b, c, d, e — read one with guide_read before you start; guide_list says \
             what each is for."
        );
    }

    #[test]
    fn a_listed_tool_has_its_guides_appended_and_the_others_are_untouched() {
        let l = Library::new(vec![guide("a", &["t"])]).unwrap();
        let mut tools = vec![
            json!({"name": "t", "description": "Does t.  "}),
            json!({"name": "u", "description": "Does u."}),
        ];
        l.decorate(&mut tools);
        assert_eq!(
            tools[0]["description"],
            "Does t.\n\nGuides: a — a work. Read one with guide_read before you start."
        );
        assert_eq!(tools[1]["description"], "Does u.");
    }

    #[test]
    fn a_tool_no_backend_serves_is_named_with_its_guides() {
        let l = Library::new(vec![guide("a", &["t", "gone"])]).unwrap();
        let served = BTreeSet::from(["t"]);
        assert_eq!(
            l.unserved(&served),
            vec![("gone".to_string(), vec!["a".to_string()])]
        );
    }
}
