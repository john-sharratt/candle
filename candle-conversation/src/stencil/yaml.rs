//! Front-end C — a declarative YAML node spec.
//!
//! Maps 1:1 to the label-based intermediate, so a hand-authored tree compiles
//! the same way the builder or the tool compiler would.

use serde::Deserialize;

use super::error::BuildError;
use super::spec::{LabeledNode, LabeledTree, TreeSpec};
use super::terminator::Terminator;
use super::tree::FreeTextLimits;

#[derive(Deserialize)]
struct YamlTree {
    label: String,
    root: String,
    nodes: Vec<YamlNode>,
}

#[derive(Deserialize)]
struct YamlNode {
    id: String,
    #[serde(rename = "static", default)]
    static_text: Option<String>,
    #[serde(default)]
    next: Option<String>,
    #[serde(default)]
    branch: Option<Vec<YamlArm>>,
    #[serde(default)]
    free_text: Option<YamlFreeText>,
    #[serde(default)]
    end: Option<bool>,
}

#[derive(Deserialize)]
struct YamlArm {
    #[serde(rename = "match")]
    matches: String,
    next: String,
}

#[derive(Deserialize)]
struct YamlFreeText {
    terminator: String,
    #[serde(default)]
    open: Option<String>,
    #[serde(default)]
    close: Option<String>,
    #[serde(default)]
    eos_ends: bool,
    #[serde(default)]
    limits: Option<YamlLimits>,
}

#[derive(Deserialize)]
struct YamlLimits {
    forced_after: u32,
    #[serde(default)]
    ramp_start: Option<u32>,
    #[serde(default)]
    ramp_len: Option<u32>,
    #[serde(default)]
    boost: Option<f32>,
}

impl TreeSpec {
    /// Parse a declarative YAML tree into a (still untokenized) [`TreeSpec`].
    pub fn from_yaml(yaml: &str) -> Result<TreeSpec, BuildError> {
        let parsed: YamlTree =
            serde_yaml::from_str(yaml).map_err(|e| BuildError::Yaml(e.to_string()))?;
        let mut nodes = Vec::with_capacity(parsed.nodes.len());
        for n in parsed.nodes {
            nodes.push((n.id.clone(), n.into_labeled()?));
        }
        LabeledTree {
            label: parsed.label,
            root: parsed.root,
            nodes,
        }
        .resolve()
    }
}

impl YamlNode {
    fn into_labeled(self) -> Result<LabeledNode, BuildError> {
        let YamlNode {
            id,
            static_text,
            next,
            branch,
            free_text,
            end,
        } = self;
        let kinds = static_text.is_some() as u8
            + branch.is_some() as u8
            + free_text.is_some() as u8
            + end.unwrap_or(false) as u8;
        if kinds != 1 {
            return Err(BuildError::Yaml(format!(
                "node {id:?} must declare exactly one of static/branch/free_text/end"
            )));
        }
        let need_next = |kind: &str| {
            next.clone()
                .ok_or_else(|| BuildError::Yaml(format!("{kind} node {id:?} needs `next`")))
        };
        if let Some(text) = static_text {
            return Ok(LabeledNode::Static {
                text,
                next: need_next("static")?,
            });
        }
        if let Some(arms) = branch {
            return Ok(LabeledNode::Branch {
                arms: arms.into_iter().map(|a| (a.matches, a.next)).collect(),
            });
        }
        if let Some(ft) = free_text {
            let term = parse_terminator(&ft)?;
            let limits = ft
                .limits
                .map(Into::into)
                .unwrap_or_else(FreeTextLimits::json_string);
            return Ok(LabeledNode::FreeText {
                term,
                eos_ends: ft.eos_ends,
                limits,
                // YAML trees are string-space; a token-closed span (the thinking
                // block) is built programmatically with the resolved id, not here.
                close_token: None,
                suppress_close: false,
                next: need_next("free_text")?,
            });
        }
        Ok(LabeledNode::End)
    }
}

fn parse_terminator(ft: &YamlFreeText) -> Result<Terminator, BuildError> {
    match ft.terminator.as_str() {
        "json_string" => Ok(Terminator::JsonString),
        "json_number" => Ok(Terminator::JsonNumber {
            integer_only: false,
        }),
        "json_integer" => Ok(Terminator::JsonNumber { integer_only: true }),
        "balanced" => {
            let one = |s: &Option<String>, which: &str| -> Result<u8, BuildError> {
                s.as_ref()
                    .and_then(|x| x.as_bytes().first().copied())
                    .ok_or_else(|| {
                        BuildError::Yaml(format!("balanced terminator needs a 1-byte `{which}`"))
                    })
            };
            Ok(Terminator::Balanced {
                open: one(&ft.open, "open")?,
                close: one(&ft.close, "close")?,
            })
        }
        other => Err(BuildError::Yaml(format!("unknown terminator {other:?}"))),
    }
}

impl From<YamlLimits> for FreeTextLimits {
    fn from(y: YamlLimits) -> Self {
        FreeTextLimits {
            ramp_start: y.ramp_start,
            ramp_len: y.ramp_len.unwrap_or(0),
            boost: y.boost.unwrap_or(0.0),
            forced_after: y.forced_after,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::tree::StencilNode;
    use crate::stencil::vocab::TestVocab;

    const YAML: &str = r#"
label: tool_call
root: open
nodes:
  - id: open
    static: "\""
    next: name
  - id: name
    branch:
      - match: "ab"
        next: close
      - match: "cd"
        next: close
  - id: close
    static: "\""
    next: val
  - id: val
    free_text:
      terminator: json_string
      eos_ends: false
      limits: { forced_after: 256 }
    next: done
  - id: done
    end: true
"#;

    #[test]
    fn parses_and_compiles() {
        let spec = TreeSpec::from_yaml(YAML).unwrap();
        let tree = compile(&spec, &TestVocab::new()).unwrap();
        match tree.node(tree.root()) {
            StencilNode::Static { tokens, .. } => assert_eq!(tokens, &[b'"' as u32]),
            _ => panic!(),
        }
    }

    #[test]
    fn balanced_terminator() {
        let yaml = r#"
label: t
root: v
nodes:
  - id: v
    free_text:
      terminator: balanced
      open: "{"
      close: "}"
      limits: { forced_after: 64 }
    next: done
  - id: done
    end: true
"#;
        let spec = TreeSpec::from_yaml(yaml).unwrap();
        let tree = compile(&spec, &TestVocab::new()).unwrap();
        match tree.node(tree.root()) {
            StencilNode::FreeText(span) => {
                assert_eq!(
                    span.term,
                    Terminator::Balanced {
                        open: b'{',
                        close: b'}'
                    }
                )
            }
            _ => panic!(),
        }
    }

    #[test]
    fn multi_kind_node_rejected() {
        let yaml = r#"
label: t
root: x
nodes:
  - id: x
    static: "a"
    end: true
    next: x
"#;
        assert!(matches!(
            TreeSpec::from_yaml(yaml),
            Err(BuildError::Yaml(_))
        ));
    }

    #[test]
    fn bad_yaml_rejected() {
        assert!(matches!(
            TreeSpec::from_yaml("not: [valid"),
            Err(BuildError::Yaml(_))
        ));
    }
}
