//! Tests that tool schemas are read into the parameter trees the grammar is built
//! from.
//!
//! Named types and nullable spellings are resolved to their definitions, so their
//! enums survive. A union of unlike arms, or a reference that cannot be resolved,
//! stays opaque rather than hanging the builder. The tests also pin the field
//! order (required first, then declared order), the minimum and maximum that bound
//! arrays, and the conditional rules that require fields depending on a mode
//! value, each applied to the field it tests.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use serde_json::json;

    use super::*;

    use crate::stencil::compile::compile;

    use crate::stencil::vocab::TestVocab;

    use super::super::tests_envelope::tests::{arms, statics};

    /// An array's element schema and an object's fields survive parsing,
    /// recursively and in the same field order a tool's parameters take — a
    /// nullable container included, which is guided with `null` beside it.
    #[test]
    fn items_and_properties_are_kept_recursively() {
        let schema: Value = serde_json::from_str(
            r#"{
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "end_line": {"type": ["number", "null"]},
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    },
                    "maybe": {"type": ["array", "null"], "items": {"type": "string"}},
                    "loose": {"type": "object"},
                    "any_items": {"type": "array", "items": true}
                },
                "required": ["files"]
            }"#,
        )
        .unwrap();
        let spec = ToolSpec::from_json_schema("read_files", &schema);
        let by_name = |n: &str| spec.params.iter().find(|p| p.name == n).unwrap();

        let item = by_name("files")
            .items
            .as_deref()
            .expect("files keeps its items");
        assert_eq!(item.ty, ParamType::Object);
        assert!(item.name.is_empty());
        let fields = item
            .properties
            .as_ref()
            .expect("the element keeps its fields");
        let names: Vec<(&str, bool)> = fields
            .iter()
            .map(|p| (p.name.as_str(), p.required))
            .collect();
        assert_eq!(names, [("path", true), ("end_line", false)]);
        assert_eq!(fields[1].ty, ParamType::Number);
        assert!(fields[1].nullable);
        assert!(!fields[0].nullable);

        let maybe = by_name("maybe");
        assert_eq!(maybe.ty, ParamType::Array);
        assert!(maybe.nullable);
        assert!(maybe.items.is_some(), "a nullable array keeps its items");
        assert!(by_name("loose").properties.is_none(), "no properties, free");
        assert!(
            by_name("any_items").items.is_none(),
            "`items: true` is free"
        );
    }

    /// Every way a schema can say "or null" is read as nullable, and the type
    /// it qualifies is kept.
    #[test]
    fn every_spelling_of_nullable_is_read() {
        let param = |schema: Value| param_of(&schema, &schema);
        for (schema, ty) in [
            (json!({"type": ["boolean", "null"]}), ParamType::Boolean),
            (json!({"type": ["null", "string"]}), ParamType::String),
            (
                json!({"type": "integer", "nullable": true}),
                ParamType::Integer,
            ),
            (
                json!({"anyOf": [{"type": "boolean"}, {"type": "null"}]}),
                ParamType::Boolean,
            ),
            (
                json!({"oneOf": [{"type": "null"}, {"type": "array", "items": {"type": "string"}}]}),
                ParamType::Array,
            ),
            (json!({"enum": ["a", "b", null]}), ParamType::Object),
        ] {
            let p = param(schema.clone());
            assert!(p.nullable, "{schema}");
            assert_eq!(p.ty, ty, "{schema}");
        }
        let e = param(json!({"enum": ["a", null]}));
        assert_eq!(e.enum_values.as_deref(), Some(&["a".to_string()][..]));
        for schema in [
            json!({"type": "boolean"}),
            json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
            json!({"enum": ["a"]}),
        ] {
            assert!(!param(schema.clone()).nullable, "{schema}");
        }
        // `anyOf` of one schema and null keeps that schema's structure.
        let object = param(json!({"anyOf": [
            {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
            {"type": "null"}
        ]}));
        assert!(object.nullable);
        assert_eq!(object.properties.map(|f| f.len()), Some(1));
    }

    /// **A named type is followed to its definition.** `schemars` writes one as
    /// a bare `$ref`, or as an `allOf` of a single `$ref` when it also carries
    /// a description. Both must reach the definition, or the `enum` the
    /// grammar exists to enforce is lost and the value becomes any JSON at all.
    #[test]
    fn a_named_type_is_resolved_to_its_definition() {
        let schema = json!({
            "type": "object",
            "definitions": {
                "CredType": {"type": "string", "enum": ["ssh_key", "http_bearer"]}
            },
            "properties": {
                "bare": {"$ref": "#/definitions/CredType"},
                "described": {
                    "allOf": [{"$ref": "#/definitions/CredType"}],
                    "description": "Credential type."
                },
                "optional": {
                    "anyOf": [{"$ref": "#/definitions/CredType"}, {"type": "null"}]
                }
            },
            "required": ["bare", "described"]
        });
        let spec = ToolSpec::from_json_schema("credential_save", &schema);
        let by_name = |n: &str| spec.params.iter().find(|p| p.name == n).unwrap();
        for name in ["bare", "described", "optional"] {
            let p = by_name(name);
            assert_eq!(p.ty, ParamType::String, "{name}");
            assert_eq!(
                p.enum_values.as_deref(),
                Some(&["ssh_key".to_string(), "http_bearer".to_string()][..]),
                "{name} lost its enum",
            );
        }
        assert!(by_name("optional").nullable);
        assert!(!by_name("bare").nullable);
    }

    /// A `$ref` that leads nowhere, or in a circle, leaves the value opaque
    /// rather than hanging the builder or panicking.
    #[test]
    fn an_unresolvable_or_circular_ref_is_left_opaque() {
        let dangling = json!({
            "type": "object",
            "properties": {"x": {"$ref": "#/definitions/Missing"}}
        });
        assert_eq!(
            ToolSpec::from_json_schema("t", &dangling).params[0].ty,
            ParamType::Object
        );

        let circular = json!({
            "type": "object",
            "definitions": {"Loop": {"$ref": "#/definitions/Loop"}},
            "properties": {"x": {"$ref": "#/definitions/Loop"}}
        });
        assert_eq!(
            ToolSpec::from_json_schema("t", &circular).params[0].ty,
            ParamType::Object
        );

        // A remote pointer is not fetched.
        let remote = json!({
            "type": "object",
            "properties": {"x": {"$ref": "https://example.com/s.json#/A"}}
        });
        assert_eq!(
            ToolSpec::from_json_schema("t", &remote).params[0].ty,
            ParamType::Object
        );
    }

    /// **A tagged union is merged into one object whose discriminator lists
    /// every tag.** Without this a revision argument reaches the grammar as
    /// "any JSON value" and the model may write any key at all; with it the
    /// tag is one of the five that exist and the keys are the keys that exist.
    ///
    /// The merge is deliberately permissive about *combinations* — `kind` and
    /// every arm's field are offered together — because `Param` carries one
    /// shape per value. A combination no arm allows is refused when the call
    /// deserializes, which names the offending field.
    #[test]
    fn a_tagged_union_merges_into_one_object_listing_every_tag() {
        let schema = json!({
            "type": "object",
            "definitions": {
                "RevArg": {
                    "oneOf": [
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["head"]}},
                         "required": ["kind"]},
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["commit"]},
                            "id": {"type": "string"}},
                         "required": ["kind", "id"]},
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["branch"]},
                            "name": {"type": "string"}},
                         "required": ["kind", "name"]}
                    ]
                }
            },
            "properties": {
                "rev": {"anyOf": [{"$ref": "#/definitions/RevArg"}, {"type": "null"}]}
            }
        });
        let spec = ToolSpec::from_json_schema("git_log", &schema);
        let rev = &spec.params[0];
        assert_eq!(rev.ty, ParamType::Object);
        assert!(rev.nullable, "the field itself is optional");
        let fields = rev.properties.as_ref().expect("the union keeps its fields");

        let kind = fields.iter().find(|f| f.name == "kind").unwrap();
        assert_eq!(
            kind.enum_values.as_deref(),
            Some(
                &[
                    "head".to_string(),
                    "commit".to_string(),
                    "branch".to_string()
                ][..]
            ),
            "the discriminator must list every arm's tag",
        );
        assert!(kind.required, "every arm requires the discriminator");

        // The arms' own fields are offered, and none is required, because no
        // single one is required by every arm.
        let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["kind", "id", "name"], "discriminator first");
        for f in fields.iter().filter(|f| f.name != "kind") {
            assert!(!f.required, "{} is required by only one arm", f.name);
            assert_eq!(f.ty, ParamType::String);
        }
    }

    /// A union whose arms are not all objects stays opaque — there is no
    /// single shape to merge them into.
    #[test]
    fn a_union_of_unlike_arms_stays_opaque() {
        let schema = json!({
            "type": "object",
            "properties": {
                "x": {"oneOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}},
                    {"type": "string"}
                ]}
            }
        });
        let x = &ToolSpec::from_json_schema("t", &schema).params[0];
        assert_eq!(x.ty, ParamType::Object);
        assert!(x.properties.is_none(), "no single shape, so no fields");
    }

    /// A union nested inside an array's elements resolves too — `git_push`
    /// takes an array of tagged push items.
    #[test]
    fn a_union_inside_an_array_resolves() {
        let schema = json!({
            "type": "object",
            "definitions": {
                "PushItem": {
                    "oneOf": [
                        {"type": "object", "properties": {
                            "action": {"type": "string", "enum": ["update_branch"]},
                            "branch": {"type": "string"}},
                         "required": ["action", "branch"]},
                        {"type": "object", "properties": {
                            "action": {"type": "string", "enum": ["delete_tag"]},
                            "tag": {"type": "string"}},
                         "required": ["action", "tag"]}
                    ]
                }
            },
            "properties": {
                "pushes": {"type": "array", "items": {"$ref": "#/definitions/PushItem"}}
            },
            "required": ["pushes"]
        });
        let spec = ToolSpec::from_json_schema("git_push", &schema);
        let element = spec.params[0]
            .items
            .as_ref()
            .expect("elements keep a schema");
        let fields = element
            .properties
            .as_ref()
            .expect("the element is an object");
        let action = fields.iter().find(|f| f.name == "action").unwrap();
        assert_eq!(
            action.enum_values.as_deref(),
            Some(&["update_branch".to_string(), "delete_tag".to_string()][..])
        );
        assert!(fields.iter().any(|f| f.name == "branch"));
        assert!(fields.iter().any(|f| f.name == "tag"));
    }

    /// **An enum field's arm spells the separator after it, so the model writes
    /// the quote and the separator as its own token.** `repo` is an optional
    /// enum ahead of the optional `path`: after the value, the arms go on to
    /// the next key or to the close, never stopping at a lone closing quote —
    /// the boundary that left a live model unable to write `",` and closing its
    /// arguments instead of naming the path it meant to.
    #[test]
    fn an_enum_field_arm_carries_the_separator_after_it() {
        let schema = json!({
            "type": "object",
            "properties": {
                "repo": {"type": "string", "enum": ["a", "b"]},
                "path": {"type": "string"}
            }
        });
        let spec = compile_tool_call_tree(
            &[ToolSpec::from_json_schema("t", &schema)],
            &ToolCallEnvelope::qwen3(),
        )
        .unwrap();
        let all = arms(&spec);
        for want in [
            " \"a\", \"path\":",
            " \"b\", \"path\":",
            " \"a\"}}\n</tool_call>",
            " \"b\"}}\n</tool_call>",
        ] {
            assert!(
                all.iter().any(|a| a == want),
                "{want:?} missing from {all:?}"
            );
        }
        assert!(
            !all.iter().any(|a| a == " \"a\"" || a == " \"b\""),
            "an enum arm must not end on its bare closing quote: {all:?}"
        );
    }

    /// A closed set compiles to a choice, with `null` among the arms exactly
    /// when the schema allows it. A string is not a closed set.
    #[test]
    fn a_closed_set_compiles_to_a_choice() {
        let arms_for = |value: Value| {
            let schema = json!({"type": "object", "properties": {"v": value}, "required": ["v"]});
            let spec = compile_tool_call_tree(
                &[ToolSpec::from_json_schema("t", &schema)],
                &ToolCallEnvelope::qwen3(),
            )
            .unwrap();
            // The one-tool name branch is not the value's.
            let mut a: Vec<String> = arms(&spec).into_iter().filter(|a| a != "t\"").collect();
            a.sort();
            a
        };
        assert_eq!(arms_for(json!({"type": "boolean"})), [" false", " true"]);
        assert_eq!(
            arms_for(json!({"type": ["boolean", "null"]})),
            [" false", " null", " true"]
        );
        // An enum arm carries what follows the value — here the call's close —
        // so the value's closing quote and the close are one arm's text.
        let close = "}}\n</tool_call>";
        assert_eq!(
            arms_for(json!({"enum": ["x", "y", null]})),
            [
                format!(" \"x\"{close}"),
                format!(" \"y\"{close}"),
                format!(" null{close}")
            ]
        );
        assert_eq!(
            arms_for(json!({"enum": ["x", "y"]})),
            [format!(" \"x\"{close}"), format!(" \"y\"{close}")]
        );
        // A nullable string is not a closed set: it is a free value, so the
        // model's own ` ""` and ` null` tokens are both reachable and no arm
        // prefills its opening quote.
        assert!(arms_for(json!({"anyOf": [{"type": "string"}, {"type": "null"}]})).is_empty());
        // An enum value is written as JSON, escapes and all.
        assert_eq!(
            arms_for(json!({"enum": ["say \"x\"", "b"]})),
            [
                format!(" \"b\"{close}"),
                format!(" \"say \\\"x\\\"\"{close}")
            ]
        );
        // An array of booleans chooses at every element.
        let a = arms_for(json!({"type": "array", "items": {"type": "boolean"}}));
        for arm in ["true", "false", "]", ", true", ", false"] {
            assert!(a.iter().any(|x| x == arm), "{arm:?} missing from {a:?}");
        }
    }

    /// **A guided array honours `minItems`**: the levels before the minimum
    /// offer no `]`, so `"pushes": []` cannot be decoded for a `git_push` that
    /// needs one push. Each unrolled level offers one `]` arm otherwise, so
    /// the count of them is the count of levels the array may close at.
    #[test]
    fn a_guided_array_offers_no_close_before_its_minimum() {
        let closes = |min: Option<u64>| {
            let mut array = json!({"type": "array", "items": {"type": "boolean"}});
            if let Some(min) = min {
                array["minItems"] = json!(min);
            }
            let schema = json!({"type": "object", "properties": {"v": array}, "required": ["v"]});
            let tool = ToolSpec::from_json_schema("t", &schema);
            assert_eq!(tool.params[0].min_items, min.unwrap_or(0) as usize);
            let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
            arms(&spec).into_iter().filter(|a| a == "]").count()
        };
        assert_eq!(closes(None), MAX_ARRAY_ELEMENTS);
        assert_eq!(closes(Some(1)), MAX_ARRAY_ELEMENTS - 1);
        assert_eq!(closes(Some(3)), MAX_ARRAY_ELEMENTS - 3);
        // A minimum past the unrolled bound could only be written as exactly
        // the bound, which the schema refuses: the array is left free.
        let schema = json!({"type": "object", "properties": {"v": {
            "type": "array", "items": {"type": "boolean"}, "minItems": 1000}}});
        let past = &ToolSpec::from_json_schema("t", &schema).params[0];
        assert!(past.items.is_none(), "{past:?}");
        assert_eq!(past.min_items, 0);
    }

    /// **A guided array honours `maxItems`**: it is unrolled to the maximum,
    /// so it offers one `]` per level it may close at and none past it — and a
    /// maximum past the unrolled bound is the bound.
    #[test]
    fn a_guided_array_offers_no_element_past_its_maximum() {
        let closes = |min: u64, max: Option<u64>| {
            let mut array = json!({"type": "array", "items": {"type": "boolean"}, "minItems": min});
            if let Some(max) = max {
                array["maxItems"] = json!(max);
            }
            let schema = json!({"type": "object", "properties": {"v": array}, "required": ["v"]});
            let tool = ToolSpec::from_json_schema("t", &schema);
            assert_eq!(tool.params[0].max_items, max.map(|m| m as usize));
            let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
            arms(&spec).into_iter().filter(|a| a == "]").count()
        };
        assert_eq!(closes(0, Some(8)), 8);
        assert_eq!(closes(2, Some(8)), 6);
        assert_eq!(closes(0, Some(1000)), MAX_ARRAY_ELEMENTS);
        assert_eq!(closes(0, None), MAX_ARRAY_ELEMENTS);
    }

    /// `git_push`'s real schema — `pushes` an array of `$ref`'d objects with
    /// `minItems: 1` — compiles with its first element required.
    #[test]
    fn a_push_needs_a_push() {
        let schema = json!({
            "type": "object",
            "definitions": {"PushItem": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["update_branch", "delete_branch"]},
                "name": {"type": "string"}},
                "required": ["action", "name"]}},
            "properties": {
                "remote": {"type": "string"},
                "pushes": {"type": "array", "minItems": 1, "maxItems": 50,
                           "items": {"$ref": "#/definitions/PushItem"}}},
            "required": ["remote", "pushes"]
        });
        let tool = ToolSpec::from_json_schema("git_push", &schema);
        let pushes = tool.params.iter().find(|p| p.name == "pushes").unwrap();
        assert_eq!((pushes.min_items, pushes.max_items), (1, Some(50)));
        let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
        // Unrolled to its fifty, every level but the first may close: the
        // first element is written.
        let closes = arms(&spec).into_iter().filter(|a| a == "]").count();
        assert_eq!(closes, 50 - 1, "an empty push list is offered");
    }

    /// A mode-switched request, `git_commit`'s shape: `from` decides which of
    /// the optional fields a call needs, through `if`/`then`.
    pub(in super::super) fn moded_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "from": {"type": "string", "enum": ["changes", "files", "patch", "revert", "noop"]},
                "message": {"type": "string"},
                "changes": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "patch": {"type": "string"},
                "commit": {"type": "string"}
            },
            "required": ["repo", "from"],
            "allOf": [
                {"if": {"properties": {"from": {"const": "files"}}},
                 "then": {"required": ["message", "changes"]}},
                {"if": {"properties": {"from": {"enum": ["changes", "patch"]}}},
                 "then": {"required": ["message"]}},
                {"if": {"properties": {"from": {"const": "patch"}}},
                 "then": {"required": ["patch"]}},
                {"if": {"properties": {"from": {"const": "revert"}}},
                 "then": {"required": ["commit"]}},
                {"if": {"properties": {"from": {"const": "files"}, "repo": {"const": "x"}}},
                 "then": {"required": ["patch"]}}
            ]
        })
    }

    /// **Each `if`/`then` lands on the field it tests**, value by value — an
    /// `enum` naming several — and a rule testing two fields is not one field's.
    #[test]
    fn value_requirements_are_read_onto_the_field_they_test() {
        let tool = ToolSpec::from_json_schema("t", &moded_schema());
        let from = tool.params.iter().find(|p| p.name == "from").unwrap();
        let s = |v: &[&str]| v.iter().map(|f| f.to_string()).collect::<Vec<_>>();
        assert_eq!(
            from.requires,
            vec![
                ("files".to_string(), s(&["message", "changes"])),
                ("changes".to_string(), s(&["message"])),
                ("patch".to_string(), s(&["message"])),
                ("patch".to_string(), s(&["patch"])),
                ("revert".to_string(), s(&["commit"])),
            ]
        );
        assert!(tool
            .params
            .iter()
            .all(|p| p.name == "from" || p.requires.is_empty()));
    }

    /// **A value can hold a sibling to its own rule**: `then.properties` keeps
    /// the sibling's type and takes the rule's pattern, for that value alone.
    #[test]
    fn then_properties_bring_a_field_shaped_by_the_value() {
        let schema = json!({
            "type": "object",
            "properties": {
                "from": {"type": "string", "enum": ["hex", "utf8"]},
                "data": {"type": "string"}
            },
            "required": ["from", "data"],
            "allOf": [{
                "if": {"properties": {"from": {"const": "hex"}}},
                "then": {"properties": {"data": {"pattern": "[0-9a-f]+"}}}
            }]
        });
        let tool = ToolSpec::from_json_schema("t", &schema);
        let from = tool.params.iter().find(|p| p.name == "from").unwrap();
        assert_eq!(from.shapes.len(), 1);
        let (value, fields) = &from.shapes[0];
        assert_eq!(value, "hex");
        assert_eq!(fields[0].name, "data");
        assert_eq!(fields[0].ty, ParamType::String);
        assert_eq!(fields[0].pattern.as_deref(), Some("[0-9a-f]+"));
        let plain = tool.params.iter().find(|p| p.name == "data").unwrap();
        assert_eq!(plain.pattern, None, "the other values keep it free");
        assert!(compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).is_ok());
    }

    pub(in super::super) fn picked(rule: &str) -> ToolSpec {
        let mut schema = json!({
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "a": {"type": ["string", "null"]},
                "b": {"type": ["string", "null"]},
                "c": {"type": "boolean"}
            },
            "required": ["id"]
        });
        schema[rule] = json!([{"required": ["a"]}, {"required": ["b"]}]);
        ToolSpec::from_json_schema("t", &schema)
    }

    /// **`anyOf` of `required` is at least one, `oneOf` exactly one**, and each
    /// member stops being nullable: `null` would satisfy it in name only.
    #[test]
    fn a_required_choice_is_read_onto_its_members() {
        for (rule, pick) in [("anyOf", Pick::AtLeastOne), ("oneOf", Pick::ExactlyOne)] {
            let tool = picked(rule);
            let of = |n: &str| tool.params.iter().find(|p| p.name == n).unwrap();
            assert_eq!(of("a").pick, pick);
            assert_eq!(of("b").pick, pick);
            assert!(!of("a").nullable);
            assert_eq!(of("c").pick, Pick::None);
            assert_eq!(of("id").pick, Pick::None);
        }
    }

    /// **The close is not offered until a member is written.** After `id` the
    /// only ways on are the members; after a member the call may close, and
    /// under `oneOf` the other member is gone.
    #[test]
    fn a_required_choice_withholds_the_close() {
        let closes = "}}\n</tool_call>";
        for (rule, other_after_a) in [("anyOf", true), ("oneOf", false)] {
            let spec = compile_tool_call_tree(&[picked(rule)], &ToolCallEnvelope::qwen3()).unwrap();
            let gate_with = |wanted: &str| -> Vec<Vec<String>> {
                spec.nodes
                    .iter()
                    .filter_map(|n| match n {
                        NodeSpec::Branch { arms } => {
                            let texts: Vec<String> = arms.iter().map(|(t, _)| t.clone()).collect();
                            texts.iter().any(|t| t.contains(wanted)).then_some(texts)
                        }
                        _ => None,
                    })
                    .collect()
            };
            let first: Vec<Vec<String>> = gate_with("\"a\"");
            assert!(
                first
                    .iter()
                    .any(|g| g.iter().all(|t| !t.starts_with(closes))),
                "{rule}: a gate offers the member and no close: {first:?}"
            );
            let after_a = gate_with("\"c\"")
                .into_iter()
                .filter(|g| g.iter().any(|t| t.starts_with(closes)))
                .collect::<Vec<_>>();
            assert!(
                !after_a.is_empty(),
                "{rule}: once satisfied the call may close"
            );
            assert_eq!(
                after_a
                    .iter()
                    .any(|g| g.iter().any(|t| t.contains("\"b\""))),
                other_after_a,
                "{rule}: {after_a:?}"
            );
        }
    }

    /// **A value cannot close the call before the fields it requires**: each
    /// arm of `from` goes on into its own first required key, and only a value
    /// that requires nothing may close.
    #[test]
    fn a_mode_value_goes_on_into_the_fields_it_requires() {
        let spec = compile_tool_call_tree(
            &[ToolSpec::from_json_schema("t", &moded_schema())],
            &ToolCallEnvelope::qwen3(),
        )
        .unwrap();
        let all = arms(&spec);
        let has = |want: &str| all.iter().any(|a| a == want);
        for want in [
            " \"files\", \"message\":",
            " \"changes\", \"message\":",
            " \"patch\", \"message\":",
            " \"revert\", \"commit\":",
            " \"noop\"}}\n</tool_call>",
            " \"noop\", \"message\":",
        ] {
            assert!(has(want), "{want:?} missing from {all:?}");
        }
        for closed in ["files", "changes", "patch", "revert"] {
            let early = format!(" \"{closed}\"}}}}\n</tool_call>");
            assert!(!has(&early), "{closed} may close the call: {all:?}");
        }
        // Required after their value, the keys are the grammar's to write.
        let fixed = statics(&spec);
        for key in [", \"changes\": [", ", \"patch\":"] {
            assert!(
                fixed.iter().any(|s| s == key),
                "{key:?} missing from {fixed:?}"
            );
        }
    }

    /// The function-block shape branches the same way, each value ending on
    /// the element's own marker.
    #[test]
    fn a_mode_value_branches_in_the_function_block_shape_too() {
        let env = ToolCallEnvelope::qwen35();
        let spec =
            compile_tool_call_tree(&[ToolSpec::from_json_schema("t", &moded_schema())], &env)
                .unwrap();
        let all = arms(&spec);
        for value in ["changes", "files", "patch", "revert", "noop"] {
            let want = format!("{value}{}", env.param_close);
            assert!(all.contains(&want), "{want:?} missing from {all:?}");
        }
    }

    /// Which arrays and objects the grammar writes, read off the spec: a free
    /// `JsonValue` span per value the model writes whole.
    #[test]
    fn only_elements_the_grammar_can_begin_are_guided() {
        let free_spans = |schema: &str| {
            let schema: Value = serde_json::from_str(schema).unwrap();
            let tool = ToolSpec::from_json_schema("t", &schema);
            let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
            spec.nodes
                .iter()
                .filter(|n| {
                    matches!(
                        n,
                        NodeSpec::FreeText {
                            term: Terminator::JsonValue,
                            ..
                        }
                    )
                })
                .count()
        };
        let array_of = |items: &str| {
            format!(
                r#"{{"type":"object","properties":{{"a":{{"type":"array","items":{items}}}}},
                    "required":["a"]}}"#
            )
        };
        // Strings, closed sets and objects with fields are written by the
        // grammar.
        assert_eq!(free_spans(&array_of(r#"{"type":"string"}"#)), 0);
        assert_eq!(free_spans(&array_of(r#"{"type":"boolean"}"#)), 0);
        assert_eq!(free_spans(&array_of(r#"{"type":["boolean","null"]}"#)), 0);
        assert_eq!(free_spans(&array_of(r#"{"enum":["a","b"]}"#)), 0);
        assert_eq!(
            free_spans(&array_of(
                r#"{"type":"object","properties":{"p":{"type":"string"}},"required":["p"]}"#
            )),
            0
        );
        // Numbers open on no delimiter: the whole array is one free value.
        assert_eq!(free_spans(&array_of(r#"{"type":"integer"}"#)), 1);
        // An array inside a guided array is free — once per unrolled element.
        assert_eq!(
            free_spans(&array_of(
                r#"{"type":"object","properties":{
                     "tags":{"type":"array","items":{"type":"string"}}},"required":["tags"]}"#
            )),
            MAX_ARRAY_ELEMENTS
        );
    }

    /// The guided shapes compile against a vocabulary that merges across their
    /// boundaries the way a BPE tokenizer does — `[{`, `[]`, `}]`, `}, {`, `["`.
    #[test]
    fn a_guided_array_of_objects_compiles_with_bracket_merges() {
        let v = TestVocab::new()
            .with_special("[{", 300)
            .with_special("[]", 301)
            .with_special("}]", 302)
            .with_special("}, {", 303)
            .with_special("{\"", 304)
            .with_special("\"]", 305)
            .with_special("[\"", 306)
            .with_special("\": [", 307)
            .with_special(" [", 308);
        let schema: Value = serde_json::from_str(
            r#"{"type":"object","properties":{
                 "files":{"type":"array","items":{"type":"object","properties":{
                   "path":{"type":"string"},"start_line":{"type":["number","null"]}},
                   "required":["path"]}},
                 "commands":{"type":"array","items":{"type":"string"}}},
               "required":["files"]}"#,
        )
        .unwrap();
        let tool = ToolSpec::from_json_schema("t", &schema);
        let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
        compile(&spec, &v).expect("guided arrays compile across bracket merges");
    }

    /// Required parameters take the `required` list's order; optionals follow
    /// in the order the schema text declares them, unsorted. Parsed from text
    /// rather than built with `json!`, because a client's schema arrives as
    /// text.
    #[test]
    fn params_follow_the_required_list_then_declared_order() {
        let schema: Value = serde_json::from_str(
            r#"{
                "type": "object",
                "properties": {
                    "width": {"type": "integer"},
                    "path": {"type": "string"},
                    "height": {"type": "integer"},
                    "note": {"type": "string"},
                    "after": {"type": "string"}
                },
                "required": ["path", "note"]
            }"#,
        )
        .unwrap();
        let spec = ToolSpec::from_json_schema("probe", &schema);
        let names: Vec<&str> = spec.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["path", "note", "width", "height", "after"]);
        assert!(spec.params[..2].iter().all(|p| p.required));
        assert!(spec.params[2..].iter().all(|p| !p.required));
    }
}
