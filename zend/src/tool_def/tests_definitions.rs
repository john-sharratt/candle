//! Tests that every bundled tool definition parses, names the fields its executor
//! reads, and offers the model only the values its schema allows.
//!
//! The YAML and the Rust request type are kept apart on purpose, so these tests hold
//! them to the same shape. A field declared but not read, or required but not declared,
//! fails at run time. No field of the git family may admit null, since null was the
//! cheapest wrong value the grammar offered. A `$ref` must resolve, since an unresolved
//! one is any JSON value to the decoder. Each `repo` parameter is constrained to the
//! workspace's repositories, and every value rule names fields and values that exist.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle_conversation::stencil::{Param, ToolSpec};

    use zend_vfs::ALL_REPOS;

    pub(in super::super) fn def(description: &str, params: Value) -> ToolDef {
        ToolDef(Definition {
            name: "file_list".to_string(),
            category: "Files".to_string(),
            description: description.to_string(),
            high_risk: false,
            mandatory: false,
            never_twice: false,
            parameters: params,
            examples: vec!["list the files<|im_end|>".to_string()],
            questions: vec!["what files are here".to_string()],
        })
    }

    /// **Every `repo` parameter becomes a choice between the workspace's
    /// repositories** — in the schema the prompt renders and in the spec the
    /// constrained decoder compiles, which are one and the same — and a tool
    /// without one is left exactly as it was.
    #[test]
    fn repo_parameters_are_constrained_to_the_workspace() {
        let repos = vec!["candle".to_string(), "mind".to_string()];
        let mut with_repo = def(
            "List the files.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "The repository."},
                    "path": {"type": "string"}
                }
            }),
        );
        with_repo.name = "file_list".into();
        let plain_params = serde_json::json!({
            "type": "object",
            "properties": {"expression": {"type": "string"}}
        });
        let mut plain = def("Calculate.", plain_params.clone());
        plain.name = "calculator".into();
        let mut defs = vec![with_repo, plain];

        constrain_repos(&mut defs, &repos);

        assert_eq!(
            defs[0].parameters["properties"]["repo"],
            serde_json::json!({
                "type": "string",
                "description": "The repository.",
                "enum": ["candle", "mind"]
            })
        );
        assert_eq!(
            defs[1].parameters, plain_params,
            "no repo parameter, no change"
        );

        let spec = ToolSpec::from_json_schema(&defs[0].name, &defs[0].parameters);
        let repo = spec.params.iter().find(|p| p.name == REPO_PARAM).unwrap();
        assert_eq!(repo.enum_values, Some(repos));
    }

    /// **A value rule names what exists.** Each `if`/`then` a definition
    /// carries — the fields one value of a field requires, which the grammar
    /// writes as required after that value — tests a field the definition
    /// has, for values that field allows, and requires fields it has. A
    /// misspelt name would leave the requirement silently off, and the call
    /// free to close without what its mode needs.
    #[test]
    fn every_value_rule_names_real_fields_and_values() {
        let mut ruled = Vec::new();
        for def in load_bundled().iter() {
            let params = &def.parameters;
            let Some(rules) = params.get("allOf").and_then(Value::as_array) else {
                continue;
            };
            ruled.push(def.name.clone());
            let props = params["properties"].as_object().unwrap();
            // A field's allowed values, through the `allOf`/`$ref` a named
            // enum is written with.
            let allowed = |field: &str| -> Vec<String> {
                let schema = &props[field];
                let named = schema["allOf"][0]["$ref"]
                    .as_str()
                    .or_else(|| schema["$ref"].as_str())
                    .and_then(|r| r.strip_prefix("#/definitions/"));
                let schema = match named {
                    Some(name) => &params["definitions"][name],
                    None => schema,
                };
                schema["enum"]
                    .as_array()
                    .map(|e| {
                        e.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default()
            };
            // Parsed once per definition, not once per rule: every rule below
            // reads the same schema, and re-parsing it per rule was pure
            // repeated work for a definition with more than one `allOf` rule.
            let spec = ToolSpec::from_json_schema(&def.name, params);
            for rule in rules {
                let tested = rule["if"]["properties"]
                    .as_object()
                    .unwrap_or_else(|| panic!("{}: a rule without `if.properties`", def.name));
                // A rule's `if` tests exactly one field, or the grammar drops
                // it — checked up front so the per-value work below can
                // assume a single `(field, values)` pair.
                assert_eq!(
                    tested.len(),
                    1,
                    "{}: a rule's `if` tests one field, or the grammar drops it",
                    def.name
                );
                let (field, test) = tested.iter().next().unwrap();
                assert!(
                    props.contains_key(field),
                    "{}: tests no field {field}",
                    def.name
                );
                let values: Vec<&str> = match (&test["const"], &test["enum"]) {
                    (Value::String(one), _) => vec![one.as_str()],
                    (_, Value::Array(many)) => many.iter().filter_map(Value::as_str).collect(),
                    _ => panic!("{}: {field} is tested by neither const nor enum", def.name),
                };
                let allows = allowed(field);
                for value in &values {
                    assert!(
                        allows.iter().any(|a| a == value),
                        "{}: {field} has no value {value} (it allows {allows:?})",
                        def.name
                    );
                }
                // The grammar acts on a rule only over a required, non-null
                // enum — anywhere else it would be dropped without a word.
                let param = spec.params.iter().find(|p| &p.name == field).unwrap();
                assert!(
                    param.required && !param.nullable && param.enum_values.is_some(),
                    "{}: the grammar does not act on the rule over {field}",
                    def.name
                );
                // And only when THIS rule's specific tested value carries its
                // own required fields — checking the field's aggregate
                // `requires` (every value any rule on it ever named) would
                // pass even if a bug dropped or mis-recorded the requirement
                // for exactly this value, as long as some other rule on the
                // same field left it non-empty.
                for value in &values {
                    assert!(
                        param
                            .requires
                            .iter()
                            .any(|(v, reqs)| v == value && !reqs.is_empty()),
                        "{}: {field}={value} carries no requirement the grammar acts on",
                        def.name
                    );
                }
                let required = rule["then"]["required"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{}: a rule without `then.required`", def.name));
                for field in required.iter().filter_map(Value::as_str) {
                    assert!(
                        props.contains_key(field),
                        "{}: requires no field {field}",
                        def.name
                    );
                }
            }
        }
        assert!(ruled.iter().any(|n| n == "git_commit"), "{ruled:?}");
    }

    /// **Every definition names exactly the fields its executor reads.**
    ///
    /// The YAML and the Rust request type are deliberately separate — the
    /// definition carries trigger-rich prose no derive could produce — but
    /// they have to agree on the *shape*, and nothing made them. A field the
    /// YAML declares and the request lacks is a parameter the model is told to
    /// send and the tool ignores; one the request requires and the YAML omits
    /// is a call the grammar can never complete. Both surface only as a failed
    /// call at run time, in a family where a failed call may be a refused
    /// commit.
    ///
    /// Every definition in the catalog, not one family: a new tool is held to
    /// it from the day it is written.
    #[test]
    fn every_definition_matches_its_executors_request_type() {
        let names = |schema: &Value| -> HashSet<String> {
            schema
                .get("properties")
                .and_then(Value::as_object)
                .map(|p| p.keys().cloned().collect())
                .unwrap_or_default()
        };
        let required = |schema: &Value| -> HashSet<String> {
            schema
                .get("required")
                .and_then(Value::as_array)
                .map(|r| {
                    r.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default()
        };

        // Every object the call is made of: the request itself, and each
        // object in its definitions block (a revision, a pushed ref, a file
        // change), since those are decoded by the same grammar.
        let objects = |schema: &Value| -> Vec<(String, Value)> {
            let mut out = vec![("the request".to_string(), schema.clone())];
            if let Some(defs) = schema.get("definitions").and_then(Value::as_object) {
                out.extend(
                    defs.iter()
                        .filter(|(_, d)| d.get("properties").is_some())
                        .map(|(n, d)| (n.clone(), d.clone())),
                );
            }
            out
        };

        for def in load_bundled().iter() {
            let tool = zend_tools::registry::find(&def.name)
                .unwrap_or_else(|| panic!("{} has no executor", def.name));
            let generated = (tool.schema)();

            for (object, want) in objects(&generated) {
                let have = if object == "the request" {
                    def.parameters.clone()
                } else {
                    def.parameters["definitions"][&object].clone()
                };
                assert_eq!(
                    names(&have),
                    names(&want),
                    "{}: {object}'s fields differ from the request type's",
                    def.name,
                );
                assert_eq!(
                    required(&have),
                    required(&want),
                    "{}: {object} and the request type disagree on what is required",
                    def.name,
                );
                // The type is what the stencil compiles, so it must be the
                // request type's — a field whose YAML type still admits `null`
                // hands the decoder a one-token way to satisfy the grammar
                // and say nothing.
                for field in names(&want) {
                    let ty = |schema: &Value| schema["properties"][&field].get("type").cloned();
                    if let Some(expected) = ty(&want) {
                        assert_eq!(
                            ty(&have),
                            Some(expected),
                            "{}: `{field}` of {object} has a different type in the \
                             definition than in its request type",
                            def.name,
                        );
                    }
                }
            }

            // **No field of the family admits `null`.** Optional fields are
            // left out, never nulled: measured live, `{"rev": null, "since":
            // null}` was the call the model made three times where it meant
            // `origin/main`, because `null` was the cheapest value the grammar
            // offered. Checked on the decoder's own reading of the definition,
            // so any spelling of nullability — a `null` type, an `anyOf` with
            // `{"type":"null"}` — is caught.
            // The git family's rule: its `null` was the cheapest wrong value.
            // Elsewhere an `Option` field may be nullable (`file_list`'s
            // `prefix`), and the stencil offers `null` beside the value.
            let spec = ToolSpec::from_json_schema(&def.name, &def.parameters);
            let mut stack: Vec<(String, &Param)> = if def.name.starts_with("git_") {
                spec.params.iter().map(|p| (p.name.clone(), p)).collect()
            } else {
                Vec::new()
            };
            while let Some((path, p)) = stack.pop() {
                assert!(
                    !p.nullable,
                    "{}: `{path}` admits null — leave an optional field out instead",
                    def.name,
                );
                for child in p.properties.iter().flatten() {
                    stack.push((format!("{path}.{}", child.name), child));
                }
                if let Some(items) = &p.items {
                    stack.push((format!("{path}[]"), items));
                }
            }

            // A `$ref` that resolves to nothing is silently "any JSON value"
            // in the decoder, which is exactly the constraint these unions
            // exist to provide.
            let text = def.parameters.to_string();
            let defined: HashSet<String> = def
                .parameters
                .get("definitions")
                .and_then(Value::as_object)
                .map(|d| d.keys().cloned().collect())
                .unwrap_or_default();
            for (_, rest) in text.split("#/definitions/").skip(1).map(|s| ("", s)) {
                let referenced: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                assert!(
                    defined.contains(&referenced),
                    "{}: $ref to {referenced:?}, which its definitions block does not define",
                    def.name,
                );
            }
        }
    }

    /// **The git tools' tagged unions reach the decoder as constrained
    /// choices, not as free JSON.**
    ///
    /// Every mode, kind and action in the family is a `$ref` to a named enum
    /// in the definitions block. Until the stencil learned to follow a `$ref`
    /// (`candle_conversation::stencil::ToolSpec::from_json_schema`), each one
    /// compiled to "any JSON value" — the model could write anything and the
    /// schema in the prompt was the only thing suggesting otherwise.
    ///
    /// This matters more now than it did: the family was collapsed from
    /// seventeen tools to nine precisely so that distinctions the model was
    /// getting wrong in the projection's top-k — where nothing enforces them —
    /// became enum fields inside a call, where the grammar does. That trade is
    /// only sound if the enums really do reach the decoder, which is what this
    /// asserts, from the bundled YAML through to the compiled spec.
    #[test]
    fn the_git_tools_modes_compile_to_constrained_choices() {
        let spec = |name: &str| {
            let def = find(name).unwrap_or_else(|| panic!("{name} is defined"));
            ToolSpec::from_json_schema(&def.name, &def.parameters)
        };
        let field = |params: &[Param], name: &str| {
            params
                .iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("no {name} parameter"))
                .clone()
        };

        let sorted = |p: &Param, what: &str| {
            let mut v = p
                .enum_values
                .clone()
                .unwrap_or_else(|| panic!("{what} is not constrained"));
            v.sort();
            v
        };

        // A revision: six forms, including the `parent` that exists so "the
        // previous commit" never needs an object id.
        let log = spec("git_log");
        let rev_fields = field(&log.params, "rev")
            .properties
            .expect("rev keeps its fields");
        let kind = field(&rev_fields, "kind");
        assert_eq!(
            sorted(&kind, "the revision kind"),
            [
                "branch",
                "commit",
                "head",
                "parent",
                "ref",
                "remote_branch",
                "tag",
                "upstream"
            ]
        );
        assert!(kind.required, "a revision always names its kind");
        for name in ["name", "back"] {
            assert!(
                rev_fields.iter().any(|f| f.name == name),
                "the revision's {name} field must be offered",
            );
        }

        // git_show's five modes — the distinction that used to be a tool
        // choice, and was measured going wrong there.
        let show = spec("git_show");
        assert_eq!(
            sorted(&field(&show.params, "what"), "git_show's what"),
            ["blame", "changes", "file", "patch", "tree"]
        );

        // git_refs' kinds — three former tools.
        let refs = spec("git_refs");
        assert_eq!(
            sorted(&field(&refs.params, "kind"), "git_refs' kind"),
            ["branches", "remote_branches", "remotes", "tags"]
        );

        // A commit's source, and the per-file action that carries `take`.
        let commit = spec("git_commit");
        assert_eq!(
            sorted(&field(&commit.params, "from"), "git_commit's from"),
            ["all_changes", "cherry_pick", "files", "patch", "revert"]
        );
        let change_fields = field(&commit.params, "changes")
            .items
            .expect("changes constrains its elements")
            .properties
            .expect("the element is an object");
        assert_eq!(
            sorted(&field(&change_fields, "action"), "the change action"),
            ["delete", "take", "write"]
        );
        assert!(
            change_fields.iter().any(|f| f.name == "content"),
            "a write's content must be offered",
        );

        // git_ref's two axes.
        let git_ref = spec("git_ref");
        assert_eq!(
            sorted(&field(&git_ref.params, "kind"), "git_ref's kind"),
            ["branch", "tag"]
        );
        assert_eq!(
            sorted(&field(&git_ref.params, "action"), "git_ref's action"),
            ["create", "delete", "move"]
        );

        // What a reset does to the work, and where a new branch starts.
        assert_eq!(
            sorted(
                &field(&spec("git_reset").params, "mode"),
                "git_reset's mode"
            ),
            ["hard", "soft"]
        );
        let switch = spec("git_switch");
        assert!(
            field(&switch.params, "from").properties.is_some(),
            "a new branch's start is a revision"
        );

        // A push item's action, inside the array.
        let item_fields = field(&spec("git_push").params, "pushes")
            .items
            .expect("pushes constrains its elements")
            .properties
            .expect("the item is an object");
        assert_eq!(
            sorted(&field(&item_fields, "action"), "the push action"),
            ["delete_branch", "delete_tag", "update_branch", "update_tag"]
        );
        for name in ["name", "source", "expected", "new"] {
            assert!(
                item_fields.iter().any(|f| f.name == name),
                "a push item's {name} field must be offered",
            );
        }
    }

    /// **A tool that can cover the whole workspace keeps its `*`** — the
    /// repositories go ahead of what the YAML declares, never in place of it.
    #[test]
    fn a_declared_all_repos_value_stays_after_the_repositories() {
        let mut grep = def(
            "Grep.",
            serde_json::json!({
                "type": "object",
                "properties": {"repo": {"type": "string", "enum": [ALL_REPOS]}},
                "required": ["repo"]
            }),
        );
        grep.name = "file_grep".into();
        let mut defs = vec![grep];
        constrain_repos(&mut defs, &["candle".to_string(), "mind".to_string()]);
        assert_eq!(
            defs[0].parameters["properties"]["repo"]["enum"],
            serde_json::json!(["candle", "mind", "*"])
        );
    }

    /// **Every bundled file tool requires a `repo`, and exactly the tools whose
    /// executor takes `*` declare it.** Nothing else ties a tool's YAML to its
    /// Rust request type, so a `repo` added to one and not the other would be a
    /// parameter the model is shown but the tool never reads, or one it needs
    /// but is never told about — and a `*` the YAML offered that the executor
    /// refused would be a call the grammar let through only to fail.
    #[test]
    fn every_repository_tool_declares_repo_as_its_executor_does() {
        let one_repo = [
            // Listing and reading are raw operations on one repository's
            // files; only a search spans the workspace.
            "file_list",
            "file_read",
            "write",
            "file_edit",
            "file_delete",
            "file_move",
            "file_image",
            // The transfers move one file between a URL and one repository.
            "file_download",
            "file_upload",
            "journal_move",
            "journal_check",
            "file_present",
            "code_run",
            "code_session_exec",
            "remote_fs_session_get",
            "remote_fs_session_put",
            // Every git tool works in exactly one repository. None takes `*`:
            // a revision, a branch and a remote only mean anything relative to
            // the repository they belong to, so a call spanning all of them
            // could not be executed even if the grammar let it through.
            "git_status",
            "git_log",
            "git_show",
            "git_grep",
            "git_refs",
            "git_commit",
            "git_merge",
            "git_ref",
            "git_switch",
            "git_reset",
            "git_fetch",
            "git_push",
            // A command runs in one repository's sandbox, and a build on an
            // agent against one repository's branch.
            "run_command",
            "run_output",
            "repo_job_start",
        ];
        let any_repo = ["file_search", "file_grep"];
        for def in load_bundled() {
            let name = def.name.as_str();
            let param = &def.parameters["properties"][REPO_PARAM];
            let is_required = def.parameters["required"]
                .as_array()
                .is_some_and(|r| r.iter().any(|v| v == REPO_PARAM));
            if one_repo.contains(&name) {
                assert!(is_required, "{name} must require `repo`");
                assert!(
                    param.get("enum").is_none(),
                    "{name} takes one repository, not `*`"
                );
            } else if any_repo.contains(&name) {
                assert!(is_required, "{name} must require `repo`");
                assert_eq!(
                    param["enum"],
                    serde_json::json!([ALL_REPOS]),
                    "{name} must offer `*`"
                );
            } else {
                assert!(
                    param.is_null(),
                    "{name} declares `repo` but its executor has none"
                );
            }
        }
    }

    /// **The grammar is built from these definitions, in their order**, not
    /// from the executor's request type: `text` declared before `old` left a
    /// call that wrote `old` no way to say what replaces it.
    #[test]
    fn file_edit_offers_old_before_text() {
        let defs = load_bundled();
        let edit = defs.iter().find(|d| d.name == "file_edit").unwrap();
        let names: Vec<&String> = edit.parameters["properties"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        let at = |n: &str| names.iter().position(|k| *k == n).unwrap();
        assert!(at("old") < at("text"), "{names:?}");
    }

    /// Execution (the registry) and definition (this folder) must agree on the
    /// tool set: every executable tool has a bundled definition, and every
    /// definition has an executor — so no model call binds to a missing runner
    /// and no surfaced tool lacks one. The registry is execution-only now, so this
    /// is the binding invariant (there are no definition fields left to compare).
    #[test]
    fn every_tool_has_definition_and_executor() {
        let defs = all();
        let reg = zend_tools::registry::all_tools();
        assert_eq!(
            defs.len(),
            reg.len(),
            "definition count {} != executor count {}",
            defs.len(),
            reg.len()
        );
        for tool in reg {
            assert!(
                find(tool.name).is_some(),
                "no definition for executable tool {}",
                tool.name
            );
        }
        for def in defs {
            assert!(
                zend_tools::registry::find(&def.name).is_some(),
                "no executor for defined tool {}",
                def.name
            );
        }
    }

    /// Every definition carries a real category (never the `"Other"` fallback the
    /// summary groups under) and a well-formed `"type": "object"` JSON Schema for
    /// `parameters` — the shape the tool-catalog summary and the constrained-decode
    /// stencil rely on. This is the bounded drift guard: it catches a `parameters`
    /// block that has degraded to a non-object or lost its schema shape. Exact
    /// conformance to the executor's `Tool::Request` is enforced at call time
    /// (args deserialize into `Request`, yielding `invalid_arguments` on mismatch).
    #[test]
    fn every_def_has_category_and_object_schema() {
        for def in all() {
            assert!(
                !def.category.trim().is_empty() && def.category != "Other",
                "tool {:?} has no real category",
                def.name
            );
            assert_eq!(
                def.parameters.get("type").and_then(|t| t.as_str()),
                Some("object"),
                "tool {:?} parameters is not an object JSON Schema",
                def.name
            );
        }
    }
}
