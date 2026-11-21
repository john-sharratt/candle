//! Places the tool registry in the system prompt, one section per tool, and
//! resolves the specs the tool grammar is compiled from.
//!
//! Each tool is one Hermes-format JSON line in a collection that the prompt's
//! top-k selection filters, and a mandatory tool is projected on every turn
//! beside that selection. The grammar admits only the names it was compiled
//! from, so every alias a tool answers to is compiled as well, with an argument
//! the alias fixes narrowed to that one value. A calibration frame shows one
//! pinned tool inside the same envelope the live prompt uses.

use super::*;

/// The names of the tools projected in `mode` — see
/// [`crate::tool_def::names_for`].
pub fn offered_tool_names(mode: ToolMode) -> HashSet<String> {
    crate::tool_def::names_for(mode)
}

// ── Tool catalog injection ───────────────────────────────────────────────────

/// Install the registry's tool catalog as a `SectionCollection` inside
/// `dialogue`'s system prompt.
///
/// On entry the schema is expected to declare a YAML-defined collection
/// named `"tools"` (typically wedged between `tools_intro` and
/// `tools_outro` static sections).  This function:
///
/// 1. Looks up the existing `tools` collection's id.
/// 2. Calls [`ProjectionBuilder::add_section_to_collection`] for each
///    tool in the registry, registering one section per tool with the
///    Hermes-format JSON line as its content.
/// 3. Returns `(tool_name, section_id, json_line)` triples in registry
///    order.  Each section's BDP sigs come from the single prefill of
///    its `json_line` content during `insert_section_collection`.
///
/// Static framing sections (mode/frame/grounding/tools_intro/
/// tools_outro) are *outside* the collection — they always emit.  The
/// collection's own selection rule (declared in YAML, typically
/// `TopK { k: 5 }`) is what filters the tool list.
pub fn install_tool_catalog(
    builder: &mut ProjectionBuilder,
) -> anyhow::Result<Vec<(String, SectionId, String)>> {
    // A system prompt without a `tools` collection is a deliberately tool-free
    // projection (e.g. a conversational mind): install nothing rather than error.
    let Some(collection_id) = builder.id_for_system_collection("tools") else {
        tracing::info!("system prompt declares no 'tools' collection — tools disabled");
        return Ok(Vec::new());
    };
    let mut out: Vec<(String, SectionId, String)> = Vec::new();
    for def in crate::tool_def::all() {
        let json_line = def.json_line();
        let id = builder
            .add_section_to_collection(collection_id, def.name.clone(), &json_line, 100.0)
            .map_err(|e| anyhow::anyhow!("add_section_to_collection({}): {}", def.name, e))?;
        // A mandatory tool projects on every turn on top of the belief top-k,
        // never taking one of its slots. The per-mode builders cloned from this
        // one drop it with the rest when their mode excludes it.
        if def.mandatory {
            builder
                .set_collection_member_mandatory(collection_id, id)
                .map_err(|e| anyhow::anyhow!("mandatory tool {}: {e}", def.name))?;
        }
        out.push((def.name.clone(), id, json_line));
    }
    // This function only lays down the per-tool sections. The tool-catalog
    // *overview* is sealed separately into the `ToolSummaryRestricted` /
    // `ToolSummary` reserved sections at session startup and associated
    // with this collection per mode in `build_mode_builder` (via
    // `set_collection_summary_section`), so projection emits the full name listing
    // ahead of the provenance-selected subset.
    Ok(out)
}

/// The system prompt a calibration case's one tool is shown inside
/// ([`host::calibration`]): the coding assistant's instruction to call, and the
/// Hermes `<tools>…</tools>` envelope the live prompt wraps its tools in.
///
/// `open` is the system-open marker, the framing prose and the opening
/// `<tools>` marker; the pinned tool's line emits next; `close` closes the
/// block, gives the call instruction, and emits the system-close marker so the
/// whole system message is well-formed.
pub fn calibration_frame(dialect: &Dialect) -> Frame {
    Frame {
        open: format!(
            "{sys_start}You are a coding assistant. Use the available tool to satisfy the \
             user's request.\n\n# Tools\n\nYou may call the tool below. Return the call as a \
             JSON object with its name and arguments inside <tool_call></tool_call> XML tags.\n\n\
             <tools>\n",
            sys_start = dialect.system_start,
        ),
        close: format!(
            "</tools>\n\nFor a tool call, return:\n<tool_call>\n\
             {{\"name\": <tool-name>, \"arguments\": <args-json-object>}}\n</tool_call>{sys_end}",
            sys_end = dialect.system_end,
        ),
    }
}

/// The tool catalog as stencil specs, resolved once.
///
/// **The same list the tool stencil is compiled from** (see `session.rs`), which
/// is the property that makes the translation the grammar's exact inverse: a
/// parameter the grammar decoded as an integer parses back as one, and a
/// string-typed parameter whose text happens to be all digits stays a string. A
/// looks-like-a-number guess gets that second case wrong, and gets it wrong on
/// exactly the arguments a coding assistant passes around.
///
/// [`crate::tool_def::all`] is itself `OnceLock`-resolved and fixed for the
/// process once `init` has run, so caching the derived specs here costs one
/// schema walk per daemon rather than one per decoded turn.
///
/// **Every alias a tool answers to is a spec too**, with the tool's own
/// parameters. The grammar admits only the names it was compiled from, so a
/// model that writes a natural alias — `file_write` for `write` — had its name
/// healed into the nearest compiled one instead: `file_` then `list`, not
/// `write`. A turn that meant to create a file listed the directory forty
/// times, its reasoning saying "write the file now" before every call.
/// Dispatch already resolves aliases ([`registry::find`]); compiling them
/// lets the call reach it as the model wrote it.
///
/// **An alias whose name fixes an argument is compiled with it fixed**
/// ([`alias_pins`]): the field is narrowed to that one value, which the grammar
/// then writes itself. `git_cherry_pick` cannot decode `from: revert`.
pub fn tool_catalog() -> &'static [ToolSpec] {
    static SPECS: OnceLock<Vec<ToolSpec>> = OnceLock::new();
    SPECS.get_or_init(|| {
        crate::tool_def::all()
            .iter()
            .flat_map(|d| {
                let schema = grammar_schema(&d.name, &d.parameters);
                iter::once(d.name.as_str())
                    .chain(registry::aliases(&d.name).iter().copied())
                    .map(move |name| ToolSpec::from_json_schema(name, &pinned(name, &schema)))
            })
            .collect()
    })
}

/// Every name — a tool's own and each alias it answers to — of the tools a turn
/// calls at most once ([`crate::tool_def::ToolDef::never_twice`]). The grammar
/// names an alias's arm by the alias, so a tool held out under its own name
/// would still be writable under another.
pub fn never_twice_names() -> Vec<String> {
    crate::tool_def::all()
        .iter()
        .filter(|d| d.never_twice)
        .flat_map(|d| {
            iter::once(d.name.as_str())
                .chain(registry::aliases(&d.name).iter().copied())
                .map(str::to_string)
        })
        .collect()
}

/// The specs of [`tool_catalog`] that `mode` offers, aliases included: what a
/// turn in `mode` compiles its tool stencil from. The grammar admits only the
/// names it was compiled from, so a turn cannot decode a call to a tool its
/// mode does not offer. Compiled from the whole catalog, a Restricted turn
/// spent a round on six `web_search` calls, every one refused.
pub fn catalog_for(mode: ToolMode) -> Vec<ToolSpec> {
    let offered = offered_tool_names(mode);
    tool_catalog()
        .iter()
        .filter(|spec| registry::find(&spec.name).is_some_and(|tool| offered.contains(tool.name)))
        .cloned()
        .collect()
}

/// `parameters` with each field `name` fixes narrowed to its one value.
pub(super) fn pinned(name: &str, parameters: &Value) -> Value {
    let mut narrowed = parameters.clone();
    for (field, value) in alias_pins::pins(name) {
        let property = &mut narrowed["properties"][*field];
        let description = property.get("description").cloned();
        *property = serde_json::json!({"type": "string", "enum": [value]});
        if let Some(description) = description {
            property["description"] = description;
        }
    }
    narrowed
}
