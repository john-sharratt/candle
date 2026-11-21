//! Tests that each tool mode offers exactly the tools its grants allow, and that the
//! file tools a coding turn needs are always among them.
//!
//! A mode never offers a tool its grants would refuse, since that would spend a
//! projection slot on a call that fails. The file tools, the two web tools, the journal
//! readers and the note an acting turn opens with are offered on every turn. That set
//! is pinned, because each tool added to it widens every prompt. Restricted mode has the
//! file tools and the git readers but no command line, and comprehensive mode offers
//! every tool.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// **`guide_read` offers exactly the guides the library holds**, so a guide
    /// added to `mcp/src/guides/` cannot be left out of Zen's grammar, nor a
    /// removed one left in it.
    #[test]
    fn guide_read_offers_every_guide() {
        let def = find("guide_read").expect("guide_read is defined");
        let offered: Vec<String> = def.parameters["properties"]["name"]["enum"]
            .as_array()
            .expect("an enum of the guides")
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        assert_eq!(offered, zend_tools::tools::guide::names());
    }

    /// A coding-agent workspace (no `projection.yaml`) uses the bundled catalog
    /// even when it contains an unrelated `tools/` directory (the regression that
    /// shadowed the built-ins with an empty catalog); a mind workspace (with a
    /// `projection.yaml`) draws from its own `tools/` — empty means tool-free.
    #[test]
    fn tools_override_only_applies_to_a_mind_workspace() {
        let base = std::env::temp_dir().join(format!("zend_toolws_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);

        let coding = base.join("coding");
        std::fs::create_dir_all(coding.join("tools")).unwrap();
        std::fs::write(coding.join("tools").join("Cargo.toml"), "x").unwrap();
        assert_eq!(
            load_effective(&coding).len(),
            load_bundled().len(),
            "coding-agent workspace must use the bundled catalog, not an unrelated tools/ dir"
        );

        let mind = base.join("mind");
        std::fs::create_dir_all(mind.join("tools")).unwrap();
        std::fs::write(mind.join("projection.yaml"), "layers: []").unwrap();
        assert!(
            load_effective(&mind).is_empty(),
            "a mind's empty tools/ folder is deliberately tool-free"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// The file tools a coding turn needs whatever the question — reading,
    /// searching, editing and writing — the two web tools, the two journal
    /// readers and the note every acting turn opens with project on every
    /// turn; nothing else does. A mandatory tool rides on top of the belief top-k, so adding one
    /// here widens every prompt — the set is pinned so that is a deliberate
    /// change.
    #[test]
    fn the_file_tools_a_coding_turn_needs_are_mandatory() {
        let mut mandatory: Vec<&str> = all()
            .iter()
            .filter(|d| d.mandatory)
            .map(|d| d.name.as_str())
            .collect();
        mandatory.sort();
        assert_eq!(
            mandatory,
            [
                "file_edit",
                "file_grep",
                "file_list",
                "file_read",
                "file_search",
                "journal_grep",
                "journal_note",
                "journal_read",
                "web_fetch",
                "web_search",
                "write"
            ]
        );
    }

    /// **Restricted is the file tools and the git readers, and no command
    /// line.** Its file changes stay in the overlay, so writing is safe; the
    /// git writers, code, the network and programs on this host are not.
    #[test]
    fn restricted_writes_files_and_reads_git_but_runs_nothing() {
        let safe = names_for(ToolMode::Restricted);
        for name in [
            "datetime",
            "write",
            "file_edit",
            "file_delete",
            "file_read",
            "git_status",
            "git_log",
            "git_show",
            "git_grep",
            "git_refs",
        ] {
            assert!(safe.contains(name), "{name} missing from restricted");
        }
        for name in [
            "git_commit",
            "git_merge",
            "git_ref",
            "git_switch",
            "git_reset",
            "git_fetch",
            "git_push",
            "code_run",
            "code_session_exec",
            "ssh_session_exec",
            "telnet_send",
            "ping_icmp",
            "trace_route",
            "sql_session_query",
            "run_command",
            "run_output",
        ] {
            assert!(!safe.contains(name), "{name} offered in restricted");
        }
    }

    /// **No mode offers what it would refuse.** Every tool a mode projects
    /// runs under that mode's grants, and the file tools a coding turn needs
    /// are in every mode that has tools.
    #[test]
    fn every_mode_offers_only_what_its_grants_cover() {
        for mode in ToolMode::ALL {
            let names = names_for(mode);
            for name in &names {
                let tool = registry::find(name).expect("defined tools execute");
                assert!(
                    access::grants(mode).require_all(tool.requires).is_ok(),
                    "{name} is offered in {} but needs {:?}",
                    mode.id(),
                    tool.requires
                );
            }
            if mode != ToolMode::None {
                for name in ["file_read", "file_list", "file_grep", "calculator"] {
                    assert!(names.contains(name), "{name} missing from {}", mode.id());
                }
            }
        }
        assert!(names_for(ToolMode::None).is_empty());
        let restricted = names_for(ToolMode::Restricted);
        for name in ["web_search", "web_fetch", "dns_lookup", "sql_session_open"] {
            assert!(!restricted.contains(name), "{name} needs a capability");
        }
    }

    /// **Comprehensive offers every tool** — programs on the host, the JS
    /// sandbox, the network, remote shells and the git writers included.
    #[test]
    fn comprehensive_offers_every_tool() {
        let comprehensive = names_for(ToolMode::Comprehensive);
        assert_eq!(
            comprehensive.len(),
            all().len(),
            "comprehensive offers every tool"
        );
    }
}
