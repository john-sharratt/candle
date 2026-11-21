//! Tests for how dispatch is scoped to a conversation, a mode and a resumed
//! round.
//!
//! They cover the rule that a conversation's sessions close with it, that one
//! conversation's file writes are never read by another, and that a conversation
//! owned through `/mcp` cannot push to a default branch. An alias that
//! contradicts its fixed argument is refused before the tool runs. A resumed
//! round refuses only the calls that cannot be re-issued, while a live round
//! runs them all.

use super::*;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use zend_tools::state::sessions::{CodeEntry, SessionMeta};

    use zend_vfs::RepoSpec;

    use crate::journal::notes::JOURNAL_NOTE_TOOL;

    use crate::journal::shelf::{Lazy, Shelf};

    /// **The note a round opens with is called at most once a turn**, under its
    /// own name and every alias it answers to — and an ordinary tool is not.
    #[test]
    fn journal_note_is_called_at_most_once_a_turn() {
        let once = never_twice_names();
        assert!(once.contains(&JOURNAL_NOTE_TOOL.to_string()));
        for alias in registry::aliases(JOURNAL_NOTE_TOOL) {
            assert!(
                once.contains(&alias.to_string()),
                "alias {alias} is held out too"
            );
        }
        assert!(!once.contains(&"file_list".to_string()));
    }

    /// A code session opened by the round `ctx` runs.
    pub(in super::super) fn open_code_session(ctx: &ToolContext, id: &str) {
        ctx.sessions.insert_code(CodeEntry {
            meta: SessionMeta {
                session_id: id.to_string(),
                conversation: ctx.lineage().to_string(),
                opened_at: String::new(),
                last_activity: String::new(),
                alive: true,
            },
            language: "javascript".to_string(),
            history: Vec::new(),
        });
    }

    /// **A conversation's sessions close with it, whatever mode opened them**,
    /// and another conversation's stay open.
    #[test]
    fn closing_a_conversations_sessions_spares_the_others() {
        let dir = tempfile::tempdir().unwrap();
        let host = ToolHost::new(&workspace_in(&dir), no_secrets()).unwrap();
        let files = host.conversation_files();
        let a_full = host.context_for(ToolMode::Comprehensive, &files, "a");
        let a_restricted = host.context_for(ToolMode::Restricted, &files, "a");
        let b = host.context_for(ToolMode::Comprehensive, &files, "b");
        open_code_session(&a_full, "s1");
        open_code_session(&a_restricted, "s2");
        open_code_session(&b, "s3");

        assert_eq!(host.close_sessions("a"), 2);
        assert!(b.sessions.get_code("s3").is_some());
        assert!(a_full.sessions.get_code("s1").is_none());
        assert_eq!(host.close_sessions("a"), 0);
    }

    /// `dir` as a workspace holding one repository, `r`.
    pub(in super::super) fn workspace_in(dir: &tempfile::TempDir) -> Workspace {
        std::fs::create_dir_all(dir.path().join("r")).unwrap();
        Workspace::new(dir.path(), vec![RepoSpec::named("r")]).unwrap()
    }

    /// No secrets: what every test host that is not about secrets gets.
    pub(in super::super) fn no_secrets() -> Arc<Secrets> {
        Arc::new(Secrets::empty())
    }

    /// **A conversation owned through `/mcp` pushes to branches only**: its
    /// context closes each repository's default branch, and an ordinary
    /// dialogue's closes nothing.
    #[test]
    fn an_mcp_conversations_context_closes_the_default_branch() {
        let dir = tempfile::tempdir().unwrap();
        let host = ToolHost::new(&workspace_in(&dir), no_secrets())
            .unwrap()
            .guarding_defaults(&BTreeMap::from([("r".to_string(), "trunk".to_string())]));
        let files = host.conversation_files();
        let owned = host.context_for(ToolMode::Comprehensive, &files, "mcp:abc:review/verify");
        for closed in ["trunk", "main", "master"] {
            assert!(owned.check_push("r", closed).is_err(), "{closed}");
        }
        assert!(owned.check_push("r", "verify-expiry").is_ok());
        assert!(owned.check_push("unknown", "verify-expiry").is_err());
        let dialogue = host.context_for(ToolMode::Comprehensive, &files, "c1");
        assert!(dialogue.check_push("r", "main").is_ok());
    }

    /// **Dispatch holds an alias to its name** for a call that did not come
    /// through the grammar: a contradicting value is refused before the tool
    /// runs.
    #[test]
    fn dispatch_refuses_an_alias_its_arguments_contradict() {
        let dir = tempfile::tempdir().unwrap();
        let host = ToolHost::new(&workspace_in(&dir), no_secrets()).unwrap();
        let ctx = host.context_for(ToolMode::Comprehensive, &host.conversation_files(), "c1");
        let out = run_tool(
            &ctx,
            &ToolCall {
                name: "git_cherry_pick".to_string(),
                arguments: serde_json::json!({"repo": "r", "from": "revert", "commit": "c"}),
            },
        );
        assert_eq!(out["error"], "invalid_arguments", "{out}");
        assert!(
            out["detail"]
                .as_str()
                .unwrap()
                .starts_with("`git_cherry_pick` means `from: cherry_pick`"),
            "{out}"
        );
        // Shaped like every other `invalid_arguments` refusal — `with_guidance`
        // must enrich this one too, not just the ones `t.call` itself returns.
        assert_eq!(
            out["hint"],
            "call again with arguments that match `parameters`"
        );
        assert!(out["parameters"].is_object(), "{out}");
    }

    // ── The daemon's secrets ────────────────────────────────────────────────

    /// **`ToolHost::new` hands the daemon's secrets to every mode's context.**
    ///
    /// The document's parsing is covered in `zend-tools` and the choice of file
    /// in `crate::secrets`; what is asserted here is the wiring, which nothing
    /// else would catch. Dropping the `.with_secrets(..)` call leaves every
    /// crate compiling and every other test passing, and shows up only as
    /// `web_search` reporting itself unconfigured on a machine whose key is
    /// sitting right there in the file.
    #[test]
    fn the_tool_host_carries_the_daemons_secrets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secrets.yaml");
        std::fs::write(&path, "tavily_api_key: tvly-wired-through\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let secrets = Arc::new(Secrets::load(&path).unwrap());

        let host = ToolHost::new(&workspace_in(&dir), secrets).unwrap();
        let files = host.conversation_files();
        for mode in ToolMode::ALL {
            assert_eq!(
                host.context_for(mode, &files, "c1")
                    .secrets
                    .tavily_api_key(),
                Some("tvly-wired-through"),
                "{} lost the daemon's secrets",
                mode.id()
            );
        }
    }

    /// **Only an admin's dialogue reads the memory's repository file by
    /// file**: below Comprehensive a dialogue's stores have every other
    /// repository and not the memory, while the ingest's context reads it
    /// whole.
    #[test]
    fn only_a_comprehensive_dialogue_reads_the_memory_repository() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["r", memory_paths::REPO] {
            std::fs::create_dir_all(dir.path().join(name)).unwrap();
        }
        let workspace = Workspace::new(
            dir.path(),
            vec![RepoSpec::named("r"), RepoSpec::named(memory_paths::REPO)],
        )
        .unwrap();
        let host = ToolHost::new(&workspace, no_secrets()).unwrap();
        let files = host.conversation_files();
        let shelf = || -> Arc<dyn JournalShelf> {
            Arc::new(Lazy::new(|| -> Shelf {
                unreachable!("no journal tool runs")
            }))
        };
        for mode in ToolMode::ALL {
            let dialogue = host.dialogue_context(mode, &files, "c1", shelf());
            let want: Vec<&str> = match mode {
                ToolMode::Comprehensive => vec!["r", memory_paths::REPO],
                _ => vec!["r"],
            };
            assert_eq!(dialogue.files.names(), want, "{}", mode.id());
            let ingest = host.context_for(mode, &files, "");
            assert_eq!(
                ingest.files.names(),
                vec!["r", memory_paths::REPO],
                "{}",
                mode.id()
            );
        }
    }

    /// **Each mode's context carries that mode's grants, over the
    /// conversation's own files.** A Restricted round handed a gated call —
    /// the network, code, a database, a git writer, a program on this host —
    /// refuses it at dispatch, whatever the prompt offered.
    #[test]
    fn each_modes_context_carries_its_grants() {
        let dir = tempfile::tempdir().unwrap();
        let host = ToolHost::new(&workspace_in(&dir), no_secrets()).unwrap();
        let files = host.conversation_files();
        for mode in ToolMode::ALL {
            let ctx = host.context_for(mode, &files, "c1");
            assert_eq!(ctx.grants(), access::grants(mode), "{}", mode.id());
            assert!(Arc::ptr_eq(&ctx.files, &files), "{}", mode.id());
            assert_eq!(ctx.lineage(), "c1", "{}", mode.id());
        }
        let restricted = host.context_for(ToolMode::Restricted, &files, "c1");
        for (name, arguments) in [
            (
                "web_fetch",
                serde_json::json!({ "url": "https://example.com" }),
            ),
            (
                "code_run",
                serde_json::json!({ "repo": "r", "language": "js", "code": "1" }),
            ),
            ("sql_session_open", serde_json::json!({})),
            (
                "git_commit",
                serde_json::json!({ "repo": "r", "message": "m" }),
            ),
            ("ping_icmp", serde_json::json!({ "host": "127.0.0.1" })),
            (
                "run_command",
                serde_json::json!({ "repo": "r", "program": "npm", "args": ["test"] }),
            ),
        ] {
            let call = ToolCall {
                name: name.to_string(),
                arguments,
            };
            assert_eq!(
                run_tool(&restricted, &call)["error"],
                "not_permitted",
                "{name} ran in Restricted"
            );
        }
        // Comprehensive reaches the sandboxes — here to be told the folder is
        // not a git repository, so there is none for it.
        let comprehensive = host.context_for(ToolMode::Comprehensive, &files, "c1");
        let call = ToolCall {
            name: "run_command".to_string(),
            arguments: serde_json::json!({ "repo": "r", "program": "npm", "args": ["test"] }),
        };
        assert_eq!(run_tool(&comprehensive, &call)["error"], "no_sandbox");
    }

    /// A workspace with no document leaves every secret unset and the daemon
    /// still starts: not configuring web search is an ordinary way to run it.
    #[test]
    fn a_workspace_without_secrets_still_builds_a_host() {
        let dir = tempfile::tempdir().unwrap();
        let host = ToolHost::new(&workspace_in(&dir), no_secrets()).unwrap();
        assert_eq!(
            host.context_for(ToolMode::Restricted, &host.conversation_files(), "c1")
                .secrets
                .tavily_api_key(),
            None
        );
    }

    /// **A file one conversation writes is never what another reads.** Two
    /// conversations' rounds run in contexts bound to their own file stores,
    /// and neither ever changes the file on disk.
    #[test]
    fn each_conversation_has_its_own_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("r")).unwrap();
        std::fs::write(dir.path().join("r/shared.txt"), "disk\n").unwrap();
        let host = ToolHost::new(&workspace_in(&dir), no_secrets()).unwrap();
        let (a, b) = (host.conversation_files(), host.conversation_files());
        let in_a = host.context_for(ToolMode::Comprehensive, &a, "a");
        let in_b = host.context_for(ToolMode::Comprehensive, &b, "b");

        in_a.files
            .repo("r")
            .unwrap()
            .write("shared.txt", "conversation a\n".into())
            .unwrap();
        in_a.files
            .repo("r")
            .unwrap()
            .write("only_a.txt", "a\n".into())
            .unwrap();
        let b_store = in_b.files.repo("r").unwrap();
        assert_eq!(
            b_store.read("shared.txt").unwrap().as_deref(),
            Some("disk\n")
        );
        assert_eq!(b_store.read("only_a.txt").unwrap(), None);
        // A later round of the same conversation sees its own change.
        let again = host.context_for(ToolMode::Comprehensive, &a, "a");
        assert_eq!(
            again
                .files
                .repo("r")
                .unwrap()
                .read("shared.txt")
                .unwrap()
                .as_deref(),
            Some("conversation a\n")
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("r/shared.txt")).unwrap(),
            "disk\n"
        );
    }

    // ── Resuming a tool round after a restart ───────────────────────────────

    /// One call that may be re-issued and one that may not.
    pub(in super::super) fn a_mixed_round() -> Vec<ToolCall> {
        vec![
            ToolCall {
                name: "calculator".to_string(),
                arguments: serde_json::json!({ "expression": "1 + 1" }),
            },
            // No such session exists, so this reaches no network either way —
            // what is being asserted is whether it is put to the tool at all.
            ToolCall {
                name: "http_request".to_string(),
                arguments: serde_json::json!({
                    "session_id": "s1",
                    "path": "/v1/things",
                    "method": "POST",
                }),
            },
        ]
    }

    /// **A resumed round refuses only what cannot be re-issued, and still
    /// answers every call.** The round keeps its shape — one response per call
    /// the model made — so a refusal is something it reads and can act on
    /// rather than a hole where a result should be.
    #[test]
    fn a_resumed_round_refuses_only_the_calls_that_cannot_be_re_issued() {
        let ctx = ToolContext::default();
        let results = run_tool_calls(&ctx, a_mixed_round(), Dispatch::Resumed);
        assert_eq!(results.len(), 2);
        assert!(
            results[0].response.get("error").is_none(),
            "the arithmetic should have been re-run: {:?}",
            results[0].response,
        );
        assert_eq!(
            results[1].response["error"], "not_replayed_after_restart",
            "a POST must not be sent a second time: {:?}",
            results[1].response,
        );
    }

    /// **The refusal belongs to resume alone.** Dispatched live, the same round
    /// runs both calls — the POST fails on its missing session, which is the
    /// tool's own answer and not a refusal.
    #[test]
    fn a_live_round_puts_every_call_to_its_tool() {
        let ctx = ToolContext::default();
        let results = run_tool_calls(&ctx, a_mixed_round(), Dispatch::Live);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_ne!(
                r.response["error"], "not_replayed_after_restart",
                "a live turn refuses nothing: {:?}",
                r.response,
            );
        }
    }

    #[test]
    fn run_tool_unknown_returns_error_shape() {
        let ctx = ToolContext::default();
        let call = ToolCall {
            name: "tool_that_does_not_exist".to_string(),
            arguments: serde_json::json!({}),
        };
        let resp = run_tool(&ctx, &call);
        assert_eq!(resp["error"], "unknown_tool");
        assert!(resp["detail"]
            .as_str()
            .unwrap()
            .contains("tool_that_does_not_exist"));
    }

    #[test]
    fn run_tool_known_dispatches() {
        // datetime is a no-state tool that should always succeed with empty args.
        let ctx = ToolContext::default();
        let call = ToolCall {
            name: "datetime".to_string(),
            arguments: serde_json::json!({}),
        };
        let resp = run_tool(&ctx, &call);
        // Should be a successful response (not an error shape).
        assert!(
            resp.get("error").is_none(),
            "datetime call returned error: {resp:?}"
        );
    }
}
