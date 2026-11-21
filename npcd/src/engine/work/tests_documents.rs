//! Tests the working set: a document a body writes is visible only to that body
//! until a commit, and an edit must match the anchor the body read.
//!
//! A commit lands in the operation's store and never on the disk, and a read
//! shows a body only its own changes. File acts take their subject from the path,
//! and paths are checked at the act. An edit with a stale anchor is refused, and
//! the body is told to read again. Two bodies on one document collide at the
//! commit, and the refused one keeps all of its work.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::sim::shelf::NO_OPERATION;

    use serde_json::json;

    use super::super::tests_mission::tests::{on_operation, stored, vault, OP};

    pub(in super::super) fn act(tool: &'static str, args: Value) -> Act {
        Act {
            tool,
            args: args.as_object().unwrap().clone(),
        }
    }

    /// The vault, with somewhere to keep documents and two documents in it.
    ///
    /// A world without a root is a world with nothing to edit, so every test
    /// below that touches a document needs this rather than [`vault`].
    pub(in super::super) fn vault_with_documents(name: &str) -> (Hosted, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("npcd-work-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("layers/eras")).unwrap();
        std::fs::write(
            root.join("layers/eras/third.md"),
            "the third era\nburned in the spring\n",
        )
        .unwrap();
        std::fs::write(root.join("layers/eras/fourth.md"), "the fourth era\n").unwrap();

        // A mood and a personality, both carrying the comments that are the
        // reason the splice exists.
        std::fs::create_dir_all(root.join("moods")).unwrap();
        std::fs::write(
            root.join("moods/undone.yaml"),
            "id: undone\ncategory: mood\ndescription: Opened by what just happened.\n\n\
             # The felt register — its KV is loaded; a spike selects it.\ntemplate: |\n  \
             Something has been opened.\n",
        )
        .unwrap();
        std::fs::create_dir_all(root.join("personalities")).unwrap();
        std::fs::write(
            root.join("personalities/ash-the-drifter.yaml"),
            "# ash-the-drifter — identity definition.\n#\n# Biography is NOT here.\n\n\
             id: ash-the-drifter\nname: Ash\n\nportrait:\n  \
             image: portraits/ash-the-drifter.png\n  prompt: >-\n    a lean sun-darkened man\n",
        )
        .unwrap();

        let h = vault();
        h.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
            &root,
        ))));
        on_operation(&h, &root);
        (h, root)
    }

    /// **A write is in the working set and nowhere else until the commit**,
    /// and the commit lands it in the operation's store, never on the disk —
    /// the record moves only at a publish (§10.3).
    #[test]
    fn a_written_document_reaches_the_store_only_at_the_commit() {
        let (h, root) = vault_with_documents("commit");
        let wrote = perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/fifth.md","content":"the fifth era\n"}),
            ),
        );
        assert!(wrote.happened(), "{wrote:?}");
        assert_eq!(
            stored(&h, "layers/eras/fifth.md"),
            None,
            "the store moved early"
        );

        let out = perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"named the fifth"})),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            stored(&h, "layers/eras/fifth.md").as_deref(),
            Some("the fifth era\n")
        );
        assert!(
            !root.join("layers/eras/fifth.md").exists(),
            "the disk moved"
        );
    }

    /// **A body on no operation has nowhere to commit**: what it writes stays
    /// in its working set, and the commit says why.
    #[test]
    fn a_commit_on_no_operation_is_refused() {
        let (h, root) = vault_with_documents("no-operation");
        h.with_sim(|s| s.missions.cancel("m1"));
        assert!(perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/fifth.md","content":"the fifth era\n"}),
            ),
        )
        .happened());
        match perform(&h, "m1", &act("bench_commit", json!({"why":"named it"}))) {
            Outcome::Refused(why) => assert!(why.contains(NO_OPERATION), "{why}"),
            other => panic!("a commit with no operation was taken: {other:?}"),
        }
        assert!(!root.join("layers/eras/fifth.md").exists());
        assert_eq!(stored(&h, "layers/eras/fifth.md"), None);
    }

    /// A body reads its own uncommitted work; nobody else can see it at all.
    #[test]
    fn a_read_shows_your_own_changes_and_nobody_elses() {
        let (h, _) = vault_with_documents("read");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"rewritten\n"}),
            ),
        );
        let mine = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(mine.line().unwrap().contains("rewritten"), "{mine:?}");

        let theirs = perform(
            &h,
            "m2",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(
            theirs.line().unwrap().contains("burned in the spring"),
            "a working set leaked: {theirs:?}"
        );
    }

    /// **The subject of a file act is its `path`, never its text.** `story_draft`
    /// once wrote a draft into a record item named after the draft's own prose
    /// because a generic subject read the text first. The same shape is here —
    /// `file_write` carries a path and a body of text — so it is pinned.
    #[test]
    fn the_document_written_is_the_one_the_path_names() {
        let (h, root) = vault_with_documents("subject");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/sixth.md","content":"a night at the gate"}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert!(stored(&h, "layers/eras/sixth.md").is_some());
        assert_eq!(
            h.op_shelf(OP).unwrap().list("layers/eras"),
            ["fourth.md", "sixth.md", "third.md"]
                .map(String::from)
                .into(),
            "wrote into the prose"
        );
        assert!(!root.join("layers/eras/sixth.md").exists());
    }

    /// **A document is written exactly as it was given.** A trailing newline is
    /// part of the file, and the generic argument reader trims — which would
    /// make every document the bench writes quietly differ from what the
    /// character said, and report success doing it.
    #[test]
    fn a_document_is_written_exactly_as_it_was_given() {
        let (h, _) = vault_with_documents("verbatim");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/fifth.md","content":"a line\n\nand another\n"}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert_eq!(
            stored(&h, "layers/eras/fifth.md").as_deref(),
            Some("a line\n\nand another\n")
        );
    }

    /// The anchor of paragraph `i` of `text`, as a read of `path` shows it.
    pub(in super::super) fn anchor(path: &str, text: &str, i: usize) -> String {
        Doc::new(path, text).units()[i].anchor.clone()
    }

    /// The same reason from the other side: an edit has to keep leading
    /// whitespace, or it cannot touch a `.yaml` document at all — the
    /// indentation *is* the structure there.
    #[test]
    fn an_edit_keeps_the_indentation_it_was_given() {
        let (h, root) = vault_with_documents("indent");
        let text = "a:\n  b: 1\n  c: 1\n";
        std::fs::write(root.join("layers/eras/third.md"), text).unwrap();
        let at = anchor("layers/eras/third.md", text, 0);
        let out = perform(
            &h,
            "m1",
            &act(
                "file_edit",
                json!({"path":"layers/eras/third.md","at":at,"do":"replace",
                       "old":"  b: 1","text":"  b: 2"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert_eq!(
            stored(&h, "layers/eras/third.md").as_deref(),
            Some("a:\n  b: 2\n  c: 1\n")
        );
    }

    /// A read is capped and says so, so a long document cannot swallow a
    /// character's context window in one act.
    #[test]
    fn a_read_is_capped_and_carries_its_own_continuation() {
        let (h, root) = vault_with_documents("cap");
        let body: String = (1..=900).map(|i| format!("line {i}\n")).collect();
        std::fs::write(root.join("layers/eras/long.md"), &body).unwrap();

        let first = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/long.md"})),
        );
        // One 900-line paragraph is cut into parts of 80 lines, two to a page.
        let shown = first.line().unwrap();
        assert!(
            shown.contains("(page 1 of 6: paragraphs 1–2 of 12)"),
            "{shown}"
        );
        assert!(
            !shown.contains("line 161"),
            "the cap did not hold at the act"
        );
        assert!(shown.contains("the next read asks for page 2"), "{shown}");

        let next = perform(
            &h,
            "m1",
            &act(
                "file_read",
                json!({"path":"layers/eras/long.md","page":"2"}),
            ),
        );
        assert!(
            next.line()
                .unwrap()
                .contains("(page 2 of 6: paragraphs 3–4 of 12)"),
            "{next:?}"
        );
    }

    /// `page` is a string because the grammar cannot bound an integer, so the
    /// act parses it — and something unparseable means the first page rather
    /// than a refusal that spends the turn on arithmetic.
    #[test]
    fn an_unparseable_page_reads_from_the_top() {
        let (h, _) = vault_with_documents("start-junk");
        for junk in ["the top", "", "-4", "one"] {
            let out = perform(
                &h,
                "m1",
                &act(
                    "file_read",
                    json!({"path":"layers/eras/third.md","page":junk}),
                ),
            );
            assert!(out.happened(), "{junk:?}: {out:?}");
            assert!(out.line().unwrap().contains("the third era"), "{junk:?}");
        }
    }

    /// **Every file act takes its subject from `path`.** The subject is picked
    /// by scanning a list of argument names in order, and the new parameters
    /// sit next to entries in that list — `page` beside `for`, `content` and
    /// `text` beside `what`, `to` an addressee's word. If one of them ever
    /// wins, the act operates on the prose instead of the document, reports
    /// success, and nobody finds out until the record is wrong. That is exactly
    /// what `story_draft` did.
    #[test]
    fn the_subject_of_every_file_act_is_its_path() {
        for args in [
            json!({"path":"layers/eras/third.md"}),
            json!({"path":"layers/eras/third.md","page":"2"}),
            json!({"path":"layers/eras/third.md","content":"what the document says"}),
            json!({"path":"layers/eras/third.md","at":"Qm7x","do":"replace","old":"a","text":"b"}),
            json!({"path":"layers/eras/third.md","at":"Qm7x","do":"move","to":"b4_K",
                   "dest":"cX0e","where":"after"}),
        ] {
            let map = args.as_object().unwrap().clone();
            assert_eq!(
                subject(&map).as_deref(),
                Some("layers/eras/third.md"),
                "the subject was taken from the wrong argument in {args}"
            );
        }
    }

    #[test]
    fn an_edit_leaves_everything_it_did_not_name_alone() {
        let (h, _) = vault_with_documents("edit");
        let path = "layers/eras/third.md";
        let at = anchor(path, "the third era\nburned in the spring\n", 0);
        let out = perform(
            &h,
            "m1",
            &act(
                "file_edit",
                json!({"path":path,"at":at,"do":"replace","old":"spring","text":"autumn"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        // The answer shows the paragraph as a block under its new anchor.
        let now = anchor(path, "the third era\nburned in the autumn\n", 0);
        assert_eq!(
            out.line().unwrap(),
            format!(
                "{path}: Changed {at}.\n```markdown file={path} part={now} lines=1-2\n +1 the \
                 third era\n +2 burned in the autumn\n```\n"
            )
        );
        perform(
            &h,
            "m1",
            &act(
                "bench_commit",
                json!({"why":"dated it against its neighbours"}),
            ),
        );
        assert_eq!(
            stored(&h, "layers/eras/third.md").as_deref(),
            Some("the third era\nburned in the autumn\n")
        );
    }

    #[test]
    fn an_ambiguous_edit_is_refused_rather_than_applied_to_the_wrong_one() {
        let (h, root) = vault_with_documents("ambiguous");
        let text = "a fire\nand a fire\n";
        std::fs::write(root.join("layers/eras/third.md"), text).unwrap();
        let at = anchor("layers/eras/third.md", text, 0);
        let out = perform(
            &h,
            "m1",
            &act(
                "file_edit",
                json!({"path":"layers/eras/third.md","at":at,"do":"replace",
                       "old":"a fire","text":"a flood"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("2 times"), "{out:?}");
    }

    #[test]
    fn an_edit_missing_what_it_needs_says_what() {
        let (h, _) = vault_with_documents("half");
        let path = "layers/eras/third.md";
        let at = anchor(path, "the third era\nburned in the spring\n", 0);
        for (args, says) in [
            (json!({"path":path,"at":at,"do":"replace"}), "`text`"),
            (json!({"path":path,"do":"delete"}), "`at`"),
            (
                json!({"path":path,"at":at,"do":"rewrite"}),
                "replace, insert_before",
            ),
            (
                json!({"path":path,"at":at,"do":"move","dest":at}),
                "`before` or `after`",
            ),
        ] {
            let out = perform(&h, "m1", &act("file_edit", args));
            assert!(!out.happened());
            assert!(out.line().unwrap().contains(says), "{says}: {out:?}");
        }
    }

    /// **An anchor read before its paragraph changed is refused**, and told to
    /// read again — never applied to text the character did not see.
    #[test]
    fn a_stale_anchor_is_refused_and_told_to_read_again() {
        let (h, _) = vault_with_documents("stale");
        let path = "layers/eras/third.md";
        let at = anchor(path, "the third era\nburned in the spring\n", 0);
        let replace = |text: &str| {
            act(
                "file_edit",
                json!({"path":path,"at":at,"do":"replace","text":text}),
            )
        };
        assert!(perform(&h, "m1", &replace("the third era\nburned in the autumn")).happened());
        // The same anchor again names the paragraph as it was.
        let out = perform(&h, "m1", &replace("the third era\nburned in winter"));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("read it again"), "{out:?}");
    }

    #[test]
    fn a_listing_shows_the_disk_and_your_own_new_documents() {
        let (h, _) = vault_with_documents("list");
        let before = perform(&h, "m1", &act("file_list", json!({"path":"layers/eras"})));
        assert!(before.line().unwrap().contains("third.md"), "{before:?}");

        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/fifth.md","content":"…"}),
            ),
        );
        let after = perform(&h, "m1", &act("file_list", json!({"path":"layers/eras"})));
        assert!(after.line().unwrap().contains("fifth.md"), "{after:?}");
        // …and not to anybody else, because it is not committed.
        let theirs = perform(&h, "m2", &act("file_list", json!({"path":"layers/eras"})));
        assert!(!theirs.line().unwrap().contains("fifth.md"), "{theirs:?}");
    }

    #[test]
    fn a_deleted_document_goes_from_the_store_at_the_commit() {
        let (h, root) = vault_with_documents("delete");
        let out = perform(
            &h,
            "m1",
            &act("file_delete", json!({"path":"layers/eras/fourth.md"})),
        );
        assert!(out.happened(), "{out:?}");
        assert!(
            stored(&h, "layers/eras/fourth.md").is_some(),
            "gone before the commit"
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"never canon"})));
        assert_eq!(stored(&h, "layers/eras/fourth.md"), None);
        assert!(
            root.join("layers/eras/fourth.md").exists(),
            "only a publish reaches the record"
        );
    }

    /// The one act somebody else's work can refuse — now over a real document,
    /// and the refusal still names the other party.
    #[test]
    fn two_makers_on_one_document_collide_at_the_commit() {
        let (h, _) = vault_with_documents("collide");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"mine\n"}),
            ),
        );
        perform(
            &h,
            "m2",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"mine too\n"}),
            ),
        );

        assert!(perform(&h, "m1", &act("bench_commit", json!({"why":"first"}))).happened());
        let out = perform(&h, "m2", &act("bench_commit", json!({"why":"second"})));
        assert!(!out.happened());
        assert!(
            out.line().unwrap().contains("m1"),
            "no other party named: {out:?}"
        );
        assert!(
            out.line().unwrap().contains("talk to"),
            "a collision read as a retryable error: {out:?}"
        );
        // The refused body still has all of its work.
        let still = perform(&h, "m2", &act("bench_diff", json!({})));
        assert!(
            still.line().unwrap().contains("layers/eras/third.md"),
            "{still:?}"
        );
    }

    #[test]
    fn setting_work_aside_and_picking_it_up_again_survives_the_round_trip() {
        let (h, _) = vault_with_documents("stash");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"half\n"}),
            ),
        );
        assert!(perform(&h, "m1", &act("bench_stash", json!({}))).happened());
        let gone = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(
            gone.line().unwrap().contains("burned in the spring"),
            "{gone:?}"
        );

        assert!(perform(&h, "m1", &act("bench_stash_pop", json!({}))).happened());
        let back = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(back.line().unwrap().contains("half"), "{back:?}");
    }

    #[test]
    fn throwing_the_work_away_puts_the_document_back() {
        let (h, _) = vault_with_documents("restore");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"wrong\n"}),
            ),
        );
        assert!(perform(&h, "m1", &act("bench_restore", json!({}))).happened());
        let back = perform(
            &h,
            "m1",
            &act("file_read", json!({"path":"layers/eras/third.md"})),
        );
        assert!(
            back.line().unwrap().contains("burned in the spring"),
            "{back:?}"
        );
    }

    #[test]
    fn status_and_diff_name_the_documents_that_are_changed() {
        let (h, _) = vault_with_documents("status");
        let quiet = perform(&h, "m1", &act("bench_status", json!({})));
        assert!(quiet.line().unwrap().contains("Nothing open"), "{quiet:?}");

        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"changed\n"}),
            ),
        );
        let diff = perform(&h, "m1", &act("bench_diff", json!({})));
        assert!(
            diff.line()
                .unwrap()
                .contains("layers/eras/third.md (changed)"),
            "{diff:?}"
        );

        perform(&h, "m1", &act("bench_stage", json!({})));
        let staged = perform(&h, "m1", &act("bench_status", json!({})));
        assert!(staged.line().unwrap().contains("Offered"), "{staged:?}");
    }

    /// The paths are written by a language model, so the guard has to hold at
    /// the act, not only in the store beneath it.
    #[test]
    fn a_path_that_leaves_the_world_is_refused_at_the_act() {
        let (h, root) = vault_with_documents("escape");
        for bad in [
            "../stolen.md",
            "layers/eras/../../stolen.md",
            "c:/windows/x.md",
            "layers/eras/x.exe",
        ] {
            let out = perform(
                &h,
                "m1",
                &act("file_write", json!({"path":bad,"content":"owned"})),
            );
            assert!(!out.happened(), "{bad} was written");
            assert!(!perform(&h, "m1", &act("file_read", json!({"path":bad}))).happened());
        }
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        assert!(!root.parent().unwrap().join("stolen.md").exists());
    }

    /// A world with nothing to edit refuses rather than inventing somewhere.
    #[test]
    fn a_world_with_no_documents_refuses_every_file_act() {
        let h = vault();
        for a in [
            act("file_read", json!({"path":"layers/eras/third.md"})),
            act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"x"}),
            ),
            act("file_list", json!({"path":"layers/eras"})),
        ] {
            let out = perform(&h, "m1", &a);
            assert!(!out.happened(), "{:?} happened with no documents", a.tool);
        }
    }

    /// Committing a document is not the same as being the one who holds it, and
    /// the history says who actually did it.
    #[test]
    fn blame_on_a_document_names_whoever_committed_it() {
        let (h, _) = vault_with_documents("blame");
        perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path":"layers/eras/third.md","content":"mine\n"}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"…"})));
        let out = perform(
            &h,
            "m2",
            &act("bench_blame", json!({"what":"layers/eras/third.md"})),
        );
        assert!(out.line().unwrap().contains("m1"), "{out:?}");
    }

    // ── the join: stations write documents ──────────────────────────────────
    //
    // The whole point of `index_canon` and `settle_path`. Before these, every
    // act at a chronicle terminal reported success against an in-memory fixture
    // and changed nothing that survived a restart.

    /// **An entry written at a chronicle terminal reaches the era document in
    /// the operation's store.** The one test that says the stations are real.
    #[test]
    fn an_entry_written_into_an_era_lands_in_the_era_document() {
        let (h, root) = vault_with_documents("era-entry");
        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"}))
        )
        .happened());
        let out = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"The redoubt fell in the spring."}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            stored(&h, "layers/eras/the-third-era.md"),
            None,
            "the store moved before the commit"
        );

        assert!(perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"dated the fall"}))
        )
        .happened());
        assert!(!root.join("layers/eras/the-third-era.md").exists());
        let written = stored(&h, "layers/eras/the-third-era.md").unwrap();
        assert!(
            written.contains("The redoubt fell in the spring."),
            "{written}"
        );
    }

    /// A second entry is added to the first, not written over it.
    #[test]
    fn a_second_entry_is_added_rather_than_replacing_the_first() {
        let (h, _) = vault_with_documents("era-append");
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"})),
        );
        perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"First."}),
            ),
        );
        perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"Second."}),
            ),
        );
        perform(&h, "m1", &act("bench_commit", json!({"why":"two entries"})));

        let written = stored(&h, "layers/eras/the-third-era.md").unwrap();
        assert!(written.contains("First."), "{written}");
        assert!(
            written.contains("Second."),
            "the second entry replaced the first: {written}"
        );
        assert!(
            written.find("First.") < written.find("Second."),
            "the record came back out of order: {written}"
        );
    }

    /// A document already on the disk *is* the thing the world names, rather
    /// than a twin standing beside it.
    #[test]
    fn an_era_document_on_disk_becomes_the_thing_the_world_already_names() {
        let (h, root) = vault_with_documents("era-index");
        std::fs::write(
            root.join("layers/eras/burning.md"),
            "# the fourth era\n\nQuiet, and then not.\n",
        )
        .unwrap();
        h.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
            &root,
        ))));

        h.sim(|s| {
            let names: Vec<&str> = s.record.iter().map(|i| i.name.as_str()).collect();
            assert_eq!(
                names.iter().filter(|n| **n == "the fourth era").count(),
                1,
                "the document made a twin: {names:?}"
            );
            assert_eq!(
                s.record.path_of("the fourth era").as_deref(),
                Some("layers/eras/burning.md"),
                "the era did not take its document"
            );
        });
    }

    /// A story is a document too, and lands in its own layer.
    #[test]
    fn a_draft_written_into_a_silence_lands_in_the_stories_layer() {
        let (h, _) = vault_with_documents("story-draft");
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third silence"})),
        );
        let out = perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"A night at the gate."}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"filled the longest silence"})),
        );

        let draft = stored(&h, "layers/stories/the-third-silence.md")
            .expect("the draft did not reach the stories layer");
        assert!(draft.contains("A night at the gate."));
    }

    /// **A filed document is not written over casually.** It refuses until the
    /// writer has opened it, which is the bench loop said by the store.
    #[test]
    fn a_filed_era_refuses_a_write_until_it_is_opened() {
        let (h, _root) = vault_with_documents("era-filed");
        let closed = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the fourth era","what":"…"}),
            ),
        );
        assert!(!closed.happened());
        assert!(
            closed.line().unwrap().contains("Open it first"),
            "{closed:?}"
        );

        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the fourth era"}))
        )
        .happened());
        let open = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the fourth era","what":"And then not."}),
            ),
        );
        assert!(open.happened(), "{open:?}");
    }

    /// Two Makers on one era collide at the commit, over a real file.
    #[test]
    fn two_makers_writing_one_era_collide_at_the_commit() {
        let (h, _) = vault_with_documents("era-collide");
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"})),
        );
        perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"Mine."}),
            ),
        );
        // m1 is holding it, so m2 is refused by custody before it reaches the
        // document at all — the record's own rule, still doing its job.
        let blocked = perform(
            &h,
            "m2",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"Mine too."}),
            ),
        );
        assert!(!blocked.happened());
        assert!(blocked.line().unwrap().contains("m1"), "{blocked:?}");
    }

    /// A thing that is a judgement rather than a document still lives in the
    /// record — the join must not sweep everything onto the disk.
    #[test]
    fn a_judgement_stays_in_the_record_and_gets_no_document() {
        let (h, root) = vault_with_documents("judgement");
        let out = perform(
            &h,
            "m1",
            &act(
                "record_appraise",
                json!({"what":"the third era","verdict":"sound"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| assert!(s.record.path_of("the western intake").is_none()));
        assert!(
            !root
                .join("layers/eras")
                .join("the-western-intake.md")
                .exists(),
            "an accession was given a document"
        );
    }

    #[test]
    fn a_filed_document_is_not_deleted_as_though_it_were_scratch() {
        let h = vault();
        perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third silence"})),
        )
        .happened();
        perform(
            &h,
            "m1",
            &act("story_draft", json!({"for":"the third silence","what":"x"})),
        );
        perform(
            &h,
            "m1",
            &act("story_file", json!({"what":"the third silence"})),
        );
        let out = perform(
            &h,
            "m1",
            &act("file_delete", json!({"path":"the third silence"})),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("record_let_go"), "{out:?}");
    }
}
