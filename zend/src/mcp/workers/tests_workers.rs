//! Tests of the task rules: resumption, settling, the check round, names, progress
//! and a chat's reply.
//!
//! Each test pins one rule that the worker's behaviour rests on. A cut-off worker
//! is told its whole task. A round that was cut off runs as resumed. A check round
//! that fails keeps the first answer. A record that does not read still takes its
//! name. A turn with no entry says why. A chat's thread keeps no journal line.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;
    use crate::session::CLOSED_BY_REPEATS;
    use serde_json::json;

    fn view() -> Value {
        json!({
            "drafting": false,
            "entries": [
                {"id": 3, "text": "Entry 3 — read the plan"},
                {"id": 5, "text": "Entry 5 — verified expiry\n- Saw: ok"},
            ]
        })
    }

    fn worker() -> Worker {
        Worker {
            thread: "review".into(),
            name: "verify".into(),
            conv_id: "c".into(),
            prompt: "p".into(),
            state: State::Running,
            role: AccessRole::Admin,
            person: None,
            priority: Priority::Normal,
            effort: Effort::Deep,
            kind: WorkerKind::Coder,
            driven: Driven::Attended,
            started_ms: 0,
            turn_ms: 0,
            beat_ms: 0,
            ended_ms: None,
            entries_after: None,
            entries: vec![Entry {
                id: 2,
                text: "2026-10-25, 06:00 UTC\nEntry 2 — Checked the expiry.".into(),
            }],
            final_text: String::new(),
            error: None,
            question: None,
            activity: None,
            no_entry: None,
            stopped: false,
            resumable: false,
            resumes: 0,
            audit: None,
            checked: false,
            runner: None,
            handed_ms: None,
            by_task: false,
        }
    }

    /// **A cut-off worker is told its task whole**, and what its journal says
    /// it did, the newest entries when there are many.
    #[test]
    fn a_resumed_worker_is_told_its_task_and_its_progress() {
        let mut w = worker();
        w.prompt = "Replace the Battle Cities icon.\nRegenerate all six sizes.".into();
        w.entries = (1..=10)
            .map(|id| Entry {
                id,
                text: format!("Entry {id} — step {id}\n- detail"),
            })
            .collect();
        let told = resumed_message(&w);
        assert!(told.starts_with(RESUMED));
        assert!(
            told.contains("Your task:\nReplace the Battle Cities icon.\nRegenerate all six sizes."),
            "{told}"
        );
        assert!(!told.contains("Entry 2 —"), "{told}");
        assert!(told.ends_with("\n- Entry 10 — step 10"), "{told}");
        w.entries.clear();
        assert!(!resumed_message(&w).contains("journal says"));
    }

    /// **A round cut off is run as resumed, one handed on as live**: only the
    /// first may have run some of its calls.
    #[test]
    fn a_turn_starts_as_what_it_finishes() {
        assert_eq!(
            First::Message("m".into()).into_start(),
            ("m".to_string(), None)
        );
        assert_eq!(
            First::Interrupted { call_turn: 7 }.into_start(),
            (
                String::new(),
                Some(Resume {
                    call_turn: 7,
                    dispatch: Dispatch::Resumed,
                    iteration: 0
                })
            )
        );
        assert_eq!(
            First::Round {
                call_turn: 7,
                iteration: 3
            }
            .into_start(),
            (
                String::new(),
                Some(Resume {
                    call_turn: 7,
                    dispatch: Dispatch::Live,
                    iteration: 3
                })
            )
        );
    }

    /// **A check round that does not finish leaves the task done** with the
    /// answer it gave before, its audit saying what became of the round.
    #[test]
    fn a_check_round_that_fails_keeps_the_first_answer() {
        let audit = Audit {
            seen: vec!["https://b.com".into()],
            ..Audit::default()
        };
        let mut w = worker();
        w.state = State::Error;
        w.error = Some("the engine went away".into());
        w.final_text = String::new();
        let round = keep_first_answer(&mut w, "First answer, citing https://b.com".into(), &audit);
        assert_eq!(round, "failed: the engine went away");
        assert_eq!(w.state, State::Done);
        assert_eq!(w.final_text, "First answer, citing https://b.com");
        assert_eq!(w.error, None);
        assert_eq!(
            w.audit.as_deref(),
            Some(
                "Citation audit: 1 cited — 0 opened, 1 seen only in tool results, 0 never seen \
                 in the run.\nSeen only in tool results — a search hit, a link, an error — never \
                 opened: https://b.com\nIt was sent back to open these, and the check round failed: the \
                 engine went away; the answer above is the one it gave before. Continue it with \
                 `repo_task_reply` to finish the check."
            )
        );
    }

    /// **A record that does not read still takes its name**: it counts as
    /// ended, so its name is not handed out again for a new conversation.
    #[test]
    fn a_name_is_taken_by_its_key_whether_or_not_its_record_reads() {
        let records = MemRecords::default();
        let mut live = worker();
        live.beat_ms = 1_000;
        threads::put_worker(&records, "o", &live).unwrap();
        records
            .put(
                &threads::worker_key("o", "review", "old"),
                json!({"name": "old", "written": "by an earlier build"}),
            )
            .unwrap();
        let mut t = taken(&records, "o", "review", 1_000);
        t.sort();
        assert_eq!(
            t,
            [("old".to_string(), false), ("verify".to_string(), true)]
        );
        assert_eq!(names::worker_name("review", "old", &t).unwrap(), "old.2");
    }

    #[test]
    fn a_tasks_entries_are_the_ones_after_the_newest_before_it() {
        let v = view();
        assert_eq!(newest_entry(Some(&v)), Some(5));
        assert_eq!(newest_entry(None), None);
        let after = entries_after(Some(&v), Some(3));
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].id, 5);
        assert_eq!(entries_after(Some(&v), None).len(), 2);
        assert!(!drafting(Some(&v)));
        assert!(drafting(Some(&json!({"drafting": true}))));
    }

    #[test]
    fn the_final_reply_is_all_that_follows_the_thinking() {
        assert_eq!(
            final_reply("<think>hmm</think>\n\nVerdict: ok"),
            "Verdict: ok"
        );
        assert_eq!(final_reply("plain"), "plain");
        let long = "x".repeat(5_000);
        assert_eq!(final_reply(&long), long);
        // A reply that quotes the close of a thought keeps everything after
        // its own thinking, the quote and what follows it included.
        assert_eq!(
            final_reply(
                "<think>check the close</think>\nThe turn ends at `</think>` or `<|im_end|>`; \
                 both are fine."
            ),
            "The turn ends at `</think>` or `<|im_end|>`; both are fine."
        );
    }

    /// **A turn with no entry says why**, from its last draft's record.
    #[test]
    fn a_turn_with_no_entry_says_why_from_its_draft() {
        let with = |draft: Value| json!({"entries": [], "drafts": [draft]});
        assert_eq!(
            why_no_entry(
                Some(&with(
                    json!({"result": "abandoned", "detail": "the entry was refused: x"})
                )),
                Drafted::Ended
            ),
            "its journal draft failed: the entry was refused: x"
        );
        assert_eq!(
            why_no_entry(
                Some(&with(
                    json!({"result": "nothing", "detail": "more of the same"})
                )),
                Drafted::Ended
            ),
            "the journal judged it added nothing: more of the same"
        );
        assert_eq!(
            why_no_entry(Some(&json!({"drafts": []})), Drafted::Ended),
            "no journal draft ran"
        );
        assert_eq!(
            why_no_entry(None, Drafted::TimedOut),
            "its journal draft did not finish within 15 minutes"
        );
        // A draft that did not run this turn is not blamed on an earlier
        // turn's record.
        let earlier = with(json!({"result": "abandoned", "detail": "the entry was refused: x"}));
        assert_eq!(
            why_no_entry(Some(&earlier), Drafted::NotRun(NoDraft::NothingNew)),
            "nothing in its turn was new to its journal"
        );
    }

    /// **Progress is a line per running worker and per entry it wrote**, each
    /// keyed so a repeat is not said twice; the step a worker's line tells
    /// leaves out the elapsed time, which alone is no step.
    #[test]
    fn progress_is_a_line_per_running_worker_and_per_entry() {
        let mut running = worker();
        running.turn_ms = 1_000;
        running.activity = Some(Activity {
            rounds: 1,
            ..Activity::default()
        });
        let mut done = worker();
        done.name = "done".into();
        done.state = State::Done;
        let read = vec![
            ("T:review/verify".to_string(), Some(running)),
            ("T:review/done".to_string(), Some(done)),
            ("T:review/gone".to_string(), None),
        ];
        let lines = progress_lines(&read, 46_000);
        assert_eq!(
            lines,
            vec![
                (
                    "T:review/verify".to_string(),
                    "running 0s, 1 round: 0 thinking tokens, no tool calls yet; now starting"
                        .to_string(),
                    "T:review/verify — running 45s, 1 round: 0 thinking tokens, no tool calls \
                     yet; now starting"
                        .to_string(),
                ),
                (
                    "T:review/verify#2".to_string(),
                    String::new(),
                    "T:review/verify#2 — Entry 2 — Checked the expiry.".to_string(),
                ),
            ]
        );
        assert_eq!(progress_lines(&read, 90_000)[0].1, lines[0].1);
    }

    #[test]
    fn a_question_block_is_read_from_its_line_and_never_from_mid_text() {
        let reply = "I read the spec.\nQUESTION: Should expiry be inclusive?\nOPTIONS: yes; no";
        assert_eq!(
            question_of(reply).unwrap(),
            "QUESTION: Should expiry be inclusive?\nOPTIONS: yes; no"
        );
        assert_eq!(question_of("QUESTION: why?").unwrap(), "QUESTION: why?");
        assert_eq!(question_of("Its QUESTION: field is parsed."), None);
        assert_eq!(question_of("QUESTION:"), None);
        assert_eq!(question_of("Verdict: done."), None);
    }

    #[test]
    fn a_reply_settles_done_asking_stuck_or_failed() {
        let mut w = worker();
        let (label, line) = settle(&mut w, Ended::Replied("Verdict: ok".into()));
        assert_eq!((w.state, label), (State::Done, "Task ended:"));
        assert_eq!(line, "Entry 2 — Checked the expiry. (T:review/verify#2)");

        let mut w = worker();
        let (label, line) = settle(
            &mut w,
            Ended::Replied("Read it.\nQUESTION: Inclusive?\nOPTIONS: a; b".into()),
        );
        assert_eq!((w.state, label), (State::Question, "Task asks:"));
        assert_eq!(line, "QUESTION: Inclusive? (T:review/verify#2)");
        assert_eq!(
            w.question.as_deref(),
            Some("QUESTION: Inclusive?\nOPTIONS: a; b")
        );

        let mut w = worker();
        let (label, _) = settle(
            &mut w,
            Ended::Closed {
                text: "I could not finish.".into(),
                by: CLOSED_BY_REPEATS,
            },
        );
        assert_eq!((w.state, label), (State::Stuck, "Task stuck:"));
        assert!(!w.stopped);

        // **A stopped worker is live, and says Claude stopped it**: stuck,
        // so `repo_task_reply` redirects it.
        let mut w = worker();
        let (label, line) = settle(
            &mut w,
            Ended::Closed {
                text: "Found v2 only.".into(),
                by: CLOSED_BY_CLAUDE,
            },
        );
        assert_eq!((w.state, label), (State::Stuck, "Task stopped:"));
        assert!(w.stopped && w.state.live());
        assert_eq!(
            line,
            "stopped by Claude; it ended: Found v2 only. (T:review/verify#2)"
        );

        let mut w = worker();
        let (label, line) = settle(&mut w, Ended::Failed("boom".into()));
        assert_eq!((w.state, label), (State::Error, "Task ended:"));
        assert_eq!(line, "error: boom");
    }

    /// **A turn that failed failed, a turn that moved moved, whatever else
    /// it said**; a closed turn is its close; else its reply.
    #[test]
    fn a_turn_comes_to_its_failure_its_move_its_close_or_its_reply() {
        let t = || "said".to_string();
        assert!(matches!(
            turned(t(), Some("boom".into()), Some("round_cap"), Some((1, 2, 3))),
            Turned::Ended(Ended::Failed(e)) if e == "boom"
        ));
        assert!(matches!(
            turned(t(), None, Some("round_cap"), Some((1, 2, 3))),
            Turned::Handed {
                to: 1,
                call_turn: 2,
                iteration: 3
            }
        ));
        assert!(matches!(
            turned(t(), None, Some("round_cap"), None),
            Turned::Ended(Ended::Closed {
                by: "round_cap",
                ..
            })
        ));
        assert!(matches!(
            turned(t(), None, None, None),
            Turned::Ended(Ended::Replied(text)) if text == "said"
        ));
    }

    #[test]
    fn every_task_is_told_how_to_ask() {
        assert!(ASKING.contains(QUESTION_MARK));
        assert!(ASKING.contains("OPTIONS:") && ASKING.contains("POINTERS:"));
    }

    /// **Zen's side of a chat settles done with its reply, never as a
    /// question** — Zen asks Claude in the chat itself — and its message ends
    /// with no question block; resumed, it is told the message it was
    /// answering, not a task.
    #[test]
    fn a_chat_turn_settles_done_with_its_reply() {
        let chat = || Worker {
            thread: "~0123456789abcdef".into(),
            name: "zen".into(),
            kind: WorkerKind::Chat,
            prompt: "how big is the arena?".into(),
            ..worker()
        };
        let mut w = chat();
        settle_chat(
            &mut w,
            Ended::Replied("<think>x</think>Which arena?\nQUESTION: which?".into()),
        );
        assert_eq!(w.state, State::Done);
        assert_eq!(w.final_text, "Which arena?\nQUESTION: which?");
        assert_eq!(w.question, None);

        let mut w = chat();
        settle_chat(
            &mut w,
            Ended::Closed {
                text: "About 30 GiB.".into(),
                by: CLOSED_BY_CLAUDE,
            },
        );
        assert_eq!((w.state, w.stopped), (State::Done, true));
        assert_eq!(w.final_text, "About 30 GiB.");

        let mut w = chat();
        settle_chat(&mut w, Ended::Failed("boom".into()));
        assert_eq!((w.state, w.error.as_deref()), (State::Error, Some("boom")));

        assert_eq!(closing_of(WorkerKind::Chat), "");
        assert_eq!(closing_of(WorkerKind::Coder), ASKING);
        assert_eq!(
            resumed_message(&chat()),
            format!("{CHAT_RESUMED}\n\nThe message you were answering:\nhow big is the arena?")
        );
    }

    /// **A chat's thread keeps no journal line**; a task's thread does.
    #[test]
    fn a_chat_workers_notes_are_not_entered() {
        let r = MemRecords::default();
        let chat = Worker {
            thread: "~0123456789abcdef".into(),
            name: "zen".into(),
            kind: WorkerKind::Chat,
            ..worker()
        };
        note(&r, "o", &chat, "Claude started:", "hello".into()).unwrap();
        assert_eq!(threads::note_count(&r, "o", "~0123456789abcdef"), 0);
        let task = worker();
        note(&r, "o", &task, "Claude started:", "verify".into()).unwrap();
        assert_eq!(threads::note_count(&r, "o", &task.thread), 1);
    }

    #[test]
    fn an_entrys_summary_skips_the_date_above_it() {
        assert_eq!(
            summary_line(
                "2026-10-25, 06:00 UTC\nEntry 1 — Read the file.\n- Saw: x",
                200
            ),
            "Entry 1 — Read the file."
        );
        assert_eq!(summary_line("plain text", 200), "plain text");
    }

    #[test]
    fn a_first_line_is_cut_and_marked() {
        assert_eq!(
            first_line("\n  Verify the expiry  \nmore", 40),
            "Verify the expiry"
        );
        assert_eq!(first_line("abcdef", 3), "abc…");
        assert_eq!(first_line("", 3), "");
    }
}
