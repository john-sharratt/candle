//! Families A, B, C, D of the behavioural catalogue
//! (`docs/recurrent_state_behavioural_tests.md`): continuity, fork semantics,
//! isolation, and the record/K-V correspondence.
//!
//! Every scenario is the same oracle in a different costume: **do the thing, do
//! the thing with an interruption in the middle, and compare.** It needs no
//! reference implementation, which is what makes it decisive when the reference
//! is itself suspect.
//!
//! ```text
//! cargo test -p zend --release --features cuda --test memory_continuity \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## How one model load serves every test
//!
//! Each engine open loads a 22 GB checkpoint — about a minute — so one load
//! per `#[test]` would put a full run near an hour of *loading*. Instead:
//!
//! - **Single-engine scenarios** are individual `#[test]`s that all reuse ONE
//!   process-wide engine via [`common::shared_session`]: the first test to run
//!   loads it, every later one takes the same guard. Separate pass/fail
//!   reporting, one scenario runnable by name for one load, and a panic in one
//!   never aborts the others. The guard also SERIALISES the bodies — even
//!   under parallel test threads they queue on the engine — though
//!   `--test-threads=1` remains the documented mode because only serial ORDER
//!   makes runs reproducible.
//! - [`restart_invariants`] owns its engines (the drop and reopen is the
//!   subject) and takes the slot exclusively via
//!   [`common::exclusive_engine_slot`], dropping the shared engine first so at
//!   most one engine is resident at a time. libtest's name-sorted order runs
//!   it after the single-engine tests, so the release costs nothing on a full
//!   run.

mod common;

use candle::Device;
use common::{
    digest_of_layers, exclusive_engine_slot, memory, memory_is_empty, memory_of,
    poison_memory_record, probe_recall, say, say_n, say_opening, scenario, sealed_memory,
    sealed_memory_at, shared_session, Workspace, RECURRENT_LAYERS,
};

// ═══ One engine ═════════════════════════════════════════════════════════════
//
// Every invariant expressible against a single live engine is its own
// `#[test]`, and they all reuse ONE engine via [`common::shared_session`]:
// the 22 GB checkpoint loads once per test process, each test's first
// statement takes the shared guard (serialising the bodies even under
// parallel test threads), and a panic in one scenario never aborts the
// others. Run a single scenario by name for one load plus that scenario.

/// **E2 — a conversation opened on a different selector branch carries that
/// branch's memory.**
///
/// The composer dials (`response_length`, `persona`, …) arrive with the FIRST
/// turn's `TurnOptions.selection`, but the branch checkpoint installs at
/// create — under the DEFAULT selection, because create cannot know the dials.
/// If nothing re-keys the branch when the first turn's dials differ, every
/// dialed conversation runs on the default branch's prompt memory while
/// projecting the dialed prompt — the same class of defect as the resume
/// belief loss, at the prompt-branch level.
///
/// The oracle is the checkpoint counters: the first turn ever submitted under
/// a new dial assignment must COMPUTE that branch (or restore it, once it is
/// durable). Neither moving means the dial never reached the branch key.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn e2_a_dialed_first_turn_gets_its_own_branch() {
    use candle_conversation::TurnOptions;
    let session = shared_session();
    scenario("E2", "a dialed first turn gets its own prompt branch");
    let mut conv = session.start();
    let (computed_before, installed_before) = candle_conversation::branch_checkpoint_counts();
    let mut options = TurnOptions::default();
    options.selection.select("response_length", "comprehensive");
    let handle = conv
        .submit_turn_with_options("Dialed turn: reply with one short sentence.", options)
        .expect("submit dialed turn");
    let resp = handle.wait().expect("dialed turn completes");
    conv.finish_turn(handle, &resp).expect("finish");
    let (computed_after, installed_after) = candle_conversation::branch_checkpoint_counts();
    eprintln!(
        "[E2] dialed first turn: computed {computed_before}->{computed_after}, \
         installed {installed_before}->{installed_after}"
    );
    assert!(
        computed_after > computed_before || installed_after > installed_before,
        "E2: the first turn under a NEW dial assignment neither computed nor \
         restored its branch's checkpoint — the conversation is running on the \
         default branch's prompt memory while projecting the dialed prompt"
    );
}

/// **E1 — a brand-new conversation's memory already reflects the system
/// prompt.**
///
/// The branch checkpoint installs at creation, after the prompt K/V is primed
/// — so before the first turn, the conversation must already carry a non-empty
/// recurrent memory (the prompt's), not zeros. A conversation that starts at
/// zeros has the whole prompt in K/V and none of it in memory: it reads its
/// instructions fluently and does not remember them — §7.8 defect 2's quiet
/// prompt-shaped variant.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn e1_new_conversation_carries_prompt_memory() {
    let session = shared_session();
    scenario(
        "E1",
        "a new conversation's memory reflects the system prompt",
    );
    let conv = session.start();
    match memory_of(&conv) {
        None => panic!(
            "E1: a brand-new conversation carries no memory at all before its \
             first turn — the prompt branch checkpoint was never installed"
        ),
        Some(_) => assert!(
            !memory_is_empty(&conv),
            "E1: a brand-new conversation's memory is all zeros — it holds the \
             system prompt in K/V and remembers none of it"
        ),
    }
}

/// **A4 — tier migration is invisible.**
///
/// Demote a conversation's hot K/V to warm (the VRAM returns to the pool),
/// then continue the conversation — the next projection elevates it back. The
/// K/V content must round-trip byte-for-byte, and memory must not move: a
/// migration is a *placement* decision, and any content change it makes is
/// silent corruption dressed as memory management.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn a4_tier_migration_is_invisible() {
    let session = shared_session();
    scenario("A4", "hot→warm demote and elevate back is invisible");
    let mut conv = session.start();
    say(&mut conv, "The migration codeword is GLACIER. Acknowledge.");
    say(&mut conv, "A second turn to give the timeline some depth.");
    let tl = conv.timeline_id();
    let before_memory = memory(&conv);
    let mut before_kv = Vec::new();
    for i in 0..2u32 {
        before_kv.push(
            conv.turn_kv_digest(tl, i)
                .expect("digest")
                .expect("turn is hot before the demote"),
        );
    }

    let demoted = session
        .engine()
        .demote_timelines_hot(&[tl], true)
        .expect("demote");
    eprintln!("[A4] residences demoted: {demoted}");
    assert!(
        demoted > 0,
        "A4: nothing demoted — the scenario did not exercise the migration path"
    );
    assert_eq!(
        memory(&conv),
        before_memory,
        "A4: demoting K/V to warm moved the conversation's MEMORY — migration \
         touched state it has no business near"
    );

    // The next turn's projection elevates the warm copies back to hot.
    say(&mut conv, "A third turn, attending the migrated history.");
    assert!(
        !memory_is_empty(&conv),
        "A4: the conversation lost its memory across a tier migration"
    );
    for (i, before) in before_kv.iter().enumerate() {
        let after = conv
            .turn_kv_digest(tl, i as u32)
            .expect("digest")
            .expect("turn is hot again after being attended");
        assert_eq!(
            after, *before,
            "A4: turn {i}'s K/V content changed across a hot→warm→hot round \
             trip — the migration altered what the conversation attends"
        );
    }
    let (recalled, reply) = probe_recall(
        &mut conv,
        "What is the migration codeword? One word.",
        "glacier",
    );
    eprintln!("[A4] post-migration recall: {}", reply.trim());
    assert!(
        recalled,
        "A4: the conversation lost its codeword across a tier migration"
    );
}

/// **A5 — a compaction is invisible.**
///
/// Compact the redo log (rewriting it to live records only), reload the
/// substrate from the compacted log in place, and continue. Memory must not
/// move and the conversation must still know its history: compaction is a
/// storage decision about *dead* bytes, and the only way it can touch live
/// state is by dropping a record it wrongly judged dead.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn a5_compaction_is_invisible() {
    let session = shared_session();
    scenario("A5", "log compaction plus reload is invisible");
    let mut conv = session.start();
    say(&mut conv, "The compaction codeword is BASALT. Acknowledge.");
    say(&mut conv, "Another turn so the log has something to keep.");
    let before = memory(&conv);

    session
        .engine()
        .compact_substrate(None)
        .expect("compact the redo log");
    session.engine().reload_substrate();

    assert_eq!(
        memory(&conv),
        before,
        "A5: compacting the log (and reloading from it) changed a live \
         conversation's memory"
    );
    let (recalled, reply) = probe_recall(
        &mut conv,
        "What is the compaction codeword? One word.",
        "basalt",
    );
    eprintln!("[A5] post-compaction recall: {}", reply.trim());
    assert!(
        recalled,
        "A5: the conversation lost its codeword across a compaction — a live \
         record was judged dead"
    );
}

/// **C2 — freeing a conversation releases its memory.**
///
/// Slot ids are recycled pool indices, so memory that outlives its
/// conversation is not merely wasted VRAM: the next conversation on that id
/// inherits a stranger's memory, fluently. `live_memory_count` is the leak
/// gauge; a freed conversation must decrement it.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn c2_freeing_a_conversation_releases_its_memory() {
    let session = shared_session();
    scenario("C2", "freeing a conversation releases its memory");
    let mut conv = session.start();
    say(&mut conv, "One turn, so the slot holds real memory.");
    let with = session.engine().live_memory_count();
    drop(conv);
    // FreeSequence is fire-and-forget; give the scheduler a moment.
    let mut after = with;
    for _ in 0..100 {
        after = session.engine().live_memory_count();
        if after < with {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    eprintln!("[C2] live memory count: {with} -> {after}");
    assert!(
        after < with,
        "C2: dropping a conversation did not release its recurrent memory — \
         the recycled slot id will hand it to a stranger"
    );
}

/// **C5 — a conversation freed while another is mid-turn does not disturb it.**
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn c5_freeing_mid_turn_does_not_disturb_others() {
    let session = shared_session();
    scenario(
        "C5",
        "freeing a conversation mid-turn leaves others undisturbed",
    );
    let mut bystander = session.start();
    let doomed = {
        let mut doomed = session.start();
        say(
            &mut doomed,
            "A turn so the doomed conversation holds state.",
        );
        doomed
    };
    let handle = bystander
        .submit_turn("Reply with one short sentence while another conversation dies.")
        .expect("submit");
    drop(doomed); // frees its slot while the bystander decodes
    let resp = handle.wait().expect("the bystander's turn must complete");
    bystander.finish_turn(handle, &resp).expect("finish");
    assert!(
        !resp.text.trim().is_empty(),
        "C5: the bystander's turn produced nothing after a concurrent free"
    );
    assert!(
        !memory_is_empty(&bystander),
        "C5: the bystander lost its memory when an unrelated conversation was \
         freed"
    );
}

/// **A7 — recall survives a compressed span, and the semantics is decided.**
///
/// The summariser replaces spans of older turns with summary nodes in the
/// projection, while the conversation's memory accumulated over the
/// *originals*. The catalogue flagged the intended semantics as an open
/// decision; this scenario decides it **by measurement**: state a fact, run
/// the conversation deep enough that the fact's span is summarised, verify the
/// projection actually shows a summary in place of the original turn (the
/// opening event's materialized pieces carry each turn's kind), then ask for
/// the fact.
///
/// Decision, recorded in the catalogue: memory being a **superset** of the
/// compressed K/V is the designed behaviour — it is the recurrent layers doing
/// exactly their job, carrying what the attention window no longer shows. The
/// oracle is that recall *works* across the seam.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn a7_recall_survives_a_compressed_span() {
    let session = shared_session();
    scenario("A7", "recall survives a span the projection has compressed");
    let mut conv = session.start();
    let tl = conv.timeline_id();
    say(
        &mut conv,
        "The archive codeword is OBSIDIAN. This matters later: remember the \
         word OBSIDIAN exactly. Acknowledge in one word.",
    );
    // Drive depth PAST the raw tail — the summariser keeps the newest
    // `RAW_TAIL_TURNS` (8) turns verbatim and only absorbs what falls behind
    // it — and probe at the EARLIEST compressed moment: the claim under test
    // is that recall survives *compression*, not arbitrary depth, and a fact
    // pushed ever deeper competes with everything after it in an O(1) state.
    // (At 15 turns the recall sat near threshold and flipped between runs.)
    let mut leaves = Vec::new();
    for i in 0..14 {
        say(
            &mut conv,
            &format!("Filler {i}: reply with one short sentence."),
        );
        leaves = session
            .engine()
            .conversation()
            .read()
            .summary_leaves_chrono(tl);
        if !leaves.is_empty() {
            break;
        }
    }
    if leaves.is_empty() {
        // The last filler may have outrun the summariser's in-flight probes —
        // give it a settling window with the foreground idle.
        for _ in 0..240 {
            leaves = session
                .engine()
                .conversation()
                .read()
                .summary_leaves_chrono(tl);
            if !leaves.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }
    eprintln!("[A7] summary leaves: {leaves:?}");
    assert!(
        !leaves.is_empty(),
        "A7: fifteen turns (seven past the raw tail) and a two-minute settling \
         window, and the summariser never produced a summary leaf — the \
         compression path this scenario exists to cross never engaged"
    );
    // The recall itself is REPORTED here and MEASURED in the cruise
    // (`recall_quality.rs`, deep arm), not gated on a single shot: recall of
    // an ~11-turn-old fact across a compressed span sits near threshold on
    // this model (observed 1-in-3 across runs), and the catalogue's own rule
    // for text assertions is "a single sample is an anecdote — fail on the
    // margin". What IS gated deterministically: compression engaged, and the
    // conversation still operates with its memory intact.
    let (recalled, reply) = probe_recall(
        &mut conv,
        "What is the archive codeword? One word.",
        "obsidian",
    );
    eprintln!(
        "[A7] recall across the compressed span (reported, gated in the cruise): \
         recalled={recalled}: {}",
        reply.trim()
    );
    assert!(
        !reply.trim().is_empty(),
        "A7: the conversation stopped producing text after its span compressed"
    );
    assert!(
        !memory_is_empty(&conv),
        "A7: the conversation's memory emptied when its span compressed"
    );
}

/// **H1/H3 — the cost numbers, taken rather than assumed.**
///
/// Prints the per-turn recurrent-state costs accumulated over this whole
/// suite: seal-export wall time and bytes, and the per-turn state forks. The
/// only assertion is the one that would change a decision (P8): the seal
/// export must stay a small fraction of turn wall time — measured at 4–5%
/// when P8 was closed, gated loosely here so only a regression that would
/// reopen P8 fails.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn h_cost_measurements() {
    let session = shared_session();
    scenario("H1/H3", "recurrent-state cost measurements");
    let (count, export_us, bytes, forks) = candle_conversation::recurrent_state_cost();
    let (computed, installed) = candle_conversation::branch_checkpoint_counts();
    eprintln!(
        "[H] seal exports: {count} ({} ms total, {:.2} MiB avg) | state forks: {forks} | \
         branch checkpoints: {computed} computed, {installed} installed",
        export_us / 1000,
        if count > 0 {
            bytes as f64 / count as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        },
    );
    let _ = session;
    if count > 0 {
        let avg_ms = export_us as f64 / 1000.0 / count as f64;
        assert!(
            avg_ms < 500.0,
            "H1: seal export averages {avg_ms:.1} ms — an order of magnitude over \
             the 40 ms measured when P8 (async staging) was closed as unneeded; \
             that decision needs revisiting"
        );
    }
}

/// **B8 (instrument) — a fresh slot continuing a conversation decodes what the
/// original slot would have.**
///
/// The A1 restart divergence isolated to: state restored byte-perfect, layout
/// block-identical, replies still differ at the first resumed decode. What a
/// restart *also* changes is the SLOT — a fresh slot has no cached glue
/// islands, no pending-user capture, and no carried belief. `fork_resuming`
/// onto the conversation's OWN timeline reproduces exactly that on one live
/// engine: same corpus, same substrate, no reload — only the slot is new.
///
/// - Fork's continuation == original's twin ⇒ the fresh slot is innocent, and
///   A1's divergence needs the restart specifically (recovered-substrate
///   materialisation).
/// - Fork's continuation != twin ⇒ A1's divergence reproduces WITHOUT a
///   restart, and the cause lives in what a fresh slot does differently —
///   glue-island recompute in one wave vs accumulated cache being the prime
///   suspect.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn b8_fresh_slot_continuation_matches() {
    let session = shared_session();
    scenario(
        "B8",
        "a fresh slot continuing a conversation matches the original (instrument)",
    );
    const TURNS: usize = 4;
    let prompt = |i: usize| format!("turn {i}: reply with one short sentence.");

    // The uninterrupted twin.
    let mut full = session.start();
    let mut full_reply = Vec::with_capacity(TURNS);
    for i in 0..TURNS {
        full_reply.push(say(&mut full, &prompt(i)));
    }
    let full_digest = memory(&full);

    // The same first half on a second conversation, then continue it on a
    // FRESH SLOT via a same-timeline fork. (The fork takes the parent's live
    // memory D2D — B4's contract — so state is not the variable; the slot is.)
    let mut parent = session.start();
    for (i, expected) in full_reply.iter().enumerate().take(TURNS / 2) {
        let reply = say(&mut parent, &prompt(i));
        assert_eq!(
            &reply, expected,
            "B8 precondition: siblings must agree before the fork (they did in \
             every A1 run)"
        );
    }
    let mut fork = parent
        .fork_resuming(parent.timeline_id())
        .expect("same-timeline fork");
    for (i, expected) in full_reply.iter().enumerate().skip(TURNS / 2) {
        let reply = say(&mut fork, &prompt(i));
        eprintln!(
            "[B8] fork turn {i}: reply {} full",
            if &reply == expected { "==" } else { "!=" }
        );
        if &reply != expected {
            eprintln!("[B8]   full: {:?}", expected.trim());
            eprintln!("[B8]   fork: {:?}", reply.trim());
        }
    }
    eprintln!(
        "[B8] final memory: fork {} full",
        if memory(&fork) == full_digest {
            "=="
        } else {
            "!="
        }
    );
}

/// **C6 (instrument) — do identical content-rich conversations diverge as the
/// corpus grows?**
///
/// The restart suite found two conversations given identical prompts sealing
/// different memory on one live engine — with *content-rich* prompts, where the
/// contentless "turn N" prompts of the A1 instrument matched bit-for-bit. The
/// hypothesis: provenance/selection scores read workspace-global state
/// (galleries, summaries), so the second conversation projects against a corpus
/// the first's turns had already grown — designed behaviour for retrieval, fatal
/// for any oracle that assumes identical prompts imply identical text.
///
/// Three identical conversations run back-to-back, replies printed per turn:
/// - all agree → the topic divergence needs another explanation;
/// - A differs from B ≡ C → the influence saturates (one crossing);
/// - all differ → cumulative corpus sensitivity.
///
/// This scenario never fails the suite — it is the measurement that decides
/// whether A1/A8 must be re-scoped to isolated workspaces. It prints; the
/// decision belongs in the catalogue, not in an assert.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn c6_content_rich_siblings_diverge_or_not() {
    let session = shared_session();
    scenario(
        "C6",
        "identical content-rich conversations, growing corpus (instrument)",
    );
    const PROMPTS: [&str; 2] = [
        "Let's talk about lighthouses. Acknowledge.",
        "They are tall. Acknowledge.",
    ];
    let mut replies: Vec<Vec<String>> = Vec::new();
    let mut digests = Vec::new();
    for c in 0..3 {
        let mut conv = session.start();
        let mut r = Vec::new();
        for (t, p) in PROMPTS.iter().enumerate() {
            let reply = say(&mut conv, p);
            if let Some(first) = replies.first() {
                eprintln!(
                    "[C6] conv {c} turn {t}: reply {} conv 0",
                    if reply == first[t] { "==" } else { "!=" }
                );
            }
            r.push(reply);
        }
        digests.push(memory(&conv));
        replies.push(r);
    }
    eprintln!(
        "[C6] memory digests: conv0==conv1: {}, conv1==conv2: {}",
        digests[0] == digests[1],
        digests[1] == digests[2],
    );
    for (c, r) in replies.iter().enumerate() {
        for (t, reply) in r.iter().enumerate() {
            if replies[0][t] != *reply {
                eprintln!("[C6] conv {c} turn {t} FULL: {:?}", reply.trim());
            }
        }
    }
}

/// **D1 — a turn's memory record describes the memory the conversation holds at
/// that turn boundary.**
///
/// The invariant every resume rests on, and it had never been asserted. A record
/// is only useful if it describes the same token stream as the K/V sealed beside
/// it; taken at different moments, every resume lands slightly off — memory
/// describing one history, attention holding another.
///
/// It also settled a suspicion by measurement rather than by reading code: that
/// the `<think>` clean re-prefill advances the conversation *after* the export,
/// leaving the record stale by one re-prefill. It does not. What did exist was a
/// harness defect — polling for a record's existence rather than for the record
/// of *this* turn — which made a correct seal look like a divergence.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn d1_record_matches_live_memory() {
    let session = shared_session();
    scenario(
        "D1",
        "the sealed record matches memory at the turn boundary",
    );
    let mut conv = session.start();
    let timeline = conv.timeline_id();

    for turn in 0..3u32 {
        say(
            &mut conv,
            &format!("Turn {turn}: reply in one short sentence."),
        );
        let live = memory(&conv);
        let sealed = sealed_memory_at(session.engine(), timeline, turn);
        let sealed_digest = digest_of_layers(
            sealed
                .layers
                .iter()
                .map(|l| (l.layer_index, l.state.as_slice(), l.conv_tail.as_slice())),
        );
        assert_eq!(
            sealed.layers.len(),
            RECURRENT_LAYERS,
            "D1: the record for turn {turn} carries {} layers, not {RECURRENT_LAYERS} — a \
             partial record is refused on resume, so this presents as an unresumable \
             conversation rather than as a bad write",
            sealed.layers.len(),
        );
        assert_eq!(
            live, sealed_digest,
            "D1: at turn {turn} the memory record does not describe the memory the \
             conversation holds. A resume would install it beside K/V sealed from a \
             different moment, so every resumed conversation starts fractionally \
             wrong — fluently."
        );
    }
}

/// **A3 — a reprojection does not disturb memory.**
///
/// Reprojection rebuilds what the model *attends to*; it must not rebuild what
/// the conversation *remembers*. Memory follows the token stream the
/// conversation actually saw, in the order it saw it — not the reprojected
/// order. Easy to get wrong in the quiet direction: a reprojection that reset
/// memory would leave every answer fluent.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn a3_reprojection_does_not_disturb_memory() {
    let session = shared_session();
    scenario("A3", "a reprojection does not disturb memory");
    const TURNS: usize = 4;

    let mut plain = session.start();
    say_n(&mut plain, TURNS, "turn");
    let without = memory(&plain);

    let mut reprojected = session.start();
    say_n(&mut reprojected, TURNS / 2, "turn");
    // Handing the schema back and setting it again asks for a reprojection
    // without changing what is projected.
    let same_schema = reprojected.projection();
    reprojected.set_projection(same_schema);
    for i in (TURNS / 2)..TURNS {
        say(
            &mut reprojected,
            &format!("turn {i}: reply with one short sentence."),
        );
    }

    assert_eq!(
        without,
        memory(&reprojected),
        "A3: a reprojection changed what the conversation remembers. Memory must \
         follow the tokens the conversation saw, not the order a later projection \
         chose to show them in."
    );
}

/// **A6 — an in-place substrate reload does not disturb memory.**
///
/// `reload_substrate` rebuilds every index from the redo log without dropping
/// the engine — a different path from a restart, and one that now also has to
/// rebuild the branch-checkpoint index rather than merely refresh it.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn a6_in_place_reload_does_not_disturb_memory() {
    let session = shared_session();
    scenario("A6", "an in-place substrate reload does not disturb memory");
    let mut conv = session.start();
    say_n(&mut conv, 3, "turn");
    let before = memory(&conv);
    sealed_memory(session.engine(), conv.timeline_id());

    session.engine().reload_substrate();

    assert_eq!(
        memory(&conv),
        before,
        "A6: reloading the substrate in place changed a live conversation's memory"
    );
    say(&mut conv, "one more turn.");
    assert!(
        !memory_is_empty(&conv),
        "A6: the conversation lost its memory across a reload and kept talking"
    );
}

/// **B1 — memory matches history, in both directions.**
///
/// A conversation's memory must describe exactly the history it holds. Two ways
/// to violate that: **K/V without memory** (the defect this area exists to
/// remove) and **memory without K/V** (its mirror, equally quiet — the model
/// recollects turns its attention layers never saw).
///
/// `fork()` mints a fresh timeline, so the child holds no dialogue history. The
/// first version of this test discovered that the hard way: asked about a
/// codeword from one turn earlier, the child reasoned about the *system prompt*,
/// because that was its entire context. Handing such a child the parent's live
/// memory produced the mirror defect on a production path — `fork_scope` shares
/// this code. A fresh-timeline fork now starts clean in both senses.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn b1_fresh_fork_has_neither_history_nor_memory() {
    let session = shared_session();
    scenario(
        "B1",
        "a fresh-timeline fork inherits neither history nor memory",
    );
    let mut parent = session.start();
    say(
        &mut parent,
        "The codeword is ALBATROSS. Acknowledge in one word.",
    );

    let at_fork = memory(&parent);
    let mut child = parent.fork().expect("fork");

    assert_ne!(
        memory_of(&child),
        Some(at_fork),
        "B1: a fork onto a fresh timeline took its parent's memory while holding \
         none of its history — it now recollects turns its attention layers never \
         saw, which is `state without K/V` and reads perfectly"
    );

    let (recalled, reply) =
        probe_recall(&mut child, "What was the codeword? One word.", "albatross");
    eprintln!("[B1] fork on the parent's turn: {}", reply.trim());
    assert!(
        !recalled,
        "B1: the fork answered from a turn it does not hold"
    );
    assert_eq!(
        memory(&parent),
        at_fork,
        "B1: taking a fork changed the parent's memory"
    );
}

/// **B2/B3 — a fork and its parent are independent in both directions.**
///
/// Asserted as a pair on purpose. Either alone passes under a naive
/// implementation: "always share" satisfies neither, but "copy once and never
/// check" satisfies B2 while failing B3.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn b2_b3_fork_and_parent_are_independent() {
    let session = shared_session();
    scenario("B2/B3", "a fork and its parent do not disturb each other");
    let mut parent = session.start();
    say(&mut parent, "Start here. Acknowledge.");

    let at_fork = memory(&parent);
    let mut child = parent.fork().expect("fork");

    say(&mut child, "A turn only the child sees.");
    assert_eq!(
        memory(&parent),
        at_fork,
        "B2: a turn on the fork moved the parent's memory — they share a buffer"
    );
    let child_after = memory(&child);
    assert_ne!(
        Some(child_after),
        Some(at_fork),
        "B2: the child's own turn did not land"
    );

    say(&mut parent, "A turn only the parent sees.");
    assert_eq!(
        memory(&child),
        child_after,
        "B3: a turn on the parent moved the fork's memory"
    );
    assert_ne!(
        memory(&parent),
        at_fork,
        "B3: the parent's own turn did not land"
    );
}

/// **B4 — a fork inherits the parent as of the fork point, not as of now.**
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn b4_fork_inherits_as_of_the_fork_point() {
    let session = shared_session();
    scenario("B4", "a fork inherits the parent as of the fork point");
    let mut parent = session.start();
    say(&mut parent, "First. Acknowledge.");
    let early = parent
        .fork_resuming(parent.timeline_id())
        .expect("fork early");
    let at_early = memory(&early);

    say(&mut parent, "Second. Acknowledge.");
    let late = parent
        .fork_resuming(parent.timeline_id())
        .expect("fork late");

    assert_ne!(
        memory(&late),
        at_early,
        "B4: a fork taken after another turn inherited the same memory as one taken \
         before it — it reads a stale record rather than the live state"
    );
    assert_eq!(
        memory(&early),
        at_early,
        "B4: the earlier fork's memory moved when the parent took another turn"
    );
}

/// **B5 — forking mid-turn is refused.**
///
/// A fork whose memory ran ahead of its K/V is unrepairable — nothing downstream
/// can detect it and no later operation can fix it — so the refusal is the
/// feature, not a limitation.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn b5_forking_mid_turn_is_refused() {
    let session = shared_session();
    scenario("B5", "forking mid-turn is refused");
    let mut conv = session.start();
    let handle = conv
        .submit_turn("A turn we do not wait for.")
        .expect("submit");
    // `Sequence` is not `Debug`, so unwrap by hand rather than via `expect_err`.
    let msg = match conv.fork() {
        Ok(_) => panic!(
            "B5: a fork with a turn in flight was allowed — its memory is ahead of \
             its K/V, and nothing downstream can detect that"
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("turn in flight"),
        "B5: the refusal must name the reason: {msg}"
    );
    let _ = handle.wait();
}

/// **B7 — a scope splice advances the parent's memory over the adopted turns.**
///
/// The `code_read` merge: a scope fork ingests against the system prompt, and
/// its sealed turn pair is adopted onto the file conversation **by reference** —
/// K/V the parent's forward never saw. The decided semantics is (c): the parent
/// re-prefills the adopted tokens through its own recurrence, which it can do
/// because they land at the tail. Three observable consequences, asserted here:
///
/// 1. The parent's memory **moves** — a splice that leaves it at the fork point
///    is K/V-without-state over the adopted span, §1's defect produced by the
///    merge instead of a restart.
/// 2. The parent can **recall the injected content**. This is the semantic
///    proof that the catch-up produced the RIGHT state, not merely a different
///    one — and on this architecture it is load-bearing, not decorative: K/V
///    alone reads as forgotten (the D5 measurement), so if the catch-up state
///    were wrong, memory would move (consequence 1 still passes) while the
///    conversation stays blind to everything it was handed.
/// 3. The catch-up is a **deterministic function of the adopted stream** — two
///    identical parent+scope+splice runs end bit-identical. Without this the
///    splice would make sibling determinism (which A3 and C-family rest on)
///    unrecoverable for every `code_read` conversation.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn b7_scope_splice_catches_memory_up() {
    let session = shared_session();
    scenario(
        "B7",
        "a scope splice advances the parent's memory over the adopted turns",
    );
    // The production round-trip, minus the parallelism: turn_sink's
    // `ingest_scopes` forks, runs `ingest_scope_roundtrip_indices` on the fork,
    // and splices the coupled pair back in scope order.
    let run_once = || {
        let mut parent = session.start();
        say(&mut parent, "Turn zero. Acknowledge in one word.");
        let at_fork = memory(&parent);

        let mut fork = parent.fork_scope().expect("fork_scope");
        let (call_idx, resp_idx, _tokens) = fork
            .ingest_scope_roundtrip_indices(
                "Read the file src/lantern.rs.",
                "fn glow() -> Lumens { Lumens(17) }",
                "Summarise what this file does in one sentence.",
                vec!["scope".into()],
                64,
            )
            .expect("scope round-trip");
        parent
            .splice_scope_turns(fork.timeline_id(), call_idx, resp_idx, vec!["scope".into()])
            .expect("splice_scope_turns");
        let after = memory(&parent);
        (at_fork, after, parent)
    };

    let (at_fork, after_first, mut parent) = run_once();
    assert_ne!(
        after_first, at_fork,
        "B7: the splice left the parent's memory at the fork point — it now holds \
         the scope's K/V with a recurrence that never saw it, and reads perfectly"
    );

    // The injected content is USABLE, not merely present: the parent never
    // decoded these turns — they arrived from another conversation's timeline
    // by reference — and it must now answer from them.
    let (recalled, reply) = probe_recall(
        &mut parent,
        "In src/lantern.rs, what number does the glow function return? Reply \
         with just the number.",
        "17",
    );
    eprintln!("[B7] recall of spliced content: {}", reply.trim());
    assert!(
        recalled,
        "B7: the parent cannot recall content spliced into its history — the \
         catch-up advanced its memory without covering what was adopted, and \
         every code_read conversation is blind to the files it just read"
    );
    drop(parent);

    let (_, after_second, _) = run_once();
    assert_eq!(
        after_first, after_second,
        "B7: two identical scope splices produced different parent memory — the \
         catch-up is not a function of the adopted stream"
    );
}

/// **C1 — conversations sharing an engine keep their own memory.**
///
/// Several conversations share one engine, one wave and one slot pool. Each must
/// remember only its own facts. Cross-contamination here is the recycled-slot
/// defect's live cousin, and it reads perfectly.
#[test]
#[ignore = "Tier 3: shares the process-wide engine (one 22 GB load per test binary); see the \
            file header for the run command"]
fn c1_conversations_keep_their_own_memory() {
    let session = shared_session();
    scenario("C1", "interleaved conversations keep their own memory");
    let facts = ["vermilion", "kestrel", "harbour"];
    let mut convs: Vec<_> = (0..facts.len()).map(|_| session.start()).collect();

    for (conv, fact) in convs.iter_mut().zip(facts) {
        say(conv, &format!("Remember the word {fact}. Acknowledge."));
    }
    for round in 0..2 {
        for conv in convs.iter_mut() {
            say(conv, &format!("Filler {round}. One short sentence."));
        }
    }

    let digests: Vec<_> = convs.iter().map(memory).collect();
    for i in 0..digests.len() {
        for j in (i + 1)..digests.len() {
            assert_ne!(
                digests[i], digests[j],
                "C1: conversations {i} and {j} hold identical memory after seeing \
                 different turns — they are sharing state"
            );
        }
    }
    for (i, (conv, fact)) in convs.iter_mut().zip(facts).enumerate() {
        let (recalled, reply) =
            probe_recall(conv, "What word did I ask you to remember? One word.", fact);
        eprintln!("[C1] conversation {i} ({fact}) -> {}", reply.trim());
        assert!(
            recalled,
            "C1: conversation {i} could not recall its own word ({fact}) while \
             sharing an engine with {} others",
            facts.len() - 1
        );
    }
}

// ═══ Stop and start ═════════════════════════════════════════════════════════

/// Everything that needs the daemon to stop and come back.
///
/// All four scenarios share **one** stop/start cycle: each sets up its timelines
/// on the first engine, the engine is dropped, and each checks its invariant on
/// the second. Two loads instead of eight.
#[test]
#[ignore = "Tier 3: loads the pinned Qwen3.6-35B-A3B GGUF (22 GB) twice and runs four restart \
            scenarios (~14 min). Run with: cargo test -p zend --release --features cuda --test \
            memory_continuity -- --ignored --nocapture --test-threads=1"]
fn restart_invariants() {
    // This test owns its engines (the drop and reopen IS the subject), so it
    // takes the engine slot exclusively: the shared engine is dropped and no
    // scenario can reload it beside ours — one engine resident at a time,
    // which is what the 16 GB machine requires.
    let _slot = exclusive_engine_slot();
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::new();

    // ── Before the restart ──────────────────────────────────────────────────
    const TURNS: usize = 6;
    let a1_prompt = |i: usize| format!("turn {i}: reply with one short sentence.");
    let (a1, half_tl, topic_tl, codeword_tl, reference_tl) = {
        let session = ws.session(&device);

        scenario("A1", "setting up: the same turns, uninterrupted");
        // Per-turn digests and replies, so a failure names the first turn that
        // diverged instead of only the last digest that differed.
        let mut straight = session.start();
        let mut straight_at = Vec::with_capacity(TURNS);
        let mut straight_reply = Vec::with_capacity(TURNS);
        let mut straight_blocks = Vec::with_capacity(TURNS);
        let mut straight_sel = Vec::with_capacity(TURNS);
        for i in 0..TURNS {
            let (reply, sel) = say_opening(&mut straight, &a1_prompt(i));
            straight_reply.push(reply);
            straight_at.push(memory(&straight));
            straight_blocks.push(straight.sealed_block_count());
            straight_sel.push(sel);
        }
        eprintln!("[A1] straight turn 0 opening: {}", straight_sel[0]);

        scenario("A1", "setting up: the first half of an interrupted run");
        let mut half = session.start();
        for i in 0..TURNS / 2 {
            let reply = say(&mut half, &a1_prompt(i));
            // Sibling determinism before any restart is involved: two live
            // conversations fed the same turns on one engine must agree
            // turn-for-turn (A3 already relies on this).
            eprintln!(
                "[A1] pre-restart turn {i}: digest {} straight, reply {} straight, \
                 blocks {}/{}",
                if memory(&half) == straight_at[i] {
                    "=="
                } else {
                    "!="
                },
                if reply == straight_reply[i] {
                    "=="
                } else {
                    "!="
                },
                half.sealed_block_count(),
                straight_blocks[i],
            );
        }
        let half_tl = half.timeline_id();
        let sealed = sealed_memory_at(session.engine(), half_tl, (TURNS / 2 - 1) as u32);
        let sealed_digest = digest_of_layers(
            sealed
                .layers
                .iter()
                .map(|l| (l.layer_index, l.state.as_slice(), l.conv_tail.as_slice())),
        );
        eprintln!(
            "[A1] half's sealed record == its live memory at the boundary: {}",
            sealed_digest == memory(&half)
        );
        // The K/V content each side actually holds hot, per turn — formats,
        // palettes, scales, and raw arena bytes. Pre-restart, half's turn i and
        // straight's turn i carry identical tokens at identical positions, so
        // their sealed K/V content should be identical bytes; post-restart the
        // same digests say whether the recovered materialisation serves those
        // bytes back faithfully.
        let mut half_kv = Vec::with_capacity(TURNS / 2);
        for i in 0..TURNS / 2 {
            let s = straight
                .turn_kv_digest(straight.timeline_id(), i as u32)
                .expect("digest straight")
                .expect("straight turn is hot");
            let h = half
                .turn_kv_digest(half_tl, i as u32)
                .expect("digest half")
                .expect("half turn is hot");
            eprintln!(
                "[A1] pre-restart turn {i} sealed K/V: half {} straight",
                if h == s { "==" } else { "!=" }
            );
            half_kv.push(h);
        }

        scenario("A8", "setting up: two identical conversations with a topic");
        // TWO timelines built identically, because resuming ONE timeline twice
        // is not symmetric: the first resume's probe turn lands on the shared
        // timeline, so the second resume would see a longer history. The two
        // builds usually seal identical memory but are not guaranteed to:
        // content-rich replies sit near argmax ties, and near-ties flip under
        // engine-instance numerics (the instrument below prints which happened).
        // A8's assertion is therefore semantic — both resumes recall the topic
        // — not byte equality of the seals.
        let mut topic_a = session.start();
        let mut topic_b = session.start();
        let mut topic_replies: Vec<Vec<String>> = Vec::new();
        for topic in [&mut topic_a, &mut topic_b] {
            topic_replies.push(vec![
                say(topic, "Let's talk about lighthouses. Acknowledge."),
                say(topic, "They are tall. Acknowledge."),
            ]);
        }
        // The C6 instrument proved three of these conversations run
        // SEQUENTIALLY (each dropped before the next starts) agree bit-for-bit
        // — so if these two, which are ALIVE SIMULTANEOUSLY on different
        // slots, disagree, the discriminating factor is slot coexistence, and
        // whether the replies or only the seals differ says whether it is the
        // decode or the bookkeeping that depends on it.
        for (t, (ra, rb)) in topic_replies[0]
            .iter()
            .zip(topic_replies[1].iter())
            .enumerate()
        {
            eprintln!(
                "[A8] topic turn {t}: reply b {} a",
                if ra == rb { "==" } else { "!=" }
            );
            if ra != rb {
                eprintln!("[A8]   a: {:?}", ra.trim());
                eprintln!("[A8]   b: {:?}", rb.trim());
            }
        }
        let topic_a_tl = topic_a.timeline_id();
        let topic_b_tl = topic_b.timeline_id();
        let sealed_a = sealed_memory_at(session.engine(), topic_a_tl, 1);
        let sealed_b = sealed_memory_at(session.engine(), topic_b_tl, 1);
        // Layer-by-layer, state and tail separately: "tail differs only" and
        // "state diverges from layer k" name different defects.
        let mut seals_agree = true;
        for (la, lb) in sealed_a.layers.iter().zip(sealed_b.layers.iter()) {
            let state_eq = la.state == lb.state;
            let tail_eq = la.conv_tail == lb.conv_tail;
            if !(state_eq && tail_eq) {
                seals_agree = false;
                eprintln!(
                    "[A8]   layer {}: state {} tail {}",
                    la.layer_index,
                    if state_eq { "==" } else { "!=" },
                    if tail_eq { "==" } else { "!=" },
                );
            }
        }
        eprintln!("[A8] the two topic conversations sealed identical memory: {seals_agree}");

        scenario("B1a", "setting up: a conversation with a codeword");
        let mut codeword = session.start();
        say(
            &mut codeword,
            "The codeword is ALBATROSS. Acknowledge in one word.",
        );
        let codeword_tl = codeword.timeline_id();
        sealed_memory_at(session.engine(), codeword_tl, 0);

        scenario("B6", "setting up: a conversation with a reference number");
        let mut reference = session.start();
        say(
            &mut reference,
            "The reference number is SEVENTEEN. Acknowledge.",
        );
        let reference_tl = reference.timeline_id();
        sealed_memory_at(session.engine(), reference_tl, 0);

        scenario(
            "D4",
            "setting up: a conversation whose record will be poisoned",
        );
        let mut amnesia = session.start();
        say(&mut amnesia, "The secret is MOONSTONE. Acknowledge.");
        let amnesia_tl = amnesia.timeline_id();
        sealed_memory_at(session.engine(), amnesia_tl, 0);
        // The amnesia control: supersede the record with one carrying a schedule
        // hash the model does not report. `import` validates it before touching
        // a tensor, so the resume below MUST refuse it and come up empty — the
        // production path for a record from a different model geometry.
        poison_memory_record(session.engine(), amnesia_tl);

        scenario(
            "D5",
            "setting up: a conversation whose record will be corrupted",
        );
        let mut corrupt = session.start();
        say(&mut corrupt, "The corrupt codeword is FLINT. Acknowledge.");
        let corrupt_tl = corrupt.timeline_id();
        sealed_memory_at(session.engine(), corrupt_tl, 0);
        // Supersede the record with garbage BYTES (not a wrong hash — undecodable
        // input). The reload must survive it and only this resume comes up empty.
        session
            .engine()
            .conversation()
            .enqueue_recurrent_snapshot(corrupt_tl, vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01]);
        std::thread::sleep(std::time::Duration::from_millis(300));

        scenario("D6", "setting up: a conversation to distill");
        let mut distilled = session.start();
        say(
            &mut distilled,
            "The distilled codeword is EMBER. Acknowledge.",
        );
        let distilled_tl = distilled.timeline_id();
        sealed_memory_at(session.engine(), distilled_tl, 0);
        session
            .engine()
            .distill_timeline(
                distilled_tl,
                candle_conversation::persistence::record::DistillMode::ProvenanceOnly,
            )
            .expect("mark for distillation");

        scenario("D7", "setting up: a conversation to tombstone");
        let mut doomed = session.start();
        say(&mut doomed, "The tombstoned codeword is ASH. Acknowledge.");
        let doomed_tl = doomed.timeline_id();
        sealed_memory_at(session.engine(), doomed_tl, 0);
        session
            .engine()
            .tombstone_timeline(doomed_tl)
            .expect("tombstone");

        // Compaction applies the distillation (content sheds at compaction, not
        // at the mark) and reclaims the tombstoned timeline — and every LIVE
        // record must survive it, which the post-restart scenarios re-prove.
        session
            .engine()
            .compact_substrate(None)
            .expect("compact before the restart");

        scenario("D2/D3", "setting up: a turn whose Tokens record will tear");
        // The one tear the §4.1 write ordering permits: the snapshot is enqueued
        // BEFORE the turn's Tokens record, so a crash between the two leaves a
        // snapshot for a turn whose records never landed. The fault hook drops
        // exactly that Tokens record — a deterministic stand-in for the crash.
        // Ordered AFTER the compaction: compaction synthesises live records
        // from state that still remembers the turn, so a tear made before it
        // would be quietly healed rather than reaching the reload.
        let mut torn = session.start();
        say(&mut torn, "The torn codeword is SHALE. Acknowledge.");
        let torn_tl = torn.timeline_id();
        sealed_memory_at(session.engine(), torn_tl, 0);
        candle_conversation::persistence::writer::fault::drop_next_tokens(
            candle_conversation::persistence::content_hash::turn_stream_id(torn_tl.raw(), 1),
        );
        say(&mut torn, "This turn's Tokens record is dropped in flight.");
        sealed_memory_at(session.engine(), torn_tl, 1);

        (
            (
                straight_at,
                straight_reply,
                straight_blocks,
                straight_sel,
                sealed_digest,
                half_kv,
            ),
            half_tl,
            (topic_a_tl, topic_b_tl),
            codeword_tl,
            (
                reference_tl,
                amnesia_tl,
                corrupt_tl,
                distilled_tl,
                doomed_tl,
                torn_tl,
            ),
        )
        // Every conversation and the engine drop here — the daemon stops, and
        // the redo log is all that is left.
    };
    let (straight_at, straight_reply, straight_blocks, straight_sel, half_sealed_digest, half_kv) =
        a1;
    let (reference_tl, amnesia_tl, corrupt_tl, distilled_tl, doomed_tl, torn_tl) = reference_tl;

    // ── After the restart ───────────────────────────────────────────────────
    // Scoped so the second engine and every conversation on it drop before C4
    // opens a third engine on a fresh workspace.
    {
        let session = ws.session(&device);

        // **A1 — a restart restores the conversation byte-for-byte, and it
        // continues from there.** The catalogue's headline oracle, restated to what
        // the engine promises after the divergence was chased to ground (see the A1
        // note in `docs/recurrent_state_behavioural_tests.md`): the restored
        // recurrent state, every recovered turn's K/V content, the projected
        // layout, and the opening selection are all BYTE-IDENTICAL to the
        // uninterrupted twin — each asserted below with its own instrument. What is
        // deliberately NOT asserted is token-identical continuation: with every
        // input byte proven equal, the residual reply divergence is argmax flipping
        // on near-ties under engine-instance numerics (allocator and library state
        // — the same class the decode-reproducibility campaigns documented), which
        // no persistence machinery can remove. It is printed, not asserted; the
        // recall gates (B1a, B6) carry the semantic half of the claim.
        scenario(
            "A1",
            "a restart restores the conversation byte-for-byte and it continues",
        );
        let base = session.start();
        let mut resumed = base.fork_resuming(half_tl).expect("resume the half run");
        // The restored state, before any new turn touches it: it must be the sealed
        // record, which the pre-restart instrument proved equals the uninterrupted
        // twin's live memory at the same boundary.
        let restored = memory_of(&resumed).expect("A1: the resume installed no memory at all");
        assert_eq!(
            restored, half_sealed_digest,
            "A1: the restored memory is not the sealed record it came from — the \
         resume installed a different state than the conversation sealed"
        );
        assert_eq!(
            restored,
            straight_at[TURNS / 2 - 1],
            "A1: the restored memory differs from the uninterrupted twin at the same \
         boundary — the two conversations were not equal at the seam"
        );
        for i in (TURNS / 2)..TURNS {
            let (reply, sel) = say_opening(&mut resumed, &a1_prompt(i));
            let d = memory(&resumed);
            eprintln!(
                "[A1] resumed turn {i}: digest {} straight, reply {} straight",
                if d == straight_at[i] { "==" } else { "!=" },
                if reply == straight_reply[i] {
                    "=="
                } else {
                    "!="
                },
            );
            if reply != straight_reply[i] {
                eprintln!("[A1]   straight: {:?}", straight_reply[i].trim());
                eprintln!("[A1]   resumed:  {:?}", reply.trim());
            }
            assert!(
                !reply.trim().is_empty(),
                "A1: the resumed conversation stopped producing text at turn {i}"
            );
            assert_eq!(
                resumed.sealed_block_count(),
                straight_blocks[i],
                "A1: at turn {i} the resumed conversation's layout diverged from the \
             uninterrupted twin — its history was laid out differently, so every \
             position (and RoPE) after the seam is shifted"
            );
            assert_eq!(
                sel, straight_sel[i],
                "A1: at turn {i} the resumed conversation projected a different \
             context than the uninterrupted twin — selection, glue, or prefill \
             differ, so it is not continuing the same conversation"
            );
        }
        // The recovered turns are materialised hot by now (the resumed turns
        // attended them): the K/V content the resumed engine serves must be
        // byte-identical — formats, palettes, scales, and raw arena bytes — to what
        // the live engine held for the same turns before the restart.
        for (i, pre) in half_kv.iter().enumerate() {
            let post = resumed
                .turn_kv_digest(half_tl, i as u32)
                .expect("digest recovered turn")
                .unwrap_or_else(|| {
                    panic!("A1: recovered turn {i} holds no hot sealed K/V after being attended")
                });
            assert_eq!(
                post, *pre,
                "A1: recovered turn {i}'s K/V content changed across the restart — \
             the cold round-trip is not byte-faithful, and the resumed \
             conversation is attending different bytes than it sealed"
            );
        }
        // **A8 — a restarted conversation carries its behaviour.** A1 asserts the
        // restored bytes; this asserts that resuming actually continues the
        // conversation the user was having. Two identically-built timelines rather
        // than one resumed twice: the first resume's probe turn lands on its
        // timeline, so a second resume of the SAME timeline would see a longer
        // history and the comparison would be asymmetric by construction. The
        // recall is the assertion; token-identity between the two resumes is
        // printed as a diagnostic and not asserted, for the same reason as A1's —
        // content-rich replies sit near argmax ties, and near-ties flip under
        // engine-instance numerics that no restored byte can pin.
        scenario(
            "A8",
            "resumes of two identically-built conversations recall their topic",
        );
        let (topic_a_tl, topic_b_tl) = topic_tl;
        const PROBE: &str = "In one word, what have we been discussing?";
        let mut resumed_a = base.fork_resuming(topic_a_tl).expect("resume topic a");
        let (recalled_a, first) = probe_recall(&mut resumed_a, PROBE, "lighthouse");
        let mut resumed_b = base.fork_resuming(topic_b_tl).expect("resume topic b");
        let (recalled_b, second) = probe_recall(&mut resumed_b, PROBE, "lighthouse");
        eprintln!(
            "[A8] two resumes agree token-for-token: {}",
            first == second
        );
        let mut failures: Vec<String> = Vec::new();
        if !(recalled_a && recalled_b) {
            eprintln!("[A8]   first:  {:?}", first.trim());
            eprintln!("[A8]   second: {:?}", second.trim());
            failures.push(format!(
                "A8: a resumed conversation lost its topic (first recalled: \
             {recalled_a}, second: {recalled_b}) — the restart did not continue \
             the conversation the user was having"
            ));
        }

        // **B1a — a fork that continues a conversation carries both.** The fork the
        // daemon actually performs.
        scenario("B1a", "a resuming fork carries history and memory");
        let mut codeword = base.fork_resuming(codeword_tl).expect("resume codeword");
        assert!(
            !memory_is_empty(&codeword),
            "B1a: a fork continuing a real conversation started with empty memory while \
         its K/V holds the whole history"
        );
        let (recalled, reply) = probe_recall(
            &mut codeword,
            "What was the codeword? One word.",
            "albatross",
        );
        eprintln!("[B1a] resuming fork: {}", reply.trim());
        assert!(
            recalled,
            "B1a: the resuming fork could not recall the conversation it is continuing"
        );

        // **B6 — a non-resident conversation can be picked up and continued.** The
        // catalogue asked for something stronger — that this match forking a *live*
        // conversation — and that turned out not to be expressible: `fork()` mints a
        // fresh timeline by design, and `fork_resuming` onto a live conversation's own
        // timeline puts two sequences on one timeline. The only fork that continues a
        // history is this one.
        scenario("B6", "a non-resident conversation can be picked up");
        let mut reference = base.fork_resuming(reference_tl).expect("resume reference");
        assert!(
            !memory_is_empty(&reference),
            "B6: a fork of a non-resident conversation came up with empty memory under a \
         full history"
        );
        let (recalled, reply) = probe_recall(
            &mut reference,
            "What is the reference number? One word.",
            "seventeen",
        );
        eprintln!("[B6] resumed fork: {}", reply.trim());
        assert!(
            recalled,
            "B6: a fork of a non-resident conversation could not recall it — the record \
         path does not continue the same conversation"
        );

        // **D4 — memory sealed under a different schedule is refused, not installed.**
        // The amnesia control, run for real: the record was superseded pre-restart by
        // one whose schedule hash the model does not report. The resume must refuse
        // it — a wrong-geometry install that "mostly works" is the quiet corruption
        // this area exists to remove — and the conversation must come up with its
        // K/V intact and its memory empty: fluent, and having forgotten.
        scenario(
            "D4",
            "a poisoned memory record is refused, leaving memory empty",
        );
        let amnesiac = base.fork_resuming(amnesia_tl).expect("resume amnesiac");
        match memory_of(&amnesiac) {
            None => eprintln!("[D4] refused install left no state at all"),
            Some(_) => assert!(
                memory_is_empty(&amnesiac),
                "D4: a memory record with a foreign schedule hash was INSTALLED — a \
             model-geometry mismatch is being scattered into live state"
            ),
        }

        // **D5 — a corrupt record fails that one resume, not the reload.** The
        // record was superseded pre-restart with undecodable bytes. The reload
        // already survived (this test is running); this conversation must come
        // up with its memory empty — refused with its own distinguishable WARN
        // ("failed to decode", distinct from D4's "hash mismatch") — while its
        // history RECORDS are intact and its K/V still materialises.
        //
        // What is deliberately NOT asserted: recall through that K/V. On this
        // architecture a conversation with K/V and no recurrent memory HAS
        // FORGOTTEN — `docs/deltanet_state_persistence.md` §1's own words:
        // "nothing errors, every shape matches, and the model reads fluently —
        // it has simply forgotten." The first version of this scenario demanded
        // recall through the 10 attention layers alone and measured exactly
        // that design sentence coming true. The forgetting IS the amnesia
        // control the G-family leans on.
        scenario("D5", "a corrupt record fails one resume, not the reload");
        let read = session.engine().conversation();
        assert!(
            read.read().turn_count(corrupt_tl) > 0,
            "D5: the corrupt-record conversation lost its HISTORY too — corruption \
         in one record class took out another's"
        );
        // Record-class inventory for the recovered turn — separates "the turn
        // counted but its content classes died" from "content fine, the
        // materialisation broke". The poisoned twin (valid encoding, wrong
        // hash) is printed beside it: if both lost a class, the culprit is the
        // snapshot supersede path, not the garbage decode.
        {
            use candle_conversation::projection::TurnIndex;
            let r = read.read();
            eprintln!(
                "[D5] recovered turn 0: {} tokens, assistant text {} chars, sealed K/V {}",
                r.token_ids_of(corrupt_tl, TurnIndex(0)).len(),
                r.assistant_text_of(corrupt_tl, TurnIndex(0)).len(),
                if r.turn_sealed_of(corrupt_tl, TurnIndex(0)).is_some() {
                    "present"
                } else {
                    "ABSENT"
                },
            );
            eprintln!(
                "[D4] recovered turn 0 (poisoned twin): {} tokens, sealed K/V {}",
                r.token_ids_of(amnesia_tl, TurnIndex(0)).len(),
                if r.turn_sealed_of(amnesia_tl, TurnIndex(0)).is_some() {
                    "present"
                } else {
                    "ABSENT"
                },
            );
        }
        let mut corrupted = base.fork_resuming(corrupt_tl).expect("resume corrupted");
        match memory_of(&corrupted) {
            None => eprintln!("[D5] corrupt record left no state at all"),
            Some(_) => assert!(
                memory_is_empty(&corrupted),
                "D5: a memory record that does not decode was INSTALLED anyway"
            ),
        }
        let reply = say(&mut corrupted, "What is the corrupt codeword? One word.");
        eprintln!("[D5] reply under refused memory: {}", reply.trim());
        assert!(
            !reply.trim().is_empty(),
            "D5: the conversation stopped producing text — a corrupt memory \
             record must cost the memory, not the conversation"
        );
        // The probe turn's materialisation must still elevate the recovered
        // turn's K/V — the corrupt snapshot is one record class, and it must
        // not take the turn's chunks down with it.
        {
            use candle_conversation::projection::TurnIndex;
            assert!(
                read.read()
                    .turn_sealed_of(corrupt_tl, TurnIndex(0))
                    .is_some(),
                "D5: the recovered turn's K/V never materialised — the corrupt \
                 snapshot took an intact record class down with it"
            );
        }

        // **D6 — a distilled conversation is unresumable.** ProvenanceOnly
        // KEEPS the `StreamDecl` (the turn still counts — it is retrievable by
        // signature) and sheds tokens + K/V at compaction, so what remains is
        // a provenance exemplar, not a conversation. The resume must come up
        // with no CONTENT rather than a half-conversation.
        scenario("D6", "a distilled conversation is unresumable");
        {
            use candle_conversation::projection::TurnIndex;
            let r = read.read();
            let tokens = r.token_ids_of(distilled_tl, TurnIndex(0)).len();
            let text = r.assistant_text_of(distilled_tl, TurnIndex(0)).len();
            eprintln!(
                "[D6] distilled turn 0 after compaction: {} recovered turns, \
                 {tokens} tokens, {text} chars",
                r.turn_count(distilled_tl),
            );
            // TOKENS are the resumability test — no tokens means no replay and
            // no re-prefill. The decl (and the text riding it) is KEPT by the
            // ProvenanceOnly spec: the turn stays retrievable by signature.
            assert_eq!(
                tokens, 0,
                "D6: a ProvenanceOnly-distilled turn still carries its token \
                 stream after compaction — the distillation did not shed, and \
                 the conversation is still resumable"
            );
        }
        let distilled = base.fork_resuming(distilled_tl).expect("fork survives");
        match memory_of(&distilled) {
            None => eprintln!("[D6] distilled resume carries no state"),
            Some(_) => assert!(
                memory_is_empty(&distilled),
                "D6: a distilled conversation resumed WITH memory — state describing \
             turns whose K/V and tokens no longer exist"
            ),
        }

        // **D7 — shedding what a record was taken over makes the record unusable.**
        // The whole timeline was tombstoned pre-restart; its memory record must be
        // refused rather than silently applied to an empty history.
        scenario(
            "D7",
            "a tombstoned conversation's record is unusable, not applied",
        );
        assert_eq!(
            read.read().turn_count(doomed_tl),
            0,
            "D7: a tombstoned timeline still has recoverable turns after compaction"
        );
        let doomed = base.fork_resuming(doomed_tl).expect("fork survives");
        match memory_of(&doomed) {
            None => eprintln!("[D7] tombstoned resume carries no state"),
            Some(_) => assert!(
                memory_is_empty(&doomed),
                "D7: a tombstoned conversation's memory record was applied to an \
             empty history — state without K/V, installed on purpose"
            ),
        }

        // **D2/D3 — the torn seal, for real.** Turn 1's Tokens record was dropped
        // in flight while its snapshot landed — the exact tear the §4.1 write
        // ordering permits. D2: the torn turn must be ABSENT from the recovered
        // history. D3: the too-new snapshot must be REFUSED — installed, it would
        // put memory one turn ahead of the K/V, the mirror of the defect this
        // whole area exists to remove, and just as fluent.
        scenario(
            "D2/D3",
            "a torn seal loses the turn and refuses its snapshot",
        );
        let recovered_turns = read.read().turn_count(torn_tl);
        assert_eq!(
            recovered_turns, 1,
            "D2: the torn turn's Tokens record never reached the log, yet the \
         reload recovered {recovered_turns} turns — an unpersisted turn came back"
        );
        let torn = base.fork_resuming(torn_tl).expect("resume torn");
        match memory_of(&torn) {
            None => eprintln!("[D3] torn snapshot refused; no state installed"),
            Some(_) => assert!(
                memory_is_empty(&torn),
                "D3: the snapshot for the torn turn was INSTALLED — memory is now \
             one turn ahead of the K/V it sits beside"
            ),
        }

        // E6/E7's same-branch half, on this engine: every conversation this
        // suite opened restored or installed the ONE known branch — nothing
        // recomputed it. (The edited-prompt half, E4, needs the edit to land
        // across a restart and runs as its own engine open below — an
        // in-process second builder aliases section ids by construction, which
        // is `set_projection`'s documented contract, not a prompt edit.)
        let (computed_now, installed_now) = candle_conversation::branch_checkpoint_counts();
        eprintln!(
            "[E] branch checkpoints so far: {computed_now} computed, {installed_now} installed"
        );
        assert!(
            installed_now > computed_now,
            "E7: conversations are recomputing the prompt branch instead of \
             restoring it — installs should far outnumber computes"
        );

        assert!(
            failures.is_empty(),
            "restart invariants failed:\n{}",
            failures.join("\n")
        );
    }

    // **C4 — workspaces do not share conversation memory.** A third engine on a
    // FRESH workspace: nothing from this test's workspace may be visible. The
    // codeword stated to `codeword_tl`'s conversation must be unknown here.
    scenario(
        "C4",
        "a fresh workspace knows nothing of another's conversations",
    );
    {
        let ws2 = Workspace::new();
        let session = ws2.session(&device);
        let mut stranger = session.start();
        let reply = say(
            &mut stranger,
            "What was the codeword? Answer with one word, or say you don't know.",
        );
        eprintln!("[C4] stranger's answer: {}", reply.trim());
        assert!(
            !common::recalls(&reply, "albatross"),
            "C4: a conversation in a FRESH workspace recalled another workspace's \
             codeword — state is bleeding across workspaces"
        );
    }

    // **E4/E6/E7 — an edited prompt is a new branch, computed once.** The
    // daemon restarted after a `projection.yaml` edit: the edited persona
    // section is content-addressed, so it seals fresh and the branch prefix
    // hash changes with it. Reusing the old branch's checkpoint here is the
    // catalogue's nightmare case — a model confidently following instructions
    // it no longer has.
    scenario("E4/E6/E7", "an edited prompt computes its own branch, once");
    let session = ws.session_with_edited_prompt(&device);
    let (computed_before, installed_before) = candle_conversation::branch_checkpoint_counts();
    let edited_a = session.start();
    let (computed_mid, installed_mid) = candle_conversation::branch_checkpoint_counts();
    eprintln!(
        "[E4] after the first edited-prompt conversation: computed \
         {computed_before}->{computed_mid}, installed {installed_before}->{installed_mid}; \
         memory installed: {}",
        memory_of(&edited_a).is_some(),
    );
    assert!(
        computed_mid > computed_before,
        "E4: a conversation opened after a prompt EDIT reused the old branch's \
         checkpoint — the model is confidently following instructions it no \
         longer has"
    );
    let edited_b = session.start();
    let (computed_after, installed_after) = candle_conversation::branch_checkpoint_counts();
    assert_eq!(
        computed_after, computed_mid,
        "E7: a second conversation on the SAME edited prompt recomputed the \
         branch instead of restoring it"
    );
    assert!(
        installed_after > installed_before,
        "E-family: branch checkpoints are being computed but never installed"
    );
    eprintln!(
        "[E] branch checkpoints: computed {computed_before}->{computed_after}, \
         installed {installed_before}->{installed_after}"
    );
    drop(edited_a);
    drop(edited_b);
}
