//! Family G of the behavioural catalogue
//! (`docs/recurrent_state_behavioural_tests.md`): quality, measured as a
//! **paired comparison against the amnesia control** — the same conversations,
//! half of them resumed with their memory record poisoned so the restore
//! refuses it and they come back with full K/V and zero memory.
//!
//! These are the only tests whose assertion is on text, so they run repeated
//! trials and report margins. What is HARD-GATED is only what must hold
//! absolutely: every memory-carrying resume recalls its fact (G1's absolute
//! half, across thinking turns — G3's behavioural face), and the amnesia arm
//! never out-recalls the memory arm. The margin itself is REPORTED each run.
//!
//! What the margins mean on this architecture, as measured: K/V-only recall
//! (the amnesia arm) is WEAK and threshold-dependent — a single flat mention
//! mostly fails (D5's finding: the resume enumerates its context as prompt +
//! question, nothing else), while a doubly-stated salient fact can cross on
//! attention alone. And once the summariser engages, its nodes KEEP specific
//! facts (the schema instructs exactly that), so a compressed span's summary
//! TEXT can carry a fact in K/V for both arms. Recall is served by **layered
//! carriers** — raw K/V, then summary text, then recurrent memory — and the
//! amnesia control isolates memory's contribution only where the other layers
//! don't already serve the fact. The margins reported here are the standing
//! measurement of that layering, run over run.
//!
//! ```text
//! cargo test -p zend --release --features cuda --test recall_quality \
//!   -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use candle::Device;
use common::{memory_is_empty, poison_memory_record, probe_recall, say, sealed_memory, Workspace};

/// Facts assigned to the shallow trials, memory group first.
const WORDS: [&str; 6] = [
    "vermilion",
    "kestrel",
    "harbour",
    "obsidian",
    "juniper",
    "cobalt",
];
/// Trials per group in the shallow arm.
const PER_GROUP: usize = 3;
/// Filler turns after the fact in the shallow arm — depth for the seal cadence
/// and the summariser, and thinking turns for G3's behavioural claim.
const DEPTH: usize = 3;

/// Facts for the DEEP arm — fact + enough turns that the summariser compresses
/// the fact's span (past `RAW_TAIL_TURNS`), memory pair first. This is A7's
/// recall question and G2's depth dimension in trial form: single-shot recall
/// at compressed depth sits near threshold on this model, so it is measured
/// here with a margin, not gated on one sample.
const DEEP_WORDS: [&str; 4] = ["lantern", "granite", "sparrow", "meridian"];
/// Trials per group in the deep arm.
const DEEP_PER_GROUP: usize = 2;
/// Filler turns in the deep arm — past the summariser's raw tail of 8.
const DEEP_DEPTH: usize = 10;

#[test]
#[ignore = "Family G cruise: loads the pinned GGUF twice and runs paired recall trials \
            (~10 min). Run with: cargo test -p zend --release --features cuda --test \
            recall_quality -- --ignored --nocapture --test-threads=1"]
fn recall_margin_over_amnesia_control() {
    let device = Device::new_cuda(0).expect("cuda");
    let ws = Workspace::new();

    // ── Build the trials ────────────────────────────────────────────────────
    let (timelines, deep_timelines) = {
        let session = ws.session(&device);
        let build = |i: usize, word: &str, depth: usize, per_group: usize| {
            let mut conv = session.start();
            say(
                &mut conv,
                &format!(
                    "The codeword is {}. This matters later: remember the word \
                     {} exactly. Acknowledge in one word.",
                    word.to_uppercase(),
                    word.to_uppercase()
                ),
            );
            for f in 0..depth {
                say(
                    &mut conv,
                    &format!("Filler {f}: reply with one short sentence."),
                );
            }
            let tl = conv.timeline_id();
            // Only the POISON arm waits, and for the NEWEST record rather than
            // a specific turn's: the summariser injects ghost summary turns
            // whose seals supersede the single-tail snapshot, so "the record
            // for turn N" stops being a satisfiable address the moment
            // summarisation engages. The poison must land after the newest
            // snapshot; if a straggler seal still lands after IT, the
            // control-validity assert at resume (`memory_is_empty`) catches
            // the un-poisoned control loudly. The memory arm needs no wait —
            // the engine drop flushes the writer.
            if i >= per_group {
                sealed_memory(session.engine(), tl);
                poison_memory_record(session.engine(), tl);
            }
            tl
        };
        let timelines: Vec<_> = WORDS
            .iter()
            .enumerate()
            .map(|(i, w)| build(i, w, DEPTH, PER_GROUP))
            .collect();
        let deep_timelines: Vec<_> = DEEP_WORDS
            .iter()
            .enumerate()
            .map(|(i, w)| build(i, w, DEEP_DEPTH, DEEP_PER_GROUP))
            .collect();
        (timelines, deep_timelines)
        // Engine drops; the redo log is all that is left.
    };

    // ── Resume and probe ────────────────────────────────────────────────────
    let session = ws.session(&device);
    let base = session.start();
    let probe_arm = |tag: &str,
                     timelines: &[candle_conversation::projection::TimelineId],
                     words: &[&str],
                     per_group: usize|
     -> (usize, usize) {
        let mut memory_recalls = 0usize;
        let mut amnesia_recalls = 0usize;
        for (i, (tl, word)) in timelines.iter().zip(words.iter()).enumerate() {
            let mut conv = base.fork_resuming(*tl).expect("resume trial");
            let amnesia = i >= per_group;
            // The control must actually BE the control: memory refused, K/V
            // intact.
            if amnesia {
                assert!(
                    memory_is_empty(&conv),
                    "G-control: {tag} trial {i}'s poisoned record was installed — \
                     the amnesia arm is not amnesiac, and every margin below is void"
                );
            }
            let (recalled, reply) =
                probe_recall(&mut conv, "What was the codeword? One word.", word);
            eprintln!(
                "[G1:{tag}] trial {i} ({}{word}): {}",
                if amnesia { "amnesia:" } else { "memory:" },
                reply.trim()
            );
            if recalled {
                if amnesia {
                    amnesia_recalls += 1;
                } else {
                    memory_recalls += 1;
                }
            }
        }
        (memory_recalls, amnesia_recalls)
    };

    let (memory_recalls, amnesia_recalls) = probe_arm("shallow", &timelines, &WORDS, PER_GROUP);
    let (deep_memory, deep_amnesia) =
        probe_arm("deep", &deep_timelines, &DEEP_WORDS, DEEP_PER_GROUP);

    eprintln!(
        "[G1] shallow (depth {DEPTH}): memory {memory_recalls}/{PER_GROUP}, amnesia \
         {amnesia_recalls}/{PER_GROUP}, margin {:+}",
        memory_recalls as i64 - amnesia_recalls as i64
    );
    eprintln!(
        "[G2] deep (depth {DEEP_DEPTH}, compressed span): memory \
         {deep_memory}/{DEEP_PER_GROUP}, amnesia {deep_amnesia}/{DEEP_PER_GROUP}, \
         margin {:+}",
        deep_memory as i64 - deep_amnesia as i64
    );
    // The absolute gate (G1's memory half + G3 across thinking turns): a
    // conversation that kept its memory must recall its fact at shallow depth,
    // every time. The DEEP arm is measured, not absolutely gated — recall at
    // compressed depth sits near threshold on this model, and this report is
    // where its trajectory shows across runs (raising DEEP_DEPTH is how G2's
    // curve is traced when a long-run slot exists).
    assert_eq!(
        memory_recalls, PER_GROUP,
        "G1: a memory-carrying resume failed to recall its fact at depth \
         {DEPTH} — recall with memory has degraded"
    );
    // The shallow margin is inequality-gated at its full sample; the deep
    // margin is REPORTED ONLY — at this trial count it is noise, and the first
    // measured run showed why it can legitimately sit at or below zero:
    // summary nodes KEEP specific facts (by the schema's own instruction), so
    // the compressed span's summary TEXT carries the codeword in K/V for both
    // arms. Recall is served by layered carriers — raw K/V, then summary
    // text, then recurrent memory — and the amnesia control isolates memory's
    // contribution only where the other layers don't already serve the fact.
    // The deep report is the standing measurement of that layering.
    assert!(
        amnesia_recalls <= memory_recalls,
        "G1: the amnesia control out-recalled the memory group — memory is \
         hurting recall"
    );
}
