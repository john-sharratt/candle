//! Working at a bench: branch, edit, diff, offer, merge — and the history that
//! makes an outcome checkable by somebody other than the one who claims it.
//!
//! # Named after git, deliberately
//!
//! Every verb here is one the model has seen millions of times. An earlier draft
//! called these `open_working`, `show_changes`, `set_aside` and `roll_back` —
//! coinages that read well and recruit nothing. A constrained grammar makes
//! every name equally *valid*, so the only thing separating two names when the
//! model chooses is how much one looks like something it has done before.
//!
//! The fiction is unharmed by it. The part's own prose stays in the vault's
//! voice; the identifier is git. That is the same split `say` keeps by taking an
//! intent while the narrator writes the words — the surface carries the story,
//! the mechanism is plain.
//!
//! # Why a merge that can fail is the point
//!
//! `bench_commit` is the only act in the catalog that can be refused by
//! **somebody else's work**. Two Makers editing overlapping ground produce a
//! conflict that names both parties, cannot be ignored, and has exactly one
//! resolution procedure — which is the trigger condition for settling.
//!
//! That matters beyond tidiness: nothing else in the world gives two characters
//! a *reason* to be in the same room. An idle nudge cannot, because it has no
//! grounds to prefer one interlocutor over another. Parallel work manufactures
//! social work as a side effect, and this is where it comes from.
//!
//! # And why the history is not the character's to write
//!
//! An outcome a character reports about itself is a label it wrote on its own
//! work. `bench_log` and `bench_blame` answer *what was actually done* and *who
//! actually did it* — neither is negotiable by whoever produced them, which is
//! what makes them worth having when everything else is self-reported.

use super::tools::{Availability, Example, Param, Plane, Tool};

/// Every station a body can work at — the six that write content.
///
/// **One line, rather than six YAML edits.** This is the whole reason the
/// attachment lives on the act: `bench_branch` reaching six stations is this
/// list, where the parts declaring it meant the same act written into six files
/// with six chances to miss one.
const BENCHES: &[&str] = &[
    "chronicle-terminal",
    "character-terminal",
    "story-desk",
    "easel",
    "survey-desk",
    "map-table",
];

/// Where working files are actually made. A story desk accumulates scratch
/// nobody else has to see; a settling table never does.
const SCRATCH: &[&str] = &["story-desk"];

/// A bench act with no arguments — it acts on what you are working on.
macro_rules! bench {
    ($name:literal, $at:expr, $desc:literal, $situation:literal, $because:literal) => {
        Tool {
            name: $name,
            at: $at,
            category: "Bench",
            plane: Plane::World,
            availability: Availability::AtPart,
            description: $desc,
            params: &[],
            examples: &[Example {
                situation: $situation,
                call: "{}",
                because: $because,
            }],
        }
    };
}

/// A bench act naming what it acts on.
macro_rules! bench_on {
    ($name:literal, $at:expr, $desc:literal, $arg:literal, $argdesc:literal,
     $situation:literal, $call:literal, $because:literal) => {
        Tool {
            name: $name,
            at: $at,
            category: "Bench",
            plane: Plane::World,
            availability: Availability::AtPart,
            description: $desc,
            params: &[Param {
                name: $arg,
                ty: "string",
                required: true,
                description: $argdesc,
            }],
            examples: &[Example {
                situation: $situation,
                call: $call,
                because: $because,
            }],
        }
    };
}

pub const BENCH_BRANCH: Tool = bench_on!(
    "bench_branch", BENCHES,
    "Start changing something. Your work is yours alone until you offer it, and while you have it \
     open nobody else can commit over you.",
    "what", "What you are opening for work.",
    "You have read enough of a thing to know what is wrong with it and you are about to change it.",
    r#"{"what":"the third era"}"#,
    "Opening the work is a separate act from doing it, and it is what tells everybody else you \
     are in there."
);

pub const BENCH_DIFF: Tool = bench!(
    "bench_diff", BENCHES,
    "See what you have actually changed against what stands. Not what you meant to change — what \
     is different.",
    "You have been working for a while and are no longer certain what you have touched.",
    "The gap between what you think you changed and what you changed is where the surprises are."
);

pub const BENCH_STASH: Tool = bench!(
    "bench_stash", BENCHES,
    "Set your changes aside without giving them up. The thing goes back to what it was and your \
     work waits for you.",
    "Something more urgent has come up and you are in the middle of a change you do not want to \
     lose.",
    "Setting aside is not discarding, and the difference is the whole reason both acts exist."
);

pub const BENCH_STASH_POP: Tool = bench!(
    "bench_stash_pop", BENCHES,
    "Pick your own set-aside work back up, exactly where you left it. Only yours, and only what \
     `bench_stash` put down — this is the other half of setting aside rather than a way into \
     anybody else's unfinished change.",
    "The urgent thing is dealt with and the change you were making is still waiting.",
    "Coming back to your own work is an act; it is not the same as starting again."
);

pub const BENCH_RESTORE: Tool = bench!(
    "bench_restore", BENCHES,
    "Throw your changes away and put the thing back to what it was. This does not ask twice.",
    "You have taken a change a long way and it is wrong from the start, not wrong in the details.",
    "Starting again from the other end is sometimes the cheapest thing there is. Knowing that is \
     judgement."
);

pub const BENCH_STAGE: Tool = bench!(
    "bench_stage", BENCHES,
    "Put your change up as done, for it to be looked at. It stops being private and is not yet \
     part of what stands.",
    "You have finished, you have read it back, and you would not change anything else without \
     being told to.",
    "Offering is a claim that it is ready — which is why it is a decision and not a save."
);

pub const BENCH_UNSTAGE: Tool = bench!(
    "bench_unstage", BENCHES,
    "Take your offered change back, because you are not finished after all.",
    "Somebody has said something about your change and they are right.",
    "Withdrawing is cheap and taking back a bad merge is not."
);

pub const BENCH_COMMIT: Tool = bench_on!(
    "bench_commit", BENCHES,
    "Merge your change into what stands. **It can fail**: if somebody else has changed the same \
     ground while you were working, it comes back with their name on it and the two of you have \
     to settle it. That is not an error, it is the situation.",
    "why", "What the change is, in one line, for whoever reads the history in a year.",
    "You have offered a change, nothing has come back against it, and it is time it was part of \
     the record.",
    r#"{"why":"dated the second burning to the year the redoubt fell, matching both neighbours"}"#,
    "The one act somebody else's work can refuse. What comes back is a person to talk to, not a \
     failure to retry."
);

pub const BENCH_STATUS: Tool = bench!(
    "bench_status", BENCHES,
    "See what you have open, what is offered, and — when a merge has come back — exactly what it \
     collided with and whose it was.",
    "Your commit came back and you do not yet know what it ran into.",
    "The collision names the other party, which is the thing that turns a failed merge into a \
     conversation."
);

pub const BENCH_BLAME: Tool = bench_on!(
    "bench_blame", BENCHES,
    "Trace a thing back through everyone who has held it, and find the point where the chain goes \
     quiet.",
    "what", "What you are tracing.",
    "You are about to rely on something and cannot tell where it came from.",
    r#"{"what":"the third era"}"#,
    "A chain of custody with a time on every link, and not one of them written by somebody \
     describing their own work."
);

pub const BENCH_LOG: Tool = bench_on!(
    "bench_log", BENCHES,
    "See what has actually been done to a thing, and when. The history is not written by whoever \
     produced it, which is what makes it worth reading back.",
    "what", "What you want the history of, by the name it is held under.",
    "Something is not the way you remember leaving it.",
    r#"{"what":"the third era"}"#,
    "What was done, as against what anybody says was done. The two diverge exactly where it \
     matters."
);

// ── Files ───────────────────────────────────────────────────────────────────
//
// Ordinary file editing, because that is what revising a draft *is*. The names
// are the ones every model has seen: read, write, edit, list, delete. The edit
// semantics are `zend-tools`' — a replacement applies only where its target
// appears exactly once, so an ambiguous edit is refused rather than applied to
// the wrong one of three.

pub const FILE_READ: Tool = Tool {
    name: "file_read",
    at: BENCHES,
    category: "Bench",
    plane: Plane::World,
    availability: Availability::AtPart,
    description: "Read a document, as it actually is on the page — including your own uncommitted \
                  changes to it, which nobody else can see yet. Comes back numbered by line, at \
                  most 200 lines at a time: a header reading `(lines 1-200 of 900)` means there \
                  is more, and the next read asks for `start_line` 201.",
    params: &[
        Param {
            name: "path",
            ty: "string",
            required: true,
            description: "Which document, as a path under the world's own documents — \
                          `layers/eras/third.md`.",
        },
        // **A count carried as a string, and parsed by the act.** `integer`
        // compiles to "any structurally-valid JSON value" in the stencil, and
        // JSON nests without limit — the grammar then never finishes building.
        // See `tools::tests::every_parameter_is_a_type_the_grammar_can_bound`.
        Param {
            name: "start_line",
            ty: "string",
            required: false,
            description: "Which line to start at, counting from 1. Leave it out for the start of \
                          the document.",
        },
    ],
    examples: &[Example {
        situation: "You are about to change something and want the exact wording rather than your \
                    memory of it.",
        call: r#"{"path":"eras/third.md"}"#,
        because: "Reading the page is not the same as reading the station's summary of it, and a \
                  change made from memory is a change made to a thing that is not there.",
    }],
};

pub const FILE_WRITE: Tool = Tool {
    name: "file_write",
    at: SCRATCH,
    category: "Bench",
    plane: Plane::World,
    availability: Availability::AtPart,
    description: "Write a document whole — for something new, or for a rewrite that keeps nothing. \
                  What you write is yours and unseen until you commit.",
    params: &[
        Param {
            name: "path",
            ty: "string",
            required: true,
            description: "Which document, as a path under the world's own documents — \
                          `eras/third.md`. Only `.md` and `.yaml` are documents.",
        },
        Param {
            name: "content",
            ty: "string",
            required: true,
            description: "The whole of what the document is to say. Everything it said before is \
                          replaced by this.",
        },
    ],
    examples: &[Example {
        situation: "You are making something that does not exist yet.",
        call: r#"{"path":"stories/the-third-silence.md","content":"A night at the gate, and nobody came through it."}"#,
        because: "Whole-document writing is for a new thing. Changing part of an existing one is \
                  `file_edit`, which cannot silently lose the rest.",
    }],
};

pub const FILE_EDIT: Tool = Tool {
    name: "file_edit",
    at: BENCHES,
    category: "Bench",
    plane: Plane::World,
    availability: Availability::AtPart,
    description: "Change one part of a document, leaving the rest exactly as it is. What you are \
                  replacing has to appear exactly once — if it appears twice you will be told how \
                  many, rather than have the wrong one changed.",
    params: &[
        Param {
            name: "path",
            ty: "string",
            required: true,
            description: "Which document to change, as a path — `layers/eras/third.md`.",
        },
        Param {
            name: "old_str",
            ty: "string",
            required: true,
            description: "The text to replace, exactly as the document has it. Give enough of \
                          what surrounds it to pick out one place and not three.",
        },
        Param {
            name: "new_str",
            ty: "string",
            required: true,
            description: "What stands there instead.",
        },
    ],
    examples: &[Example {
        situation: "One line in a long entry is wrong and everything around it is fine.",
        call: r#"{"path":"eras/third.md","old_str":"burned in the spring","new_str":"burned in the autumn, the year the redoubt fell"}"#,
        because: "The uniqueness rule is what makes an edit safe to make without reading the \
                  whole thing back — so the target carries its neighbours, not just the words \
                  being changed.",
    }],
};

pub const FILE_LIST: Tool = bench_on!(
    "file_list", SCRATCH,
    "See what documents are actually in a place — including the ones you have made and not \
     committed, and without the ones you have taken out. What exists, rather than what anything \
     says exists.",
    "path", "Which directory to look in — `eras`, or nothing at all for the top.",
    "You are looking for something and are not certain what it is called.",
    r#"{"path":"eras"}"#,
    "What exists, rather than what the index says exists — the two drift, and that drift is a \
     thing worth finding."
);

pub const FILE_DELETE: Tool = bench_on!(
    "file_delete", SCRATCH,
    "Remove a document. Retiring something from the record is `record_let_go`, which keeps the \
     reason; this is for a thing that should never have been a document at all.",
    "path", "Which document to take out, as a path.",
    "You made a working file to think in and it is still sitting there looking like canon.",
    r#"{"path":"stories/notes-to-self.md"}"#,
    "Named apart from retiring on purpose. One is a decision about the record; this is tidying."
);

/// Everything in this module, in the order it is offered.
pub const BENCH_ACTS: &[Tool] = &[
    BENCH_BRANCH,
    BENCH_DIFF,
    BENCH_STASH,
    BENCH_STASH_POP,
    BENCH_RESTORE,
    BENCH_STAGE,
    BENCH_UNSTAGE,
    BENCH_COMMIT,
    BENCH_STATUS,
    BENCH_BLAME,
    BENCH_LOG,
    FILE_READ,
    FILE_WRITE,
    FILE_EDIT,
    FILE_LIST,
    FILE_DELETE,
];
