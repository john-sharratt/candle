//! Operation workflows: a mind's named sequences of steps, written in
//! `missions.yaml`, and the pure state machine an operation runs through them.
//! `docs/npcd_workflows.md` is the reference.
//!
//! A workflow is an ordered map of steps; the first listed is where an
//! operation starts. Each step says who takes it (`by`: `maker`, `another`, or
//! the `table` through a Rust-registered `call`), the prompt its taker is
//! given, what it does to the document (`edits`), and where it leads — `next:
//! <target>` for one result, or a map of outcome → target. A target is a step,
//! `done` or `failed`; `cancelled` is the engine's alone. A route to an earlier
//! step, or the same one, is a send-back and opens a new round, within which
//! `another` is judged; a workflow's `send-backs` bounds them. Every actor step
//! also offers `stuck`, routed by its `stuck` key once `stuck-limit` reports
//! are made in a round. Nothing here knows what a step is for: a story
//! pipeline and a repair job are the same machinery with different YAML.
//!
//! Prompts are resolved in two stages. At load, includes — a shared prompt
//! from `prompts:` as `{name}`, another step's prompt as
//! `{[workflow.]step.prompt[.variant]}` — are spliced in. When a step is
//! offered, the variant for the incoming outcome is selected and the
//! operation's own placeholders (`{objective}`, `{findings}`, …) are filled.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`config`] | the typed shape: [`Missions`], [`Generator`], [`Workflow`], [`Step`] |
//! | [`parse`] | reading `missions.yaml` into it |
//! | [`validate`] | the load-time rules workflows and generators must satisfy |
//! | [`includes`] | resolving prompt includes at load |
//! | [`run`] | an operation's persisted [`Run`], and the functions that move it |
//! | [`prompt`] | filling a prompt's operation placeholders |

pub mod config;
pub mod includes;
pub mod parse;
pub mod prompt;
pub mod run;
pub mod validate;

#[cfg(test)]
mod examples;
#[cfg(test)]
mod template;

pub use config::{
    By, Edits, Generator, Missions, Next, OnFailed, Step, Variants, Workflow, CANCELLED, DEFAULT,
    DONE, FAILED, START, STUCK,
};
pub use parse::{parse_missions, parse_workflows};
pub use prompt::{fill, placeholders};
pub use run::{
    advance, cancel, may_take, offered, reopen, start, start_at, Offered, Run, Taken, Taker, Where,
};
