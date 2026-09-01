//! Writing a character's life with the model, so an author does not have to.
//!
//! # The ladder
//!
//! A life is generated at four resolutions, each one built from the ones above
//! it:
//!
//! | Phase | Produces | Fan-out | Job |
//! |---|---|---|---|
//! | Story | the arc, the cast, the shape | 1 | invents everything — the only rung that may |
//! | Years | one document per year of the life | 1 | allocates the arc to years |
//! | Months | one document per month | one fork per month | expands what the year allocated |
//! | Days | one document per defining day | one fork per day | writes prose that earns its consequences |
//!
//! # Assign above, expand below
//!
//! **The rule the whole thing rests on.** A year *allocates* — month by month,
//! what happens where. A month writes out what it was handed. It does not
//! invent.
//!
//! The reason is parallelism. Siblings are generated concurrently and never see
//! each other, so if a child may invent, two months in the same year both
//! introduce a stranger at the gate and neither knows about the other. Coherence
//! comes from the parent having already decided, which is what makes the fan-out
//! safe — and it is why every system prompt below the story is written as an
//! expansion instruction rather than as "continue the story".
//!
//! # One prefix per phase, forked N ways
//!
//! Each phase builds one system prompt from the phases above it, primes it once,
//! and forks it. [`candle_conversation::Conversation`]'s fork shares the system
//! prompt and deliberately inherits no turns — which is exactly the shape wanted
//! here: the shared half is the prompt, the divergent half is each fork's own
//! instruction. The prompt K/V is Arc-injected, so twelve months attend one copy
//! of the story rather than twelve.
//!
//! That makes the *variable* part physically incapable of contaminating the
//! shared part: you cannot put "expand month 3" in the prefix, because the
//! prefix **is** the system prompt.
//!
//! # The prefix is built from edited artifacts, never inherited
//!
//! A phase's prompt is composed from what the plan holds *now* — including
//! whatever the operator corrected by hand — and not from the conversation that
//! generated it. An operator's edit is the point of the review loop, and a
//! prefix inherited from the original decode would silently ignore it.
//!
//! [`plan::Plan::prefix_hash`] is what makes that enforceable rather than
//! hoped-for: a fork carries the hash it was planned against, and a mismatch is
//! refused. It is the one way this design can be wrong without looking wrong.
//!
//! # Nothing here touches the substrate
//!
//! The generator writes markdown into `layers/life/<who>/` and stops. The
//! watcher notices, [`crate::engine::ingest`] runs, and beliefs form through the
//! same path a hand-authored life takes. There is no privileged write: a life
//! that came out badly is a text file to delete.
//!
//! | File | Concern |
//! |---|---|
//! | [`calendar`] | the date arithmetic a life needs, and nothing else |
//! | [`seed`] | what an operator hands the generator to start |
//! | [`plan`] | the tree under construction, and what is stale |
//! | [`consequence`] | operator-authored tool calls, injected rather than decoded |
//! | [`document`] | a node written out as a life document on disk |
//! | [`prompt`] | each phase's shared prefix and its per-fork instruction |
//! | [`progress`] | what the generating overlay reads |
//! | [`job`] | one generation run: its state, and cancelling it |
//! | [`generate`] | priming a prefix and fanning it out over the engine |
//! | [`routes`] | the API the life editor talks to |

pub mod calendar;
pub mod consequence;
pub mod document;
pub mod generate;
pub mod job;
pub mod plan;
pub mod progress;
pub mod prompt;
pub mod routes;
pub mod seed;
