//! Priming one shared prefix per phase and fanning it out over the engine.
//!
//! # The optimisation, and why it is a property of the API rather than a trick
//!
//! [`Sequence::fork`] mints a fresh timeline that **shares the parent's system
//! prompt and inherits none of its turns**. Elsewhere that is a trap worth a
//! warning in its own doc comment. Here it is exactly the shape wanted: the
//! shared half is the prompt, the divergent half is each fork's own
//! instruction.
//!
//! And the sharing is real rather than nominal — a new conversation Arc-injects
//! its sealed prompt K/V and every fork clones that same primed prefix, so five
//! hundred months attend one copy of the story instead of five hundred.
//!
//! The consequence for prompt construction is not a rule to remember: a
//! per-fork detail *cannot* reach the shared context, because the shared
//! context is the system prompt and the detail travels in the turn.
//!
//! # What is not free on this model
//!
//! A model carrying recurrent state cannot have that state Arc-injected — it
//! has to be computed by running the prompt tokens. Qwen3.5's hybrid is mostly
//! DeltaNet layers, so priming costs a real forward. It is paid **once per
//! phase**, not once per fork, because forks start from the prompt branch
//! checkpoint the parent already built. That is why the progress overlay
//! reports priming as its own stage rather than pretending the phase has begun.
//!
//! # Concurrency is what makes the fan-out a wave
//!
//! Each fork's `send_turn` blocks, so the forks are driven from a pool of
//! threads. They submit to the same scheduler, which batches them into waves —
//! the fan-out is parallel because several sequences are in flight at once, not
//! because any one call is asynchronous. [`MAX_IN_FLIGHT`] caps the pool at the
//! engine's wave width; beyond it the extra forks would queue inside the
//! scheduler anyway, while holding KV for a turn that has not started.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};

use super::document;
use super::plan::{CastMember, NodeId, Phase, Plan};
use super::progress::{GenProgress, Stage};
use super::prompt;
use super::seed::check;

/// How many forks decode at once.
///
/// The engine's wave width. Past it, extra forks do not decode any sooner —
/// they queue inside the scheduler while holding a slot's worth of KV for a
/// turn that has not started.
pub const MAX_IN_FLIGHT: usize = 64;

/// A fork's answer, or why it has none.
struct Answer {
    id: NodeId,
    text: Result<String, String>,
}

/// Prime a phase's prefix and run every pending node of it.
///
/// Returns how many nodes were written. A phase with nothing pending primes
/// nothing and returns zero — regenerating one month must not pay for a prefix
/// no fork is going to use.
pub fn run_phase(
    engine: &Arc<Mutex<ConversationEngine>>,
    cfg: &SequenceConfig,
    mind: &Path,
    plan: &Mutex<Plan>,
    phase: Phase,
    progress: &GenProgress,
) -> anyhow::Result<usize> {
    progress.enter(phase);

    // Everything the prompts are built from is read once, here, from the plan
    // as it stands — **including whatever the operator corrected by hand**. A
    // prefix inherited from the decode that produced the parent would silently
    // ignore the edit that is the entire point of the review loop.
    let (snapshot, primed_at, pending) = {
        let p = plan.lock().unwrap();
        (p.clone(), p.prefix_hash(), p.pending(phase))
    };
    if pending.is_empty() {
        return Ok(0);
    }

    let prefix = match phase {
        Phase::Story => {
            let checked = check(&snapshot.seed).map_err(|e| {
                anyhow::anyhow!(
                    "seed: {}",
                    e.iter().map(|b| b.message()).collect::<Vec<_>>().join("; ")
                )
            })?;
            prompt::story_prefix(&checked)
        }
        Phase::Years => prompt::years_prefix(&snapshot),
        Phase::Months => prompt::months_prefix(&snapshot),
        Phase::Days => prompt::days_prefix(&snapshot),
    };

    // One turn per fork, so no fork carries any other's history.
    let mut fork_cfg = cfg.clone();
    fork_cfg.context_window_turns = 0;

    progress.detail(format!("{} node(s)", pending.len()));
    let parent = engine
        .lock()
        .unwrap()
        .new_conversation(&prefix, fork_cfg.clone())?;

    let turns: Vec<(NodeId, String)> = pending
        .iter()
        .map(|id| (*id, turn_for(&snapshot, *id)))
        .collect();

    // **What was actually asked, in characters.** A phase that produces nothing looks
    // identical whether the prompt was wrong, the turn was empty, or the model declined — and
    // the first two are the common causes. Sizes here separate them at a glance.
    tracing::debug!(
        target: "npcd::lifegen",
        phase = ?phase,
        prefix_chars = prefix.len(),
        forks = turns.len(),
        first_turn_chars = turns.first().map(|(_, t)| t.len()).unwrap_or(0),
        "phase primed"
    );
    progress.fanning_out(turns.len() as u64);
    let answers = fan_out(&parent, turns, progress)?;
    for a in &answers {
        match &a.text {
            Ok(t) => tracing::debug!(
                target: "npcd::lifegen",
                node = ?a.id,
                reply_chars = t.len(),
                "answered"
            ),
            Err(e) => tracing::debug!(target: "npcd::lifegen", node = ?a.id, "refused: {e}"),
        }
    }

    // **The prefix a fork was planned against must still be the prefix.** An
    // operator editing the story while a phase runs would otherwise have every
    // node of it written against an arc that no longer exists — plausibly, and
    // with nothing anywhere to say so.
    let mut p = plan.lock().unwrap();
    if p.prefix_hash() != primed_at {
        anyhow::bail!(
            "the life story changed while the {} phase was running — nothing was written; \
             run it again against the story as it stands now",
            phase.unit()
        );
    }

    let mut written = 0;
    for a in answers {
        let Ok(text) = a.text else {
            tracing::warn!(
                "life {}: {:?} not generated — {}",
                p.seed.who,
                a.id,
                a.text.unwrap_err()
            );
            continue;
        };
        // **The decode as the model produced it, before any parsing.**
        //
        // Everything downstream — `strip_reasoning`, `parse_story`, the document write — can
        // only report what survived it, and every failure in this phase so far has been
        // diagnosed from the wreckage rather than the event. The distinction this exists to
        // draw: prose that is wrong from its first token is a prompt or a weights problem,
        // while prose that starts clean and degrades is a state one, and the parsed output
        // cannot tell the two apart. At `debug`, so it costs nothing in a normal run.
        tracing::debug!(
            target: "npcd::lifegen",
            who = %p.seed.who,
            node = ?a.id,
            chars = text.len(),
            head = %text.chars().take(400).collect::<String>(),
            "decode"
        );
        if apply(&mut p, a.id, &text) {
            written += 1;
        } else {
            // **A decode that parses to nothing is a failure, and it used to be a
            // silent one.** `apply` refuses empty prose, but saying so nowhere meant a
            // node that produced 311 seconds of tokens and no document was
            // indistinguishable from one that worked — the job still reported `done`.
            // The length is here because it separates the two ways to arrive at
            // nothing: a model that stopped immediately, and one that reasoned to the
            // token cap without ever closing its `<think>` block.
            tracing::warn!(
                "life {}: {:?} produced no document from {} chars of decode",
                p.seed.who,
                a.id,
                text.len()
            );
        }
    }

    progress.stage(Stage::Writing);
    // Only a story that was actually generated is written. Without the guard a redo
    // that cleared the node and then failed to refill it wrote the *empty* node over
    // the good document already on disk — a failed regeneration destroying the thing
    // it was regenerating. `routes.rs::commit` has always tested this; this call site
    // did not.
    if phase == Phase::Story && p.story.content.is_generated() {
        document::write_story(mind, &p.seed.who, &p.story.content.text)?;
    }
    let report = document::sync(mind, &p)?;
    if !report.removed.is_empty() {
        tracing::info!(
            "life {}: {} document(s) removed as no longer part of the plan",
            p.seed.who,
            report.removed.len()
        );
    }
    super::plan::save(mind, &p)?;
    Ok(written)
}

/// The instruction one fork gets.
fn turn_for(plan: &Plan, id: NodeId) -> String {
    match id {
        NodeId::Story => match check(&plan.seed) {
            Ok(c) => prompt::story_turn(&c),
            // Unreachable in practice: `run_phase` checks the seed before it
            // primes. Kept total rather than panicking, because a fork that
            // asks for nothing is recoverable and a poisoned pool is not.
            Err(_) => String::new(),
        },
        NodeId::Year { year } => prompt::year_turn(plan, year),
        NodeId::Month { year, month } => prompt::month_turn(plan, year, month),
        NodeId::Day { year, month, day } => prompt::day_turn(plan, year, month, day),
    }
}

/// Fork the primed parent once per node and decode a wave at a time.
///
/// # Why the forking happens here and the decoding does not
///
/// A [`Sequence`] is `Send` but not `Sync`, so the primed parent stays on this
/// thread and each *child* is moved into a worker. That is not a workaround —
/// it is the shape the cap wants anyway: forks are minted one wave at a time,
/// so a five-hundred-month phase holds sixty-four slots rather than five
/// hundred, and the rest wait as instructions in a `Vec` instead of as
/// conversations holding KV for a turn that has not started.
///
/// Cancellation is checked between waves rather than inside a decode. A turn
/// already in flight is left to finish, because tearing one down leaves a slot
/// the engine still believes is busy.
fn fan_out(
    parent: &Sequence,
    turns: Vec<(NodeId, String)>,
    progress: &GenProgress,
) -> anyhow::Result<Vec<Answer>> {
    let out: Mutex<Vec<Answer>> = Mutex::new(Vec::with_capacity(turns.len()));
    let flight = AtomicUsize::new(0);

    for wave in turns.chunks(MAX_IN_FLIGHT) {
        if progress.is_cancelled() {
            break;
        }
        // Mint this wave's forks. Each shares the parent's primed prompt K/V
        // and inherits none of its turns, so they are the same context and
        // separate histories.
        let mut forked = Vec::with_capacity(wave.len());
        for (id, turn) in wave {
            forked.push((*id, parent.fork()?, turn.as_str()));
        }
        progress.in_flight(forked.len() as u64);
        flight.store(forked.len(), Ordering::Relaxed);

        std::thread::scope(|scope| {
            for (id, mut seq, turn) in forked {
                let (out, flight, progress) = (&out, &flight, &progress);
                scope.spawn(move || {
                    let answer = seq
                        .send_turn(turn)
                        .map(|r| r.text)
                        .map_err(|e| e.to_string());
                    // The slot goes back as soon as the answer is out of it.
                    drop(seq);
                    progress.in_flight(flight.fetch_sub(1, Ordering::Relaxed) as u64 - 1);
                    progress.completed_one(detail_for(id));
                    out.lock().unwrap().push(Answer { id, text: answer });
                });
            }
        });
    }

    let mut answers = out.into_inner().unwrap();
    // Workers finish out of order; the plan is applied in reading order so a
    // log of one run reads like the life.
    answers.sort_by_key(|a| a.id);
    Ok(answers)
}

fn detail_for(id: NodeId) -> String {
    match id {
        NodeId::Story => "the life story".to_string(),
        _ => id.date_key().unwrap_or_default(),
    }
}

/// Fold one fork's answer into the plan. Returns whether anything landed.
fn apply(plan: &mut Plan, id: NodeId, raw: &str) -> bool {
    match id {
        NodeId::Story => {
            let out = prompt::parse_story(raw);
            if out.prose.trim().is_empty() {
                return false;
            }
            // The seed's own people are merged in first and marked, so a
            // relationship formed against one of them resolves to the NPC it
            // already names rather than to a slug the story reinvented.
            let mut cast: Vec<CastMember> = plan
                .seed
                .cast
                .iter()
                .map(|c| CastMember {
                    entity_id: c.entity_id.clone(),
                    display: c.display.clone(),
                    what: c.what.clone(),
                    npc_id: c.npc_id,
                    from_seed: true,
                })
                .collect();
            for c in out.cast {
                if !cast.iter().any(|x| x.entity_id == c.entity_id) {
                    cast.push(c);
                }
            }
            plan.story.cast = cast;
            // Only years the life actually has. A model that outlines 1997 for
            // a character born in 1998 has written a year that no document can
            // be named for.
            let real: Vec<i32> = plan.years.iter().map(|y| y.year).collect();
            plan.story.outline = out
                .outline
                .into_iter()
                .filter(|b| real.contains(&b.year))
                .collect();
            plan.story
                .content
                .generated("Life story".to_string(), out.prose);
            true
        }
        NodeId::Year { year } => {
            let (title, prose) = prompt::parse_titled(raw);
            if prose.trim().is_empty() {
                return false;
            }
            match plan.content_mut(NodeId::Year { year }) {
                Some(c) => {
                    c.generated(title, prose);
                    true
                }
                None => false,
            }
        }
        NodeId::Month { year, month } => {
            let out = prompt::parse_month(raw, year, month);
            if out.prose.trim().is_empty() {
                return false;
            }
            // The month names its own defining days, which is the allocation
            // the day phase then expands. A day the operator has already edited
            // keeps its title.
            for (day, title) in out.days {
                if let Some(d) = plan.ensure_day(year, month, day) {
                    if !d.content.edited && d.content.title.is_empty() {
                        d.content.title = title;
                    }
                }
            }
            match plan.content_mut(NodeId::Month { year, month }) {
                Some(c) => {
                    c.generated(out.title, out.prose);
                    true
                }
                None => false,
            }
        }
        NodeId::Day { year, month, day } => {
            let (title, prose) = prompt::parse_titled(raw);
            if prose.trim().is_empty() {
                return false;
            }
            match plan.content_mut(NodeId::Day { year, month, day }) {
                Some(c) => {
                    // A day's title came from its month's allocation; a decode
                    // that supplies a better one may improve it, but never
                    // replaces one a human set.
                    let keep = c.edited && !c.title.is_empty();
                    let title = if keep { c.title.clone() } else { title };
                    c.generated(title, prose);
                    true
                }
                None => false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifegen::plan::{Content, YearBeat};
    use crate::lifegen::seed::{Cadence, Seed, SeedCast};

    fn seed() -> Seed {
        Seed {
            who: "cindy-tan".into(),
            display: "Cindy Tan".into(),
            born: "1998-09-14".into(),
            through: "2000-03-02".into(),
            place: "Nanyang".into(),
            role: "a clerk".into(),
            cadence: Cadence::Even,
            facts: Vec::new(),
            world: Vec::new(),
            cast: vec![SeedCast {
                entity_id: "prof-lim".into(),
                display: "Professor Lim".into(),
                what: "taught her".into(),
                npc_id: Some(7),
            }],
            eras: Vec::new(),
        }
    }

    fn plan() -> Plan {
        Plan::new(&check(&seed()).unwrap())
    }

    #[test]
    fn a_story_answer_lands_as_prose_cast_and_outline() {
        let mut p = plan();
        let raw = "\
You were born in the rain.

## Cast
- hess | Hess | the man at the granary

## Years
- 1998 | Born | You arrive.
- 1999 | The Quiet Year | Nothing arrives.
- 2000 | The Fire | It goes.
";
        assert!(apply(&mut p, NodeId::Story, raw));
        assert_eq!(p.story.content.text, "You were born in the rain.");
        assert!(p.story.content.is_generated());
        assert_eq!(p.story.outline.len(), 3);
        assert_eq!(p.story.beat(1999).unwrap().title, "The Quiet Year");
    }

    /// **The seed's people keep their NPC binding.** A relationship formed
    /// against one of them must resolve to the character it names, not to a
    /// slug the story reinvented.
    #[test]
    fn the_seeds_cast_survives_the_story_with_its_npc_id() {
        let mut p = plan();
        apply(
            &mut p,
            NodeId::Story,
            "Prose.\n\n## Cast\n- prof-lim | Prof Lim | someone else\n- hess | Hess | x\n",
        );
        let lim = p
            .story
            .cast
            .iter()
            .find(|c| c.entity_id == "prof-lim")
            .unwrap();
        assert_eq!(lim.npc_id, Some(7), "the binding was lost");
        assert!(lim.from_seed);
        assert_eq!(lim.display, "Professor Lim", "the seed's name wins");
        assert_eq!(p.story.cast.len(), 2);
        assert!(p.seeded_entities().contains(&"prof-lim".to_string()));
    }

    /// A year the life does not have cannot be written to a document, so it is
    /// dropped rather than carried as an outline entry nothing expands.
    #[test]
    fn an_outline_year_outside_the_life_is_dropped() {
        let mut p = plan();
        apply(
            &mut p,
            NodeId::Story,
            "Prose.\n\n## Years\n- 1997 | Before | x\n- 1998 | Born | y\n- 2050 | After | z\n",
        );
        assert_eq!(
            p.story.outline.iter().map(|b| b.year).collect::<Vec<_>>(),
            vec![1998]
        );
    }

    /// **The month allocates its defining days**, which is the assignment the
    /// day phase expands.
    #[test]
    fn a_month_answer_creates_the_days_it_named() {
        let mut p = plan();
        let raw = "# First Term\n\nYou arrived late.\n\n## Days\n- 14 | First Week | x\n- 30 | The Argument | y\n";
        assert!(apply(
            &mut p,
            NodeId::Month {
                year: 1998,
                month: 9
            },
            raw
        ));
        let m = p.month(1998, 9).unwrap();
        assert_eq!(m.content.title, "First Term");
        assert_eq!(
            m.days.iter().map(|d| d.day).collect::<Vec<_>>(),
            vec![14, 30]
        );
        assert_eq!(m.days[0].content.title, "First Week");
        // The days exist but are not written, so they are what the day phase
        // will pick up.
        assert_eq!(p.pending(Phase::Days).len(), 2);
    }

    /// A human's title outranks a regenerated month's opinion of it.
    #[test]
    fn regenerating_a_month_does_not_retitle_a_day_a_human_named() {
        let mut p = plan();
        apply(
            &mut p,
            NodeId::Month {
                year: 1998,
                month: 9,
            },
            "# T\n\nP.\n\n## Days\n- 14 | Model Title | x\n",
        );
        p.day_mut(1998, 9, 14)
            .unwrap()
            .content
            .edit("My Title".into(), "I wrote this.".into());
        apply(
            &mut p,
            NodeId::Month {
                year: 1998,
                month: 9,
            },
            "# T2\n\nP2.\n\n## Days\n- 14 | Another Model Title | x\n",
        );
        assert_eq!(p.month(1998, 9).unwrap().days[0].content.title, "My Title");
    }

    #[test]
    fn an_empty_answer_lands_nothing_rather_than_an_empty_document() {
        let mut p = plan();
        assert!(!apply(&mut p, NodeId::Year { year: 1998 }, "   \n\n  "));
        assert!(!p.year(1998).unwrap().content.is_generated());
        assert!(!apply(&mut p, NodeId::Story, "## Cast\n- a | A | x\n"));
    }

    #[test]
    fn a_year_answer_lands_with_its_title() {
        let mut p = plan();
        assert!(apply(
            &mut p,
            NodeId::Year { year: 1999 },
            "# The Quiet Year\n\nYou learned to wait.\n"
        ));
        let y = p.year(1999).unwrap();
        assert_eq!(y.content.title, "The Quiet Year");
        assert_eq!(y.content.text, "You learned to wait.");
    }

    /// Applying to a node the plan does not have is a miss, not a panic — a
    /// model can answer about a month outside the life.
    #[test]
    fn an_answer_for_a_node_outside_the_life_is_dropped() {
        let mut p = plan();
        assert!(!apply(&mut p, NodeId::Year { year: 2050 }, "# X\n\nY.\n"));
        assert!(!apply(
            &mut p,
            NodeId::Month {
                year: 1998,
                month: 1
            },
            "# X\n\nY.\n"
        ));
    }

    /// A day the operator wrote keeps its prose and its title through a
    /// regeneration of the day itself only if it was not asked for — the
    /// pending filter is what protects it, and this asserts the title half.
    #[test]
    fn a_day_answer_keeps_a_title_a_human_set() {
        let mut p = plan();
        p.ensure_day(1999, 1, 3).unwrap();
        p.day_mut(1999, 1, 3)
            .unwrap()
            .content
            .edit("My Day".into(), "mine".into());
        apply(
            &mut p,
            NodeId::Day {
                year: 1999,
                month: 1,
                day: 3,
            },
            "# Model Day\n\nThe model's prose.\n",
        );
        let d = &p.month(1999, 1).unwrap().days[0];
        assert_eq!(d.content.title, "My Day");
        assert_eq!(d.content.text, "The model's prose.");
    }

    /// The turn for each node carries that node's parent text — the per-fork
    /// half, which is exactly what must not be in the shared prefix.
    #[test]
    fn each_nodes_turn_carries_its_own_parent() {
        let mut p = plan();
        p.story.content = Content {
            title: "S".into(),
            text: "An arc.".into(),
            edited: false,
            stale: false,
        };
        p.story.outline.push(YearBeat {
            year: 1999,
            title: "Quiet".into(),
            premise: "Nothing.".into(),
        });
        p.content_mut(NodeId::Year { year: 1999 })
            .unwrap()
            .generated("Quiet".into(), "The year's own text.".into());
        p.content_mut(NodeId::Month {
            year: 1999,
            month: 4,
        })
        .unwrap()
        .generated("April".into(), "The month's own text.".into());
        p.ensure_day(1999, 4, 2).unwrap();

        assert!(turn_for(&p, NodeId::Year { year: 1999 }).contains("Quiet | Nothing."));
        assert!(turn_for(
            &p,
            NodeId::Month {
                year: 1999,
                month: 4
            }
        )
        .contains("The year's own text."));
        assert!(turn_for(
            &p,
            NodeId::Day {
                year: 1999,
                month: 4,
                day: 2
            }
        )
        .contains("The month's own text."));
    }

    /// The fan-out never spawns more workers than there is work, and never
    /// fewer than one.
    #[test]
    fn the_worker_pool_is_bounded_by_the_work_and_the_wave_width() {
        for (work, want) in [(0, 1), (1, 1), (10, 10), (500, MAX_IN_FLIGHT)] {
            assert_eq!(MAX_IN_FLIGHT.min(work).max(1), want);
        }
    }
}
