//! One generation, and the loop that keeps the table stocked.
//!
//! A generation is: draw a generator by weight, find its next target in the
//! corpus that the ledger does not block, show the model the generator's prompt
//! and the target's material, hold its answer to `mission` / `no_mission`, check
//! it, and put the mission on the table — or record that there was nothing to
//! do. The loop runs one at a time per world while the table is open and its
//! pool is below the configured level.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_conversation::guest::resolve_seed;
use candle_conversation::{ConversationEngine, SequenceConfig};
use serde::Serialize;
use tokio::task::spawn_blocking;

use super::answer::{self, Answer, Desk};
use super::canon;
use super::config::{Config, Generator};
use super::corpus::Corpus;
use super::gates;
use super::material;
use super::reading;
use super::target::{self, Kind, Target};
use crate::engine::journal::tools::arguments;
use crate::engine::mind::Minds;
use crate::engine::mind_record;
use crate::engine::mission::{time_step_text, Mission, Todo};
use crate::engine::prose::{self, decode_call, CallAsk};
use crate::engine::runtime::Runtime;
use crate::engine::work::what_happens;
use crate::prose::Request;
use crate::sim::operations::{AfterReading, Operation};
use crate::world::Hosted;

/// The most tokens one answer may run to. The table's reading and its faults
/// run to a thousand words between them, and an answer cut off by the cap is not a
/// call at all — a life event was refused "not a call" three times running.
const ANSWER_TOKENS: usize = 2400;

/// The most tokens the table's reading may run to: an answer's room, and the
/// reader's working in `notes` before it — see [`reading::specs`].
const READING_TOKENS: usize = 4800;

/// How many times an answer is asked for before the target is given up.
const ATTEMPTS: usize = 3;

/// The sampling temperature a mission is written at.
///
/// Below the checkpoint's own, because a mission carries paths and dates that
/// must be copied exactly: at the default a story brief about "the vanishing
/// silence" named its own file `the-vanishing-silage.md`.
const TEMPERATURE: f32 = 0.5;

/// The sampling temperature the table reads a draft at.
///
/// **A judge gives the same verdict twice.** At the generator's temperature
/// the table failed a Zen given hands and a chair — and, reading the same text
/// an hour later, found it sound. A reading decides whether work goes on, back
/// to a reviewer, or out of the record; it is held near-deterministic.
const READING_TEMPERATURE: f32 = 0.2;

/// How often the loop looks at the table.
const TICK: Duration = Duration::from_secs(20);

/// What one generation came to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Generation {
    /// An operation is open, its draft mission on the table.
    Offered {
        operation: u64,
        name: String,
        generator: String,
        target: String,
        brief: String,
        writes: String,
        reads: Vec<String>,
        attempts: usize,
    },
    /// The generator looked and found nothing to do.
    Declined {
        generator: String,
        target: String,
        why: String,
    },
    /// No generator found a target the ledger does not block.
    NoWork,
    /// Every answer was refused; the target is set aside until it changes.
    Failed {
        generator: String,
        target: String,
        why: String,
        /// The last answer as the model wrote it, for whoever tunes the prompt.
        last: String,
    },
}

/// Run one generation for `hosted`. `only` names the generator to use; `None`
/// draws by weight.
pub async fn generate(
    rt: &Runtime,
    hosted: &Hosted,
    config: &Config,
    only: Option<&str>,
) -> anyhow::Result<Generation> {
    let mind = rt
        .mind
        .clone()
        .ok_or_else(|| anyhow::anyhow!("this daemon has no mind to generate from"))?;
    let minds = rt
        .minds
        .read()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("the engine is not loaded yet"))?;
    let corpus = Corpus::read(&mind, hosted.id());
    let turn = hosted.with_sim(|s| s.missions.next_draw());
    let order: Vec<&Generator> = match only {
        Some(id) => vec![config
            .generator(id)
            .ok_or_else(|| anyhow::anyhow!("no generator called `{id}`"))?],
        None => config.order(turn),
    };
    // **Chosen and held in one step.** The loop and an operator's request can
    // generate at the same moment, and both chose the same target when the
    // choice was only recorded once the model had answered.
    let Some((generator, target)) = order.into_iter().find_map(|g| {
        hosted.with_sim(|s| {
            let t = target::next(g.kind, &corpus, &|key, fp| s.missions.blocks(key, fp), turn)?;
            s.missions
                .reserve(&t.key, &g.id, t.fingerprint)
                .then_some((g, t))
        })
    }) else {
        return Ok(Generation::NoWork);
    };
    let written = write_up(hosted, config, &minds, &corpus, generator, &target).await;
    if written.is_err() {
        hosted.with_sim(|s| s.missions.release(&target.key));
    }
    written
}

/// Ask for the mission `target` needs, check it, and settle the target with
/// what came of it.
async fn write_up(
    hosted: &Hosted,
    config: &Config,
    minds: &Minds,
    corpus: &Corpus,
    generator: &Generator,
    target: &Target,
) -> anyhow::Result<Generation> {
    // **A target with nothing to show is set aside, not released.** Released, it
    // stays first in line — a review queued for a life whose character has no
    // personality file was chosen again on every draw, and nothing behind it was
    // ever reached.
    let Some(material) = material::render(generator.kind, target, corpus) else {
        let why = format!("{} has nothing to show for {}", generator.id, target.key);
        decline(hosted, generator, target, &why);
        tracing::warn!(generator = %generator.id, target = %target.key, "mission generator: no material, target set aside");
        return Ok(Generation::Declined {
            generator: generator.id.clone(),
            target: target.key.clone(),
            why,
        });
    };
    let asked = format!(
        "# The record\n\n{material}\n\n# What you are asked\n\n{}",
        generator.prompt.trim()
    );
    tracing::info!(generator = %generator.id, target = %target.key, "mission generator: asking");

    let (engine, base) = (minds.engine(), minds.base_config());
    let mut prompt = asked.clone();
    let mut last_fault = String::new();
    let mut last_raw = String::new();
    for attempt in 1..=ATTEMPTS {
        let calls = answer::specs(generator.kind);
        let ask = CallAsk {
            system: &config.system,
            prompt: &prompt,
            calls: &calls,
            max_tokens: ANSWER_TOKENS,
            temperature: Some(TEMPERATURE),
            seed: resolve_seed(None),
        };
        let raw = decode_call(&engine, &base, &ask).await?;
        let checked = match answer::check(&raw, generator.kind, target, corpus) {
            // **A correction is put to a second reading before it is set.** The
            // generator found "the rank system solidified" against "the Duke
            // system consolidated" and called it a contradiction; asked plainly
            // whether both could be true, a reader says yes.
            Ok(Answer::Mission(p)) if generator.kind == Kind::Contradiction => {
                match contradict(&engine, &base, &raw).await? {
                    true => Ok(Answer::Mission(p)),
                    false => Ok(Answer::Nothing(
                        "On a second reading the two statements do not contradict each other, so \
                         there is nothing to correct."
                            .into(),
                    )),
                }
            }
            other => other,
        };
        match checked {
            Ok(Answer::Mission(p)) => {
                let desk = desk_for(hosted, &p.writes);
                let mission = through_time(
                    hosted,
                    answer::mission(&p, &generator.id, target, desk.as_ref()),
                    canon::set_in(&target.key, &p.writes, corpus),
                );
                // What its document says now, when the record holds it — put
                // back if the operation fails (`Sim::set_aside_failed`).
                let before = corpus.on_disk(&p.writes);
                let (operation, name) = hosted.with_sim(|s| {
                    let id = s.missions.launch(
                        mission,
                        target.fingerprint,
                        &p.objective,
                        before.as_deref(),
                    );
                    let name = s.missions.operations().get(id).map(|o| o.name.clone());
                    (id, name.unwrap_or_default())
                });
                tracing::info!(
                    generator = %generator.id, target = %target.key, writes = %p.writes, attempt,
                    %name, "mission generator: an operation is open — {}", p.objective
                );
                return Ok(Generation::Offered {
                    operation,
                    name,
                    generator: generator.id.clone(),
                    target: target.key.clone(),
                    brief: p.brief,
                    writes: p.writes,
                    reads: p.reads,
                    attempts: attempt,
                });
            }
            Ok(Answer::Nothing(why)) => {
                decline(hosted, generator, target, &why);
                tracing::info!(generator = %generator.id, target = %target.key, %why, "mission generator: nothing to do");
                return Ok(Generation::Declined {
                    generator: generator.id.clone(),
                    target: target.key.clone(),
                    why,
                });
            }
            Err(fault) => {
                tracing::info!(generator = %generator.id, target = %target.key, attempt, %fault, "mission generator: answer refused");
                prompt = format!(
                    "{asked}\n\n## Your last answer was refused\n\n{fault}\n\nIt was:\n\n{}\n\nAnswer \
                     again.",
                    raw.trim()
                );
                last_fault = fault;
                last_raw = raw;
            }
        }
    }
    let why = format!("no acceptable answer in {ATTEMPTS} tries — last: {last_fault}");
    decline(hosted, generator, target, &why);
    tracing::warn!(generator = %generator.id, target = %target.key, %why, "mission generator: target set aside");
    Ok(Generation::Failed {
        generator: generator.id.clone(),
        target: target.key.clone(),
        why,
        last: last_raw,
    })
}

/// The judge a correction is put to: do the two quoted statements contradict
/// each other? One token, `yes` or `no`, held by a grammar.
///
/// **Asked so that the doubtful answer declines.** Put as "can both be true?",
/// a no-leaning reading kept "the last Keeper instance was lost in 2534" against
/// "the Portal Retreat began in 2537" as a contradiction to fix; put as "do they
/// contradict?", the same leaning declines it. A false correction costs a Maker
/// an afternoon spent breaking a page that was right; a missed one is found
/// again when either document changes.
const JUDGE_SYSTEM: &str = "You are a careful reader of a fictional world's record. You judge \
                            whether two statements contradict each other. Two statements \
                            contradict only when they cannot both be true: different emphasis, \
                            one adding detail the other lacks, or two events at two different \
                            times do not contradict.";

/// Whether a correction's two quotes contradict each other, by a second
/// reading.
async fn contradict(
    engine: &Arc<Mutex<ConversationEngine>>,
    base: &SequenceConfig,
    raw: &str,
) -> anyhow::Result<bool> {
    let (_, args) = arguments(raw).unwrap_or_default();
    let quote = |k: &str| {
        args.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    };
    let request = Request {
        system: JUDGE_SYSTEM.into(),
        prompt: format!(
            "Statement A: \"{}\"\n\nStatement B: \"{}\"\n\nDo these two statements contradict \
             each other — is it impossible for both to be true? Answer yes or no.",
            quote("quote_a"),
            quote("quote_b")
        ),
        max_tokens: 4,
        temperature: Some(0.0),
        seed: Some(1),
        choices: Some(vec!["yes".into(), "no".into()]),
    };
    let answer = prose::decode(engine, base, &request, 1, &mut |_: &str| {}).await?;
    Ok(answer.text.trim().eq_ignore_ascii_case("yes"))
}

fn decline(hosted: &Hosted, generator: &Generator, target: &Target, why: &str) {
    hosted.with_sim(|s| {
        s.missions
            .decline(&target.key, &generator.id, target.fingerprint, why)
    });
}

/// The part a document is written at, by where it lives in the mind.
///
/// The vault's own division of labour: lives and personalities on the casting
/// level, stories at the story desks, places at the map table, and everything
/// of the world's history at a chronicle terminal.
pub fn bench_for(path: &str) -> &'static str {
    match path {
        p if p.starts_with("layers/life/") || p.starts_with("personalities/") => {
            "character-terminal"
        }
        p if p.starts_with("layers/stories/") => "story-desk",
        p if p.starts_with("map/") => "map-table",
        _ => "chronicle-terminal",
    }
}

/// The first room in the world holding the bench `path` is written at, and the
/// level it is on.
fn desk_for(hosted: &Hosted, path: &str) -> Option<Desk> {
    room_with(hosted, bench_for(path))
}

/// The same mission, sent first to a time machine to work in `year` — when it
/// has a year and the world has a time machine.
///
/// **What happened later is not recalled while the past is written.** A Maker
/// writing a life in 2950 was handed eras and stories from the following century
/// by its own recall; standing in 2950 at a time machine, nothing after it
/// reaches it (see `bearings`).
fn through_time(hosted: &Hosted, mut mission: Mission, year: Option<u32>) -> Mission {
    let (Some(year), Some(machine)) = (year, room_with(hosted, "time-machine")) else {
        return mission;
    };
    mission.todo.splice(
        0..0,
        [
            Todo::new(format!("go to {} on {}", machine.room, machine.level)),
            Todo::new(time_step_text(year)),
        ],
    );
    mission
}

/// The first room in the world holding `part`, and the level it is on.
fn room_with(hosted: &Hosted, part: &str) -> Option<Desk> {
    hosted.read(|w| {
        w.map().areas().find_map(|area| {
            let node = area
                .nodes
                .iter()
                .find(|n| w.map().part_ids_at(n).contains(&part))?;
            Some(Desk {
                room: node.name.clone(),
                level: area.name.clone(),
            })
        })
    })
}

/// What the table is asked when it reads a draft: the record it answers to,
/// what it was to tell when its brief says, and the configured question.
///
/// **What it was to tell, beside what it tells.** Read against the lore
/// alone, a draft that tells nothing reads as one that tells it quietly; the
/// brief's own event is what it answers to first.
fn reading_prompt(material: &str, brief: &str, question: &str) -> String {
    let told = match what_happens(brief) {
        Some(h) => format!("# What it was to tell\n\n{h}\n\n"),
        None => String::new(),
    };
    format!(
        "# The record\n\n{material}\n\n{told}# What you are asked\n\n{}",
        question.trim()
    )
}

/// One reading of `op`'s draft by the table, asked up to [`ATTEMPTS`] times
/// until it is an acceptable one — `None` when none was.
async fn table_reading(
    engine: &Arc<Mutex<ConversationEngine>>,
    base: &SequenceConfig,
    config: &Config,
    asked: &str,
    op: &Operation,
    corpus: &Corpus,
) -> anyhow::Result<Option<reading::Reading>> {
    let mut prompt = asked.to_string();
    for attempt in 1..=ATTEMPTS {
        let calls = reading::specs();
        let ask = CallAsk {
            system: &config.system,
            prompt: &prompt,
            calls: &calls,
            max_tokens: READING_TOKENS,
            temperature: Some(READING_TEMPERATURE),
            seed: resolve_seed(None),
        };
        let raw = decode_call(engine, base, &ask).await?;
        match reading::check(&raw, &op.document, corpus, attempt < ATTEMPTS) {
            Ok(r) => return Ok(Some(r)),
            Err(fault) => {
                tracing::info!(operation = %op.name, attempt, %fault, "operation: reading refused");
                if attempt == ATTEMPTS {
                    // What it actually said, for whoever tunes the prompt or
                    // the check: the fault alone does not say which was wrong.
                    tracing::warn!(operation = %op.name, answer = %raw.trim(), "operation: no acceptable reading");
                }
                prompt = format!(
                    "{asked}\n\n## Your last answer was refused\n\n{fault}\n\nIt was:\n\n{}\n\n\
                     Answer again.",
                    raw.trim()
                );
            }
        }
    }
    Ok(None)
}

/// Read operation `id`'s draft and put its review on the table.
///
/// **A reading that will not come is not a review withheld.** When the table
/// cannot give an acceptable reading in [`ATTEMPTS`], the review is still set,
/// and the reviewer is told the table had nothing to add: the second Maker's
/// reading is the gate that matters.
pub async fn read_draft(
    rt: &Runtime,
    hosted: &Hosted,
    config: &Config,
    id: u64,
) -> anyhow::Result<()> {
    let mind = rt
        .mind
        .clone()
        .ok_or_else(|| anyhow::anyhow!("this daemon has no mind to read from"))?;
    let minds = rt
        .minds
        .read()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("the engine is not loaded yet"))?;
    let Some(op) = hosted.sim(|s| s.missions.operations().get(id).cloned()) else {
        return Ok(());
    };
    // The form put right before the table reads it, so the reading is of what
    // the draft says rather than of a heading line.
    if gates::tidy_on_disk(&mind, &op.document) {
        hosted.with_sim(|s| s.bench.rebase(&op.document));
    }
    let corpus = Corpus::read(&mind, hosted.id());
    let Some(material) = material::draft(&corpus, &op.document) else {
        hosted.with_sim(|s| {
            s.missions
                .cancel_operation(id, "its draft is no longer on the record")
        });
        tracing::warn!(operation = %op.name, document = %op.document, "operation: draft gone, called off");
        return Ok(());
    };
    let asked = reading_prompt(&material, &op.brief, &config.reading);
    let (engine, base) = (minds.engine(), minds.base_config());
    let mut reading = table_reading(&engine, &base, config, &asked, &op, &corpus).await?;
    // **A draft stands on two sound readings, not one.** A single reading
    // found a life event sound in which a door was opened on an empty room
    // and shut again — "while quiet, it has a clear beginning and end" — and
    // a second reader is how a judge that talked itself into a verdict is
    // caught. A second reading that finds faults is the one the review gets.
    if reading
        .as_ref()
        .is_some_and(|r| r.verdict == reading::Verdict::Sound)
    {
        let again = table_reading(&engine, &base, config, &asked, &op, &corpus).await?;
        if let Some(second) = again.filter(|r| r.verdict != reading::Verdict::Sound) {
            tracing::info!(operation = %op.name, verdict = ?second.verdict, "operation: the second reading disagreed");
            reading = Some(second);
        }
    }
    let rendered = match &reading {
        Some(r) => r.render(),
        None => "The table could not read it; read it wholly for yourself.".to_string(),
    };
    let desk = desk_for(hosted, &op.document);
    let sound = reading
        .as_ref()
        .is_some_and(|r| r.verdict == reading::Verdict::Sound);
    let review = through_time(
        hosted,
        reading::review_mission(
            &op,
            &rendered,
            reading.as_ref().map(|r| r.verdict),
            &corpus,
            desk.as_ref(),
        ),
        canon::set_in(&op.target, &op.document, &corpus),
    );
    let next = hosted.with_sim(|s| s.missions.offer_review(id, review, &rendered, sound));
    // Read again after a review and still found wanting, past the limit: the
    // operation has failed, and its document is settled with every other
    // failed one's — see `spawn`.
    if next == AfterReading::Failed {
        tracing::info!(operation = %op.name, "operation FAILED on the table's reading");
    }
    tracing::info!(
        operation = %op.name, verdict = ?reading.as_ref().map(|r| r.verdict), ?next,
        "operation: read"
    );
    Ok(())
}

/// Keep every hosted world's table stocked while the daemon runs.
///
/// **Only while the table is open.** A shut table hands out nothing, so a
/// mission written then would only wait — and the corpus it was written from
/// may have moved on by the time anybody collects it.
pub fn spawn(rt: Arc<Runtime>) {
    tokio::spawn(async move {
        let mut told: Option<String> = None;
        loop {
            tokio::time::sleep(TICK).await;
            if rt.stopping() {
                return;
            }
            let Some(mind) = rt.mind.clone() else {
                return;
            };
            let config = match Config::load(&mind) {
                Ok(Some(c)) => c,
                Ok(None) => continue,
                Err(e) => {
                    if told.as_deref() != Some(e.as_str()) {
                        tracing::warn!("mission generator: {e}");
                        told = Some(e);
                    }
                    continue;
                }
            };
            told = None;
            for id in rt.hosted.ids() {
                let Some(hosted) = rt.hosted.get(&id) else {
                    continue;
                };
                // **Every failed operation's document is settled here**, however
                // it failed — rejected, read and found wanting past the limit,
                // or stuck: a draft is moved out of the record and leaves
                // memory now, not at the next boot (the substrate holds
                // whatever was ingested of it, and a gather could otherwise
                // still surface lore the review threw out); a correction's
                // document is put back (`Sim::set_aside_failed`).
                let retiring = hosted.sim(|s| s.missions.operations().to_retire());
                // Taken out of the lock first, so the read guard is not held
                // across the scans; each scan holds the engine, and runs off
                // the async workers so it does not stall them while it does.
                let minds = rt.minds.read().unwrap().clone();
                if let (false, Some(minds)) = (retiring.is_empty(), minds) {
                    for op in retiring {
                        let doc = op.document.clone();
                        let set_aside = hosted.with_sim(|s| s.set_aside_failed(&op));
                        tracing::info!(operation = %op.name, %doc, ?set_aside, "operation: failed document settled");
                        let scan = match op.leaves_on_failure() {
                            true => {
                                let engine = minds.engine();
                                let rel = doc.clone();
                                spawn_blocking(move || {
                                    mind_record::retire(&engine.lock().unwrap(), &rel)
                                })
                                .await
                            }
                            false => Ok(0),
                        };
                        // Marked only once it has run, so a scan that failed is
                        // tried again on the next pass.
                        match scan {
                            Ok(gone) => {
                                hosted.with_sim(|s| s.missions.mark_retired(op.id));
                                tracing::info!(%doc, gone, "operation: failed draft retired from memory");
                            }
                            Err(e) => {
                                tracing::warn!(%doc, "operation: retiring the draft failed: {e}")
                            }
                        }
                    }
                }
                // Drafts waiting for the table's reading come first: an
                // operation half done is worth more than a new one opened.
                if hosted.sim(|s| s.table_open && s.bench.has_root()) {
                    for op in hosted.sim(|s| s.missions.operations().awaiting_reading()) {
                        if rt.stopping() {
                            return;
                        }
                        if let Err(e) = read_draft(&rt, &hosted, &config, op).await {
                            tracing::warn!("operation reading: {e:#}");
                            break;
                        }
                    }
                    // Reviewed lore goes to its check against the storyline.
                    // Nothing to decode: the engine knows which eras to set.
                    let reviewed = hosted.sim(|s| s.missions.operations().awaiting_canon());
                    if !reviewed.is_empty() {
                        let corpus = Corpus::read(&mind, hosted.id());
                        for id in reviewed {
                            let Some(op) = hosted.sim(|s| s.missions.operations().get(id).cloned())
                            else {
                                continue;
                            };
                            let desk = desk_for(&hosted, &op.document);
                            let check = through_time(
                                &hosted,
                                canon::canon_mission(&op, &corpus, desk.as_ref()),
                                canon::set_in(&op.target, &op.document, &corpus),
                            );
                            hosted.with_sim(|s| s.missions.offer_check(id, check));
                            tracing::info!(operation = %op.name, "operation: canon check on the table");
                        }
                    }
                }
                // Work waiting for a Maker this world does not have is called
                // off, so it neither blocks its target nor counts as stock.
                let makers: Vec<String> = rt
                    .bodies
                    .in_world(hosted.id())
                    .into_iter()
                    .map(|(_, body)| body)
                    .collect();
                for op in hosted.with_sim(|s| s.missions.call_off_untakeable(&makers)) {
                    tracing::info!(
                        operation = op,
                        "operation: no Maker left who may take it, called off"
                    );
                }
                while hosted.sim(|s| s.table_open && s.bench.has_root())
                    && hosted.sim(|s| s.missions.pooled().len()) < config.keep
                    && !rt.stopping()
                {
                    match generate(&rt, &hosted, &config, None).await {
                        Ok(Generation::NoWork) => break,
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!("mission generator: {e:#}");
                            break;
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{bench_for, reading_prompt};

    #[test]
    fn the_table_reads_a_draft_beside_what_it_was_to_tell() {
        assert_eq!(
            reading_prompt(
                "the draft",
                "Write it.\n\nWhat happens: Keeper orders the retreat.\n\nGo.",
                " Read it. "
            ),
            "# The record\n\nthe draft\n\n# What it was to tell\n\nKeeper orders the \
             retreat.\n\n# What you are asked\n\nRead it."
        );
        assert_eq!(
            reading_prompt("the draft", "", "Read it."),
            "# The record\n\nthe draft\n\n# What you are asked\n\nRead it.",
            "a draft put through review by hand has no brief"
        );
    }

    #[test]
    fn each_kind_of_document_is_written_at_its_own_bench() {
        assert_eq!(
            bench_for("layers/life/keeper/2488 X.md"),
            "character-terminal"
        );
        assert_eq!(bench_for("layers/stories/x.md"), "story-desk");
        assert_eq!(bench_for("layers/eras/x.md"), "chronicle-terminal");
        assert_eq!(bench_for("layers/world/combat.md"), "chronicle-terminal");
        assert_eq!(bench_for("map/battle-cities.yaml"), "map-table");
    }
}
