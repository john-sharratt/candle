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
use candle_conversation::stencil::ThinkMode;
use candle_conversation::{ConversationEngine, SequenceConfig};
use serde::Serialize;
use tokio::task::spawn_blocking;

use super::answer::{self, Answer, Desk};
use super::canon;
use super::config::{Config, Generator, READING};
use super::corpus::Corpus;
use super::gates;
use super::material;
use super::reading;
use super::step;
use super::target::{self, Kind, Target};
use crate::engine::journal::tools::arguments;
use crate::engine::mind::Minds;
use crate::engine::mind_record;
use crate::engine::mission::{time_step_text, Mission, Todo};
use crate::engine::prose::{self, decode_call, CallAsk};
use crate::engine::runtime::Runtime;
use crate::engine::work::what_happens;
use crate::engine::workflow::{OnFailed, Where};
use crate::prose::Request;
use crate::sim::operations::Operation;
use crate::world::Hosted;

/// The most tokens one answer may run to. The table's reading and its faults
/// run to a thousand words between them, and an answer cut off by the cap is not a
/// call at all — a life event was refused "not a call" three times running.
const ANSWER_TOKENS: usize = 2400;

/// The most tokens the table's reading may run to, past its thinking.
pub(super) const READING_TOKENS: usize = 2400;

/// How many times an answer is asked for before the target is given up.
const ATTEMPTS: usize = 3;

/// How much a mission and the table's reading are thought through before the
/// call is written — see [`crate::engine::prose::decode_call`].
///
/// **Both decode on the checkpoint's own sampling.** Held below its
/// temperature — 0.5 for a mission, 0.2 for a reading — the model went round:
/// a reading at 0.2 wrote "the draft's voice is not Keeper's voice" until the
/// cap, three tries running, the near-greedy loop the checkpoint's own card
/// warns against. A reading's verdict is read twice before a draft stands on it
/// (`table_step`), which is what holding the table near-deterministic was for.
pub(super) const THINK: ThinkMode = ThinkMode::Balanced;

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
    let Some(material) = material::render(generator.kind, target, corpus, &generator.context)
    else {
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
    let mut faults: Vec<String> = Vec::new();
    let mut last_raw = String::new();
    for attempt in 1..=ATTEMPTS {
        let calls = answer::specs(generator.kind);
        let ask = CallAsk {
            system: &config.system,
            prompt: &prompt,
            calls: &calls,
            max_tokens: ANSWER_TOKENS,
            temperature: None,
            seed: resolve_seed(None),
            think: THINK,
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
                // What its document says now, when the record holds it — put
                // back if the operation fails and its workflow says `restore`
                // (`Sim::set_aside_failed`).
                let before = corpus.on_disk(&p.writes);
                let opened = hosted.with_sim(|s| {
                    let id = s.missions.launch(
                        &generator.workflow,
                        &generator.id,
                        &target.key,
                        target.fingerprint,
                        &p.objective,
                        &p.writes,
                        &p.brief,
                        p.fields.clone(),
                        p.reads.clone(),
                        before.as_deref(),
                    )?;
                    let name = s.missions.operations().get(id).map(|o| o.name.clone());
                    Ok::<_, String>((id, name.unwrap_or_default()))
                });
                let (operation, name) = opened.map_err(|e| anyhow::anyhow!("{e}"))?;
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
                // How the answer opened and closed, so a refusal can be told
                // from a decode that never produced a call.
                let (head, tail) = ends(&raw, 160);
                tracing::info!(generator = %generator.id, target = %target.key, attempt, %fault, %head, %tail, "mission generator: answer refused");
                faults.push(fault);
                prompt = retry_prompt(&asked, &faults);
                last_raw = raw;
            }
        }
    }
    let why = format!(
        "no acceptable answer in {ATTEMPTS} tries — last: {}",
        faults.last().map_or("", String::as_str)
    );
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

/// The first and last `chars` characters of `raw`, trimmed — all of it in
/// `head` when it is short.
fn ends(raw: &str, chars: usize) -> (String, String) {
    let raw = raw.trim();
    let n = raw.chars().count();
    if n <= chars * 2 {
        return (raw.to_string(), String::new());
    }
    let head: String = raw.chars().take(chars).collect();
    let tail: String = raw.chars().skip(n - chars).collect();
    (head, tail)
}

fn decline(hosted: &Hosted, generator: &Generator, target: &Target, why: &str) {
    hosted.with_sim(|s| {
        s.missions
            .decline(&target.key, &generator.id, target.fingerprint, why)
    });
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
/// what it was to tell when that is known, and the step's question.
///
/// **What it was to tell, beside what it tells.** Read against the lore
/// alone, a draft that tells nothing reads as one that tells it quietly; the
/// brief's own event is what it answers to first.
pub(super) fn reading_prompt(material: &str, told: Option<&str>, question: &str) -> String {
    let told = match told {
        Some(h) => format!("# What it was to tell\n\n{h}\n\n"),
        None => String::new(),
    };
    format!(
        "# The record\n\n{material}\n\n{told}# What you are asked\n\n{}",
        question.trim()
    )
}

/// What a call is asked again after `faults` refused its answers: the ask and
/// every fault found so far — not the refused answer itself.
///
/// **Every refusal, not only the last.** Told only why its last answer was
/// refused, a story proposal refused for reusing other stories' names was
/// refused next for its turn — and on its third try took back the very names
/// the first refusal had ruled out.
///
/// **The refused answer is not shown.** An answer in front of it to copy is
/// what it copied: a reading refused for a 1165-word `checked` sent back the
/// same 1165 words, and a story refused for "Kess, Jorik" sent back Kess and
/// Jorik twice, with the refusal naming them right above. Each fault quotes
/// what it is about; the answer is written fresh against them.
fn retry_prompt(asked: &str, faults: &[String]) -> String {
    let said: Vec<String> = faults
        .iter()
        .enumerate()
        .map(|(i, f)| format!("{}. {}", i + 1, f.trim()))
        .collect();
    format!(
        "{asked}\n\n## Your answers so far were refused\n\nEvery one of these still holds — \
         write a new answer, from the start, so that none of them is true of it:\n\n{}\n\n\
         Answer again.",
        said.join("\n")
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
    let mut faults: Vec<String> = Vec::new();
    for attempt in 1..=ATTEMPTS {
        let calls = reading::specs();
        let ask = CallAsk {
            system: &config.reader,
            prompt: &prompt,
            calls: &calls,
            max_tokens: READING_TOKENS,
            temperature: None,
            seed: resolve_seed(None),
            think: THINK,
        };
        let raw = decode_call(engine, base, &ask).await?;
        match reading::check(&raw, &op.document, corpus, attempt < ATTEMPTS) {
            Ok(r) => return Ok(Some(r)),
            Err(fault) => {
                let (head, tail) = ends(&raw, 160);
                tracing::info!(operation = %op.name, attempt, %fault, %head, %tail, "operation: reading refused");
                if attempt == ATTEMPTS {
                    // What it actually said, for whoever tunes the prompt or
                    // the check: the fault alone does not say which was wrong.
                    tracing::warn!(operation = %op.name, answer = %raw.trim(), "operation: no acceptable reading");
                }
                faults.push(fault);
                prompt = retry_prompt(asked, &faults);
            }
        }
    }
    Ok(None)
}

/// The outcome of a reading the table could not give in [`ATTEMPTS`].
pub const UNREAD: &str = "unread";

/// The findings of a reading the table could not give.
const NOT_READ: &str = "The table could not read it; read it wholly for yourself.";

/// Take the table step operation `id` waits on, running its `call`, and move
/// the operation on by the call's verdict.
///
/// **A reading that will not come is an outcome, not a stall.** When the table
/// cannot give an acceptable reading in [`ATTEMPTS`], the step is taken
/// [`UNREAD`], and the workflow says where that leads — in npcd's, to a review
/// told the table had nothing to add.
pub async fn table_step(
    rt: &Runtime,
    hosted: &Hosted,
    config: &Config,
    id: u64,
    call: &str,
) -> anyhow::Result<()> {
    if call != READING {
        anyhow::bail!("the table has no call `{call}`");
    }
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
    let Some((op, prompt, context)) = hosted.sim(|s| {
        let ops = s.missions.operations();
        let offer = ops.offer(id).ok()?;
        Some((
            ops.get(id)?.clone(),
            offer.prompt.to_string(),
            offer.step.context.clone(),
        ))
    }) else {
        return Ok(());
    };
    // The form put right before the table reads it, so the reading is of what
    // the draft says rather than of a heading line.
    if gates::tidy_on_disk(&mind, &op.document) {
        hosted.with_sim(|s| s.bench.rebase(&op.document));
    }
    let corpus = Corpus::read(&mind, hosted.id());
    let Some(material) = material::draft(
        &corpus,
        &op.document,
        op.target.strip_prefix("era:"),
        &context,
    ) else {
        hosted.with_sim(|s| {
            s.missions
                .cancel_operation(id, "its document is no longer on the record")
        });
        tracing::warn!(operation = %op.name, document = %op.document, "operation: document gone, called off");
        return Ok(());
    };
    let told = match context.iter().any(|c| c == "told") {
        true => op
            .fields
            .get("happens")
            .cloned()
            .or_else(|| what_happens(&op.brief)),
        false => None,
    };
    let asked = reading_prompt(&material, told.as_deref(), &prompt);
    let (engine, base) = (minds.engine(), minds.base_config());
    let mut reading = table_reading(&engine, &base, config, &asked, &op, &corpus).await?;
    // **A draft stands on two sound readings, not one.** A single reading
    // found a life event sound in which a door was opened on an empty room
    // and shut again — "while quiet, it has a clear beginning and end" — and
    // a second reader is how a judge that talked itself into a verdict is
    // caught. A second reading that finds faults is the one that counts.
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
    let (outcome, found) = match &reading {
        Some(r) => (r.verdict.name(), r.render()),
        None => (UNREAD, NOT_READ.to_string()),
    };
    let at = hosted.with_sim(|s| s.missions.table_took(id, outcome, &found));
    match at {
        Ok(at) => {
            tracing::info!(operation = %op.name, verdict = outcome, ?at, "operation: read");
            if matches!(at, Where::Failed(_)) {
                tracing::info!(operation = %op.name, "operation FAILED on the table's reading");
            }
        }
        Err(e) => {
            tracing::warn!(operation = %op.name, "operation: the workflow refused the reading: {e}")
        }
    }
    Ok(())
}

/// Put the Maker's step operation `id` waits on at the table, as a mission
/// built from the step ([`step::step_mission`]), at its workflow's desk and in
/// its workflow's year.
fn offer_step(hosted: &Hosted, corpus: &Corpus, id: u64) -> Result<(), String> {
    let (desk, year) = hosted
        .sim(|s| {
            let ops = s.missions.operations();
            let wf = ops.workflow_of(ops.get(id)?)?;
            Some((wf.desk.clone(), wf.year.is_some()))
        })
        .ok_or("no such operation, or its workflow is not loaded")?;
    let desk = desk.and_then(|d| room_with(hosted, &d));
    let (op, mission) = hosted.sim(|s| {
        let ops = s.missions.operations();
        let op = ops.get(id).ok_or("no such operation")?.clone();
        let offer = ops.offer(id)?;
        let mission = step::step_mission(&op, &offer, corpus, desk.as_ref())?;
        Ok::<_, String>((op, mission))
    })?;
    let year = year
        .then(|| canon::set_in(&op.target, &op.document, corpus))
        .flatten();
    let mission = through_time(hosted, mission, year);
    let step = op.step().unwrap_or_default().to_string();
    hosted.with_sim(|s| s.missions.offer_step(id, mission));
    tracing::info!(operation = %op.name, %step, desk = ?desk.map(|d| d.room), "operation: step on the table");
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
                // The workflows as the mind's file has them now: an edit to
                // `missions.yaml` applies to every operation's next step.
                hosted.with_sim(|s| s.missions.set_workflows(config.workflows.clone()));
                // **Every failed operation's document is settled here**, however
                // it failed — rejected, read and found wanting past the limit,
                // or stuck — as its workflow's `on-failed` says: a draft set
                // aside leaves the record and memory now, not at the next boot
                // (the substrate holds whatever was ingested of it, and a
                // gather could otherwise still surface lore the review threw
                // out); a correction's document is put back
                // (`Sim::set_aside_failed`).
                let retiring = hosted.sim(|s| {
                    let ops = s.missions.operations();
                    ops.to_retire()
                        .into_iter()
                        .map(|op| {
                            let leaves = ops.on_failed(&op) == OnFailed::SetAside;
                            (op, leaves)
                        })
                        .collect::<Vec<_>>()
                });
                // Taken out of the lock first, so the read guard is not held
                // across the scans; each scan holds the engine, and runs off
                // the async workers so it does not stall them while it does.
                let minds = rt.minds.read().unwrap().clone();
                if let (false, Some(minds)) = (retiring.is_empty(), minds) {
                    for (op, leaves) in retiring {
                        let doc = op.document.clone();
                        let set_aside = hosted.with_sim(|s| s.set_aside_failed(&op));
                        tracing::info!(operation = %op.name, %doc, ?set_aside, "operation: failed document settled");
                        let scan = match leaves {
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
                // A document can leave the record under a mission written to
                // read it — a failed draft set aside above, or by any other
                // hand — and a read of nothing is a step no Maker can finish.
                let struck = hosted.with_sim(|s| {
                    let root = s.bench.mind_root()?.to_path_buf();
                    Some(s.missions.strike_gone_reads(&|p| root.join(p).is_file()))
                });
                for (whose, gone) in struck.unwrap_or_default() {
                    tracing::info!(body = ?whose, ?gone, "mission: reads of documents gone from the record struck");
                }
                // Steps of operations already open come first: an operation
                // half done is worth more than a new one opened. The table's
                // own steps are taken here; a Maker's is put on the table.
                if hosted.sim(|s| s.table_open && s.bench.has_root()) {
                    for (op, call) in hosted.sim(|s| s.missions.operations().awaiting_table()) {
                        if rt.stopping() {
                            return;
                        }
                        if let Err(e) = table_step(&rt, &hosted, &config, op, &call).await {
                            tracing::warn!("operation: the table's step: {e:#}");
                            break;
                        }
                    }
                    let owed = hosted.sim(|s| s.missions.operations().awaiting_offer());
                    if !owed.is_empty() {
                        let corpus = Corpus::read(&mind, hosted.id());
                        for op in owed {
                            // A step that cannot be written up never will be:
                            // called off, so it does not hold its target.
                            if let Err(e) = offer_step(&hosted, &corpus, op) {
                                let why = format!("its step could not be put on the table: {e}");
                                hosted.with_sim(|s| s.missions.cancel_operation(op, &why));
                                tracing::warn!(operation = op, "operation: called off — {why}");
                            }
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
                    && !hosted.sim(|s| s.missions.stocked_for(&makers, config.keep))
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
    use super::{ends, reading_prompt, retry_prompt};

    /// **An answer asked again carries every refusal so far**, numbered, and
    /// none of the refused answers to copy back.
    #[test]
    fn a_retry_carries_every_refusal_so_far() {
        let faults = vec!["Names reused.".to_string(), " Not an act. ".to_string()];
        assert_eq!(
            retry_prompt("Ask.", &faults),
            "Ask.\n\n## Your answers so far were refused\n\nEvery one of these still holds — \
             write a new answer, from the start, so that none of them is true of it:\n\n1. \
             Names reused.\n2. Not an act.\n\nAnswer again."
        );
    }

    #[test]
    fn a_refused_answer_is_logged_by_its_two_ends() {
        assert_eq!(ends("  short  ", 4), ("short".to_string(), String::new()));
        assert_eq!(
            ends("abcdefghijkl", 3),
            ("abc".to_string(), "jkl".to_string())
        );
    }

    #[test]
    fn the_table_reads_a_draft_beside_what_it_was_to_tell() {
        assert_eq!(
            reading_prompt(
                "the draft",
                Some("Keeper orders the retreat."),
                " Read it. "
            ),
            "# The record\n\nthe draft\n\n# What it was to tell\n\nKeeper orders the \
             retreat.\n\n# What you are asked\n\nRead it."
        );
        assert_eq!(
            reading_prompt("the draft", None, "Read it."),
            "# The record\n\nthe draft\n\n# What you are asked\n\nRead it.",
            "a draft put through review by hand has no brief"
        );
    }
}
