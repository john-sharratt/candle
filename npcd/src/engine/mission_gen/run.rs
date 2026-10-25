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

use super::answer::{self, Answer, Desk};
use super::canon;
use super::config::{Config, Generator, READING};
use super::corpus::Corpus;
use super::gates;
use super::material;
use super::reading;
use super::step;
use super::target::{self, Kind, Target};
use crate::api::now_ms;
use crate::engine::mind::Minds;
use crate::engine::mission::{time_step_text, Mission, Todo};
use crate::engine::prose::{self, decode_call, CallAsk};
use crate::engine::runtime::Runtime;
use crate::engine::work::what_happens;
use crate::engine::workflow::{OnFailed, Where};
use crate::mind::files::MindFiles;
use crate::opstore::owner::Owner;
use crate::prose::Request;
use crate::sim::operations::Operation;
use crate::sim::shelf::Shelf;
use crate::world::Hosted;
use journal_core::tools::arguments;

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
        .mind_files()
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
                    )?;
                    let name = s.missions.operations().get(id).map(|o| o.name.clone());
                    Ok::<_, String>((id, name.unwrap_or_default()))
                });
                let (operation, name) = opened.map_err(|e| anyhow::anyhow!("{e}"))?;
                // Its store takes the mind as it stands now as its base (§10.3).
                if let Some(stores) = hosted.op_stores() {
                    stores.pin(&Owner::operation(hosted.id(), operation));
                }
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
        .mind_files()
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
    // The table reads the draft in the operation's store (§10.3). The form is
    // put right first, so the reading is of what the draft says rather than of
    // a heading line.
    let Some(shelf) = hosted.op_shelf(id) else {
        anyhow::bail!("this world has no operation stores");
    };
    if gates::tidy_on(&shelf, &op.document) {
        hosted.with_sim(|s| s.bench.rebase(&op.document, &shelf));
    }
    let held = shelf.read(&op.document);
    let corpus = Corpus::read(&mind, hosted.id());
    let material = held.and_then(|text| {
        let corpus = corpus.clone().holding(&op.document, text);
        let drafted = material::draft(
            &corpus,
            &op.document,
            op.target.strip_prefix("era:"),
            &context,
        )?;
        Some((corpus, drafted))
    });
    let Some((corpus, material)) = material else {
        hosted.with_sim(|s| {
            s.missions
                .cancel_operation(id, "its document is no longer in its store")
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
/// built from the step ([`step::step_mission`]), at the step's own desk or its
/// workflow's, and in its workflow's year.
fn offer_step(hosted: &Hosted, corpus: &Corpus, id: u64) -> Result<(), String> {
    let (desk, year) = hosted
        .sim(|s| {
            let ops = s.missions.operations();
            let op = ops.get(id)?;
            let wf = ops.workflow_of(op)?;
            let step = wf.step(op.step()?)?;
            let desk = step.desk.clone().or_else(|| wf.desk.clone());
            // A publish lands the work in the present record; it is written
            // in no year.
            Some((desk, wf.year.is_some() && !step.publishes()))
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

/// Put every Maker's step the operations of `hosted` owe on the table. A step
/// that cannot be written up never will be: its operation is called off, so it
/// does not hold its target.
fn offer_owed(hosted: &Hosted, mind: &Arc<MindFiles>) {
    let owed = hosted.sim(|s| s.missions.operations().awaiting_offer(now_ms()));
    if owed.is_empty() {
        return;
    }
    let corpus = Corpus::read(mind, hosted.id());
    for op in owed {
        if let Err(e) = offer_step(hosted, &corpus, op) {
            let why = format!("its step could not be put on the table: {e}");
            hosted.with_sim(|s| s.missions.cancel_operation(op, &why));
            tracing::warn!(operation = op, "operation: called off — {why}");
        }
    }
}

/// Settle every failed operation of `hosted` not yet settled, as its
/// workflow's `on-failed` says: its store discarded, or kept unpublished for
/// an operator to reopen (`docs/replicated_substrate.md` §10.3).
pub fn settle_failed(hosted: &Hosted) {
    let failed = hosted.sim(|s| {
        let ops = s.missions.operations();
        ops.to_retire()
            .into_iter()
            .map(|op| (op.id, op.name.clone(), ops.on_failed(&op)))
            .collect::<Vec<_>>()
    });
    for (id, name, on_failed) in failed {
        if on_failed == OnFailed::Discard {
            if let Some(stores) = hosted.op_stores() {
                stores.discard(&Owner::operation(hosted.id(), id));
            }
        }
        hosted.with_sim(|s| s.missions.mark_retired(id));
        tracing::info!(operation = %name, ?on_failed, "operation: failed work settled");
    }
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
            let Some(mind) = rt.mind_files() else {
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
                // `missions.yaml` applies to every operation's next step. Every
                // member reads them, for the worlds it follows as well as the
                // ones it runs — they are the mind's, not the world's.
                hosted.with_sim(|s| s.missions.set_workflows(config.workflows.clone()));
                // The table is the world's authority's to keep: a member that
                // only follows the world takes its missions from the records.
                if !rt.standing.role(&id).is_authority() {
                    continue;
                }
                // **Every failed operation's store is settled here**, however it
                // failed — rejected, read and found wanting past the limit, or
                // stuck — as its workflow's `on-failed` says: discarded, or
                // kept unpublished for an operator to reopen. Nothing of it
                // reached the record, so nothing there needs putting back.
                settle_failed(&hosted);
                // A document can leave the record under a mission written to
                // read it, and a read of nothing is a step no Maker can finish.
                // A step reads through its operation's store (§10.3), so its
                // own document counts there while it is a draft.
                let struck = hosted.sim(|s| s.bench.files());
                let struck = struck.map(|root| {
                    let holds = |op: Option<u64>, p: &str| {
                        root.is_file(p)
                            || op
                                .and_then(|id| hosted.op_shelf(id))
                                .is_some_and(|shelf| shelf.read(p).is_some())
                    };
                    hosted.with_sim(|s| s.missions.strike_gone_reads(&holds))
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
                    offer_owed(&hosted, &mind);
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
                // **One generation a tick.** Each can take minutes, and the open
                // operations' steps above come first: stocking the table in a
                // loop here held a draft's reading behind every generation it
                // took to stock, while the Makers drew routines.
                if hosted.sim(|s| s.table_open && s.bench.has_root())
                    && !hosted.sim(|s| s.missions.stocked_for(&makers, config.keep))
                    && !rt.stopping()
                {
                    match generate(&rt, &hosted, &config, None).await {
                        Ok(Generation::NoWork) => {}
                        // An operation just opened has its first step put on
                        // the table now rather than a tick later, while a
                        // Maker at the table would be handed a routine.
                        Ok(_) => offer_owed(&hosted, &mind),
                        Err(e) => tracing::warn!("mission generator: {e:#}"),
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ends, reading_prompt, retry_prompt, settle_failed};
    use crate::engine::mission::{Mission, Origin, Outcome, Todo};
    use crate::engine::workflow::OnFailed;
    use crate::opstore::owner::Owner;
    use crate::opstore::shelf::StoreShelf;
    use crate::opstore::MindStores;
    use crate::sim::operations::tests::workflows;
    use crate::sim::shelf::Shelf;
    use crate::world::Hosted;

    const DRAFT: &str = "layers/stories/the-gate.md";

    /// **A failed operation's store is settled as its workflow says**: under
    /// `discard` its work is gone, under `keep` it stays unpublished — and
    /// sent back to a step, the kept work is there to go on with, while the
    /// discarded one starts afresh on the record.
    #[test]
    fn a_failed_store_is_discarded_or_kept_and_a_kept_one_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let mind = dir.path().join("mind");
        std::fs::create_dir_all(mind.join("layers/eras")).unwrap();
        let h = Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .unwrap();
        h.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
            &mind,
        ))));
        let stores = MindStores::new(Some(&mind));
        stores.set_quiet(Box::new(|_| true));
        h.set_op_stores(Arc::clone(&stores));
        let mut wfs = workflows();
        let mut kept = wfs.iter().find(|w| w.name == "story").unwrap().clone();
        kept.name = "kept-story".into();
        kept.on_failed = OnFailed::Keep;
        wfs.push(kept);

        let fail = |workflow: &str, body: &str| -> u64 {
            let id = h.with_sim(|s| {
                s.missions.set_workflows(wfs.clone());
                s.missions
                    .review_document(workflow, "write", DRAFT)
                    .unwrap()
            });
            StoreShelf {
                stores: Arc::clone(&stores),
                owner: Owner::operation(h.id(), id),
            }
            .commit(&[(DRAFT.into(), Some(format!("{workflow}'s draft\n")))])
            .unwrap();
            h.with_sim(|s| {
                s.missions.assign(
                    body,
                    Mission::new(
                        "Write it.",
                        vec![Todo::report("report it")],
                        Origin::Generated {
                            generator: "operator".into(),
                            target: format!("doc:{DRAFT}"),
                            operation: id,
                            step: "write".into(),
                        },
                    ),
                );
                s.missions
                    .report(body, Outcome::Fail, "it cannot be written", None);
                assert!(s.missions.operations().get(id).unwrap().failed());
            });
            id
        };
        let discarded = fail("story", "m1");
        let kept = fail("kept-story", "m2");
        let read = |op: u64| h.op_shelf(op).unwrap().read(DRAFT);

        settle_failed(&h);
        assert_eq!(read(discarded), None, "discarded");
        assert_eq!(read(kept).as_deref(), Some("kept-story's draft\n"), "kept");
        assert!(
            h.sim(|s| s.missions.operations().to_retire().is_empty()),
            "settled once"
        );

        for op in [discarded, kept] {
            h.with_sim(|s| s.missions.send_to_step(op, "write").unwrap());
            stores.pin(&Owner::operation(h.id(), op));
        }
        assert_eq!(
            read(kept).as_deref(),
            Some("kept-story's draft\n"),
            "resumed"
        );
        assert_eq!(read(discarded), None, "afresh on the record");
    }

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
