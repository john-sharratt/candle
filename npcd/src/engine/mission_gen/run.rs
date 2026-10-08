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

use super::answer::{self, Answer, Desk};
use super::config::{Config, Generator};
use super::corpus::Corpus;
use super::material;
use super::target::{self, Kind, Target};
use crate::engine::journal::tools::arguments;
use crate::engine::mind::Minds;
use crate::engine::prose::{self, decode_call, CallAsk};
use crate::engine::runtime::Runtime;
use crate::prose::Request;
use crate::world::Hosted;

/// The most tokens one answer may run to. A review's reading and faults run to
/// a thousand words between them, and an answer cut off by the cap is not a
/// call at all — a life event was refused "not a call" three times running.
const ANSWER_TOKENS: usize = 2400;

/// How many times an answer is asked for before the target is given up.
const ATTEMPTS: usize = 3;

/// The sampling temperature a mission is written at.
///
/// Below the checkpoint's own, because a mission carries paths and dates that
/// must be copied exactly: at the default a story brief about "the vanishing
/// silence" named its own file `the-vanishing-silage.md`.
const TEMPERATURE: f32 = 0.5;

/// How often the loop looks at the table.
const TICK: Duration = Duration::from_secs(20);

/// What one generation came to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Generation {
    /// A mission is on the table.
    Offered {
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
    let mut corpus = Corpus::read(&mind, hosted.id());
    corpus.reviewable = hosted.sim(|s| s.missions.written().to_vec());
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
                let mission = answer::mission(&p, &generator.id, target, desk.as_ref());
                hosted.with_sim(|s| s.missions.offer(mission, target.fingerprint));
                tracing::info!(
                    generator = %generator.id, target = %target.key, writes = %p.writes, attempt,
                    "mission generator: a mission is on the table — {}", p.brief
                );
                return Ok(Generation::Offered {
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
    let part = bench_for(path);
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
    use super::bench_for;

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
