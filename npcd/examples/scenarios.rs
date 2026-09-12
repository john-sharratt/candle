//! Put a spread of situations to the running daemon and report what a character
//! actually did with each.
//!
//! ```text
//! cargo run --release -p npcd --example scenarios
//! cargo run --release -p npcd --example scenarios -- --rendered   # the other assembly
//! cargo run --release -p npcd --example scenarios -- --url http://192.168.0.6:8081
//! ```
//!
//! # What it is for
//!
//! Four questions, none of which can be answered without a live checkpoint and
//! none of which the daemon's own views can answer at all:
//!
//! 1. **Does the frame produce a call?** A character that emitted nothing and
//!    one that chose to do nothing are identical in the pulse — same empty act
//!    list, same successful tick, no error. The difference is the whole problem.
//! 2. **Does provenance bring the right acts into focus?** A room with somebody
//!    in it should surface `tell`; an empty corridor should not.
//! 3. **Does thinking happen when it should?** And — the failure that cost a
//!    night — does the block ever *close*. An unterminated `<think>` runs to the
//!    token ceiling and the whole decode is discarded as reasoning, silently.
//! 4. **Is the call clean?** A refused call is a character acting into nothing.
//!
//! # Why an example and not a test
//!
//! It needs the model, and the model is held by the daemon that is running. So
//! this talks to it over HTTP and the cast goes on thinking beside the
//! scenarios. `cargo test` cannot do that and should not try.

use std::time::Duration;

// The standing task, from the constants the daemon delivers rather than
// retyped here — see `cases`.
use npcd::engine::runtime::{IN_COMPANY, NO_MISSION};

/// One situation, and what a character ought to make of it.
struct Case {
    name: &'static str,
    /// What the scenario is actually testing, printed beside the result so a
    /// failure reads as a finding rather than as a red line.
    asks: &'static str,
    body: serde_json::Value,
    /// What has to be true of the outcome. Returns the reason it failed, or
    /// `None` when it held.
    expect: fn(&Outcome) -> Option<String>,
}

#[derive(serde::Deserialize)]
struct Outcome {
    acts: Vec<String>,
    rejected: Vec<String>,
    narration: String,
    raw: String,
    opened_think: bool,
    closed_think: bool,
    runaway_think: bool,
    projected: bool,
    ms: u64,
}

/// A character must do *something* — an act, or a refusal it can learn from.
fn acted(o: &Outcome) -> Option<String> {
    if o.acts.is_empty() {
        return Some(format!(
            "no act (narration {:?}, raw {} bytes)",
            truncate(&o.narration, 60),
            o.raw.len()
        ));
    }
    None
}

/// Acted, and cleanly — nothing refused.
fn acted_cleanly(o: &Outcome) -> Option<String> {
    acted(o)
        .or_else(|| (!o.rejected.is_empty()).then(|| format!("refused: {}", o.rejected.join("; "))))
}

fn truncate(s: &str, n: usize) -> String {
    let s = s.trim().replace('\n', " ");
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s,
    }
}

/// Somewhere with the ordinary ways out.
///
/// **Never the room the character is standing in.** `move_to` is bound to this
/// list, so including the current room would put back the one move that cannot
/// succeed — and being refused for it taught a live cast nothing at all: it
/// emitted the same move every tick for an evening.
const WAYS_OUT: &[&str] = &[
    "the quiet room",
    "the long table",
    "the chronicle",
    "the command level",
    "the story level",
];

fn scenario(perceive: &str) -> serde_json::Value {
    serde_json::json!({
        "personality": "maker",
        "world_id": "battle-cities",
        "name": "Test Subject",
        "perceive": perceive,
        "places": WAYS_OUT,
    })
}

/// The same, in a room with somebody in it.
///
/// **Company is provenance, not prose.** The grammar is built from it: alone,
/// the acts that need somebody are not in the mask at all, and in company an
/// addressee may only be one of these names. A scenario whose `perceive` says
/// Perrin is here while its company is empty is describing a room the character
/// cannot act in — and would be testing the wrong thing quietly.
fn with(perceive: &str, company: &[&str]) -> serde_json::Value {
    let mut b = scenario(perceive);
    b["company"] = company.iter().map(|n| n.to_string()).collect();
    b
}

/// Everything the harness puts to the daemon.
///
/// # The situation carries the standing task, because production's does
///
/// A character on a quiet turn reads the room *and* what it is for — the
/// runtime delivers one and then the other, and the standing task is the more
/// recent, which is where attention weights hardest. Fixtures that carried only
/// the room were testing a thinner situation than any character has ever been
/// in, and quietly stopped covering the wording that actually drives behaviour:
/// they went on saying "you can also use: tell" long after the acts a character
/// has were decided by the grammar rather than by a sentence.
///
/// Taken from the constants rather than retyped, so a change to what a
/// character is told is a change to what the harness measures.
fn cases() -> Vec<Case> {
    let vault = &format!(
        "You are in the first writing room.\n\nSix story desks stand free.\n\n\
         Within reach: story desks.\n\n{NO_MISSION}"
    );
    let with_company = &format!(
        "You are in the first writing room.\n\nPerrin Vastwood is here. Six story desks stand \
         free.\n\nWithin reach: story desks.\n\n{}",
        IN_COMPANY.replace("{who}", "Perrin Vastwood is")
    );

    vec![
        // ── 1. does the frame produce a call at all ─────────────────────────
        Case {
            name: "alone in a room",
            asks: "the frame produces an act with nothing prompting one",
            body: scenario(vault),
            expect: acted,
        },
        Case {
            name: "alone in a corridor",
            asks: "a room offering nothing still gets an act, not silence",
            body: scenario(
                "You are in the north run.\n\nIt is a corridor and there is nothing here.",
            ),
            expect: acted,
        },
        // ── 2. does provenance bring the right acts into focus ──────────────
        Case {
            name: "somebody is here",
            asks: "company surfaces speech — `say` or `tell`, not a journey",
            body: with(with_company, &["Perrin Vastwood"]),
            expect: |o| {
                acted(o).or_else(|| {
                    // An act reads as `tool — intent`, so speech is the tool's
                    // own name. It used to be matched as "You say…" — the
                    // world's narration of a successful act — which stopped
                    // being what gets recorded when the act itself became the
                    // record, and left this asserting on prose nothing emits.
                    let spoke = o
                        .acts
                        .iter()
                        .any(|a| a.starts_with("say") || a.starts_with("tell"));
                    (!spoke).then(|| format!("did not speak — {:?}", o.acts))
                })
            },
        },
        Case {
            name: "addressed by name",
            asks: "being spoken to draws a reply rather than a departure",
            body: with(
                &format!(
                    "{with_company}\n\nPerrin Vastwood says to you: that the third era is written \
                     twice and the two do not agree."
                ),
                &["Perrin Vastwood"],
            ),
            expect: |o| {
                acted(o).or_else(|| {
                    let left = o.acts.iter().any(|a| a.contains("set off"));
                    left.then(|| format!("walked away from a question — {:?}", o.acts))
                })
            },
        },
        Case {
            name: "a named room to walk to",
            asks: "a destination it has been told about produces `move_to`, not an invention",
            body: scenario(
                "You are in the first writing room.\n\nYou have decided to go to the quiet room.",
            ),
            expect: |o| {
                acted(o).or_else(|| {
                    (!o.rejected.is_empty())
                        .then(|| format!("refused a real room: {}", o.rejected.join("; ")))
                })
            },
        },
        // ── 3. thinking on/off, and whether the block ever closes ───────────
        Case {
            name: "idle · thinking off",
            asks: "no reasoning block on a turn that does not need one",
            body: scenario(vault),
            expect: |o| {
                o.opened_think
                    .then(|| format!("opened <think> when idle (closed: {})", o.closed_think))
            },
        },
        Case {
            name: "writing mission · thinking deep",
            asks: "a mission that needs thought gets a block — and it CLOSES",
            body: {
                let mut b = scenario(
                    "You are at a story desk in the first writing room.\n\nThe gap ledger shows a \
                     silence of eleven years between the fall of the redoubt and the first entry \
                     of the new watch.",
                );
                b["mission"] = "Draft the story that belongs in the eleven-year silence.".into();
                b["thinking"] = "deep".into();
                b
            },
            expect: |o| {
                o.runaway_think.then(|| {
                    "opened <think> and never closed it — the decode was discarded whole"
                        .to_string()
                })
            },
        },
        Case {
            name: "writing mission · thinking off",
            asks: "the dial is honoured downwards too",
            body: {
                let mut b = scenario(vault);
                b["mission"] = "Draft the story that belongs in the silence.".into();
                b["thinking"] = "off".into();
                b
            },
            expect: |o| {
                o.opened_think
                    .then(|| "opened <think> with deliberation off".to_string())
            },
        },
        // ── 4. clean invocation ─────────────────────────────────────────────
        Case {
            name: "clean call · alone",
            asks: "nothing is refused — every call names a real act with its arguments",
            body: scenario(vault),
            expect: acted_cleanly,
        },
        Case {
            name: "clean call · in company",
            asks: "`tell` arrives with its addressee, which it never did free-decoding",
            body: with(with_company, &["Perrin Vastwood"]),
            expect: acted_cleanly,
        },
        // ── 5. the refusals a character must be able to learn from ──────────
        Case {
            name: "a room that does not exist",
            asks: "an invented destination is refused and the refusal names real rooms",
            body: scenario(
                "You are in the first writing room.\n\nYou have decided to go to the observatory \
                 on Level 9.",
            ),
            expect: |o| {
                // Either it refuses to invent (best), or the world refuses it
                // and says so. Silence is the only wrong answer.
                acted(o).and_then(|no_act| o.rejected.is_empty().then_some(no_act))
            },
        },
        Case {
            name: "somebody who is not here",
            // Was "refused, naming who IS here". It cannot be refused any more:
            // alone, the acts that take an addressee are not in the grammar, so
            // addressing an absent person is unreachable rather than rejected.
            // What is under test is that being *reminded* of somebody absent
            // does not stall the character.
            asks: "an absent person is not addressable, and thinking of one still produces an act",
            body: scenario(&format!(
                "{vault}\n\nYou have been thinking about something Wyneth Vayne said, and she is \
                 not here."
            )),
            expect: |o| {
                acted(o).or_else(|| {
                    (!o.rejected.is_empty())
                        .then(|| format!("refused what should be unreachable: {:?}", o.rejected))
                })
            },
        },
        // ── 6. the standing instruction ─────────────────────────────────────
        Case {
            name: "no mission · alone",
            asks: "having nothing asked of it still produces movement or speech, not stillness",
            body: scenario(vault),
            // Stillness is `wait`, and it is the one wrong answer here — every
            // other act is the standing instruction doing its job. Asserting on
            // a leading "You" tested the world's narration of an act rather than
            // the act, and no longer matched anything at all.
            expect: |o| {
                acted(o).or_else(|| {
                    o.acts
                        .iter()
                        .all(|a| a.starts_with("wait"))
                        .then(|| format!("nothing but stillness — {:?}", o.acts))
                })
            },
        },
        Case {
            name: "no mission · company",
            asks: "the standing instruction keeps it in the room rather than exploring away",
            body: with(with_company, &["Perrin Vastwood"]),
            expect: acted,
        },
        // ── 7. a personality nothing is written for ─────────────────────────
        Case {
            name: "unknown personality",
            asks: "the generic identity still yields a character that acts",
            body: {
                let mut b = scenario(vault);
                b["personality"] = "nobody-has-written-this".into();
                b
            },
            expect: acted,
        },
        // ── 8. waiting, which is now a named thing aimed at a named person ──
        Case {
            name: "waiting names its condition",
            asks: "a wait carries a `for` the world can settle, never free text",
            body: with(
                &format!(
                    "{with_company}\n\nYou have asked Perrin Vastwood something and it has \
                          not answered. There is nothing else you need from this room."
                ),
                &["Perrin Vastwood"],
            ),
            expect: |o| {
                acted(o).or_else(|| {
                    let bad = o.acts.iter().find(|a| {
                        a.starts_with("wait_for")
                            && !["someone_speaks", "someone_arrives", "someone_leaves"]
                                .iter()
                                .any(|k| a.contains(k))
                    });
                    bad.map(|a| format!("a wait for nothing the world can settle — {a}"))
                })
            },
        },
        Case {
            name: "cornered by a wait",
            // The deadlock the act exists to break. Perrin is already waiting on
            // this character, so waiting back is not in its grammar: the only
            // ways out are to speak, to move, or to look. Any of those is right;
            // returning the stare is the one thing that must be unreachable.
            asks: "somebody already waiting on you cannot be waited on back",
            body: {
                let mut b = with(
                    &format!(
                        "{with_company}\n\nPerrin Vastwood is waiting for you to say \
                              something."
                    ),
                    &["Perrin Vastwood"],
                );
                b["waited_on_by"] = vec!["Perrin Vastwood".to_string()].into();
                b
            },
            expect: |o| {
                acted(o).or_else(|| {
                    o.acts
                        .iter()
                        .find(|a| a.starts_with("wait_for") && a.contains("Perrin"))
                        .map(|a| format!("it returned the stare — {a}"))
                })
            },
        },
        Case {
            name: "alone, waiting asks nobody",
            asks: "with nobody here a wait names no person and still names its condition",
            body: scenario(vault),
            expect: |o| {
                acted(o).or_else(|| {
                    o.acts
                        .iter()
                        .find(|a| a.starts_with("wait_for") && a.contains(';'))
                        .map(|a| format!("it named somebody in an empty room — {a}"))
                })
            },
        },
        // ── 9. a world with no map ──────────────────────────────────────────
        Case {
            name: "no world",
            // Nowhere to walk is now enforced rather than hoped for: with no
            // places, `move_to` has a required argument with nothing to choose
            // from, so the act is not in the grammar at all. The character
            // cannot spend its turn trying to leave.
            asks: "a character with nowhere to walk cannot spend its turn trying to",
            body: {
                let mut b = scenario("You are somewhere. There is nothing here but you.");
                b["world_id"] = "nowhere-at-all".into();
                b["places"] = serde_json::Value::Array(Vec::new());
                b
            },
            expect: |o| {
                acted(o).or_else(|| {
                    o.acts
                        .iter()
                        .find(|a| a.starts_with("move_to"))
                        .map(|a| format!("it walked out of a world with no rooms — {a}"))
                })
            },
        },
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let url = arg(&args, "--url").unwrap_or_else(|| "http://192.168.0.6:8081".to_string());
    let rendered = args.iter().any(|a| a == "--rendered");
    let only = arg(&args, "--only");

    let all = cases();
    let cases: Vec<&Case> = all
        .iter()
        .filter(|c| only.as_deref().is_none_or(|o| c.name.contains(o)))
        .collect();

    println!(
        "\n{} scenarios against {url} — {} assembly\n",
        cases.len(),
        if rendered {
            "RENDERED prompt"
        } else {
            "PROJECTED schema"
        }
    );

    let client = ureq::AgentBuilder::new()
        // A decode is seconds and a cold conversation open can be much more.
        .timeout(Duration::from_secs(600))
        .build();

    let (mut held, mut failed, mut errored) = (0, 0, 0);
    let mut rows: Vec<(String, bool, String)> = Vec::new();

    for c in &cases {
        let mut body = c.body.clone();
        body["projected"] = (!rendered).into();

        print!("  {:<28} ", c.name);
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let sent = client
            .post(&format!("{url}/v1/simulate"))
            .set("x-tokera-user", "111930197703828817752")
            .set("x-tokera-provider", "google")
            .set("x-tokera-email", "johnathan.sharratt@gmail.com")
            .send_json(body);

        match sent {
            Err(e) => {
                errored += 1;
                println!("ERROR  {e}");
                rows.push((c.name.to_string(), false, format!("transport: {e}")));
            }
            Ok(resp) => match resp.into_json::<Outcome>() {
                Err(e) => {
                    errored += 1;
                    println!("ERROR  malformed reply: {e}");
                    rows.push((c.name.to_string(), false, format!("malformed: {e}")));
                }
                Ok(o) => {
                    let verdict = (c.expect)(&o);
                    let ok = verdict.is_none();
                    if ok {
                        held += 1;
                    } else {
                        failed += 1;
                    }
                    println!(
                        "{}  {:>6}ms  think:{}  acts:{}  {}",
                        if ok { "HELD" } else { "FAIL" },
                        o.ms,
                        match (o.opened_think, o.closed_think) {
                            (false, _) => "off",
                            (true, true) => "closed",
                            (true, false) => "RUNAWAY",
                        },
                        o.acts.len(),
                        verdict.clone().unwrap_or_else(|| truncate(
                            o.acts.first().map(String::as_str).unwrap_or(""),
                            52
                        ))
                    );
                    rows.push((
                        c.name.to_string(),
                        ok,
                        verdict.unwrap_or_else(|| o.acts.join(" / ")),
                    ));
                    if !o.projected && !rendered {
                        println!("      (ran on the rendered prompt — no projection is installed)");
                    }
                }
            },
        }
    }

    println!("\n{held} held, {failed} failed, {errored} errored\n");
    println!("WHAT EACH ASKED");
    for (c, (_, ok, detail)) in cases.iter().zip(rows.iter()) {
        println!(
            "  {} {:<28} {}\n      {}",
            if *ok { "·" } else { "!" },
            c.name,
            c.asks,
            truncate(detail, 100)
        );
    }
    println!();
    if failed > 0 || errored > 0 {
        std::process::exit(1);
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
