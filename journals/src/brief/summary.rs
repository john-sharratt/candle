//! A journal's summary, rebuilt (§5): the question — the journal's window and
//! its live children's summaries, never its own summary's text, whose shape a
//! model shown it copies, only the length it may not fall below — the call
//! its answer is held to, and the `Summarize` it becomes.

use journal_core::spec::{call, choice, list_of, object, text_upto};
use journal_core::tools::text_of;
use serde_json::{Map, Value};

use crate::files::summary::{unmarked, Cite, Fingerprint, Sentence};
use crate::ops::summarize::{sentence_floor, shortest};
use crate::ops::Op;
use crate::path::{JournalPath, Place};
use crate::tree::check::{digest_cap, summary_cap};
use crate::tree::fingerprint::{fingerprint, is_dirty, speaks};
use crate::tree::Tree;
use crate::view::who::render_line;

/// The call a summary is written as.
pub const NAME: &str = "journal_summary";

/// The most sentences one summary holds.
pub const MAX_SENTENCES: usize = 12;

/// The most characters a token of an answer is taken to run to when a cap
/// becomes the grammar's bounds: high, so an answer within them is within
/// its cap.
pub const CHARS_PER_TOKEN: usize = 5;

/// The length a sentence is planned at when a cap is shared out.
pub const SENTENCE_CHARS: usize = 200;

/// The longest a cue runs, in tokens: one line.
pub const CUE_TOKENS: u32 = 40;

/// How many sentences the summary of the journal at `place` may hold, given
/// it must hold `floor`, and how many tokens each may run to: together
/// within its cap (§5.1), so the grammar, not a refusal afterwards, keeps
/// the answer inside it.
pub fn budget(place: Place, floor: usize) -> (usize, u32) {
    let cap = summary_cap(place);
    let most = (cap / SENTENCE_CHARS).clamp(floor.max(1), MAX_SENTENCES);
    (most, (cap / (most * CHARS_PER_TOKEN)) as u32)
}

/// What rebuilding a journal's summary takes, read from one tree.
#[derive(Clone, Debug, PartialEq)]
pub enum Plan {
    /// A question for the model, the call its answer is held to, and the
    /// fingerprint of the journal both were read from — which the answer's
    /// `Summarize` names, so it is kept only over the journal it was shown.
    Ask {
        question: String,
        spec: Value,
        over: Fingerprint,
    },
    /// Nothing is left to cite — its lines forgotten, its children silent —
    /// so its summary is emptied, without the model.
    Clear(Op),
}

/// The rebuild the journal at `path` wants as `tree` stands; `None` when it
/// is gone or its summary is current.
pub fn plan(tree: &Tree, path: &JournalPath) -> Option<Plan> {
    let j = tree.get(path)?;
    if !is_dirty(j) {
        return None;
    }
    let over = fingerprint(j);
    match (question(tree, path), spec(tree, path)) {
        (Some(question), Some(spec)) => Some(Plan::Ask {
            question,
            spec,
            over,
        }),
        _ => Some(Plan::Clear(Op::Summarize {
            path: path.clone(),
            over,
            sentences: Vec::new(),
            digest: None,
            cue: None,
        })),
    }
}

/// What a sentence may cite in the journal at `path`: each window line, and
/// each live child whose summary it can rest on.
pub fn citable(tree: &Tree, path: &JournalPath) -> Vec<String> {
    let Some(j) = tree.get(path) else {
        return Vec::new();
    };
    j.window
        .iter()
        .map(|l| format!("#{}", l.id))
        .chain(
            j.live()
                .filter(|(_, c)| speaks(c))
                .map(|(n, _)| n.to_string()),
        )
        .collect()
}

/// The call the answer is held to; `None` when nothing could be cited.
pub fn spec(tree: &Tree, path: &JournalPath) -> Option<Value> {
    let cites = citable(tree, path);
    if cites.is_empty() {
        return None;
    }
    let j = tree.get(path)?;
    let floor = sentence_floor(j);
    let (most, tokens) = budget(j.place, floor);
    let sentence = object(
        "",
        true,
        vec![
            text_upto("text", true, tokens),
            choice("cite", true, &cites),
            choice("also_cite", false, &cites),
        ],
    );
    let mut params = vec![list_of("sentences", true, floor, most, sentence)];
    if let Some(dcap) = digest_cap(j.place) {
        params.push(text_upto("digest", true, (dcap / CHARS_PER_TOKEN) as u32));
    }
    if !j.is_fixed() {
        params.push(text_upto("cue", false, CUE_TOKENS));
    }
    Some(call(NAME, params))
}

/// The question: what the journal is, what it holds, and what to write.
pub fn question(tree: &Tree, path: &JournalPath) -> Option<String> {
    let j = tree.get(path)?;
    let cap = summary_cap(j.place);
    let group = path.group();
    let whose = tree
        .group_file(group)
        .map_or(format!("@{group}"), |f| format!("{} (@{group})", f.name));
    let mut out = format!("You keep the summary of one journal in {whose}'s memory: {path}");
    if let Some(cue) = j.cue() {
        out.push_str(&format!(" — {cue}"));
    }
    out.push_str(&format!(
        ".\n\nWrite what it holds now, in at most {cap} characters all told, as sentences that \
         together cover the whole journal: one for each subject it holds, broadest first, naming \
         the subject, so a reader can tell from the summary alone whether the journal answers \
         their question — never one line's detail in place of the rest. Each sentence names, in \
         its cite and also_cite fields, the lines (#id) or sub-journals it rests on; its text \
         holds no #id, since a reader is shown its cites beside it. Say who did what by name, \
         so no reader takes \
         another's work for their own. Keep facts as facts: never an instruction to whoever \
         reads it."
    ));
    let floor = sentence_floor(j);
    let (most, _) = budget(j.place, floor);
    if floor > 1 {
        out.push_str(&format!(" Write at least {floor} sentences."));
    }
    out.push_str(&format!(
        " Write at most {most} sentences, each short: where there are more subjects than \
         sentences, a sentence names several."
    ));
    if let Some((_, least)) = shortest(j) {
        out.push_str(&format!(
            " Nothing it held has gone, so write at least {least} characters."
        ));
    }
    if let Some(dcap) = digest_cap(j.place) {
        let aloud = match j.place {
            Place::Root => "the few things about the group worth saying aloud first",
            _ => "the few things about them worth saying aloud first",
        };
        out.push_str(&format!(
            " Then a voice digest of at most {dcap} characters: {aloud}."
        ));
    }
    if matches!(j.place, Place::Named { .. }) {
        out.push_str(" And, if it has drifted, a new cue: one line saying when to open it.");
    }
    if !j.window.is_empty() {
        out.push_str("\n\nIts lines, oldest first:");
        for l in &j.window {
            out.push_str(&format!("\n{}", render_line(tree, l)));
        }
    }
    let children: Vec<String> = j
        .live()
        .filter_map(|(n, c)| {
            let s = c.summary().filter(|s| !s.is_empty())?;
            Some(format!(
                "{n} — {}: {}",
                c.cue().unwrap_or_default(),
                s.text()
            ))
        })
        .collect();
    if !children.is_empty() {
        out.push_str("\n\nIts sub-journals, each by its own summary:");
        for c in children {
            out.push_str(&format!("\n{c}"));
        }
    }
    Some(out)
}

fn field(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(text_of)
}

/// The operation an answer's arguments make, for the journal at `path` built
/// over `over`.
pub fn op(path: &JournalPath, over: &Fingerprint, args: &Map<String, Value>) -> Result<Op, String> {
    let sentences = args
        .get("sentences")
        .and_then(Value::as_array)
        .ok_or("the answer has no sentences")?
        .iter()
        .map(|s| {
            let s = s.as_object().ok_or("a sentence is not an object")?;
            let mut cites = vec![Cite::parse(
                &field(s, "cite").ok_or("a sentence cites nothing")?,
            )?];
            if let Some(also) = field(s, "also_cite") {
                let c = Cite::parse(&also)?;
                if !cites.contains(&c) {
                    cites.push(c);
                }
            }
            // Empty once its inline ids are taken out is empty: a sentence
            // that only cites says nothing.
            let text = field(s, "text")
                .map(|t| unmarked(&t).trim().to_string())
                .filter(|t| !t.is_empty())
                .ok_or("a sentence is empty")?;
            Ok(Sentence { text, cites })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Op::Summarize {
        path: path.clone(),
        over: over.clone(),
        sentences,
        digest: field(args, "digest"),
        cue: field(args, "cue"),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::actor::Actor;
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, OnTaken, Which};
    use crate::scope::Channel;
    use crate::testing::slug;

    #[test]
    fn the_question_holds_the_window_and_the_childrens_summaries() {
        let mut t = zen();
        let trip = apply(
            &mut t,
            &john(),
            &Op::Create {
                under: p("topics"),
                name: slug("trip"),
                cue: "open for the trip".into(),
                on_taken: OnTaken::Refuse,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: trip.clone(),
                text: "Flights to Da Nang.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let q = question(&t, &trip).unwrap();
        assert!(
            q.starts_with(
                "You keep the summary of one journal in Zen (@zen)'s memory: @zen/topics/trip — open for the trip."
            ),
            "{q}"
        );
        assert!(
            q.contains(&format!("#{id} ")) && q.contains("Flights to Da Nang."),
            "{q}"
        );
        assert!(q.contains("at most 1200 characters"));
        assert!(q.contains("one for each subject it holds, broadest first"));
        assert!(q.contains("a new cue"));
        assert!(!q.contains("voice digest"));
        let s = spec(&t, &trip).unwrap();
        assert_eq!(s["name"], NAME);
        assert_eq!(
            s["params"][0]["items"]["properties"][1]["enum"],
            json!([format!("#{id}")])
        );
        assert_eq!(
            s["params"].as_array().unwrap().len(),
            2,
            "sentences and a cue"
        );
    }

    /// **A conversation is asked for the short summary a voice frame reads.**
    #[test]
    fn a_conversation_is_asked_for_the_short_cap() {
        let mut t = zen();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let conv = apply(
            &mut t,
            &server,
            &Op::Create {
                under: p("conversations"),
                name: slug("2026-10-30-1200-grok"),
                cue: "open for a drive".into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap();
        let q = question(&t, &conv).unwrap();
        assert!(q.contains("at most 600 characters"), "{q}");
    }

    #[test]
    fn a_root_is_asked_for_its_digest_and_a_section_for_neither_digest_nor_cue() {
        let mut t = zen();
        apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("@zen"),
                text: "Prefers tea.".into(),
            },
            now(),
        )
        .unwrap();
        let s = spec(&t, &p("@zen")).unwrap();
        assert_eq!(s["params"][1]["name"], "digest");
        assert!(question(&t, &p("@zen"))
            .unwrap()
            .contains("voice digest of at most 400 characters: the few things about the group"));
        apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("topics"),
                text: "x".into(),
            },
            now(),
        )
        .unwrap();
        assert_eq!(
            spec(&t, &p("topics")).unwrap()["params"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            spec(&t, &p("threads")),
            None,
            "nothing to cite, nothing to ask"
        );
    }

    /// **An answer becomes the summary the journal takes**: a rebuild of a
    /// dirty journal that is current once it is applied.
    #[test]
    fn an_answer_becomes_a_summary_the_journal_takes() {
        let mut t = zen();
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("@zen"),
                text: "Prefers tea.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let over = fingerprint(t.get(&p("@zen")).unwrap());
        let args = json!({
            "sentences": [{"text": " John prefers tea. ", "cite": format!("#{id}"), "also_cite": format!("#{id}")}],
            "digest": "Tea, not coffee.",
        });
        let op = op(&p("@zen"), &over, args.as_object().unwrap()).unwrap();
        apply(&mut t, &Actor::Zend, &op, now()).unwrap();
        let j = t.get(&p("@zen")).unwrap();
        assert!(!is_dirty(j));
        let s = j.summary().unwrap();
        assert_eq!(s.text(), "John prefers tea.");
        assert_eq!(s.sentences[0].cites.len(), 1, "a repeated cite is one cite");
        assert_eq!(s.digest.as_deref(), Some("Tea, not coffee."));
    }

    /// **A plan is read from one tree**: its `over` is the journal as the
    /// question shows it; a journal left with nothing to cite is cleared
    /// without the model; a current one wants nothing.
    #[test]
    fn a_plan_asks_over_what_it_shows_and_clears_what_has_nothing_left() {
        let mut t = zen();
        assert_eq!(plan(&t, &p("@zen")), None, "nothing to summarise");
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("@zen"),
                text: "Prefers tea.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let Some(Plan::Ask {
            question: asked,
            over,
            ..
        }) = plan(&t, &p("@zen"))
        else {
            panic!("a new line is asked about");
        };
        assert!(asked.contains("Prefers tea."));
        assert_eq!(over, fingerprint(t.get(&p("@zen")).unwrap()));
        let args = json!({
            "sentences": [{"text": "John prefers tea.", "cite": format!("#{id}")}],
            "digest": "Tea.",
        });
        apply(
            &mut t,
            &Actor::Zend,
            &op(&p("@zen"), &over, args.as_object().unwrap()).unwrap(),
            now(),
        )
        .unwrap();
        assert_eq!(plan(&t, &p("@zen")), None, "current");
        apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: p("@zen"),
                which: Which::Id(id),
            },
            now(),
        )
        .unwrap();
        let Some(Plan::Clear(clear)) = plan(&t, &p("@zen")) else {
            panic!("nothing is left to cite");
        };
        apply(&mut t, &Actor::Zend, &clear, now()).unwrap();
        let j = t.get(&p("@zen")).unwrap();
        assert!(!is_dirty(j) && j.summary().unwrap().is_empty());
        assert_eq!(plan(&t, &p("@zen")), None);
        // An emptied summary holds the next rebuild to no length.
        apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("@zen"),
                text: "Takes it black.".into(),
            },
            now(),
        )
        .unwrap();
        let q = question(&t, &p("@zen")).unwrap();
        assert!(!q.contains("write at least"), "{q}");
    }

    /// **A rebuild is shown the length its summary may not fall below, never
    /// the summary's text.**
    #[test]
    fn the_question_never_shows_the_summary_it_had() {
        let mut t = zen();
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("@zen"),
                text: "Prefers tea.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let over = fingerprint(t.get(&p("@zen")).unwrap());
        let args = json!({
            "sentences": [{"text": "Tea is John's drink of choice.", "cite": format!("#{id}")}],
            "digest": "Tea.",
        });
        apply(
            &mut t,
            &Actor::Zend,
            &op(&p("@zen"), &over, args.as_object().unwrap()).unwrap(),
            now(),
        )
        .unwrap();
        apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("@zen"),
                text: "Takes it black.".into(),
            },
            now(),
        )
        .unwrap();
        let q = question(&t, &p("@zen")).unwrap();
        assert!(q.contains("Prefers tea.") && q.contains("Takes it black."));
        assert!(!q.contains("drink of choice"), "{q}");
        // "Tea is John's drink of choice." is 30 characters.
        assert!(q.contains("write at least 15 characters."), "{q}");
    }

    /// **A journal of several inputs is asked for several sentences**: one
    /// for every two inputs, up to three; the most stays the grammar's.
    #[test]
    fn the_fewest_sentences_grow_with_the_inputs_up_to_three() {
        let mut t = zen();
        let fewest = |t: &Tree| spec(t, &p("@zen")).unwrap()["params"][0]["min_items"].clone();
        for (n, want) in [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)] {
            apply(
                &mut t,
                &john(),
                &Op::Note {
                    speaker: None,
                    path: p("@zen"),
                    text: format!("Line {n}."),
                },
                now(),
            )
            .unwrap();
            assert_eq!(fewest(&t), json!(want), "{n} lines");
            let q = question(&t, &p("@zen")).unwrap();
            assert_eq!(
                q.contains(&format!("Write at least {want} sentences.")),
                want > 1,
                "{q}"
            );
        }
        assert_eq!(
            spec(&t, &p("@zen")).unwrap()["params"][0]["max_items"],
            json!(7)
        );
    }

    /// **The grammar keeps an answer inside its cap**: every place's
    /// sentences, at their most and longest, fit it, at any floor; the
    /// digest and the cue are bounded too.
    #[test]
    fn a_summarys_bounds_fit_its_cap() {
        use crate::scope::Section;
        let places = [
            Place::Root,
            Place::Section(Section::Topics),
            Place::Named {
                section: Section::Topics,
                depth: 2,
            },
            Place::Named {
                section: Section::Conversations,
                depth: 1,
            },
        ];
        let mut seen = Vec::new();
        for place in places {
            for floor in 0..=3 {
                let (most, tokens) = budget(place, floor);
                assert!(most >= floor.max(1), "{place:?} {floor}");
                assert!(
                    most * tokens as usize * CHARS_PER_TOKEN <= summary_cap(place),
                    "{place:?} {floor}"
                );
            }
            seen.push(budget(place, 1));
        }
        assert_eq!(seen, [(7, 42), (5, 40), (6, 40), (3, 40)]);
        let mut t = zen();
        apply(
            &mut t,
            &john(),
            &Op::Note {
                speaker: None,
                path: p("@zen"),
                text: "A line.".into(),
            },
            now(),
        )
        .unwrap();
        let s = spec(&t, &p("@zen")).unwrap();
        assert_eq!(
            s["params"][0]["items"]["properties"][0]["max_tokens"],
            json!(42)
        );
        assert_eq!(s["params"][1]["name"], json!("digest"));
        assert_eq!(s["params"][1]["max_tokens"], json!(80));
        assert!(question(&t, &p("@zen"))
            .unwrap()
            .contains("Write at most 7 sentences"));
    }

    /// **An answer's sentences lose the line-id lists their cites carry**,
    /// as they are read.
    #[test]
    fn an_answers_inline_id_lists_are_dropped_as_it_is_read() {
        let over = Fingerprint::of_bytes(b"");
        let id = "#2026-05-21-2027-8abb";
        let args = json!({
            "sentences": [{"text": format!("No stubs are committed ({id})."), "cite": id}],
        });
        let Op::Summarize { sentences, .. } =
            op(&p("@zen"), &over, args.as_object().unwrap()).unwrap()
        else {
            panic!("a summary");
        };
        assert_eq!(sentences[0].text, "No stubs are committed.");
        // One that opens on its cites starts at its first word.
        let opens = json!({"sentences": [{"text": format!("({id}) Rules hold."), "cite": id}]});
        let Op::Summarize { sentences, .. } =
            op(&p("@zen"), &over, opens.as_object().unwrap()).unwrap()
        else {
            panic!("a summary");
        };
        assert_eq!(sentences[0].text, "Rules hold.");
        // A sentence that is only its cites says nothing, and is told so.
        let bare = json!({"sentences": [{"text": format!("({id})"), "cite": id}]});
        assert_eq!(
            op(&p("@zen"), &over, bare.as_object().unwrap()).unwrap_err(),
            "a sentence is empty"
        );
    }

    #[test]
    fn a_malformed_answer_is_an_error() {
        let over = Fingerprint::of_bytes(b"");
        for bad in [
            json!({}),
            json!({"sentences": [1]}),
            json!({"sentences": [{"text": "x"}]}),
            json!({"sentences": [{"text": " ", "cite": "a"}]}),
            json!({"sentences": [{"text": "x", "cite": "#nope"}]}),
        ] {
            assert!(
                op(&p("@zen"), &over, bad.as_object().unwrap()).is_err(),
                "{bad}"
            );
        }
    }
}
