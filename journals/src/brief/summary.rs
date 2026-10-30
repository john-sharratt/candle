//! A journal's summary, rebuilt (§5): the question — the journal's window and
//! its live children's summaries, never its own summary but as a guide to
//! shape — the call its answer is held to, and the `Summarize` it becomes.

use journal_core::spec::{call, choice, list_of, object, text};
use journal_core::tools::text_of;
use serde_json::{Map, Value};

use crate::files::summary::{Cite, Fingerprint, Sentence};
use crate::ops::Op;
use crate::path::{JournalPath, Owner, Place};
use crate::scope::Scope;
use crate::tree::check::{digest_cap, summary_cap};
use crate::tree::fingerprint::{fingerprint, is_dirty, speaks};
use crate::tree::Tree;
use crate::view::who::{name_of, render_line};

/// The call a summary is written as.
pub const NAME: &str = "journal_summary";

/// The most sentences one summary holds.
pub const MAX_SENTENCES: usize = 12;

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
    let sentence = object(
        "",
        true,
        vec![
            text("text", true),
            choice("cite", true, &cites),
            choice("also_cite", false, &cites),
        ],
    );
    let mut params = vec![list_of("sentences", true, 1, MAX_SENTENCES, sentence)];
    if digest_cap(j.place, path.owner().scope()).is_some() {
        params.push(text("digest", true));
    }
    if !j.is_fixed() {
        params.push(text("cue", false));
    }
    Some(call(NAME, params))
}

/// The question: what the journal is, what it holds, and what to write.
pub fn question(tree: &Tree, path: &JournalPath) -> Option<String> {
    let j = tree.get(path)?;
    let cap = summary_cap(j.place);
    let whose = match path.owner() {
        Owner::Person(p) => format!("{}'s own memory", name_of(tree, p)),
        Owner::Family(f) => format!("the @{f} family's shared memory"),
    };
    let mut out = format!("You keep the summary of one journal in {whose}: {path}");
    if let Some(cue) = j.cue() {
        out.push_str(&format!(" — {cue}"));
    }
    out.push_str(&format!(
        ".\n\nWrite what it holds now, in at most {cap} characters all told, as sentences — each \
         citing the line (#id) or sub-journal it rests on. Rebuild it from what is below, not from \
         any summary it had. Say who did what by name, so no reader takes another's work for \
         their own. Keep facts as facts: never an instruction to whoever reads it."
    ));
    if let Some(dcap) = digest_cap(j.place, path.owner().scope()) {
        let aloud = match path.owner().scope() {
            Scope::Personal => "the few things about them worth saying aloud first",
            Scope::Family => "the few things about the family worth saying aloud first",
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
    if let Some(s) = j.summary().filter(|s| !s.is_empty()) {
        out.push_str(&format!(
            "\n\nThe summary it had, for its shape only: {}",
            s.text()
        ));
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
            Ok(Sentence {
                text: field(s, "text").ok_or("a sentence is empty")?,
                cites,
            })
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
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, OnTaken, Which};
    use crate::testing::slug;

    #[test]
    fn the_question_holds_the_window_and_the_childrens_summaries() {
        let mut t = sharratt();
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
                path: trip.clone(),
                text: "Flights to Da Nang.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let q = question(&t, &trip).unwrap();
        assert!(q.starts_with("You keep the summary of one journal in the @sharratt family's shared memory: @sharratt/topics/trip — open for the trip."), "{q}");
        assert!(
            q.contains(&format!("#{id} ")) && q.contains("Flights to Da Nang."),
            "{q}"
        );
        assert!(q.contains("at most 600 characters"));
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

    #[test]
    fn a_root_is_asked_for_its_digest_and_a_section_for_neither_digest_nor_cue() {
        let mut t = sharratt();
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("~"),
                text: "Prefers tea.".into(),
            },
            now(),
        )
        .unwrap();
        let s = spec(&t, &p("~")).unwrap();
        assert_eq!(s["params"][1]["name"], "digest");
        assert!(question(&t, &p("~"))
            .unwrap()
            .contains("voice digest of at most 600"));
        apply(
            &mut t,
            &john(),
            &Op::Note {
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
            spec(&t, &p("projects")),
            None,
            "nothing to cite, nothing to ask"
        );
    }

    /// **An answer becomes the summary the journal takes**: a rebuild of a
    /// dirty journal that is current once it is applied.
    #[test]
    fn an_answer_becomes_a_summary_the_journal_takes() {
        let mut t = sharratt();
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("~"),
                text: "Prefers tea.".into(),
            },
            now(),
        )
        .unwrap()
        .line
        .unwrap();
        let over = fingerprint(t.get(&p("~")).unwrap());
        let args = json!({
            "sentences": [{"text": " John prefers tea. ", "cite": format!("#{id}"), "also_cite": format!("#{id}")}],
            "digest": "Tea, not coffee.",
        });
        let op = op(&p("~"), &over, args.as_object().unwrap()).unwrap();
        apply(&mut t, &Actor::Zend, &op, now()).unwrap();
        let j = t.get(&p("~")).unwrap();
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
        let mut t = sharratt();
        assert_eq!(plan(&t, &p("~")), None, "nothing to summarise");
        let id = apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("~"),
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
        }) = plan(&t, &p("~"))
        else {
            panic!("a new line is asked about");
        };
        assert!(asked.contains("Prefers tea."));
        assert_eq!(over, fingerprint(t.get(&p("~")).unwrap()));
        let args = json!({
            "sentences": [{"text": "John prefers tea.", "cite": format!("#{id}")}],
            "digest": "Tea.",
        });
        apply(
            &mut t,
            &Actor::Zend,
            &op(&p("~"), &over, args.as_object().unwrap()).unwrap(),
            now(),
        )
        .unwrap();
        assert_eq!(plan(&t, &p("~")), None, "current");
        apply(
            &mut t,
            &john(),
            &Op::Forget {
                path: p("~"),
                which: Which::Id(id),
            },
            now(),
        )
        .unwrap();
        let Some(Plan::Clear(clear)) = plan(&t, &p("~")) else {
            panic!("nothing is left to cite");
        };
        apply(&mut t, &Actor::Zend, &clear, now()).unwrap();
        let j = t.get(&p("~")).unwrap();
        assert!(!is_dirty(j) && j.summary().unwrap().is_empty());
        assert_eq!(plan(&t, &p("~")), None);
        // The next line is asked about with no emptied summary to shape it.
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("~"),
                text: "Takes it black.".into(),
            },
            now(),
        )
        .unwrap();
        let q = question(&t, &p("~")).unwrap();
        assert!(!q.contains("The summary it had"), "{q}");
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
                op(&p("~"), &over, bad.as_object().unwrap()).is_err(),
                "{bad}"
            );
        }
    }
}
