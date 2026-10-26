//! Writing a piece in one sitting, with the brief and the sources open.
//!
//! **Where the writing happens is in the writing.** A Maker drafted a life
//! event the way it does everything else: one `file_write` among its turns,
//! the piece composed as an argument in the middle of its own running
//! conversation — the room it stood in, the colleague who just walked past, the
//! refusal it last read — with the brief somewhere far above. What came out
//! was that conversation: the vault's light rings and air handlers, the Makers
//! by name, a scene whose plot was somebody writing, a mood where an event
//! should be. Judged against the lore, nine pieces in eleven carried the vault.
//!
//! `compose` is the Maker sitting down to write. The sitting is a clean one: the
//! mind's writing voice (`writing` in `missions.yaml`) is put the whole of the
//! job at once — the brief, every document the research steps sent it to, in
//! full, and the piece as it stands when there is one — and answers with the
//! piece and nothing else. The Maker's own running conversation is not in it:
//! asked on a fork of that conversation, thirty thousand tokens of lifts,
//! desks and colleagues sat above the brief, and the pieces still carried the
//! vault. What it writes goes into the Maker's working set exactly as a
//! `file_write` would, through the same checks, and the Maker reads it back,
//! mends it and commits it.
//!
//! **A piece that stands is mended, not written again**
//! (`docs/npcd_anchored_edits.md` §7). Sent back to put a draft right, the
//! sitting is shown the piece under its paragraphs' anchors and answers with
//! `mend` — a list of edits naming the paragraphs it changes — which the engine
//! makes one by one: the decode is the changes, not the 800 words around them.
//! A sitting that has to change most of the piece answers with `draft`
//! instead, and writes it whole.

use candle_conversation::stencil::{Param, ThinkMode, ToolSpec};
use doc_units::render::units;
use doc_units::{apply, Doc};
use serde_json::{json, Map, Value};

use crate::engine::mission_gen::gates::Voice;
use crate::engine::{act, work};

/// The call a composition answers with.
pub const DRAFT: &str = "draft";

/// The most of one source document put in front of the writer, in words.
pub const SOURCE_WORDS: usize = 900;

/// Room for the longest piece the gates allow, and a little over.
pub const COMPOSE_TOKENS: usize = 2400;

/// How much a sitting thinks the piece through before writing it: a little,
/// so the piece is planned in the block rather than in its own sentences.
pub const THINK: ThinkMode = ThinkMode::Quick;

/// How many times one sitting writes: the piece, then — while it is short of
/// its floor — what comes next, up to this many in all. See [`fuller`].
pub const SITTINGS: usize = 3;

/// The one call a composition is written through: `draft`, whose `text` is the
/// whole piece.
pub fn spec() -> ToolSpec {
    let params: Vec<Param> =
        serde_json::from_value(json!([{ "name": "text", "type": "string", "required": true }]))
            .expect("a fixed parameter list");
    ToolSpec {
        name: DRAFT.to_string(),
        params,
    }
}

/// The piece, read out of a composition's decode — `None` when it holds no
/// `draft` call with any text in it.
pub fn read(raw: &str) -> Option<String> {
    let (_, args) = act::raw_calls(raw)
        .into_iter()
        .find(|(name, _)| name == DRAFT)?;
    match args.get("text") {
        Some(Value::String(s)) => Some(s.trim().to_string()).filter(|t| !t.is_empty()),
        _ => None,
    }
}

/// What the writer is put: the brief, every source in full (each cut at
/// [`SOURCE_WORDS`]), the piece as it stands when there is one, and what to
/// write.
///
/// **A piece that stands is in front of whoever writes it again.** A reviewer
/// told to write a draft again in the right voice was given the brief and the
/// sources but not the draft, so it wrote a different piece from nothing.
/// `standing` is the record's text, the one the table read: a Maker's working
/// copy holds what it last typed by hand, and given that the sitting wrote the
/// vault back in. It is `None` for a piece written anew — a draft the table
/// failed.
///
/// **The voice a life is told in is said, not left to be found.** The gate
/// holds a life event to the voice its other events are written in, and the
/// writer was never told which: a reviewer composed Marek's year in the third
/// person six sittings running, each refused at the report because every other
/// event of that life says "you". `voice` is that voice, when the piece is a
/// life event and the life has one.
pub fn question(
    brief: &str,
    writes: &str,
    min_words: usize,
    sources: &[(String, String)],
    standing: Option<&str>,
    voice: Option<Voice>,
) -> String {
    let mut q = format!("# Your brief\n\n{}\n", brief.trim());
    if !sources.is_empty() {
        q.push_str("\n# What you read for it\n");
        for (path, text) in sources {
            q.push_str(&format!("\n## `{path}`\n\n{}\n", text.trim()));
        }
    }
    let standing = standing.map(str::trim).filter(|t| !t.is_empty());
    if let Some(text) = standing {
        // Each paragraph under its anchor, as a read shows it: what a `mend`
        // names (`docs/npcd_anchored_edits.md` §7).
        let doc = Doc::new(writes, text);
        q.push_str(&format!(
            "\n# `{writes}` as it stands, each paragraph under its anchor\n\n{}",
            units(&doc, 0..doc.units().len())
        ));
    }
    let length = match min_words {
        0 => String::new(),
        n => format!(" At least {n} words."),
    };
    let told = match voice {
        Some(v) => format!(
            " Every other event of this life is told in {}: tell this one in it too, from the \
             first sentence to the last.",
            v.word()
        ),
        None => String::new(),
    };
    let job = match standing {
        Some(_) => format!(
            "Mend `{writes}`: keep everything in it that is right, and put right what your brief \
             says is wrong with it."
        ),
        None => format!(
            "Write `{writes}` whole — the piece your brief asks for, as it is to stand in the \
             record."
        ),
    };
    let answer = match standing {
        Some(_) => format!(
            "Answer with `{MEND}`, its `edits` the changes to make, each naming a paragraph by \
             its anchor above as `at` and saying what to `do` there: `replace` it with `text` — \
             or, with `old`, change just those words inside it — `insert_after` or \
             `insert_before` it a new paragraph `text`, or `delete` it; `to` carries a replace \
             or a delete on through the paragraphs to that anchor. Only if most of the piece has \
             to change, answer instead with `{DRAFT}`, its `text` the whole piece written again \
             and nothing but the piece."
        ),
        None => {
            format!("Answer with `{DRAFT}`, its `text` the whole piece and nothing but the piece.")
        }
    };
    // **The craft is said where the writing starts.** Put only in the voice,
    // above a brief whose "what happens" is itself a summary, it was not what
    // the writer followed: a life event came back as "the decision arrived not
    // as emotion but as arithmetic" and "the last thing Keeper processed was
    // the realization that…" — the summary, retold.
    q.push_str(&format!(
        "\n# Write it now\n\n{job}{length}{told} Build it from your brief and from what you read above: \
         its people, its places, its time. The brief tells you what happens; you show it \
         happening — what is done and said, moment by moment, as somebody there would see and \
         hear it — and never name a feeling or say what the moment means. {answer}"
    ));
    q
}

/// The call a mending sitting answers with: a list of anchored edits.
pub const MEND: &str = "mend";

/// The most edits one mending sitting makes. A mend that would change more is
/// a piece written again, which is `draft`.
pub const MOST_EDITS: usize = 12;

/// The call a mending sitting may answer with besides `draft`: `mend`, whose
/// `edits` each name a paragraph of the standing piece by one of `anchors` —
/// bound in the grammar, so an anchor the piece does not have cannot be
/// written — and say what to do there.
pub fn mend_spec(anchors: &[String]) -> ToolSpec {
    let params: Vec<Param> = serde_json::from_value(json!([{
        "name": "edits",
        "type": "array",
        "required": true,
        "min_items": 1,
        "max_items": MOST_EDITS,
        "items": {
            "type": "object",
            "properties": [
                { "name": "at", "type": "string", "required": true, "enum": anchors },
                {
                    "name": "do",
                    "type": "string",
                    "required": true,
                    "enum": ["replace", "insert_before", "insert_after", "delete"],
                    "requires": [
                        ["replace", ["text"]],
                        ["insert_before", ["text"]],
                        ["insert_after", ["text"]]
                    ]
                },
                { "name": "text", "type": "string" },
                { "name": "old", "type": "string" },
                { "name": "to", "type": "string", "enum": anchors }
            ]
        }
    }]))
    .expect("a fixed parameter list");
    ToolSpec {
        name: MEND.to_string(),
        params,
    }
}

/// The edits a mending sitting's decode answered with — `None` when it holds
/// no `mend` call, so the sitting wrote a `draft` or nothing.
pub fn read_mend(raw: &str) -> Option<Vec<Map<String, Value>>> {
    let (_, args) = act::raw_calls(raw)
        .into_iter()
        .find(|(name, _)| name == MEND)?;
    let edits = args.get("edits")?.as_array()?;
    Some(
        edits
            .iter()
            .filter_map(|e| e.as_object().cloned())
            .collect(),
    )
}

/// A mending sitting's edits, made one by one on the standing piece.
pub struct Mended {
    /// The piece after every edit that could be made.
    pub text: String,
    /// How many were made.
    pub made: usize,
    /// Why each of the rest was not.
    pub refused: Vec<String>,
}

/// Make `edits`, in order, on `doc`. An edit that names a paragraph an
/// earlier edit already changed finds its anchor gone and is refused, never
/// applied to the changed text.
pub fn mended(doc: &Doc, edits: &[Map<String, Value>]) -> Mended {
    let mut now = doc.clone();
    let mut made = 0;
    let mut refused = Vec::new();
    for args in edits {
        let edit = match work::edit_of(args) {
            Ok(edit) => edit,
            Err(why) => {
                refused.push(why);
                continue;
            }
        };
        match apply(&now, &edit) {
            Ok(done) => {
                now = done.doc;
                made += 1;
            }
            Err(e) => refused.push(e.to_string()),
        }
    }
    Mended {
        text: now.text(),
        made,
        refused,
    }
}

/// The refusal for a mending sitting none of whose edits could be made.
pub fn nothing_mended(writes: &str, refused: &[String]) -> String {
    format!(
        "You sat down to mend {writes} and none of your changes could be made: {}. Sit down to \
         it again.",
        refused.join("; ")
    )
}

/// The same sitting, carried on: the piece came to `words`, short of
/// `min_words`, and the writer goes on from where it stopped. What it answers
/// is what comes next, which [`joined`] puts after what it wrote.
///
/// **One sitting writes the whole piece.** Told "at least 380 words", the
/// writer stopped at 199 and 233, and the commit refused it for being short
/// — the Maker then padded it by hand, between other acts, which is the very
/// writing `compose` exists to replace. Asked to write it whole again at full
/// length, it wrote the same 203 words back three sittings running; asked to
/// go on, it has the rest of the scene to write and nothing to copy.
pub fn fuller(question: &str, draft: &str, words: usize, min_words: usize) -> String {
    format!(
        "{question}\n\n# What you have written so far\n\n{}\n\n# Go on\n\nThat is {words} words; \
         the piece is to be at least {min_words}. Carry the scene on from exactly where it \
         stops — what is done and said next, moment by moment, what it costs, and how the moment \
         ends differently than it began. Answer with `{DRAFT}`, its `text` only what comes next, \
         not what you have already written.",
        draft.trim()
    )
}

/// The refusal for sitting down to a piece already written: `words` of it
/// stand uncommitted in the working copy, up to its floor.
pub fn already_written(writes: &str, words: usize) -> String {
    format!(
        "{writes} is already written — {words} words in your working copy, not yet committed. \
         Read it back with `file_read`; if it stands, `bench_commit` it. Put one passage right \
         with `file_edit` if one is wrong. To start it over instead, throw the working copy away \
         with `bench_restore` first."
    )
}

/// Whether two pieces are the same words in the same order — spacing, line
/// breaks and a heading's `#` aside.
pub fn same_words(a: &str, b: &str) -> bool {
    let words = |t: &str| {
        t.split_whitespace()
            .filter(|w| !w.chars().all(|c| c == '#'))
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    words(a) == words(b)
}

/// The refusal for sitting down to a piece this mission has already committed.
pub fn already_committed(writes: &str) -> String {
    format!(
        "{writes} is written and committed — that step of your mission is done. What is left \
         is your report at the table."
    )
}

/// The refusal for a sitting that wrote the standing document back as it was.
pub fn unchanged(writes: &str) -> String {
    format!(
        "You wrote {writes} out again exactly as it stands — nothing in it changed, so there is \
         nothing to commit. If it can stand as it is, go back to the table and give your verdict. \
         If something in it is wrong, put that passage right with `file_edit`, or sit down again \
         and write it changed."
    )
}

/// The piece so far with what the writer went on to write after it — less
/// anything of it the piece already says.
///
/// **Carrying on is not starting over.** Asked to go on from where a piece
/// stopped, the writer wrote it again from its first line, and joined as it
/// came the piece said every passage twice: the gate refused it five faults
/// deep, the reviewer cut the copies, fell short of the floor, was carried on,
/// and wrote it all again — six sittings over one story.
pub fn joined(so_far: &str, next: &str) -> String {
    let said = |s: &str| {
        let s = normal(s);
        !s.is_empty() && normal(so_far).contains(&s)
    };
    let fresh: Vec<String> = next
        .trim()
        .split("\n\n")
        .filter_map(|para| {
            let para = para.trim();
            if para.starts_with('#') {
                return (!said(para)).then(|| para.to_string());
            }
            let kept: Vec<&str> = sentences(para).into_iter().filter(|s| !said(s)).collect();
            (!kept.is_empty()).then(|| kept.join(" "))
        })
        .collect();
    match fresh.is_empty() {
        true => so_far.trim_end().to_string(),
        false => format!("{}\n\n{}", so_far.trim_end(), fresh.join("\n\n")),
    }
}

/// A passage as [`joined`] compares it: its words, single-spaced.
fn normal(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A paragraph's sentences, each with its own closing mark.
fn sentences(para: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = para.as_bytes();
    for (i, c) in para.char_indices() {
        let ends = matches!(c, '.' | '!' | '?')
            && bytes.get(i + 1).is_none_or(|b| b.is_ascii_whitespace());
        if ends {
            let s = para[start..=i].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = i + 1;
        }
    }
    let rest = para[start..].trim();
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// `piece` without a first line that only names the file it is written to —
/// the path, or its file name, bare or as a heading.
///
/// **The question names the file, and the piece echoed it.** Told "write
/// `layers/life/conan-the-eloquent-barbarian/2950-06-12 The Unlocked Door.md`
/// whole", the writer opened the life event with that path as its first line,
/// and the table passed it with the path standing in the record.
pub fn unlabelled(piece: &str, writes: &str) -> String {
    let file = writes.rsplit('/').next().unwrap_or(writes);
    let stem = file.strip_suffix(".md").unwrap_or(file);
    let mut lines = piece.trim_start().splitn(2, '\n');
    let first = lines.next().unwrap_or_default();
    let named = first
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_matches('`');
    if [writes, file, stem].contains(&named) {
        return lines.next().unwrap_or_default().trim_start().to_string();
    }
    // **Or as a label on the first sentence**: "layers/life/marek-the-ordnance/
    // 2805 The First Refusal.md: The command post smelled of ionized air…"
    let rest = lines.next();
    let labelled = [writes, file, stem].iter().find_map(|name| {
        named
            .strip_prefix(name)
            .and_then(|after| after.trim_start_matches('`').strip_prefix(':'))
    });
    match labelled {
        Some(sentence) => match rest {
            Some(r) => format!("{}\n{r}", sentence.trim_start()),
            None => sentence.trim_start().to_string(),
        },
        None => piece.to_string(),
    }
}

/// `text` cut to about `words` words.
pub fn cut(text: &str, words: usize) -> String {
    let all: Vec<&str> = text.split_whitespace().collect();
    match all.len() > words {
        true => format!("{} …", all[..words].join(" ")),
        false => text.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_writer_is_put_the_brief_the_sources_and_the_job() {
        let q = question(
            "Write Zen's life where the record has nothing: the year 2491.",
            "layers/life/zen/2491 The Count.md",
            250,
            &[(
                "layers/eras/the-revenge-fleet.md".into(),
                "# The Revenge Fleet\n\nThe fleet arrived in 2487.".into(),
            )],
            None,
            Some(Voice::Second),
        );
        assert!(q.starts_with("# Your brief\n\nWrite Zen's life where the record has nothing"));
        // The voice the life is told in is said with the job.
        assert!(q.contains(
            "At least 250 words. Every other event of this life is told in the second person \
             (\"you\"): tell this one in it too, from the first sentence to the last."
        ));
        assert!(q.contains("## `layers/eras/the-revenge-fleet.md`\n\n# The Revenge Fleet"));
        assert!(q.contains("Write `layers/life/zen/2491 The Count.md` whole — the piece"));
        assert!(!q.contains("as it stands"));
        assert!(q.contains("At least 250 words."));
        // The craft is said at the point of writing, after the brief.
        assert!(q.contains(
            "The brief tells you what happens; you show it happening — what is done and said, \
             moment by moment, as somebody there would see and hear it — and never name a \
             feeling or say what the moment means."
        ));
        assert!(q.ends_with("its `text` the whole piece and nothing but the piece."));
    }

    /// **The same words are the same piece**, however they are spaced.
    #[test]
    fn a_piece_written_back_unchanged_is_known() {
        let standing =
            "# The First Refusal\n\nThe command post smelled of ionized air.\n\nMarek waited.";
        assert!(same_words(
            standing,
            "The First Refusal\nThe command post smelled of ionized air. Marek waited.\n"
        ));
        assert!(!same_words(
            standing,
            "The command post smelled of ionized air. Marek waited."
        ));
        assert!(!same_words(
            standing,
            "# The First Refusal\n\nThe post smelled of ionized air.\n\nMarek waited."
        ));
        assert!(unchanged("x.md").starts_with("You wrote x.md out again exactly as it stands"));
        assert_eq!(
            already_committed("x.md"),
            "x.md is written and committed — that step of your mission is done. What is left is \
             your report at the table."
        );
    }

    #[test]
    fn a_piece_written_again_is_in_front_of_its_writer() {
        let q = question(
            "Mend the draft: it is told as summary.",
            "layers/stories/the-count.md",
            380,
            &[],
            Some("  You count the fleet.  "),
            None,
        );
        assert!(
            !q.contains("Every other event"),
            "a story has no life's voice"
        );
        // The piece is shown under its anchors, and mended by them.
        let a = &Doc::new("layers/stories/the-count.md", "You count the fleet.").anchors()[0];
        assert!(
            q.contains(&format!(
                "\n# `layers/stories/the-count.md` as it stands, each paragraph under its \
                 anchor\n\n{a}| You count the fleet.\n"
            )),
            "{q}"
        );
        assert!(q.contains(
            "Mend `layers/stories/the-count.md`: keep everything in it that is right, and put \
             right what your brief says is wrong with it."
        ));
        assert!(
            q.contains("Answer with `mend`, its `edits` the changes to make"),
            "{q}"
        );
        assert!(q.contains("answer instead with `draft`"), "{q}");
        let blank = question("b", "x.md", 0, &[], Some("  "), None);
        assert!(!blank.contains("as it stands"), "{blank}");
        assert!(!blank.contains("`mend`"), "{blank}");
    }

    /// **A mend is read from its call and made edit by edit**, and an edit
    /// that names a paragraph an earlier one changed is refused rather than
    /// applied to the changed text.
    #[test]
    fn a_mend_is_read_and_made_edit_by_edit() {
        let piece = "The bay was warm.\n\nThe order came down.\n\nYou said fourteen.\n";
        let doc = Doc::new("x.md", piece);
        let [a, b, c] = [0, 1, 2].map(|i| doc.units()[i].anchor.clone());
        let raw = format!(
            "<tool_call>\n{{\"name\": \"mend\", \"arguments\": {{\"edits\": [\
             {{\"at\": \"{c}\", \"do\": \"replace\", \"old\": \"fourteen\", \"text\": \"fifteen\"}}, \
             {{\"at\": \"{b}\", \"do\": \"delete\"}}, \
             {{\"at\": \"{c}\", \"do\": \"replace\", \"text\": \"Again.\"}}, \
             {{\"at\": \"{a}\", \"do\": \"insert_before\", \"text\": \"Dawn.\"}}]}}}}\n</tool_call>"
        );
        let edits = read_mend(&raw).expect("a mend call");
        assert_eq!(edits.len(), 4);
        let m = mended(&doc, &edits);
        assert_eq!(m.text, "Dawn.\n\nThe bay was warm.\n\nYou said fifteen.\n");
        assert_eq!(m.made, 3);
        assert_eq!(m.refused.len(), 1);
        assert!(m.refused[0].contains("read it again"), "{:?}", m.refused);
        assert_eq!(read_mend("<tool_call>\n{\"name\": \"draft\", \"arguments\": {\"text\": \"x\"}}\n</tool_call>"), None);
    }

    /// The mend's grammar binds `at` and `to` to the piece's own anchors.
    #[test]
    fn a_mends_anchors_are_the_pieces_own() {
        let anchors = vec!["Qm7x".to_string(), "b4_K".to_string()];
        let spec = mend_spec(&anchors);
        assert_eq!(spec.name, MEND);
        let items = spec.params[0].items.as_ref().unwrap();
        let fields = items.properties.as_ref().unwrap();
        let at = fields.iter().find(|f| f.name == "at").unwrap();
        assert_eq!(at.enum_values.as_deref(), Some(&anchors[..]));
        assert!(at.required);
        let to = fields.iter().find(|f| f.name == "to").unwrap();
        assert_eq!(to.enum_values.as_deref(), Some(&anchors[..]));
        assert!(!to.required);
        let act = fields.iter().find(|f| f.name == "do").unwrap();
        assert_eq!(act.requires.len(), 3);
        assert_eq!(spec.params[0].max_items, Some(MOST_EDITS));
    }

    #[test]
    fn the_piece_is_read_from_its_call() {
        let raw = r#"<tool_call>
{"name": "draft", "arguments": {"text": "  You count the fleet as it comes in.  "}}
</tool_call>"#;
        assert_eq!(
            read(raw).as_deref(),
            Some("You count the fleet as it comes in.")
        );
        assert_eq!(read("no call at all"), None);
        let empty = r#"<tool_call>
{"name": "draft", "arguments": {"text": "  "}}
</tool_call>"#;
        assert_eq!(read(empty), None);
    }

    #[test]
    fn a_short_piece_is_carried_on_in_the_same_sitting() {
        let q = fuller("# Your brief\n\nWrite it.", "You count the fleet.", 4, 380);
        assert!(q.starts_with(
            "# Your brief\n\nWrite it.\n\n# What you have written so far\n\nYou count the fleet."
        ));
        assert!(
            q.contains("That is 4 words; the piece is to be at least 380."),
            "{q}"
        );
        assert!(q.contains("its `text` only what comes next"), "{q}");
        assert_eq!(
            joined("You count the fleet.\n", "  The last ship docks.  "),
            "You count the fleet.\n\nThe last ship docks."
        );
    }

    /// **Carrying on adds only what is new**: a continuation that starts the
    /// piece over keeps none of what the piece already says, heading included.
    #[test]
    fn a_continuation_that_starts_over_adds_only_what_is_new() {
        let so_far = "# The Ledger\n\nAutumn, 3012. The vault drips.\n\nElian holds the blueprint.";
        let again =
            "# The Ledger\n\nAutumn, 3012.  The vault drips. Kael takes it.\n\nElian holds \
                     the blueprint.\n\nThe door shuts behind them.";
        assert_eq!(
            joined(so_far, again),
            "# The Ledger\n\nAutumn, 3012. The vault drips.\n\nElian holds the blueprint.\n\nKael \
             takes it.\n\nThe door shuts behind them."
        );
        assert_eq!(joined(so_far, so_far), so_far, "nothing new, nothing added");
    }

    #[test]
    fn a_written_piece_is_pointed_at_its_commit() {
        assert_eq!(
            already_written("layers/stories/x.md", 561),
            "layers/stories/x.md is already written — 561 words in your working copy, not yet \
             committed. Read it back with `file_read`; if it stands, `bench_commit` it. Put one \
             passage right with `file_edit` if one is wrong. To start it over instead, throw the \
             working copy away with `bench_restore` first."
        );
    }

    #[test]
    fn a_first_line_that_names_the_file_is_dropped() {
        const W: &str = "layers/life/conan/2950-06-12 The Unlocked Door.md";
        for first in [
            "layers/life/conan/2950-06-12 The Unlocked Door.md",
            "# 2950-06-12 The Unlocked Door",
            "`2950-06-12 The Unlocked Door.md`",
        ] {
            assert_eq!(
                unlabelled(&format!("{first}\n\nYou open the door."), W),
                "You open the door.",
                "{first}"
            );
        }
        let titled = "# The Unlocked Door\n\nYou open the door.";
        assert_eq!(unlabelled(titled, W), titled);
        // A name glued on the first sentence as its label goes; the sentence stays.
        assert_eq!(
            unlabelled(&format!("{W}: You open the door.\n\nIt is dark."), W),
            "You open the door.\n\nIt is dark."
        );
        assert_eq!(
            unlabelled("`2950-06-12 The Unlocked Door.md`: You open the door.", W),
            "You open the door."
        );
        let colon = "The Unlocked Door: you open it.";
        assert_eq!(
            unlabelled(colon, W),
            colon,
            "a title is not the file's name"
        );
    }

    #[test]
    fn a_long_source_is_cut() {
        assert_eq!(cut("a b c d e", 3), "a b c …");
        assert_eq!(cut("a b", 3), "a b");
    }
}
