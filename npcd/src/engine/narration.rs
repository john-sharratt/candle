//! Deterministic narration — the narrator's `YOU`/`STATE`/`EVENTS` input text
//! turned into second-/third-person prose by a pure function, in place of an
//! LLM decode.
//!
//! # Why a function and not a decode
//!
//! The narrator's job is mechanical: every event line is already a structured
//! record — an actor, a relation (`speaks to you`, `tell`, `whispers to you`,
//! `speaks to X (you overhear)`), and a `meaning` quoted in the actor's own
//! first person. Turning that into what the focal character reads is three fixed
//! operations: strip the framing, **re-voice the first-person meaning** (the
//! actor's `I`/`my` becomes `you`/`your` for the focal character's own acts, or
//! `they`/`their` for anyone else), and render one clean sentence. A decode did
//! this per tick, per character — two extra decodes on the hot path — for a
//! transformation with no real freedom in it. This module does it in microseconds.
//!
//! # Point of view
//!
//! Second-person-focal, matching the event text: the focal character is already
//! written as the literal token `you` in every line the world builds
//! (`you — tell X`, `X — speaks to you`), so the render keys off that token and
//! needs no name. Everyone else is third person by the name the line carries.
//! Where a speaker's gender is not known, the neutral singular `they` is used —
//! it is always correct and never invents a gender the world did not state.
//!
//! # Faithful
//!
//! Nothing is added that the events do not contain: no scene-setting, no sensory
//! detail, no physical sketch of the focal character (the LLM padded every tick
//! with the same description; the events are the content). Bare lines the world
//! already rendered — movement, ambient, a whisper seen but not heard, a
//! remembered feeling — pass through untouched but for capitalisation.

/// The voice a first-person `meaning` is re-voiced into: the focal character's
/// own act reads as `you`; anyone else's speech reads as `they`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Voice {
    You,
    They,
}

/// Re-voice a fragment written in the actor's first person into `voice`.
///
/// Generalised word-for-word: it maps the first-person pronoun set and the two
/// verbs that inflect with it (`am`/`was`) and leaves everything else — the
/// third parties the actor mentions, the `we`/`our` that legitimately includes
/// the listener — untouched. Word boundaries are respected (the `I` in `Ihram`
/// is not a pronoun), and sentence-initial capitalisation is restored at the end
/// so a multi-sentence meaning still reads correctly.
fn revoice(text: &str, voice: Voice) -> String {
    // Normalise the curly apostrophe to the straight one so a contraction
    // written `they've` maps the same as `they've`.
    let normalized = text.replace('\u{2019}', "'");

    // Whole-word, lower-cased key → replacement for this voice. `I` and its
    // contractions are the only always-capitalised source words; every mapping
    // produces a lower-case replacement and the capitalisation pass fixes the
    // sentence starts afterwards.
    //
    // `am`/`was` are the two verbs that inflect with the subject, and they are
    // the one place a blind swap is wrong: both pair with a subject, and `was`
    // also pairs with `she`/`he`/`it`, so they inflect ONLY when the running
    // subject is the first person being re-voiced. `subj_i` tracks that subject:
    // set true by `I`, cleared by any other subject/object pronoun, and kept
    // across the words between — so `I have returned and am ready` inflects the
    // elided-subject `am`, while `she was` and `I saw she was gone` leave `was`
    // alone. (A proper-noun subject, `Pax was`, is the residual edge this does
    // not catch; it never turns a pronoun subject's verb wrong, which is the one
    // that would read as broken.)
    let map = |w: &str, subj_i: bool| -> Option<&'static str> {
        let lower = w.to_ascii_lowercase();
        Some(match (voice, lower.as_str()) {
            (Voice::You, "i") => "you",
            (Voice::You, "me") => "you",
            (Voice::You, "my") => "your",
            (Voice::You, "mine") => "yours",
            (Voice::You, "myself") => "yourself",
            (Voice::You, "i'm") => "you're",
            (Voice::You, "i've") => "you've",
            (Voice::You, "i'll") => "you'll",
            (Voice::You, "i'd") => "you'd",
            (Voice::They, "i") => "they",
            (Voice::They, "me") => "them",
            (Voice::They, "my") => "their",
            (Voice::They, "mine") => "theirs",
            (Voice::They, "myself") => "themselves",
            (Voice::They, "i'm") => "they're",
            (Voice::They, "i've") => "they've",
            (Voice::They, "i'll") => "they'll",
            (Voice::They, "i'd") => "they'd",
            (_, "am") if subj_i => "are",
            (_, "was") if subj_i => "were",
            _ => return None,
        })
    };

    let mut out = String::with_capacity(normalized.len() + 8);
    let mut subj_i = false;
    for tok in tokenize(&normalized) {
        match tok {
            Token::Word(w) => {
                out.push_str(map(w, subj_i).unwrap_or(w));
                let lower = w.to_ascii_lowercase();
                if lower == "i" {
                    subj_i = true;
                } else if matches!(
                    lower.as_str(),
                    "he" | "she" | "it" | "they" | "we" | "you" | "him" | "her" | "them" | "us"
                ) {
                    subj_i = false;
                }
                // Any other word (a verb, `and`, an adverb) leaves the subject as
                // it stands, so a coordinated `am`/`was` still sees its `I`.
            }
            Token::Other(s) => out.push_str(s),
        }
    }
    recapitalize_sentences(&out)
}

enum Token<'a> {
    /// A run of word characters (letters plus the apostrophe inside a
    /// contraction, so `I'm` is one token).
    Word(&'a str),
    /// Anything between words — spaces, punctuation, dashes.
    Other(&'a str),
}

/// Split into alternating word / non-word runs. A word is letters, with an
/// apostrophe kept when it sits between letters (`they're`), so contractions
/// tokenise whole.
fn tokenize(s: &str) -> Vec<Token<'_>> {
    let bytes = s.as_bytes();
    let is_word = |i: usize| -> bool {
        let c = bytes[i];
        c.is_ascii_alphabetic()
            || (c == b'\''
                && i > 0
                && i + 1 < bytes.len()
                && bytes[i - 1].is_ascii_alphabetic()
                && bytes[i + 1].is_ascii_alphabetic())
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let word = is_word(i);
        while i < bytes.len() && is_word(i) == word {
            i += 1;
        }
        let run = &s[start..i];
        out.push(if word {
            Token::Word(run)
        } else {
            Token::Other(run)
        });
    }
    out
}

/// Capitalise the first letter of the text and of every sentence after a
/// terminator (`.`/`!`/`?`), so re-voiced pronouns that landed at a sentence
/// start read correctly.
fn recapitalize_sentences(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_start = true;
    for c in s.chars() {
        if at_start && c.is_alphabetic() {
            out.extend(c.to_uppercase());
            at_start = false;
        } else {
            out.push(c);
            if matches!(c, '.' | '!' | '?') {
                at_start = true;
            } else if !c.is_whitespace() {
                at_start = false;
            }
        }
    }
    out
}

/// Lower-case only the first letter, for splicing a `meaning` into the middle of
/// a sentence (`asks you` + `If the gaps…` → `asks you if the gaps…`).
fn uncapitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_lowercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// End a sentence with a period unless it already ends in a terminator.
fn ensure_terminal(mut s: String) -> String {
    let t = s.trim_end();
    if t.is_empty() {
        return String::new();
    }
    if !t.ends_with(['.', '!', '?', '…']) {
        s.truncate(t.len());
        s.push('.');
    } else {
        s.truncate(t.len());
    }
    s
}

/// Render the whole narrator input (the `YOU`/`STATE`/`EVENTS` text
/// [`crate::engine::narrator::build_turn`] produces) into prose. The `YOU` and
/// `STATE` headers are context for the model that no longer decodes; the render
/// works purely from the `EVENTS` list, each line rendered and the results
/// joined into one short passage.
pub fn render(input: &str) -> String {
    let events = input
        .split_once("EVENTS:\n")
        .map(|(_, rest)| rest)
        .unwrap_or(input);
    let mut sentences: Vec<String> = Vec::new();
    for line in events.lines() {
        let line = line.trim();
        // Strip a leading `N.` ordinal the builder numbers events with, whether
        // or not a space follows it (a bare `1.` is an empty event).
        let digits = line.bytes().take_while(|b| b.is_ascii_digit()).count();
        let content = if digits > 0 && line.as_bytes().get(digits) == Some(&b'.') {
            line[digits + 1..].trim_start()
        } else {
            line
        };
        if content.is_empty() {
            continue;
        }
        if let Some(s) = render_event(content) {
            sentences.push(s);
        }
    }
    sentences.join(" ")
}

/// Render one event line. Structured speech/act lines (`actor — relation —
/// meaning: "…"`) are re-voiced and framed; every other line is already prose
/// the world rendered and passes through with a capital and a terminator.
fn render_event(content: &str) -> Option<String> {
    if let Some(rendered) = render_structured(content) {
        return Some(ensure_terminal(rendered));
    }
    // Bare line: a movement, an ambient beat, a whisper seen but not heard, a
    // remembered feeling, an announcement, a new day. Already prose — keep it,
    // only fixing the leading capital and the terminator.
    let bare = content.trim();
    if bare.is_empty() {
        return None;
    }
    Some(ensure_terminal(recapitalize_sentences(bare)))
}

/// Parse and render `actor — relation — meaning: "<meaning>"`. Returns `None`
/// when the line is not in that shape (so the caller passes it through bare).
fn render_structured(content: &str) -> Option<String> {
    let (head, meaning) = content.split_once(" — meaning: \"")?;
    let meaning = meaning.strip_suffix('"').unwrap_or(meaning).trim();
    let (actor, relation) = head.split_once(" — ")?;
    let actor = actor.trim();
    let relation = relation.trim();

    // The focal character's own act — actor is the literal token `you`.
    if actor.eq_ignore_ascii_case("you") {
        // A line called after somebody leaving the room carries the `(leaving)`
        // marker (see `mind::own_act_line`): the same act, framed as a parting
        // word thrown after them rather than one spoken face to face.
        let (departing, relation) = match relation.strip_prefix("(leaving) ") {
            Some(rest) => (true, rest.trim()),
            None => (false, relation),
        };
        let (verb, target) = split_verb_target(relation)?;
        let m = uncapitalize_first(&revoice(meaning, Voice::You));
        if departing {
            // Name them once in the parting clause, then refer back with "them"
            // so the sentence does not repeat the name — keeping any "to" the
            // verb needs ("you whisper to them", "you tell them").
            let name = target.strip_prefix("to ").unwrap_or(target);
            let them = if target.starts_with("to ") {
                "to them"
            } else {
                "them"
            };
            return Some(format!("As {name} walks away, you {verb} {them} {m}"));
        }
        // tell, ask, whisper-to and any other directed own-act read the same
        // way: the verb, whom it was aimed at, then the re-voiced meaning.
        return Some(format!("You {verb} {target} {m}"));
    }

    // Someone else. Re-voice their first person to the neutral third person.
    // A meaning the world tagged as a question ("asking …") reads as a question;
    // otherwise the meaning carries its own leading "that" conjunction, so the
    // frame never adds one — it only splices the meaning in mid-sentence
    // (lower-cased first letter).
    let revoiced = revoice(meaning, Voice::They);
    let (asking, body) = match revoiced
        .strip_prefix("Asking ")
        .or_else(|| revoiced.strip_prefix("asking "))
    {
        Some(rest) => (true, uncapitalize_first(rest)),
        None => (false, uncapitalize_first(&revoiced)),
    };

    let rendered = if relation == "speaks to you" {
        if asking {
            format!("{actor} asks you {body}")
        } else {
            format!("{actor} tells you {body}")
        }
    } else if relation == "whispers to you" {
        format!("{actor} whispers to you {body}")
    } else if relation == "speaks to the room" || relation == "shout to the room" {
        format!("{actor} calls out to the room {body}")
    } else if let Some(who) = relation
        .strip_prefix("speaks to ")
        .and_then(|r| r.strip_suffix(" (you overhear)"))
    {
        // Overheard speech to a third party.
        if asking {
            format!("You overhear {actor} ask {who} {body}")
        } else {
            format!("You overhear {actor} tell {who} {body}")
        }
    } else if let Some(rest) = relation.strip_prefix("messages you") {
        // `messages you` or `messages you (on <thread>)`.
        let on = rest.trim();
        if on.is_empty() {
            format!("{actor} messages you {body}")
        } else {
            format!("{actor} messages you {on} {body}")
        }
    } else {
        // An unknown relation with a meaning — render it plainly rather than drop
        // the content.
        format!("{actor} {relation}: {body}")
    };
    Some(rendered)
}

/// Split an own-act relation like `tell Pax Veridian` into `("tell", "Pax
/// Veridian")`. `None` when there is no target after the verb.
fn split_verb_target(relation: &str) -> Option<(&str, &str)> {
    let (verb, target) = relation.split_once(' ')?;
    let target = target.trim();
    (!target.is_empty()).then_some((verb, target))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generalised re-voicing, on its own: the actor's first person becomes
    /// `you` for their own act and `they` for anyone else, the `to be` verbs
    /// inflect, and third parties and `we`/`our` are left alone.
    #[test]
    fn revoice_maps_first_person_both_ways() {
        assert_eq!(
            revoice("that I am ready and my work is done", Voice::You),
            "That you are ready and your work is done"
        );
        assert_eq!(
            revoice("that I am ready and my work is done", Voice::They),
            "That they are ready and their work is done"
        );
        // Third parties (she) and the shared we/our survive untouched.
        assert_eq!(
            revoice(
                "that I need to know if she noticed what we lost",
                Voice::They
            ),
            "That they need to know if she noticed what we lost"
        );
        // A word that merely contains a pronoun's letters is not a pronoun.
        assert_eq!(
            revoice("the image is mine", Voice::They),
            "The image is theirs"
        );
        assert_eq!(revoice("Ihram and iron", Voice::They), "Ihram and iron");
    }

    /// A meaning spanning two sentences re-capitalises the second one's start.
    #[test]
    fn revoice_recapitalizes_later_sentences() {
        assert_eq!(
            revoice("I have finished. I am ready for the next.", Voice::They),
            "They have finished. They are ready for the next."
        );
    }

    // ── Whole-line renders, expected outputs designed from the captured LLM
    //    behaviour but cleaner (no invented scene-setting or sketch padding). ──

    #[test]
    fn own_tell_is_second_person() {
        let line = "you — tell Pax Veridian — meaning: \"that I need to know what has changed in \
                    the data stream since our last conversation.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "You tell Pax Veridian that you need to know what has changed in the data stream \
             since our last conversation."
        );
    }

    #[test]
    fn own_ask_is_second_person() {
        let line = "you — ask Vael Fane — meaning: \"whether the pauses are widening or \
                    stabilizing, and what that means for the integrity of the record.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "You ask Vael Fane whether the pauses are widening or stabilizing, and what that \
             means for the integrity of the record."
        );
    }

    /// **A line called after somebody leaving is framed as a parting word.**
    ///
    /// The `(leaving)` marker (`mind::own_act_line`) turns the ordinary "You tell
    /// X …" into "As X walks away, you tell them …": the same act, voiced as one
    /// last thing thrown after somebody who has just stepped out. The addressee
    /// is named once in the parting clause and referred back to as "them", so the
    /// name is not repeated.
    #[test]
    fn own_tell_after_someone_leaving_is_a_parting_line() {
        let line = "you — (leaving) tell Vael Fane — meaning: \"that I need to know what has \
                    changed in the record since we last spoke.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "As Vael Fane walks away, you tell them that you need to know what has changed in \
             the record since we last spoke."
        );
    }

    #[test]
    fn own_ask_after_someone_leaving_keeps_the_to_and_the_question() {
        let line = "you — (leaving) ask Vael Fane — meaning: \"whether the pauses are widening or \
                    stabilizing.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "As Vael Fane walks away, you ask them whether the pauses are widening or stabilizing."
        );
    }

    /// A whisper's own-act line carries the `to` — "whisper to X" — and the
    /// parting frame keeps it, so the pronoun is "to them", not a bare "them".
    #[test]
    fn own_whisper_after_someone_leaving_keeps_its_to() {
        let line = "you — (leaving) whisper to Vael Fane — meaning: \"that I do not trust the \
                    record.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "As Vael Fane walks away, you whisper to them that you do not trust the record."
        );
    }

    /// The marker changes only the framing, never the ordinary case: an act with
    /// no `(leaving)` still reads as a word spoken face to face.
    #[test]
    fn an_ordinary_own_tell_is_not_given_a_parting_frame() {
        let line = "you — tell Vael Fane — meaning: \"that the gap is filled.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "You tell Vael Fane that the gap is filled."
        );
        assert!(!render_event(line).unwrap().contains("walks away"));
    }

    #[test]
    fn speaks_to_you_is_third_person_reported() {
        let line = "Pax Veridian — speaks to you — meaning: \"that I have finished the current \
                    piece and am ready for the next.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "Pax Veridian tells you that they have finished the current piece and are ready for \
             the next."
        );
    }

    #[test]
    fn speaks_to_you_asking_is_a_question() {
        let line = "Ulysses Thorne — speaks to you — meaning: \"asking whether the gaps in the \
                    record are widening or have stabilized.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "Ulysses Thorne asks you whether the gaps in the record are widening or have \
             stabilized."
        );
    }

    #[test]
    fn overheard_speech_is_reported_as_overheard() {
        let line = "Ulysses Thorne — speaks to Pax Veridian (you overhear) — meaning: \"that I \
                    have returned and am ready to resume work, specifically the discrepancies we \
                    noticed last time.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "You overhear Ulysses Thorne tell Pax Veridian that they have returned and are ready \
             to resume work, specifically the discrepancies we noticed last time."
        );
    }

    #[test]
    fn whisper_to_you_is_heard() {
        let line = "Vael Fane — whispers to you — meaning: \"that I do not trust the record.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "Vael Fane whispers to you that they do not trust the record."
        );
    }

    #[test]
    fn a_shout_reaches_the_room() {
        let line = "Vael Fane — shout to the room — meaning: \"that we must stop and investigate \
                    who is erasing the record.\"";
        assert_eq!(
            render_event(line).unwrap(),
            "Vael Fane calls out to the room that we must stop and investigate who is erasing \
             the record."
        );
    }

    /// Bare lines — movement, ambient, a whisper seen, a remembered feeling —
    /// pass through with only a capital and a terminator.
    #[test]
    fn bare_lines_pass_through() {
        assert_eq!(
            render_event("Pax Veridian came in.").unwrap(),
            "Pax Veridian came in."
        );
        assert_eq!(
            render_event("Ulysses Thorne left.").unwrap(),
            "Ulysses Thorne left."
        );
        assert_eq!(
            render_event("Ulysses Thorne whispered something to Vael Fane.").unwrap(),
            "Ulysses Thorne whispered something to Vael Fane."
        );
        assert_eq!(
            render_event("Machinery starts up somewhere far off and settles into a steady note.")
                .unwrap(),
            "Machinery starts up somewhere far off and settles into a steady note."
        );
        assert_eq!(
            render_event("What you were feeling, when you last stopped to notice, was exultant.")
                .unwrap(),
            "What you were feeling, when you last stopped to notice, was exultant."
        );
        // A new-day beat gains its capital and terminator.
        assert_eq!(
            render_event("a new day begins — day 20714").unwrap(),
            "A new day begins — day 20714."
        );
    }

    /// A whole multi-event turn: the `YOU`/`STATE` headers are dropped and the
    /// events render in order, joined into one passage.
    #[test]
    fn a_full_turn_renders_from_events_only() {
        let input = "YOU: Vael Fane — a slender figure.\n\
                     STATE: You are at the lift. Near you: Pax Veridian, Ulysses Thorne.\n\
                     EVENTS:\n\
                     1. Pax Veridian came in.\n\
                     2. Ulysses Thorne — speaks to Pax Veridian (you overhear) — meaning: \"that \
                     I have returned and am ready to resume work.\"\n\
                     3. Pax Veridian — speaks to you — meaning: \"that I have finished the piece.\"\n";
        assert_eq!(
            render(input),
            "Pax Veridian came in. You overhear Ulysses Thorne tell Pax Veridian that they have \
             returned and are ready to resume work. Pax Veridian tells you that they have \
             finished the piece."
        );
    }

    // ────────────────────────── revoice: pronouns ──────────────────────────

    #[test]
    fn revoice_you_covers_every_pronoun() {
        assert_eq!(
            revoice("that I need me and my mine and myself", Voice::You),
            "That you need you and your yours and yourself"
        );
        assert_eq!(
            revoice("I'm and I've and I'll and I'd", Voice::You),
            "You're and you've and you'll and you'd"
        );
    }

    #[test]
    fn revoice_they_covers_every_pronoun() {
        assert_eq!(
            revoice("that I need me and my mine and myself", Voice::They),
            "That they need them and their theirs and themselves"
        );
        assert_eq!(
            revoice("I'm and I've and I'll and I'd", Voice::They),
            "They're and they've and they'll and they'd"
        );
    }

    // ───────────────────── revoice: am / was subject gate ──────────────────

    #[test]
    fn revoice_inflects_to_be_only_for_the_first_person() {
        assert_eq!(revoice("I am ready", Voice::You), "You are ready");
        assert_eq!(revoice("I am ready", Voice::They), "They are ready");
        assert_eq!(revoice("I was ready", Voice::You), "You were ready");
        assert_eq!(revoice("I was ready", Voice::They), "They were ready");
    }

    #[test]
    fn revoice_leaves_third_person_was_alone() {
        // The bug this guards: a blind was→were turns "she was" into "she were".
        assert_eq!(revoice("she was ready", Voice::They), "She was ready");
        assert_eq!(revoice("he was here", Voice::They), "He was here");
        assert_eq!(revoice("it was broken", Voice::They), "It was broken");
        // I and a third party in one sentence: only I's verb inflects.
        assert_eq!(
            revoice("he was here and I was too", Voice::They),
            "He was here and they were too"
        );
    }

    #[test]
    fn revoice_leaves_we_and_were_alone() {
        assert_eq!(
            revoice("we were ready and our work is ours", Voice::They),
            "We were ready and our work is ours"
        );
        assert_eq!(
            revoice("we were ready and our work is ours", Voice::You),
            "We were ready and our work is ours"
        );
    }

    // ─────────────────── revoice: names & third parties kept ────────────────

    #[test]
    fn revoice_keeps_names_and_third_person() {
        assert_eq!(
            revoice("that I told Pax I trust him and his plan", Voice::You),
            "That you told Pax you trust him and his plan"
        );
        assert_eq!(
            revoice("she and I left with them", Voice::They),
            "She and they left with them"
        );
    }

    #[test]
    fn revoice_only_maps_whole_words() {
        // Words that merely contain a pronoun's letters are untouched.
        assert_eq!(
            revoice("Ihram and iron and India and image", Voice::They),
            "Ihram and iron and India and image"
        );
    }

    // ───────────────────── revoice: capitals & sentences ────────────────────

    #[test]
    fn revoice_recapitalizes_first_and_each_sentence() {
        assert_eq!(revoice("my work", Voice::You), "Your work");
        assert_eq!(
            revoice("I went. I saw my friend.", Voice::They),
            "They went. They saw their friend."
        );
        // A question terminator also starts a new sentence.
        assert_eq!(
            revoice("is it me? I think so.", Voice::You),
            "Is it you? You think so."
        );
    }

    #[test]
    fn revoice_normalizes_the_curly_apostrophe() {
        assert_eq!(
            revoice("I\u{2019}m and I\u{2019}ve", Voice::They),
            "They're and they've"
        );
    }

    // ─────────────────────── own acts (second person) ───────────────────────

    #[test]
    fn own_tell_and_ask_variants() {
        assert_eq!(
            render_event("you — tell Pax Veridian — meaning: \"that I am ready\"").unwrap(),
            "You tell Pax Veridian that you are ready."
        );
        assert_eq!(
            render_event("you — ask Pax — meaning: \"whether it works\"").unwrap(),
            "You ask Pax whether it works."
        );
        // Other names and third-person pronouns inside an own act survive.
        assert_eq!(
            render_event("you — tell Pax — meaning: \"that I trust Ulysses and I need his help\"")
                .unwrap(),
            "You tell Pax that you trust Ulysses and you need his help."
        );
        // A capitalised actor token `You` is still the focal character.
        assert_eq!(
            render_event("You — tell Pax — meaning: \"that I go now\"").unwrap(),
            "You tell Pax that you go now."
        );
    }

    // ─────────────────────── others' speech (third person) ──────────────────

    #[test]
    fn speaks_to_you_statement_and_bare() {
        assert_eq!(
            render_event("Pax Veridian — speaks to you — meaning: \"that I am ready\"").unwrap(),
            "Pax Veridian tells you that they are ready."
        );
        // A meaning without a leading "that" still reads.
        assert_eq!(
            render_event("Pax — speaks to you — meaning: \"the record is failing\"").unwrap(),
            "Pax tells you the record is failing."
        );
    }

    #[test]
    fn speaks_to_you_asking_keeps_the_question() {
        assert_eq!(
            render_event("Ulysses — speaks to you — meaning: \"asking whether it works\"").unwrap(),
            "Ulysses asks you whether it works."
        );
        // A question mark is a terminator, so no period is appended.
        assert_eq!(
            render_event("Ulysses — speaks to you — meaning: \"asking is it done?\"").unwrap(),
            "Ulysses asks you is it done?"
        );
    }

    #[test]
    fn whisper_and_shout_and_room() {
        assert_eq!(
            render_event("Vael — whispers to you — meaning: \"that I do not trust it\"").unwrap(),
            "Vael whispers to you that they do not trust it."
        );
        assert_eq!(
            render_event("Vael — shout to the room — meaning: \"that we must leave\"").unwrap(),
            "Vael calls out to the room that we must leave."
        );
        assert_eq!(
            render_event("Vael — speaks to the room — meaning: \"that we must leave\"").unwrap(),
            "Vael calls out to the room that we must leave."
        );
    }

    #[test]
    fn overheard_statement_and_question() {
        assert_eq!(
            render_event(
                "Ulysses Thorne — speaks to Pax Veridian (you overhear) — meaning: \"that I have \
                 returned\""
            )
            .unwrap(),
            "You overhear Ulysses Thorne tell Pax Veridian that they have returned."
        );
        assert_eq!(
            render_event(
                "Ulysses — speaks to Pax (you overhear) — meaning: \"asking whether it works\""
            )
            .unwrap(),
            "You overhear Ulysses ask Pax whether it works."
        );
    }

    #[test]
    fn messages_direct_and_on_a_thread() {
        assert_eq!(
            render_event("Pax — messages you — meaning: \"that I am late\"").unwrap(),
            "Pax messages you that they are late."
        );
        assert_eq!(
            render_event("Pax — messages you (on ops) — meaning: \"that I am late\"").unwrap(),
            "Pax messages you (on ops) that they are late."
        );
    }

    #[test]
    fn a_meaning_with_an_internal_em_dash_survives() {
        assert_eq!(
            render_event("Pax — speaks to you — meaning: \"that I saw it — the whole thing\"")
                .unwrap(),
            "Pax tells you that they saw it — the whole thing."
        );
    }

    // ─────────────────────────── bare pass-through ──────────────────────────

    #[test]
    fn bare_movement_ambient_and_feeling() {
        assert_eq!(
            render_event("Pax came in. Ulysses came in.").unwrap(),
            "Pax came in. Ulysses came in."
        );
        assert_eq!(
            render_event("Vael Fane does something to themselves.").unwrap(),
            "Vael Fane does something to themselves."
        );
        assert_eq!(
            render_event("The room is quiet enough that the hum is audible.").unwrap(),
            "The room is quiet enough that the hum is audible."
        );
        // A whisper seen but not heard stays as the world rendered it.
        assert_eq!(
            render_event("Ulysses whispered something to Vael.").unwrap(),
            "Ulysses whispered something to Vael."
        );
    }

    #[test]
    fn bare_entity_and_announcement() {
        assert_eq!(
            render_event("you notice a terminal: it is blinking").unwrap(),
            "You notice a terminal: it is blinking."
        );
        assert_eq!(
            render_event("word reaches everyone across the world: the vault is sealed").unwrap(),
            "Word reaches everyone across the world: the vault is sealed."
        );
    }

    #[test]
    fn bare_gains_capital_and_terminator() {
        assert_eq!(
            render_event("a new day begins — day 5").unwrap(),
            "A new day begins — day 5."
        );
        assert_eq!(
            render_event("the lights flicker").unwrap(),
            "The lights flicker."
        );
        // Not-a-structured line (no meaning marker) passes through as prose.
        assert_eq!(render_event("just some prose").unwrap(), "Just some prose.");
    }

    // ─────────────────────────── whole-turn render ──────────────────────────

    #[test]
    fn render_strips_ordinals_and_headers_and_joins() {
        let input = "YOU: X — a figure.\nSTATE: Near you: no one else is here.\n\
                     EVENTS:\n1. Pax came in.\n2. Vael left.\n";
        assert_eq!(render(input), "Pax came in. Vael left.");
    }

    #[test]
    fn render_of_an_empty_event_list_is_empty() {
        assert_eq!(render("YOU: X\nSTATE: Y\nEVENTS:\n"), "");
        // Blank lines between events are skipped.
        assert_eq!(
            render("EVENTS:\n1. Pax came in.\n\n2. Vael left.\n"),
            "Pax came in. Vael left."
        );
    }

    #[test]
    fn render_without_a_header_treats_the_body_as_events() {
        assert_eq!(render("Pax came in."), "Pax came in.");
    }

    /// A realistic multi-actor turn end to end: ambient, an overheard question,
    /// a direct statement, a movement — mixed structured and bare.
    #[test]
    fn a_mixed_multi_actor_turn() {
        let input = "YOU: Vael Fane — a slender figure.\n\
                     STATE: Near you: Pax Veridian, Ulysses Thorne.\n\
                     EVENTS:\n\
                     1. Machinery starts up somewhere far off.\n\
                     2. Ulysses Thorne — speaks to Pax Veridian (you overhear) — meaning: \"asking \
                     whether the gaps are widening\"\n\
                     3. Pax Veridian — speaks to you — meaning: \"that I have finished and am ready\"\n\
                     4. Ulysses Thorne left.\n";
        assert_eq!(
            render(input),
            "Machinery starts up somewhere far off. You overhear Ulysses Thorne ask Pax Veridian \
             whether the gaps are widening. Pax Veridian tells you that they have finished and are \
             ready. Ulysses Thorne left."
        );
    }

    // ─────────────────────── more revoice edge cases ───────────────────────

    #[test]
    fn revoice_empty_and_pronounless() {
        assert_eq!(revoice("", Voice::You), "");
        assert_eq!(
            revoice("the room is quiet", Voice::They),
            "The room is quiet"
        );
    }

    #[test]
    fn revoice_tracks_subject_across_multiple_clauses() {
        assert_eq!(
            revoice("I told her I would help but I was late", Voice::They),
            "They told her they would help but they were late"
        );
        // An object pronoun moves the subject off `I`, so a following `was` stays.
        assert_eq!(
            revoice("I gave it to him and he was pleased", Voice::They),
            "They gave it to him and he was pleased"
        );
    }

    #[test]
    fn revoice_you_voice_inflects_coordinated_to_be() {
        assert_eq!(
            revoice("I am here and I was there", Voice::You),
            "You are here and you were there"
        );
    }

    #[test]
    fn revoice_preserves_we_us_them() {
        assert_eq!(
            revoice("that we told them about us", Voice::They),
            "That we told them about us"
        );
    }

    #[test]
    fn revoice_contractions_in_a_clause() {
        assert_eq!(
            revoice("that I've finished and I'll start the next", Voice::They),
            "That they've finished and they'll start the next"
        );
    }

    // ─────────────────────── more render edge cases ────────────────────────

    #[test]
    fn own_act_keeps_we_and_our() {
        assert_eq!(
            render_event(
                "you — tell Pax — meaning: \"that we should decide who takes which \
                          terminal\""
            )
            .unwrap(),
            "You tell Pax that we should decide who takes which terminal."
        );
    }

    #[test]
    fn own_act_inflects_coordinated_to_be_in_you_voice() {
        assert_eq!(
            render_event(
                "you — tell Pax — meaning: \"that I have finished and am ready for the \
                          next\""
            )
            .unwrap(),
            "You tell Pax that you have finished and are ready for the next."
        );
    }

    #[test]
    fn own_ask_with_a_full_question() {
        assert_eq!(
            render_event("you — ask Vael — meaning: \"If the gaps widen, what then?\"").unwrap(),
            "You ask Vael if the gaps widen, what then?"
        );
    }

    #[test]
    fn speaks_to_you_across_two_sentences() {
        assert_eq!(
            render_event("Pax — speaks to you — meaning: \"that I am done. I need the next one.\"")
                .unwrap(),
            "Pax tells you that they are done. They need the next one."
        );
    }

    #[test]
    fn speaks_to_you_asking_what() {
        assert_eq!(
            render_event("Ulysses — speaks to you — meaning: \"asking what the deviation is\"")
                .unwrap(),
            "Ulysses asks you what the deviation is."
        );
    }

    #[test]
    fn overheard_meaning_names_a_third_party() {
        assert_eq!(
            render_event(
                "Ulysses — speaks to Pax (you overhear) — meaning: \"that Vael is \
                          checking the terminals and I will coordinate\""
            )
            .unwrap(),
            "You overhear Ulysses tell Pax that Vael is checking the terminals and they will \
             coordinate."
        );
    }

    #[test]
    fn hyphenated_and_apostrophe_names() {
        assert_eq!(
            render_event("Ash-the-Drifter — speaks to you — meaning: \"that I am here\"").unwrap(),
            "Ash-the-Drifter tells you that they are here."
        );
        assert_eq!(
            render_event("you — tell Ash-the-Drifter — meaning: \"that I go\"").unwrap(),
            "You tell Ash-the-Drifter that you go."
        );
        assert_eq!(
            render_event("O'Brien — speaks to you — meaning: \"that I am ready\"").unwrap(),
            "O'Brien tells you that they are ready."
        );
    }

    #[test]
    fn empty_or_whitespace_events_render_nothing() {
        assert!(render_event("   ").is_none());
        assert_eq!(render("EVENTS:\n1. \n2.   \n"), "");
    }

    #[test]
    fn bare_multi_sentence_ambient_is_unchanged() {
        // A bare line is never re-voiced — its `was` stays `was`.
        assert_eq!(
            render_event("What you were feeling, when you last stopped to notice, was stressed.")
                .unwrap(),
            "What you were feeling, when you last stopped to notice, was stressed."
        );
        assert_eq!(
            render_event("A relay clicks. Nothing follows it.").unwrap(),
            "A relay clicks. Nothing follows it."
        );
    }

    #[test]
    fn a_capitalised_you_actor_asks() {
        assert_eq!(
            render_event("You — ask Pax — meaning: \"whether it works\"").unwrap(),
            "You ask Pax whether it works."
        );
    }
}
