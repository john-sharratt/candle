//! Probe generation: the prompt that asks for candidates, and the parser that
//! reads them back.
//!
//! # Why generation gets its own conversation
//!
//! The obvious place to ask for questions is the folder's own conversation — it
//! already holds the listing, the anchor excerpt and the summary. Two things
//! rule that out:
//!
//! * The generator's answer is a **list of questions**, and it would seal into
//!   the folder's timeline as retrievable content. A scan would then match the
//!   list rather than any single probe, and the folder's strongest signal would
//!   be a numbered block that answers nothing.
//! * [`Sequence::fork`] does not inherit the parent's turns — it projects the
//!   system prompt and nothing else — so there is no way to borrow that context
//!   for a throwaway branch and discard it afterwards.
//!
//! So generation runs in a fresh conversation carrying a **compact evidence
//! block** ([`evidence`]): the folder's summary, the head of its listing, and
//! the terms the directory-frequency index found distinctive. A few hundred
//! tokens against the full chain's ~1,500, which matters at 415 directories, and
//! the conversation is tombstoned once its candidates are parsed.
//!
//! # Why the distinctive terms are an input
//!
//! Asked simply for good questions, a model writes "What does this module do?"
//! sixteen times — questions that match every folder equally and retrieve
//! nothing. Specificity has to be *supplied*, not hoped for: [`super::idf`]
//! computes which terms characterise this folder against the whole workspace,
//! and the prompt hands them over with an instruction to use them. The filters
//! in [`super::filters`] then verify the instruction was followed, because a
//! prompt is a request and a gate is a guarantee.

use super::{Register, CANDIDATES_PER_REGISTER, PROBES_PER_REGISTER};
use crate::repo_scan::dir_unit::DirUnit;

/// Listing entries shown in the compact evidence block.
///
/// Enough to convey what kind of folder this is; not so many that the block
/// becomes the folder's whole file list, which would tempt the generator into
/// writing questions that name files — the one thing every register forbids.
const EVIDENCE_FILES: usize = 12;

/// Distinctive terms offered as seeds.
///
/// The three specific registers each need a term per question, so
/// `3 * PROBES_PER_REGISTER` is the floor for non-repeating coverage; a further
/// register's worth of slack lets the generator pass over terms it cannot build
/// a natural question around instead of forcing the awkward ones.
///
/// Derived rather than written down: the literal 16 sat under its own stated
/// floor of eighteen, so the three registers could not be covered without
/// repeating a term.
const SEED_TERMS: usize = 4 * PROBES_PER_REGISTER;

/// The floor the constant's reasoning states, enforced where it cannot drift:
/// a term per question for each of the three specific registers, no repeats.
const _: () = assert!(SEED_TERMS >= 3 * PROBES_PER_REGISTER);

/// Characters of summary text carried into the evidence block.
const SUMMARY_CHARS: usize = 600;

/// The compact evidence block a generation conversation is prefilled with.
///
/// Deliberately NOT the folder's full chain: the generator needs to know what
/// the folder is and what it calls things, and nothing more.
pub fn evidence(unit: &DirUnit, summary: &str, seeds: &[&str]) -> String {
    let mut out = String::new();

    out.push_str("Here is what is known about one folder in a software project.\n\n");

    // The folder is named by its LAST component, not its path. Naming it in full
    // puts the one string every register forbids at the top of the prompt, and a
    // generator that has just read `candle-nn/src/kv_cache/chunked/` reaches for
    // it. The leaf alone conveys what the folder is called without supplying an
    // address to copy.
    if unit.list_prefix().is_empty() {
        out.push_str("FOLDER: the root folder of the project\n");
    } else {
        let leaf = unit
            .dir
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(&unit.dir);
        out.push_str(&format!("FOLDER: {leaf}\n"));
    }
    if let Some(hint) = unit.module_hint() {
        out.push_str(&format!("PACKAGE: {}\n", hint.render()));
    }

    out.push_str("\nWHAT IT IS FOR:\n");
    let summary = summary.trim();
    if summary.is_empty() {
        out.push_str("(no summary available)\n");
    } else {
        let clipped: String = summary.chars().take(SUMMARY_CHARS).collect();
        out.push_str(&clipped);
        out.push('\n');
    }

    // BASENAMES, never full paths. The evidence block is the one place the
    // generator could read this folder's path, and a model shown
    // `candle-nn/src/kv_cache/chunked/alloc.rs` writes questions containing
    // `kv_cache/chunked` — which [`super::filters`] then discards as
    // self-address. The filter catches it either way; showing basenames means
    // the candidate slot is not wasted in the first place.
    out.push_str("\nFILES IN IT:\n");
    for file in unit.files.iter().take(EVIDENCE_FILES) {
        let base = file.path.rsplit('/').next().unwrap_or(&file.path);
        out.push_str(&format!("  {base}\n"));
    }
    if unit.files.len() > EVIDENCE_FILES {
        out.push_str(&format!(
            "  … and {} more\n",
            unit.files.len() - EVIDENCE_FILES
        ));
    }

    if !seeds.is_empty() {
        out.push_str(
            "\nDISTINCTIVE TERMS (these names appear here and almost nowhere else in the \
             project — they are what makes a question about this folder findable):\n",
        );
        for chunk in seeds.chunks(6) {
            out.push_str("  ");
            out.push_str(&chunk.join(", "));
            out.push('\n');
        }
    }

    out
}

/// The instruction turn: what to write, in what shape, and what disqualifies a
/// question.
///
/// The prohibitions are stated to the generator even though [`super::filters`]
/// enforces them independently. Both are needed and they do different jobs: the
/// prompt raises the share of usable candidates, and the filter guarantees no
/// unusable one survives. Relying on the prompt alone was the original design of
/// this layer's summaries, and it is why they never retrieved.
///
/// All four registers in ONE instruction.
///
/// Generation was briefly split into a turn per register, to stop one bad decode
/// costing a folder every probe it has. That is the right failure isolation and
/// the wrong economics, because **the reasoning block's size barely depends on
/// what is asked**: a twelve-question ask drew 12,752 characters where a
/// forty-eight-question ask drew 15,140. Four turns therefore paid four
/// reasoning blocks — ~15,200 tokens per directory — to produce the same
/// questions one turn produces after ~3,800.
///
/// So the registers are asked for together, and failure isolation comes from
/// [`parse`] instead: it reads each register's block independently, so a section
/// the model mangles costs that register and no other.
pub fn instruction(unit: &DirUnit, seeds: &[&str]) -> String {
    let mut out = String::new();
    let n = CANDIDATES_PER_REGISTER;

    out.push_str(
        "Write the questions that a developer would ask, whose ANSWER is this folder.\n\n\
         Someone asking has NOT seen the folder and does not know where it lives — they only \
         know what they are trying to do. Write what they would type.\n\n",
    );

    out.push_str("RULES — a question breaking any of these is discarded:\n");
    out.push_str("  1. Never write the folder's path, and never write a file name.\n");
    out.push_str(
        "  2. Never write \"this folder\", \"this module\", \"this directory\" or \
         \"this file\" — the asker cannot see it.\n",
    );
    out.push_str("  3. Every question ends with a question mark.\n");
    out.push_str("  4. One question per line. No commentary, no explanation, no answers.\n\n");

    out.push_str(&format!(
        "Write exactly {n} questions under EACH of the four headings below. Reproduce each \
         heading exactly as written, in square brackets, on its own line.\n\n",
    ));

    for (i, register) in Register::ALL.into_iter().enumerate() {
        out.push_str(&format!("[{}]\n", register.id()));
        out.push_str(register.brief());
        out.push('\n');
        if let Some(example) = worked_example(register, seeds, i) {
            out.push_str(&format!("Example: {example}\n"));
        }
        out.push('\n');
    }

    if !seeds.is_empty() {
        // Every seed the evidence block showed, not a shorter prefix of them:
        // two lists of different lengths in one prompt read as a disagreement
        // about which names are usable, and the shorter one is the instruction
        // — the half the generator is actually told to obey.
        out.push_str(&format!(
            "Use these names in the first three sections, and NONE of them in the fourth: {}\n\n",
            seeds.join(", "),
        ));
    }

    let _ = unit;
    out.push_str("Begin with the [locational] heading.\n");
    out
}

/// Questions per register, read from a block carrying all four headings.
///
/// The reasoning block is stripped first, so nothing the model deliberated can
/// be mistaken for a question — and an unterminated block (the cap was hit)
/// yields nothing, which is the honest answer.
pub fn parse(body: &str) -> Vec<(Register, Vec<String>)> {
    let answer = match body.rfind("</think>") {
        Some(end) => &body[end + "</think>".len()..],
        None if body.contains("<think>") => "",
        None => body,
    };
    let mut out: Vec<(Register, Vec<String>)> = Vec::new();
    let mut current: Option<Register> = None;
    for line in answer.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(register) = heading(trimmed) {
            if !out.iter().any(|(r, _)| *r == register) {
                out.push((register, Vec::new()));
            }
            current = Some(register);
            continue;
        }
        let Some(register) = current else { continue };
        let Some(question) = list_item(trimmed) else {
            continue;
        };
        if let Some((_, items)) = out.iter_mut().find(|(r, _)| *r == register) {
            items.push(question);
        }
    }
    out
}

/// A register heading, in any of the shapes a model actually emits.
fn heading(line: &str) -> Option<Register> {
    let stripped = line
        .trim_start_matches(['#', '*', '-', ' '])
        .trim_end_matches([':', '*', ' ']);
    let inner = stripped
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(stripped);
    // A heading is the bare register word. Without this guard a question that
    // merely opens with the word is read as a section break, silently splitting
    // the block and dropping everything after it.
    if inner.split_whitespace().count() != 1 {
        return None;
    }
    Register::from_id(inner)
}

/// One worked question per register, built from this folder's own vocabulary.
///
/// Each specific register gets a *different* seed (`nth`), so the three examples
/// do not all point at the same term and invite twelve questions about it.
/// The systemic example carries no identifier at all — it is the shape that
/// register exists for, and showing it beside three identifier-bearing examples
/// is what stops the model writing a fourth specific block.
fn worked_example(register: Register, seeds: &[&str], nth: usize) -> Option<String> {
    if register == Register::Systemic {
        return Some(
            "How does the project keep track of what it has already worked out?".to_string(),
        );
    }
    let term = seeds.get(nth).or_else(|| seeds.first())?;
    Some(match register {
        Register::Locational => format!("Where is {term} set up when the process starts?"),
        Register::Mechanistic => format!("What happens to {term} when the cache runs out of room?"),
        Register::Conceptual => format!("What is {term} for?"),
        Register::Systemic => unreachable!("handled above"),
    })
}

/// The question text of a line, with any bullet or number removed.
///
/// **The question mark is what identifies a question — not a list marker.** The
/// instruction asks for a numbered list, and the model frequently ignores that
/// and writes bare lines:
///
/// ```text
/// [locational]
/// Where can I find the unit tests for the allocation strategies?
/// In which source files are the selection table algorithms tested?
/// ```
///
/// Requiring `1.` or `-` discarded every one of those as prose. Measured on the
/// first live run: seven of nine directories lost their **entire** probe set to
/// this, and it was invisible — a parse failure and a generation failure both
/// show up as `candidates=0`, which is why the raw-text warning in
/// [`super::super::probe_pass::generate`] exists.
///
/// Keying on the trailing `?` instead is both more permissive and stricter in
/// the right place: every register already requires it ([`super::filters`]), so
/// nothing admissible is lost, while the model's commentary — "Here are the
/// questions:", "That covers the locational ones." — has no question mark and is
/// still correctly ignored.
fn list_item(line: &str) -> Option<String> {
    let rest = if let Some(stripped) = line.strip_prefix(['-', '*', '•']) {
        stripped
    } else {
        // An optional `12.` / `12)` prefix; a bare line keeps all of itself.
        let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            line
        } else {
            let after = &line[digits.len()..];
            after
                .strip_prefix('.')
                .or_else(|| after.strip_prefix(')'))
                .unwrap_or(line)
        }
    };
    let text = rest.trim();
    if text.is_empty() || !text.ends_with('?') {
        return None;
    }
    Some(text.to_string())
}

/// Terms a folder's seeds are drawn from, capped for the prompt.
pub fn seed_count() -> usize {
    SEED_TERMS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repo_scan::dir_unit::build_units;
    use crate::repo_scan::types::{FileEntry, Language, RepoMap};

    fn unit(paths: &[&str]) -> DirUnit {
        let map = RepoMap {
            files: paths
                .iter()
                .map(|p| FileEntry {
                    path: p.to_string(),
                    line_count: 1,
                    language: Language::Rust,
                    size_bytes: 1,
                    module_hint: None,
                })
                .collect(),
            ..Default::default()
        };
        let d = tempfile::tempdir().unwrap();
        build_units(&map, d.path()).into_iter().next().unwrap()
    }

    #[test]
    fn evidence_carries_folder_summary_files_and_seeds() {
        let u = unit(&["kv/backing.rs", "kv/arena.rs"]);
        let body = evidence(
            &u,
            "Holds the chunked KV cache state.",
            &["ChunkGid", "GidPool"],
        );
        assert!(body.contains("FOLDER: kv"), "{body}");
        assert!(body.contains("Holds the chunked KV cache state."), "{body}");
        assert!(body.contains("backing.rs"), "{body}");
        assert!(body.contains("ChunkGid"), "{body}");
    }

    /// The evidence block is the only place a generator could read this folder's
    /// address, and a model shown one writes it back. Neither the folder path nor
    /// any file path may appear — only leaf names.
    #[test]
    fn evidence_never_shows_a_path() {
        let u = unit(&["candle-nn/src/kv_cache/chunked/backing.rs"]);
        let body = evidence(&u, "The chunked KV backing store.", &["ChunkGid"]);
        assert!(
            !body.contains("candle-nn/src/kv_cache"),
            "the folder path leaked into the prompt:\n{body}",
        );
        assert!(body.contains("FOLDER: chunked"), "{body}");
        assert!(body.contains("  backing.rs"), "{body}");
    }

    /// The root folder is named in words. Asked about "the `.` folder" the model
    /// writes about a directory literally called `.` — the same trap the
    /// summarise request already documents.
    #[test]
    fn the_root_folder_is_named_in_words() {
        let u = unit(&["top.rs"]);
        let body = evidence(&u, "The workspace root.", &[]);
        assert!(body.contains("the root folder of the project"), "{body}");
        assert!(!body.contains("FOLDER: ."), "{body}");
    }

    /// A folder with hundreds of files must not turn the evidence block into its
    /// file list — that is what tempts the generator into naming files.
    #[test]
    fn a_large_listing_is_clipped_with_a_count() {
        let paths: Vec<String> = (0..40).map(|i| format!("a/f{i:02}.rs")).collect();
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let u = unit(&refs);
        let body = evidence(&u, "Many files.", &[]);
        assert!(
            body.contains(&format!("and {} more", 40 - EVIDENCE_FILES)),
            "{body}",
        );
    }

    /// **The two lists in one prompt must agree.** The evidence block shows the
    /// seeds and the instruction says to use them; when the instruction named a
    /// shorter prefix, the generator was told to use twelve of the sixteen it
    /// could see, and the four it was shown but not told about read as usable.
    #[test]
    fn the_instruction_names_every_seed_the_evidence_shows() {
        let u = unit(&["a/x.rs"]);
        let seeds: Vec<String> = (0..SEED_TERMS)
            .map(|i| format!("widget_{i:02}_thing"))
            .collect();
        let refs: Vec<&str> = seeds.iter().map(String::as_str).collect();
        let shown = evidence(&u, "A folder.", &refs);
        let told = instruction(&u, &refs);
        for seed in &seeds {
            assert!(shown.contains(seed.as_str()), "{seed} not shown:\n{shown}");
            assert!(told.contains(seed.as_str()), "{seed} not named:\n{told}");
        }
    }

    #[test]
    fn the_instruction_names_every_register_and_its_count() {
        let u = unit(&["a/x.rs"]);
        let text = instruction(&u, &["Widget"]);
        for register in Register::ALL {
            assert!(
                text.contains(&format!("[{}]", register.id())),
                "{}",
                register.id()
            );
            assert!(text.contains(register.brief()), "{}", register.id());
        }
        assert!(
            text.contains(&format!("exactly {CANDIDATES_PER_REGISTER} questions")),
            "{text}",
        );
    }

    /// The briefs state the rule; the examples show the shape, and the shape is
    /// what gets copied. Each specific register must get a DIFFERENT seed, or
    /// all three examples point at one term and the folder gets thirty-six
    /// questions about it.
    #[test]
    fn each_specific_register_gets_a_worked_example_from_a_different_seed() {
        let u = unit(&["a/x.rs"]);
        let text = instruction(&u, &["GidPool", "ChunkGid", "ArenaKey"]);
        assert!(text.contains("Where is GidPool set up"), "{text}");
        assert!(text.contains("What happens to ChunkGid when"), "{text}");
        assert!(text.contains("What is ArenaKey for?"), "{text}");
    }

    /// The systemic example must carry no identifier — it is the one register
    /// whose shape is "a newcomer who cannot name anything", and showing it an
    /// identifier-bearing example produces a fourth specific block.
    #[test]
    fn the_systemic_example_names_no_identifier() {
        let u = unit(&["a/x.rs"]);
        let text = instruction(&u, &["GidPool", "ChunkGid", "ArenaKey"]);
        let systemic = text.split("[systemic]").nth(1).expect("systemic section");
        let example = systemic.split("Use these names").next().unwrap_or(systemic);
        assert!(example.contains("Example:"), "{example}");
        for term in ["GidPool", "ChunkGid", "ArenaKey"] {
            assert!(!example.contains(term), "{term} leaked:\n{example}");
        }
    }

    /// A folder with nothing distinctive still gets its systemic example; the
    /// specific registers simply have nothing to demonstrate with.
    #[test]
    fn a_folder_without_seeds_still_gets_the_systemic_example() {
        let u = unit(&["a/x.rs"]);
        let text = instruction(&u, &[]);
        assert!(text.contains("How does the project keep track"), "{text}");
        assert!(!text.contains("Where is  set up"), "no empty-term example");
        assert!(!text.contains("Use these names:"), "no empty seed list");
    }

    /// The prohibitions must reach the generator, not only the filter — the
    /// filter caps the damage, the prompt is what raises the yield.
    #[test]
    fn the_instruction_states_the_disqualifying_rules() {
        let u = unit(&["a/x.rs"]);
        let text = instruction(&u, &[]);
        assert!(text.contains("never write a file name"), "{text}");
        assert!(text.contains("this folder"), "{text}");
        assert!(text.contains("question mark"), "{text}");
    }

    /// The shape the model ACTUALLY emits: bare question lines, no numbering.
    /// Requiring a list marker discarded seven of nine directories' entire probe
    /// sets on the first live run, and reported it indistinguishably from the
    /// generator having failed.
    #[test]
    fn parses_bare_unnumbered_question_lines() {
        let body = "\
[locational]
Where can I find the unit tests for the allocation strategies?
In which source files are the selection table algorithms tested?

[systemic]
How does the project decide what to keep in memory?
";
        let parsed = parse(body);
        assert_eq!(parsed.len(), 2, "{parsed:?}");
        assert_eq!(parsed[0].1.len(), 2, "{parsed:?}");
        assert_eq!(parsed[1].1.len(), 1, "{parsed:?}");
    }

    /// Numbering and bullets still work — the model uses them about half the
    /// time, and which half is not predictable.
    #[test]
    fn parses_numbered_and_bulleted_lines_too() {
        let body = "[locational]\n1. Where is the arena allocated?\n- Which file handles sealing?\n* And this one?\n2) And this?\n";
        assert_eq!(parse(body)[0].1.len(), 4);
    }

    /// Commentary is not a question. Without the question-mark rule a model's
    /// "Here are the questions:" line becomes probe number one.
    #[test]
    fn prose_around_the_questions_is_ignored() {
        let body = "\
[locational]
Here are some good questions:
Where is the arena allocated?
That covers the locational ones.
";
        assert_eq!(
            parse(body)[0].1,
            vec!["Where is the arena allocated?".to_string()],
        );
    }

    /// A question opening with a register word must not be read as a heading —
    /// that would split the block and silently drop everything after it.
    #[test]
    fn a_question_starting_with_a_register_word_is_not_a_heading() {
        let body = "[conceptual]\nConceptual overview of what a ChunkGid represents?\nWhat does GidPool allocate?\n";
        let parsed = parse(body);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].1.len(), 2, "{parsed:?}");
    }

    /// The reasoning block is deliberation, not output. Everything before
    /// `</think>` is discarded — and the model opens nearly every generation
    /// with one, running to 15,140 characters.
    #[test]
    fn a_closed_think_block_contributes_nothing() {
        let body = "<think>\nShould I ask about arenas? Maybe about sealing?\n[locational] is first.\n</think>\n\n[locational]\nWhere is it kept?\n";
        let parsed = parse(body);
        assert_eq!(parsed.len(), 1, "{parsed:?}");
        assert_eq!(parsed[0].1, vec!["Where is it kept?".to_string()]);
    }

    /// An UNTERMINATED think block yields nothing, which is the honest answer:
    /// the cap was hit and the model never reached the questions. Taking its
    /// interrogative lines instead would seed the corpus with the generator's
    /// own deliberation.
    #[test]
    fn an_unterminated_think_block_yields_nothing() {
        let body = "<think>\n[locational]\nShould I ask about arenas?\nWhat about sealing?\n";
        assert!(
            parse(body).iter().all(|(_, q)| q.is_empty()),
            "{:?}",
            parse(body)
        );
    }

    #[test]
    fn an_empty_or_shapeless_body_parses_to_nothing() {
        assert!(parse("").is_empty());
        assert!(parse("I cannot help with that.").is_empty());
        // Questions before any heading have no register to belong to.
        assert!(parse("Stray question?\n").is_empty());
    }
}
