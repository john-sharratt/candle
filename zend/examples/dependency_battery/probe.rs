//! The probe arms' context (`arms::PROBES`): where the files provenance
//! selected are placed. Each is asked on the daemon's passthrough route, which
//! runs the request's own system prompt and messages with none of the
//! daemon's projection; every system message joins the one system prompt. So
//! the four differ in nothing but where the files sit and how many fit:
//!
//! - S1 — inside the system prompt, after the persona, under P1's budget;
//! - M1 — in a system block of their own after the persona's, closed and
//!   reopened with the chat template's own markers, under the same budget;
//! - U1 — in the user message ahead of the question, as P1 carries them, the
//!   system prompt the persona alone;
//! - S2 — inside the system prompt as S1, the repository's seed files
//!   ([`SEED_FILES`]) first and the selection after them, under
//!   [`WIDE_BUDGET_TOKENS`]: the pinned prompt a conversation would open on;
//! - Q1 — as S1, the same files rendered in Qwen-Agent's knowledge format
//!   ([`qwen_knowledge`]): the system prompt's `# Knowledge Base`, each file
//!   under `## The content from <path>:` in a fence.
//!
//! The user message's instruction is P1's in all five
//! (`prompt::RETRIEVED_PREFIX`).

use crate::arms::Arm;
use crate::prompt::RETRIEVED_PREFIX;
use crate::rag::context::{first_chars, header, Packed, BUDGET_CHARS, CHARS_PER_TOKEN};

/// The persona every probe's system prompt opens with: the daemon's own first
/// sentence.
pub const PERSONA: &str = "You are Zen, an AI coding assistant running on Qwen3.8-Flash-Next, \
     working in the prog workspace, which holds one or more repositories.";

/// The heading the files sit under in the system prompt.
pub const FILES_HEADING: &str = "# Files you have read\n\n";

/// Where M1 closes the persona's system block and opens the files' own.
pub const NEXT_SYSTEM_BLOCK: &str = "<|im_end|>\n<|im_start|>system\n";

/// The repository's files the seed chain reads, pinned ahead of S2's
/// selection.
pub const SEED_FILES: [&str; 3] = ["README.md", "ARCHITECTURE.md", "AGENTS.md"];

/// S2's budget: four times P1's.
pub const WIDE_BUDGET_TOKENS: usize = 96_000;

/// The characters of files `arm` carries.
pub fn budget_chars(arm: Arm) -> usize {
    match arm {
        Arm::S2 => WIDE_BUDGET_TOKENS * CHARS_PER_TOKEN,
        _ => BUDGET_CHARS,
    }
}

/// The files `arm` packs, best first: `selected` itself, or for S2 the seed
/// files ahead of it, each once.
pub fn paths(arm: Arm, selected: &[String]) -> Vec<String> {
    let seed = match arm {
        Arm::S2 => SEED_FILES.as_slice(),
        _ => &[],
    };
    let mut out: Vec<String> = Vec::new();
    for p in seed
        .iter()
        .map(|s| s.to_string())
        .chain(selected.iter().cloned())
    {
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}

/// `packed`'s files in Qwen-Agent's knowledge format (`qwen_agent/agents/
/// assistant.py`, `KNOWLEDGE_TEMPLATE_EN` and `KNOWLEDGE_SNIPPET_EN`), each
/// file's source its repository-relative path.
pub fn qwen_knowledge(packed: &Packed) -> String {
    let mut snippets: Vec<String> = Vec::new();
    let mut at = 0;
    for f in &packed.files {
        let head = header(&f.path);
        let Some(found) = packed.text[at..].find(&head) else {
            continue;
        };
        let start = at + found + head.len();
        let body = first_chars(&packed.text[start..], f.chars);
        snippets.push(format!(
            "## The content from {}:\n\n```\n{body}\n```",
            f.path
        ));
        at = start + body.len();
    }
    format!("# Knowledge Base\n\n{}", snippets.join("\n\n"))
}

/// `(system, user)` for `question` under `arm`, `packed` its files.
pub fn messages(arm: Arm, question: &str, packed: &Packed) -> (String, String) {
    let user = format!("{RETRIEVED_PREFIX}\n\n{question}");
    let files = packed.text.as_str();
    match arm {
        Arm::Q1 => (format!("{PERSONA}\n\n{}", qwen_knowledge(packed)), user),
        Arm::U1 => (PERSONA.to_string(), format!("{files}{user}")),
        Arm::M1 => (
            format!("{PERSONA}{NEXT_SYSTEM_BLOCK}{FILES_HEADING}{files}"),
            user,
        ),
        _ => (format!("{PERSONA}\n\n{FILES_HEADING}{files}"), user),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::context::pack;

    const FILES: &str = "=== a.rs ===\nfn x() {}\n\n";

    fn packed() -> Packed {
        pack(
            [(1, "a.rs".to_string(), 0.0, "fn x() {}".to_string())],
            1_000,
        )
    }
    const PERSONA_TEXT: &str = "You are Zen, an AI coding assistant running on \
         Qwen3.8-Flash-Next, working in the prog workspace, which holds one or more repositories.";
    const USER: &str = "Answer from the files above and what you know of the repository — call \
         no tool. If you do not know, say so.\n\nWhere?";

    /// **S1 and S2 carry the files inside the system prompt, M1 in a system
    /// block of their own, U1 in the user message**; the instruction and the
    /// question are the same in all.
    #[test]
    fn each_probe_places_the_files_where_it_says() {
        let inside =
            format!("{PERSONA_TEXT}\n\n# Files you have read\n\n=== a.rs ===\nfn x() {{}}\n\n");
        assert_eq!(packed().text, FILES);
        assert_eq!(
            messages(Arm::S1, "Where?", &packed()),
            (inside.clone(), USER.to_string())
        );
        assert_eq!(
            messages(Arm::S2, "Where?", &packed()),
            (inside, USER.to_string())
        );
        assert_eq!(
            messages(Arm::M1, "Where?", &packed()),
            (
                format!(
                    "{PERSONA_TEXT}<|im_end|>\n<|im_start|>system\n# Files you have read\n\n\
                     === a.rs ===\nfn x() {{}}\n\n"
                ),
                USER.to_string()
            )
        );
        assert_eq!(
            messages(Arm::U1, "Where?", &packed()),
            (PERSONA_TEXT.to_string(), format!("{FILES}{USER}"))
        );
        assert_eq!(
            messages(Arm::Q1, "Where?", &packed()),
            (
                format!(
                    "{PERSONA_TEXT}\n\n# Knowledge Base\n\n## The content from a.rs:\n\n\
                     ```\nfn x() {{}}\n```"
                ),
                USER.to_string()
            )
        );
    }

    /// **Each file is one snippet, only what the budget let in**, joined by a
    /// blank line as Qwen-Agent joins them.
    #[test]
    fn the_knowledge_holds_each_file_as_packed() {
        let p = pack(
            [
                (1, "a.rs".to_string(), 0.0, "one".to_string()),
                (2, "b/c.rs".to_string(), 0.0, "two three".to_string()),
            ],
            37,
        );
        assert_eq!(p.text, "=== a.rs ===\none\n\n=== b/c.rs ===\ntw\n\n");
        assert_eq!(
            qwen_knowledge(&p),
            "# Knowledge Base\n\n## The content from a.rs:\n\n```\none\n```\n\n\
             ## The content from b/c.rs:\n\n```\ntw\n```"
        );
    }

    /// **Only S2 pins the seed files, ahead of the selection**, and a seed
    /// file the selection also names is carried once, where the seed puts it.
    #[test]
    fn s2_pins_the_seed_ahead_of_the_selection() {
        let sel = vec!["zend/src/a.rs".to_string(), "AGENTS.md".to_string()];
        assert_eq!(paths(Arm::S1, &sel), sel);
        assert_eq!(
            paths(Arm::S2, &sel),
            vec!["README.md", "ARCHITECTURE.md", "AGENTS.md", "zend/src/a.rs"]
        );
    }

    #[test]
    fn only_s2_is_wide() {
        assert_eq!(budget_chars(Arm::S1), 96_000);
        assert_eq!(budget_chars(Arm::M1), 96_000);
        assert_eq!(budget_chars(Arm::U1), 96_000);
        assert_eq!(budget_chars(Arm::S2), 384_000);
    }
}
