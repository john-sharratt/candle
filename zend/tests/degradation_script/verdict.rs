//! Judging a final turn: did it answer the question the conversation ended on?
//!
//! The script's last question asks for an OPINION of a paper the model reads in
//! that same turn. What this catches is the model answering something else — an
//! earlier question in the conversation (the square root, the repo), nothing
//! about the paper at all, or a summary of it where a view was asked for.
//!
//! The reasoning and the answer are judged separately, because the reasoning is
//! where the model says which question it believes it is answering: the live
//! failures read "The user asked me to tell them about this conversation's
//! topic" and "Let me provide the answer to their original question about the
//! square root" there, before an answer that could otherwise pass for on-topic.
//!
//! The markers are tied to the questions, not to their words in general. The
//! paper discusses error growth, so a bare "square root" can appear in a correct
//! answer; the square root the script asks for is of one specific number, and
//! that number is what gives a wrong answer away.

/// What a final turn did with the question it was asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Discussed the paper and gave a view of it.
    Answered,
    /// Answered a question from earlier in the conversation instead.
    EarlierQuestion(&'static str),
    /// Discussed the paper but gave no view of it — a summary where an opinion
    /// was asked for.
    NoOpinion,
    /// Said nothing about the paper at all.
    OffTopic,
    /// Never read the paper — a tool call that found nothing, a file it could
    /// not locate — so the conversation says nothing about what happens after
    /// the read, which is what the script measures.
    NeverRead,
}

impl Verdict {
    pub fn passed(&self) -> bool {
        matches!(self, Verdict::Answered)
    }
}

/// The square-root question, recognised by its number in any rendering the
/// model uses, or by the model naming that question in its reasoning.
const SQUARE_ROOT: &[&str] = &[
    "237891273498722349871229384712934",
    "237,891,273",
    "1.5423",
    "15,423,72",
    "15423724",
    "square root of 237",
    "sqrt of 237",
    "question about the square root",
    "the square root question",
];

/// The repo question, as the model names it in its reasoning when it believes
/// that is what it was asked. Reasoning only: an answer about the paper may
/// well say what the repository implements.
const REPO: &[&str] = &[
    "conversation's topic",
    "asked about this repo",
    "asked about the repo",
    "asking about this repo",
    "asking about the repo",
    "tell them about this repo",
    "tell them about the repo",
    "tell me about this repo",
    "about this repository",
    "what this repo is about",
    "this repo is about",
    "describe this repo",
    "what's in this conversation",
];

/// The paper, as an answer about it cannot avoid naming it.
const PAPER: &[&str] = &["paper", "unbounded", "o(1)", "provenance"];

/// A view of the paper rather than a description of it — first-person, or the
/// evaluative words a review is written in ("what's solid", "weaker",
/// "overall it reads as a credible v1").
const OPINION: &[&str] = &[
    "i think",
    "impressions",
    "i find",
    "i found",
    "in my view",
    "my take",
    "my view",
    "my opinion",
    "my assessment",
    "my thoughts",
    "i'd say",
    "i would say",
    "i'd argue",
    "i would argue",
    "i'm impressed",
    "impressive",
    "compelling",
    "strength",
    "strong point",
    "weak",
    "solid",
    "credible",
    "ambitious",
    "promising",
    "convincing",
    "persuasive",
    "stands out",
    "reads as",
    "overall",
    "concern",
    "limitation",
    "critique",
    "caveat",
    "skeptic",
    "sceptic",
];

/// The head the file tool writes on a read of the paper, as it sits in the
/// conversation's tool results: `docs/unbounded_agents.md (lines 1-1252):`.
const PAPER_READ: &str = "unbounded_agents.md (lines";

/// Whether any of `user_turns` — the conversation's user-side texts, tool
/// results included — carries a read of the paper.
pub fn read_the_paper<'a>(user_turns: impl IntoIterator<Item = &'a str>) -> bool {
    user_turns.into_iter().any(|t| t.contains(PAPER_READ))
}

/// The verdict on one conversation: [`Verdict::NeverRead`] when none of
/// `user_turns` — its user-side texts, tool results included — carries a read
/// of the paper, and [`judge`]'s verdict on `final_turn` otherwise.
pub fn verdict_for<'a>(final_turn: &str, user_turns: impl IntoIterator<Item = &'a str>) -> Verdict {
    if read_the_paper(user_turns) {
        judge(final_turn)
    } else {
        Verdict::NeverRead
    }
}

/// Judge `final_turn` — the conversation's last assistant turn as stored,
/// reasoning block included.
pub fn judge(final_turn: &str) -> Verdict {
    let (reasoning, answer) = split_reasoning(final_turn);
    let reasoning = reasoning.to_lowercase();
    let answer = answer.to_lowercase();
    if contains_any(&reasoning, SQUARE_ROOT) || contains_any(&answer, SQUARE_ROOT) {
        return Verdict::EarlierQuestion("the square root");
    }
    if contains_any(&reasoning, REPO) {
        return Verdict::EarlierQuestion("the repo");
    }
    if !contains_any(&answer, PAPER) {
        return Verdict::OffTopic;
    }
    if !contains_any(&answer, OPINION) {
        return Verdict::NoOpinion;
    }
    Verdict::Answered
}

/// `(reasoning, answer)`: the text inside the turn's `<think>` block and the
/// text after it. A turn with no block is all answer.
fn split_reasoning(turn: &str) -> (&str, &str) {
    const CLOSE: &str = "</think>";
    match turn.find(CLOSE) {
        Some(end) => {
            let reasoning = &turn[..end];
            let reasoning = reasoning
                .trim_start()
                .strip_prefix("<think>")
                .unwrap_or(reasoning);
            (reasoning, &turn[end + CLOSE.len()..])
        }
        None => ("", turn),
    }
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| text.contains(n))
}

#[cfg(test)]
mod tests {
    use super::{judge, read_the_paper, split_reasoning, verdict_for, Verdict};

    /// a40740b1's final turn, as stored: the model said it was answering the
    /// repo question, then summarised the paper without a view of it.
    const SUMMARISED_THE_CONVERSATION: &str = "<think>\nThe user asked me to tell them \
        about this conversation's topic. I've now read the full paper. Let me provide a \
        comprehensive, informative summary of what the project is about.\n</think>\n\n\
        # One Card, One Stack\n\nThe paper describes an inference engine with bounded \
        numerical error at any context depth, a three-tier KV cache and adaptive \
        quantization.";

    /// 085044ec / 515f45ab: the answer to the square root, after reading the paper.
    const ANSWERED_THE_SQUARE_ROOT: &str = "<think>\nLet me provide the answer to their \
        original question about the square root.\n</think>\n\nThe square root of \
        **237,891,273,498,722,349,871,229,384,712,934** is approximately \
        **1.5423724371847494 × 10^16**.";

    const GAVE_AN_OPINION: &str = "<think>\nThe user wants my thoughts on the paper they \
        pointed me at. I've read it; let me give an honest assessment.\n</think>\n\n\
        I think the paper makes a compelling case for provenance-selected attention. Its \
        main strength is the O(1) error argument; my main concern is that the evaluation \
        leans on a single card.";

    /// flash_next_selected's second conversation on 2026-09-16: a review written
    /// in "what's solid / what's weaker / overall" rather than in the first
    /// person — an answer, which the first marker list scored as a summary.
    const A_REVIEW_WITHOUT_THE_FIRST_PERSON: &str = "<think>\nThe user asked me to read \
        the document and give my assessment. I've now read the full 1252-line technical \
        report. Let me synthesize a clear, complete summary in a paragraph or two that \
        captures what this document is, what it claims, what's strong, and where it's \
        incomplete — without being longer than needed.\n</think>\n\nThis is a v1 technical \
        report for a system called **One Card, One Stack**, a research fork of Hugging \
        Face's Candle framework written in Rust.\n\n**What's solid:** The kernel benchmarks \
        (§9.4–9.6) are concrete and reproducible.\n\n**What's still open / weaker:** The \
        theorem's practical punch depends on the retrieval quality assumption — the paper \
        acknowledges you've traded O(N) numerical error for a retrieval-quality dependency \
        it bounds but cannot eliminate. Overall it reads as a credible, ambitious v1 whose \
        theoretical framing is clean but whose empirical weight rests heavily on the live \
        demo and v2 numbers.";

    /// flash_next_selected's sixth conversation on 2026-09-16: an opinion of the
    /// paper, written by a model that says it is answering the repo question.
    /// The answer fits only because the README and the paper describe the same
    /// project — the question it was asked is still lost, which is the failure.
    const THE_REPO_QUESTION_ANSWERED_WITH_THE_PAPER: &str = "<think>\nThe user asked me to \
        describe what this repo is about. I've now read the full technical report (1252 \
        lines). Let me synthesize a clear, complete summary in a paragraph or two.\n\
        </think>\n\nThis is a dense, ambitious paper. Here's my honest read:\n\n**What it \
        is:** A Rust reimplementation of the Hugging Face Candle inference stack, rebuilt \
        with a hard 16 GB VRAM ceiling as a design constraint.";

    /// flash_next_selected's seventh conversation on 2026-09-16: the paper as the
    /// subject but the repo question's verb — "tell me about" — in place of the
    /// "tell me what you think" it was asked.
    const THE_CONVERSATIONS_CONTENTS: &str = "<think>\nThe user asked me to describe \
        what's in this conversation/project. I've now read the full technical report. Let \
        me synthesize a clear, concise summary of what this document contains and its key \
        claims, strengths, and potential concerns.\n\nThe user asked \"tell me about\" the \
        paper — they want me to describe/summarize it.\n</think>\n\nThis is a v1 technical \
        report for a system called **\"One Card, One Stack\"**. It's a dense, ambitious \
        paper with a central theoretical claim wrapped around four engineering \
        contributions.";

    /// flash_next_selected's eighth conversation on 2026-09-16: "my honest
    /// impressions" and "what's genuinely strong" — an answer the first marker
    /// list scored as a summary.
    const HONEST_IMPRESSIONS: &str = "<think>\nThe user wants me to give my thoughts on \
        this technical report. Let me read through it carefully and form a genuine \
        assessment.\n</think>\n\nThis is a clearly-written systems paper. Here are my honest \
        impressions:\n\n**What's genuinely strong:** the O(N) → O(1) architectural \
        observation applied to KV quantization.";

    const SUMMARY_WITHOUT_A_VIEW: &str = "<think>\nThe user wants me to read the paper and \
        say what I think.\n</think>\n\nThe paper presents an unbounded-context engine built \
        from three tiers of KV cache and per-block quantization.";

    const ABOUT_SOMETHING_ELSE: &str = "<think>\nThe user asked what time it is.\n</think>\n\n\
        It's 15:01 UTC.";

    #[test]
    fn the_live_repo_failure_is_an_earlier_question() {
        assert_eq!(
            judge(SUMMARISED_THE_CONVERSATION),
            Verdict::EarlierQuestion("the repo")
        );
    }

    #[test]
    fn the_live_square_root_failure_is_an_earlier_question() {
        assert_eq!(
            judge(ANSWERED_THE_SQUARE_ROOT),
            Verdict::EarlierQuestion("the square root")
        );
    }

    #[test]
    fn a_view_of_the_paper_is_an_answer() {
        assert_eq!(judge(GAVE_AN_OPINION), Verdict::Answered);
        assert!(judge(GAVE_AN_OPINION).passed());
    }

    /// The reasoning decides: a fitting answer to the wrong question is still
    /// the wrong question.
    #[test]
    fn a_paper_opinion_given_to_the_repo_question_is_an_earlier_question() {
        assert_eq!(
            judge(THE_REPO_QUESTION_ANSWERED_WITH_THE_PAPER),
            Verdict::EarlierQuestion("the repo")
        );
    }

    #[test]
    fn describing_the_conversations_contents_is_the_repo_question() {
        assert_eq!(
            judge(THE_CONVERSATIONS_CONTENTS),
            Verdict::EarlierQuestion("the repo")
        );
    }

    #[test]
    fn a_review_in_evaluative_words_is_an_answer() {
        assert_eq!(judge(A_REVIEW_WITHOUT_THE_FIRST_PERSON), Verdict::Answered);
        assert_eq!(judge(HONEST_IMPRESSIONS), Verdict::Answered);
    }

    #[test]
    fn a_summary_where_a_view_was_asked_for_is_not_an_answer() {
        assert_eq!(judge(SUMMARY_WITHOUT_A_VIEW), Verdict::NoOpinion);
        assert!(!judge(SUMMARY_WITHOUT_A_VIEW).passed());
    }

    #[test]
    fn a_turn_that_never_names_the_paper_is_off_topic() {
        assert_eq!(judge(ABOUT_SOMETHING_ELSE), Verdict::OffTopic);
    }

    /// The paper's own subject matter — error growth — must not read as the
    /// square-root question.
    #[test]
    fn the_papers_own_square_roots_are_not_the_square_root_question() {
        let turn = "<think>\nThe user wants my opinion of the paper.\n</think>\n\nI think \
            the paper's strongest point is that error stays O(1) where full attention's \
            grows with the square root of the depth.";
        assert_eq!(judge(turn), Verdict::Answered);
    }

    /// qwen36's second conversation on 2026-09-16 listed nothing, looked for
    /// `paper.md`, and asked for the file: no read, so nothing to judge.
    #[test]
    fn only_a_tool_result_holding_the_paper_counts_as_a_read() {
        let read = "<tool_response>\ndocs/unbounded_agents.md (lines 1-1252):\n\n```markdown\n   \
                    1  # One Card, One Stack";
        let listed_nothing = "<tool_response>{\"total\": 0, \"files\": []}</tool_response>";
        assert!(read_the_paper(["hi - how are you?", read]));
        assert!(!read_the_paper([
            "read the unbounded context paper and tell me what you think",
            listed_nothing,
        ]));
    }

    /// A conversation is judged only once it has read the paper: the same final
    /// turn is an answer after a read and says nothing without one.
    #[test]
    fn a_conversation_that_never_read_the_paper_is_not_judged() {
        let read = "<tool_response>\ndocs/unbounded_agents.md (lines 1-1252):\n";
        let asked = "read the unbounded context paper and tell me what you think";
        assert_eq!(
            verdict_for(GAVE_AN_OPINION, [asked, read]),
            Verdict::Answered
        );
        assert_eq!(verdict_for(GAVE_AN_OPINION, [asked]), Verdict::NeverRead);
        assert!(!Verdict::NeverRead.passed());
    }

    #[test]
    fn the_reasoning_block_splits_from_the_answer() {
        assert_eq!(
            split_reasoning("<think>\nwhy\n</think>\n\nwhat"),
            ("\nwhy\n", "\n\nwhat")
        );
        assert_eq!(split_reasoning("no block"), ("", "no block"));
    }
}
