//! The checked-in prompt bodies, available outside `#[cfg(test)]`.
//!
//! These live here rather than in [`super::utils`] because they have two kinds of
//! consumer. The forward gates reach them from inside this crate's test build,
//! and measurement harnesses in crates *above* this one — which cannot see a
//! `#[cfg(test)]` module — need the identical bytes: a throughput figure from one
//! prompt is not comparable to a throughput figure from another, however similar
//! the two look.
//!
//! **One definition, deliberately.** `story_prompt` normalises the file's line
//! endings once. `story.md` is checked in with CRLF on Windows, and a prompt
//! carrying `\r\n` tokenises differently from the same text with `\n`, so a second
//! `include_str!` of it anywhere would be a different prompt that merely looked
//! identical in the source — and the two would then be silently measuring
//! different work.

/// The story a `TestMode::StoryRewrite` prompt asks the model to reproduce with
/// its character renamed.
///
/// Line endings are normalised here, once — see the module note on why a second
/// `include_str!` of this file is not the same prompt.
pub fn story_prompt() -> String {
    include_str!("story.md")
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

/// The system prompt the batched gates frame their sessions with.
pub fn system_prompt() -> String {
    include_str!("system.md")
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

/// The per-session names a `StoryRewrite` assigns, in file order.
///
/// Session `n` takes `names[n % names.len()]`, so a harness that wants the same
/// session identities as the gate indexes this the same way. The list runs from
/// two-character names to twenty-character ones on purpose: a rename that only
/// works for short single-token names passes a gate built from `Bo` and fails on
/// `Benjaminchristopher`.
pub fn session_names() -> Vec<String> {
    include_str!("names.md")
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The user prompt for session `n`: the story with its rename instruction filled
/// in.
pub fn story_rewrite_prompt(story: &str, name: &str) -> String {
    story.replace("{INSERT_NAME}", name)
}

/// What session `n` must produce: the story with the protagonist renamed.
///
/// All three cases are replaced because the story refers to the protagonist in
/// each, and a rewrite that gets one wrong is a different string — the same three
/// replacements the gate's own `create_test_run` performs.
pub fn story_rewrite_expected(story: &str, name: &str) -> String {
    let capitalized = capitalize(name);
    story
        .replace("Marcus", &capitalized)
        .replace("marcus", &name.to_lowercase())
        .replace("MARCUS", &name.to_uppercase())
}

/// The capitalisation the gate assigns a session name: first letter upper, rest
/// lower.
pub fn capitalize(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str().to_lowercase()),
    }
}
