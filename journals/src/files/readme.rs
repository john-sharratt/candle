//! `README.md` at the top of each group's folder: what the group is, who
//! reads it, and what goes where — for a person or an AI opening the
//! repository cold. Written by the service from `group.yaml` and the sections
//! fixed in code, so it is never out of step with either; a hand edit to it is
//! written over.

use crate::files::group::GroupFile;
use crate::scope::{Naming, Section};

/// The file's name.
pub const README: &str = "README.md";

/// What every journal folder holds, the same in every README.
const LAYOUT: &str = "In every journal's folder, `journal.yaml` says what the journal is, \
lists its references to other journals, and summarises it; each `yyyy-mm-dd-hhmm-xxxx.yaml` \
beside it is one entry, named for the minute it was written, so the folder lists its entries \
in order; `archive/<period>/` holds the entries of a period that has folded; and each other \
folder is a sub-journal. A section's folder appears once something is written in it. The \
tools only add to the tree; moving, merging or renaming journals is a coding change to this \
repository.";

fn sections() -> String {
    Section::ALL
        .into_iter()
        .map(|s| {
            let named = match s.naming(1) {
                Naming::Dated => ", each named for the minute it began and the channel it began on",
                Naming::Named => "",
            };
            format!(
                "- `{}/` — {}: {}{named}.\n",
                s.name(),
                cue_noun(s),
                s.kept_by()
            )
        })
        .collect()
}

/// A section's cue without its `open for`.
fn cue_noun(s: Section) -> &'static str {
    s.cue().strip_prefix("open for ").unwrap_or(s.cue())
}

pub fn group_readme(g: &GroupFile) -> String {
    let about = g
        .about
        .as_deref()
        .map(|a| format!("{a}\n\n"))
        .unwrap_or_default();
    format!(
        "# @{group} — {name}\n\n\
         {about}\
         This folder holds the group's journals, and its master journal is `journal.yaml` here. \
         Every member's AIs read all of it and write in it; a bot reads only what `group.yaml` \
         grants it. An AI addresses them as `@{group}/…`.\n\n\
         - `group.yaml` — the group: what it is for, and who is in it as what.\n\
         - `journal.yaml` — the master journal and its summary.\n\
         {sections}\n\
         {LAYOUT}\n",
        group = g.group,
        name = g.name,
        sections = sections(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::slug;

    #[test]
    fn a_groups_readme_says_what_it_is_and_what_goes_where() {
        let g = GroupFile {
            group: slug("zen"),
            name: "Zen".into(),
            about: Some("John and Yen's household and Tokera's work.".into()),
            members: vec![],
        };
        assert_eq!(
            group_readme(&g),
            format!(
                "# @zen — Zen\n\n\
                 John and Yen's household and Tokera's work.\n\n\
                 This folder holds the group's journals, and its master journal is \
                 `journal.yaml` here. Every member's AIs read all of it and write in it; a bot \
                 reads only what `group.yaml` grants it. An AI addresses them as `@zen/…`.\n\n\
                 - `group.yaml` — the group: what it is for, and who is in it as what.\n\
                 - `journal.yaml` — the master journal and its summary.\n\
                 - `people/` — the people the group talks to and about: a journal each, kept by every member's AIs.\n\
                 - `conversations/` — each conversation, from any client: a conversation each, opened by the connector as it begins, each named for the minute it began and the channel it began on.\n\
                 - `topics/` — everything durable the group talks about: a journal each, kept by every member's AIs.\n\
                 - `threads/` — Zen feature threads, their workers and parts: kept by zend for Zen's threads and workers.\n\
                 - `bots/` — the group's bots and what each did: kept by npcd for the group's bots.\n\n\
                 {LAYOUT}\n"
            )
        );
    }
}
