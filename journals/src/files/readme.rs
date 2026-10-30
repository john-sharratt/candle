//! `README.md` at the top of each person's and group's folder: what the
//! folder is, who reads it, and what goes where — for a person or an AI
//! opening the repository cold. Written by the service from the owner's file
//! and the sections fixed in code, so it is never out of step with either;
//! a hand edit to it is written over.

use crate::files::group::GroupFile;
use crate::files::person::PersonFile;
use crate::scope::{Naming, Scope, Section};

/// The file's name.
pub const README: &str = "README.md";

/// What every journal folder holds, the same in every README.
const LAYOUT: &str = "In every journal's folder, `journal.yaml` says what the journal is and \
summarises it; each `yyyy-mm-dd-hhmm-xxxx.yaml` beside it is one entry, named for the minute \
it was written, so the folder lists its entries in order; `archive/<period>/` holds the \
entries of a period that has folded; and each other folder is a sub-journal. A section's \
folder appears once something is written in it.";

fn sections(scope: Scope) -> String {
    Section::of(scope)
        .map(|s| {
            let named = match s.naming(2) {
                Naming::Dated => ", each conversation named for when it began",
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
        .map(|a| format!(" {a}"))
        .unwrap_or_default();
    format!(
        "# @{group} — {name}\n\n\
         A {kind}.{about}\n\n\
         This folder holds the group's journals. Every member's AIs read and write them — \
         a member listed as a reader only reads — and each bot `group.yaml` lists reads them \
         too. An AI addresses them as `@{group}/…`.\n\n\
         - `group.yaml` — the group: its kind, what it is for, and who is in it as what.\n\
         - `journal.yaml` — the group's own journal and its summary.\n\
         {sections}\n\
         {LAYOUT}\n",
        group = g.group,
        name = g.name,
        kind = g.kind.name(),
        sections = sections(Scope::Group),
    )
}

pub fn person_readme(p: &PersonFile) -> String {
    format!(
        "# ~ — {name}\n\n\
         This folder holds {name}'s own journals. Only {name}'s AIs read them; nothing \
         here is shared with a group. An AI of {name}'s addresses them as `~/…`.\n\n\
         - `person.yaml` — who {name} is here: their accounts and their groups.\n\
         - `journal.yaml` — their own journal and its summary.\n\
         {sections}\n\
         {LAYOUT}\n",
        name = p.name,
        sections = sections(Scope::Personal),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::group::Kind;
    use crate::testing::slug;

    #[test]
    fn a_groups_readme_says_what_it_is_and_what_goes_where() {
        let g = GroupFile {
            group: slug("tokera"),
            name: "Tokera".into(),
            kind: Kind::Team,
            about: Some("John's company: the candle engine, Zen Code and their papers.".into()),
            members: vec![],
        };
        assert_eq!(
            group_readme(&g),
            format!(
                "# @tokera — Tokera\n\n\
                 A team. John's company: the candle engine, Zen Code and their papers.\n\n\
                 This folder holds the group's journals. Every member's AIs read and write \
                 them — a member listed as a reader only reads — and each bot `group.yaml` \
                 lists reads them too. An AI addresses them as `@tokera/…`.\n\n\
                 - `group.yaml` — the group: its kind, what it is for, and who is in it as what.\n\
                 - `journal.yaml` — the group's own journal and its summary.\n\
                 - `projects/` — repositories and durable projects: a journal each, kept by every member's AIs.\n\
                 - `topics/` — themes that span conversations: a journal each, kept by every member's AIs.\n\
                 - `threads/` — Zen feature threads, their workers and parts: kept by zend for Zen's threads and workers.\n\
                 - `bots/` — the group's bots and what each did: kept by npcd for the group's bots.\n\n\
                 {LAYOUT}\n"
            )
        );
    }

    #[test]
    fn a_persons_readme_says_it_is_theirs_alone() {
        let p = PersonFile {
            person: slug("john"),
            name: "John".into(),
            accounts: vec![],
            groups: vec![],
            default_group: None,
            framed: vec![],
        };
        let r = person_readme(&p);
        assert!(
            r.starts_with("# ~ — John\n\nThis folder holds John's own journals."),
            "{r}"
        );
        assert!(
            r.contains(
                "- `channels/` — each channel they speak through and its conversations: a \
             conversation each, opened by the connector as it begins, each conversation named \
             for when it began.\n"
            ),
            "{r}"
        );
        assert!(
            r.contains(
                "- `people/` — the people in their life: a journal each, kept by your AIs.\n"
            ),
            "{r}"
        );
    }
}
