//! A conversation's catch-up (`docs/npc_conversations.md` §6.1): what the
//! character's journal says happened since it last spoke with its party,
//! carried as one section, `catchup/now`, of the `catchup` collection.
//!
//! Read from the bookmark less the overlap, leaving out what was seen, fitted
//! to [`CATCHUP_CHARS`] with what matters to a conversation first — by
//! [`journals::view::since`]. This member's lines not yet written are merged
//! in after it. With more than [`VERBATIM_LINES`] read, the turn is given a
//! summary the character makes of them; with fewer, the lines themselves.

use std::collections::BTreeSet;

use journals::actor::Actor;
use journals::line::LineId;
use journals::ops::episodes::Lived;
use journals::path::JournalPath;
use journals::time::Stamp;
use journals::tree::Tree;
use journals::view::since::{since, OVERLAP};

/// The catch-up's budget, in characters: between the frame's Chat and Worker
/// budgets.
pub const CATCHUP_CHARS: usize = 12_000;

/// The most lines a turn is given verbatim; more are summarised.
pub const VERBATIM_LINES: usize = 6;

/// The collection a catch-up is carried in.
pub const COLLECTION: &str = "catchup";

/// Its one section.
pub const SECTION: &str = "catchup/now";

/// The most a summary decodes.
pub const SUMMARY_TOKENS: usize = 192;

/// What a turn catches up on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CatchUp {
    /// The section's text: one line each, with its `#id`.
    pub text: String,
    /// How many lines it holds.
    pub lines: usize,
    /// The ids read within the overlap before `now`: the next turn's
    /// `talk.seen`.
    pub seen: BTreeSet<LineId>,
}

impl CatchUp {
    /// Whether the turn is given a summary rather than the lines.
    pub fn summarised(&self) -> bool {
        self.lines > VERBATIM_LINES
    }
}

/// One line as the section shows it.
fn shown(id: Option<&LineId>, at: Stamp, text: &str) -> String {
    match id {
        Some(id) => format!("#{id} {at} {text}"),
        None => format!("{at} {text}"),
    }
}

/// The catch-up for a turn at `now` on the bookmark `after`, having seen
/// `seen`, over `paths`, with this member's unwritten lines `pending`.
pub fn gather(
    tree: &Tree,
    actor: &Actor,
    paths: &[JournalPath],
    after: Stamp,
    seen: &BTreeSet<LineId>,
    pending: &[Lived],
    now: Stamp,
) -> Result<CatchUp, String> {
    let read = since(tree, actor, paths, after, seen, CATCHUP_CHARS).map_err(|e| e.to_string())?;
    let start = after.plus(-OVERLAP);
    let mut rows: Vec<(Stamp, String)> = read
        .lines
        .iter()
        .map(|r| (r.line.at, shown(Some(&r.line.id), r.line.at, &r.line.text)))
        .collect();
    rows.extend(
        pending
            .iter()
            .filter(|l| l.at >= start)
            .map(|l| (l.at, shown(None, l.at, &l.text))),
    );
    rows.sort_by_key(|(at, _)| *at);
    let next_seen = read
        .lines
        .iter()
        .filter(|r| r.line.at >= now.plus(-OVERLAP))
        .map(|r| r.line.id.clone())
        .collect();
    Ok(CatchUp {
        lines: rows.len(),
        text: rows
            .into_iter()
            .map(|(_, line)| line)
            .collect::<Vec<_>>()
            .join("\n"),
        seen: next_seen,
    })
}

/// A gap as a conversation speaks of it: `3 days`, `an hour`, `a moment`.
pub fn gap(secs: i64) -> String {
    let plural = |n: i64, unit: &str| match n {
        1 => format!("a{} {unit}", if unit == "hour" { "n" } else { "" }),
        n => format!("{n} {unit}s"),
    };
    match secs {
        s if s < 60 => "a moment".to_string(),
        s if s < 3_600 => plural(s / 60, "minute"),
        s if s < 86_400 => plural(s / 3_600, "hour"),
        s => plural(s / 86_400, "day"),
    }
}

/// The question a long gap is summarised by.
pub fn summary_question(party: &str, gap: &str) -> String {
    format!(
        "What has happened since you last spoke with {party}, {gap} ago, that matters to this \
         conversation? Cite each line you use by its #id."
    )
}

#[cfg(test)]
mod tests {
    use journals::experience::{Experience, ExperienceKind};

    use super::*;

    #[test]
    fn a_gap_reads_as_a_person_would_say_it() {
        assert_eq!(gap(5), "a moment");
        assert_eq!(gap(60), "a minute");
        assert_eq!(gap(600), "10 minutes");
        assert_eq!(gap(3_600), "an hour");
        assert_eq!(gap(7_200), "2 hours");
        assert_eq!(gap(3 * 86_400), "3 days");
        assert_eq!(gap(86_400), "a day");
    }

    #[test]
    fn a_long_gap_is_asked_about_with_citations() {
        assert_eq!(
            summary_question("Mira", "3 days"),
            "What has happened since you last spoke with Mira, 3 days ago, that matters to this \
             conversation? Cite each line you use by its #id."
        );
    }

    #[test]
    fn unwritten_lines_join_the_catch_up_in_time_order_without_an_id() {
        let t = Tree::default();
        let at = Stamp::parse("2026-11-10T09:00:00Z").unwrap();
        let pending = [
            Lived {
                at: at.plus(30),
                text: "Mira, on the phone: is it in?".into(),
                experience: Experience::of(ExperienceKind::Heard),
            },
            Lived {
                at: at.plus(-3_600),
                text: "long before".into(),
                experience: Experience::of(ExperienceKind::Did),
            },
        ];
        let c = gather(
            &t,
            &Actor::Zend,
            &[],
            at,
            &BTreeSet::new(),
            &pending,
            at.plus(60),
        )
        .unwrap();
        assert_eq!(
            c,
            CatchUp {
                text: "2026-11-10T09:00:30Z Mira, on the phone: is it in?".into(),
                lines: 1,
                seen: BTreeSet::new(),
            }
        );
        assert!(!c.summarised());
    }
}
