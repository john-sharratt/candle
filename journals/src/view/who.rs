//! How a line's writer and time read: "Yen via Grok on John's account",
//! "John via Claude Code", "the server for John via Grok", "Zen" — and how
//! long ago.

use crate::line::{Line, Writer};
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::Tree;

/// A person by their own name, or their slug when their file does not read.
pub fn name_of(tree: &Tree, person: &Slug) -> String {
    tree.person_file(person)
        .map(|f| f.name.clone())
        .unwrap_or_else(|| person.to_string())
}

/// Who wrote `line`, for whom, through what. The speaker is named first when
/// the focus had one; the account it came through is said when it is
/// someone else's (§3.2).
pub fn who(tree: &Tree, line: &Line) -> String {
    let account = line.account.as_ref().map(|p| name_of(tree, p));
    let speaker = line.speaker.as_ref().map(|p| name_of(tree, p));
    match (line.by, account) {
        (Writer::Channel(c), account) => match (speaker, account) {
            (Some(s), Some(a)) if s != a => format!("{s} via {} on {a}'s account", c.spoken()),
            (Some(s), _) | (None, Some(s)) => format!("{s} via {}", c.spoken()),
            (None, None) => c.spoken().to_string(),
        },
        (Writer::Server, Some(a)) => match line.channel {
            Some(c) => format!("the server for {a} via {}", c.spoken()),
            None => format!("the server for {a}"),
        },
        (Writer::Worker, Some(a)) => format!("a worker for {a}"),
        (Writer::Npcd, Some(a)) => format!("npcd for {a}"),
        (Writer::Zend, _) => "Zen".to_string(),
        (w, None) => w.name().to_string(),
    }
}

/// `then` from `now`: "just now", "5 min ago", "2 h ago", "3 d ago".
pub fn ago(then: Stamp, now: Stamp) -> String {
    let s = (now.secs() - then.secs()).max(0);
    match s {
        0..=59 => "just now".into(),
        60..=3_599 => format!("{} min ago", s / 60),
        3_600..=86_399 => format!("{} h ago", s / 3_600),
        _ => format!("{} d ago", s / 86_400),
    }
}

/// `2026-10-30 09:12`.
pub fn minute(at: Stamp) -> String {
    let s = at.to_string();
    format!("{} {}", &s[..10], &s[11..16])
}

/// One line as a reader sees it: `#id 2026-10-30 09:12 note, Yen via Grok: text`.
pub fn render_line(tree: &Tree, line: &Line) -> String {
    let kind = if line.proposed {
        "proposal"
    } else {
        line.kind.name()
    };
    format!(
        "#{} {} {kind}, {}: {}",
        line.id,
        minute(line.at),
        who(tree, line),
        line.text
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::fixture::zen;
    use crate::scope::Channel;
    use crate::testing::{at, note, slug};

    #[test]
    fn writers_read_as_who_they_are() {
        let t = zen();
        let mut l = note("2026-10-30T09:12:00Z", "x");
        assert_eq!(who(&t, &l), "john via Claude Code");
        l.speaker = Some(slug("john"));
        assert_eq!(who(&t, &l), "john via Claude Code", "his own account");
        l.by = Writer::Channel(Channel::Grok);
        l.speaker = Some(slug("yen"));
        assert_eq!(who(&t, &l), "yen via Grok on john's account");
        l.account = None;
        assert_eq!(who(&t, &l), "yen via Grok");
        l.speaker = None;
        assert_eq!(who(&t, &l), "Grok");
        l.by = Writer::Server;
        l.account = Some(slug("john"));
        l.channel = Some(Channel::Grok);
        assert_eq!(who(&t, &l), "the server for john via Grok");
        l.by = Writer::Zend;
        l.account = None;
        assert_eq!(who(&t, &l), "Zen");
        l.by = Writer::Worker;
        l.account = Some(slug("yen"));
        assert_eq!(who(&t, &l), "a worker for yen");
        l.account = Some(slug("nobody"));
        assert_eq!(who(&t, &l), "a worker for nobody");
    }

    #[test]
    fn ages_round_down() {
        let now = at("2026-10-30T12:00:00Z");
        assert_eq!(ago(now, now), "just now");
        assert_eq!(ago(now.plus(-59), now), "just now");
        assert_eq!(ago(now.plus(-60), now), "1 min ago");
        assert_eq!(ago(now.plus(-7_199), now), "1 h ago");
        assert_eq!(ago(now.plus(-86_400 * 3), now), "3 d ago");
        assert_eq!(
            ago(now.plus(100), now),
            "just now",
            "a clock behind is not the future"
        );
    }

    #[test]
    fn a_line_reads_on_one_line() {
        let t = zen();
        let mut l = note("2026-10-30T09:12:40Z", "Prefers tea.");
        l.proposed = true;
        assert_eq!(
            render_line(&t, &l),
            format!(
                "#{} 2026-10-30 09:12 proposal, john via Claude Code: Prefers tea.",
                l.id
            )
        );
    }
}
