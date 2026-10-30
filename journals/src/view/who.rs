//! How a line's writer and time read: "Yen via grok", "the server for John
//! via grok", "zend" — and how long ago.

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

/// Who wrote `line`, for whom, through what.
pub fn who(tree: &Tree, line: &Line) -> String {
    let person = line.person.as_ref().map(|p| name_of(tree, p));
    match (line.by, person) {
        (Writer::Channel(c), Some(p)) => format!("{p} via {c}"),
        (Writer::Channel(c), None) => c.to_string(),
        (Writer::Server, Some(p)) => match line.channel {
            Some(c) => format!("the server for {p} via {c}"),
            None => format!("the server for {p}"),
        },
        (Writer::Worker, Some(p)) => format!("a worker for {p}"),
        (Writer::Npcd, Some(p)) => format!("npcd for {p}"),
        (w, _) => w.name().to_string(),
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

/// One line as a reader sees it: `#id 2026-10-30 09:12 note, Yen via grok: text`.
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
    use crate::ops::fixture::sharratt;
    use crate::scope::Channel;
    use crate::testing::{at, note, slug};

    #[test]
    fn writers_read_as_who_they_are() {
        let t = sharratt();
        let mut l = note("2026-10-30T09:12:00Z", "x");
        assert_eq!(who(&t, &l), "john via claude-code");
        l.by = Writer::Server;
        l.channel = Some(Channel::Grok);
        assert_eq!(who(&t, &l), "the server for john via grok");
        l.by = Writer::Zend;
        l.person = None;
        assert_eq!(who(&t, &l), "zend");
        l.by = Writer::Worker;
        l.person = Some(slug("yen"));
        assert_eq!(who(&t, &l), "a worker for yen");
        l.person = Some(slug("nobody"));
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
        let t = sharratt();
        let mut l = note("2026-10-30T09:12:40Z", "Prefers tea.");
        l.proposed = true;
        assert_eq!(
            render_line(&t, &l),
            format!(
                "#{} 2026-10-30 09:12 proposal, john via claude-code: Prefers tea.",
                l.id
            )
        );
    }
}
