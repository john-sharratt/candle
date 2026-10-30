//! A conversation's frame (§6.4, §6.5): what it is given at its first call —
//! its own journal, each journal above it to the person, the person, and
//! each group they frame — nearest first, within its channel's budget. A
//! bot's frame is the groups that list it, and nothing of anyone's own.

use crate::actor::{framed_groups, groups_of_bot};
use crate::path::{JournalPath, Owner};
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{Journal, Tree};
use crate::view::index::{group_lines, index};
use crate::view::who::render_line;
use crate::view::NOT_INSTRUCTIONS;

/// How much a channel is given (§6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Budget {
    /// Spoken: digests and the latest summaries, all already in memory.
    Voice,
    /// claude.ai, Claude Code.
    Chat,
    /// A Zen worker, in its prompt.
    Worker,
}

impl Budget {
    pub fn chars(self) -> usize {
        match self {
            Budget::Voice => 1_600,
            Budget::Chat => 6_000,
            Budget::Worker => 32_000,
        }
    }
}

/// Blocks added in priority order until the budget is spent; the block
/// that does not fit is cut, and nothing after it is added.
struct Filler {
    out: String,
    left: usize,
    full: bool,
}

impl Filler {
    fn new(budget: usize) -> Filler {
        Filler {
            out: String::new(),
            left: budget,
            full: false,
        }
    }

    fn push(&mut self, block: &str) {
        if self.full || block.is_empty() {
            return;
        }
        let need = block.chars().count() + usize::from(!self.out.is_empty());
        if need <= self.left {
            if !self.out.is_empty() {
                self.out.push('\n');
            }
            self.out.push_str(block);
            self.left -= need;
            return;
        }
        self.full = true;
        // Room for the separator and the ellipsis, or nothing worth adding.
        if self.left > 16 {
            if !self.out.is_empty() {
                self.out.push('\n');
            }
            let cut: String = block.chars().take(self.left - 2).collect();
            self.out.push_str(cut.trim_end());
            self.out.push('…');
        }
        self.left = 0;
    }
}

fn summary_of(j: &Journal) -> Option<String> {
    j.summary().map(|s| s.text()).filter(|t| !t.is_empty())
}

fn digest_of(j: &Journal) -> Option<String> {
    j.summary()
        .and_then(|s| s.digest.clone())
        .or_else(|| summary_of(j))
}

/// How a frame names a group to its reader: "Your family @sharratt".
/// How a group's block is headed: what kind it is, and what it is for when
/// its `group.yaml` says.
fn group_label(tree: &Tree, group: &Slug) -> String {
    match tree.group_file(group) {
        Some(f) => match &f.about {
            Some(about) => format!("Your {} @{group} — {about}", f.kind),
            None => format!("Your {} @{group}", f.kind),
        },
        None => format!("Your group @{group}"),
    }
}

/// The frame for `person` at `at` — a conversation or a worker's journal —
/// with `also` (a worker's workspace projects) after the groups.
pub fn frame(
    tree: &Tree,
    person: &Slug,
    at: Option<&JournalPath>,
    also: &[JournalPath],
    budget: Budget,
    now: Stamp,
) -> String {
    let mut f = Filler::new(budget.chars());
    f.push(NOT_INSTRUCTIONS);
    let me = JournalPath::root(Owner::Person(person.clone()));
    if budget == Budget::Voice {
        voice(tree, &mut f, person, &me, at);
        return f.out;
    }
    if let Some(at) = at {
        if let Some(j) = tree.get(at).filter(|j| j.problems.is_empty()) {
            let mut block = vec![format!("This conversation, {at}:")];
            if let Some(s) = summary_of(j) {
                block.push(format!("  so far: {s}"));
            }
            // The newest lines that fit in half of what is left.
            let mut lines: Vec<String> = Vec::new();
            let mut room = f.left / 2;
            for l in j.window.iter().rev() {
                let r = format!("  {}", render_line(tree, l));
                let n = r.chars().count() + 1;
                if n > room {
                    break;
                }
                room -= n;
                lines.push(r);
            }
            lines.reverse();
            block.extend(lines);
            f.push(&block.join("\n"));
        }
        let mut up = at.parent();
        while let Some(p) = up {
            if p.is_root() {
                break;
            }
            if let Some(s) = tree.get(&p).and_then(summary_of) {
                f.push(&format!("{p}: {s}"));
            }
            up = p.parent();
        }
    }
    if let Some(root) = tree.get(&me) {
        let mut block = vec![match summary_of(root) {
            Some(s) => format!("You ({me}): {s}"),
            None => format!("You ({me}): nothing summarised yet."),
        }];
        block.extend(
            index(&me, root, now)
                .into_iter()
                .map(|l| format!("  - {l}")),
        );
        block.extend(
            group_lines(tree, person)
                .into_iter()
                .map(|l| format!("  - {l}")),
        );
        f.push(&block.join("\n"));
    }
    for group in framed_groups(tree, person) {
        let path = JournalPath::root(Owner::Group(group.clone()));
        if let Some(root) = tree.get(&path) {
            root_block(&mut f, &path, root, &group_label(tree, &group), now);
        }
    }
    for p in also {
        if let Some(j) = tree.get(p) {
            let s = summary_of(j).unwrap_or_else(|| "no summary yet".into());
            f.push(&format!("{p}: {s}"));
        }
    }
    f.out
}

/// The frame for `bot`: each group that lists it, its summary and index.
pub fn bot_frame(tree: &Tree, bot: &Slug, budget: Budget, now: Stamp) -> String {
    let mut f = Filler::new(budget.chars());
    f.push(NOT_INSTRUCTIONS);
    for group in groups_of_bot(tree, bot) {
        let path = JournalPath::root(Owner::Group(group.clone()));
        if let Some(root) = tree.get(&path) {
            root_block(&mut f, &path, root, &group_label(tree, &group), now);
        }
    }
    f.out
}

fn root_block(f: &mut Filler, path: &JournalPath, root: &Journal, who: &str, now: Stamp) {
    let mut block = vec![match summary_of(root) {
        Some(s) => format!("{who} ({path}): {s}"),
        None => format!("{who} ({path}): nothing summarised yet."),
    }];
    block.extend(
        index(path, root, now)
            .into_iter()
            .map(|l| format!("  - {l}")),
    );
    f.push(&block.join("\n"));
}

/// Spoken: digests and summaries only — no index, no lines.
fn voice(tree: &Tree, f: &mut Filler, person: &Slug, me: &JournalPath, at: Option<&JournalPath>) {
    if let Some(d) = tree.get(me).and_then(digest_of) {
        f.push(&format!("You: {d}"));
    }
    if let Some(group) = framed_groups(tree, person).first() {
        let path = JournalPath::root(Owner::Group(group.clone()));
        if let Some(d) = tree.get(&path).and_then(digest_of) {
            let kind = tree.group_file(group).map_or("group", |g| g.kind.name());
            f.push(&format!("Your {kind}: {d}"));
        }
    }
    let Some(at) = at else { return };
    let channel = at.parent();
    if let Some(c) = channel.as_ref() {
        if let Some(s) = tree.get(c).and_then(summary_of) {
            f.push(&format!("With this channel: {s}"));
        }
        // The last conversation before this one that has a summary.
        if let Some(cj) = tree.get(c) {
            let last = cj
                .children
                .iter()
                .filter(|(n, j)| Some(*n) != at.last() && j.summary().is_some())
                .max_by_key(|(_, j)| j.meta.as_ref().map(|m| m.created));
            if let Some((_, j)) = last {
                if let Some(s) = summary_of(j) {
                    f.push(&format!("Last time: {s}"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::files::summary::{Cite, Sentence};
    use crate::files::Kind;
    use crate::ops::fixture::{john, now, p, sharratt};
    use crate::ops::{apply, OnTaken, Op};
    use crate::scope::Channel;
    use crate::testing::slug;
    use crate::tree::fingerprint::fingerprint;

    fn summarise(t: &mut Tree, path: &JournalPath, text: &str, digest: Option<&str>) {
        let j = t.get(path).unwrap();
        let cite = match j.window.first() {
            Some(l) => Cite::Line(l.id.clone()),
            None => Cite::Child(
                j.live()
                    .find(|(_, c)| c.summary().is_some())
                    .unwrap()
                    .0
                    .clone(),
            ),
        };
        let over = fingerprint(j);
        let op = Op::Summarize {
            path: path.clone(),
            over,
            sentences: vec![Sentence {
                text: text.into(),
                cites: vec![cite],
            }],
            digest: digest.map(Into::into),
            cue: None,
        };
        apply(t, &Actor::Zend, &op, now()).unwrap();
    }

    /// John in the car: two conversations with Grok, a profile, a family.
    fn drive() -> (Tree, JournalPath) {
        let mut t = sharratt();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        let mut convs = Vec::new();
        for (i, (name, said)) in [
            ("2026-10-29-0800", "Drew a cat."),
            ("2026-10-30-0800", "Asked about permits."),
        ]
        .iter()
        .enumerate()
        {
            let c = apply(
                &mut t,
                &server,
                &Op::Create {
                    under: p("~/channels/grok"),
                    name: slug(name),
                    cue: "a drive".into(),
                    on_taken: OnTaken::Number,
                },
                now().plus(i as i64),
            )
            .unwrap()
            .created
            .unwrap();
            apply(
                &mut t,
                &server,
                &Op::Episode {
                    path: c.clone(),
                    text: said.to_string(),
                },
                now().plus(i as i64),
            )
            .unwrap();
            convs.push(c);
        }
        summarise(&mut t, &convs[0], "They drew a cat together.", None);
        summarise(
            &mut t,
            &p("~/channels/grok"),
            "In the car they draw and plan.",
            None,
        );
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("~"),
                text: "Prefers short answers.".into(),
            },
            now(),
        )
        .unwrap();
        summarise(
            &mut t,
            &p("~"),
            "John builds an inference engine and prefers short answers.",
            Some("Prefers short answers."),
        );
        apply(
            &mut t,
            &john(),
            &Op::Note {
                path: p("@sharratt"),
                text: "Building an extension.".into(),
            },
            now(),
        )
        .unwrap();
        summarise(
            &mut t,
            &p("@sharratt"),
            "The Sharratts are extending the house.",
            Some("Extending the house."),
        );
        (t, convs[1].clone())
    }

    #[test]
    fn a_voice_frame_is_digests_and_the_last_conversation() {
        let (t, here) = drive();
        let f = frame(&t, &slug("john"), Some(&here), &[], Budget::Voice, now());
        assert_eq!(
            f,
            format!(
                "{NOT_INSTRUCTIONS}\nYou: Prefers short answers.\nYour family: Extending the house.\n\
                 With this channel: In the car they draw and plan.\nLast time: They drew a cat together."
            )
        );
    }

    #[test]
    fn a_chat_frame_is_nearest_first() {
        let (t, here) = drive();
        let f = frame(&t, &slug("john"), Some(&here), &[], Budget::Chat, now());
        let order = [
            "This conversation, ~/channels/grok/2026-10-30-0800:",
            "Asked about permits.",
            "~/channels/grok: In the car they draw and plan.",
            "You (~): John builds an inference engine",
            "  - ~/channels — ",
            "Your family @sharratt (@sharratt): The Sharratts are extending the house.",
            "  - bots, projects, threads, topics: nothing written yet",
        ];
        let mut from = 0;
        for want in order {
            let at = f[from..]
                .find(want)
                .unwrap_or_else(|| panic!("{want:?} after {from} in\n{f}"));
            from += at + want.len();
        }
        assert!(f.chars().count() <= Budget::Chat.chars());
    }

    #[test]
    fn a_frame_never_exceeds_its_budget_and_ends_on_a_cut() {
        let (mut t, here) = drive();
        let long = "word ".repeat(300);
        for _ in 0..20 {
            apply(
                &mut t,
                &Actor::Server {
                    person: slug("john"),
                    channel: Channel::Grok,
                },
                &Op::Episode {
                    path: here.clone(),
                    text: long.clone(),
                },
                now(),
            )
            .unwrap();
        }
        for budget in [Budget::Voice, Budget::Chat, Budget::Worker] {
            let f = frame(&t, &slug("john"), Some(&here), &[], budget, now());
            assert!(
                f.chars().count() <= budget.chars(),
                "{budget:?}: {}",
                f.chars().count()
            );
        }
        // A group block too big for what is left is cut, and ends the frame.
        let mut fl = Filler::new(40);
        fl.push("0123456789");
        fl.push(&"x".repeat(100));
        fl.push("never");
        assert!(fl.out.ends_with('…') && !fl.out.contains("never"));
        assert!(fl.out.chars().count() <= 40);
    }

    #[test]
    fn a_person_alone_has_a_frame_with_no_group() {
        let t = sharratt();
        let f = frame(&t, &slug("stranger"), None, &[], Budget::Chat, now());
        assert!(f.contains("You (~): nothing summarised yet."));
        assert!(!f.contains("@sharratt"));
    }

    /// **Only framed groups open a conversation**: the default, and those
    /// `framed` names; another group is read when it is opened.
    #[test]
    fn a_group_opens_the_frame_only_when_framed() {
        let (mut t, here) = drive();
        apply(
            &mut t,
            &john(),
            &Op::CreateGroup {
                group: slug("battle-cities"),
                name: "Battle Cities".into(),
                kind: Kind::World,
            },
            now(),
        )
        .unwrap();
        let chat = |t: &Tree| frame(t, &slug("john"), Some(&here), &[], Budget::Chat, now());
        let unframed = chat(&t);
        assert!(
            !unframed.contains("Your world @battle-cities"),
            "{unframed}"
        );
        assert!(
            unframed.contains("  - @battle-cities — Battle Cities, a world; opened when needed"),
            "known, though not opened: {unframed}"
        );
        t.persons
            .get_mut(&slug("john"))
            .unwrap()
            .file
            .as_mut()
            .unwrap()
            .framed = vec![slug("battle-cities")];
        let f = chat(&t);
        let family = f.find("Your family @sharratt").unwrap();
        let world = f
            .find("Your world @battle-cities (@battle-cities): nothing summarised yet.")
            .unwrap_or_else(|| panic!("{f}"));
        assert!(family < world, "the default first");
        // What a group is for heads its block.
        apply(
            &mut t,
            &john(),
            &Op::DescribeGroup {
                group: slug("battle-cities"),
                about: "The game's world and its bots.".into(),
            },
            now(),
        )
        .unwrap();
        let f = chat(&t);
        assert!(
            f.contains(
                "Your world @battle-cities — The game's world and its bots. (@battle-cities):"
            ),
            "{f}"
        );
    }

    /// **A bot's frame is its groups and nothing of anyone's own**.
    #[test]
    fn a_bots_frame_is_its_groups_alone() {
        let (t, _) = drive();
        let f = bot_frame(&t, &slug("c3f1a07e"), Budget::Chat, now());
        assert!(f.starts_with(&format!(
            "{NOT_INSTRUCTIONS}\nYour family @sharratt (@sharratt): The Sharratts are extending the house."
        )));
        assert!(!f.contains("~"), "no one's own folder: {f}");
        assert_eq!(
            bot_frame(&t, &slug("b9"), Budget::Chat, now()),
            NOT_INSTRUCTIONS
        );
    }

    #[test]
    fn a_workers_frame_adds_its_projects() {
        let (t, _) = drive();
        let f = frame(
            &t,
            &slug("john"),
            Some(&p("threads")),
            &[p("projects")],
            Budget::Worker,
            now(),
        );
        assert!(f.ends_with("@sharratt/projects: no summary yet"), "{f}");
    }
}
