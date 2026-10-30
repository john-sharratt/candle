//! A conversation's frame (§6.4–§6.6): built afresh at every call from the
//! conversation's focus — the group's master, who it could be talking to,
//! the conversation itself, the speaker's journal, the journals it is
//! about, the conversation before it on the same channel, and the cues of
//! what lies below — within its channel's budget. Grok's is spoken, and
//! opens with how to work. A bot's frame is what its group grants it; a
//! guest's is `@guest`.

use crate::actor::grants_of_bot;
use crate::files::Focus;
use crate::path::{JournalPath, Place};
use crate::scope::{conversation_channel, Channel, Section};
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{Journal, Tree};
use crate::view::index::{index, refs_lines, written};
use crate::view::open::focus_line;
use crate::view::who::{name_of, render_line};
use crate::view::NOT_INSTRUCTIONS;

/// How much a channel is given (§6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Budget {
    /// Spoken: Grok's primed first call and every call after, all from
    /// memory.
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

/// How to work, in a few spoken-length lines: what a voice frame opens with,
/// never cut (§6.6).
pub const VOICE_SPINE: &str = "This is Zen's memory. Work out who is speaking from what they say, \
and ask if nothing says; then say it with journal_focus. Keep what you learn on the journal it is \
about with journal_note, and this drive's open tasks on here.";

/// The newest lines a focused journal shows in a frame.
const FOCUSED_LINES: usize = 5;

/// Blocks added in priority order until the budget is spent.
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

    fn need(&self, block: &str) -> usize {
        block.chars().count() + usize::from(!self.out.is_empty())
    }

    fn put(&mut self, block: &str) {
        let need = self.need(block);
        if !self.out.is_empty() {
            self.out.push('\n');
        }
        self.out.push_str(block);
        self.left -= need;
    }

    /// `block`, or as much of it as fits, cut and marked: the block that does
    /// not fit is the last, and nothing after it is added.
    fn push(&mut self, block: &str) {
        if self.full || block.is_empty() {
            return;
        }
        if self.need(block) <= self.left {
            self.put(block);
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

    /// `block` whole, or nothing: what follows still has its chance. A voice
    /// frame is built so, every block a sentence that is said or not.
    fn push_whole(&mut self, block: &str) {
        if !self.full && !block.is_empty() && self.need(block) <= self.left {
            self.put(block);
        }
    }

    /// `header`, then as many of `lines` as fit whole, in order; nothing
    /// when not even the first fits. A line is never cut, so a path it names
    /// is whole, and what does not fit leaves the rest of the budget to what
    /// comes after.
    fn push_lines(&mut self, header: &str, lines: &[String]) {
        let Some(first) = lines.first() else { return };
        let sep = usize::from(!self.out.is_empty());
        let head = header.chars().count() + sep;
        if self.full || head + 1 + first.chars().count() > self.left {
            return;
        }
        if sep == 1 {
            self.out.push('\n');
        }
        self.out.push_str(header);
        self.left -= head;
        for l in lines {
            let n = l.chars().count() + 1;
            if n > self.left {
                break;
            }
            self.out.push('\n');
            self.out.push_str(l);
            self.left -= n;
        }
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

/// A cue as a voice says it: without its `open for`.
fn spoken_cue(cue: &str) -> &str {
    cue.strip_prefix("open for ").unwrap_or(cue)
}

/// How a frame names a group to its reader, with what `group.yaml` says it
/// is for.
fn group_label(tree: &Tree, group: &Slug) -> String {
    match tree.group_file(group) {
        Some(f) => match &f.about {
            Some(about) => format!("{} (@{group}) — {about}", f.name),
            None => format!("{} (@{group})", f.name),
        },
        None => format!("@{group}"),
    }
}

/// What a voice calls a group: its name, as `group.yaml` gives it.
fn group_name(tree: &Tree, group: &Slug) -> String {
    tree.group_file(group)
        .map_or_else(|| format!("@{group}"), |f| f.name.clone())
}

/// The people of `group`, each `people/<p>` journal with its cue, its
/// members first and said to be: who a conversation could be talking to.
fn people_lines(tree: &Tree, group: &Slug) -> Vec<String> {
    let people = JournalPath::of_section(group, Section::People);
    let Some(j) = tree.get(&people) else {
        return Vec::new();
    };
    let member = |p: &Slug| {
        tree.group_file(group)
            .is_some_and(|f| f.member(p).is_some())
    };
    let mut lines: Vec<(bool, String)> = written(j)
        .into_iter()
        .filter(|(_, c)| c.problems.is_empty())
        .map(|(n, c)| {
            let is_member = member(n);
            let cue = c.cue().unwrap_or_default();
            let tag = if is_member { ", a member" } else { "" };
            (is_member, format!("  - people/{n}{tag} — {cue}"))
        })
        .collect();
    lines.sort_by_key(|(m, _)| !m);
    lines.into_iter().map(|(_, l)| l).collect()
}

/// The conversation before `here` on the same channel that has a summary:
/// where a conversation on that client picks up.
fn last_time(tree: &Tree, here: &JournalPath) -> Option<(JournalPath, String)> {
    let channel = here.last().and_then(|n| conversation_channel(n.as_str()))?;
    let conversations = JournalPath::of_section(here.group(), Section::Conversations);
    let j = tree.get(&conversations)?;
    j.children
        .iter()
        .filter(|(n, _)| Some(*n) != here.last())
        .filter(|(n, _)| conversation_channel(n.as_str()) == Some(channel))
        .filter_map(|(n, c)| summary_of(c).map(|s| (n, c, s)))
        .max_by_key(|(_, c, _)| c.meta.as_ref().map(|m| m.created))
        .and_then(|(n, _, s)| Some((conversations.child(n.clone()).ok()?, s)))
}

fn focus_of<'a>(tree: &'a Tree, at: Option<&JournalPath>) -> Option<&'a Focus> {
    tree.get(at?)?.meta.as_ref()?.focus.as_ref()
}

/// The speaker's journal, `people/<speaker>`, when the focus has one.
fn speaker_path(group: &Slug, f: &Focus) -> Option<JournalPath> {
    JournalPath::person(group, f.speaker.as_ref()?).ok()
}

/// A journal in focus: its summary, its index, its references and its
/// newest lines.
fn focused_block(tree: &Tree, path: &JournalPath, j: &Journal, label: &str, now: Stamp) -> String {
    let mut block = vec![match summary_of(j) {
        Some(s) => format!("{label} {path}: {s}"),
        None => format!("{label} {path}: nothing summarised yet."),
    }];
    block.extend(index(path, j, now).into_iter().map(|l| format!("  - {l}")));
    block.extend(refs_lines(j).into_iter().map(|l| format!("  - {l}")));
    let from = j.window.len().saturating_sub(FOCUSED_LINES);
    block.extend(
        j.window[from..]
            .iter()
            .map(|l| format!("  {}", render_line(tree, l))),
    );
    block.join("\n")
}

/// The frame for a call in `group`, at `at` — a conversation, or a worker's
/// journal — with `also` (a worker's workspace topics) after the focus.
pub fn frame(
    tree: &Tree,
    group: &Slug,
    at: Option<&JournalPath>,
    also: &[JournalPath],
    budget: Budget,
    now: Stamp,
) -> String {
    let mut f = Filler::new(budget.chars());
    f.push(NOT_INSTRUCTIONS);
    if budget == Budget::Voice {
        voice(tree, &mut f, group, at);
        return f.out;
    }
    let master = JournalPath::root(group.clone());
    if let Some(root) = tree.get(&master) {
        let mut block = vec![match summary_of(root) {
            Some(s) => format!("{}: {s}", group_label(tree, group)),
            None => format!("{}: nothing summarised yet.", group_label(tree, group)),
        }];
        block.extend(
            index(&master, root, now)
                .into_iter()
                .map(|l| format!("  - {l}")),
        );
        block.extend(refs_lines(root).into_iter().map(|l| format!("  - {l}")));
        f.push(&block.join("\n"));
    }
    f.push_lines("Who it could be:", &people_lines(tree, group));
    let focus = focus_of(tree, at);
    if let Some(at) = at {
        if let Some(j) = tree.get(at).filter(|j| j.problems.is_empty()) {
            let conversation = j.place.is_conversation();
            let mut block = vec![if conversation {
                format!("This conversation, {at} (here):")
            } else {
                format!("This journal, {at}:")
            }];
            if let Some(fo) = focus {
                block.push(format!("  {}", focus_line(fo)));
            } else if conversation {
                block.push(
                    "  Focus: who is speaking and what it is about are not known yet.".into(),
                );
            }
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
            // A journal below the master that is no conversation is read
            // with what is above it.
            if !conversation {
                let mut up = at.parent();
                while let Some(p) = up.filter(|p| !p.is_root()) {
                    if let Some(s) = tree.get(&p).and_then(summary_of) {
                        f.push(&format!("{p}: {s}"));
                    }
                    up = p.parent();
                }
            }
        }
    }
    if let Some(fo) = focus {
        let speaking = speaker_path(group, fo);
        if let Some(sp) = &speaking {
            if let Some(j) = tree.get(sp) {
                f.push(&focused_block(tree, sp, j, "Speaking,", now));
            }
        }
        for a in &fo.about {
            // The speaker's own journal is shown once, as theirs.
            let about = a.at(group).ok().filter(|p| Some(p) != speaking.as_ref());
            if let Some((p, j)) = about.and_then(|p| tree.get(&p).map(|j| (p, j))) {
                f.push(&focused_block(tree, &p, j, "About:", now));
            }
        }
    }
    for p in also {
        if let Some(j) = tree.get(p) {
            f.push(&focused_block(tree, p, j, "Workspace:", now));
        }
    }
    if let Some((p, s)) = at.and_then(|a| last_time(tree, a)) {
        let channel = at
            .and_then(|a| a.last())
            .and_then(|n| conversation_channel(n.as_str()))
            .map_or("this channel", Channel::spoken);
        f.push(&format!("Last time on {channel}, {p}: {s}"));
    }
    // Last, in what is left: the cues of the journals the indexes name only
    // by name, so a reader picks one from the frame.
    if let Some(root) = tree.get(&master) {
        f.push_lines(&format!("Below {master}, by cue:"), &deeper(&master, root));
    }
    f.out
}

/// Spoken: how to work, never cut; then what the group is, who it could be,
/// where the last drive left off, the topics, and this conversation's path —
/// each said whole or left out.
fn voice(tree: &Tree, f: &mut Filler, group: &Slug, at: Option<&JournalPath>) {
    f.push(VOICE_SPINE);
    let master = JournalPath::root(group.clone());
    if let Some(d) = tree.get(&master).and_then(digest_of) {
        f.push_whole(&format!("{}: {d}", group_name(tree, group)));
    }
    let focus = focus_of(tree, at);
    if let Some(fo) = focus {
        if let Some(sp) = speaker_path(group, fo) {
            if let (Some(j), Some(s)) = (tree.get(&sp), &fo.speaker) {
                let said = digest_of(j).unwrap_or_else(|| "nothing kept about them yet".into());
                f.push_whole(&format!("Speaking: {}. {said}", name_of(tree, s)));
            }
        }
        for a in &fo.about {
            if let Some(s) = a
                .at(group)
                .ok()
                .and_then(|p| tree.get(&p))
                .and_then(summary_of)
            {
                f.push_whole(&format!("About {a}: {s}"));
            }
        }
    }
    let members: Vec<String> = tree
        .group_file(group)
        .map(|g| {
            g.persons()
                .map(|(p, _)| {
                    let cue = JournalPath::person(group, p)
                        .ok()
                        .and_then(|path| tree.get(&path))
                        .and_then(Journal::cue);
                    match cue {
                        Some(c) => format!("{} ({})", name_of(tree, p), spoken_cue(&c)),
                        None => name_of(tree, p),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    if !members.is_empty() && focus.and_then(|fo| fo.speaker.as_ref()).is_none() {
        f.push_whole(&format!("It could be: {}.", members.join("; ")));
    }
    if let Some((_, s)) = at.and_then(|a| last_time(tree, a)) {
        f.push_whole(&format!("Last time: {s}"));
    }
    let topics = JournalPath::of_section(group, Section::Topics);
    let names: Vec<String> = tree
        .get(&topics)
        .map(|j| written(j).into_iter().map(|(n, _)| n.to_string()).collect())
        .unwrap_or_default();
    if !names.is_empty() {
        f.push_whole(&format!("Topics: {}.", names.join(", ")));
    }
    if let Some(at) = at {
        f.push_whole(&format!("This conversation is {at}, written as here."));
    }
}

/// The frame for `bot`: each subtree its group grants it, its summary and
/// index.
pub fn bot_frame(tree: &Tree, bot: &Slug, budget: Budget, now: Stamp) -> String {
    let mut f = Filler::new(budget.chars());
    f.push(NOT_INSTRUCTIONS);
    for path in grants_of_bot(tree, bot) {
        if let Some(j) = tree.get(&path) {
            let mut block = vec![match summary_of(j) {
                Some(s) => format!("{path}: {s}"),
                None => format!("{path}: nothing summarised yet."),
            }];
            block.extend(index(&path, j, now).into_iter().map(|l| format!("  - {l}")));
            f.push(&block.join("\n"));
        }
    }
    f.out
}

/// Each named journal directly under `root`'s topics and people with its
/// cue, and under it each of its own, one line each — the journals an index
/// line names without their cues. Conversations, threads and bots are not
/// walked: a conversation is found by its channel and date, and threads and
/// bots are zend's own. A closed or unreadable journal is left out.
fn deeper(root: &JournalPath, j: &Journal) -> Vec<String> {
    let open = |c: &Journal| !c.is_closed() && c.problems.is_empty();
    let said = |name: &str, c: &Journal| match c.cue() {
        Some(cue) => format!("{name} — {cue}"),
        None => name.to_string(),
    };
    let mut lines = Vec::new();
    for (s, sec) in j.live() {
        if !matches!(sec.place, Place::Section(Section::Topics | Section::People)) {
            continue;
        }
        for (n, named) in written(sec).into_iter().filter(|(_, c)| open(c)) {
            let path = root
                .child(s.clone())
                .and_then(|p| p.child(n.clone()))
                .expect("a named journal is at a place");
            lines.push(format!("  - {}", said(&path.show(), named)));
            for (m, below) in written(named).into_iter().filter(|(_, c)| open(c)) {
                lines.push(format!("    - {}", said(m.as_str(), below)));
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actor::Actor;
    use crate::files::summary::{Cite, Sentence};
    use crate::ops::fixture::{john, now, p, zen};
    use crate::ops::{apply, OnTaken, Op};
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

    fn noted(t: &mut Tree, path: &str, text: &str) {
        apply(
            t,
            &john(),
            &Op::Note {
                path: p(path),
                text: text.into(),
                speaker: None,
            },
            now(),
        )
        .unwrap();
    }

    fn made(t: &mut Tree, actor: &Actor, under: &str, name: &str, cue: &str) -> JournalPath {
        apply(
            t,
            actor,
            &Op::Create {
                under: p(under),
                name: slug(name),
                cue: cue.into(),
                on_taken: OnTaken::Number,
            },
            now(),
        )
        .unwrap()
        .created
        .unwrap()
    }

    /// The household on the road: John and Yen's journals, the engine, a
    /// drive yesterday and one today.
    fn drive() -> (Tree, JournalPath) {
        let mut t = zen();
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        made(
            &mut t,
            &john(),
            "people",
            "john",
            "open for John: the engine, the house",
        );
        made(
            &mut t,
            &john(),
            "people",
            "yen",
            "open for Yen: the extension, the kids",
        );
        made(
            &mut t,
            &john(),
            "people",
            "builder",
            "open for the extension's builder",
        );
        made(
            &mut t,
            &john(),
            "topics",
            "candle",
            "open for the inference engine",
        );
        made(
            &mut t,
            &john(),
            "topics/candle",
            "cluster",
            "open for the machines it runs on",
        );
        noted(&mut t, "people/yen", "Yen prefers mornings.");
        let people_yen = p("people/yen");
        summarise(
            &mut t,
            &people_yen,
            "Yen runs the extension.",
            Some("Runs the extension."),
        );
        noted(&mut t, "@zen", "Zen is the household's memory.");
        summarise(
            &mut t,
            &p("@zen"),
            "Zen is John and Yen's memory.",
            Some("John and Yen's memory."),
        );
        let yesterday = made(
            &mut t,
            &server,
            "conversations",
            "2026-10-29-0800-grok",
            "open for a drive",
        );
        apply(
            &mut t,
            &server,
            &Op::Episode {
                path: yesterday.clone(),
                text: "image_create — a cat".into(),
                speaker: None,
            },
            now(),
        )
        .unwrap();
        summarise(&mut t, &yesterday, "They drew a cat together.", None);
        let today = made(
            &mut t,
            &server,
            "conversations",
            "2026-10-30-0800-grok",
            "open for a drive",
        );
        (t, today)
    }

    /// **A voice names the group the frame is for**, as its `group.yaml`
    /// calls it — never Zen for another group.
    #[test]
    fn a_voice_frame_names_its_own_group() {
        let (mut t, here) = drive();
        let g = t.groups.get_mut(&slug("zen")).unwrap();
        g.file.as_mut().unwrap().name = "The Sharratts".into();
        let f = frame(&t, &slug("zen"), Some(&here), &[], Budget::Voice, now());
        assert!(
            f.contains(&format!(
                "{VOICE_SPINE}\nThe Sharratts: John and Yen's memory.\n"
            )),
            "{f}"
        );
    }

    #[test]
    fn a_first_voice_frame_is_primed_and_spoken() {
        let (t, here) = drive();
        let f = frame(&t, &slug("zen"), Some(&here), &[], Budget::Voice, now());
        assert_eq!(
            f,
            format!(
                "{NOT_INSTRUCTIONS}\n{VOICE_SPINE}\nZen: John and Yen's memory.\n\
                 It could be: john (John: the engine, the house); yen (Yen: the extension, the kids); \
                 thanhlong; zen.\n\
                 Last time: They drew a cat together.\n\
                 Topics: candle.\n\
                 This conversation is @zen/conversations/2026-10-30-0800-grok, written as here."
            )
        );
        assert!(f.chars().count() <= Budget::Voice.chars());
    }

    /// **Once the focus has a speaker and a topic, the voice frame says
    /// them**, and stops listing who it could be.
    #[test]
    fn a_focused_voice_frame_says_who_and_what() {
        let (mut t, here) = drive();
        apply(
            &mut t,
            &john(),
            &Op::Focus {
                path: here.clone(),
                speaker: Some(slug("yen")),
                about: Some(vec![p("people/yen")]),
            },
            now(),
        )
        .unwrap();
        let f = frame(&t, &slug("zen"), Some(&here), &[], Budget::Voice, now());
        assert!(f.contains("\nSpeaking: yen. Runs the extension.\n"), "{f}");
        assert!(
            f.contains("\nAbout people/yen: Yen runs the extension.\n"),
            "{f}"
        );
        assert!(!f.contains("It could be"), "{f}");
    }

    /// **The spine is never cut**: a block that does not fit is left out
    /// whole, and what follows still has its chance.
    #[test]
    fn a_voice_frame_leaves_out_whole_blocks() {
        let (mut t, here) = drive();
        noted(&mut t, "@zen", "Zen keeps everything.");
        summarise(
            &mut t,
            &p("@zen"),
            "Zen is John and Yen's memory.",
            Some(&"d".repeat(380)),
        );
        noted(&mut t, "topics/candle", "x");
        // Within a topic's cap of 1,200, beyond what a voice frame has left.
        let long = "a ".repeat(599);
        let candle = p("topics/candle");
        summarise(&mut t, &candle, long.trim(), None);
        apply(
            &mut t,
            &john(),
            &Op::Focus {
                path: here.clone(),
                speaker: None,
                about: Some(vec![candle]),
            },
            now(),
        )
        .unwrap();
        let f = frame(&t, &slug("zen"), Some(&here), &[], Budget::Voice, now());
        assert!(f.contains(VOICE_SPINE));
        assert!(
            !f.contains("About topics/candle"),
            "too long to say: left out"
        );
        assert!(f.ends_with("written as here."), "{f}");
        assert!(f.chars().count() <= Budget::Voice.chars());
    }

    #[test]
    fn a_chat_frame_starts_at_the_master_and_follows_the_focus() {
        let (mut t, here) = drive();
        apply(
            &mut t,
            &john(),
            &Op::Focus {
                path: here.clone(),
                speaker: Some(slug("yen")),
                about: Some(vec![p("topics/candle")]),
            },
            now(),
        )
        .unwrap();
        let f = frame(&t, &slug("zen"), Some(&here), &[], Budget::Chat, now());
        let order = [
            "Zen (@zen): Zen is John and Yen's memory.",
            "Who it could be:",
            "  - people/john, a member — open for John: the engine, the house",
            "  - people/yen, a member — open for Yen: the extension, the kids",
            "  - people/builder — open for the extension's builder",
            "This conversation, @zen/conversations/2026-10-30-0800-grok (here):",
            "  Focus: yen is speaking; it is about topics/candle.",
            "Speaking, @zen/people/yen: Yen runs the extension.",
            "note, john via Claude Code: Yen prefers mornings.",
            "About: @zen/topics/candle: nothing summarised yet.",
            "Last time on Grok, @zen/conversations/2026-10-29-0800-grok: They drew a cat together.",
            "Below @zen, by cue:",
            "  - @zen/people/builder — open for the extension's builder",
            "  - @zen/topics/candle — open for the inference engine",
            "    - cluster — open for the machines it runs on",
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

    /// **A focus about the speaker's own journal shows it once**, as
    /// theirs.
    #[test]
    fn a_speakers_journal_in_the_focus_is_shown_once() {
        let (mut t, here) = drive();
        apply(
            &mut t,
            &john(),
            &Op::Focus {
                path: here.clone(),
                speaker: Some(slug("yen")),
                about: Some(vec![p("people/yen"), p("topics/candle")]),
            },
            now(),
        )
        .unwrap();
        let f = frame(&t, &slug("zen"), Some(&here), &[], Budget::Chat, now());
        assert_eq!(
            f.matches("@zen/people/yen: Yen runs the extension.")
                .count(),
            1,
            "{f}"
        );
        assert!(f.contains("Speaking, @zen/people/yen"), "{f}");
        assert!(f.contains("About: @zen/topics/candle"), "{f}");
    }

    #[test]
    fn a_first_chat_frame_says_the_focus_is_unknown() {
        let (t, here) = drive();
        let f = frame(&t, &slug("zen"), Some(&here), &[], Budget::Chat, now());
        assert!(
            f.contains("\n  Focus: who is speaking and what it is about are not known yet.\n"),
            "{f}"
        );
        assert!(!f.contains("Speaking,"));
    }

    #[test]
    fn a_frame_never_exceeds_its_budget_and_ends_on_a_cut() {
        let (mut t, here) = drive();
        let long = "word ".repeat(300);
        let server = Actor::Server {
            person: slug("john"),
            channel: Channel::Grok,
        };
        for _ in 0..20 {
            apply(
                &mut t,
                &server,
                &Op::Episode {
                    path: here.clone(),
                    text: long.clone(),
                    speaker: None,
                },
                now(),
            )
            .unwrap();
        }
        for budget in [Budget::Voice, Budget::Chat, Budget::Worker] {
            let f = frame(&t, &slug("zen"), Some(&here), &[], budget, now());
            assert!(
                f.chars().count() <= budget.chars(),
                "{budget:?}: {}",
                f.chars().count()
            );
        }
        let mut fl = Filler::new(40);
        fl.push("0123456789");
        fl.push(&"x".repeat(100));
        fl.push("never");
        assert!(fl.out.ends_with('…') && !fl.out.contains("never"));
        assert!(fl.out.chars().count() <= 40);
    }

    /// **Lines are added whole or not at all**, and what does not fit leaves
    /// the budget to what comes after.
    #[test]
    fn lines_fill_whole_and_leave_the_rest() {
        let mut fl = Filler::new(30);
        fl.push("0123456789");
        fl.push_lines("head:", &["  - a".into(), "  - far too long a line".into()]);
        assert_eq!(fl.out, "0123456789\nhead:\n  - a");
        fl.push_lines("next:", &["  - b".into()]);
        assert_eq!(
            fl.out, "0123456789\nhead:\n  - a",
            "no room for a header and a line"
        );
        fl.push_whole("y");
        fl.push_whole(&"z".repeat(50));
        fl.push("w");
        assert_eq!(fl.out, "0123456789\nhead:\n  - a\ny\nw");
    }

    /// **A bot's frame is its grants and nothing else**.
    #[test]
    fn a_bots_frame_is_its_grants_alone() {
        let mut t = zen();
        made(&mut t, &john(), "topics", "game", "open for the game");
        made(&mut t, &john(), "topics", "house", "open for the house");
        let f = bot_frame(&t, &slug("c3f1a07e"), Budget::Chat, now());
        assert_eq!(
            f,
            format!("{NOT_INSTRUCTIONS}\n@zen/topics/game: nothing summarised yet.")
        );
        assert_eq!(
            bot_frame(&t, &slug("b9"), Budget::Chat, now()),
            NOT_INSTRUCTIONS
        );
    }

    #[test]
    fn a_workers_frame_reads_up_its_lineage_and_adds_its_topics() {
        let (mut t, _) = drive();
        let thread = made(
            &mut t,
            &Actor::Zend,
            "threads",
            "user-context",
            "open for the thread",
        );
        let worker = made(
            &mut t,
            &Actor::Zend,
            "threads/user-context",
            "implement",
            "open for a worker",
        );
        noted_zend(&mut t, &thread, "Thread began.");
        summarise(&mut t, &thread, "The user-context thread.", None);
        let f = frame(
            &t,
            &slug("zen"),
            Some(&worker),
            &[p("topics/candle")],
            Budget::Worker,
            now(),
        );
        let order = [
            "This journal, @zen/threads/user-context/implement:",
            "@zen/threads/user-context: The user-context thread.",
            "Workspace: @zen/topics/candle: nothing summarised yet.",
        ];
        let mut from = 0;
        for want in order {
            let at = f[from..]
                .find(want)
                .unwrap_or_else(|| panic!("{want:?} after {from} in\n{f}"));
            from += at + want.len();
        }
    }

    fn noted_zend(t: &mut Tree, path: &JournalPath, text: &str) {
        apply(
            t,
            &Actor::Zend,
            &Op::Note {
                path: path.clone(),
                text: text.into(),
                speaker: None,
            },
            now(),
        )
        .unwrap();
    }
}
