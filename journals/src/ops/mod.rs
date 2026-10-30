//! Every change to the tree, one module each, applied in memory and checked
//! against the rules before anything is written. An operation answers what
//! it touched, so the store writes exactly those journals' files and the
//! commit says what it was.
//!
//! An operation is checked against who asks ([`permit`]) and then applied; a
//! refused one changes nothing.
//!
//! **The operations only add to the tree's structure** (§2.6): a line, a
//! sub-journal, a reference, a focus. Renaming, moving, merging and unfolding
//! journals is a restructuring — a coding change to the journals repository —
//! and the service folds only what its upkeep folds (§2.3, §4).

pub mod accept;
pub mod append;
pub mod close;
pub mod create;
pub mod edit;
pub mod focus;
pub mod fold;
pub mod forget;
pub mod group;
pub mod guard;
pub mod link;
pub mod period_fold;
pub mod permit;
pub mod remap;
pub mod summarize;
pub mod wire;

use crate::actor::Actor;
use crate::error::{Error, Result};
use crate::files::{AccountKey, Email, Fingerprint, Role, Sentence, Who};
use crate::line::{LineId, Writer};
use crate::path::{Address, Below, JournalPath};
use crate::period::Period;
use crate::slug::Slug;
use crate::time::Stamp;
use crate::tree::{Journal, Tree};

/// Which line to forget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Which {
    Id(LineId),
    /// The one line whose text holds this phrase, ignoring case.
    Says(String),
}

/// What a name already taken among siblings does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnTaken {
    /// Take the next free number: conversations, workers.
    Number,
    /// Refuse: a topic or a project named twice is a mistake.
    Refuse,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// `speaker` is who the conversation's focus says is talking (§2.4).
    Note {
        path: JournalPath,
        text: String,
        speaker: Option<Slug>,
    },
    Episode {
        path: JournalPath,
        text: String,
        speaker: Option<Slug>,
    },
    Edit {
        address: Address,
        text: String,
    },
    Forget {
        path: JournalPath,
        which: Which,
    },
    Accept {
        address: Address,
    },
    Create {
        under: JournalPath,
        name: Slug,
        cue: String,
        on_taken: OnTaken,
    },
    /// A reference from `under` to the journal at `to`, in the same group
    /// (§2.5).
    Link {
        under: JournalPath,
        name: Slug,
        to: JournalPath,
        cue: String,
    },
    /// A conversation's focus: who is speaking, what it is about. A part
    /// left `None` keeps what the focus has.
    Focus {
        path: JournalPath,
        speaker: Option<Slug>,
        about: Option<Vec<JournalPath>>,
    },
    Close {
        path: JournalPath,
    },
    Summarize {
        path: JournalPath,
        over: Fingerprint,
        sentences: Vec<Sentence>,
        digest: Option<String>,
        cue: Option<String>,
    },
    /// `why` is the refusal that left the summary standing, which its commit
    /// keeps, so a stale summary's history says what to mend.
    MarkStale {
        path: JournalPath,
        why: String,
    },
    /// `over` is [`period_fold::lines_over`] of the lines the fold was
    /// written from: one edited meanwhile makes it a conflict.
    FoldPeriod {
        path: JournalPath,
        period: Period,
        ids: Vec<LineId>,
        over: Fingerprint,
        text: String,
    },
    /// A new group, its creator its first owner.
    CreateGroup {
        group: Slug,
        name: String,
        about: Option<String>,
    },
    /// What a group is for: its `about`, an owner's to write.
    DescribeGroup {
        group: Slug,
        about: String,
    },
    AddMember {
        group: Slug,
        person: Slug,
        name: Option<String>,
        emails: Vec<Email>,
        role: Role,
    },
    /// The actor's person's own choice: the group their conversations open
    /// in.
    ChooseDefault {
        group: Slug,
    },
    /// A bot, as a reader of the subtrees `reads` grants it.
    AddBot {
        group: Slug,
        bot: Slug,
        reads: Vec<Below>,
    },
    RemoveMember {
        group: Slug,
        who: Who,
    },
    Bind {
        person: Slug,
        email: Email,
        key: AccountKey,
    },
}

/// An owner file an operation changed: a `person.yaml` or a `group.yaml`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OwnerFile {
    Person(Slug),
    Group(Slug),
}

/// What an operation changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// The commit's subject line.
    pub subject: String,
    /// Journals whose own files changed.
    pub touched: Vec<JournalPath>,
    /// Owner files that changed.
    pub owners: Vec<OwnerFile>,
    /// The line written, edited or forgotten.
    pub line: Option<LineId>,
    /// The journal created.
    pub created: Option<JournalPath>,
    /// Journals folded to make room.
    pub folded: Vec<JournalPath>,
}

impl Applied {
    fn touching(subject: impl Into<String>, paths: Vec<JournalPath>) -> Applied {
        Applied {
            subject: subject.into(),
            touched: paths,
            ..Applied::default()
        }
    }
}

/// Apply `op` for `actor` at `now`: checked, then done; refused, nothing.
pub fn apply(tree: &mut Tree, actor: &Actor, op: &Op, now: Stamp) -> Result<Applied> {
    apply_keeping(tree, actor, op, now, None)
}

/// [`apply`], the line it writes taking the id `keep` when that is free in
/// its journal — as a line replayed onto a newer origin keeps the id it was
/// first written with.
pub fn apply_keeping(
    tree: &mut Tree,
    actor: &Actor,
    op: &Op,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    let (next, applied) = applied_to(tree, actor, op, now, keep)?;
    *tree = next;
    Ok(applied)
}

/// The tree `op` makes of `tree`, and what it changed, leaving `tree` as it
/// was: one copy of the tree, which is the one returned.
pub fn applied_to(
    tree: &Tree,
    actor: &Actor,
    op: &Op,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<(Tree, Applied)> {
    permit::check(tree, actor, op)?;
    let mut next = tree.clone();
    let applied = run(&mut next, actor, op, now, keep)?;
    Ok((next, applied))
}

fn run(
    tree: &mut Tree,
    actor: &Actor,
    op: &Op,
    now: Stamp,
    keep: Option<&LineId>,
) -> Result<Applied> {
    match op {
        Op::Note {
            path,
            text,
            speaker,
        } => append::note(tree, actor, path, speaker.as_ref(), text, now, keep),
        Op::Episode {
            path,
            text,
            speaker,
        } => append::episode(tree, actor, path, speaker.as_ref(), text, now, keep),
        Op::Edit { address, text } => edit::edit(tree, address, text),
        Op::Forget { path, which } => forget::forget(tree, path, which),
        Op::Accept { address } => accept::accept(tree, address),
        Op::Create {
            under,
            name,
            cue,
            on_taken,
        } => create::create(tree, under, name, cue, *on_taken, now),
        Op::Link {
            under,
            name,
            to,
            cue,
        } => link::link(tree, under, name, to, cue, now),
        Op::Focus {
            path,
            speaker,
            about,
        } => focus::focus(tree, path, speaker.as_ref(), about.as_deref()),
        Op::Close { path } => close::close(tree, path, now),
        Op::Summarize {
            path,
            over,
            sentences,
            digest,
            cue,
        } => summarize::summarize(
            tree,
            path,
            over,
            sentences,
            digest.as_deref(),
            cue.as_deref(),
            now,
        ),
        Op::MarkStale { path, why } => summarize::mark_stale(tree, path, why),
        Op::FoldPeriod {
            path,
            period,
            ids,
            over,
            text,
        } => period_fold::fold_period(tree, path, *period, ids, over, text, now, keep),
        Op::CreateGroup { group, name, about } => {
            let creator = actor
                .person()
                .ok_or_else(|| Error::forbidden("only a person makes a group"))?;
            group::create_group(tree, creator, group, name, about.as_deref(), now)
        }
        Op::DescribeGroup { group, about } => group::describe_group(tree, group, about),
        Op::AddMember {
            group,
            person,
            name,
            emails,
            role,
        } => group::add_member(tree, group, person, name.as_deref(), emails, *role, now),
        Op::ChooseDefault { group } => {
            let person = actor
                .person()
                .ok_or_else(|| Error::forbidden("only a person chooses their default group"))?;
            group::choose_default(tree, person, group)
        }
        Op::AddBot { group, bot, reads } => group::add_bot(tree, group, bot, reads, now),
        Op::RemoveMember { group, who } => group::remove_member(tree, group, who),
        Op::Bind { person, email, key } => group::bind(tree, person, email, key),
    }
}

/// An id for a new line in `j`, salted until it is free there.
pub(crate) fn mint(j: &Journal, at: Stamp, by: Writer, text: &str) -> LineId {
    (0u32..)
        .map(|salt| LineId::mint(at, by, text, salt))
        .find(|id| !j.has_id(id))
        .expect("an id is free within u32 salts")
}

/// `keep` when it is free in `j`, else a fresh id.
pub(crate) fn mint_keeping(
    j: &Journal,
    keep: Option<&LineId>,
    at: Stamp,
    by: Writer,
    text: &str,
) -> LineId {
    match keep {
        Some(id) if !j.has_id(id) => id.clone(),
        _ => mint(j, at, by, text),
    }
}

#[cfg(test)]
pub(crate) mod fixture;
