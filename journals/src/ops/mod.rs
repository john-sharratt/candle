//! Every change to the tree, one module each, applied in memory and checked
//! against the rules before anything is written. An operation answers what
//! it touched, so the store writes exactly those journals' files and the
//! commit says what it was.
//!
//! An operation is checked against who asks ([`permit`]) and then applied; a
//! refused one changes nothing.

pub mod accept;
pub mod append;
pub mod close;
pub mod create;
pub mod edit;
pub mod family;
pub mod fold;
pub mod forget;
pub mod guard;
pub mod merge;
pub mod period_fold;
pub mod permit;
pub mod place;
pub mod remap;
pub mod share;
pub mod summarize;
pub mod wire;

use crate::actor::Actor;
use crate::error::Result;
use crate::files::{AccountKey, Email, Fingerprint, Role, Sentence};
use crate::line::{LineId, Writer};
use crate::path::{Address, JournalPath, Owner};
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
    Note {
        path: JournalPath,
        text: String,
    },
    Episode {
        path: JournalPath,
        text: String,
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
    Close {
        path: JournalPath,
    },
    Fold {
        path: JournalPath,
    },
    Unfold {
        path: JournalPath,
    },
    Merge {
        from: JournalPath,
        into: JournalPath,
    },
    Summarize {
        path: JournalPath,
        over: Fingerprint,
        sentences: Vec<Sentence>,
        digest: Option<String>,
        cue: Option<String>,
    },
    MarkStale {
        path: JournalPath,
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
    Share {
        from: JournalPath,
        into: JournalPath,
        name: Option<Slug>,
    },
    AddMember {
        family: Slug,
        person: Slug,
        name: Option<String>,
        emails: Vec<Email>,
        role: Role,
    },
    RemoveMember {
        family: Slug,
        person: Slug,
    },
    Bind {
        person: Slug,
        email: Email,
        key: AccountKey,
    },
}

/// What an operation changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// The commit's subject line.
    pub subject: String,
    /// Journals whose own files changed.
    pub touched: Vec<JournalPath>,
    /// Journals removed, with everything below them.
    pub removed: Vec<JournalPath>,
    /// Owners whose `person.yaml` or `family.yaml` changed.
    pub owners: Vec<Owner>,
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
        Op::Note { path, text } => append::note(tree, actor, path, text, now, keep),
        Op::Episode { path, text } => append::episode(tree, actor, path, text, now, keep),
        Op::Edit { address, text } => edit::edit(tree, address, text),
        Op::Forget { path, which } => forget::forget(tree, path, which),
        Op::Accept { address } => accept::accept(tree, address),
        Op::Create {
            under,
            name,
            cue,
            on_taken,
        } => create::create(tree, under, name, cue, *on_taken, now),
        Op::Close { path } => close::close(tree, path, now),
        Op::Fold { path } => fold::fold(tree, path, now),
        Op::Unfold { path } => fold::unfold(tree, path),
        Op::Merge { from, into } => merge::merge(tree, from, into),
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
        Op::MarkStale { path } => summarize::mark_stale(tree, path),
        Op::FoldPeriod {
            path,
            period,
            ids,
            over,
            text,
        } => period_fold::fold_period(tree, path, *period, ids, over, text, now, keep),
        Op::Share { from, into, name } => share::share(tree, from, into, name.as_ref(), now),
        Op::AddMember {
            family,
            person,
            name,
            emails,
            role,
        } => family::add_member(tree, family, person, name.as_deref(), emails, *role, now),
        Op::RemoveMember { family, person } => family::remove_member(tree, family, person),
        Op::Bind { person, email, key } => family::bind(tree, person, email, key),
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
