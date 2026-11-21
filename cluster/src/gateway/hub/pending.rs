//! The state that holds a node's replacement session until its handshake
//! succeeds or fails.
//!
//! While a node's new session is handshaking, its standing session is marked
//! replaced, so its reports are held back rather than applied, and the relays
//! for the node are kept so the new socket's table can be brought up to date.
//! If the handshake fails, the replacement is undone, the standing session
//! reports again, and the held reports apply under the same guard. If it
//! succeeds, the held reports are dropped. A guard removes the pending
//! admission when its serving task ends before taking it, so a dropped task
//! cannot leave the node with a pending that nothing will clear.

use super::*;

impl Pending {
    /// Keep `frame`, a relay that brings the session's table up to date.
    pub(super) fn relay(&mut self, frame: Frame) {
        if self.overflowed {
            return;
        }
        if self.queue.len() >= PENDING_CAP {
            self.overflowed = true;
            tracing::warn!(
                session = self.session,
                "more than {PENDING_CAP} relays arrived during a handshake: its admission fails"
            );
            return;
        }
        self.queue.push(frame);
    }
}

/// What `run` finds of its session's [`Pending`].
pub(in super::super) enum Taken {
    /// Its own: the frames to hand the new socket.
    Ready(Vec<Frame>),
    /// Too much arrived during the handshake: the session goes, and the
    /// state reports held back have been applied to the node's standing
    /// session; what follows from them is left to do.
    Overflowed(Vec<Applied>),
}

/// `report`, from the session `replaced` marks, under the state's guard: held
/// back in the node's [`Pending`] while the replacement can still be undone and
/// this is the node's standing session, and otherwise dropped.
pub(super) fn hold_back(
    s: &mut State,
    app: &str,
    node: u64,
    replaced: &AtomicBool,
    report: Report,
) {
    let key = (app.to_string(), node);
    let standing = s
        .sockets
        .get(&key)
        .is_some_and(|x| ptr::eq(Arc::as_ptr(&x.replaced), replaced));
    if let Some(p) = s.pending.get_mut(&key).filter(|_| standing) {
        p.held.keep(report);
    }
}

/// Apply `held`, what `node`'s standing session reported while a replacement
/// of it was pending, under the guard that undid the replacement: before any
/// report it sends after, so nothing it said lands out of order. A standing
/// session that has since left takes what it reported with it: its node is no
/// member here, and naming it a key's holder or a build's would send work to
/// a socket that is gone. What is held was checked and decoded on arrival
/// ([`check_report`]), so it applies.
pub(super) fn apply_held(s: &mut State, app: &str, node: u64, held: Held) -> Vec<Applied> {
    if !s.sockets.contains_key(&(app.to_string(), node)) {
        return Vec::new();
    }
    held.into_reports()
        .into_iter()
        .map(|report| apply_report(s, app, node, report))
        .collect()
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let applied = {
            let mut s = self.hub.state();
            undo_pending(&mut s, &self.app, self.node, &self.admitted)
        };
        self.hub.follow(&self.app, self.node, applied);
    }
}

/// The node's pending admission, which must be `session`'s: only the session
/// that `decide` admitted removes it, through its `run` or its guard. `None`,
/// with the invariant's breach logged and the pending left as it was, when it
/// is not — the caller decides how loudly to fail: a guard dropping may be
/// unwinding a panic already, and a second would abort the whole gateway.
pub(super) fn own_pending(s: &mut State, app: &str, node: u64, session: u64) -> Option<Pending> {
    let key = (app.to_string(), node);
    match s.pending.get(&key).map(|p| p.session) {
        Some(owner) if owner == session => s.pending.remove(&key),
        owner => {
            tracing::error!(
                app,
                node = %hex(node),
                session,
                ?owner,
                "a node's pending admission is not this session's: an admission's pending \
                 stands until its own run or guard removes it"
            );
            None
        }
    }
}

/// Undo the replacement [`Hub::decide`] began for `admitted`, whose handshake
/// failed before its socket was in place: the node's admitted session is the
/// one it replaced again (`Admission::undo`), so a group of one keeps its
/// model lock; the relays kept for it are dropped; the node's standing
/// session reports again; and what it reported meanwhile is applied under
/// this same guard ([`apply_held`]). What follows from that is returned, to
/// do. A pending not `admitted`'s is left alone ([`own_pending`]).
pub(in super::super) fn undo_pending(
    s: &mut State,
    app: &str,
    node: u64,
    admitted: &Admitted,
) -> Vec<Applied> {
    let Some(p) = own_pending(s, app, node, admitted.session) else {
        return Vec::new();
    };
    let mut applied = restore_standing(s, app, node, admitted);
    applied.extend(apply_held(s, app, node, p.held));
    applied
}

/// `node`'s standing session, the one `admitted` was to replace, stands again:
/// admission names it the node's session again, and it reports again — only
/// when the node's socket here is that session's; one that left meanwhile has
/// nothing to restore. When that leaves the app with no session, its model
/// lock clears, and what follows says so.
pub(super) fn restore_standing(
    s: &mut State,
    app: &str,
    node: u64,
    admitted: &Admitted,
) -> Vec<Applied> {
    let socket = s.sockets.get(&(app.to_string(), node));
    let standing = socket
        .map(|x| x.session)
        .filter(|session| Some(*session) == admitted.replaces);
    if let Some(x) = socket.filter(|_| standing.is_some()) {
        x.replaced.store(false, Ordering::SeqCst);
    }
    if !s.admission.undo(app, node, admitted, standing) {
        return Vec::new();
    }
    vec![Applied {
        notes: vec![(hex(node), "model lock cleared: the group is empty".into())],
        ..Applied::default()
    }]
}

/// The [`Pending`] `run` puts its session's socket in place with. Overflowed,
/// its replacement is undone as a failed handshake's is. The caller has
/// disarmed its guard: a pending not its own fails it loudly, and only it.
pub(in super::super) fn take_pending(
    s: &mut State,
    app: &str,
    node: u64,
    admitted: &Admitted,
) -> Taken {
    let p = own_pending(s, app, node, admitted.session)
        .expect("an admission's pending stands until its own run or guard removes it");
    if p.overflowed {
        let mut applied = restore_standing(s, app, node, admitted);
        applied.extend(apply_held(s, app, node, p.held));
        return Taken::Overflowed(applied);
    }
    Taken::Ready(p.queue)
}
