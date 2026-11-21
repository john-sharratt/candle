//! The lock that serialises checking and beginning a worker's turn, so two calls
//! cannot take one name or answer one question.
//!
//! The lock is held while a worker's state is read and its turn begun, and by
//! archiving, so no turn begins on a worker between its being read and its record
//! saying it is archived. It is held only across reads, claims and record writes,
//! never across a distillation or a turn. A lock poisoned by a panic is taken as it
//! is, because every record written under it is whole on its own.

use super::*;

/// Held while a worker's state is checked and its turn begun — naming a new
/// worker, answering one — so two calls at once (Claude issuing them in
/// parallel) cannot both take one name, or both answer one question, and run
/// two turns on one conversation.
pub(super) static BEGIN: Mutex<()> = Mutex::new(());

/// Hold [`BEGIN`] — what archiving takes too, so no turn begins on a worker
/// between its being read and its record saying archived
/// (`super::archiving`). Held only across reads, claims and record writes,
/// never a distil or a turn; every record written under it is whole on its
/// own, so a lock poisoned by a panic is taken as it is rather than wedging
/// every later task.
pub fn holding() -> MutexGuard<'static, ()> {
    BEGIN.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub(in super::super) fn loading() -> String {
    "the model is still loading; try again when `cluster_status` says it is ready".into()
}
