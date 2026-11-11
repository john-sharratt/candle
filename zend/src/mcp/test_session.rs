//! A zend session for the `/mcp` tools' tests: a real substrate, the memory
//! open on a real journal repository with its origin, and call contexts for
//! John — the repository's owner, an estate admin — in a conversation or
//! none.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use candle_conversation::projection::Conversation;
use journals::diaries::crypto::KdfParams;
use mcp_server::protocol::MODERN;
use mcp_server::refs::Ref;
use mcp_server::{Client, Session, Store};
use tempfile::TempDir;
use web::auth::Role;

use super::caller::{Caller, Person};
use super::diaries::Diaries;
use super::memory::tests::repo;
use super::state::RecordStore;
use super::tools::Ctx;
use super::workers::now_ms;
use crate::config::DaemonConfig;
use crate::session::ZendSession;
use crate::workspace::single_repo;

/// A session whose memory is open on a fresh journal repository, given to
/// `prepare` before the memory opens it; what must outlive the session, the
/// session, and the repository's working folder.
pub(crate) fn session(prepare: impl FnOnce(&Path)) -> (Vec<TempDir>, Arc<ZendSession>, PathBuf) {
    let work = TempDir::new().unwrap();
    let session = Arc::new(ZendSession::new(DaemonConfig::new(
        single_repo(work.path(), "project").unwrap(),
    )));
    session.hold_substrate(Conversation::new());
    // Passphrases stretched cheaply: the production cost has its own test
    // (`journals::diaries::crypto`).
    session.mcp().install_diaries(Diaries::with(KdfParams {
        m_kib: 64,
        t: 1,
        p: 1,
    }));
    let (journals, dir) = repo();
    prepare(&dir);
    let memory = session.mcp().memory();
    memory.start(dir.clone());
    let until = Instant::now() + Duration::from_secs(30);
    while memory.tree().is_err() {
        assert!(Instant::now() < until, "the memory never opened");
        thread::sleep(Duration::from_millis(20));
    }
    (vec![work, journals], session, dir)
}

/// John, an estate admin, as the gateway names him.
pub(crate) fn john() -> Caller {
    Caller {
        person: Person {
            provider: "google".into(),
            sub: "john".into(),
            email: "john@example.com".into(),
            name: "John".into(),
        },
        role: Role::Admin,
    }
}

/// A call by `caller` through Claude Code on `session`, in `conversation`.
pub(crate) fn ctx(session: &Arc<ZendSession>, caller: Caller, conversation: Option<Ref>) -> Ctx {
    let records = session.shared_substrate().unwrap();
    let store: Arc<dyn Store> =
        Arc::new(RecordStore::new(records, Arc::clone(session.mcp().used())));
    Ctx {
        mcp: Session::new(&caller.key(), Client::ClaudeCode, MODERN, now_ms()),
        mcp_id: "tests".into(),
        conversation,
        session: Arc::clone(session),
        caller,
        store,
        progress: None,
    }
}

/// A new conversation of `caller`'s, private or not, as `get_user_context`
/// would begin it.
pub(crate) fn conversation(session: &ZendSession, caller: &Caller, private: bool) -> (String, Ref) {
    session.mcp().refs().issue(&caller.key(), private, now_ms())
}
