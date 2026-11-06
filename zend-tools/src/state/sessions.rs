//! In-memory registry for all open network sessions across every protocol.
//!
//! [`SessionRegistry`] is the single data structure that owns every live
//! connection.  It has ten typed sub-maps (one per protocol/resource class),
//! each protected by an `RwLock` so parallel reads and serialised writes can
//! coexist across concurrent tool calls.
//!
//! # Session types
//!
//! | Struct | Protocol | Key dependency |
//! |--------|----------|----------------|
//! | [`SshEntry`] | SSH (ssh2 + TcpStream) | `ssh_key` or `ssh_password` credential |
//! | [`SshProcess`] | Async SSH command | parent `SshEntry` |
//! | [`TelnetEntry`] | Telnet (raw TCP) | optional `telnet_password` credential |
//! | [`HttpEntry`] | HTTP/HTTPS (reqwest) | optional `http_*` credential |
//! | [`TcpEntry`] | Raw TCP | none |
//! | [`UdpEntry`] | UDP socket | none |
//! | [`TlsEntry`] | TLS over TCP (native-tls) | optional `tls_client_cert` credential |
//! | [`SqlEntry`] | SQLite (rusqlite) | optional `sql_password` credential |
//! | [`RemoteFsEntry`] | SFTP over SSH (ssh2) | `ssh_key` or `remote_fs_password` credential |
//! | [`CodeEntry`] | Embedded JavaScript (boa) REPL | none |
//!
//! # Thread safety
//!
//! `SshConn` and `SqlConn` are not `Send` by the standard rules, so we use
//! `unsafe impl Send` guarded by `Mutex` at the call site.  Every access to
//! the underlying connection goes through `entry.lock().unwrap()`.
//!
//! # Resource limits
//!
//! The session tool implementations (not this registry itself) enforce a cap of
//! 5 active sessions per user per protocol; tools manage their own caps before
//! inserting.  The one exception is [`RemoteFsEntry`], which carries an optional
//! idle timeout that `get_remote_fs` enforces lazily — expiring the session on
//! access once it has been idle past the limit.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::net::{TcpStream, UdpSocket};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use native_tls::TlsStream;

extern crate rusqlite;

// ── Common metadata ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SessionMeta {
    pub session_id: String,
    /// The conversation that opened it (its context's lineage); empty for
    /// none. Archiving the conversation closes it
    /// ([`SessionRegistry::close_conversation`]).
    pub conversation: String,
    pub opened_at: String,
    pub last_activity: String,
    pub alive: bool,
}

// ── SSH ──────────────────────────────────────────────────────────────────────

pub struct SshConn {
    pub session: ssh2::Session,
    pub _stream: TcpStream,
}

// ssh2::Session is not Send by default, but we gate all access behind Mutex
unsafe impl Send for SshConn {}
unsafe impl Sync for SshConn {}

pub struct SshEntry {
    pub meta: SessionMeta,
    pub host: String,
    pub port: u16,
    pub credential_id: String,
    pub credential_name: String,
    pub cwd: String,
    pub conn: SshConn,
}

// ── SSH async process ────────────────────────────────────────────────────────

pub struct SshProcess {
    pub process_id: String,
    pub session_id: String,
    pub command: String,
    pub started_at: String,
    pub stdout_buf: Arc<Mutex<Vec<u8>>>,
    pub stderr_buf: Arc<Mutex<Vec<u8>>>,
    pub exit_code: Arc<Mutex<Option<i32>>>,
    pub running: Arc<Mutex<bool>>,
}

// ── Telnet ───────────────────────────────────────────────────────────────────

pub struct TelnetEntry {
    pub meta: SessionMeta,
    pub host: String,
    pub port: u16,
    pub prompt_pattern: String,
    pub stream: TcpStream,
}

// ── HTTP ─────────────────────────────────────────────────────────────────────

pub struct HttpEntry {
    pub meta: SessionMeta,
    pub base_url: Option<String>,
    pub credential_name: Option<String>,
    pub client: reqwest::blocking::Client,
}

// ── TCP ──────────────────────────────────────────────────────────────────────

pub struct TcpEntry {
    pub meta: SessionMeta,
    pub peer_addr: String,
    pub local_addr: String,
    pub stream: TcpStream,
}

// ── UDP ──────────────────────────────────────────────────────────────────────

pub struct UdpEntry {
    pub meta: SessionMeta,
    pub default_peer: String,
    pub local_addr: String,
    pub socket: UdpSocket,
}

// ── TLS ──────────────────────────────────────────────────────────────────────

pub struct TlsEntry {
    pub meta: SessionMeta,
    pub host: String,
    pub port: u16,
    pub local_addr: String,
    pub stream: TlsStream<TcpStream>,
}

// ── SQL ──────────────────────────────────────────────────────────────────────

pub enum SqlConn {
    Sqlite(rusqlite::Connection),
}
// rusqlite::Connection is Send (but not Sync); Mutex provides Sync
unsafe impl Send for SqlConn {}
unsafe impl Sync for SqlConn {}

pub struct SqlEntry {
    pub meta: SessionMeta,
    pub dsn: String,
    pub driver: String, // "sqlite"
    pub conn: SqlConn,
}

// ── Remote FS ─────────────────────────────────────────────────────────────────

pub struct RemoteFsConn {
    pub ssh: SshConn, // reuse SshConn (ssh2::Session + TcpStream)
}
// Safety: access is always under Mutex
unsafe impl Send for RemoteFsConn {}
unsafe impl Sync for RemoteFsConn {}

pub struct RemoteFsEntry {
    pub meta: SessionMeta,
    pub uri: String,
    pub protocol: String, // "sftp"
    pub host: String,
    pub port: u16,
    pub credential_id: String,
    pub credential_name: String,
    pub remote_prefix: String, // leading path from URI (e.g. "/home/user")
    pub conn: RemoteFsConn,
    /// If set, the session lazily expires after this many seconds with no access.
    pub idle_timeout_secs: Option<u32>,
}

// ── Code ─────────────────────────────────────────────────────────────────────

/// A persistent JavaScript REPL session. There is no live VM here — the session
/// state is the accumulated source of every successful `code_session_exec`,
/// replayed into a fresh `boa_engine::Context` on each call. This keeps the
/// entry `Send` (a `boa` `Context` is not) so it can live in the shared
/// registry. See `tools::code`.
pub struct CodeEntry {
    pub meta: SessionMeta,
    pub language: String,
    /// The source of every successful exec, one entry per snippet, replayed in
    /// order to rebuild variable / function state before each new snippet.
    /// Kept apart so each replays under its own call budget, as it first ran.
    pub history: Vec<String>,
}

// ── Registry ─────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct SessionRegistry {
    pub ssh: RwLock<HashMap<String, Arc<Mutex<SshEntry>>>>,
    pub ssh_processes: RwLock<HashMap<String, Arc<Mutex<SshProcess>>>>,
    pub telnet: RwLock<HashMap<String, Arc<Mutex<TelnetEntry>>>>,
    pub http: RwLock<HashMap<String, Arc<Mutex<HttpEntry>>>>,
    pub tcp: RwLock<HashMap<String, Arc<Mutex<TcpEntry>>>>,
    pub udp: RwLock<HashMap<String, Arc<Mutex<UdpEntry>>>>,
    pub tls: RwLock<HashMap<String, Arc<Mutex<TlsEntry>>>>,
    pub sql: RwLock<HashMap<String, Arc<Mutex<SqlEntry>>>>,
    pub remote_fs: RwLock<HashMap<String, Arc<Mutex<RemoteFsEntry>>>>,
    pub code: RwLock<HashMap<String, Arc<Mutex<CodeEntry>>>>,
    /// Who opened each session, kept beside the maps so closing a
    /// conversation's sessions reads no entry's lock: a tool call may be
    /// holding one.
    owners: Mutex<Owners>,
}

/// Each session's conversation, and the conversations already closed.
#[derive(Default)]
struct Owners {
    /// Session and SSH-process id → the conversation that opened it.
    by_session: HashMap<String, String>,
    /// Conversations whose sessions have been closed: a round still in
    /// flight when its conversation was archived opens nothing more.
    closed: HashSet<String>,
}

/// Which kinds of session have at least one open.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenSessions {
    pub ssh: bool,
    pub ssh_processes: bool,
    pub telnet: bool,
    pub http: bool,
    pub tcp: bool,
    pub udp: bool,
    pub tls: bool,
    pub sql: bool,
    pub remote_fs: bool,
    pub code: bool,
}

impl SessionRegistry {
    /// Which kinds of session have at least one open, across every conversation.
    pub fn open(&self) -> OpenSessions {
        fn any<V>(map: &RwLock<HashMap<String, V>>) -> bool {
            !map.read().unwrap_or_else(|e| e.into_inner()).is_empty()
        }
        OpenSessions {
            ssh: any(&self.ssh),
            ssh_processes: any(&self.ssh_processes),
            telnet: any(&self.telnet),
            http: any(&self.http),
            tcp: any(&self.tcp),
            udp: any(&self.udp),
            tls: any(&self.tls),
            sql: any(&self.sql),
            remote_fs: any(&self.remote_fs),
            code: any(&self.code),
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    fn owners(&self) -> MutexGuard<'_, Owners> {
        self.owners.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Put `entry`, opened as `meta` says, in `map` — unless its conversation
    /// has already been closed, when it is dropped and its connection with
    /// it. The owner and the entry go in under one hold of the owners, so a
    /// close sees both or neither.
    fn admit<V>(&self, map: &RwLock<HashMap<String, Arc<Mutex<V>>>>, meta: &SessionMeta, entry: V) {
        let mut owners = self.owners();
        if !meta.conversation.is_empty() {
            if owners.closed.contains(&meta.conversation) {
                tracing::info!(
                    conversation = %meta.conversation,
                    "a session opened after its conversation closed is dropped"
                );
                return;
            }
            owners
                .by_session
                .insert(meta.session_id.clone(), meta.conversation.clone());
        }
        map.write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(meta.session_id.clone(), Arc::new(Mutex::new(entry)));
    }

    fn disown(&self, id: &str) {
        self.owners().by_session.remove(id);
    }

    /// Close every session `conversation` opened, of every kind, and the SSH
    /// processes running on them, answering how many sessions it closed. A
    /// session is dropped from its map here and its connection closes when
    /// the last holder lets go — at once, unless a tool call is using it,
    /// and then when that call ends. An empty `conversation` names none.
    pub fn close_conversation(&self, conversation: &str) -> usize {
        if conversation.is_empty() {
            return 0;
        }
        let ids: Vec<String> = {
            let mut owners = self.owners();
            owners.closed.insert(conversation.to_string());
            let ids: Vec<String> = owners
                .by_session
                .iter()
                .filter(|(_, c)| c.as_str() == conversation)
                .map(|(id, _)| id.clone())
                .collect();
            for id in &ids {
                owners.by_session.remove(id);
            }
            ids
        };
        fn take<V>(map: &RwLock<HashMap<String, V>>, id: &str) -> bool {
            map.write()
                .unwrap_or_else(|e| e.into_inner())
                .remove(id)
                .is_some()
        }
        let mut closed = 0;
        for id in &ids {
            // A process is owned like a session but is not one: it is counted
            // with the SSH session it runs on.
            take(&self.ssh_processes, id);
            let session = take(&self.ssh, id)
                | take(&self.telnet, id)
                | take(&self.http, id)
                | take(&self.tcp, id)
                | take(&self.udp, id)
                | take(&self.tls, id)
                | take(&self.sql, id)
                | take(&self.remote_fs, id)
                | take(&self.code, id);
            closed += usize::from(session);
        }
        closed
    }

    // SSH
    pub fn insert_ssh(&self, entry: SshEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.ssh, &meta, entry);
    }
    pub fn get_ssh(&self, id: &str) -> Option<Arc<Mutex<SshEntry>>> {
        self.ssh.read().unwrap().get(id).cloned()
    }
    pub fn remove_ssh(&self, id: &str) -> bool {
        self.disown(id);
        self.ssh.write().unwrap().remove(id).is_some()
    }
    pub fn list_ssh(&self) -> Vec<Arc<Mutex<SshEntry>>> {
        self.ssh.read().unwrap().values().cloned().collect()
    }

    // SSH processes
    pub fn insert_ssh_process(&self, proc: SshProcess) {
        let id = proc.process_id.clone();
        let mut owners = self.owners();
        if let Some(conversation) = owners.by_session.get(&proc.session_id).cloned() {
            owners.by_session.insert(id.clone(), conversation);
        }
        self.ssh_processes
            .write()
            .unwrap()
            .insert(id, Arc::new(Mutex::new(proc)));
    }
    pub fn get_ssh_process(&self, id: &str) -> Option<Arc<Mutex<SshProcess>>> {
        self.ssh_processes.read().unwrap().get(id).cloned()
    }
    pub fn remove_ssh_process(&self, id: &str) -> bool {
        self.disown(id);
        self.ssh_processes.write().unwrap().remove(id).is_some()
    }

    // Telnet
    pub fn insert_telnet(&self, entry: TelnetEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.telnet, &meta, entry);
    }
    pub fn get_telnet(&self, id: &str) -> Option<Arc<Mutex<TelnetEntry>>> {
        self.telnet.read().unwrap().get(id).cloned()
    }
    pub fn remove_telnet(&self, id: &str) -> bool {
        self.disown(id);
        self.telnet.write().unwrap().remove(id).is_some()
    }
    pub fn list_telnet(&self) -> Vec<Arc<Mutex<TelnetEntry>>> {
        self.telnet.read().unwrap().values().cloned().collect()
    }

    // HTTP
    pub fn insert_http(&self, entry: HttpEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.http, &meta, entry);
    }
    pub fn get_http(&self, id: &str) -> Option<Arc<Mutex<HttpEntry>>> {
        self.http.read().unwrap().get(id).cloned()
    }
    pub fn remove_http(&self, id: &str) -> bool {
        self.disown(id);
        self.http.write().unwrap().remove(id).is_some()
    }
    pub fn list_http(&self) -> Vec<Arc<Mutex<HttpEntry>>> {
        self.http.read().unwrap().values().cloned().collect()
    }

    // TCP
    pub fn insert_tcp(&self, entry: TcpEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.tcp, &meta, entry);
    }
    pub fn get_tcp(&self, id: &str) -> Option<Arc<Mutex<TcpEntry>>> {
        self.tcp.read().unwrap().get(id).cloned()
    }
    pub fn remove_tcp(&self, id: &str) -> bool {
        self.disown(id);
        self.tcp.write().unwrap().remove(id).is_some()
    }
    pub fn list_tcp(&self) -> Vec<Arc<Mutex<TcpEntry>>> {
        self.tcp.read().unwrap().values().cloned().collect()
    }

    // UDP
    pub fn insert_udp(&self, entry: UdpEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.udp, &meta, entry);
    }
    pub fn get_udp(&self, id: &str) -> Option<Arc<Mutex<UdpEntry>>> {
        self.udp.read().unwrap().get(id).cloned()
    }
    pub fn remove_udp(&self, id: &str) -> bool {
        self.disown(id);
        self.udp.write().unwrap().remove(id).is_some()
    }
    pub fn list_udp(&self) -> Vec<Arc<Mutex<UdpEntry>>> {
        self.udp.read().unwrap().values().cloned().collect()
    }

    // TLS
    pub fn insert_tls(&self, entry: TlsEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.tls, &meta, entry);
    }
    pub fn get_tls(&self, id: &str) -> Option<Arc<Mutex<TlsEntry>>> {
        self.tls.read().unwrap().get(id).cloned()
    }
    pub fn remove_tls(&self, id: &str) -> bool {
        self.disown(id);
        self.tls.write().unwrap().remove(id).is_some()
    }
    pub fn list_tls(&self) -> Vec<Arc<Mutex<TlsEntry>>> {
        self.tls.read().unwrap().values().cloned().collect()
    }

    // SQL
    pub fn insert_sql(&self, entry: SqlEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.sql, &meta, entry);
    }
    pub fn get_sql(&self, id: &str) -> Option<Arc<Mutex<SqlEntry>>> {
        self.sql.read().unwrap().get(id).cloned()
    }
    pub fn remove_sql(&self, id: &str) -> bool {
        self.disown(id);
        self.sql.write().unwrap().remove(id).is_some()
    }
    pub fn list_sql(&self) -> Vec<Arc<Mutex<SqlEntry>>> {
        self.sql.read().unwrap().values().cloned().collect()
    }

    // Remote FS
    pub fn insert_remote_fs(&self, entry: RemoteFsEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.remote_fs, &meta, entry);
    }
    pub fn get_remote_fs(&self, id: &str) -> Option<Arc<Mutex<RemoteFsEntry>>> {
        let arc = self.remote_fs.read().unwrap().get(id).cloned()?;
        // Lazily enforce the per-session idle timeout: expire the session if it
        // has gone untouched for longer than the limit, otherwise refresh its
        // activity timestamp so the window slides on each use.
        {
            let mut entry = arc.lock().unwrap();
            if let Some(timeout) = entry.idle_timeout_secs {
                let within_window = chrono::DateTime::parse_from_rfc3339(&entry.meta.last_activity)
                    .map(|t| {
                        (chrono::Utc::now() - t.with_timezone(&chrono::Utc)).num_seconds()
                            <= i64::from(timeout)
                    })
                    .unwrap_or(true);
                if !within_window {
                    drop(entry);
                    self.disown(id);
                    self.remote_fs.write().unwrap().remove(id);
                    return None;
                }
                entry.meta.last_activity = chrono::Utc::now().to_rfc3339();
            }
        }
        Some(arc)
    }
    pub fn remove_remote_fs(&self, id: &str) -> bool {
        self.disown(id);
        self.remote_fs.write().unwrap().remove(id).is_some()
    }
    pub fn list_remote_fs(&self) -> Vec<Arc<Mutex<RemoteFsEntry>>> {
        self.remote_fs.read().unwrap().values().cloned().collect()
    }

    // Code
    pub fn insert_code(&self, entry: CodeEntry) {
        let meta = entry.meta.clone();
        self.admit(&self.code, &meta, entry);
    }
    pub fn get_code(&self, id: &str) -> Option<Arc<Mutex<CodeEntry>>> {
        self.code.read().unwrap().get(id).cloned()
    }
    pub fn remove_code(&self, id: &str) -> bool {
        self.disown(id);
        self.code.write().unwrap().remove(id).is_some()
    }
    pub fn list_code(&self) -> Vec<Arc<Mutex<CodeEntry>>> {
        self.code.read().unwrap().values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::*;

    fn meta(id: &str, conversation: &str) -> SessionMeta {
        SessionMeta {
            session_id: id.to_string(),
            conversation: conversation.to_string(),
            opened_at: String::new(),
            last_activity: String::new(),
            alive: true,
        }
    }

    fn tcp(id: &str, conversation: &str, listener: &TcpListener) -> TcpEntry {
        TcpEntry {
            meta: meta(id, conversation),
            peer_addr: String::new(),
            local_addr: String::new(),
            stream: TcpStream::connect(listener.local_addr().unwrap()).unwrap(),
        }
    }

    fn http(id: &str, conversation: &str) -> HttpEntry {
        HttpEntry {
            meta: meta(id, conversation),
            base_url: None,
            credential_name: Some("yenbot".to_string()),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn code(id: &str, conversation: &str) -> CodeEntry {
        CodeEntry {
            meta: meta(id, conversation),
            language: "javascript".to_string(),
            history: Vec::new(),
        }
    }

    /// **Archiving a conversation closes what it opened, and only that.**
    #[test]
    fn closing_a_conversation_closes_its_sessions_and_no_others() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let reg = SessionRegistry::new();
        reg.insert_tcp(tcp("t-a", "a", &listener));
        reg.insert_http(http("h-a", "a"));
        reg.insert_code(code("c-a", "a"));
        reg.insert_tcp(tcp("t-b", "b", &listener));
        reg.insert_http(http("h-b", "b"));

        assert_eq!(reg.close_conversation("a"), 3);
        assert!(reg.get_tcp("t-a").is_none());
        assert!(reg.get_http("h-a").is_none());
        assert!(reg.get_code("c-a").is_none());
        assert!(reg.get_tcp("t-b").is_some());
        assert!(reg.get_http("h-b").is_some());

        assert_eq!(reg.close_conversation("a"), 0, "nothing is closed twice");
        assert_eq!(reg.close_conversation("b"), 2);
    }

    /// **A round still in flight when its conversation closed opens nothing
    /// more**: the session it opens is dropped, and another conversation's
    /// opens as ever.
    #[test]
    fn a_session_opened_after_its_conversation_closed_is_dropped() {
        let reg = SessionRegistry::new();
        assert_eq!(reg.close_conversation("a"), 0);
        reg.insert_http(http("late", "a"));
        reg.insert_code(code("late-code", "a"));
        reg.insert_http(http("other", "b"));
        assert!(reg.get_http("late").is_none());
        assert!(reg.get_code("late-code").is_none());
        assert!(reg.get_http("other").is_some());
        assert_eq!(reg.close_conversation("a"), 0);
    }

    /// **A session that belongs to no conversation is never swept.**
    #[test]
    fn an_unowned_session_is_closed_by_no_conversation() {
        let reg = SessionRegistry::new();
        reg.insert_http(http("h", ""));
        assert_eq!(reg.close_conversation(""), 0);
        assert!(reg.get_http("h").is_some());
    }

    /// **A session closed by hand leaves no owner behind**, so its id is not
    /// counted when the conversation is archived later.
    #[test]
    fn a_session_closed_by_hand_is_not_counted_again() {
        let reg = SessionRegistry::new();
        reg.insert_http(http("h1", "a"));
        reg.insert_http(http("h2", "a"));
        assert!(reg.remove_http("h1"));
        assert_eq!(reg.close_conversation("a"), 1);
    }

    /// **A process on an SSH session is the session's conversation's**, and
    /// goes when it does without being counted as a session.
    #[test]
    fn an_ssh_process_goes_with_its_conversation() {
        let reg = SessionRegistry::new();
        reg.owners()
            .by_session
            .insert("s".to_string(), "a".to_string());
        reg.insert_ssh_process(SshProcess {
            process_id: "p".to_string(),
            session_id: "s".to_string(),
            command: String::new(),
            started_at: String::new(),
            stdout_buf: Arc::default(),
            stderr_buf: Arc::default(),
            exit_code: Arc::default(),
            running: Arc::default(),
        });
        assert_eq!(reg.close_conversation("a"), 0);
        assert!(reg.get_ssh_process("p").is_none());
    }
}
