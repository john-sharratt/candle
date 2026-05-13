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
//! | [`CodeEntry`] | Python/Node subprocess | none |
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
//! 5 active sessions per user per protocol.  The registry does no eviction; tools
//! manage their own caps before inserting.

#![allow(dead_code)]

use std::collections::HashMap;
use std::net::{TcpStream, UdpSocket};
use std::sync::{Arc, Mutex, RwLock};

use native_tls::TlsStream;

extern crate rusqlite;

// ── Common metadata ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SessionMeta {
    pub session_id: String,
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
    pub protocol: String,       // "sftp"
    pub host: String,
    pub port: u16,
    pub credential_id: String,
    pub credential_name: String,
    pub remote_prefix: String,  // leading path from URI (e.g. "/home/user")
    pub conn: RemoteFsConn,
}

// ── Code ─────────────────────────────────────────────────────────────────────

pub struct CodeEntry {
    pub meta: SessionMeta,
    pub language: String,
    pub child: std::process::Child,
    pub stdin: std::process::ChildStdin,
    pub stdout_reader: std::io::BufReader<std::process::ChildStdout>,
    /// Temp script file that must stay alive until the process exits.
    pub _temp_script: Option<std::path::PathBuf>,
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
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    // SSH
    pub fn insert_ssh(&self, entry: SshEntry) {
        let id = entry.meta.session_id.clone();
        self.ssh.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_ssh(&self, id: &str) -> Option<Arc<Mutex<SshEntry>>> {
        self.ssh.read().unwrap().get(id).cloned()
    }
    pub fn remove_ssh(&self, id: &str) -> bool {
        self.ssh.write().unwrap().remove(id).is_some()
    }
    pub fn list_ssh(&self) -> Vec<Arc<Mutex<SshEntry>>> {
        self.ssh.read().unwrap().values().cloned().collect()
    }

    // SSH processes
    pub fn insert_ssh_process(&self, proc: SshProcess) {
        let id = proc.process_id.clone();
        self.ssh_processes.write().unwrap().insert(id, Arc::new(Mutex::new(proc)));
    }
    pub fn get_ssh_process(&self, id: &str) -> Option<Arc<Mutex<SshProcess>>> {
        self.ssh_processes.read().unwrap().get(id).cloned()
    }
    pub fn remove_ssh_process(&self, id: &str) -> bool {
        self.ssh_processes.write().unwrap().remove(id).is_some()
    }

    // Telnet
    pub fn insert_telnet(&self, entry: TelnetEntry) {
        let id = entry.meta.session_id.clone();
        self.telnet.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_telnet(&self, id: &str) -> Option<Arc<Mutex<TelnetEntry>>> {
        self.telnet.read().unwrap().get(id).cloned()
    }
    pub fn remove_telnet(&self, id: &str) -> bool {
        self.telnet.write().unwrap().remove(id).is_some()
    }
    pub fn list_telnet(&self) -> Vec<Arc<Mutex<TelnetEntry>>> {
        self.telnet.read().unwrap().values().cloned().collect()
    }

    // HTTP
    pub fn insert_http(&self, entry: HttpEntry) {
        let id = entry.meta.session_id.clone();
        self.http.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_http(&self, id: &str) -> Option<Arc<Mutex<HttpEntry>>> {
        self.http.read().unwrap().get(id).cloned()
    }
    pub fn remove_http(&self, id: &str) -> bool {
        self.http.write().unwrap().remove(id).is_some()
    }
    pub fn list_http(&self) -> Vec<Arc<Mutex<HttpEntry>>> {
        self.http.read().unwrap().values().cloned().collect()
    }

    // TCP
    pub fn insert_tcp(&self, entry: TcpEntry) {
        let id = entry.meta.session_id.clone();
        self.tcp.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_tcp(&self, id: &str) -> Option<Arc<Mutex<TcpEntry>>> {
        self.tcp.read().unwrap().get(id).cloned()
    }
    pub fn remove_tcp(&self, id: &str) -> bool {
        self.tcp.write().unwrap().remove(id).is_some()
    }
    pub fn list_tcp(&self) -> Vec<Arc<Mutex<TcpEntry>>> {
        self.tcp.read().unwrap().values().cloned().collect()
    }

    // UDP
    pub fn insert_udp(&self, entry: UdpEntry) {
        let id = entry.meta.session_id.clone();
        self.udp.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_udp(&self, id: &str) -> Option<Arc<Mutex<UdpEntry>>> {
        self.udp.read().unwrap().get(id).cloned()
    }
    pub fn remove_udp(&self, id: &str) -> bool {
        self.udp.write().unwrap().remove(id).is_some()
    }
    pub fn list_udp(&self) -> Vec<Arc<Mutex<UdpEntry>>> {
        self.udp.read().unwrap().values().cloned().collect()
    }

    // TLS
    pub fn insert_tls(&self, entry: TlsEntry) {
        let id = entry.meta.session_id.clone();
        self.tls.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_tls(&self, id: &str) -> Option<Arc<Mutex<TlsEntry>>> {
        self.tls.read().unwrap().get(id).cloned()
    }
    pub fn remove_tls(&self, id: &str) -> bool {
        self.tls.write().unwrap().remove(id).is_some()
    }
    pub fn list_tls(&self) -> Vec<Arc<Mutex<TlsEntry>>> {
        self.tls.read().unwrap().values().cloned().collect()
    }

    // SQL
    pub fn insert_sql(&self, entry: SqlEntry) {
        let id = entry.meta.session_id.clone();
        self.sql.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_sql(&self, id: &str) -> Option<Arc<Mutex<SqlEntry>>> {
        self.sql.read().unwrap().get(id).cloned()
    }
    pub fn remove_sql(&self, id: &str) -> bool {
        self.sql.write().unwrap().remove(id).is_some()
    }
    pub fn list_sql(&self) -> Vec<Arc<Mutex<SqlEntry>>> {
        self.sql.read().unwrap().values().cloned().collect()
    }

    // Remote FS
    pub fn insert_remote_fs(&self, entry: RemoteFsEntry) {
        let id = entry.meta.session_id.clone();
        self.remote_fs.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_remote_fs(&self, id: &str) -> Option<Arc<Mutex<RemoteFsEntry>>> {
        self.remote_fs.read().unwrap().get(id).cloned()
    }
    pub fn remove_remote_fs(&self, id: &str) -> bool {
        self.remote_fs.write().unwrap().remove(id).is_some()
    }
    pub fn list_remote_fs(&self) -> Vec<Arc<Mutex<RemoteFsEntry>>> {
        self.remote_fs.read().unwrap().values().cloned().collect()
    }

    // Code
    pub fn insert_code(&self, entry: CodeEntry) {
        let id = entry.meta.session_id.clone();
        self.code.write().unwrap().insert(id, Arc::new(Mutex::new(entry)));
    }
    pub fn get_code(&self, id: &str) -> Option<Arc<Mutex<CodeEntry>>> {
        self.code.read().unwrap().get(id).cloned()
    }
    pub fn remove_code(&self, id: &str) -> bool {
        self.code.write().unwrap().remove(id).is_some()
    }
    pub fn list_code(&self) -> Vec<Arc<Mutex<CodeEntry>>> {
        self.code.read().unwrap().values().cloned().collect()
    }
}
