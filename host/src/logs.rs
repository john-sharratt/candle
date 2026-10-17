//! The daemon's own log lines, as a bus the console can read.
//!
//! What `/ws/logs` streams and `/v1/logs` searches: the lines the log file
//! holds, tapped as they are written to it.
//!
//! # Fed by the file, not beside it
//!
//! The bus is the log file's tap (`web::log_file`): every write the
//! file takes is handed here under the file's lock, split into lines and
//! broadcast. There is exactly one formatter and one record of what the
//! daemon said — the file — and the console reads that record rather than a
//! copy of it. The cost is that the structured fields are already flattened
//! into text and have to be picked apart again, which [`parse`] does,
//! tolerantly.
//!
//! # Replay, from the file
//!
//! A console that connects to a live stream and shows nothing until the next
//! event looks broken — and a quiet process is exactly when you go looking at
//! the logs. So a reader starts with the file's last lines and then follows
//! the stream. The bus keeps no lines of its own: the replay is read back
//! from disk ([`LogBus::replay`]), and the subscription is taken under the
//! file's lock in the same moment, so a line is in the replay or the stream —
//! never both, never neither.
//!
//! # One bus for every tenant
//!
//! A host serves several tenants from one process and one subscriber, so each
//! line names the tenant it came from ([`LogLine::tenant`]), read off the
//! target's crate. Engine lines name none: the engine is every tenant's.

use std::path::PathBuf;
use std::str::from_utf8;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Serialize;
use tokio::sync::broadcast;
use web::log_file::RotatingFileLog;

/// Broadcast depth. Generous because a slow console must not cost the *daemon*
/// anything: when it overflows, the reader is told it lagged rather than the
/// writer being made to wait.
const CHANNEL: usize = 2048;

/// One line, already taken apart.
///
/// The console never parses a formatted string — filtering by level is then a
/// property test rather than a regex over presentation, and a message that
/// happens to contain the word `ERROR` cannot masquerade as one.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LogLine {
    pub ts: String,
    pub level: String,
    pub target: String,
    pub msg: String,
    /// The tenant the line came from — `zend`, `npcd` — or empty for the
    /// engine and the host, which belong to every tenant.
    pub tenant: String,
    /// Position on the bus, strictly increasing across the live stream; zero
    /// for a line replayed from the file, which the bus never carried.
    pub seq: u64,
}

pub const LEVELS: [&str; 5] = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];

pub struct LogBus {
    tx: broadcast::Sender<LogLine>,
    next: AtomicU64,
    /// The tenants a line's target can name.
    tenants: Vec<&'static str>,
    /// Lines a slow reader missed, process-wide — reported in the stream.
    dropped: AtomicU64,
    /// Bytes the file took since its last newline. Only the file's tap writes
    /// here, under the file's lock.
    partial: Mutex<String>,
    /// The file feeding the bus, once [`Self::attach`] has run.
    file: OnceLock<RotatingFileLog>,
}

impl LogBus {
    /// A bus whose lines are attributed to `tenants` by target.
    pub fn new(tenants: &[&'static str]) -> Arc<Self> {
        let (tx, _) = broadcast::channel(CHANNEL);
        Arc::new(Self {
            tx,
            next: AtomicU64::new(1),
            tenants: tenants.to_vec(),
            dropped: AtomicU64::new(0),
            partial: Mutex::new(String::new()),
            file: OnceLock::new(),
        })
    }

    /// Feed the bus from `file`: every line written to it is a line here, and
    /// a reader's replay is read back from it. The tap holds the bus weakly,
    /// so the file outliving it costs nothing.
    pub fn attach(self: &Arc<Self>, file: &RotatingFileLog) {
        let bus = Arc::downgrade(self);
        file.set_tap(Box::new(move |bytes| {
            if let Some(bus) = bus.upgrade() {
                bus.take(bytes);
            }
        }));
        let _ = self.file.set(file.clone());
    }

    /// A reader's start: a subscription, and the lines among the file's last
    /// `max_bytes` — split at the moment the subscription was taken, so each
    /// line is in exactly one of the two.
    pub fn replay(&self, max_bytes: u64) -> (broadcast::Receiver<LogLine>, Vec<LogLine>) {
        let Some(file) = self.file.get() else {
            return (self.tx.subscribe(), Vec::new());
        };
        match file.tail(max_bytes, || self.tx.subscribe()) {
            Ok((bytes, rx)) => (rx, self.lines(&bytes)),
            // Said in the stream rather than swallowed: an empty pane with no
            // reason reads as a silent daemon.
            Err(e) => (
                self.tx.subscribe(),
                vec![LogLine {
                    ts: String::new(),
                    level: "WARN".to_owned(),
                    target: "host::logs".to_owned(),
                    msg: format!("the log file could not be read back: {e}"),
                    tenant: String::new(),
                    seq: 0,
                }],
            ),
        }
    }

    /// Every file of the log, oldest first; none before [`Self::attach`].
    pub fn files(&self) -> Vec<PathBuf> {
        self.file
            .get()
            .map(RotatingFileLog::files)
            .unwrap_or_default()
    }

    /// One formatted line as the bus carries it: taken apart, its tenant named.
    pub fn line(&self, text: &str) -> LogLine {
        let mut line = parse(text);
        line.tenant = self.tenant_of(&line.target);
        line
    }

    /// Note that a reader fell `n` lines behind.
    pub fn note_dropped(&self, n: u64) {
        self.dropped.fetch_add(n, Ordering::Relaxed);
    }

    /// Every non-blank line in `bytes`, in order.
    fn lines(&self, bytes: &[u8]) -> Vec<LogLine> {
        String::from_utf8_lossy(bytes)
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| self.line(l))
            .collect()
    }

    /// The tenant a target names, or empty: its crate, or one of the crates
    /// named after it (`zend_tools`, `zend_vfs` are zend's).
    fn tenant_of(&self, target: &str) -> String {
        let krate = target.split("::").next().unwrap_or("");
        self.tenants
            .iter()
            .find(|t| {
                krate == **t
                    || krate
                        .strip_prefix(**t)
                        .is_some_and(|rest| rest.starts_with('_'))
            })
            .map(|t| t.to_string())
            .unwrap_or_default()
    }

    /// What the file took. Nothing guarantees one write is one line, so bytes
    /// accumulate and only a newline completes a line — treating each write as
    /// a line would shred one split across writes into meaningless fragments.
    /// Non-UTF-8 is discarded rather than escaped: the formatter writes a
    /// `String`, so it cannot happen, and a lossy conversion would put mojibake
    /// on the console under a name that reads as a real log line.
    fn take(&self, bytes: &[u8]) {
        let Ok(s) = from_utf8(bytes) else {
            return;
        };
        let mut partial = self.partial.lock().unwrap_or_else(|e| e.into_inner());
        partial.push_str(s);
        while let Some(nl) = partial.find('\n') {
            let text: String = partial.drain(..=nl).collect();
            if !text.trim().is_empty() {
                self.push(self.line(&text));
            }
        }
    }

    fn push(&self, mut line: LogLine) {
        line.seq = self.next.fetch_add(1, Ordering::Relaxed);
        // Nobody listening is the normal case — the console is usually closed.
        let _ = self.tx.send(line);
    }
}

/// Split one formatted line into its parts.
///
/// Tolerant on purpose. The formatter's exact layout is not a contract — ANSI
/// colour, a span list, a changed time format all alter it — and a log viewer
/// that drops lines it cannot parse hides exactly the unusual output somebody
/// went looking for. Anything unrecognised becomes the message, whole.
pub fn parse(line: &str) -> LogLine {
    // The fmt layer writes ANSI escapes when the terminal takes them; they are
    // noise in JSON and would break level detection by splitting the word.
    let clean = strip_ansi(line);
    let ts = hms(clean.split_whitespace().next().unwrap_or(""));

    // The level is the earliest level word in the line, standing alone: the
    // formatter writes it before the target and the message, so a level word
    // later in the line is the message's — "the INFO field was empty" on an
    // ERROR line stays an ERROR. As a whole word, so `zend::warn_util` is not
    // one either.
    let (level, after) = LEVELS
        .iter()
        .filter_map(|lv| find_token(&clean, lv).map(|idx| (idx, *lv)))
        .min_by_key(|(idx, _)| *idx)
        .map_or(("INFO", clean.trim()), |(idx, lv)| {
            (lv, clean[idx + lv.len()..].trim_start())
        });

    let (target, msg) = match after.split_once(": ") {
        Some((t, m)) => (t.trim().to_owned(), m.trim().to_owned()),
        None => (String::new(), after.trim().to_owned()),
    };

    LogLine {
        ts,
        level: level.to_owned(),
        target,
        msg,
        tenant: String::new(),
        seq: 0,
    }
}

/// `2026-08-28T06:35:04.123456Z` → `06:35:04`. The date is the same for every
/// line on screen, so it is noise in a column that has to stay narrow.
fn hms(stamp: &str) -> String {
    stamp
        .split('T')
        .nth(1)
        .map(|t| t.split('.').next().unwrap_or(t).trim_end_matches('Z'))
        .unwrap_or("")
        .to_owned()
}

/// The byte offset of `token` when it stands alone.
fn find_token(hay: &str, token: &str) -> Option<usize> {
    let bytes = hay.as_bytes();
    let mut from = 0;
    while let Some(rel) = hay[from..].find(token) {
        let at = from + rel;
        let before_ok = at == 0 || bytes[at - 1].is_ascii_whitespace();
        let end = at + token.len();
        let after_ok = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return Some(at);
        }
        from = end;
    }
    None
}

/// Drop CSI escape sequences. Only the `ESC [ … <final>` form, which is all the
/// fmt layer emits.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        if chars.next() == Some('[') {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::TempDir;

    use super::*;

    /// A bus fed by a log file in `dir`, as `init_logging` wires it.
    fn attached(dir: &TempDir, tenants: &[&'static str]) -> (Arc<LogBus>, RotatingFileLog) {
        let bus = LogBus::new(tenants);
        let file = RotatingFileLog::new(dir.path(), "test.log").unwrap();
        bus.attach(&file);
        (bus, file)
    }

    #[test]
    fn a_formatted_line_comes_apart_into_its_fields() {
        let l = parse("2026-08-28T06:35:04.123456Z  INFO npcd::api: world: 2 loaded");
        assert_eq!(l.ts, "06:35:04");
        assert_eq!(l.level, "INFO");
        assert_eq!(l.target, "npcd::api");
        assert_eq!(l.msg, "world: 2 loaded");
    }

    /// The console filters on `level`, so a message that merely says "ERROR"
    /// must not be filed as one — that is a line appearing in a filter the
    /// reader set to catch real failures.
    #[test]
    fn a_level_word_inside_a_message_is_not_the_level() {
        let l = parse("2026-08-28T06:35:04.1Z  INFO npcd::api: the ERROR count is 0");
        assert_eq!(l.level, "INFO");
        assert_eq!(l.msg, "the ERROR count is 0");

        // Nor is one glued into a longer word.
        let l = parse("2026-08-28T06:35:04.1Z  INFO npcd::api: ERRORS were counted");
        assert_eq!(l.level, "INFO");

        // A higher level's line naming a lower level keeps its own, and its
        // target and message come apart where the formatter put them.
        let l = parse("2026-08-28T06:35:04.1Z ERROR zend::x: the INFO field was empty");
        assert_eq!(l.level, "ERROR");
        assert_eq!(l.target, "zend::x");
        assert_eq!(l.msg, "the INFO field was empty");
    }

    #[test]
    fn colour_escapes_do_not_reach_the_wire() {
        let l = parse("2026-08-28T06:35:04.1Z \u{1b}[32m INFO\u{1b}[0m npcd: up");
        assert_eq!(l.level, "INFO");
        assert_eq!(l.target, "npcd");
        assert_eq!(l.msg, "up");
        assert!(!l.msg.contains('\u{1b}'));
    }

    /// Nothing is dropped for being unrecognisable. A viewer that hides what it
    /// cannot parse hides the unusual output somebody went looking for.
    #[test]
    fn an_unparseable_line_survives_whole() {
        let l = parse("a bare line with no shape at all");
        assert_eq!(l.level, "INFO");
        assert_eq!(l.target, "");
        assert_eq!(l.msg, "a bare line with no shape at all");
    }

    /// **The replay is the file's, and the stream starts where it ends.** A
    /// line written before the reader arrived is replayed and not streamed;
    /// one written after is streamed and not replayed.
    #[tokio::test]
    async fn the_replay_is_read_from_the_file_and_the_stream_follows_it() {
        let dir = TempDir::new().unwrap();
        let (bus, mut file) = attached(&dir, &["zend"]);
        writeln!(file, "2026-01-01T00:00:01.0Z  INFO zend::a: before").unwrap();

        let (mut rx, replay) = bus.replay(1 << 20);
        writeln!(file, "2026-01-01T00:00:02.0Z  WARN zend::a: after").unwrap();

        let replayed: Vec<(&str, &str, u64)> = replay
            .iter()
            .map(|l| (l.msg.as_str(), l.tenant.as_str(), l.seq))
            .collect();
        assert_eq!(replayed, [("before", "zend", 0)]);
        let streamed = rx.recv().await.unwrap();
        assert_eq!(
            (streamed.msg.as_str(), streamed.level.as_str()),
            ("after", "WARN")
        );
        assert!(rx.try_recv().is_err(), "nothing else streamed");
    }

    /// The replay is bounded by bytes, taken from the end, in whole lines.
    #[test]
    fn the_replay_is_the_files_last_lines() {
        let dir = TempDir::new().unwrap();
        let (bus, mut file) = attached(&dir, &["npcd"]);
        for i in 0..100 {
            writeln!(file, "2026-08-28T06:35:04.1Z  INFO npcd: line {i:03}").unwrap();
        }
        // Each line is 44 bytes: 100 bytes back cuts into the third from last,
        // which is dropped, and holds the last two whole.
        let (_rx, replay) = bus.replay(100);
        let msgs: Vec<&str> = replay.iter().map(|l| l.msg.as_str()).collect();
        assert_eq!(msgs, ["line 098", "line 099"]);
    }

    /// The converse, and the one that actually bit: a line may arrive across
    /// several writes. `write_fmt` splits on the format string's fragments, so
    /// this is what `writeln!(w, "line {i}")` really does.
    #[tokio::test]
    async fn a_line_split_across_writes_is_reassembled() {
        let dir = TempDir::new().unwrap();
        let (bus, mut file) = attached(&dir, &["npcd"]);
        let (mut rx, _) = bus.replay(0);
        file.write_all(b"2026-01-01T00:00:01.0Z  INFO npcd: line ")
            .unwrap();
        file.write_all(b"41").unwrap();
        // Nothing yet — no newline has arrived, so the line is not finished.
        assert!(rx.try_recv().is_err());
        file.write_all(b"\n").unwrap();

        let got = rx.recv().await.unwrap();
        assert_eq!(got.msg, "line 41");
        assert_eq!(got.target, "npcd");
        assert!(rx.try_recv().is_err(), "one line, not one per write call");
    }

    /// A line names its tenant by its target's crate, and the engine's lines
    /// name none; sequence numbers count up across all of them.
    #[tokio::test]
    async fn lines_name_their_tenant_and_count_up() {
        let dir = TempDir::new().unwrap();
        let (bus, mut file) = attached(&dir, &["zend", "npcd"]);
        let (mut rx, _) = bus.replay(0);
        writeln!(file, "2026-01-01T00:00:01.0Z  INFO zend::session: a").unwrap();
        writeln!(file, "2026-01-01T00:00:01.0Z  INFO npcd::api: b").unwrap();
        writeln!(
            file,
            "2026-01-01T00:00:01.0Z  INFO candle_conversation::scheduler: c"
        )
        .unwrap();
        writeln!(file, "2026-01-01T00:00:01.0Z  INFO zend_tools::fs: d").unwrap();
        writeln!(file, "2026-01-01T00:00:01.0Z  INFO zendish::x: e").unwrap();
        let mut got = Vec::new();
        for _ in 0..5 {
            let l = rx.recv().await.unwrap();
            got.push((l.tenant, l.seq));
        }
        let got: Vec<(&str, u64)> = got.iter().map(|(t, s)| (t.as_str(), *s)).collect();
        assert_eq!(
            got,
            vec![("zend", 1), ("npcd", 2), ("", 3), ("zend", 4), ("", 5)]
        );
    }

    /// One write may carry several lines, and each is its own event.
    #[tokio::test]
    async fn a_multi_line_write_becomes_multiple_lines() {
        let dir = TempDir::new().unwrap();
        let (bus, mut file) = attached(&dir, &["npcd"]);
        let (mut rx, _) = bus.replay(0);
        file.write_all(
            b"2026-01-01T00:00:01.0Z  INFO a: one\n2026-01-01T00:00:02.0Z  WARN b: two\n",
        )
        .unwrap();
        let one = rx.recv().await.unwrap();
        let two = rx.recv().await.unwrap();
        assert_eq!((one.level.as_str(), one.msg.as_str()), ("INFO", "one"));
        assert_eq!((two.level.as_str(), two.msg.as_str()), ("WARN", "two"));
    }

    /// A bus with no file has nothing to replay and no files to search: it
    /// keeps no record of its own.
    #[test]
    fn an_unattached_bus_replays_nothing() {
        let bus = LogBus::new(&["zend"]);
        let (_rx, replay) = bus.replay(1 << 20);
        assert!(replay.is_empty());
        assert!(bus.files().is_empty());
    }
}
