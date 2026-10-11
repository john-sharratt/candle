//! The daemon's own log lines, as a bus the console can read.
//!
//! What `/ws/logs` streams: the same lines `tracing` writes to the terminal,
//! tapped on the way out.
//!
//! # Why a writer rather than a `Layer`
//!
//! A `tracing_subscriber::Layer` would see events *before* formatting and would
//! have to render them itself — a second formatter, guaranteed to drift from
//! the one on stderr, so the console and the terminal would disagree about what
//! the daemon said. Tapping the writer means there is exactly one formatter and
//! both readers see its output. The cost is that the structured fields are
//! already flattened into text and have to be picked apart again, which
//! [`parse`] does, tolerantly.
//!
//! # Replay, and why the ring is here rather than in the socket
//!
//! A console that connects to a live stream and shows nothing until the next
//! event looks broken — and a quiet process is exactly when you go looking at
//! the logs. So the bus keeps the last [`RING`] lines and hands them over on
//! connect, on the same socket as the tail. Every line carries a [`LogLine::seq`]:
//! the reader subscribes first, replays the ring, and skips whatever the
//! subscription then delivers that the replay already covered — so a line is
//! neither lost in the gap between the two nor shown twice.
//!
//! # One bus for every tenant
//!
//! A host serves several tenants from one process and one subscriber, so each
//! line names the tenant it came from ([`LogLine::tenant`]), read off the
//! target's crate. Engine lines name none: the engine is every tenant's.

use std::collections::VecDeque;
use std::io::{Result as IoResult, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tokio::sync::broadcast;

/// Lines kept for replay. Enough to see what happened just before you looked,
/// not a log store — that is what the terminal and the log file are for.
const RING: usize = 500;

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
    /// Position on the bus, strictly increasing — what lets a reader that
    /// replays the ring and then tails the stream skip what it has already
    /// shown.
    pub seq: u64,
}

const LEVELS: [&str; 5] = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];

pub struct LogBus {
    tx: broadcast::Sender<LogLine>,
    /// Guarded together with the next sequence number, so ring order and
    /// `seq` order are one order.
    ring: Mutex<(VecDeque<LogLine>, u64)>,
    /// The tenants a line's target can name.
    tenants: Vec<&'static str>,
    /// Lines a slow reader missed, process-wide — reported in the stream.
    dropped: AtomicU64,
}

impl LogBus {
    /// A bus whose lines are attributed to `tenants` by target.
    pub fn new(tenants: &[&'static str]) -> Arc<Self> {
        let (tx, _) = broadcast::channel(CHANNEL);
        Arc::new(Self {
            tx,
            ring: Mutex::new((VecDeque::with_capacity(RING), 1)),
            tenants: tenants.to_vec(),
            dropped: AtomicU64::new(0),
        })
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LogLine> {
        self.tx.subscribe()
    }

    /// The most recent lines, oldest first.
    pub fn recent(&self) -> Vec<LogLine> {
        self.ring.lock().unwrap().0.iter().cloned().collect()
    }

    /// Note that a reader fell `n` lines behind.
    pub fn note_dropped(&self, n: u64) {
        self.dropped.fetch_add(n, Ordering::Relaxed);
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

    fn push(&self, mut line: LogLine) {
        line.tenant = self.tenant_of(&line.target);
        {
            let mut g = self.ring.lock().unwrap();
            let (ring, next) = &mut *g;
            line.seq = *next;
            *next += 1;
            if ring.len() >= RING {
                ring.pop_front();
            }
            ring.push_back(line.clone());
            // Sent under the lock, so the stream's order is the ring's.
            // Nobody listening is the normal case — the console is usually
            // closed.
            let _ = self.tx.send(line);
        }
    }
}

/// Plugs into `tracing_subscriber::fmt::layer().with_writer(..)`.
///
/// # It buffers, because `Write` is a byte stream
///
/// Nothing guarantees that one `write` call is one line. `write_fmt` — what
/// every `write!`/`writeln!` expands to — issues a separate call per fragment
/// of the format string, so `writeln!(w, "line {i}")` arrives as three. Treating
/// each call as a line would shred that into three log entries, two of them
/// meaningless.
///
/// The fmt layer as it stands renders into a `String` and writes it in one go,
/// so this would have worked by luck. Luck is not a contract, and the failure
/// mode — silently mangled log lines — is one nobody would think to look for.
/// So bytes accumulate here and only a newline completes a line.
pub struct BusWriter {
    bus: Arc<LogBus>,
    /// Bytes seen since the last newline. A writer is made per event and
    /// dropped after it, so this is normally empty by the time it dies — and
    /// when it is not, `Drop` still delivers what it holds.
    partial: String,
}

impl BusWriter {
    pub fn new(bus: Arc<LogBus>) -> Self {
        Self {
            bus,
            partial: String::new(),
        }
    }

    /// Emit whatever is buffered as a final line, complete or not. A line
    /// without its newline is still something the daemon said.
    fn finish(&mut self) {
        if !self.partial.trim().is_empty() {
            let line = std::mem::take(&mut self.partial);
            self.bus.push(parse(&line));
        }
        self.partial.clear();
    }
}

impl Write for BusWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        // Non-UTF-8 is discarded rather than escaped: the fmt layer produces a
        // formatted `String`, so this cannot happen, and a lossy conversion
        // would put mojibake on the console under a name that reads as a real
        // log line.
        if let Ok(s) = std::str::from_utf8(buf) {
            self.partial.push_str(s);
            while let Some(nl) = self.partial.find('\n') {
                let line: String = self.partial.drain(..=nl).collect();
                // A line that is only whitespace is formatting, not content.
                if !line.trim().is_empty() {
                    self.bus.push(parse(&line));
                }
            }
        }
        // Always the full length: refusing bytes would make `tracing` retry a
        // line the daemon has already printed.
        Ok(buf.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        self.finish();
        Ok(())
    }
}

impl Drop for BusWriter {
    fn drop(&mut self) {
        self.finish();
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
    use super::*;

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

    #[test]
    fn the_ring_replays_the_most_recent_lines_oldest_first() {
        let bus = LogBus::new(&["npcd"]);
        let mut w = BusWriter::new(bus.clone());
        for i in 0..(RING + 10) {
            writeln!(w, "2026-08-28T06:35:04.1Z  INFO t: line {i}").unwrap();
        }
        let recent = bus.recent();
        assert_eq!(recent.len(), RING);
        assert_eq!(recent[0].msg, format!("line {}", 10));
        assert_eq!(recent[RING - 1].msg, format!("line {}", RING + 9));
    }

    /// The converse, and the one that actually bit: a line may arrive across
    /// several `write` calls. `write_fmt` splits on the format string's
    /// fragments, so this is what `writeln!(w, "line {i}")` really does.
    #[test]
    fn a_line_split_across_writes_is_reassembled() {
        let bus = LogBus::new(&["npcd"]);
        let mut w = BusWriter::new(bus.clone());
        w.write_all(b"2026-01-01T00:00:01.0Z  INFO npcd: line ")
            .unwrap();
        w.write_all(b"41").unwrap();
        // Nothing yet — no newline has arrived, so the line is not finished.
        assert!(bus.recent().is_empty());
        w.write_all(b"\n").unwrap();

        let r = bus.recent();
        assert_eq!(r.len(), 1, "one line, not one per write call");
        assert_eq!(r[0].msg, "line 41");
        assert_eq!(r[0].target, "npcd");
    }

    /// A writer dropped mid-line still delivers what it had. Losing the last
    /// thing a daemon said is losing the one line most likely to explain why it
    /// stopped saying anything.
    #[test]
    fn an_unterminated_line_survives_the_writers_death() {
        let bus = LogBus::new(&["npcd"]);
        {
            let mut w = BusWriter::new(bus.clone());
            w.write_all(b"2026-01-01T00:00:01.0Z ERROR npcd: cut off mid-")
                .unwrap();
        }
        let r = bus.recent();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].level, "ERROR");
        assert_eq!(r[0].msg, "cut off mid-");
    }

    /// A line names its tenant by its target's crate, and the engine's lines
    /// name none; sequence numbers count up across all of them.
    #[test]
    fn lines_name_their_tenant_and_count_up() {
        let bus = LogBus::new(&["zend", "npcd"]);
        let mut w = BusWriter::new(bus.clone());
        writeln!(w, "2026-01-01T00:00:01.0Z  INFO zend::session: a").unwrap();
        writeln!(w, "2026-01-01T00:00:01.0Z  INFO npcd::api: b").unwrap();
        writeln!(
            w,
            "2026-01-01T00:00:01.0Z  INFO candle_conversation::scheduler: c"
        )
        .unwrap();
        writeln!(w, "2026-01-01T00:00:01.0Z  INFO zend_tools::fs: d").unwrap();
        writeln!(w, "2026-01-01T00:00:01.0Z  INFO zendish::x: e").unwrap();
        let r = bus.recent();
        let got: Vec<(&str, u64)> = r.iter().map(|l| (l.tenant.as_str(), l.seq)).collect();
        assert_eq!(
            got,
            vec![("zend", 1), ("npcd", 2), ("", 3), ("zend", 4), ("", 5)]
        );
    }

    /// One `write` may carry several lines, and each is its own event.
    #[test]
    fn a_multi_line_write_becomes_multiple_lines() {
        let bus = LogBus::new(&["npcd"]);
        let mut w = BusWriter::new(bus.clone());
        w.write_all(b"2026-01-01T00:00:01.0Z  INFO a: one\n2026-01-01T00:00:02.0Z  WARN b: two\n")
            .unwrap();
        let r = bus.recent();
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].level.as_str(), r[0].msg.as_str()), ("INFO", "one"));
        assert_eq!((r[1].level.as_str(), r[1].msg.as_str()), ("WARN", "two"));
    }

    /// A subscriber taken before the write sees it; the ring keeps it too, so a
    /// console that connects a moment later still gets it.
    #[tokio::test]
    async fn a_line_reaches_both_the_stream_and_the_replay() {
        let bus = LogBus::new(&["npcd"]);
        let mut rx = bus.subscribe();
        let mut w = BusWriter::new(bus.clone());
        writeln!(w, "2026-01-01T00:00:01.0Z  WARN npcd: careful").unwrap();

        let got = rx.recv().await.unwrap();
        assert_eq!(got.level, "WARN");
        assert_eq!(got.msg, "careful");
        assert_eq!(bus.recent().last().unwrap().msg, "careful");
    }
}
