//! `GET /v1/logs` — this member's log searched: the active file and then the
//! rotated ones, newest first, read off disk a line at a time, until the
//! answer has its `limit` of lines or `since` is passed — so a search for what
//! happened lately reads what was written lately, not the whole log. The
//! gateway asks every member and merges their answers (`Merge::Envelope`), so
//! one request searches the cluster.
//!
//! | Parameter | |
//! |---|---|
//! | `grep` | a regular expression the whole line must match |
//! | `exclude` | a regular expression no match may match — the noise a `grep` alone cannot leave out |
//! | `level` | the least level kept: `TRACE` … `ERROR` |
//! | `tenant` | only lines one tenant wrote (`zend`, `npcd`) |
//! | `since` | only lines stamped at or after this, an RFC 3339 prefix such as `2026-10-16T00:20` |
//! | `limit` | the most matches returned, the latest kept — [`DEFAULT_LIMIT`] when absent, at most [`MAX_LIMIT`] |
//! | `lines_before`, `lines_after` | the file's lines around each match, as `grep -B`/`-A` — at most [`MAX_CONTEXT`] |
//!
//! Lines come back as written, full timestamps and all, oldest first: what a
//! reader greps for is the text, and the file is already its own best format.
//! With context asked for, runs of lines that do not touch are parted by a
//! `--` line, as `grep` parts them; a match's context stays inside its file,
//! and is the file's lines around it whatever the filters — `tenant`, `level`
//! and `exclude` choose the matches, not what stands beside them, as `grep -B`
//! shows the lines before a match whether or not they match.
//! The answer says how far back the log reaches (`held_from`), so a search
//! finding nothing older than that is read as the log not reaching back, not
//! as nothing having happened.
//!
//! Only the last `limit` matches are held while the files are read, so a
//! search costs its answer's memory, not the log's. A message's unstamped
//! line at the head of a file is taken as stamped with that file's first
//! stamp: the line it continues was written just before, in the file before.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::task::spawn_blocking;

use super::Shared;
use crate::logs::{parse, LogBus, LEVELS};

/// Matches returned when the request names no `limit`.
pub const DEFAULT_LIMIT: usize = 200;

/// The most matches one request returns.
pub const MAX_LIMIT: usize = 5000;

/// The most lines of context asked for on either side of a match.
pub const MAX_CONTEXT: usize = 20;

/// The line that parts two runs of lines that do not touch.
pub const SEPARATOR: &str = "--";

#[derive(Debug, Default, Deserialize)]
pub struct LogQuery {
    pub grep: Option<String>,
    pub exclude: Option<String>,
    pub level: Option<String>,
    pub tenant: Option<String>,
    pub since: Option<String>,
    pub limit: Option<usize>,
    pub lines_before: Option<usize>,
    pub lines_after: Option<usize>,
}

/// A search, checked.
pub struct Search {
    pattern: Option<Regex>,
    exclude: Option<Regex>,
    /// Index into [`LEVELS`] of the least level kept.
    min_level: usize,
    tenant: Option<String>,
    since: Option<String>,
    limit: usize,
    before: usize,
    after: usize,
}

/// What a search found.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct Found {
    /// The last `limit` matching lines, oldest first, each with the context
    /// asked for and runs that do not touch parted by [`SEPARATOR`].
    pub lines: Vec<String>,
    /// Every line that matched in the files read, those answered and those
    /// before them.
    pub matched: u64,
    /// Every line read.
    pub scanned: u64,
    /// How many files were read.
    pub files: usize,
    /// Every file that could hold a match was read: `false` when the search
    /// stopped at its `limit` with older files left unread, which may hold
    /// earlier matches.
    pub complete: bool,
    /// The stamp of the log's oldest line: every line written since is in it.
    pub held_from: Option<String>,
}

/// One match and the context around it, each line by its place in its file.
struct Hit {
    lines: Vec<(u64, String)>,
}

/// What one file held for a search.
struct InFile {
    /// Its last matches, at most as many as were asked for, oldest first.
    kept: VecDeque<Hit>,
    matched: u64,
    scanned: u64,
    /// It holds a line stamped before `since`: every older file is wholly
    /// before it.
    passed_since: bool,
}

/// The first stamp a line opens with: the first word, when it starts with a
/// digit.
fn stamp_of(text: &str) -> Option<&str> {
    text.split_whitespace()
        .next()
        .filter(|w| w.as_bytes()[0].is_ascii_digit())
}

/// The stamp of the oldest line the file at `path` holds, read from its head.
pub fn first_stamp(path: &Path) -> Option<String> {
    let reader = BufReader::new(File::open(path).ok()?);
    reader
        .split(b'\n')
        .map_while(Result::ok)
        .find_map(|line| stamp_of(&String::from_utf8_lossy(&line)).map(str::to_string))
}

impl Search {
    /// `q`, checked: a pattern that does not compile or a level that does not
    /// exist is refused, not ignored.
    pub fn new(q: &LogQuery) -> Result<Self, String> {
        let compile = |p: &Option<String>, name: &str| match p {
            Some(p) => Regex::new(p)
                .map(Some)
                .map_err(|e| format!("`{name}` is not a pattern: {e}")),
            None => Ok(None),
        };
        let pattern = compile(&q.grep, "grep")?;
        let exclude = compile(&q.exclude, "exclude")?;
        let min_level = match &q.level {
            Some(l) => LEVELS
                .iter()
                .position(|x| x.eq_ignore_ascii_case(l))
                .ok_or_else(|| format!("`level` is one of {}", LEVELS.join(", ")))?,
            None => 0,
        };
        Ok(Self {
            pattern,
            exclude,
            min_level,
            tenant: q.tenant.clone(),
            since: q.since.clone(),
            limit: q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT),
            before: q.lines_before.unwrap_or(0).min(MAX_CONTEXT),
            after: q.lines_after.unwrap_or(0).min(MAX_CONTEXT),
        })
    }

    /// Search `files`, given oldest first, newest first: the last `limit`
    /// matches, read no further back than they need.
    pub fn run(&self, bus: &LogBus, files: &[PathBuf]) -> io::Result<Found> {
        let mut newest_first: Vec<VecDeque<Hit>> = Vec::new();
        let (mut kept, mut matched, mut scanned, mut read) = (0usize, 0u64, 0u64, 0usize);
        let mut complete = true;
        for (i, path) in files.iter().enumerate().rev() {
            let file = self.in_file(bus, path, self.limit - kept)?;
            read += 1;
            matched += file.matched;
            scanned += file.scanned;
            kept += file.kept.len();
            newest_first.push(file.kept);
            if file.passed_since {
                break;
            }
            if kept == self.limit {
                complete = i == 0;
                break;
            }
        }
        Ok(Found {
            lines: self.assemble(newest_first.into_iter().rev()),
            matched,
            scanned,
            files: read,
            complete,
            held_from: files.first().and_then(|p| first_stamp(p)),
        })
    }

    /// The files' hits, oldest file first, as the answer's lines: each line
    /// once, and with context asked for, runs that do not touch — or that are
    /// in different files — parted by [`SEPARATOR`].
    fn assemble(&self, files: impl Iterator<Item = VecDeque<Hit>>) -> Vec<String> {
        let context = self.before > 0 || self.after > 0;
        let mut out = Vec::new();
        for hits in files {
            let mut last: Option<u64> = None;
            for hit in hits {
                for (at, text) in hit.lines {
                    if last.is_some_and(|l| at <= l) {
                        continue;
                    }
                    let touches = last.is_some_and(|l| at == l + 1);
                    if context && !out.is_empty() && !touches {
                        out.push(SEPARATOR.to_string());
                    }
                    out.push(text);
                    last = Some(at);
                }
            }
        }
        out
    }

    /// The last `want` matches in the file at `path`, with their context.
    fn in_file(&self, bus: &LogBus, path: &Path, want: usize) -> io::Result<InFile> {
        let mut scan = Scan {
            found: InFile {
                kept: VecDeque::with_capacity(want),
                matched: 0,
                scanned: 0,
                passed_since: false,
            },
            want,
            next: 0,
            ring: VecDeque::with_capacity(self.before + 1),
            open_after: 0,
        };
        // A line without a stamp — the rest of a message that spans lines —
        // is as old as the stamped line before it; at the head of the file,
        // as its first stamped line.
        let mut stamp: Option<String> = None;
        let mut unstamped: Vec<String> = Vec::new();
        let mut reader = BufReader::new(File::open(path)?);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            if reader.read_until(b'\n', &mut buf)? == 0 {
                break;
            }
            let text = String::from_utf8_lossy(&buf);
            let text = text.trim_end_matches(['\n', '\r']);
            if text.trim().is_empty() {
                continue;
            }
            if let Some(s) = stamp_of(text) {
                stamp = Some(s.to_string());
            }
            let Some(at) = &stamp else {
                unstamped.push(text.to_string());
                continue;
            };
            for head in unstamped.drain(..) {
                self.take(bus, &head, at, &mut scan);
            }
            self.take(bus, text, at, &mut scan);
        }
        for head in unstamped {
            self.take(bus, &head, "", &mut scan);
        }
        Ok(scan.found)
    }

    /// Read `text`, stamped `at`: kept in `scan` as a match with the lines
    /// before it, or as context after the last match.
    fn take(&self, bus: &LogBus, text: &str, at: &str, scan: &mut Scan) {
        scan.found.scanned += 1;
        let place = scan.next;
        scan.next += 1;
        if self.since.as_deref().is_some_and(|s| at < s) {
            scan.found.passed_since = true;
        }
        if self.admits(bus, text, at) {
            scan.found.matched += 1;
            if scan.found.kept.len() == scan.want {
                scan.found.kept.pop_front();
            }
            let mut lines: Vec<(u64, String)> = scan.ring.iter().cloned().collect();
            lines.push((place, text.to_string()));
            scan.found.kept.push_back(Hit { lines });
            scan.open_after = self.after;
        } else if scan.open_after > 0 {
            if let Some(hit) = scan.found.kept.back_mut() {
                hit.lines.push((place, text.to_string()));
            }
            scan.open_after -= 1;
        }
        if self.before > 0 {
            if scan.ring.len() == self.before {
                scan.ring.pop_front();
            }
            scan.ring.push_back((place, text.to_string()));
        }
    }

    fn admits(&self, bus: &LogBus, text: &str, stamp: &str) -> bool {
        if self.since.as_deref().is_some_and(|s| stamp < s) {
            return false;
        }
        if self.pattern.as_ref().is_some_and(|p| !p.is_match(text)) {
            return false;
        }
        if self.exclude.as_ref().is_some_and(|x| x.is_match(text)) {
            return false;
        }
        if self.min_level > 0 {
            let level = parse(text).level;
            if LEVELS.iter().position(|l| *l == level).unwrap_or(0) < self.min_level {
                return false;
            }
        }
        if let Some(t) = &self.tenant {
            if bus.line(text).tenant != *t {
                return false;
            }
        }
        true
    }
}

/// One file's read in progress.
struct Scan {
    found: InFile,
    /// The most matches kept.
    want: usize,
    /// The place of the next line read.
    next: u64,
    /// The last lines read, for the context before the next match.
    ring: VecDeque<(u64, String)>,
    /// Lines still owed to the last match's context after it.
    open_after: usize,
}

pub async fn grep(State(s): State<Arc<Shared>>, Query(q): Query<LogQuery>) -> Response {
    let search = match Search::new(&q) {
        Ok(search) => search,
        Err(detail) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "bad_request", "detail": detail})),
            )
                .into_response()
        }
    };
    let bus = Arc::clone(&s.log);
    let found = spawn_blocking(move || {
        let files = bus.files();
        search.run(&bus, &files)
    })
    .await;
    match found {
        Ok(Ok(found)) => Json(found).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "unreadable", "detail": format!("reading the log: {e}")})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "failed", "detail": format!("the search stopped: {e}")})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    const LOG: &str = "\
2026-10-16T00:01:00.0Z  INFO zend::repo_scan: repo map: ingesting
2026-10-16T00:02:00.0Z  WARN cluster::replicate::tick: a catch-up went quiet: asking again
2026-10-16T00:03:00.0Z  INFO npcd::engine::mind: dreamt
the dream's second line
2026-10-16T00:04:00.0Z ERROR cluster::replicate::tick: a catch-up went quiet: asking again
";

    fn files(dir: &TempDir) -> Vec<PathBuf> {
        let older = dir.path().join("m.log.1");
        let active = dir.path().join("m.log");
        let (a, b) = LOG.split_at(LOG.find("2026-10-16T00:03").unwrap());
        fs::write(&older, a).unwrap();
        fs::write(&active, b).unwrap();
        vec![older, active]
    }

    fn run(q: LogQuery) -> Found {
        let dir = TempDir::new().unwrap();
        let bus = LogBus::new(&["zend", "npcd"]);
        Search::new(&q).unwrap().run(&bus, &files(&dir)).unwrap()
    }

    /// The pattern is matched against the whole line, across every file, oldest
    /// first, and the answer says how far back the log reaches.
    #[test]
    fn a_pattern_finds_its_lines_across_the_files() {
        let f = run(LogQuery {
            grep: Some("went quiet".into()),
            ..LogQuery::default()
        });
        assert_eq!(
            f.lines,
            [
                "2026-10-16T00:02:00.0Z  WARN cluster::replicate::tick: a catch-up went quiet: asking again",
                "2026-10-16T00:04:00.0Z ERROR cluster::replicate::tick: a catch-up went quiet: asking again",
            ]
        );
        assert_eq!((f.matched, f.scanned, f.files, f.complete), (2, 5, 2, true));
        assert_eq!(f.held_from.as_deref(), Some("2026-10-16T00:01:00.0Z"));
    }

    /// **The search reads back no further than its answer needs**: the
    /// latest `limit` matches, found in the newest file, leave the older
    /// unread — and the answer says it may not be every match.
    #[test]
    fn the_limit_keeps_the_latest_matches_and_reads_no_further() {
        let f = run(LogQuery {
            grep: Some("catch-up".into()),
            limit: Some(1),
            ..LogQuery::default()
        });
        assert_eq!(
            f.lines,
            ["2026-10-16T00:04:00.0Z ERROR cluster::replicate::tick: a catch-up went quiet: asking again"]
        );
        assert_eq!((f.matched, f.files, f.complete), (1, 1, false));
        let f = run(LogQuery {
            grep: Some("catch-up".into()),
            limit: Some(2),
            ..LogQuery::default()
        });
        assert_eq!((f.matched, f.files, f.complete), (2, 2, true));
    }

    /// **A search past `since` stops there**: the file that holds a line
    /// older than it is the last read.
    #[test]
    fn a_search_stops_once_it_passes_since() {
        let f = run(LogQuery {
            since: Some("2026-10-16T00:04".into()),
            ..LogQuery::default()
        });
        assert_eq!(
            f.lines,
            ["2026-10-16T00:04:00.0Z ERROR cluster::replicate::tick: a catch-up went quiet: asking again"]
        );
        assert_eq!((f.files, f.complete), (1, true));
    }

    /// **A message's unstamped line at the head of a file is as old as the
    /// file's first stamp**, so a search reading the newest file first keeps
    /// it with the lines around it.
    #[test]
    fn an_unstamped_line_at_a_files_head_takes_its_first_stamp() {
        let dir = TempDir::new().unwrap();
        let active = dir.path().join("m.log");
        fs::write(
            &active,
            "the rest of a message\n2026-10-16T00:05:00.0Z  INFO zend::x: next\n",
        )
        .unwrap();
        let bus = LogBus::new(&["zend", "npcd"]);
        let q = LogQuery {
            since: Some("2026-10-16T00:05".into()),
            ..LogQuery::default()
        };
        let f = Search::new(&q).unwrap().run(&bus, &[active]).unwrap();
        assert_eq!(
            f.lines,
            [
                "the rest of a message",
                "2026-10-16T00:05:00.0Z  INFO zend::x: next"
            ]
        );
    }

    /// `level` is a floor, and a level word inside a message does not count.
    #[test]
    fn the_level_is_a_floor() {
        let f = run(LogQuery {
            level: Some("warn".into()),
            ..LogQuery::default()
        });
        assert_eq!(f.lines.len(), 2);
        assert!(f.lines.iter().all(|l| l.contains("went quiet")));
    }

    /// `tenant` keeps one tenant's lines; `since` drops what is older, and a
    /// message's unstamped second line is as old as its first.
    #[test]
    fn tenant_and_since_narrow_the_search() {
        let f = run(LogQuery {
            tenant: Some("npcd".into()),
            ..LogQuery::default()
        });
        assert_eq!(
            f.lines,
            ["2026-10-16T00:03:00.0Z  INFO npcd::engine::mind: dreamt"]
        );
        let f = run(LogQuery {
            since: Some("2026-10-16T00:03".into()),
            ..LogQuery::default()
        });
        assert_eq!(
            f.lines,
            [
                "2026-10-16T00:03:00.0Z  INFO npcd::engine::mind: dreamt",
                "the dream's second line",
                "2026-10-16T00:04:00.0Z ERROR cluster::replicate::tick: a catch-up went quiet: asking again",
            ]
        );
    }

    /// **`exclude` leaves its lines out of the matches**: the replication
    /// lines without the one that is an error.
    #[test]
    fn exclude_leaves_its_lines_out() {
        let f = run(LogQuery {
            grep: Some("cluster::".into()),
            exclude: Some("ERROR".into()),
            ..LogQuery::default()
        });
        assert_eq!(
            f.lines,
            ["2026-10-16T00:02:00.0Z  WARN cluster::replicate::tick: a catch-up went quiet: asking again"]
        );
        assert_eq!(f.matched, 1);
    }

    const FILE: &str = "\
2026-10-16T00:01:00.0Z  INFO a: one
2026-10-16T00:02:00.0Z  INFO a: two
2026-10-16T00:03:00.0Z  WARN a: three
2026-10-16T00:04:00.0Z  INFO a: four
2026-10-16T00:05:00.0Z  INFO a: five
2026-10-16T00:06:00.0Z  INFO a: six
2026-10-16T00:07:00.0Z  WARN a: seven
2026-10-16T00:08:00.0Z  INFO a: eight
";

    fn one_file(q: LogQuery) -> Vec<String> {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("m.log");
        fs::write(&path, FILE).unwrap();
        let bus = LogBus::new(&["zend"]);
        Search::new(&q).unwrap().run(&bus, &[path]).unwrap().lines
    }

    fn line(n: &str, level: &str, word: &str) -> String {
        format!("2026-10-16T00:0{n}:00.0Z  {level} a: {word}")
    }

    /// **Context comes from the file around each match**, as `grep -B`/`-A`
    /// gives it, and runs that do not touch are parted by `--`.
    #[test]
    fn context_lines_come_around_each_match() {
        assert_eq!(
            one_file(LogQuery {
                grep: Some("WARN".into()),
                lines_before: Some(1),
                lines_after: Some(1),
                ..LogQuery::default()
            }),
            [
                line("2", "INFO", "two"),
                line("3", "WARN", "three"),
                line("4", "INFO", "four"),
                SEPARATOR.to_string(),
                line("6", "INFO", "six"),
                line("7", "WARN", "seven"),
                line("8", "INFO", "eight"),
            ]
        );
    }

    /// **Context that overlaps is one run**, each line once: three lines
    /// after the first match reach the second's lines before it.
    #[test]
    fn overlapping_context_is_one_run() {
        assert_eq!(
            one_file(LogQuery {
                grep: Some("WARN".into()),
                lines_before: Some(1),
                lines_after: Some(3),
                ..LogQuery::default()
            }),
            [
                line("2", "INFO", "two"),
                line("3", "WARN", "three"),
                line("4", "INFO", "four"),
                line("5", "INFO", "five"),
                line("6", "INFO", "six"),
                line("7", "WARN", "seven"),
                line("8", "INFO", "eight"),
            ]
        );
    }

    /// **Context counts no match against the limit**: the newest match is
    /// kept with the line before it, and the older one is gone with its own.
    #[test]
    fn context_is_not_counted_against_the_limit() {
        assert_eq!(
            one_file(LogQuery {
                grep: Some("WARN".into()),
                lines_before: Some(1),
                limit: Some(1),
                ..LogQuery::default()
            }),
            [line("6", "INFO", "six"), line("7", "WARN", "seven")]
        );
    }

    /// A file's oldest stamp is its first stamped line's, past any unstamped
    /// head.
    #[test]
    fn a_files_first_stamp_is_its_first_stamped_line() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("m.log");
        fs::write(
            &path,
            "a continued line\n2026-10-16T00:05:00.0Z  INFO a: x\n",
        )
        .unwrap();
        assert_eq!(
            first_stamp(&path).as_deref(),
            Some("2026-10-16T00:05:00.0Z")
        );
        assert_eq!(first_stamp(&dir.path().join("none.log")), None);
    }

    /// A pattern that does not compile, or a level that does not exist, is
    /// refused rather than searched as nothing.
    #[test]
    fn a_bad_pattern_or_level_is_refused() {
        let bad = LogQuery {
            grep: Some("(".into()),
            ..LogQuery::default()
        };
        assert!(Search::new(&bad).is_err());
        let bad = LogQuery {
            exclude: Some("(".into()),
            ..LogQuery::default()
        };
        assert!(Search::new(&bad).err().unwrap().starts_with("`exclude`"));
        let bad = LogQuery {
            level: Some("LOUD".into()),
            ..LogQuery::default()
        };
        assert_eq!(
            Search::new(&bad).err().unwrap(),
            "`level` is one of TRACE, DEBUG, INFO, WARN, ERROR"
        );
    }
}
