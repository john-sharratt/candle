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
//! | `level` | the least level kept: `TRACE` … `ERROR` |
//! | `tenant` | only lines one tenant wrote (`zend`, `npcd`) |
//! | `since` | only lines stamped at or after this, an RFC 3339 prefix such as `2026-10-16T00:20` |
//! | `limit` | the most matches returned, the latest kept — [`DEFAULT_LIMIT`] when absent, at most [`MAX_LIMIT`] |
//!
//! Lines come back as written, full timestamps and all, oldest first: what a
//! reader greps for is the text, and the file is already its own best format.
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

#[derive(Debug, Default, Deserialize)]
pub struct LogQuery {
    pub grep: Option<String>,
    pub level: Option<String>,
    pub tenant: Option<String>,
    pub since: Option<String>,
    pub limit: Option<usize>,
}

/// A search, checked.
pub struct Search {
    pattern: Option<Regex>,
    /// Index into [`LEVELS`] of the least level kept.
    min_level: usize,
    tenant: Option<String>,
    since: Option<String>,
    limit: usize,
}

/// What a search found.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct Found {
    /// The last `limit` matching lines, oldest first.
    pub lines: Vec<String>,
    /// Every line that matched in the files read, `lines` and those before it.
    pub matched: u64,
    /// Every line read.
    pub scanned: u64,
    /// How many files were read.
    pub files: usize,
    /// Every file that could hold a match was read: `false` when the search
    /// stopped at its `limit` with older files left unread, which may hold
    /// earlier matches.
    pub complete: bool,
}

/// What one file held for a search.
struct InFile {
    /// Its last matches, at most as many as were asked for, oldest first.
    kept: VecDeque<String>,
    matched: u64,
    scanned: u64,
    /// It holds a line stamped before `since`: every older file is wholly
    /// before it.
    passed_since: bool,
}

impl Search {
    /// `q`, checked: a pattern that does not compile or a level that does not
    /// exist is refused, not ignored.
    pub fn new(q: &LogQuery) -> Result<Self, String> {
        let pattern = match &q.grep {
            Some(p) => Some(Regex::new(p).map_err(|e| format!("`grep` is not a pattern: {e}"))?),
            None => None,
        };
        let min_level = match &q.level {
            Some(l) => LEVELS
                .iter()
                .position(|x| x.eq_ignore_ascii_case(l))
                .ok_or_else(|| format!("`level` is one of {}", LEVELS.join(", ")))?,
            None => 0,
        };
        Ok(Self {
            pattern,
            min_level,
            tenant: q.tenant.clone(),
            since: q.since.clone(),
            limit: q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT),
        })
    }

    /// Search `files`, given oldest first, newest first: the last `limit`
    /// matches, read no further back than they need.
    pub fn run(&self, bus: &LogBus, files: &[PathBuf]) -> io::Result<Found> {
        let mut newest_first: Vec<VecDeque<String>> = Vec::new();
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
            lines: newest_first.into_iter().rev().flatten().collect(),
            matched,
            scanned,
            files: read,
            complete,
        })
    }

    /// The last `want` matches in the file at `path`.
    fn in_file(&self, bus: &LogBus, path: &Path, want: usize) -> io::Result<InFile> {
        let mut found = InFile {
            kept: VecDeque::with_capacity(want),
            matched: 0,
            scanned: 0,
            passed_since: false,
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
            found.scanned += 1;
            if let Some(first) = text.split_whitespace().next() {
                if first.as_bytes()[0].is_ascii_digit() {
                    stamp = Some(first.to_string());
                }
            }
            let Some(at) = &stamp else {
                unstamped.push(text.to_string());
                continue;
            };
            for head in unstamped.drain(..) {
                self.take(bus, &head, at, want, &mut found);
            }
            self.take(bus, text, at, want, &mut found);
        }
        for head in unstamped {
            self.take(bus, &head, "", want, &mut found);
        }
        Ok(found)
    }

    /// Keep `text`, stamped `at`, in `found` when it matches.
    fn take(&self, bus: &LogBus, text: &str, at: &str, want: usize, found: &mut InFile) {
        if self.since.as_deref().is_some_and(|s| at < s) {
            found.passed_since = true;
        }
        if !self.admits(bus, text, at) {
            return;
        }
        found.matched += 1;
        if found.kept.len() == want {
            found.kept.pop_front();
        }
        found.kept.push_back(text.to_string());
    }

    fn admits(&self, bus: &LogBus, text: &str, stamp: &str) -> bool {
        if self.since.as_deref().is_some_and(|s| stamp < s) {
            return false;
        }
        if self.pattern.as_ref().is_some_and(|p| !p.is_match(text)) {
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
    /// first.
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
            level: Some("LOUD".into()),
            ..LogQuery::default()
        };
        assert_eq!(
            Search::new(&bad).err().unwrap(),
            "`level` is one of TRACE, DEBUG, INFO, WARN, ERROR"
        );
    }
}
