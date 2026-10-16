//! `GET /v1/logs` — this member's log searched: every rotated file and the
//! active one, oldest first, read off disk a line at a time. The gateway asks
//! every member and merges their answers (`Merge::Envelope`), so one request
//! searches the cluster.
//!
//! | Parameter | |
//! |---|---|
//! | `grep` | a regular expression the whole line must match |
//! | `level` | the least level kept: `TRACE` … `ERROR` |
//! | `tenant` | only lines one tenant wrote (`zend`, `npcd`) |
//! | `since` | only lines stamped at or after this, an RFC 3339 prefix such as `2026-10-16T00:20` |
//! | `limit` | the most matches returned, the latest kept — [`DEFAULT_LIMIT`] when absent, at most [`MAX_LIMIT`] |
//!
//! Lines come back as written, full timestamps and all: what a reader greps
//! for is the text, and the file is already its own best format. Only the
//! last `limit` matches are held while the files are read, so a search costs
//! its answer's memory, not the log's.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;
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
    /// Every line that matched, `lines` and those before it.
    pub matched: u64,
    /// Every line read.
    pub scanned: u64,
    /// How many files were read.
    pub files: usize,
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

    /// Read `files` in order and keep the last matches.
    pub fn run(&self, bus: &LogBus, files: &[PathBuf]) -> io::Result<Found> {
        let mut kept = VecDeque::with_capacity(self.limit);
        let (mut matched, mut scanned) = (0u64, 0u64);
        // A line without a stamp — the rest of a message that spans lines —
        // is as old as the stamped line before it.
        let mut stamp = String::new();
        let mut buf = Vec::new();
        for path in files {
            let mut reader = BufReader::new(File::open(path)?);
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
                scanned += 1;
                if let Some(first) = text.split_whitespace().next() {
                    if first.as_bytes()[0].is_ascii_digit() {
                        stamp = first.to_string();
                    }
                }
                if !self.admits(bus, text, &stamp) {
                    continue;
                }
                matched += 1;
                if kept.len() == self.limit {
                    kept.pop_front();
                }
                kept.push_back(text.to_string());
            }
        }
        Ok(Found {
            lines: kept.into(),
            matched,
            scanned,
            files: files.len(),
        })
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
        assert_eq!((f.matched, f.scanned, f.files), (2, 5, 2));
    }

    /// Only the latest `limit` matches are kept; `matched` still counts them all.
    #[test]
    fn the_limit_keeps_the_latest_matches() {
        let f = run(LogQuery {
            grep: Some("catch-up".into()),
            limit: Some(1),
            ..LogQuery::default()
        });
        assert_eq!(
            f.lines,
            ["2026-10-16T00:04:00.0Z ERROR cluster::replicate::tick: a catch-up went quiet: asking again"]
        );
        assert_eq!(f.matched, 2);
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
