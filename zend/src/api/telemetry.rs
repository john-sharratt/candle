//! `GET /v1/telemetry` — live wave-timing + KV-pressure series for the
//! telemetry dashboard (`/telemetry.html`).
//!
//! The scheduler already emits per-wave summaries, KV-pool snapshots, pool
//! format breakdowns and hot→warm migration timings to the daemon's tracing
//! stream. Rather than stand up a parallel in-memory metrics pipeline, this
//! endpoint tail-parses the rolling `.substrate/zend.log` (the same source the
//! offline dashboard used) and returns a compact JSON time-series the page
//! renders with Canvas. Only the tail is read, so cost is bounded regardless of
//! run length; timestamps are normalised to elapsed seconds from the first
//! sample in the window.
//!
//! Requires the daemon to be logging at `debug` (the `-v` flag): the KV-pool
//! and migration lines are `tracing::debug!`. Without it only the `info`-level
//! wave series is present and the pressure panels read empty.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use axum::{extract::State, Json};
use regex::Regex;
use serde::Serialize;

use crate::session::ZendSession;

/// Bytes read off the tail of the active log per request — a rolling window of
/// recent waves. The active file is size-capped at 32 MiB (see `log_file`), so
/// this reads at most a fraction of it.
const TAIL_BYTES: u64 = 6 * 1024 * 1024;

#[derive(Serialize, Default, Clone)]
pub struct Telemetry {
    waves: Vec<Wave>,
    phases: Vec<Phase>,
    fmt: Vec<Fmt>,
    migrate: Vec<Migrate>,
    host: Host,
    /// `true` when the log carried the debug-level pressure lines (pool +
    /// migration). `false` means the daemon is running without `-v`.
    has_pressure: bool,
}

#[derive(Serialize, Clone)]
struct Wave { t: f64, ws: f64, fwd: f64, tok: f64, budget: f64, used: f64, backlog: f64 }
#[derive(Serialize, Clone)]
struct Phase { t: f64, prefill: f64, snap: f64, sig: f64, flush: f64, presolve: f64, pkern: f64, pasm: f64 }
#[derive(Serialize, Clone)]
struct Fmt { t: f64, fa: f64, fm: f64, fl: f64, qa: f64, qm: f64, ql: f64 }
#[derive(Serialize, Clone)]
struct Migrate { t: f64, res: f64, mib: f64, migrate: f64, quant: f64, copy: f64, select: f64, convert: f64, install: f64, total: f64 }
#[derive(Serialize, Default, Clone)]
struct Host { free_gib: f64, total_gib: f64 }

/// Cached log-derived body keyed by the active log's `(size, mtime)` — so
/// repeated polls of an *unchanged* log (idle daemon, or several dashboards)
/// skip the 6 MiB read + reparse. Host RAM is never cached — it's read fresh on
/// every request since it moves independently of the log.
struct CacheEntry {
    size: u64,
    mtime: u64,
    body: Telemetry,
}
static CACHE: OnceLock<Mutex<Option<CacheEntry>>> = OnceLock::new();

pub async fn telemetry(State(session): State<Arc<ZendSession>>) -> Json<Telemetry> {
    let path = session.tracing_log_path();
    // The read + parse is blocking I/O + CPU; keep it off the async runtime's
    // worker threads (a 6 MiB read must never stall in-flight requests).
    let body = tokio::task::spawn_blocking(move || build_body(&path))
        .await
        .unwrap_or_default();
    Json(body)
}

/// Assemble the telemetry body: reuse the cache when the log is byte-for-byte
/// unchanged, else re-read + reparse; always stamp fresh host RAM.
fn build_body(path: &Path) -> Telemetry {
    let (size, mtime) = std::fs::metadata(path)
        .map(|m| {
            let mt = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (m.len(), mt)
        })
        .unwrap_or((0, 0));
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    let mut body = {
        let hit = {
            let guard = cache.lock().unwrap();
            match guard.as_ref() {
                Some(c) if c.size == size && c.mtime == mtime => Some(c.body.clone()),
                _ => None,
            }
        };
        hit.unwrap_or_else(|| {
            let parsed = parse(&read_tail(path, TAIL_BYTES).unwrap_or_default());
            *cache.lock().unwrap() = Some(CacheEntry { size, mtime, body: parsed.clone() });
            parsed
        })
    };
    body.host = host_memory();
    body
}

/// Read the last `cap` bytes of `path` as (lossy) UTF-8, discarding the leading
/// partial line so parsing starts on a clean record.
fn read_tail(path: &Path, cap: u64) -> Option<String> {
    let mut f = File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let start = len.saturating_sub(cap);
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::with_capacity(cap.min(len) as usize);
    f.read_to_end(&mut buf).ok()?;
    let mut s = String::from_utf8_lossy(&buf).into_owned();
    if start > 0 {
        if let Some(nl) = s.find('\n') {
            s.drain(..=nl);
        }
    }
    Some(s)
}

/// Seconds-of-day from a leading ISO timestamp (`…THH:MM:SS.ffffffZ`).
fn ts(line: &str) -> Option<f64> {
    let t = line.find('T')?;
    let rest = &line[t + 1..];
    let z = rest.find('Z')?;
    let mut it = rest[..z].split(':');
    let h: f64 = it.next()?.parse().ok()?;
    let m: f64 = it.next()?.parse().ok()?;
    let s: f64 = it.next()?.parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

/// Value of a `key=<number>` token (handles the spaced keys `tok total`,
/// `fwd avg`). Returns 0.0 when absent so a partly-formed line still parses.
fn num(line: &str, key: &str) -> f64 {
    let Some(i) = line.find(key) else { return 0.0 };
    let after = &line[i + key.len()..];
    let after = after.strip_prefix('=').unwrap_or(after);
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '.'))
        .unwrap_or(after.len());
    after[..end].parse().unwrap_or(0.0)
}

fn parse(text: &str) -> Telemetry {
    let mut out = Telemetry::default();
    let mut t0: Option<f64> = None;
    let mut days = 0.0;
    let mut prev = 0.0;
    for line in text.lines() {
        let Some(sod) = ts(line) else { continue };
        if sod + days * 86400.0 < prev - 60.0 {
            days += 1.0; // crossed midnight
        }
        let abs = sod + days * 86400.0;
        prev = abs;
        let base = *t0.get_or_insert(abs);
        let t = ((abs - base) * 100.0).round() / 100.0;

        if line.contains("prefill fwds=") {
            let ws = wave_seconds(line);
            out.waves.push(Wave {
                t, ws,
                // Leading space: the line also carries `tok/fwd avg=` and
                // `kv/fwd avg=`, so match the space-prefixed standalone one.
                fwd: num(line, " fwd avg"),
                tok: num(line, "tok total"),
                budget: num(line, "budget"),
                used: num(line, "used"),
                backlog: num(line, "backlog"),
            });
        } else if line.contains("wave phase breakdown") {
            out.phases.push(Phase {
                t,
                prefill: num(line, "prefill_ms"),
                snap: num(line, "seal_snapshot_ms"),
                sig: num(line, "seal_sig_ms"),
                flush: num(line, "seal_flush_ms"),
                presolve: num(line, "prov_resolve_ms"),
                pkern: num(line, "prov_kernel_ms"),
                pasm: num(line, "prov_assemble_ms"),
            });
        } else if line.contains("kv-pool fmt:") {
            if let Some(f) = fmt_line(line, t) {
                out.fmt.push(f);
                out.has_pressure = true;
            }
        } else if line.contains("pass timing") && line.contains("migrate_ms=") {
            out.migrate.push(Migrate {
                t,
                res: num(line, "residences"),
                mib: num(line, "mib"),
                migrate: num(line, "migrate_ms"),
                quant: num(line, "quantize_ms"),
                copy: num(line, "copy_ms"),
                select: num(line, "select_ms"),
                convert: num(line, "convert_ms"),
                install: num(line, "install_ms"),
                total: num(line, "total_ms"),
            });
            out.has_pressure = true;
        }
    }
    out
}

/// `wave 2.1s:` → 2.1.
fn wave_seconds(line: &str) -> f64 {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"wave ([0-9.]+)s").unwrap());
    re.captures(line)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok())
        .unwrap_or(0.0)
}

/// `float=Narenas/NMiB (live NMiB) quant=Narenas/NMiB (live NMiB)`.
fn fmt_line(line: &str, t: f64) -> Option<Fmt> {
    static FLOAT: OnceLock<Regex> = OnceLock::new();
    static QUANT: OnceLock<Regex> = OnceLock::new();
    let fr = FLOAT.get_or_init(|| Regex::new(r"float=(\d+)arenas/(\d+)MiB \(live (\d+)MiB\)").unwrap());
    let qr = QUANT.get_or_init(|| Regex::new(r"quant=(\d+)arenas/(\d+)MiB \(live (\d+)MiB\)").unwrap());
    let g = |c: &regex::Captures, i| c.get(i).unwrap().as_str().parse::<f64>().unwrap_or(0.0);
    let f = fr.captures(line)?;
    let q = qr.captures(line)?;
    Some(Fmt {
        t,
        fa: g(&f, 1), fm: g(&f, 2), fl: g(&f, 3),
        qa: g(&q, 1), qm: g(&q, 2), ql: g(&q, 3),
    })
}

fn host_memory() -> Host {
    use sysinfo::System;
    // Reuse one System across requests — refreshing memory on it is far cheaper
    // than allocating a fresh System per poll.
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();
    let mut sys = SYS.get_or_init(|| Mutex::new(System::new())).lock().unwrap();
    sys.refresh_memory();
    let gib = |b: u64| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    Host {
        free_gib: (gib(sys.available_memory()) * 10.0).round() / 10.0,
        total_gib: (gib(sys.total_memory()) * 10.0).round() / 10.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verbatim lines from a real `.substrate/zend.log` run.
    const LOG: &str = "\
2026-07-17T23:42:45.350440Z  INFO candle_conversation::scheduler: wave 2.1s: prefill fwds=16 seqs avg=1.0 max=1 tok/fwd avg=185 tok total=2953 kv/fwd avg=0 kv total=0 fwd avg=116ms | kv-vram budget=14579MiB used=57112MiB | backlog=145tok
2026-07-17T23:42:45.360000Z  INFO candle_conversation::scheduler::timing: wave phase breakdown drain_ms=0 prefill_ms=1628 seal_count=20 seal_snapshot_ms=68 seal_sig_ms=56 seal_flush_ms=113 prov_resolve_ms=42 prov_kernel_ms=5 prov_assemble_ms=7
2026-07-17T23:42:46.294092Z DEBUG candle_conversation::scheduler::run: kv-pool fmt: float=70arenas/1120MiB (live 1001MiB) quant=467arenas/7472MiB (live 3360MiB)
2026-07-17T23:42:47.788803Z DEBUG candle_conversation::persistence::tier: hot\u{2192}warm pass timing residences=308 mib=2314 n_layers=48 sync_pre_ms=3 migrate_ms=77540 quantize_ms=72352 select_ms=916 alloc_ms=505 convert_ms=284 copy_ms=5151 sync_post_ms=0 install_ms=238 total_ms=77782";

    #[test]
    fn parses_wave_fields_exactly() {
        let w = &parse(LOG).waves[0];
        assert_eq!(w.ws, 2.1);
        assert_eq!(w.fwd, 116.0);
        assert_eq!(w.tok, 2953.0);
        assert_eq!(w.budget, 14579.0);
        assert_eq!(w.used, 57112.0);
        assert_eq!(w.backlog, 145.0);
        assert_eq!(w.t, 0.0); // first sample defines the elapsed-time origin
    }

    #[test]
    fn parses_phase_fields_exactly() {
        let p = &parse(LOG).phases[0];
        assert_eq!(p.prefill, 1628.0);
        assert_eq!(p.snap, 68.0);
        assert_eq!(p.sig, 56.0);
        assert_eq!(p.flush, 113.0);
        assert_eq!(p.presolve, 42.0);
        assert_eq!(p.pkern, 5.0);
        assert_eq!(p.pasm, 7.0);
    }

    #[test]
    fn parses_pool_fmt_fields_exactly() {
        let f = &parse(LOG).fmt[0];
        assert_eq!((f.fa, f.fm, f.fl), (70.0, 1120.0, 1001.0));
        assert_eq!((f.qa, f.qm, f.ql), (467.0, 7472.0, 3360.0));
    }

    #[test]
    fn parses_migration_fields_exactly() {
        let m = &parse(LOG).migrate[0];
        assert_eq!(m.res, 308.0);
        assert_eq!(m.mib, 2314.0);
        assert_eq!(m.migrate, 77540.0);
        assert_eq!(m.quant, 72352.0);
        assert_eq!(m.copy, 5151.0);
        assert_eq!(m.total, 77782.0);
    }

    #[test]
    fn pressure_flag_and_relative_time() {
        let out = parse(LOG);
        assert!(out.has_pressure, "pool + migration lines set has_pressure");
        // Migration line is ~2.44s after the first wave sample.
        assert!((out.migrate[0].t - 2.44).abs() < 0.01, "t={}", out.migrate[0].t);
    }

    #[test]
    fn empty_input_yields_empty_series() {
        let out = parse("no timestamps here\njust noise");
        assert!(out.waves.is_empty() && out.migrate.is_empty() && !out.has_pressure);
    }
}
