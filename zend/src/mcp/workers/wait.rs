//! Waits on workers for a bounded time, and reports each step they take while the
//! caller waits.
//!
//! The wait subscribes before it reads, so a change between the read and the wait
//! still wakes it. It returns each waited worker as it stands once one of them stops
//! working or the wait runs out. A worker's line is said again once it goes stale,
//! since another member can change its record without this process hearing of it.
//! A journal entry is said once, because it never changes.

use super::*;

/// What a read of the waited workers says: per worker still running, what its
/// turn has done, keyed by its path; per journal entry it has written, the
/// entry's summary line, keyed by its address. Each with the line, and the
/// step it tells — the line less its elapsed time, which alone is no step.
pub fn progress_lines(
    read: &[(String, Option<Worker>)],
    now: u64,
) -> Vec<(String, String, String)> {
    let mut lines = Vec::new();
    for w in read.iter().filter_map(|(_, w)| w.as_ref()) {
        if !w.state.working() {
            continue;
        }
        let path = w.path();
        // The step is the sentence with no time passed: what the turn did.
        let step = w
            .activity
            .as_ref()
            .map_or_else(|| Activity::default().sentence(0), |a| a.sentence(0));
        lines.push((path.clone(), step, format!("{path} — {}", w.doing(now))));
        for e in &w.entries {
            let at = format!("{path}#{}", e.id);
            let line = format!("{at} — {}", summary_line(&e.text, 200));
            lines.push((at, String::new(), line));
        }
    }
    lines
}

/// Wait up to `wait` for any of `wanted` — `(thread, worker)` pairs of
/// `owner`'s — to stop working; every one of them as it then stands, `None`
/// for a name that is not a worker. Each step the waited workers take while it
/// waits is said on `progress`, once.
pub async fn wait(
    session: &ZendSession,
    owner: &str,
    wanted: &[(String, String)],
    wait: Duration,
    progress: Option<&UnboundedSender<String>>,
) -> Result<Vec<(String, Option<Worker>)>, String> {
    let records = session.shared_substrate().ok_or_else(loading)?;
    let deadline = Instant::now() + wait;
    let mut told: HashMap<String, (String, Instant)> = HashMap::new();
    loop {
        // Subscribe before reading, so a change between the read and the wait
        // still wakes it.
        let mut changed = session.mcp().watch();
        let now = now_ms();
        let read: Vec<(String, Option<Worker>)> = wanted
            .iter()
            .map(|(thread, name)| {
                let path = format!("task:{thread}/{name}");
                (
                    path,
                    threads::worker(&records, owner, thread, name).map(|w| w.observed(now)),
                )
            })
            .collect();
        let any_settled = read
            .iter()
            .any(|(_, w)| w.as_ref().is_none_or(|w| !w.state.working()));
        let left = deadline.saturating_duration_since(Instant::now());
        if any_settled || left.is_zero() {
            return Ok(read);
        }
        if let Some(progress) = progress {
            for (key, step, line) in progress_lines(&read, now) {
                // A worker's line is said again once it is stale; an entry's
                // never changes, so it is said once.
                let fresh = told.get(&key).is_some_and(|(said, at)| {
                    *said == step && (step.is_empty() || at.elapsed() < PROGRESS_AGAIN)
                });
                if !fresh {
                    told.insert(key, (step, Instant::now()));
                    let _ = progress.send(line);
                }
            }
        }
        // Wake on a change, and at least every [`PROGRESS_AGAIN`]: a task
        // another member runs changes the record without bumping this
        // process, and a stale line is said again.
        let nap = left.min(PROGRESS_AGAIN);
        let _ = tokio::time::timeout(nap, changed.changed()).await;
    }
}
