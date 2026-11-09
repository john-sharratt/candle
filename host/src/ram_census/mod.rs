//! This process's committed host memory, allocation by allocation: walked every
//! 30 s through the first ten minutes and every ten minutes after, logged as
//! the `host::ram` line, and kept as the latest [`RamReading`] for
//! `/v1/memory`.
//!
//! **The commit charge is what fails first, and nothing else reported it.** On
//! WhiteWorld (2026-10-20) motherd held ~130 GB committed against a 17 GB
//! working set, the machine's commit stood 0.5 GB under its limit, and the
//! process aborted three times in twenty minutes on an allocation it could not
//! commit — with no panic, since an allocation failure aborts without one, and
//! stderr detached. The pinned experts and the warm tier explained about 77 GB
//! of it; this names the rest.
//!
//! **Served from the last walk, never walked for a request.** A walk visits
//! every region of the address space; `/v1/memory` is read by the MCP's
//! `member_memory`, which answers within the request-path budget, so the page
//! reads the reading the census thread last stored.

mod reading;

pub use reading::{Bucket, Bytes, Held, RamReading, LARGEST};

use std::sync::Arc;
use std::thread::{sleep, Builder};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwapOption;
use candle::vram::process_ram::ProcessRam;

/// The latest reading; empty until the first walk.
static LATEST: ArcSwapOption<RamReading> = ArcSwapOption::const_empty();

/// How many allocations the log line names: the reading keeps more.
const LOGGED: usize = 12;

/// How long the walk runs at its first pace, and the two paces.
const EARLY: Duration = Duration::from_secs(600);
const EARLY_EVERY: Duration = Duration::from_secs(30);
const LATER_EVERY: Duration = Duration::from_secs(600);

/// The latest reading, if a walk has run.
pub fn latest() -> Option<Arc<RamReading>> {
    LATEST.load_full()
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The `host::ram` line for reading `r`.
fn log(r: &RamReading) {
    let top: Vec<String> = r
        .largest
        .iter()
        .take(LOGGED)
        .map(|a| {
            format!(
                "{} {} {:.2}/{:.2} GiB{}",
                a.base,
                a.kind,
                gib(a.committed),
                gib(a.resident),
                a.name.as_deref().map_or(String::new(), |n| format!(" {n}"))
            )
        })
        .collect();
    let buckets: Vec<String> = r
        .private_buckets
        .iter()
        .filter(|b| b.committed >= 1 << 30)
        .map(|b| {
            format!(
                "<={:.2}GiB x{} {:.2}GiB",
                gib(b.upper),
                b.count,
                gib(b.committed)
            )
        })
        .collect();
    tracing::info!(
        target: "host::ram",
        private_committed_gib = format!("{:.2}", gib(r.private.committed)),
        private_resident_gib = format!("{:.2}", gib(r.private.resident)),
        mapped_committed_gib = format!("{:.2}", gib(r.mapped.committed)),
        image_committed_gib = format!("{:.2}", gib(r.image.committed)),
        largest = %top.join(" | "),
        private_buckets = %buckets.join(" | "),
        "process committed memory"
    );
}

/// Start the census thread.
pub fn spawn() {
    let spawned = Builder::new().name("ram-census".into()).spawn(|| {
        let started = Instant::now();
        loop {
            if let Some(ram) = ProcessRam::capture() {
                let r = RamReading::of(&ram, now_unix_ms());
                log(&r);
                LATEST.store(Some(Arc::new(r)));
            }
            sleep(if started.elapsed() < EARLY {
                EARLY_EVERY
            } else {
                LATER_EVERY
            });
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("could not start the RAM census: {e}");
    }
}
