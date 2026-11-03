//! The disk sweep: when a checkout's volume runs low, take back what its
//! build outputs hold — and only what is worth taking.
//!
//! It runs as a run takes its checkout ([`crate::jobs`]), the moment a build is
//! about to add to a disk that may be filling. When the volume of a checkout's
//! `target` folder has less free than its bar ([`Bars::floor`]):
//!
//! 1. **Prune** every checkout on a low volume whose stale build generations
//!    ([`target_prune`]) would bring back at least the worth bar. A prune is
//!    safe beside a build: it leaves alone anything written in the last two
//!    minutes, and keeps the two newest generations of every artifact.
//! 2. **Clean** — remove the `target` folder, as `cargo clean` does — only
//!    while a volume is still low after the prune, only a checkout no run is
//!    using, largest first, and only one whose build outputs are at least the
//!    worth bar. A clean costs the next build its whole cache, so it is the
//!    second step, never the first.
//!
//! Every step it takes, and every one it declines and why, is a line of what
//! it returns, for the agent's log.

use std::fs;
use std::path::{Path, PathBuf};

use target_prune::prune_target;

pub mod bars;
mod size;

pub use bars::Bars;

use bars::gb;
use size::dir_bytes;

/// One checkout's build outputs, as a sweep sees them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// The checkout's name.
    pub name: String,
    /// Its `target` folder.
    pub dir: PathBuf,
    /// A run is using the checkout, so its outputs may not be cleaned.
    pub held: bool,
}

/// A volume's `(free, total)` bytes, for the directory it holds.
pub type Volume<'a> = &'a dyn Fn(&Path) -> Option<(u64, u64)>;

/// Whether `t`'s volume is low, as `volume` measures it now.
fn low(t: &Target, bars: &Bars, volume: Volume) -> Option<(u64, u64)> {
    volume(&t.dir).filter(|(free, total)| bars.low(*free, *total))
}

/// Sweep `targets` against `bars`, measuring volumes with `volume`: what it
/// did and what it declined, a line each — none when no volume is low.
pub fn sweep(targets: &[Target], bars: &Bars, volume: Volume) -> Vec<String> {
    let present: Vec<&Target> = targets.iter().filter(|t| t.dir.is_dir()).collect();
    let mut lines = Vec::new();
    for t in &present {
        let Some((free, total)) = low(t, bars, volume) else {
            continue;
        };
        if lines.is_empty() {
            lines.push(format!(
                "the disk is low: {} free of {}, under the {} bar",
                gb(free),
                gb(total),
                gb(bars.floor(total))
            ));
        }
        let stale = prune_target(&t.dir, true).freed;
        if bars.worth(stale) {
            let pruned = prune_target(&t.dir, false);
            lines.push(format!(
                "pruned {}: {} back, {} stale files removed",
                t.name,
                gb(pruned.freed),
                pruned.removed
            ));
        } else {
            lines.push(format!(
                "{}: a prune would bring back {}, under the {} worth it",
                t.name,
                gb(stale),
                gb(bars.worth)
            ));
        }
    }
    if lines.is_empty() {
        return lines;
    }
    let mut sized: Vec<(&Target, u64)> = present
        .iter()
        .filter(|t| low(t, bars, volume).is_some())
        .map(|t| (*t, dir_bytes(&t.dir)))
        .collect();
    sized.sort_by_key(|(_, bytes)| std::cmp::Reverse(*bytes));
    for (t, bytes) in sized {
        if low(t, bars, volume).is_none() {
            continue;
        }
        if t.held {
            lines.push(format!(
                "{}: still low, but a run is using it, so its {} is not cleaned",
                t.name,
                gb(bytes)
            ));
        } else if !bars.worth(bytes) {
            lines.push(format!(
                "{}: still low, but its build outputs are {}, under the {} worth a clean",
                t.name,
                gb(bytes),
                gb(bars.worth)
            ));
        } else {
            match fs::remove_dir_all(&t.dir) {
                Ok(()) => lines.push(format!("cleaned {}: {} back", t.name, gb(bytes))),
                Err(e) => lines.push(format!("{}: the clean failed: {e}", t.name)),
            }
        }
    }
    if let Some((free, total)) = present.first().and_then(|t| volume(&t.dir)) {
        lines.push(format!("now {} free of {}", gb(free), gb(total)));
    }
    lines
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fs::File;
    use std::slice;
    use std::time::{Duration, SystemTime};

    use super::*;

    /// A `target` with `generations` of one artifact of `bytes` each, all
    /// written an hour ago, so a prune may take the stale ones.
    fn target(root: &Path, name: &str, generations: usize, bytes: usize) -> Target {
        let deps = root.join(name).join("target/debug/deps");
        fs::create_dir_all(&deps).unwrap();
        let old = SystemTime::now() - Duration::from_secs(3600);
        for g in 0..generations {
            let path = deps.join(format!("libthing-{:016x}.rlib", g + 1));
            fs::write(&path, vec![0u8; bytes]).unwrap();
            let when = old + Duration::from_secs(g as u64);
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(when)
                .unwrap();
        }
        Target {
            name: name.into(),
            dir: root.join(name).join("target"),
            held: false,
        }
    }

    fn bars(worth: u64) -> Bars {
        Bars {
            low_free: 1_000,
            low_free_percent: 0,
            worth,
        }
    }

    /// **A volume with room is not swept**: nothing is measured or removed.
    #[test]
    fn a_volume_with_room_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let t = target(root.path(), "candle", 4, 100);
        let lines = sweep(slice::from_ref(&t), &bars(1), &|_| Some((5_000, 10_000)));
        assert!(lines.is_empty(), "{lines:?}");
        assert_eq!(dir_bytes(&t.dir), 400);
    }

    /// **A low volume is pruned first**: the stale generations go, the two
    /// newest stay, and a prune that brings it back above the bar ends it.
    #[test]
    fn a_low_volume_is_pruned_and_a_prune_that_suffices_ends_it() {
        let root = tempfile::tempdir().unwrap();
        let t = target(root.path(), "candle", 5, 100);
        let free = Cell::new(500u64);
        let volume = |_: &Path| {
            let f = free.get();
            free.set(5_000);
            Some((f, 10_000))
        };
        let lines = sweep(slice::from_ref(&t), &bars(200), &volume);
        assert_eq!(
            lines[0],
            "the disk is low: 0.0 GB free of 0.0 GB, under the 0.0 GB bar"
        );
        assert!(
            lines[1].starts_with("pruned candle:") && lines[1].ends_with("3 stale files removed"),
            "{lines:?}"
        );
        assert_eq!(dir_bytes(&t.dir), 200, "the two newest generations stay");
        assert!(lines.iter().all(|l| !l.starts_with("cleaned")), "{lines:?}");
    }

    /// **A prune not worth doing is declined**, and a volume still low is
    /// cleaned only where the outputs are worth it and no run is using them.
    #[test]
    fn a_clean_is_the_second_step_and_spares_a_held_checkout() {
        let root = tempfile::tempdir().unwrap();
        let idle = target(root.path(), "site", 2, 300);
        let mut busy = target(root.path(), "candle", 2, 400);
        busy.held = true;
        let small = target(root.path(), "notes", 1, 10);
        let lines = sweep(
            &[idle.clone(), busy.clone(), small.clone()],
            &bars(100),
            &|_| Some((500, 10_000)),
        );
        assert!(
            lines.contains(
                &"site: a prune would bring back 0.0 GB, under the 0.0 GB worth it".to_string()
            ),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("candle: still low, but a run is using it")),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.starts_with("cleaned site:")),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("notes: still low, but its build outputs are")),
            "{lines:?}"
        );
        assert!(!idle.dir.exists());
        assert_eq!(dir_bytes(&busy.dir), 800);
        assert_eq!(dir_bytes(&small.dir), 10);
    }

    /// **The largest idle outputs are cleaned first, and the clean stops
    /// once the volume has room.**
    #[test]
    fn the_largest_is_cleaned_first_and_the_clean_stops_with_room() {
        let root = tempfile::tempdir().unwrap();
        let big = target(root.path(), "big", 1, 900);
        let other = target(root.path(), "other", 1, 300);
        let cleaned = Cell::new(false);
        let volume = |_: &Path| Some((if cleaned.get() { 5_000 } else { 500 }, 10_000));
        let probe = big.dir.clone();
        let lines = sweep(&[other.clone(), big.clone()], &bars(100), &|p| {
            if !probe.exists() {
                cleaned.set(true);
            }
            volume(p)
        });
        assert!(
            lines.iter().any(|l| l.starts_with("cleaned big:")),
            "{lines:?}"
        );
        assert!(other.dir.exists(), "{lines:?}");
    }
}
