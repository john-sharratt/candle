//! The machine's reservation (`docs/house_modes.md` §4): while a guard — a
//! process this machine runs for something other than the cluster, `zenj`
//! while rekordbox is open — says `refuse` on its status, the machine is
//! **reserved**, and the heavy work on it steps aside for the guard.
//!
//! What steps aside is each process's to say, from [`machine`]: keep refuses
//! switches, restarts, stops and trials; a model member drains while another
//! member is ready for every app it serves, and stays serving when none is —
//! a party is run by Zen, so Zen is never taken down for one; its agent takes
//! no builds or tests; its tenants read and calibrate nothing.
//!
//! **A reservation outlasts its guard by [`RELEASE`].** It ends only once every
//! guard has been clear that long: a guard that crashes and is restarted in a
//! moment must not let a build or a deploy land in the gap. A guard that cannot
//! be read is clear — a process not answering is reserving nothing — which is
//! what makes a machine with no guard running exactly a machine with no guard.
//!
//! | Module | |
//! |---|---|
//! | [`guard`] | a guard's status, read |
//! | [`watch`] | the thread that reads the guards into the reservation |
//! | [`step_aside`] | whether a model member drains for it |

pub mod guard;
pub mod step_aside;
pub mod watch;

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::upgrade::switch::FORCE_HEADER;

/// How long every guard must have been clear before the reservation ends.
pub const RELEASE: Duration = Duration::from_secs(10 * 60);

#[derive(Default)]
struct State {
    /// What the guard last said while it refused; `None` while not reserved.
    why: Option<String>,
    /// When every guard was first seen clear again, while still reserved.
    clear_since: Option<Instant>,
}

/// Whether this machine is reserved, as [`watch::watch`] last read its guards.
#[derive(Default)]
pub struct Reservation {
    state: Mutex<State>,
}

/// This process's reservation: never reserved until [`watch::watch`] is given
/// a guard to read. One per process, so the parts that step aside for it —
/// the agent, the tenants, the drain — read the same one without being handed
/// it.
pub fn machine() -> Arc<Reservation> {
    static MACHINE: OnceLock<Arc<Reservation>> = OnceLock::new();
    Arc::clone(MACHINE.get_or_init(|| Arc::new(Reservation::new())))
}

impl Reservation {
    pub fn new() -> Reservation {
        Reservation::default()
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One reading of the guards at `now`: the first one's reason for
    /// refusing, or `None` when every one is clear.
    pub fn observe(&self, refusing: Option<String>, now: Instant) {
        let mut s = self.state();
        match refusing {
            Some(why) => {
                s.why = Some(why);
                s.clear_since = None;
            }
            None if s.why.is_some() => {
                let since = *s.clear_since.get_or_insert(now);
                if now.duration_since(since) >= RELEASE {
                    *s = State::default();
                }
            }
            None => {}
        }
    }

    /// Whether the machine is reserved.
    pub fn holds(&self) -> bool {
        self.state().why.is_some()
    }

    /// What the guard said, while the machine is reserved.
    pub fn why(&self) -> Option<String> {
        self.state().why.clone()
    }

    /// Why a request is refused at `now` because the machine is reserved,
    /// worded with the way past it — `None` when it is not reserved.
    pub fn refusal(&self, now: Instant) -> Option<String> {
        let s = self.state();
        let why = s.why.as_deref()?;
        Some(match s.clear_since {
            None => format!(
                "refused: this machine is reserved — {why}. Wait for it to clear, or send \
                 `{FORCE_HEADER}: 1` to do it anyway\n"
            ),
            Some(since) => {
                let left = RELEASE.saturating_sub(now.duration_since(since));
                format!(
                    "refused: this machine was reserved — {why} — and is released once it has \
                     been clear for {} minutes: {} to go. Wait, or send `{FORCE_HEADER}: 1` to \
                     do it anyway\n",
                    RELEASE.as_secs() / 60,
                    minutes(left)
                )
            }
        })
    }
}

/// `d` as whole minutes, rounded up, in words.
fn minutes(d: Duration) -> String {
    match d.as_secs().div_ceil(60) {
        1 => "1 minute".to_string(),
        n => format!("{n} minutes"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A reservation begins with the guard's refusal and ends only once it
    /// has been clear for [`RELEASE`]**: a refusal inside that time starts the
    /// count again, and a guard never seen refusing reserves nothing.
    #[test]
    fn a_reservation_outlasts_its_guard_by_the_release() {
        let r = Reservation::new();
        let t0 = Instant::now();
        r.observe(None, t0);
        assert!(!r.holds(), "never refused");
        assert_eq!(r.refusal(t0), None);

        r.observe(Some("rekordbox is open".into()), t0);
        assert!(r.holds());
        assert_eq!(r.why().as_deref(), Some("rekordbox is open"));
        assert_eq!(
            r.refusal(t0).as_deref(),
            Some(
                "refused: this machine is reserved — rekordbox is open. Wait for it to clear, or \
                 send `X-Keep-Force: 1` to do it anyway\n"
            )
        );

        // The guard goes quiet — crashed, or rekordbox closed.
        let t1 = t0 + Duration::from_secs(60);
        r.observe(None, t1);
        assert!(r.holds(), "still reserved in the release");
        assert_eq!(
            r.refusal(t1 + Duration::from_secs(30)).as_deref(),
            Some(
                "refused: this machine was reserved — rekordbox is open — and is released once \
                 it has been clear for 10 minutes: 10 minutes to go. Wait, or send \
                 `X-Keep-Force: 1` to do it anyway\n"
            )
        );

        // Back inside the release: the count starts again.
        let t2 = t1 + RELEASE - Duration::from_secs(1);
        r.observe(Some("rekordbox is open".into()), t2);
        r.observe(None, t2 + Duration::from_secs(1));
        r.observe(None, t2 + RELEASE);
        assert!(r.holds(), "the release counts from the last refusal");

        r.observe(None, t2 + Duration::from_secs(1) + RELEASE);
        assert!(!r.holds(), "clear for the whole release");
        assert_eq!(r.refusal(t2 + RELEASE * 2), None);
        assert_eq!(r.why(), None);
    }

    #[test]
    fn minutes_round_up() {
        assert_eq!(minutes(Duration::from_secs(1)), "1 minute");
        assert_eq!(minutes(Duration::from_secs(60)), "1 minute");
        assert_eq!(minutes(Duration::from_secs(61)), "2 minutes");
        assert_eq!(minutes(Duration::ZERO), "0 minutes");
    }

    /// **A process with no guard is never reserved.**
    #[test]
    fn the_machine_starts_unreserved() {
        assert!(!machine().holds());
    }
}
