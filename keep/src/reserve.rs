//! The machine's reservation (`docs/house_modes.md` §4): while a process keep
//! does not run says `refuse` on its status — one of `keep.yaml`'s `guards:`,
//! `zenj` while rekordbox is open — the machine is **reserved**. Nothing is
//! switched, restarted, stopped, started or tried on it unless forced, and
//! every service marked `yields: true` is held down, so the machine's memory,
//! cores and card are the guard's.
//!
//! **A reservation outlasts its guard by [`RELEASE`].** It ends only once every
//! guard has been clear that long: a guard that crashes and is restarted in a
//! moment must not let a service the size of motherd start under it, nor a
//! deploy land in the gap. A guard that cannot be read is clear — a process not
//! answering is reserving nothing — which is what makes a machine with no
//! guard running exactly a machine with no guard.
//!
//! [`watch`] reads the guards every [`POLL`] on a thread of its own; everything
//! else asks the [`Reservation`] what it last saw, so no request waits on a
//! guard's socket.

use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use crate::guard::{self, FORCE_HEADER};

/// How long every guard must have been clear before the reservation ends.
pub const RELEASE: Duration = Duration::from_secs(10 * 60);

/// How often the guards are read.
pub const POLL: Duration = Duration::from_secs(1);

#[derive(Default)]
struct State {
    /// What the guard last said while it refused; `None` while not reserved.
    why: Option<String>,
    /// When every guard was first seen clear again, while still reserved.
    clear_since: Option<Instant>,
}

/// Whether this machine is reserved, as [`watch`] last read its guards.
#[derive(Default)]
pub struct Reservation {
    state: Mutex<State>,
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

/// Read `guards` every [`POLL`] into `reservation`, on a thread of its own,
/// for as long as keep runs. With no guards there is nothing to read and the
/// machine is never reserved.
pub fn watch(guards: Vec<String>, reservation: Arc<Reservation>) {
    if guards.is_empty() {
        return;
    }
    let spawned = thread::Builder::new()
        .name("keep-reservation".into())
        .spawn(move || {
            let mut was = false;
            loop {
                let refusing = guards.iter().find_map(|addr| guard::own_reason(addr));
                reservation.observe(refusing.clone(), Instant::now());
                let now = reservation.holds();
                if now != was {
                    match now {
                        true => eprintln!(
                            "keep: this machine is reserved: {}",
                            refusing.as_deref().unwrap_or("")
                        ),
                        false => eprintln!("keep: this machine's reservation is released"),
                    }
                    was = now;
                }
                thread::sleep(POLL);
            }
        });
    if let Err(e) = spawned {
        eprintln!("keep: the guards are not watched: {e}");
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
    }

    #[test]
    fn minutes_round_up() {
        assert_eq!(minutes(Duration::from_secs(1)), "1 minute");
        assert_eq!(minutes(Duration::from_secs(60)), "1 minute");
        assert_eq!(minutes(Duration::from_secs(61)), "2 minutes");
        assert_eq!(minutes(Duration::ZERO), "0 minutes");
    }
}
