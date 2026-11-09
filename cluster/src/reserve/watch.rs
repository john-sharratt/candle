//! The thread that reads the guards into the machine's reservation, every
//! [`POLL`]. Everything else asks the [`Reservation`] what it last saw, so no
//! request and no forward waits on a guard's socket.

use std::io;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use super::guard::reason_at;
use super::Reservation;

/// How often the guards are read.
pub const POLL: Duration = Duration::from_secs(1);

/// Read `guards` once into `reservation`, now: what a process does before it
/// starts anything, so nothing starts under a reservation already standing.
pub fn read_once(guards: &[String], reservation: &Reservation) {
    let refusing = guards.iter().find_map(|addr| reason_at(addr));
    reservation.observe(refusing, Instant::now());
}

/// Read `guards` every [`POLL`] into `reservation`, on a thread of its own,
/// for as long as the process runs, telling `changed` each time the machine
/// becomes reserved — with the guard's reason — or is released. With no
/// guards there is nothing to read and the machine is never reserved.
pub fn watch(
    guards: Vec<String>,
    reservation: Arc<Reservation>,
    changed: impl Fn(Option<&str>) + Send + 'static,
) -> io::Result<()> {
    if guards.is_empty() {
        return Ok(());
    }
    thread::Builder::new()
        .name("reservation".into())
        .spawn(move || {
            let mut was = false;
            loop {
                read_once(&guards, &reservation);
                let now = reservation.holds();
                if now != was {
                    changed(reservation.why().as_deref());
                    was = now;
                }
                thread::sleep(POLL);
            }
        })
        .map(|_| ())
}
