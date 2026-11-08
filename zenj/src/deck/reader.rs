//! The deck reader's thread (§3.3): every few milliseconds, each deck's
//! tempo, position and loaded track read out of rekordbox's memory through
//! the offsets for its release, judged (`state::Judge`), and published for
//! the conductor, the brain and the screens. rekordbox restarted or not yet
//! running is a blind reader, retried each second; whether it runs at all is
//! what reserves the machine (`http::keep_status`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;

use super::memory::{MemError, Memory, Process};
use super::offsets::Offsets;
use super::state::{Judge, Raw, Snapshot, Tracks};

/// How often the decks are read.
pub const PERIOD: Duration = Duration::from_millis(5);
/// How long a reader waits to open rekordbox again.
const REOPEN: Duration = Duration::from_secs(1);
/// The executable read.
pub const EXE: &str = "rekordbox.exe";
/// The longest analysis path read.
const PATH_MAX: usize = 500;

/// One reading of every deck through `offsets`.
pub fn read(mem: &dyn Memory, offsets: &Offsets, decks: usize) -> Result<(Vec<Raw>, u8), MemError> {
    let master = mem.u8_at(&offsets.master)?;
    let mut raws = Vec::with_capacity(decks);
    for d in offsets.decks.iter().take(decks) {
        raws.push(Raw {
            bpm: mem.f32_at(&d.bpm)?,
            samples: mem.i64_at(&d.samples)?,
            anlz: mem.text_at(&d.anlz, PATH_MAX)?,
        });
    }
    Ok((raws, master))
}

/// Run the reader until `stop`, publishing each snapshot to `out`, and to
/// `running` whether rekordbox runs at all — open, whether or not its memory
/// reads.
pub fn run(
    offsets: Offsets,
    decks: usize,
    tracks: Arc<dyn Tracks + Send + Sync>,
    out: Arc<ArcSwap<Snapshot>>,
    running: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) {
    let mut judge = Judge::new();
    while !stop.load(Ordering::Relaxed) {
        let process = match Process::open(EXE) {
            Ok(p) => p,
            Err(e) => {
                running.store(e != MemError::NotRunning, Ordering::Relaxed);
                out.store(Arc::new(Snapshot::blind(e.to_string())));
                thread::sleep(REOPEN);
                continue;
            }
        };
        running.store(true, Ordering::Relaxed);
        tracing::info!(version = %offsets.version, "deck reader: reading rekordbox");
        let mut next = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            match read(&process, &offsets, decks) {
                Ok((raws, master)) => {
                    let s = judge.judge(&raws, master, Instant::now(), tracks.as_ref());
                    out.store(Arc::new(s));
                }
                Err(e) => {
                    tracing::warn!("deck reader: {e}");
                    out.store(Arc::new(Snapshot::blind(e.to_string())));
                    break;
                }
            }
            next += PERIOD;
            let now = Instant::now();
            if next > now {
                thread::sleep(next - now);
            } else {
                next = now;
            }
        }
        thread::sleep(REOPEN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::memory::fake::Image;
    use crate::deck::offsets::{parse, Chain, DeckChains, Offsets};

    fn chain(last: usize) -> Chain {
        Chain { hops: vec![], last }
    }

    /// **One reading takes every deck's chains in order**, through the same
    /// image a running rekordbox would present.
    #[test]
    fn a_reading_takes_each_decks_chains() {
        let mut m = Image::default();
        m.write(0x10, &[1]);
        m.write(0x100, &124.0f32.to_le_bytes());
        m.write(0x200, &88_200i64.to_le_bytes());
        // A path is read at its longest: the region holds that many bytes.
        let mut padded = b"/PIONEER/USBANLZ/x/ANLZ0000.DAT\0".to_vec();
        padded.resize(PATH_MAX, 0);
        m.write(0x300, &padded);
        let deck = DeckChains {
            bpm: chain(0x100),
            samples: chain(0x200),
            info: chain(0x300),
            anlz: chain(0x300),
            uuid: None,
        };
        let o = Offsets {
            version: "t".into(),
            master: chain(0x10),
            decks: vec![deck],
        };
        let (raws, master) = read(&m, &o, 2).unwrap();
        assert_eq!(master, 1);
        assert_eq!(
            raws,
            vec![Raw {
                bpm: 124.0,
                samples: 88_200,
                anlz: "/PIONEER/USBANLZ/x/ANLZ0000.DAT".into()
            }]
        );
        assert!(parse("").unwrap().is_empty());
    }
}
