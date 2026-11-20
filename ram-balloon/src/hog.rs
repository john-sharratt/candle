//! The RAM the balloon holds: chunks of active memory a sweeper thread keeps
//! touching, and chunks of idle memory written once.
//!
//! Each chunk is [`CHUNK`] bytes, so a target is met to within one chunk and a
//! change frees or allocates whole chunks. A page is touched by writing one byte
//! of it: enough to make it resident, and to make the OS read it back from the
//! pagefile if it had been trimmed.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// One allocation: 256 MiB.
pub const CHUNK: u64 = 256 * 1024 * 1024;
const PAGE: usize = 4096;

type Chunk = Box<[u8]>;

/// A chunk of `CHUNK` bytes with every page written, or `None` when the
/// allocator refuses.
fn chunk() -> Option<Chunk> {
    let mut v: Vec<u8> = Vec::new();
    v.try_reserve_exact(CHUNK as usize).ok()?;
    v.resize(CHUNK as usize, 0);
    touch(&mut v, 1);
    Some(v.into_boxed_slice())
}

/// Write one byte of every page of `buf`.
fn touch(buf: &mut [u8], value: u8) {
    for page in buf.chunks_mut(PAGE) {
        page[0] = value;
    }
    std::hint::black_box(&buf);
}

/// Active and idle RAM, held to targets.
pub struct Hog {
    active: Arc<Mutex<Vec<Chunk>>>,
    idle: Vec<Chunk>,
    stop: Arc<AtomicBool>,
    sweeps: Arc<AtomicU64>,
    sweeper: Option<JoinHandle<()>>,
}

impl Hog {
    pub fn new() -> Self {
        let active: Arc<Mutex<Vec<Chunk>>> = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let sweeps = Arc::new(AtomicU64::new(0));
        let sweeper = {
            let (active, stop, sweeps) = (active.clone(), stop.clone(), sweeps.clone());
            std::thread::Builder::new()
                .name("balloon-sweeper".into())
                .spawn(move || {
                    let mut value = 2u8;
                    while !stop.load(Ordering::Relaxed) {
                        let mut held = active.lock().unwrap();
                        if held.is_empty() {
                            drop(held);
                            std::thread::sleep(Duration::from_millis(50));
                            continue;
                        }
                        for c in held.iter_mut() {
                            touch(c, value);
                        }
                        drop(held);
                        value = value.wrapping_add(1).max(2);
                        sweeps.fetch_add(1, Ordering::Relaxed);
                    }
                })
                .expect("the sweeper starts")
        };
        Self {
            active,
            idle: Vec::new(),
            stop,
            sweeps,
            sweeper: Some(sweeper),
        }
    }

    /// Hold `active` and `idle` bytes, to the chunk below each. Returns the
    /// bytes actually held: an allocation the machine refuses stops the growth.
    pub fn set(&mut self, active: u64, idle: u64) -> (u64, u64) {
        let want = |bytes: u64| (bytes / CHUNK) as usize;
        {
            let mut held = self.active.lock().unwrap();
            resize(&mut held, want(active));
        }
        resize(&mut self.idle, want(idle));
        self.held()
    }

    /// Bytes held, active and idle.
    pub fn held(&self) -> (u64, u64) {
        let a = self.active.lock().unwrap().len() as u64 * CHUNK;
        (a, self.idle.len() as u64 * CHUNK)
    }

    /// Full sweeps of the active RAM so far.
    pub fn sweeps(&self) -> u64 {
        self.sweeps.load(Ordering::Relaxed)
    }
}

impl Default for Hog {
    fn default() -> Self {
        Self::new()
    }
}

fn resize(chunks: &mut Vec<Chunk>, want: usize) {
    chunks.truncate(want);
    while chunks.len() < want {
        match chunk() {
            Some(c) => chunks.push(c),
            None => break,
        }
    }
}

impl Drop for Hog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(s) = self.sweeper.take() {
            let _ = s.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Targets are met to the chunk, and the sweeper touches the active
    /// part**: two chunks active and one idle are held, and sweeps run.
    #[test]
    fn targets_are_held_to_the_chunk_and_the_active_part_is_swept() {
        let mut h = Hog::new();
        assert_eq!(h.set(2 * CHUNK + 1, CHUNK), (2 * CHUNK, CHUNK));
        let before = h.sweeps();
        std::thread::sleep(Duration::from_millis(300));
        assert!(h.sweeps() > before, "the active chunks are swept");
        assert_eq!(h.set(0, 0), (0, 0));
    }
}
