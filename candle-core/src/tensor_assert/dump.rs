//! Writing a failing kernel call's operands to disk, so it can be replayed.
//!
//! A dump exists to answer one question that logs cannot: **is the fault
//! deterministic?** Replaying the same kernel over the same bytes either
//! reproduces the bad output — in which case the inputs really do produce it
//! and the arithmetic is the thing to look at — or it does not, in which case
//! the bytes that reached the kernel in production were not these, and the
//! fault is a race, a lifetime, or a copy that had not landed. Those two
//! answers point in opposite directions, and nothing short of a replay
//! separates them.
//!
//! The format is deliberately dumb: one raw little-endian `.bin` per buffer
//! plus a `manifest.txt` of `key=value` lines. No serialization library, no
//! dtype registry, no versioning. A dump is read exactly once, by a test
//! written against the dump that produced it, on the machine that produced it —
//! so the only property that matters is that reading it back cannot silently
//! reinterpret it.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::cuda_backend::CudaDevice;
use crate::Result;
use cudarc::driver::CudaSlice;

/// A directory being filled with one failing call's operands.
pub struct Dump {
    dir: PathBuf,
    manifest: BTreeMap<String, String>,
}

impl Dump {
    /// Create `dir` (removing any previous contents) and start a manifest.
    pub fn create(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        // A stale buffer from an earlier dump silently replayed as if it
        // belonged to this one is the worst failure this can have, so the
        // directory starts empty rather than being written over.
        if dir.exists() {
            fs::remove_dir_all(&dir)
                .map_err(|e| crate::Error::Msg(format!("dump: clearing {dir:?}: {e}")))?;
        }
        fs::create_dir_all(&dir)
            .map_err(|e| crate::Error::Msg(format!("dump: creating {dir:?}: {e}")))?;
        Ok(Self {
            dir,
            manifest: BTreeMap::new(),
        })
    }

    /// Record a scalar fact — a shape, a dtype, a launch bound.
    pub fn note(&mut self, key: &str, value: impl std::fmt::Display) {
        self.manifest.insert(key.to_string(), value.to_string());
    }

    /// Write `bytes` as `name.bin`, noting its length.
    pub fn bytes(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        let path = self.dir.join(format!("{name}.bin"));
        fs::write(&path, bytes)
            .map_err(|e| crate::Error::Msg(format!("dump: writing {path:?}: {e}")))?;
        self.note(&format!("{name}.bytes"), bytes.len());
        Ok(())
    }

    /// Copy a device buffer to host and write it.
    pub fn device<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone>(
        &mut self,
        name: &str,
        dev: &CudaDevice,
        slice: &CudaSlice<T>,
    ) -> Result<()> {
        let host = dev
            .memcpy_dtov(slice)
            .map_err(|e| crate::Error::Msg(format!("dump: reading {name}: {e}")))?;
        self.note(&format!("{name}.elems"), host.len());
        let bytes = unsafe {
            std::slice::from_raw_parts(
                host.as_ptr() as *const u8,
                std::mem::size_of_val(host.as_slice()),
            )
        };
        self.bytes(name, bytes)
    }

    /// Copy `len` bytes from a raw device address and write them.
    ///
    /// For buffers that are reachable only as an address — an entry of a device
    /// pointer table, an arena lease — which is most of what a kernel actually
    /// reads.
    ///
    /// # Safety
    ///
    /// `ptr` must name at least `len` readable bytes on `dev`.
    pub unsafe fn device_ptr(
        &mut self,
        name: &str,
        dev: &CudaDevice,
        ptr: u64,
        len: usize,
    ) -> Result<()> {
        // Viewed as `u8` regardless of the buffer's real element type: a dump
        // records bytes, and the manifest records what they mean.
        let stream = dev.cuda_stream();
        let view: CudaSlice<u8> = stream.upgrade_device_ptr::<u8>(ptr, len);
        let host = dev
            .memcpy_dtov(&view)
            .map_err(|e| crate::Error::Msg(format!("dump: reading {name} at {ptr:#x}: {e}")))?;
        // The view is a borrow of memory this dump does not own; leaking it is
        // what stops the drop from freeing someone else's allocation.
        std::mem::forget(view);
        self.bytes(name, &host)
    }

    /// Flush the manifest. Consumes the dump: nothing may be added after the
    /// index of what it contains has been written.
    pub fn finish(self) -> Result<PathBuf> {
        let path = self.dir.join("manifest.txt");
        let mut f = fs::File::create(&path)
            .map_err(|e| crate::Error::Msg(format!("dump: creating {path:?}: {e}")))?;
        for (k, v) in &self.manifest {
            writeln!(f, "{k}={v}")
                .map_err(|e| crate::Error::Msg(format!("dump: writing {path:?}: {e}")))?;
        }
        f.flush()
            .map_err(|e| crate::Error::Msg(format!("dump: flushing {path:?}: {e}")))?;
        Ok(self.dir)
    }
}

/// A dump read back, for a replay test.
pub struct Replay {
    dir: PathBuf,
    manifest: BTreeMap<String, String>,
}

impl Replay {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join("manifest.txt");
        let text = fs::read_to_string(&path)
            .map_err(|e| crate::Error::Msg(format!("replay: reading {path:?}: {e}")))?;
        let manifest = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Ok(Self { dir, manifest })
    }

    /// A recorded scalar, parsed. Missing or unparseable is an error rather
    /// than a default — a replay that quietly substituted a zero launch bound
    /// would "pass" by doing nothing.
    pub fn get<T: std::str::FromStr>(&self, key: &str) -> Result<T> {
        let raw = self
            .manifest
            .get(key)
            .ok_or_else(|| crate::Error::Msg(format!("replay: manifest has no {key:?}")))?;
        raw.parse::<T>()
            .map_err(|_| crate::Error::Msg(format!("replay: {key}={raw:?} does not parse")))
    }

    pub fn has(&self, key: &str) -> bool {
        self.manifest.contains_key(key)
    }

    pub fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let path = self.dir.join(format!("{name}.bin"));
        fs::read(&path).map_err(|e| crate::Error::Msg(format!("replay: reading {path:?}: {e}")))
    }

    /// Reinterpret a recorded buffer as `T`.
    ///
    /// Refuses a length that is not a whole number of `T`, because the common
    /// way to misread a dump is to name the wrong buffer, and a partial element
    /// is the cheapest place to catch it.
    pub fn typed<T: Clone>(&self, name: &str) -> Result<Vec<T>> {
        let raw = self.bytes(name)?;
        let sz = std::mem::size_of::<T>();
        if sz == 0 || raw.len() % sz != 0 {
            crate::bail!(
                "replay: {name} is {} bytes, not a whole number of {}-byte elements",
                raw.len(),
                sz
            );
        }
        let n = raw.len() / sz;
        let mut out = Vec::with_capacity(n);
        // SAFETY: `raw` holds `n * sz` bytes and is read as `n` values of `T`;
        // `T: Clone` with no invalid bit patterns is the caller's obligation,
        // and every use here is a plain numeric type.
        unsafe {
            let src = raw.as_ptr() as *const T;
            for i in 0..n {
                out.push((*src.add(i)).clone());
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{Dump, Replay};

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("candle_dump_test_{name}"))
    }

    #[test]
    fn a_dump_round_trips_its_notes_and_buffers() {
        let dir = tmp("roundtrip");
        let mut d = Dump::create(&dir).expect("create");
        d.note("rows", 7usize);
        d.note("dtype", "Q4_KO");
        d.bytes("payload", &[1u8, 2, 3, 4, 5, 6, 7, 8])
            .expect("bytes");
        d.finish().expect("finish");

        let r = Replay::open(&dir).expect("open");
        assert_eq!(r.get::<usize>("rows").unwrap(), 7);
        assert_eq!(r.get::<String>("dtype").unwrap(), "Q4_KO");
        assert_eq!(r.get::<usize>("payload.bytes").unwrap(), 8);
        assert_eq!(r.bytes("payload").unwrap(), vec![1u8, 2, 3, 4, 5, 6, 7, 8]);
        // 8 bytes read as u32 is exactly 2 elements, little-endian.
        assert_eq!(
            r.typed::<u32>("payload").unwrap(),
            vec![0x04030201, 0x08070605]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_key_is_an_error_not_a_default() {
        let dir = tmp("missing");
        let d = Dump::create(&dir).expect("create");
        d.finish().expect("finish");
        let r = Replay::open(&dir).expect("open");
        assert!(r.get::<usize>("launch_tiles").is_err());
        assert!(!r.has("launch_tiles"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_partial_element_is_refused_rather_than_truncated() {
        let dir = tmp("partial");
        let mut d = Dump::create(&dir).expect("create");
        d.bytes("odd", &[1u8, 2, 3]).expect("bytes");
        d.finish().expect("finish");
        let r = Replay::open(&dir).expect("open");
        assert!(
            r.typed::<u32>("odd").is_err(),
            "3 bytes is not a whole number of u32"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn creating_over_an_existing_dump_clears_the_stale_buffers() {
        let dir = tmp("stale");
        let mut d = Dump::create(&dir).expect("create");
        d.bytes("ghost", &[9u8; 16]).expect("bytes");
        d.finish().expect("finish");

        let d = Dump::create(&dir).expect("recreate");
        d.finish().expect("finish");
        let r = Replay::open(&dir).expect("open");
        assert!(
            r.bytes("ghost").is_err(),
            "a buffer from the previous dump must not survive into this one"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
