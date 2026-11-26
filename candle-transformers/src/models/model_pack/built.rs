//! Tensors a family's build computes rather than copies.
//!
//! A load that builds a weight on the card — stacks parts, pads a projection
//! to the int8 tiling, folds a scale in, repacks it to its KO twin — pays that
//! work on every load, for bytes that come out the same every time. A family
//! whose load does that builds those tensors once, at pack build, with the
//! loader's own code on the device, and writes the results here: a scratch file
//! beside the pack, one tensor after another, read back through a mapping when
//! the GGUF part is written. The pack then stores the built form, and the load
//! copies it to its destination as it is.
//!
//! The scratch exists because a GGUF part is planned before a byte of it is
//! written, and a built tensor's bytes exist only once the device has made
//! them. Holding them in memory instead would put the whole dense stack — about
//! 5 GiB on Flash-Next — in host RAM beside the mapped sources.

use super::compose::tensor_bytes;
use candle::quantized::{GgmlDType, QTensor};
use candle::Result;
use memmap2::{Mmap, MmapOptions};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// One built tensor: its name, the type and shape the pack stores it as, and
/// where its bytes sit in the scratch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltTensor {
    pub name: String,
    pub dtype: GgmlDType,
    pub dims: Vec<usize>,
    pub offset: u64,
}

impl BuiltTensor {
    /// Bytes the tensor holds.
    pub fn byte_len(&self) -> u64 {
        tensor_bytes(self.dtype, &self.dims)
    }
}

/// The scratch being written.
pub struct BuiltWriter {
    path: PathBuf,
    file: Option<BufWriter<File>>,
    at: u64,
    tensors: Vec<BuiltTensor>,
    names: HashSet<String>,
}

impl BuiltWriter {
    /// A scratch in `dir`, named apart from any other build's — the pid and a
    /// nanosecond stamp, as the pack's own temp file is.
    pub fn create(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|e| {
            candle::Error::Msg(format!("built tensors mkdir {}: {e}", dir.display()))
        })?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = dir.join(format!("{}.{stamp:x}.built.partial", std::process::id()));
        let file = File::create(&path).map_err(|e| {
            candle::Error::Msg(format!("built tensors create {}: {e}", path.display()))
        })?;
        Ok(Self {
            path,
            file: Some(BufWriter::with_capacity(16 << 20, file)),
            at: 0,
            tensors: Vec::new(),
            names: HashSet::new(),
        })
    }

    /// Append tensor `name` of `dtype` and `dims`, whose bytes are `bytes` as
    /// GGUF lays them out. Refuses a second tensor of one name and bytes of any
    /// length but the tensor's.
    pub fn push(
        &mut self,
        name: &str,
        dtype: GgmlDType,
        dims: &[usize],
        bytes: &[u8],
    ) -> Result<()> {
        let len = tensor_bytes(dtype, dims);
        if bytes.len() as u64 != len {
            candle::bail!(
                "built tensor {name}: {} bytes for a {dims:?} {dtype:?} tensor of {len}",
                bytes.len()
            );
        }
        if !self.names.insert(name.to_string()) {
            candle::bail!("built tensor {name} was built twice");
        }
        let Some(file) = self.file.as_mut() else {
            candle::bail!("built tensors: the scratch is already finished");
        };
        file.write_all(bytes)
            .map_err(|e| candle::Error::Msg(format!("built tensor {name}: {e}")))?;
        self.tensors.push(BuiltTensor {
            name: name.to_string(),
            dtype,
            dims: dims.to_vec(),
            offset: self.at,
        });
        self.at += len;
        Ok(())
    }

    /// Append `qt` under `name`, its bytes read back from wherever it lives.
    pub fn push_qtensor(&mut self, name: &str, qt: &QTensor) -> Result<()> {
        let bytes = qt.data()?;
        self.push(name, qt.dtype(), qt.shape().dims(), &bytes)
    }

    /// Close the scratch and map it for reading. On any failure the writer
    /// still owns the scratch, and removes it as it drops.
    pub fn finish(mut self) -> Result<BuiltSource> {
        let file = self.file.take().ok_or_else(|| {
            candle::Error::Msg("built tensors: the scratch is already finished".into())
        })?;
        let file = file
            .into_inner()
            .map_err(|e| candle::Error::Msg(format!("built tensors flush: {e}")))?;
        file.sync_all()
            .map_err(|e| candle::Error::Msg(format!("built tensors sync: {e}")))?;
        drop(file);
        let file = File::open(&self.path).map_err(|e| {
            candle::Error::Msg(format!("built tensors open {}: {e}", self.path.display()))
        })?;
        // An empty file cannot be mapped on every platform, and there is
        // nothing to read from one.
        let mmap = if self.at == 0 {
            None
        } else {
            // SAFETY: the scratch is this build's own file, named by its pid
            // and a stamp, written above and never written again; it is
            // removed only when the mapping is dropped with its source.
            Some(unsafe { MmapOptions::new().map(&file) }.map_err(|e| {
                candle::Error::Msg(format!("built tensors mmap {}: {e}", self.path.display()))
            })?)
        };
        // Mapped: the scratch passes to its source, which removes it.
        Ok(BuiltSource {
            path: mem::take(&mut self.path),
            mmap,
            tensors: mem::take(&mut self.tensors),
        })
    }
}

impl Drop for BuiltWriter {
    fn drop(&mut self) {
        // A scratch abandoned before it was finished is removed with its
        // writer; a finished one belongs to its `BuiltSource`.
        if !self.path.as_os_str().is_empty() {
            drop(self.file.take());
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// The finished scratch, mapped: the built tensors a pack's GGUF part copies.
/// The file is removed when this is dropped.
pub struct BuiltSource {
    path: PathBuf,
    mmap: Option<Mmap>,
    tensors: Vec<BuiltTensor>,
}

impl BuiltSource {
    /// Every built tensor, in the order it was built.
    pub fn tensors(&self) -> &[BuiltTensor] {
        &self.tensors
    }

    /// The bytes of `tensor`, one of [`Self::tensors`].
    pub fn bytes(&self, tensor: &BuiltTensor) -> Result<&[u8]> {
        let (at, len) = (tensor.offset as usize, tensor.byte_len() as usize);
        let mapped: &[u8] = self.mmap.as_deref().unwrap_or(&[]);
        mapped.get(at..at + len).ok_or_else(|| {
            candle::Error::Msg(format!(
                "built tensor {} runs past the end of its scratch",
                tensor.name
            ))
        })
    }
}

impl Drop for BuiltSource {
    fn drop(&mut self) {
        drop(self.mmap.take());
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tensors land one after another in build order, read back byte for byte,
    /// and the scratch is gone once its source is dropped.
    #[test]
    fn built_tensors_read_back_in_order_and_the_scratch_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = BuiltWriter::create(dir.path()).unwrap();
        w.push("a", GgmlDType::F32, &[2], &[1, 2, 3, 4, 5, 6, 7, 8])
            .unwrap();
        w.push("b", GgmlDType::F16, &[3], &[9, 10, 11, 12, 13, 14])
            .unwrap();
        let src = w.finish().unwrap();
        assert_eq!(
            src.tensors(),
            [
                BuiltTensor {
                    name: "a".into(),
                    dtype: GgmlDType::F32,
                    dims: vec![2],
                    offset: 0,
                },
                BuiltTensor {
                    name: "b".into(),
                    dtype: GgmlDType::F16,
                    dims: vec![3],
                    offset: 8,
                },
            ]
        );
        assert_eq!(
            src.bytes(&src.tensors()[0]).unwrap(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(
            src.bytes(&src.tensors()[1]).unwrap(),
            [9, 10, 11, 12, 13, 14]
        );
        let path = src.path.clone();
        assert!(path.exists());
        drop(src);
        assert!(!path.exists());
    }

    /// A length that is not the tensor's, or a name built twice, is refused.
    #[test]
    fn a_wrong_length_or_a_repeated_name_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = BuiltWriter::create(dir.path()).unwrap();
        assert!(w.push("a", GgmlDType::F32, &[2], &[0; 7]).is_err());
        w.push("a", GgmlDType::F32, &[2], &[0; 8]).unwrap();
        let e = w
            .push("a", GgmlDType::F32, &[2], &[0; 8])
            .unwrap_err()
            .to_string();
        assert!(e.contains("built twice"), "{e}");
    }

    /// A writer dropped before it finishes takes its scratch with it.
    #[test]
    fn an_abandoned_scratch_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let w = BuiltWriter::create(dir.path()).unwrap();
        let path = w.path.clone();
        assert!(path.exists());
        drop(w);
        assert!(!path.exists());
    }

    /// A scratch with nothing in it finishes, and holds nothing.
    #[test]
    fn an_empty_scratch_finishes() {
        let dir = tempfile::tempdir().unwrap();
        let src = BuiltWriter::create(dir.path()).unwrap().finish().unwrap();
        assert!(src.tensors().is_empty());
    }
}
