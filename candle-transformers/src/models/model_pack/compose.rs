//! What a model pack's GGUF part holds, and where each tensor's bytes come from.
//!
//! A pack is built from one checkpoint and, for some models, files that complete
//! it — a draft head shipped beside the trunk, a base checkpoint's recurrent
//! gates, a tensor taken from another release. A [`Composition`] is the result
//! of resolving all of them into one set of named tensors and one metadata
//! block: what a load of the finished pack reads, with no second file to open.

use super::keys::CHECKPOINT_BYTES;
use candle::quantized::gguf_file::{Content, Value};
use candle::quantized::GgmlDType;
use candle::Result;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// One source file, mapped, with its GGUF directory read.
pub struct MappedSource {
    pub path: PathBuf,
    pub content: Content,
    pub mmap: Mmap,
}

impl MappedSource {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            candle::Error::Msg(format!("model pack source {}: {e}", path.display()))
        })?;
        // SAFETY: the mapping is read-only and the build holds it for as long as
        // it reads from it; a source changed under a running build is caught by
        // the SHA-256 the build records against the bytes it read.
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| candle::Error::Msg(format!("model pack mmap {}: {e}", path.display())))?;
        let content = Content::read(&mut Cursor::new(&mmap[..]))?;
        Ok(Self {
            path: path.to_path_buf(),
            content,
            mmap,
        })
    }

    /// The absolute offset and length of tensor `name`'s bytes in this file.
    pub fn tensor_span(&self, name: &str) -> Result<(u64, u64)> {
        let info = self.content.tensor_infos.get(name).ok_or_else(|| {
            candle::Error::Msg(format!(
                "model pack: {} has no tensor {name}",
                self.path.display()
            ))
        })?;
        let len = tensor_bytes(info.ggml_dtype, info.shape.dims());
        Ok((self.content.tensor_data_offset + info.offset, len))
    }
}

/// Bytes of a tensor of `dtype` and `dims`.
pub fn tensor_bytes(dtype: GgmlDType, dims: &[usize]) -> u64 {
    let elems: usize = dims.iter().product();
    (elems / dtype.block_size() * dtype.type_size()) as u64
}

/// Every tensor `content` carries, in bytes.
pub fn checkpoint_bytes(content: &Content) -> u64 {
    content
        .tensor_infos
        .values()
        .map(|i| tensor_bytes(i.ggml_dtype, i.shape.dims()))
        .sum()
}

/// The checkpoint's weight in bytes, for a `content` that is either a model
/// pack's GGUF part — which records it, the experts and streamed projections it
/// no longer lists included — or a checkpoint itself, whose tensors are all
/// there to sum.
pub fn checkpoint_bytes_of(content: &Content) -> u64 {
    content
        .metadata
        .get(CHECKPOINT_BYTES)
        .and_then(|v| v.to_u64().ok())
        .unwrap_or_else(|| checkpoint_bytes(content))
}

/// A tensor of the pack's GGUF part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedTensor {
    pub name: String,
    /// Index of the [`MappedSource`] its bytes are copied from.
    pub source: usize,
    pub dtype: GgmlDType,
    pub dims: Vec<usize>,
    /// Absolute offset of its bytes in that source.
    pub offset: u64,
}

impl PackedTensor {
    pub fn byte_len(&self) -> u64 {
        tensor_bytes(self.dtype, &self.dims)
    }
}

/// Metadata keys a source's own block carries that a pack must not repeat: the
/// split's bookkeeping (a pack is one file), the alignment (the pack sets its
/// own) and anything already under `zen.` (the pack writes its own).
fn passes_through(key: &str) -> bool {
    !(key.starts_with("split.") || key == "general.alignment" || key.starts_with("zen."))
}

/// The GGUF part of a pack: its metadata and its tensors, in write order.
#[derive(Debug, Clone, Default)]
pub struct Composition {
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<PackedTensor>,
}

impl Composition {
    /// Every tensor of `sources[0]` that `keep` admits, and its metadata.
    ///
    /// Tensors are ordered by where they sit in the checkpoint, so the build
    /// reads it front to back; metadata by key. Both orders are a function of
    /// the checkpoint alone, so two builds of one model write identical bytes.
    pub fn from_checkpoint(sources: &[MappedSource], keep: &dyn Fn(&str) -> bool) -> Self {
        let src = &sources[0];
        let mut metadata: Vec<(String, Value)> = src
            .content
            .metadata
            .iter()
            .filter(|(k, _)| passes_through(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        metadata.sort_by(|a, b| a.0.cmp(&b.0));
        let mut tensors: Vec<PackedTensor> = src
            .content
            .tensor_infos
            .iter()
            .filter(|(name, _)| keep(name))
            .map(|(name, info)| PackedTensor {
                name: name.clone(),
                source: 0,
                dtype: info.ggml_dtype,
                dims: info.shape.dims().to_vec(),
                offset: src.content.tensor_data_offset + info.offset,
            })
            .collect();
        tensors.sort_by(|a, b| a.offset.cmp(&b.offset).then_with(|| a.name.cmp(&b.name)));
        Self { metadata, tensors }
    }

    /// Take tensor `name` from `sources[source]` — replacing this composition's
    /// own when it has one, adding it when it does not.
    pub fn take(&mut self, sources: &[MappedSource], source: usize, name: &str) -> Result<()> {
        let src = &sources[source];
        let info = src.content.tensor_infos.get(name).ok_or_else(|| {
            candle::Error::Msg(format!(
                "model pack: {} has no tensor {name}",
                src.path.display()
            ))
        })?;
        let t = PackedTensor {
            name: name.to_string(),
            source,
            dtype: info.ggml_dtype,
            dims: info.shape.dims().to_vec(),
            offset: src.content.tensor_data_offset + info.offset,
        };
        match self.tensors.iter_mut().find(|x| x.name == name) {
            Some(slot) => *slot = t,
            None => self.tensors.push(t),
        }
        Ok(())
    }

    /// Set a metadata entry, replacing the source's value when it has one.
    pub fn set_metadata(&mut self, key: &str, value: Value) {
        match self.metadata.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.metadata.push((key.to_string(), value)),
        }
    }

    /// Bytes of every tensor the composition holds.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.byte_len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::quantized::gguf_writer::{GgufPlan, GgufStreamWriter, PlannedTensor};

    /// A source file with tensors `names` (F32, two elements each) and one
    /// metadata entry per key in `meta`.
    fn source(dir: &Path, file: &str, names: &[&str], meta: &[&str]) -> MappedSource {
        let mut plan = GgufPlan::new(32).unwrap();
        for k in meta {
            plan.push_metadata(*k, Value::U32(1)).unwrap();
        }
        for n in names {
            plan.push_tensor(PlannedTensor {
                name: (*n).into(),
                dtype: GgmlDType::F32,
                dims: vec![2],
            })
            .unwrap();
        }
        let path = dir.join(file);
        let mut w = GgufStreamWriter::new(plan, File::create(&path).unwrap()).unwrap();
        for (i, _) in names.iter().enumerate() {
            let v = [i as f32, -(i as f32)];
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.write_tensor_bytes(&bytes).unwrap();
        }
        w.finish().unwrap();
        MappedSource::open(&path).unwrap()
    }

    /// The checkpoint's tensors in file order, its metadata sorted, with the
    /// split's and the alignment's keys dropped and the excluded tensors gone.
    #[test]
    fn a_checkpoint_composes_in_file_order() {
        let dir = tempfile::tempdir().unwrap();
        let s = source(
            dir.path(),
            "a.gguf",
            &["b.weight", "a.weight", "blk.0.ffn_gate_exps.weight"],
            &["z.key", "split.count", "a.key"],
        );
        let c = Composition::from_checkpoint(&[s], &|n| !n.contains("_exps"));
        let names: Vec<&str> = c.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["b.weight", "a.weight"]);
        let keys: Vec<&str> = c.metadata.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["a.key", "z.key"]);
        assert_eq!(c.tensor_bytes(), 16);
    }

    /// A tensor taken from another file replaces the checkpoint's own, and one
    /// the checkpoint lacks is added.
    #[test]
    fn a_taken_tensor_replaces_or_adds() {
        let dir = tempfile::tempdir().unwrap();
        let sources = [
            source(dir.path(), "a.gguf", &["x", "y"], &[]),
            source(dir.path(), "b.gguf", &["y", "head"], &[]),
        ];
        let mut c = Composition::from_checkpoint(&sources, &|_| true);
        c.take(&sources, 1, "y").unwrap();
        c.take(&sources, 1, "head").unwrap();
        let from: Vec<(&str, usize)> = c
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t.source))
            .collect();
        assert_eq!(from, [("x", 0), ("y", 1), ("head", 1)]);
        assert!(c.take(&sources, 0, "missing").is_err());
    }
}
