//! Opening a model pack: its GGUF part mapped for the loaders, its sections
//! located, its provenance and tokenizer read.

use super::keys::{
    CHECKPOINT_BYTES, EXPERTS_LEN, EXPERTS_OFFSET, GGUF_LEN, INT8_MODE, LAYERS_LEN, LAYERS_OFFSET,
    NARROW, TOKENIZER_JSON, TOKENIZER_REPO, TOKENIZER_REV, VERSION, VERSION_KEY,
};
use super::provenance::{from_metadata, SourceRecord};
use candle::quantized::gguf_file::{Content, Value};
use candle::quantized::Int8Mode;
use candle::Result;
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Where a section sits in the pack file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionRef {
    pub offset: u64,
    pub len: u64,
}

/// An open model pack.
pub struct ModelPack {
    pub path: PathBuf,
    /// The GGUF part's directory: every tensor but the experts and the streamed
    /// projections, and the checkpoint's metadata.
    pub content: Content,
    /// The GGUF part, mapped — and only it: the sections are read with direct
    /// I/O, never through a mapping.
    pub mmap: Arc<Mmap>,
    pub int8_mode: Int8Mode,
    pub narrow: Option<usize>,
    pub checkpoint_bytes: u64,
    pub experts: Option<SectionRef>,
    pub layers: Option<SectionRef>,
    pub sources: Vec<SourceRecord>,
    pub tokenizer_repo: String,
    pub tokenizer_rev: String,
}

fn u64_of(m: &HashMap<String, Value>, key: &str) -> Result<u64> {
    m.get(key)
        .ok_or_else(|| candle::Error::Msg(format!("model pack: no {key}")))?
        .to_u64()
}

fn string_of(m: &HashMap<String, Value>, key: &str) -> Result<String> {
    Ok(m.get(key)
        .ok_or_else(|| candle::Error::Msg(format!("model pack: no {key}")))?
        .to_string()?
        .clone())
}

/// The `Int8Mode` a pack records.
fn mode_of(v: u64) -> Result<Int8Mode> {
    [Int8Mode::Off, Int8Mode::Performance, Int8Mode::Precision]
        .into_iter()
        .find(|m| *m as u64 == v)
        .ok_or_else(|| {
            candle::Error::Msg(format!(
                "model pack: int8 mode {v} is not one this build knows"
            ))
        })
}

fn section(m: &HashMap<String, Value>, offset: &str, len: &str) -> Result<Option<SectionRef>> {
    match m.get(offset) {
        None => Ok(None),
        Some(o) => Ok(Some(SectionRef {
            offset: o.to_u64()?,
            len: u64_of(m, len)?,
        })),
    }
}

impl ModelPack {
    /// Read `path`'s GGUF directory — no tensor bytes — and refuse a file that is
    /// not a pack of this version.
    pub fn read_header(path: &Path) -> Result<Content> {
        let file = File::open(path)
            .map_err(|e| candle::Error::Msg(format!("model pack open {}: {e}", path.display())))?;
        let content = Content::read(&mut BufReader::new(file))?;
        let version = content
            .metadata
            .get(VERSION_KEY)
            .ok_or_else(|| candle::Error::Msg(format!("{} is not a model pack", path.display())))?
            .to_u32()?;
        if version != VERSION {
            candle::bail!(
                "model pack {} is version {version}, this build reads {VERSION}",
                path.display()
            );
        }
        Ok(content)
    }

    pub fn open(path: &Path) -> Result<Self> {
        let content = Self::read_header(path)?;
        let m = &content.metadata;
        let gguf_len = u64_of(m, GGUF_LEN)?;
        let file_len = std::fs::metadata(path)
            .map_err(|e| candle::Error::Msg(format!("model pack stat {}: {e}", path.display())))?
            .len();
        let experts = section(m, EXPERTS_OFFSET, EXPERTS_LEN)?;
        let layers = section(m, LAYERS_OFFSET, LAYERS_LEN)?;
        let end = [experts, layers]
            .into_iter()
            .flatten()
            .map(|s| s.offset + s.len)
            .fold(gguf_len, u64::max);
        if file_len < end {
            candle::bail!(
                "model pack {} is {file_len} bytes, its layout needs {end} — truncated",
                path.display()
            );
        }
        let file = File::open(path)
            .map_err(|e| candle::Error::Msg(format!("model pack open {}: {e}", path.display())))?;
        // SAFETY: read-only, held by the loaders for as long as they read
        // through it. The pack is replaced only by a rename, which leaves an
        // open mapping's file in place.
        let mmap = unsafe { MmapOptions::new().len(gguf_len as usize).map(&file) }
            .map_err(|e| candle::Error::Msg(format!("model pack mmap {}: {e}", path.display())))?;
        let narrow = match u64_of(m, NARROW)? {
            0 => None,
            n => Some(n as usize),
        };
        Ok(Self {
            path: path.to_path_buf(),
            int8_mode: mode_of(u64_of(m, INT8_MODE)?)?,
            narrow,
            checkpoint_bytes: u64_of(m, CHECKPOINT_BYTES)?,
            experts,
            layers,
            sources: from_metadata(m)?,
            tokenizer_repo: string_of(m, TOKENIZER_REPO)?,
            tokenizer_rev: string_of(m, TOKENIZER_REV)?,
            mmap: Arc::new(mmap),
            content,
        })
    }

    /// `tokenizer.json`'s text, as the pack carries it.
    pub fn tokenizer_json(&self) -> Result<&str> {
        Ok(self
            .content
            .metadata
            .get(TOKENIZER_JSON)
            .ok_or_else(|| candle::Error::Msg(format!("model pack: no {TOKENIZER_JSON}")))?
            .to_string()?
            .as_str())
    }

    /// The GGUF part's length in bytes — what the loaders map.
    pub fn gguf_len(&self) -> u64 {
        self.mmap.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::{write_pack, PackBuild, Section, TokenizerSource};
    use super::super::compose::{Composition, MappedSource};
    use super::super::provenance::{Provenance, SourceRecord};
    use super::*;
    use candle::quantized::gguf_writer::{GgufPlan, GgufStreamWriter, PlannedTensor};
    use candle::quantized::GgmlDType;
    use std::io::Write;

    fn checkpoint(dir: &Path) -> MappedSource {
        let mut plan = GgufPlan::new(32).unwrap();
        plan.push_metadata("general.architecture", Value::String("test".into()))
            .unwrap();
        for n in ["a.weight", "blk.0.ffn_gate_exps.weight"] {
            plan.push_tensor(PlannedTensor {
                name: n.into(),
                dtype: GgmlDType::F32,
                dims: vec![4],
            })
            .unwrap();
        }
        let path = dir.join("ckpt.gguf");
        let mut w = GgufStreamWriter::new(plan, File::create(&path).unwrap()).unwrap();
        w.write_tensor_bytes(&[1u8; 16]).unwrap();
        w.write_tensor_bytes(&[2u8; 16]).unwrap();
        w.finish().unwrap();
        MappedSource::open(&path).unwrap()
    }

    /// A pack written with an expert section reads back with its GGUF part
    /// mapped exactly, the section where the metadata says, and every recorded
    /// fact intact — the checkpoint's own tensor excluded.
    #[test]
    fn a_written_pack_opens_with_its_layout() {
        let dir = tempfile::tempdir().unwrap();
        let sources = [checkpoint(dir.path())];
        let composition = Composition::from_checkpoint(&sources, &|n: &str| !n.contains("_exps"));
        let record = SourceRecord {
            role: "checkpoint".into(),
            repo: "org/m".into(),
            rev: "r1".into(),
            file: "ckpt.gguf".into(),
            len: 1,
            sha256: "00".repeat(32),
        };
        let out = dir.path().join("m.performance.pack.gguf");
        write_pack(
            PackBuild {
                sources: &sources,
                provenance: Provenance::complete(vec![record.clone()]),
                composition,
                int8_mode: Int8Mode::Performance,
                narrow: Some(64),
                checkpoint_bytes: 32,
                tokenizer: TokenizerSource {
                    repo: "org/tok".into(),
                    rev: "t1".into(),
                    json: "{\"model\":{}}".into(),
                },
                experts: Some(Section {
                    len: 5000,
                    write: Box::new(|w: &mut dyn Write| {
                        w.write_all(&[7u8; 5000])?;
                        Ok(5000)
                    }),
                }),
                layers: None,
            },
            &out,
        )
        .unwrap();

        let pack = ModelPack::open(&out).unwrap();
        assert_eq!(pack.int8_mode, Int8Mode::Performance);
        assert_eq!(pack.narrow, Some(64));
        assert_eq!(pack.checkpoint_bytes, 32);
        assert_eq!(pack.sources, vec![record]);
        assert_eq!(pack.tokenizer_repo, "org/tok");
        assert_eq!(pack.tokenizer_json().unwrap(), "{\"model\":{}}");
        assert!(pack.content.tensor_infos.contains_key("a.weight"));
        assert!(!pack
            .content
            .tensor_infos
            .contains_key("blk.0.ffn_gate_exps.weight"));
        let experts = pack.experts.unwrap();
        assert_eq!(experts.offset, pack.gguf_len());
        assert_eq!(experts.offset % 4096, 0);
        assert_eq!(experts.len, 5000);
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(bytes.len() as u64, experts.offset + 5000);
        assert!(bytes[experts.offset as usize..].iter().all(|&b| b == 7));
        // The one tensor's bytes, through the mapping.
        let info = &pack.content.tensor_infos["a.weight"];
        let at = (pack.content.tensor_data_offset + info.offset) as usize;
        assert_eq!(&pack.mmap[at..at + 16], &[1u8; 16]);
        // No temp file is left beside it.
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(names.iter().all(|n| !n.ends_with(".partial")), "{names:?}");
    }

    /// A GGUF that is not a pack is refused by name.
    #[test]
    fn a_plain_gguf_is_not_a_pack() {
        let dir = tempfile::tempdir().unwrap();
        let s = checkpoint(dir.path());
        let e = ModelPack::open(&s.path).err().unwrap().to_string();
        assert!(e.contains("not a model pack"), "{e}");
    }
}
