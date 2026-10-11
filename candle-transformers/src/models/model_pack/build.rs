//! Writing a model pack: the GGUF part, then the sections after it, into a
//! private temp file that is synced and renamed into place only when complete.
//!
//! Everything is placed before anything is written. The GGUF part's length is a
//! function of its metadata and tensor directory ([`GgufPlan`]), and the section
//! offsets the metadata records are fixed-width integers — so the plan is laid
//! out once with the offsets at zero, measured, and laid out again with the real
//! values at an identical length.

use super::compose::{Composition, MappedSource};
use super::digest::patch_digests;
use super::keys::{
    CHECKPOINT_BYTES, EXPERTS_LEN, EXPERTS_OFFSET, GGUF_LEN, INT8_MODE, LAYERS_LEN, LAYERS_OFFSET,
    NARROW, TOKENIZER_JSON, TOKENIZER_REPO, TOKENIZER_REV, VERSION, VERSION_KEY,
};
use super::provenance::{to_metadata, Provenance};
use candle::direct_io::round_up_sector;
use candle::quantized::gguf_file::Value;
use candle::quantized::gguf_writer::{GgufPlan, GgufStreamWriter, PlannedTensor};
use candle::quantized::Int8Mode;
use candle::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// The GGUF part's alignment — a direct-I/O sector, so the sections that follow
/// it start on one.
pub const PACK_ALIGNMENT: u64 = 4096;

/// Where the tokenizer came from, and its text.
pub struct TokenizerSource {
    pub repo: String,
    pub rev: String,
    pub json: String,
}

/// Writes one section's bytes and answers how many it wrote.
pub type SectionWriter<'a> = Box<dyn FnOnce(&mut dyn Write) -> Result<u64> + 'a>;

/// A section of the pack after its GGUF part: its length, known up front, and
/// the writer that produces exactly that many bytes.
pub struct Section<'a> {
    pub len: u64,
    pub write: SectionWriter<'a>,
}

/// Everything a pack is made of.
pub struct PackBuild<'a> {
    pub sources: &'a [MappedSource],
    /// What each source is. A record whose SHA-256 is still being taken holds
    /// its [`placeholder`](super::digest::placeholder), replaced before the pack
    /// is published.
    pub provenance: Provenance,
    pub composition: Composition,
    pub int8_mode: Int8Mode,
    pub narrow: Option<usize>,
    /// Every tensor the checkpoint carries — see [`CHECKPOINT_BYTES`].
    pub checkpoint_bytes: u64,
    pub tokenizer: TokenizerSource,
    pub experts: Option<Section<'a>>,
    pub layers: Option<Section<'a>>,
}

/// Where each section of a pack whose GGUF part is `gguf_len` bytes starts.
fn place(gguf_len: u64, experts: Option<u64>, layers: Option<u64>) -> (Option<u64>, Option<u64>) {
    let mut at = gguf_len;
    let experts_at = experts.map(|len| {
        let here = at;
        at = round_up_sector((here + len) as usize) as u64;
        here
    });
    let layers_at = layers.map(|_| at);
    (experts_at, layers_at)
}

/// The GGUF part's plan, with the sections at `experts_at` / `layers_at`.
fn plan(
    build: &PackBuild<'_>,
    gguf_len: u64,
    experts: Option<(u64, u64)>,
    layers: Option<(u64, u64)>,
) -> Result<GgufPlan> {
    let mut plan = GgufPlan::new(PACK_ALIGNMENT)?;
    for (k, v) in &build.composition.metadata {
        plan.push_metadata(k.clone(), v.clone())?;
    }
    plan.push_metadata(VERSION_KEY, Value::U32(VERSION))?;
    plan.push_metadata(INT8_MODE, Value::U32(build.int8_mode as u32))?;
    plan.push_metadata(NARROW, Value::U32(build.narrow.unwrap_or(0) as u32))?;
    plan.push_metadata(CHECKPOINT_BYTES, Value::U64(build.checkpoint_bytes))?;
    plan.push_metadata(GGUF_LEN, Value::U64(gguf_len))?;
    if let Some((at, len)) = experts {
        plan.push_metadata(EXPERTS_OFFSET, Value::U64(at))?;
        plan.push_metadata(EXPERTS_LEN, Value::U64(len))?;
    }
    if let Some((at, len)) = layers {
        plan.push_metadata(LAYERS_OFFSET, Value::U64(at))?;
        plan.push_metadata(LAYERS_LEN, Value::U64(len))?;
    }
    plan.push_metadata(TOKENIZER_REPO, Value::String(build.tokenizer.repo.clone()))?;
    plan.push_metadata(TOKENIZER_REV, Value::String(build.tokenizer.rev.clone()))?;
    plan.push_metadata(TOKENIZER_JSON, Value::String(build.tokenizer.json.clone()))?;
    for (k, v) in to_metadata(&build.provenance.records) {
        plan.push_metadata(k, v)?;
    }
    for t in &build.composition.tensors {
        plan.push_tensor(PlannedTensor {
            name: t.name.clone(),
            dtype: t.dtype,
            dims: t.dims.clone(),
        })?;
    }
    Ok(plan)
}

/// The temp file a build writes, removed unless the build publishes it.
struct Partial {
    path: PathBuf,
    published: bool,
}

impl Drop for Partial {
    fn drop(&mut self) {
        if !self.published {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Write `build` as the pack at `out`.
///
/// The temp name carries the pid and a nanosecond stamp: two processes building
/// one pack at the same moment write separate files and the rename picks a
/// winner, rather than interleaving into one.
pub fn write_pack(build: PackBuild<'_>, out: &Path) -> Result<()> {
    let experts_len = build.experts.as_ref().map(|s| s.len);
    let layers_len = build.layers.as_ref().map(|s| s.len);
    let measure = |gguf_len: u64| -> Result<GgufPlan> {
        let (ea, la) = place(gguf_len, experts_len, layers_len);
        plan(&build, gguf_len, ea.zip(experts_len), la.zip(layers_len))
    };
    let gguf_len = measure(0)?.total_len()?;
    let plan = measure(gguf_len)?;
    if plan.total_len()? != gguf_len {
        candle::bail!("model pack: the GGUF part's length moved when its offsets were filled in");
    }
    let (experts_at, layers_at) = place(gguf_len, experts_len, layers_len);

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| candle::Error::Msg(format!("model pack mkdir {}: {e}", dir.display())))?;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut partial = Partial {
        path: out.with_extension(format!("{}.{stamp:x}.partial", std::process::id())),
        published: false,
    };
    let file = File::create(&partial.path).map_err(|e| {
        candle::Error::Msg(format!("model pack create {}: {e}", partial.path.display()))
    })?;
    let t0 = std::time::Instant::now();
    let mut w = GgufStreamWriter::new(plan, BufWriter::with_capacity(16 << 20, file))?;
    for t in &build.composition.tensors {
        let src = &build.sources[t.source].mmap;
        let (at, len) = (t.offset as usize, t.byte_len() as usize);
        let bytes = src.get(at..at + len).ok_or_else(|| {
            candle::Error::Msg(format!(
                "model pack: {} runs past the end of its source",
                t.name
            ))
        })?;
        w.write_tensor_bytes(bytes)?;
    }
    let mut out_w = w.finish()?;
    let mut written = gguf_len;
    for (section, at) in [(build.experts, experts_at), (build.layers, layers_at)] {
        let (Some(section), Some(at)) = (section, at) else {
            continue;
        };
        out_w.write_all(&vec![0u8; (at - written) as usize])?;
        let len = (section.write)(&mut out_w)?;
        if len != section.len {
            candle::bail!(
                "model pack: a section wrote {len} bytes against the {} it was placed for",
                section.len
            );
        }
        written = at + len;
    }
    out_w.flush()?;
    let file = out_w
        .into_inner()
        .map_err(|e| candle::Error::Msg(format!("model pack flush: {e}")))?;
    file.sync_all()
        .map_err(|e| candle::Error::Msg(format!("model pack sync: {e}")))?;
    drop(file);
    // The sources' SHA-256s were taken beside the write; they go over their
    // placeholders now, before the rename, so a published pack is never
    // missing one.
    let digests = build.provenance.digests.wait()?;
    patch_digests(&partial.path, &digests)?;
    // An existing pack under the final name is being replaced — it was stale,
    // or the resolver would not have built — and Windows will not rename over it.
    let _ = std::fs::remove_file(out);
    std::fs::rename(&partial.path, out).map_err(|e| {
        candle::Error::Msg(format!(
            "model pack publish {} → {}: {e}",
            partial.path.display(),
            out.display()
        ))
    })?;
    partial.published = true;
    tracing::info!(
        target: "candle_transformers::model_pack",
        path = %out.display(),
        gib = written as f64 / (1u64 << 30) as f64,
        secs = t0.elapsed().as_secs_f64(),
        "model pack written"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A routed model's sections: experts straight after the GGUF part, which
    /// ends on a sector; a streamed one's layers likewise; with both, layers
    /// after the experts, rounded to a sector.
    #[test]
    fn sections_are_placed_after_the_gguf_part_on_sectors() {
        assert_eq!(place(8192, Some(5000), None), (Some(8192), None));
        assert_eq!(place(8192, None, Some(9)), (None, Some(8192)));
        assert_eq!(place(8192, Some(5000), Some(9)), (Some(8192), Some(16384)));
        assert_eq!(place(8192, None, None), (None, None));
    }
}
