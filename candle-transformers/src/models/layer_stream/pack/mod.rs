//! The cold tier: a repacked layer pack holding every streamable layer, always.
//!
//! The layer analogue of [`expert_lre::pack`](crate::models::expert_lre), and
//! it exists for the same three reasons, in the same order of weight:
//!
//! 1. **The repack is hot-path poison.** A layer's projections dequantize to
//!    F32 and requantize to their KO twins; doing that on a miss would cost
//!    seconds inside a forward.
//! 2. **A repacked layer is one blob**, so a load is an offset and a copy. In
//!    the GGUF the same layer is six or seven separately-named tensors.
//! 3. **It decouples the hot path from the checkpoint format**, so a GGUF
//!    packing decision cannot become a streaming regression.
//!
//! # The invariant this exists to hold
//!
//! > **The cold tier holds a valid copy of every layer outside the pinned
//! > head, always.**
//!
//! Everything residency does follows from it: eviction is a bookkeeping change
//! with no copy and no destination to find, the warm tier needs no eviction
//! policy, and "where do I load this from" is a total function. See
//! `docs/qwen38_layer_streaming.md` §4.
//!
//! # Records are sector-aligned because the reads bypass the page cache
//!
//! Reads go through [`candle::direct_io`], which requires the file offset, the
//! length and the destination pointer to be 4 KiB-aligned. A record's stride is
//! the slot image padded up to a sector, so every record starts on one, and the
//! warm pool's slots are cut to the same stride — which is what lets a cold read
//! land *directly* in a pinned slot with no bounce buffer.
//!
//! # One record width for two layer kinds
//!
//! Unlike an expert record, a layer record's *contents* vary: a DeltaNet layer
//! holds six projections and an attention layer seven. The stride does not vary
//! — it is the widest image, padded — so the shorter kind leaves a zeroed tail.
//! That is the same uniformity the weight zone requires of its slots, arrived at
//! for the same reason, and it costs ~2% on Qwen3.8-27B.

mod header;

use candle::direct_io::{round_up_sector, DirectFile, StripeRead};
use candle::fletcher::fletcher32;
use candle::quantized::Int8Mode;
use candle::Result;
use header::{LayerSpans, PackHeader, ProjectionSpan};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::models::layer_stream::descriptor::LayerImage;

/// Bytes of the GGUF that go into the identity checksum.
///
/// The header and tensor table live at the front of the file and carry every
/// tensor's name, dtype, shape and offset — so a checkpoint that differs in any
/// way that matters differs inside this window. Hashing the whole 16.5 GiB
/// would cost a full sequential read at every startup to re-derive a fact the
/// length and this window already settle.
const IDENTITY_SAMPLE: usize = 4 * 1024 * 1024;

/// Bytes read from the front of a pack to decode its header.
///
/// The header is variable-width — a per-layer table whose rows carry one entry
/// per projection — so its length is not known until it is decoded, and this is
/// the window that decode happens inside. A 64-layer model's header is under
/// 8 KiB, so a mebibyte is four orders of magnitude of slack and still nothing
/// beside the records that follow it. A header past the window is refused rather
/// than chased with a second read: it would mean thousands of layers, which is
/// a different problem than this file has.
const HEADER_WINDOW: usize = 1024 * 1024;

/// What a pack claims to have been built from.
///
/// Two halves answering two questions: the first three fields are *which
/// checkpoint*, and `repack_fp` is *which repack of it*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackIdentity {
    pub source_len: u64,
    pub source_sum: u32,
    pub int8_mode: u32,
    pub repack_fp: u64,
}

impl PackIdentity {
    /// The identity of the GGUF mapped at `gguf`, repacked for `int8mode`, by a
    /// build whose repack fingerprints as `repack_fp`.
    ///
    /// The caller supplies the fingerprint rather than this computing it,
    /// because the sweep needs a CUDA device — which this module deliberately
    /// knows nothing about.
    pub fn of(gguf: &[u8], int8mode: Int8Mode, repack_fp: u64) -> Self {
        let sample = &gguf[..gguf.len().min(IDENTITY_SAMPLE)];
        Self {
            source_len: gguf.len() as u64,
            source_sum: fletcher32(sample),
            int8_mode: int8mode as u32,
            repack_fp,
        }
    }
}

/// A read request: one record into one stride-long aligned destination.
pub(crate) struct PackRead<'a> {
    pub layer: usize,
    pub dest: &'a mut [u8],
}

/// The cold tier, open for reading.
pub(crate) struct LayerPack {
    reader: DirectFile,
    path: PathBuf,
    header: PackHeader,
    /// Byte offset of record 0 — the header, padded to a sector.
    body: u64,
    stride: usize,
    /// One checksum per record, read from the trailer at open.
    sums: Vec<u32>,
}

impl std::fmt::Debug for LayerPack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerPack")
            .field("path", &self.path)
            .field("layers", &self.header.num_layers)
            .field("pinned", &self.header.pinned_layers)
            .field("stride", &self.stride)
            .finish()
    }
}

impl LayerPack {
    /// Record-to-record distance, and so the size of every destination buffer.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Leading layers the file holds no record for.
    pub(crate) fn pinned_layers(&self) -> usize {
        self.header.pinned_layers as usize
    }

    /// Layers the file holds records for.
    ///
    /// The runtime never asks — it addresses records by layer index, and the
    /// header's own `open` check has already established the count agrees with
    /// the images. The pack's round-trip test does ask, which is the whole
    /// reason it is here.
    #[cfg(test)]
    pub(crate) fn stored_layers(&self) -> usize {
        self.header.stored_layers()
    }

    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Where each projection of `layer` sits inside its record, **as the file
    /// says** rather than as the image the caller passed in.
    ///
    /// The runtime does not need this — a record is addressed by layer index
    /// and its projections are read at the image's offsets, which `open` has
    /// already checked against the header. The round-trip test does need it:
    /// asserting the payload lands where the *decoded header* says closes the
    /// loop through the bytes on disk, where asserting against the image the
    /// test wrote from would only check the test against itself.
    #[cfg(test)]
    pub(crate) fn spans(&self, layer: usize) -> Result<&LayerSpans> {
        self.header.layers.get(layer).ok_or_else(|| {
            candle::Error::Msg(format!(
                "layer pack: layer {layer} is past the model's {} layers",
                self.header.num_layers
            ))
        })
    }

    /// Byte offset of `layer`'s record.
    fn offset_of(&self, layer: usize) -> Result<u64> {
        Ok(self.body + self.header.record_index(layer)? as u64 * self.stride as u64)
    }

    /// Read one layer's record into `dest`, which must be exactly one stride.
    ///
    /// # `dest` must be sector-aligned
    ///
    /// The read bypasses the page cache, so the kernel requires the destination
    /// *pointer* — not merely its length — to be a multiple of
    /// [`candle::direct_io::DIRECT_IO_SECTOR`]. A plain `Vec<u8>` is not, and
    /// the failure is an assertion inside `read_at` rather than a short read.
    ///
    /// In production this is free: the destination is a pinned warm-pool slot,
    /// and the pool's slots are cut to this pack's stride, which is a whole
    /// number of sectors — so every slot base is aligned by construction. That
    /// is the same property that lets a cold read land in a warm slot with no
    /// bounce buffer. Callers outside that path want
    /// [`candle::direct_io::AlignedScratch`].
    pub(crate) fn read_into(&self, layer: usize, dest: &mut [u8]) -> Result<()> {
        if dest.len() != self.stride {
            candle::bail!(
                "layer pack read wants a {}-byte destination, got {}",
                self.stride,
                dest.len()
            );
        }
        let at = self.offset_of(layer)?;
        self.reader.read_at(at, dest).map_err(|e| {
            candle::Error::Msg(format!(
                "layer pack read L{layer} from {}: {e}",
                self.path.display()
            ))
        })
    }

    /// Read many records at once, spread across the file's handles so the drive
    /// sees a full queue.
    ///
    /// The startup fill, where every record moves once with idle cores — so it
    /// verifies, unlike [`Self::read_into`], which is the runtime miss path and
    /// would be paying a full-record checksum inside a forward.
    pub(crate) fn read_many(&self, mut targets: Vec<PackRead<'_>>) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }
        // Everything fallible that only needs to *look* at a target happens
        // first, so the concurrent read below is issued over a list already
        // known to be well-formed and the checksum pass has its expectations in
        // hand. Resolving these inside the read would mean either aborting a
        // partly-issued queue or checking on the forward thread's behalf later.
        let mut plan = Vec::with_capacity(targets.len());
        for t in targets.iter() {
            if t.dest.len() != self.stride {
                candle::bail!(
                    "layer pack read wants a {}-byte destination, got {}",
                    self.stride,
                    t.dest.len()
                );
            }
            let idx = self.header.record_index(t.layer)?;
            let want = self.sums.get(idx).copied().ok_or_else(|| {
                candle::Error::Msg(format!("layer pack: no checksum for record {idx}"))
            })?;
            plan.push((self.offset_of(t.layer)?, want));
        }

        // **One queue over every handle, not a record at a time.** This is the
        // startup fill: dozens of ~240 MB records, and issuing them serially
        // through a single handle runs the drive at queue depth 1 — the whole
        // reason `DirectFile` holds independent descriptors. The borrow of
        // `targets` ends with this scope so the checksum pass can read them.
        {
            let mut stripes: Vec<StripeRead<'_>> = targets
                .iter_mut()
                .zip(&plan)
                .map(|(t, &(at, _))| StripeRead {
                    file_offset: at,
                    dest: t.dest,
                })
                .collect();
            self.reader
                .read_stripes_concurrent(&mut stripes)
                .map_err(|e| {
                    candle::Error::Msg(format!("layer pack read from {}: {e}", self.path.display()))
                })?;
        }

        // Verified, unlike `read_into` — this is the one path with idle cores,
        // and fletcher32 over gigabytes is worth spreading across them.
        targets
            .par_iter()
            .zip(plan.par_iter())
            .try_for_each(|(t, &(_, want))| {
                let got = fletcher32(t.dest);
                if got != want {
                    candle::bail!(
                        "layer pack L{}: checksum {got:#010x} does not match the trailer's \
                         {want:#010x} — the file is damaged and must be rewritten",
                        t.layer
                    );
                }
                Ok(())
            })
    }

    /// Open a pack and check it describes `identity` and `images`.
    ///
    /// Every mismatch is the same answer — rewrite — so they are one error type
    /// and the caller does not branch on which.
    pub(crate) fn open(
        path: &Path,
        identity: PackIdentity,
        images: &[LayerImage],
        pinned: usize,
    ) -> Result<Self> {
        let mut file = File::open(path)
            .map_err(|e| candle::Error::Msg(format!("layer pack open {}: {e}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| candle::Error::Msg(format!("layer pack stat {}: {e}", path.display())))?
            .len();

        // **A prefix, not the file.** This used to `read_to_end`, which pulls the
        // entire pack into a host `Vec` to read a few kilobytes of header — 16 GiB
        // on the 27B, at every startup, on the way to deciding whether the pack
        // is even usable. Worse than slow: an allocation that large aborts the
        // process rather than returning the `Err` that would have fallen through
        // to rebuilding the pack.
        let window = len.min(HEADER_WINDOW as u64) as usize;
        let mut head = vec![0u8; window];
        file.read_exact(&mut head)
            .map_err(|e| candle::Error::Msg(format!("layer pack read {}: {e}", path.display())))?;
        let header = PackHeader::decode(&head)?;
        if header.encoded_len() > head.len() {
            candle::bail!(
                "layer pack {} declares a {}-byte header, past the {HEADER_WINDOW}-byte \
                 window this reads — a model with more layers than the window allows for",
                path.display(),
                header.encoded_len()
            );
        }

        if header.source_len != identity.source_len
            || header.source_sum != identity.source_sum
            || header.int8_mode != identity.int8_mode
            || header.repack_fp != identity.repack_fp
        {
            candle::bail!(
                "layer pack {} was built from a different checkpoint, numeric mode or repack \
                 formula than the one in front of us",
                path.display()
            );
        }
        // **The pinned prefix has to match, and nothing else would catch it.**
        //
        // It is not in the identity (the checkpoint and repack are unchanged
        // when it moves) and not in the geometry (the per-layer spans are the
        // same either way). What it decides is `record_index`: the file holds no
        // record for the first `pinned` layers, so a pack written with a
        // different count has every record offset by the difference. The loader
        // would upload its own `PINNED_LAYERS` head, the residency would mark a
        // different set resident over memory nothing wrote, and the run would
        // report them as hits while computing on whatever the zone last held.
        if header.pinned_layers as usize != pinned {
            candle::bail!(
                "layer pack {} holds no records for its first {} layers but this build \
                 pins {pinned} — every record is offset by the difference",
                path.display(),
                header.pinned_layers
            );
        }
        check_geometry(&header, images)?;
        // **`stride` is derived, so it is checked rather than trusted.**
        //
        // Every other decoded field is now compared against something: the
        // identity against the checkpoint, `pinned_layers` against this build,
        // the spans against the images. `stride` alone came from the file
        // verbatim, and it is what turns a record index into a file offset and
        // sizes every pinned staging buffer. Corrupted larger, every record
        // after the first straddles two on disk — and the runtime miss path
        // (`read_into`) does not checksum, so on a host whose warm tier is empty
        // nothing would ever notice and the cache would upload the straddle as
        // weights. Corrupted smaller it trips a release assert deep in
        // `PinnedPool::slot_ref`, on the forward thread, far from the cause.
        // The writer's rule is one line (`round_up_sector(slot_bytes)`), so
        // restating it here costs nothing and turns both into a named error.
        let want_stride = round_up_sector(header.slot_bytes as usize) as u64;
        if header.stride != want_stride {
            candle::bail!(
                "layer pack {} declares a {}-byte record stride against {}-byte records, \
                 which should be {want_stride} — the header is corrupt",
                path.display(),
                header.stride,
                header.slot_bytes
            );
        }

        let body = round_up_sector(header.encoded_len()) as u64;
        let stride = header.stride as usize;
        let trailer_at = body + (header.stored_layers() * stride) as u64;
        let want_trailer = header.stored_layers() * 4;
        if len < trailer_at + want_trailer as u64 {
            candle::bail!(
                "layer pack {} is {len} bytes, short of the {} its header describes",
                path.display(),
                trailer_at as usize + want_trailer
            );
        }
        // Seek to the trailer rather than index into a whole-file buffer: it is
        // the last few hundred bytes of a file measured in gigabytes.
        file.seek(SeekFrom::Start(trailer_at))
            .map_err(|e| candle::Error::Msg(format!("layer pack seek {}: {e}", path.display())))?;
        let mut trailer = vec![0u8; want_trailer];
        file.read_exact(&mut trailer).map_err(|e| {
            candle::Error::Msg(format!("layer pack read trailer {}: {e}", path.display()))
        })?;
        let sums = trailer
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let reader = DirectFile::open(path).map_err(|e| {
            candle::Error::Msg(format!("layer pack direct open {}: {e}", path.display()))
        })?;
        Ok(Self {
            reader,
            path: path.to_path_buf(),
            header,
            body,
            stride,
            sums,
        })
    }
}

/// Check a pack's stored geometry against the images this build computed.
///
/// Geometry catches a change to *where* the bytes go; `repack_fp` (checked by
/// the caller) catches a change to *what they are*. Both are needed: a repack
/// that emits a different permutation at identical offsets would pass this and
/// serve subtly wrong weights for the whole model.
fn check_geometry(header: &PackHeader, images: &[LayerImage]) -> Result<()> {
    if header.num_layers as usize != images.len() {
        candle::bail!(
            "layer pack describes {} layers, the checkpoint has {}",
            header.num_layers,
            images.len()
        );
    }
    // The slot size, before the per-projection walk. It is the max over image
    // totals, so it can differ while every *offset* still agrees — the last
    // projection's reserved extent moves the total and nothing else. That is not
    // cosmetic: `stride` is derived from it and sizes the staging buffer a cold
    // read lands in, so a header claiming less than the caller will ask for
    // turns into an over-read of that buffer at the first miss.
    let want = super::descriptor::slot_bytes_for_layers(images);
    if header.slot_bytes as usize != want {
        candle::bail!(
            "layer pack slots are {} B, the checkpoint's images need {want} B",
            header.slot_bytes
        );
    }
    for (li, (spans, img)) in header.layers.iter().zip(images).enumerate() {
        if spans.kind != img.kind {
            candle::bail!(
                "layer pack L{li}: file says {:?}, the checkpoint says {:?}",
                spans.kind,
                img.kind
            );
        }
        if spans.projections.len() != img.placements.len() {
            candle::bail!(
                "layer pack L{li}: file has {} projections, the checkpoint has {}",
                spans.projections.len(),
                img.placements.len()
            );
        }
        for (p, q) in spans.projections.iter().zip(&img.placements) {
            if p.role != q.role
                || p.offset as usize != q.offset
                || p.bytes as usize != q.bytes
                || p.dtype != q.dtype
            {
                candle::bail!(
                    "layer pack L{li}: {:?} at {}+{} {:?} does not match the checkpoint's \
                     {:?} at {}+{} {:?}",
                    p.role,
                    p.offset,
                    p.bytes,
                    p.dtype,
                    q.role,
                    q.offset,
                    q.bytes,
                    q.dtype
                );
            }
        }
    }
    Ok(())
}

/// Build the header a pack of `images` needs.
pub(crate) fn header_for(
    images: &[LayerImage],
    identity: PackIdentity,
    pinned_layers: usize,
    slot_bytes: usize,
) -> PackHeader {
    PackHeader {
        num_layers: images.len() as u32,
        slot_bytes: slot_bytes as u32,
        stride: round_up_sector(slot_bytes) as u64,
        source_len: identity.source_len,
        source_sum: identity.source_sum,
        int8_mode: identity.int8_mode,
        repack_fp: identity.repack_fp,
        pinned_layers: pinned_layers as u32,
        layers: images
            .iter()
            .map(|img| LayerSpans {
                kind: img.kind,
                projections: img
                    .placements
                    .iter()
                    .map(|p| ProjectionSpan {
                        role: p.role,
                        offset: p.offset as u32,
                        bytes: p.bytes as u32,
                        dtype: p.dtype,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Writes a pack through a sibling temp file and publishes it by rename.
///
/// The temp name carries the pid and a nanosecond stamp rather than a fixed
/// `.partial`: the pack sits beside the checkpoint so several workspaces share
/// one, and two processes deciding to build it at the same moment would
/// otherwise interleave their writes into one file and both rename it into
/// place. The headers are identical, so the result would validate and serve
/// garbage.
pub(crate) struct PackWriter {
    out: Option<BufWriter<File>>,
    tmp: PathBuf,
    final_path: PathBuf,
    header: PackHeader,
    /// One record, reused and zeroed per layer — the gaps between projections
    /// move between layers, not within one.
    record: Vec<u8>,
    written: usize,
    sums: Vec<u32>,
}

impl PackWriter {
    pub(crate) fn create(final_path: &Path, header: PackHeader) -> Result<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp =
            final_path.with_extension(format!("pack.{}.{stamp:x}.partial", std::process::id()));
        if let Some(dir) = final_path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| {
                candle::Error::Msg(format!("layer pack mkdir {}: {e}", dir.display()))
            })?;
        }
        let file = File::create(&tmp)
            .map_err(|e| candle::Error::Msg(format!("layer pack create {}: {e}", tmp.display())))?;
        let mut out = BufWriter::with_capacity(8 << 20, file);
        let mut head = header.encode();
        head.resize(round_up_sector(head.len()), 0);
        out.write_all(&head)
            .map_err(|e| candle::Error::Msg(format!("layer pack write header: {e}")))?;
        let stride = header.stride as usize;
        let total = header.stored_layers();
        Ok(Self {
            out: Some(out),
            tmp,
            final_path: final_path.to_path_buf(),
            header,
            record: vec![0u8; stride],
            written: 0,
            sums: Vec::with_capacity(total),
        })
    }

    /// Append `layer`'s record. Must be called in ascending layer order, once
    /// per layer, for every layer **outside the pinned prefix**.
    ///
    /// `projections` are in image order and must match the layer's geometry
    /// exactly. Offering a pinned layer is caught here rather than silently
    /// writing a record the reader's index arithmetic does not expect.
    pub(crate) fn write_layer(&mut self, layer: usize, projections: &[&[u8]]) -> Result<()> {
        let pinned = self.header.pinned_layers as usize;
        let Some(rel) = layer.checked_sub(pinned) else {
            candle::bail!(
                "layer pack writer: L{layer} is inside the {pinned} permanently resident \
                 layers, which this pack deliberately holds no records for"
            );
        };
        if rel != self.written {
            candle::bail!(
                "layer pack writes must be sequential: L{layer} is record {rel}, {} are written",
                self.written
            );
        }
        let spans = self.header.layers.get(layer).ok_or_else(|| {
            candle::Error::Msg(format!("layer pack writer: L{layer} has no geometry"))
        })?;
        if projections.len() != spans.projections.len() {
            candle::bail!(
                "layer pack L{layer}: {} projections offered, the geometry has {}",
                projections.len(),
                spans.projections.len()
            );
        }
        // Zeroed per record: the tail past a short kind's image is padding the
        // reader checksums, so it must be deterministic rather than whatever
        // the previous layer left.
        self.record.fill(0);
        for (span, src) in spans.projections.iter().zip(projections) {
            if src.len() != span.bytes as usize {
                candle::bail!(
                    "layer pack L{layer} {:?}: projection is {} bytes, the geometry says {}",
                    span.role,
                    src.len(),
                    span.bytes
                );
            }
            let at = span.offset as usize;
            self.record[at..at + src.len()].copy_from_slice(src);
        }
        let Some(out) = self.out.as_mut() else {
            candle::bail!("layer pack writer L{layer}: already published");
        };
        out.write_all(&self.record)
            .map_err(|e| candle::Error::Msg(format!("layer pack write L{layer}: {e}")))?;
        self.sums.push(fletcher32(&self.record));
        self.written += 1;
        Ok(())
    }

    /// Flush, append the checksum trailer, and publish under the final name.
    pub(crate) fn finish(mut self) -> Result<PathBuf> {
        let expect = self.header.stored_layers();
        if self.written != expect {
            candle::bail!(
                "layer pack is short: {} of {expect} records written",
                self.written
            );
        }
        let Some(mut out) = self.out.take() else {
            candle::bail!("layer pack writer: finished twice");
        };
        // Appended rather than placed in front: the writer streams and never
        // seeks, so the only place a per-record value can go is after the
        // records it describes.
        for s in &self.sums {
            out.write_all(&s.to_le_bytes())
                .map_err(|e| candle::Error::Msg(format!("layer pack write trailer: {e}")))?;
        }
        let file = out
            .into_inner()
            .map_err(|e| candle::Error::Msg(format!("layer pack flush: {e}")))?;
        file.sync_all()
            .map_err(|e| candle::Error::Msg(format!("layer pack fsync: {e}")))?;
        drop(file);
        std::fs::rename(&self.tmp, &self.final_path).map_err(|e| {
            candle::Error::Msg(format!(
                "layer pack publish {} -> {}: {e}",
                self.tmp.display(),
                self.final_path.display()
            ))
        })?;
        Ok(std::mem::take(&mut self.final_path))
    }
}

impl Drop for PackWriter {
    fn drop(&mut self) {
        // A writer dropped without a *successful* `finish` leaves no half-written
        // pack behind: the temp is removed and the final name was never created.
        //
        // Unconditional, and deliberately so. `finish` takes `out` before it
        // writes the trailer, fsyncs and renames, so a failure anywhere in that
        // window — the volume filling on the last 256 bytes is the easy one —
        // would leave `out == None` with the temp still on disk. Its name
        // carries pid and nanos, so nothing ever reuses or overwrites it, and
        // every retry would strand another multi-GiB file beside the
        // checkpoint. On the success path the rename has already moved the temp
        // away and this removes nothing.
        let _ = std::fs::remove_file(&self.tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::layer_stream::descriptor::{
        layer_image, FfnForm, LayerTensor, MixKind, Projection,
    };
    use candle::direct_io::{AlignedScratch, DIRECT_IO_SECTOR};
    use candle::quantized::GgmlDType;

    fn proj(role: LayerTensor, rows: usize, cols: usize, bytes: usize) -> Projection {
        Projection {
            role,
            shape: [rows, cols],
            dtype: GgmlDType::Q4_KO,
            payload: bytes,
            extent: bytes,
        }
    }

    fn dn_image() -> LayerImage {
        layer_image(
            MixKind::DeltaNet,
            FfnForm::Fused,
            &[
                proj(LayerTensor::Wqkv, 10240, 5120, 512),
                proj(LayerTensor::Wz, 6144, 5120, 256),
                proj(LayerTensor::WOut, 5120, 6144, 256),
                proj(LayerTensor::FfnGateUp, 34816, 5120, 1024),
                proj(LayerTensor::FfnDown, 5120, 17408, 512),
            ],
        )
        .unwrap()
    }

    fn attn_image() -> LayerImage {
        layer_image(
            MixKind::Attention,
            FfnForm::Fused,
            &[
                proj(LayerTensor::Wq, 12288, 5120, 512),
                proj(LayerTensor::Wk, 1024, 5120, 256),
                proj(LayerTensor::Wv, 1024, 5120, 256),
                proj(LayerTensor::Wo, 5120, 6144, 256),
                proj(LayerTensor::FfnGateUp, 34816, 5120, 1024),
                proj(LayerTensor::FfnDown, 5120, 17408, 512),
            ],
        )
        .unwrap()
    }

    fn identity() -> PackIdentity {
        PackIdentity {
            source_len: 1234,
            source_sum: 0xABCD_1234,
            int8_mode: 3,
            repack_fp: 0x5566_7788_99AA_BBCC,
        }
    }

    /// Four layers, DN/DN/DN/attention — the lineage's 3:1 interleave in
    /// miniature, so both kinds land in one file.
    fn images() -> Vec<LayerImage> {
        vec![dn_image(), dn_image(), dn_image(), attn_image()]
    }

    fn payloads(img: &LayerImage, seed: u8) -> Vec<Vec<u8>> {
        img.placements
            .iter()
            .enumerate()
            .map(|(i, p)| vec![seed.wrapping_add(i as u8); p.bytes])
            .collect()
    }

    fn write_pack(dir: &Path, imgs: &[LayerImage], pinned: usize) -> PathBuf {
        let slot = crate::models::layer_stream::slot_bytes_for_layers(imgs);
        let header = header_for(imgs, identity(), pinned, slot);
        let path = dir.join("layers.pack");
        let mut w = PackWriter::create(&path, header).unwrap();
        for (li, img) in imgs.iter().enumerate().skip(pinned) {
            let p = payloads(img, li as u8 * 16);
            let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
            w.write_layer(li, &refs).unwrap();
        }
        w.finish().unwrap()
    }

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "candle_layer_pack_{}_{}_{name}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A sector-aligned destination of exactly one stride — what the warm pool
    /// gives for free and a `Vec` does not. See [`LayerPack::read_into`].
    fn scratch(pack: &LayerPack) -> AlignedScratch {
        let mut s = AlignedScratch::new();
        s.ensure(pack.stride()).unwrap();
        s
    }

    #[test]
    fn a_written_pack_reads_its_records_back() {
        let dir = tmpdir("roundtrip");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        let pack = LayerPack::open(&path, identity(), &imgs, 2).unwrap();

        assert_eq!(pack.pinned_layers(), 2);
        assert_eq!(pack.stored_layers(), 2);
        assert_eq!(pack.path(), path.as_path());

        let mut s = scratch(&pack);
        for (li, img) in imgs.iter().enumerate().take(4).skip(2) {
            let buf = s.as_mut_slice(pack.stride());
            pack.read_into(li, buf).unwrap();
            let expect = payloads(img, li as u8 * 16);
            let spans = pack.spans(li).unwrap();
            for (span, want) in spans.projections.iter().zip(&expect) {
                let (at, n) = (span.offset as usize, span.bytes as usize);
                assert_eq!(
                    &buf[at..at + n],
                    want.as_slice(),
                    "L{li} projection at {at}"
                );
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_records_stride_is_a_whole_number_of_sectors() {
        // The property the alignment contract rests on: a warm slot cut to this
        // stride starts on a sector however many slots precede it.
        let dir = tmpdir("stride");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        let pack = LayerPack::open(&path, identity(), &imgs, 2).unwrap();
        assert_eq!(pack.stride() % DIRECT_IO_SECTOR, 0);
        assert!(pack.stride() >= crate::models::layer_stream::slot_bytes_for_layers(&imgs));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_identity_tracks_the_checkpoint_the_mode_and_the_repack() {
        let gguf = vec![7u8; 8192];
        let a = PackIdentity::of(&gguf, Int8Mode::Performance, 0x1234);
        assert_eq!(a.source_len, 8192);
        // Same bytes, same mode, same formula → same identity.
        assert_eq!(a, PackIdentity::of(&gguf, Int8Mode::Performance, 0x1234));
        // A different repack formula at identical bytes is a different pack —
        // the case geometry alone cannot catch.
        assert_ne!(a, PackIdentity::of(&gguf, Int8Mode::Performance, 0x1235));
        // A different numeric mode targets a different KO twin.
        assert_ne!(a, PackIdentity::of(&gguf, Int8Mode::Precision, 0x1234));
        // A changed byte inside the identity window changes the sum.
        let mut other = gguf.clone();
        other[16] ^= 1;
        assert_ne!(a, PackIdentity::of(&other, Int8Mode::Performance, 0x1234));
    }

    #[test]
    fn a_pinned_layer_has_no_record() {
        let dir = tmpdir("pinned");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        let pack = LayerPack::open(&path, identity(), &imgs, 2).unwrap();
        let mut s = scratch(&pack);
        let err = pack
            .read_into(1, s.as_mut_slice(pack.stride()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("inside the pinned prefix"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_short_kinds_tail_is_zeroed_not_left_over() {
        // A DeltaNet record is narrower than the attention image the stride is
        // cut to. The tail must be deterministic, or the trailer's checksum
        // depends on write order.
        let dir = tmpdir("tail");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 0);
        let pack = LayerPack::open(&path, identity(), &imgs, 0).unwrap();
        let mut s = scratch(&pack);
        let buf = s.as_mut_slice(pack.stride());
        buf.fill(0xFF);
        pack.read_into(0, buf).unwrap();
        let img = &imgs[0];
        let end = img.placements.last().unwrap().offset + img.placements.last().unwrap().bytes;
        assert!(
            buf[end..].iter().all(|&b| b == 0),
            "the tail past a DeltaNet image must be zero"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupt_record_stride_is_refused_at_open() {
        // `stride` decides every record offset and every staging-buffer size,
        // and it is derived (`round_up_sector(slot_bytes)`), so the file is not
        // its authority. Corrupted larger, every record after the first
        // straddles two on disk — and `read_into`, the runtime miss path, does
        // not checksum, so a host with an empty warm tier would upload the
        // straddle as weights and never notice.
        let dir = tmpdir("stride");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        let mut raw = std::fs::read(&path).unwrap();
        let bad = u64::from_le_bytes(raw[20..28].try_into().unwrap()) + 4096;
        raw[20..28].copy_from_slice(&bad.to_le_bytes());
        std::fs::write(&path, &raw).unwrap();

        let err = LayerPack::open(&path, identity(), &imgs, 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("record stride"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_trailer_catches_a_damaged_record() {
        let dir = tmpdir("damaged");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        // Flip a byte inside record 0 (layer 2), past the padded header.
        let mut raw = std::fs::read(&path).unwrap();
        let body = round_up_sector(
            header_for(
                &imgs,
                identity(),
                2,
                crate::models::layer_stream::slot_bytes_for_layers(&imgs),
            )
            .encoded_len(),
        );
        raw[body] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();

        let pack = LayerPack::open(&path, identity(), &imgs, 2).unwrap();
        let mut s = scratch(&pack);
        let stride = pack.stride();
        let err = pack
            .read_many(vec![PackRead {
                layer: 2,
                dest: s.as_mut_slice(stride),
            }])
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not match the trailer"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_pack_from_another_checkpoint_is_refused() {
        let dir = tmpdir("identity");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        let mut other = identity();
        other.source_sum ^= 1;
        let err = LayerPack::open(&path, other, &imgs, 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("different checkpoint"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A pack written with a different pinned prefix is refused.
    ///
    /// Nothing else catches it: the checkpoint and repack are unchanged, so the
    /// identity matches, and the per-layer spans are identical, so the geometry
    /// matches. What differs is `record_index` — the file holds no record for
    /// the pinned head, so a different count offsets **every** record. The
    /// residency would then mark a set resident over memory nothing wrote and
    /// report it as hits.
    #[test]
    fn a_pack_with_a_different_pinned_prefix_is_refused() {
        let dir = tmpdir("pinned_prefix");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        let err = LayerPack::open(&path, identity(), &imgs, 3)
            .unwrap_err()
            .to_string();
        assert!(err.contains("holds no records for its first 2"), "{err}");
        // The count it was written with still opens.
        assert!(LayerPack::open(&path, identity(), &imgs, 2).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_pack_whose_layer_kind_moved_is_refused() {
        let dir = tmpdir("geometry");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        // Same checkpoint, same repack, but this build now places the layer
        // differently — the case a version number would not catch. A different
        // kind is also a different image size, so the slot check is what names
        // it; either refusal is the pack being rejected, which is the point.
        let mut moved = images();
        moved[3] = dn_image();
        let err = LayerPack::open(&path, identity(), &moved, 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("slots are"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_pack_whose_placement_moved_under_an_unchanged_slot_size_is_refused() {
        // The narrow case the slot-size check cannot see: the images still total
        // the same, so the stride and every allocation agree, and only a single
        // projection's *form* has changed. Left unchecked this serves a weight
        // the kernels will read as the wrong dtype.
        let dir = tmpdir("placement");
        let imgs = images();
        let path = write_pack(&dir, &imgs, 2);
        let mut moved = images();
        moved[3].placements[0].dtype = GgmlDType::Q8_KO;
        assert_eq!(
            crate::models::layer_stream::slot_bytes_for_layers(&moved),
            crate::models::layer_stream::slot_bytes_for_layers(&imgs),
            "the fixture must change only the form, or the slot check answers first"
        );
        let err = LayerPack::open(&path, identity(), &moved, 2)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("file says") || err.contains("does not match"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn out_of_order_writes_are_refused() {
        let dir = tmpdir("order");
        let imgs = images();
        let slot = crate::models::layer_stream::slot_bytes_for_layers(&imgs);
        let header = header_for(&imgs, identity(), 2, slot);
        let path = dir.join("layers.pack");
        let mut w = PackWriter::create(&path, header).unwrap();
        let p = payloads(&imgs[3], 48);
        let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
        let err = w.write_layer(3, &refs).unwrap_err().to_string();
        assert!(err.contains("must be sequential"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_wrong_sized_projection_is_refused() {
        let dir = tmpdir("size");
        let imgs = images();
        let slot = crate::models::layer_stream::slot_bytes_for_layers(&imgs);
        let header = header_for(&imgs, identity(), 2, slot);
        let path = dir.join("layers.pack");
        let mut w = PackWriter::create(&path, header).unwrap();
        let mut p = payloads(&imgs[2], 32);
        p[0].push(0);
        let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
        let err = w.write_layer(2, &refs).unwrap_err().to_string();
        assert!(err.contains("the geometry says"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unfinished_writer_leaves_nothing_behind() {
        let dir = tmpdir("abandoned");
        let imgs = images();
        let slot = crate::models::layer_stream::slot_bytes_for_layers(&imgs);
        let header = header_for(&imgs, identity(), 2, slot);
        let path = dir.join("layers.pack");
        {
            let mut w = PackWriter::create(&path, header).unwrap();
            let p = payloads(&imgs[2], 32);
            let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
            w.write_layer(2, &refs).unwrap();
            // dropped without finish
        }
        assert!(!path.exists(), "the final name must never appear");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect();
        assert!(leftovers.is_empty(), "temp files left: {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_short_pack_is_refused() {
        let dir = tmpdir("short");
        let imgs = images();
        let slot = crate::models::layer_stream::slot_bytes_for_layers(&imgs);
        let header = header_for(&imgs, identity(), 2, slot);
        let path = dir.join("layers.pack");
        let mut w = PackWriter::create(&path, header).unwrap();
        let p = payloads(&imgs[2], 32);
        let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
        w.write_layer(2, &refs).unwrap();
        let err = w.finish().unwrap_err().to_string();
        assert!(err.contains("is short"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
