//! The cold tier: a section of the model pack holding every streamed layer,
//! always, repacked.
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
//! The section is the model's **only** copy of its streamed projections: the
//! model pack carries no checkpoint beside it, and every layer — the
//! permanently resident head included — has a record.
//!
//! # The invariant this exists to hold
//!
//! > **The cold tier holds a valid copy of every layer, always.**
//!
//! Everything residency does follows from it: eviction is a bookkeeping change
//! with no copy and no destination to find, the warm tier needs no eviction
//! policy, and "where do I load this from" is a total function. See
//! `docs/archived/qwen38_layer_streaming.md` §4.
//!
//! # Records are sector-aligned because the reads bypass the page cache
//!
//! Reads go through [`candle::direct_io`], which requires the file offset, the
//! length and the destination pointer to be 4 KiB-aligned. The section starts
//! on a sector, a record's stride is the slot image padded up to a sector, and
//! the warm pool's slots are cut to the same stride — which is what lets a cold
//! read land *directly* in a pinned slot with no bounce buffer.
//!
//! # One record width for two layer kinds
//!
//! A layer record's *contents* vary: a DeltaNet layer holds fewer projections
//! than an attention layer. The stride does not vary — it is the widest image,
//! padded — so the shorter kind leaves a zeroed tail. That is the same
//! uniformity the weight zone requires of its slots, arrived at for the same
//! reason, and it costs ~2% on Qwen3.8-27B.

mod header;

use candle::direct_io::{round_up_sector, DirectFile, StripeRead, DIRECT_IO_SECTOR};
use candle::fletcher::fletcher32;
use candle::quantized::GgmlDType;
use candle::Result;
use header::{LayerSpans, ProjectionSpan};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(crate) use header::{pairs_in, PackHeader};

use crate::models::layer_stream::descriptor::{layer_image, LayerImage, Projection};
use crate::models::repack_fingerprint::{check_prints, PairPrint};

/// Bytes read from the start of a section to decode its header.
///
/// The header is variable-width, so its length is not known until it is
/// decoded, and this is the window that decode happens inside. A 64-layer
/// model's header is under 16 KiB, so a mebibyte is three orders of magnitude
/// of slack. A header past the window is refused rather than chased with a
/// second read: it would mean thousands of layers.
const HEADER_WINDOW: usize = 1024 * 1024;

/// A read request: one record into one stride-long aligned destination.
pub(crate) struct PackRead<'a> {
    pub layer: usize,
    pub dest: &'a mut [u8],
}

/// Where the records start, from the start of the section.
fn records_offset(header: &PackHeader) -> u64 {
    round_up_sector(header.encoded_len()) as u64
}

/// The whole section's length: the padded header, a record per layer, and a
/// checksum per record.
pub(crate) fn section_len(header: &PackHeader) -> u64 {
    let n = header.num_layers as u64;
    records_offset(header) + n * header.stride + n * 4
}

/// The open layer section.
pub(crate) struct LayerPack {
    reader: DirectFile,
    path: PathBuf,
    header: PackHeader,
    /// Byte offset of record 0 from the start of the file.
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
            .field("stride", &self.stride)
            .finish()
    }
}

impl LayerPack {
    /// Record-to-record distance, and so the size of every destination buffer.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    pub(crate) fn header(&self) -> &PackHeader {
        &self.header
    }

    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Byte offset of `layer`'s record.
    fn offset_of(&self, layer: usize) -> Result<u64> {
        if layer >= self.header.num_layers as usize {
            candle::bail!(
                "layer pack: layer {layer} is past the model's {} layers",
                self.header.num_layers
            );
        }
        Ok(self.body + layer as u64 * self.stride as u64)
    }

    /// Read one layer's record into `dest`, which must be exactly one stride
    /// and sector-aligned — a warm-pool or staging slot, which the pools cut to
    /// this stride. Not checksummed: this is the runtime miss path.
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
    /// sees a full queue, and verify each against the trailer.
    ///
    /// The startup fill, where every record moves once with idle cores — so it
    /// verifies, unlike [`Self::read_into`], which is the runtime miss path and
    /// would be paying a full-record checksum inside a forward.
    pub(crate) fn read_many(&self, mut targets: Vec<PackRead<'_>>) -> Result<()> {
        if targets.is_empty() {
            return Ok(());
        }
        let mut plan = Vec::with_capacity(targets.len());
        for t in targets.iter() {
            if t.dest.len() != self.stride {
                candle::bail!(
                    "layer pack read wants a {}-byte destination, got {}",
                    self.stride,
                    t.dest.len()
                );
            }
            let want = self.sums.get(t.layer).copied().ok_or_else(|| {
                candle::Error::Msg(format!("layer pack: no checksum for layer {}", t.layer))
            })?;
            plan.push((self.offset_of(t.layer)?, want));
        }
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

    /// Open the layer section at byte `base` of `path`.
    ///
    /// Refuses a section that is not this build's: wrong magic or version, a
    /// repack whose `fingerprint` no longer matches, a stride that is not its
    /// slot size padded to a sector, or a file too short for its records.
    pub(crate) fn open_section(
        path: &Path,
        base: u64,
        fingerprint: impl Fn(GgmlDType, GgmlDType) -> u64,
    ) -> Result<Self> {
        if !base.is_multiple_of(DIRECT_IO_SECTOR as u64) {
            candle::bail!(
                "layer pack: the section at {base} in {} does not start on a sector",
                path.display()
            );
        }
        let at = |e: std::io::Error| {
            candle::Error::Msg(format!("layer pack read {}: {e}", path.display()))
        };
        let mut file = File::open(path).map_err(at)?;
        let len = file.metadata().map_err(at)?.len();
        // **A prefix, not the file**: the header is a few KiB at the front of a
        // section measured in GiB.
        let window = len.saturating_sub(base).min(HEADER_WINDOW as u64) as usize;
        let mut head = vec![0u8; window];
        file.seek(SeekFrom::Start(base)).map_err(at)?;
        file.read_exact(&mut head).map_err(at)?;
        let header = PackHeader::decode(&head)?;
        check_prints(&header.pairs, fingerprint)?;
        // **`stride` is derived, so it is checked rather than trusted.** It turns
        // a record index into a file offset and sizes every staging buffer;
        // corrupted larger, every record after the first straddles two on disk,
        // and the runtime miss path does not checksum.
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
        let body = base + records_offset(&header);
        let n = header.num_layers as usize;
        let stride = header.stride as usize;
        let trailer_at = body + (n * stride) as u64;
        if len < trailer_at + (n * 4) as u64 {
            candle::bail!(
                "layer pack {} is {len} bytes, short of the {} its header describes — truncated",
                path.display(),
                trailer_at + (n * 4) as u64
            );
        }
        file.seek(SeekFrom::Start(trailer_at)).map_err(at)?;
        let mut trailer = vec![0u8; n * 4];
        file.read_exact(&mut trailer).map_err(at)?;
        let sums = trailer
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        let reader = DirectFile::open(path).map_err(at)?;
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

/// Every layer's image, rebuilt from a section header by this build's own
/// placement rules — and checked against where the header says the build that
/// wrote it placed each projection.
///
/// A section written by a build whose placement has since changed is refused
/// here by comparing offsets, rather than trusted to a version someone had to
/// remember to bump.
pub(crate) fn images_of(header: &PackHeader) -> Result<Vec<LayerImage>> {
    let mut images = Vec::with_capacity(header.layers.len());
    for (li, spans) in header.layers.iter().enumerate() {
        let projections: Vec<Projection> = spans
            .projections
            .iter()
            .map(|p| Projection {
                role: p.role,
                shape: [p.rows as usize, p.cols as usize],
                dtype: p.dtype,
                payload: p.bytes as usize,
                extent: p.extent as usize,
            })
            .collect();
        let image = layer_image(spans.kind, spans.ffn, &projections)
            .map_err(|e| candle::Error::Msg(format!("layer pack L{li}: {e}")))?;
        for (placed, recorded) in image.placements.iter().zip(&spans.projections) {
            if placed.role != recorded.role || placed.offset != recorded.offset as usize {
                candle::bail!(
                    "layer pack L{li}: {:?} recorded at {}, this build places it at {} — the \
                     layout changed since the pack was built",
                    recorded.role,
                    recorded.offset,
                    placed.offset
                );
            }
        }
        images.push(image);
    }
    let want = super::descriptor::slot_bytes_for_layers(&images);
    if header.slot_bytes as usize != want {
        candle::bail!(
            "layer pack slots are {} B, this build's images need {want} B",
            header.slot_bytes
        );
    }
    Ok(images)
}

/// The header a section of `images` needs. `src_dtype(layer, role)` is the
/// checkpoint's dtype for each projection; `prints` the fingerprint of every
/// pair, in [`pairs_in`] order.
pub(crate) fn header_for(
    images: &[LayerImage],
    src_dtype: &dyn Fn(usize, &Projection) -> GgmlDType,
    int8_mode: u32,
    fingerprint: &dyn Fn(GgmlDType, GgmlDType) -> u64,
) -> PackHeader {
    let slot_bytes = super::descriptor::slot_bytes_for_layers(images);
    let layers: Vec<LayerSpans> = images
        .iter()
        .enumerate()
        .map(|(li, img)| LayerSpans {
            kind: img.kind,
            ffn: img.ffn,
            projections: img
                .placements
                .iter()
                .map(|p| {
                    let proj = Projection {
                        role: p.role,
                        shape: p.shape,
                        dtype: p.dtype,
                        payload: p.bytes,
                        extent: p.extent,
                    };
                    ProjectionSpan {
                        role: p.role,
                        offset: p.offset as u32,
                        bytes: p.bytes as u32,
                        extent: p.extent as u32,
                        dtype: p.dtype,
                        src_dtype: src_dtype(li, &proj),
                        rows: p.shape[0] as u32,
                        cols: p.shape[1] as u32,
                    }
                })
                .collect(),
        })
        .collect();
    let pairs = pairs_in(&layers)
        .into_iter()
        .map(|(src_dtype, dtype)| PairPrint {
            src_dtype,
            dtype,
            fp: fingerprint(src_dtype, dtype),
        })
        .collect();
    PackHeader {
        num_layers: images.len() as u32,
        slot_bytes: slot_bytes as u32,
        stride: round_up_sector(slot_bytes) as u64,
        int8_mode,
        layers,
        pairs,
    }
}

/// Writes a layer section, record by record in layer order, into whatever the
/// model pack's build hands it.
pub(crate) struct PackWriter<'a, W: Write + ?Sized> {
    out: &'a mut W,
    header: PackHeader,
    /// One record, reused and zeroed per layer — the tail past a short kind's
    /// image is checksummed, so it must be deterministic.
    record: Vec<u8>,
    written: usize,
    sums: Vec<u32>,
}

impl<'a, W: Write + ?Sized> PackWriter<'a, W> {
    /// Write `header`, padded to a sector, and stand ready for the records.
    pub(crate) fn new(out: &'a mut W, header: PackHeader) -> Result<Self> {
        let mut head = header.encode();
        head.resize(round_up_sector(head.len()), 0);
        out.write_all(&head)
            .map_err(|e| candle::Error::Msg(format!("layer pack write header: {e}")))?;
        let stride = header.stride as usize;
        let n = header.num_layers as usize;
        Ok(Self {
            out,
            header,
            record: vec![0u8; stride],
            written: 0,
            sums: Vec::with_capacity(n),
        })
    }

    /// Append `layer`'s record. Must be called in ascending layer order, once
    /// per layer, starting at layer 0. `projections` are in image order.
    pub(crate) fn write_layer(&mut self, layer: usize, projections: &[&[u8]]) -> Result<()> {
        if layer != self.written {
            candle::bail!(
                "layer pack writes must be sequential: L{layer} offered, {} are written",
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
        self.out
            .write_all(&self.record)
            .map_err(|e| candle::Error::Msg(format!("layer pack write L{layer}: {e}")))?;
        self.sums.push(fletcher32(&self.record));
        self.written += 1;
        Ok(())
    }

    /// Append the checksum trailer, refusing a section with layers unwritten,
    /// and return the section's length.
    pub(crate) fn finish(self) -> Result<u64> {
        let expect = self.header.num_layers as usize;
        if self.written != expect {
            candle::bail!(
                "layer pack is short: {} of {expect} records written",
                self.written
            );
        }
        for s in &self.sums {
            self.out
                .write_all(&s.to_le_bytes())
                .map_err(|e| candle::Error::Msg(format!("layer pack write trailer: {e}")))?;
        }
        Ok(section_len(&self.header))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::layer_stream::descriptor::{FfnForm, LayerTensor, MixKind};
    use candle::direct_io::AlignedScratch;
    use std::io::BufWriter;

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

    /// Four layers, DN/DN/DN/attention — the lineage's 3:1 interleave in
    /// miniature, so both kinds land in one section.
    fn images() -> Vec<LayerImage> {
        vec![dn_image(), dn_image(), dn_image(), attn_image()]
    }

    fn src(_: usize, _: &Projection) -> GgmlDType {
        GgmlDType::Q4_K
    }

    fn fp(_: GgmlDType, _: GgmlDType) -> u64 {
        0x77
    }

    fn payloads(img: &LayerImage, seed: u8) -> Vec<Vec<u8>> {
        img.placements
            .iter()
            .enumerate()
            .map(|(i, p)| vec![seed.wrapping_add(i as u8); p.bytes])
            .collect()
    }

    /// The fixture section after `prefix` bytes of something else, as it sits
    /// in a model pack. Returns the file and the section's offset.
    fn write_section(tag: &str, prefix: usize) -> (PathBuf, u64) {
        let path = std::env::temp_dir().join(format!(
            "candle_layer_section_{tag}_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let mut out = BufWriter::new(File::create(&path).unwrap());
        out.write_all(&vec![0xEE; prefix]).unwrap();
        let imgs = images();
        let mut w = PackWriter::new(&mut out, header_for(&imgs, &src, 3, &fp)).unwrap();
        for (li, img) in imgs.iter().enumerate() {
            let p = payloads(img, li as u8 * 16);
            let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
            w.write_layer(li, &refs).unwrap();
        }
        let len = w.finish().unwrap();
        out.flush().unwrap();
        drop(out);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), prefix as u64 + len);
        (path, prefix as u64)
    }

    fn scratch(pack: &LayerPack) -> AlignedScratch {
        let mut s = AlignedScratch::new();
        s.ensure(pack.stride()).unwrap();
        s
    }

    /// Every layer — the first included — reads back at the offsets the decoded
    /// header names, closing the loop through the bytes on disk.
    #[test]
    fn a_written_section_reads_every_layer_back() {
        let (path, base) = write_section("roundtrip", 8192);
        let pack = LayerPack::open_section(&path, base, fp).unwrap();
        assert_eq!(pack.path(), path.as_path());
        let imgs = images();
        let mut s = scratch(&pack);
        for (li, img) in imgs.iter().enumerate() {
            let buf = s.as_mut_slice(pack.stride());
            pack.read_into(li, buf).unwrap();
            let expect = payloads(img, li as u8 * 16);
            for (span, want) in pack.header().layers[li].projections.iter().zip(&expect) {
                let (at, n) = (span.offset as usize, span.bytes as usize);
                assert_eq!(&buf[at..at + n], want.as_slice(), "L{li} at {at}");
            }
        }
        std::fs::remove_file(&path).ok();
    }

    /// The images rebuilt from the header are the images the section was
    /// written from.
    #[test]
    fn the_images_rebuild_from_the_header() {
        let (path, base) = write_section("images", 4096);
        let pack = LayerPack::open_section(&path, base, fp).unwrap();
        assert_eq!(images_of(pack.header()).unwrap(), images());
        std::fs::remove_file(&path).ok();
    }

    /// A header whose recorded offsets no longer match this build's placement
    /// is refused, naming the projection.
    #[test]
    fn a_moved_placement_is_refused() {
        let mut h = header_for(&images(), &src, 3, &fp);
        h.layers[0].projections[1].offset += 256;
        let e = images_of(&h).unwrap_err().to_string();
        assert!(e.contains("layout changed"), "{e}");
    }

    #[test]
    fn a_records_stride_is_a_whole_number_of_sectors() {
        let (path, base) = write_section("stride", 4096);
        let pack = LayerPack::open_section(&path, base, fp).unwrap();
        assert_eq!(pack.stride() % DIRECT_IO_SECTOR, 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_short_kinds_tail_is_zeroed_not_left_over() {
        let (path, base) = write_section("tail", 4096);
        let pack = LayerPack::open_section(&path, base, fp).unwrap();
        let mut s = scratch(&pack);
        let buf = s.as_mut_slice(pack.stride());
        buf.fill(0xFF);
        pack.read_into(0, buf).unwrap();
        let img = &images()[0];
        let last = img.placements.last().unwrap();
        assert!(buf[last.offset + last.bytes..].iter().all(|&b| b == 0));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_corrupt_record_stride_is_refused_at_open() {
        let (path, base) = write_section("badstride", 4096);
        let mut raw = std::fs::read(&path).unwrap();
        let at = base as usize + 20;
        let bad = u64::from_le_bytes(raw[at..at + 8].try_into().unwrap()) + 4096;
        raw[at..at + 8].copy_from_slice(&bad.to_le_bytes());
        std::fs::write(&path, &raw).unwrap();
        let err = LayerPack::open_section(&path, base, fp)
            .unwrap_err()
            .to_string();
        assert!(err.contains("record stride"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_trailer_catches_a_damaged_record() {
        let (path, base) = write_section("damaged", 4096);
        let body = {
            let pack = LayerPack::open_section(&path, base, fp).unwrap();
            pack.offset_of(2).unwrap() as usize
        };
        let mut raw = std::fs::read(&path).unwrap();
        raw[body] ^= 0xFF;
        std::fs::write(&path, &raw).unwrap();
        let pack = LayerPack::open_section(&path, base, fp).unwrap();
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
        std::fs::remove_file(&path).ok();
    }

    /// A repack whose output moved for a pair the section holds refuses the open.
    #[test]
    fn a_changed_repack_refuses_the_open() {
        let (path, base) = write_section("fingerprint", 4096);
        let err = LayerPack::open_section(&path, base, |_, _| 0x78)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Q4_K"), "{err}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn out_of_order_writes_are_refused() {
        let imgs = images();
        let mut out = Vec::new();
        let mut w = PackWriter::new(&mut out, header_for(&imgs, &src, 3, &fp)).unwrap();
        let p = payloads(&imgs[1], 16);
        let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
        let err = w.write_layer(1, &refs).unwrap_err().to_string();
        assert!(err.contains("must be sequential"), "{err}");
    }

    #[test]
    fn a_wrong_sized_projection_is_refused() {
        let imgs = images();
        let mut out = Vec::new();
        let mut w = PackWriter::new(&mut out, header_for(&imgs, &src, 3, &fp)).unwrap();
        let mut p = payloads(&imgs[0], 0);
        p[0].push(0);
        let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
        let err = w.write_layer(0, &refs).unwrap_err().to_string();
        assert!(err.contains("the geometry says"), "{err}");
    }

    #[test]
    fn a_short_section_is_refused() {
        let imgs = images();
        let mut out = Vec::new();
        let mut w = PackWriter::new(&mut out, header_for(&imgs, &src, 3, &fp)).unwrap();
        let p = payloads(&imgs[0], 0);
        let refs: Vec<&[u8]> = p.iter().map(|v| v.as_slice()).collect();
        w.write_layer(0, &refs).unwrap();
        let err = w.finish().unwrap_err().to_string();
        assert!(err.contains("is short"), "{err}");
    }
}
