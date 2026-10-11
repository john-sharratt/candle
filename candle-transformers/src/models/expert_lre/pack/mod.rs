//! The cold tier: a section of the model pack that holds every expert, always,
//! repacked.
//!
//! The section stores experts in the **repacked layout the kernels consume** —
//! one contiguous record per expert, gate/up/down at the offsets a VRAM slot
//! uses — rather than the original GGUF tensors. Three reasons, in order of
//! weight:
//!
//! 1. **The repack is hot-path poison.** Repacking 6,144 experts takes ~42 s,
//!    about 7 ms each, and a forward issues on the order of 1,150 expert loads.
//!    Repacking on load would cost seconds per forward.
//! 2. **A repacked expert is already one blob**, so a load is an offset and a
//!    copy. In the GGUF the same expert is a strided slice of three stacked
//!    per-tensor arrays, so loading one is a gather plus a dequantise.
//! 3. **It decouples the hot path from the checkpoint format**, so GGUF packing
//!    decisions cannot become cache performance regressions.
//!
//! The section is the model's **only** copy of its experts: the model pack
//! (`crate::models::model_pack`) carries no checkpoint beside it. The checkpoint
//! is read once, by the build, and every expert of every layer is written here.
//!
//! # The invariant this exists to hold
//!
//! > The cold tier holds a valid copy of every expert, always.
//!
//! Everything the cache does with residency follows from it: eviction from VRAM
//! is `vram = None` with no copy and no destination to find, the warm tier needs
//! no eviction policy, and "where do I load this from" is a total function.
//! `docs/expert_cache_design.md` is the design; this module is its floor.
//!
//! # Records are sector-aligned because the reads bypass the page cache
//!
//! Reads go through [`candle::direct_io`], which requires the file offset, the
//! length and the destination pointer to be 4 KiB-aligned. So the section starts
//! on a sector, a record's stride is the slot image padded up to a sector, and
//! every record therefore starts on one. The warm pool's slots are cut to the
//! same stride, which is what lets a cold read land *directly* in a pinned slot
//! with no bounce buffer.

mod header;

use candle::direct_io::{round_up_sector, DirectFile, StripeRead, DIRECT_IO_SECTOR};
use candle::fletcher::fletcher32;
use candle::quantized::GgmlDType;
use candle::Result;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub(crate) use header::{pairs_in, LayerSpans, PackHeader, ProjectionSpan};

use crate::models::repack_fingerprint::check_prints;

/// Where one of an expert's three projections sits inside a record.
///
/// The caller reads a record into a buffer and then issues three H2D copies
/// against these spans, so this is the only thing it needs to know about the
/// section's interior.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecordSpan {
    pub offset: usize,
    pub bytes: usize,
}

impl From<ProjectionSpan> for RecordSpan {
    fn from(p: ProjectionSpan) -> Self {
        Self {
            offset: p.offset as usize,
            bytes: p.bytes as usize,
        }
    }
}

/// The geometry of one expert record — where each projection lives in it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecordLayout {
    pub gate: RecordSpan,
    pub up: RecordSpan,
    pub down: RecordSpan,
}

impl From<LayerSpans> for RecordLayout {
    fn from(l: LayerSpans) -> Self {
        Self {
            gate: l.gate.into(),
            up: l.up.into(),
            down: l.down.into(),
        }
    }
}

/// One record to fetch in a batch: which expert, and where its bytes go.
///
/// `dest` must be exactly one stride long and 4 KiB-aligned — in practice a
/// warm-pool slot or a pinned staging buffer, both of which satisfy that by
/// construction.
pub(crate) struct PackRead<'a> {
    pub layer: usize,
    pub expert: usize,
    pub dest: &'a mut [u8],
}

/// How the section after `header` is laid out: the padded header, the records,
/// then one checksum per record.
fn records_offset(header: &PackHeader) -> u64 {
    round_up_sector(header.encoded_len()) as u64
}

/// The whole section's length in bytes, for a model pack laying out its file.
pub(crate) fn section_len(header: &PackHeader) -> u64 {
    let total = header.total_experts() as u64;
    records_offset(header) + total * header.stride + total * 4
}

/// The open expert section: every expert, in kernel-ready form, readable at any
/// time.
pub(crate) struct ExpertPack {
    path: PathBuf,
    /// Every read goes to the drive; the pack keeps no host copy of a record.
    ///
    /// Two in-process record caches were measured here and removed. One sized
    /// at a quarter of physical RAM, never evicting and outside the host budget,
    /// held 8.2 GiB beside a 13 GiB warm tier on the 31.5 GiB box and drove free
    /// RAM to 1 GiB. A 4 GiB LRU served 7 % of pack loads — each forward sweeps
    /// every layer, a cycle far larger than the cache, which is LRU's worst
    /// case — while the store on every miss cost 12–25 % of throughput.
    reader: DirectFile,
    /// Where the first record starts, from the start of the file.
    records_at: u64,
    stride: usize,
    experts_per_layer: usize,
    header: PackHeader,
    layouts: Vec<RecordLayout>,
    /// One checksum per record, in index order, read from the trailer at open.
    ///
    /// **A trailer rather than a table between the header and the records**,
    /// because the writer never seeks: it accumulates these as it streams the
    /// records out and appends them at the end.
    sums: Vec<u32>,
}

impl ExpertPack {
    /// Record-to-record distance, and therefore the size of a read.
    ///
    /// Also the warm pool's slot size, so a cold read can land straight in a
    /// pinned slot.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// The section's header — the geometry the expert cache sizes itself from.
    pub(crate) fn header(&self) -> &PackHeader {
        &self.header
    }

    /// Where the three projections sit inside `layer`'s records.
    pub(crate) fn layout(&self, layer: usize) -> RecordLayout {
        self.layouts[layer]
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Flat record index of `(layer, expert)`.
    fn record_index(&self, layer: usize, expert: usize) -> Result<usize> {
        if layer >= self.layouts.len() || expert >= self.experts_per_layer {
            candle::bail!(
                "expert pack: L{layer}E{expert} is outside the section's {} layers of {} experts",
                self.layouts.len(),
                self.experts_per_layer
            );
        }
        Ok(layer * self.experts_per_layer + expert)
    }

    /// Byte offset of `(layer, expert)`'s record.
    fn offset_of(&self, layer: usize, expert: usize) -> Result<u64> {
        Ok(self.records_at + (self.record_index(layer, expert)? * self.stride) as u64)
    }

    /// Check a record against the checksum written with it.
    ///
    /// **What this catches is the storage, not the writer.** A half-built model
    /// pack cannot be published (the build streams to a private temp file and
    /// only renames a complete one). What is left is the medium: bit rot, a bad
    /// sector, a truncating filesystem — on a file whose contents become weights
    /// with no further validation.
    ///
    /// # Why only the bulk path calls this
    ///
    /// It is checked on [`Self::read_many`] — the startup fill, where thousands
    /// of records move at once, the cores are idle waiting on the drive, and the
    /// work parallelises. It is **not** checked on [`Self::read_into_with_handle`],
    /// the stager's per-miss path, and that is a measured decision rather than
    /// an oversight: a `fletcher32` over 2.9 MB costs about as much as the read
    /// it follows, in front of a worker that is waiting for it. On the miss path
    /// it once cost the gate **more than half its throughput** — 723 → 299 t/s on
    /// the narrowest config — for ~850 records per config.
    fn verify(&self, layer: usize, expert: usize, record: &[u8]) -> Result<()> {
        let idx = self.record_index(layer, expert)?;
        let Some(&want) = self.sums.get(idx) else {
            return Ok(());
        };
        let got = fletcher32(record);
        if got != want {
            candle::bail!(
                "expert pack L{layer}E{expert} is corrupt in {}: checksum {got:#010x}, \
                 expected {want:#010x}. Delete the model pack to rebuild it.",
                self.path.display()
            );
        }
        Ok(())
    }

    /// Read one expert's record into `dest`, which must be exactly one stride
    /// long and 4 KiB-aligned.
    ///
    /// A blocking positioned direct read. It does **not** verify the record's
    /// checksum — see [`Self::verify`] for the measurement behind that.
    pub(crate) fn read_into(&self, layer: usize, expert: usize, dest: &mut [u8]) -> Result<()> {
        if dest.len() != self.stride {
            candle::bail!(
                "expert pack read wants a {}-byte destination, got {}",
                self.stride,
                dest.len()
            );
        }
        self.reader
            .read_at(self.offset_of(layer, expert)?, dest)
            .map_err(|e| {
                candle::Error::Msg(format!(
                    "expert pack read L{layer}E{expert} from {}: {e}",
                    self.path.display()
                ))
            })
    }

    /// [`Self::read_into`] on file handle `handle` of the pool, so concurrent
    /// readers each keep their own kernel I/O queue — the stager's reader
    /// threads, one handle each.
    pub(crate) fn read_into_with_handle(
        &self,
        handle: usize,
        layer: usize,
        expert: usize,
        dest: &mut [u8],
    ) -> Result<()> {
        if dest.len() != self.stride {
            candle::bail!(
                "expert pack read wants a {}-byte destination, got {}",
                self.stride,
                dest.len()
            );
        }
        self.reader
            .read_at_with_handle(handle, self.offset_of(layer, expert)?, dest)
            .map_err(|e| {
                candle::Error::Msg(format!(
                    "expert pack read L{layer}E{expert} from {}: {e}",
                    self.path.display()
                ))
            })
    }

    /// Read many records at once, each into its own stride-long aligned buffer.
    ///
    /// The reads are spread across the file handles so the drive sees a full
    /// queue — this is the startup fill, where thousands of records move and
    /// per-read latency would otherwise dominate.
    pub(crate) fn read_many(&self, targets: Vec<PackRead<'_>>) -> Result<()> {
        for t in targets.iter() {
            if t.dest.len() != self.stride {
                candle::bail!(
                    "expert pack batch read wants {}-byte destinations, L{}E{} got {}",
                    self.stride,
                    t.layer,
                    t.expert,
                    t.dest.len()
                );
            }
        }
        // `(layer, expert)` per stripe, for the error a checksum failure prints.
        let mut ids: Vec<(usize, usize)> = Vec::with_capacity(targets.len());
        let mut stripes: Vec<StripeRead<'_>> = Vec::with_capacity(targets.len());
        for t in targets {
            let file_offset = self.offset_of(t.layer, t.expert)?;
            ids.push((t.layer, t.expert));
            stripes.push(StripeRead {
                file_offset,
                dest: t.dest,
            });
        }
        if stripes.is_empty() {
            return Ok(());
        }
        self.reader
            .read_stripes_concurrent(&mut stripes)
            .map_err(|e| {
                candle::Error::Msg(format!(
                    "expert pack batch read from {}: {e}",
                    self.path.display()
                ))
            })?;
        // Verified across the pool: this is the whole warm tier, ~14 GB, and a
        // checksum is memory-bound, so one thread would add seconds to startup
        // where the cores are otherwise idle waiting on the drive.
        ids.par_iter()
            .zip(stripes.par_iter())
            .try_for_each(|(&(layer, expert), stripe)| self.verify(layer, expert, stripe.dest))
    }

    /// Open the expert section at byte `base` of `path`.
    ///
    /// Refuses a section that is not this build's: wrong magic or version, a
    /// repack whose `fingerprint` no longer matches what is recorded, or a file
    /// too short for the records its header promises. Every refusal has the
    /// same remedy — rebuild the model pack — so each is an `Err` naming its
    /// reason.
    pub(crate) fn open_section(
        path: &Path,
        base: u64,
        fingerprint: impl Fn(GgmlDType, GgmlDType) -> u64,
    ) -> Result<Self> {
        if !base.is_multiple_of(DIRECT_IO_SECTOR as u64) {
            candle::bail!(
                "expert pack: the section at {base} in {} does not start on a sector",
                path.display()
            );
        }
        let at = |e: std::io::Error| {
            candle::Error::Msg(format!("expert pack read {}: {e}", path.display()))
        };
        let mut f = File::open(path).map_err(at)?;
        f.seek(SeekFrom::Start(base)).map_err(at)?;
        let mut fixed = vec![0u8; header::FIXED_BYTES];
        f.read_exact(&mut fixed).map_err(at)?;
        let len = header::encoded_len_from_fixed(&fixed)?;
        let mut head = fixed;
        head.resize(len, 0);
        f.read_exact(&mut head[header::FIXED_BYTES..]).map_err(at)?;
        let header = PackHeader::decode(&head)?;
        check_prints(&header.pairs, fingerprint)?;

        let records_at = base + records_offset(&header);
        let total = header.total_experts();
        let trailer_at = records_at + total as u64 * header.stride;
        let need = trailer_at + (total * 4) as u64;
        let have = f.metadata().map_err(at)?.len();
        if have < need {
            candle::bail!(
                "expert pack {} is {have} bytes, the section needs {need} — truncated",
                path.display()
            );
        }
        // The checksum trailer, read buffered: it is small and read once, so it
        // has none of the reasons the records bypass the page cache.
        let mut raw = vec![0u8; total * 4];
        f.seek(SeekFrom::Start(trailer_at))
            .and_then(|_| f.read_exact(&mut raw))
            .map_err(at)?;
        let sums: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        drop(f);
        let reader = DirectFile::open(path).map_err(at)?;
        Ok(Self {
            path: path.to_path_buf(),
            reader,
            records_at,
            stride: header.stride as usize,
            experts_per_layer: header.experts_per_layer as usize,
            layouts: header
                .layers
                .iter()
                .copied()
                .map(RecordLayout::from)
                .collect(),
            header,
            sums,
        })
    }
}

impl std::fmt::Debug for ExpertPack {
    /// Where the section is and how it is cut — the read handles have no
    /// printable form and the layout table is per-layer noise in a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertPack")
            .field("path", &self.path)
            .field("records_at", &self.records_at)
            .field("stride", &self.stride)
            .field("slot_bytes", &self.header.slot_bytes)
            .field("layers", &self.layouts.len())
            .field("experts_per_layer", &self.experts_per_layer)
            .finish()
    }
}

/// Writes an expert section record by record, in ascending expert order, into
/// whatever the model pack's build hands it.
///
/// Sequential by construction — the writer never seeks, so the section is one
/// streaming pass and the OS sees it as such. Publishing is the model pack's
/// job: this writes bytes and says how many.
pub(crate) struct PackWriter<'a, W: Write + ?Sized> {
    out: &'a mut W,
    header: PackHeader,
    /// One record, reused. Zeroed whenever the layer changes, because the gaps
    /// between projections are fixed within a layer and only move between them.
    record: Vec<u8>,
    current_layer: Option<usize>,
    written: usize,
    /// One checksum per record written, appended as the trailer at `finish`.
    sums: Vec<u32>,
}

impl<'a, W: Write + ?Sized> PackWriter<'a, W> {
    /// Write `header`, padded to a sector, and stand ready for the records.
    pub(crate) fn new(out: &'a mut W, header: PackHeader) -> Result<Self> {
        let mut head = header.encode();
        head.resize(round_up_sector(head.len()), 0);
        out.write_all(&head)
            .map_err(|e| candle::Error::Msg(format!("expert pack write header: {e}")))?;
        let stride = header.stride as usize;
        let total = header.total_experts();
        Ok(Self {
            out,
            header,
            record: vec![0u8; stride],
            current_layer: None,
            written: 0,
            sums: Vec::with_capacity(total),
        })
    }

    /// Append `(layer, expert)`'s record. Must be called in ascending index
    /// order, once per expert of every layer.
    pub(crate) fn write_expert(
        &mut self,
        layer: usize,
        expert: usize,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
    ) -> Result<()> {
        let expect = layer * self.header.experts_per_layer as usize + expert;
        if expect != self.written {
            candle::bail!(
                "expert pack writes must be sequential: L{layer}E{expert} is index {expect}, \
                 {} records are written",
                self.written
            );
        }
        if self.current_layer != Some(layer) {
            self.record.fill(0);
            self.current_layer = Some(layer);
        }
        let spans = self.header.layers[layer];
        for (span, src) in [(spans.gate, gate), (spans.up, up), (spans.down, down)] {
            let at = span.offset as usize;
            if src.len() != span.bytes as usize {
                candle::bail!(
                    "expert pack L{layer}E{expert}: projection is {} bytes, the layer's geometry \
                     says {}",
                    src.len(),
                    span.bytes
                );
            }
            self.record[at..at + src.len()].copy_from_slice(src);
        }
        self.out
            .write_all(&self.record)
            .map_err(|e| candle::Error::Msg(format!("expert pack write L{layer}E{expert}: {e}")))?;
        // Over the whole record including its zero padding, which is what the
        // reader has in hand and so what it can check without knowing the
        // layer's geometry.
        self.sums.push(fletcher32(&self.record));
        self.written += 1;
        Ok(())
    }

    /// Append the checksum trailer, refusing a section with records unwritten,
    /// and return the section's length.
    pub(crate) fn finish(self) -> Result<u64> {
        let expect = self.header.total_experts();
        if self.written != expect {
            candle::bail!(
                "expert pack is short: {} of {expect} records written",
                self.written
            );
        }
        let mut trailer = Vec::with_capacity(self.sums.len() * 4);
        for s in &self.sums {
            trailer.extend_from_slice(&s.to_le_bytes());
        }
        self.out
            .write_all(&trailer)
            .map_err(|e| candle::Error::Msg(format!("expert pack write trailer: {e}")))?;
        Ok(section_len(&self.header))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::repack_fingerprint::PairPrint;
    use candle::direct_io::AlignedScratch;
    use std::io::BufWriter;

    fn tmp_file(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("candle_expert_section_{tag}_{nanos}.bin"))
    }

    fn proj(offset: u32, bytes: u32, dtype: GgmlDType) -> ProjectionSpan {
        ProjectionSpan {
            offset,
            bytes,
            dtype,
            src_dtype: GgmlDType::Q4_K,
            rows: 8,
            cols: 256,
        }
    }

    /// Two layers, three projections each, sized so the record needs padding to
    /// reach a sector — the interesting case for both the writer and the reader.
    fn header() -> PackHeader {
        let layers = vec![
            LayerSpans {
                block: 0,
                gate: proj(0, 300, GgmlDType::Q4_KO),
                up: proj(512, 300, GgmlDType::Q4_KO),
                down: proj(1024, 200, GgmlDType::Q4_KO),
            },
            LayerSpans {
                block: 1,
                gate: proj(0, 256, GgmlDType::Q4_KO),
                up: proj(512, 256, GgmlDType::Q4_KO),
                down: proj(1024, 256, GgmlDType::Q4_KO),
            },
        ];
        let pairs = pairs_in(&layers)
            .into_iter()
            .map(|(src_dtype, dtype)| PairPrint {
                src_dtype,
                dtype,
                fp: 0x5A5A,
            })
            .collect();
        PackHeader {
            num_layers: 2,
            experts_per_layer: 2,
            slot_bytes: SLOT_BYTES as u32,
            stride: round_up_sector(SLOT_BYTES) as u64,
            int8_mode: 3,
            layers,
            pairs,
        }
    }

    const SLOT_BYTES: usize = 1024 + 256;

    /// The fingerprint every fixture pair was recorded with.
    fn same(_: GgmlDType, _: GgmlDType) -> u64 {
        0x5A5A
    }

    /// Write `prefix` bytes of something else, then the fixture section, and
    /// return the file and where the section starts — the shape of a model pack.
    fn build(tag: &str, prefix: usize) -> (PathBuf, u64) {
        let path = tmp_file(tag);
        let mut out = BufWriter::new(File::create(&path).unwrap());
        out.write_all(&vec![0xEE; prefix]).unwrap();
        let mut w = PackWriter::new(&mut out, header()).unwrap();
        for layer in 0..2 {
            let s = header().layers[layer];
            for expert in 0..2 {
                let tag = (layer * 2 + expert) as u8;
                w.write_expert(
                    layer,
                    expert,
                    &vec![tag; s.gate.bytes as usize],
                    &vec![tag.wrapping_add(0x40); s.up.bytes as usize],
                    &vec![tag.wrapping_add(0x80); s.down.bytes as usize],
                )
                .unwrap();
            }
        }
        let len = w.finish().unwrap();
        out.flush().unwrap();
        drop(out);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            prefix as u64 + len,
            "finish reports the section's true length"
        );
        (path, prefix as u64)
    }

    fn open(path: &Path, base: u64) -> ExpertPack {
        ExpertPack::open_section(path, base, same).unwrap()
    }

    /// The record stride is the slot image padded to a direct-I/O sector, and
    /// the first record starts after an equally padded header — both required
    /// for a positioned read to be legal at every record, wherever in the file
    /// the section sits.
    #[test]
    fn every_record_starts_on_a_sector() {
        let (path, base) = build("sectors", 8192);
        let pack = open(&path, base);
        assert_eq!(pack.stride(), round_up_sector(SLOT_BYTES));
        for layer in 0..2 {
            for expert in 0..2 {
                assert_eq!(
                    pack.offset_of(layer, expert).unwrap() % DIRECT_IO_SECTOR as u64,
                    0
                );
            }
        }
        drop(pack);
        std::fs::remove_file(&path).ok();
    }

    /// A section that does not start on a sector cannot be read directly.
    #[test]
    fn a_misaligned_section_is_refused() {
        let (path, _) = build("misaligned", 4096);
        let e = ExpertPack::open_section(&path, 100, same)
            .unwrap_err()
            .to_string();
        assert!(e.contains("sector"), "{e}");
        std::fs::remove_file(&path).ok();
    }

    /// A record read back carries exactly the bytes written, at exactly the
    /// offsets the layer's geometry names, with the gaps zeroed — the first
    /// layer included, since every layer has records.
    #[test]
    fn a_record_round_trips_byte_for_byte() {
        let (path, base) = build("roundtrip", 4096);
        let pack = open(&path, base);
        let mut scratch = AlignedScratch::new();
        scratch.ensure(pack.stride()).unwrap();
        for (layer, expert, tag) in [(0usize, 0usize, 0u8), (1, 1, 3)] {
            let dest = scratch.as_mut_slice(pack.stride());
            pack.read_into(layer, expert, dest).unwrap();
            let layout = pack.layout(layer);
            let n = layout.gate.bytes;
            assert_eq!(&dest[..n], vec![tag; n].as_slice());
            assert_eq!(
                &dest[layout.up.offset..layout.up.offset + layout.up.bytes],
                vec![tag + 0x40; layout.up.bytes].as_slice()
            );
            assert_eq!(
                &dest[layout.down.offset..layout.down.offset + layout.down.bytes],
                vec![tag + 0x80; layout.down.bytes].as_slice()
            );
            assert!(dest[n..512].iter().all(|&b| b == 0), "gap not zeroed");
        }
        drop(pack);
        std::fs::remove_file(&path).ok();
    }

    /// Two layers with different projection sizes must not bleed into each
    /// other through the reused record buffer.
    #[test]
    fn a_layer_change_clears_the_previous_layers_tail() {
        let (path, base) = build("layerchange", 0);
        let pack = open(&path, base);
        let mut scratch = AlignedScratch::new();
        scratch.ensure(pack.stride()).unwrap();
        let dest = scratch.as_mut_slice(pack.stride());
        // Layer 0's gate is 300 bytes, layer 1's is 256.
        pack.read_into(1, 0, dest).unwrap();
        assert!(
            dest[256..300].iter().all(|&b| b == 0),
            "layer 0's tail survived into layer 1: {:?}",
            &dest[256..300]
        );
        drop(pack);
        std::fs::remove_file(&path).ok();
    }

    /// The batch path reads the same bytes as the single-record path.
    #[test]
    fn a_batch_read_agrees_with_the_single_reads() {
        let (path, base) = build("batch", 4096);
        let pack = open(&path, base);
        let stride = pack.stride();
        let mut batch = AlignedScratch::new();
        batch.ensure(stride * 4).unwrap();
        let mut rest = batch.as_mut_slice(stride * 4);
        let mut chunks: Vec<&mut [u8]> = Vec::new();
        for _ in 0..4 {
            let (head, tail) = rest.split_at_mut(stride);
            chunks.push(head);
            rest = tail;
        }
        let targets: Vec<PackRead<'_>> = chunks
            .into_iter()
            .enumerate()
            .map(|(i, dest)| PackRead {
                layer: i / 2,
                expert: i % 2,
                dest,
            })
            .collect();
        pack.read_many(targets).unwrap();

        let mut one = AlignedScratch::new();
        one.ensure(stride).unwrap();
        let got = batch.as_slice(stride * 4);
        for i in 0..4 {
            let want = one.as_mut_slice(stride);
            pack.read_into(i / 2, i % 2, want).unwrap();
            assert_eq!(
                &got[i * stride..(i + 1) * stride],
                want,
                "record {i} differs"
            );
        }
        drop(pack);
        std::fs::remove_file(&path).ok();
    }

    /// **The runtime paths read the drive every time; the pack keeps no host
    /// copy of a record.** Host RAM for experts is the warm tier's alone, so a
    /// record read once and then clobbered on disk must come back as the
    /// clobbered bytes.
    #[test]
    fn a_runtime_reread_reads_the_drive() {
        let (path, base) = build("reread", 4096);
        let pack = open(&path, base);
        let stride = pack.stride();
        let mut scratch = AlignedScratch::new();
        scratch.ensure(stride).unwrap();
        {
            let dest = scratch.as_mut_slice(stride);
            pack.read_into(1, 1, dest).unwrap();
            assert!(dest.iter().any(|&b| b != 0), "fixture record is all zero");
        }
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(pack.path())
                .unwrap();
            f.seek(SeekFrom::Start(pack.offset_of(1, 1).unwrap()))
                .unwrap();
            f.write_all(&vec![0u8; stride]).unwrap();
            f.sync_all().unwrap();
        }
        let zeroed = vec![0u8; stride];
        let dest = scratch.as_mut_slice(stride);
        dest.fill(0xEE);
        pack.read_into(1, 1, dest).unwrap();
        assert_eq!(
            dest,
            zeroed.as_slice(),
            "read_into served a remembered copy"
        );
        let dest = scratch.as_mut_slice(stride);
        dest.fill(0xEE);
        pack.read_into_with_handle(3, 1, 1, dest).unwrap();
        assert_eq!(
            scratch.as_slice(stride),
            zeroed.as_slice(),
            "read_into_with_handle served a remembered copy"
        );
        drop(pack);
        std::fs::remove_file(&path).ok();
    }

    /// **Bit rot in a record is caught, not served**, on the bulk path.
    #[test]
    fn a_corrupted_record_is_refused() {
        let (path, base) = build("corrupt", 4096);
        let at = open(&path, base).offset_of(1, 1).unwrap() + 16;
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let mut b = [0u8; 1];
            f.seek(SeekFrom::Start(at)).unwrap();
            f.read_exact(&mut b).unwrap();
            f.seek(SeekFrom::Start(at)).unwrap();
            f.write_all(&[b[0] ^ 0x01]).unwrap();
            f.sync_all().unwrap();
        }
        let pack = open(&path, base);
        let mut scratch = AlignedScratch::new();
        scratch.ensure(pack.stride()).unwrap();
        let dest = scratch.as_mut_slice(pack.stride());
        let e = pack
            .read_many(vec![PackRead {
                layer: 1,
                expert: 1,
                dest,
            }])
            .unwrap_err()
            .to_string();
        assert!(e.contains("corrupt") && e.contains("L1E1"), "{e}");
        let dest = scratch.as_mut_slice(pack.stride());
        pack.read_many(vec![PackRead {
            layer: 1,
            expert: 0,
            dest,
        }])
        .unwrap();
        drop(pack);
        std::fs::remove_file(&path).ok();
    }

    /// **The case geometry cannot catch.** A repack that emits different bytes
    /// for a format the section holds refuses the open, naming the format.
    #[test]
    fn a_changed_repack_for_a_held_format_refuses_the_open() {
        let (path, base) = build("fingerprint", 4096);
        let e = ExpertPack::open_section(&path, base, |_, _| 0x5A5B)
            .unwrap_err()
            .to_string();
        assert!(e.contains("Q4_K"), "{e}");
        std::fs::remove_file(&path).ok();
    }

    /// The header read back is the one written, so the cache sizes itself from
    /// exactly the geometry the build used.
    #[test]
    fn the_open_header_is_the_written_one() {
        let (path, base) = build("header", 4096);
        assert_eq!(open(&path, base).header(), &header());
        std::fs::remove_file(&path).ok();
    }

    /// A file cut short of the trailer is refused at open rather than when a
    /// read runs off its end.
    #[test]
    fn a_truncated_section_is_refused() {
        let (path, base) = build("truncated", 4096);
        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 1).unwrap();
        drop(f);
        let e = ExpertPack::open_section(&path, base, same)
            .unwrap_err()
            .to_string();
        assert!(e.contains("truncated"), "{e}");
        std::fs::remove_file(&path).ok();
    }

    /// Records must be written in index order, once each, starting at layer 0:
    /// a gap would leave a record of zeroes that reads back as a valid-looking
    /// expert.
    #[test]
    fn out_of_order_writes_are_refused() {
        let mut out = Vec::new();
        let mut w = PackWriter::new(&mut out, header()).unwrap();
        let s = header().layers[0];
        let e = w
            .write_expert(
                0,
                1,
                &vec![1; s.gate.bytes as usize],
                &vec![2; s.up.bytes as usize],
                &vec![3; s.down.bytes as usize],
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("sequential"), "{e}");
    }

    /// A projection whose length disagrees with the layer's geometry must not
    /// be padded or truncated into place.
    #[test]
    fn a_wrong_sized_projection_is_refused() {
        let mut out = Vec::new();
        let mut w = PackWriter::new(&mut out, header()).unwrap();
        let s = header().layers[0];
        let e = w
            .write_expert(
                0,
                0,
                &vec![1; s.gate.bytes as usize - 1],
                &vec![2; s.up.bytes as usize],
                &vec![3; s.down.bytes as usize],
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("geometry"), "{e}");
    }

    /// `finish` on a short build fails rather than writing a trailer for
    /// records that were never written.
    #[test]
    fn finishing_early_is_refused() {
        let mut out = Vec::new();
        let mut w = PackWriter::new(&mut out, header()).unwrap();
        let s = header().layers[0];
        w.write_expert(
            0,
            0,
            &vec![1; s.gate.bytes as usize],
            &vec![2; s.up.bytes as usize],
            &vec![3; s.down.bytes as usize],
        )
        .unwrap();
        let e = w.finish().unwrap_err().to_string();
        assert!(e.contains("short"), "{e}");
    }

    /// The section's length is its padded header, every record and the trailer.
    #[test]
    fn the_section_length_is_header_records_and_trailer() {
        let h = header();
        let stride = round_up_sector(SLOT_BYTES) as u64;
        assert_eq!(
            section_len(&h),
            round_up_sector(h.encoded_len()) as u64 + 4 * stride + 4 * 4
        );
    }
}
