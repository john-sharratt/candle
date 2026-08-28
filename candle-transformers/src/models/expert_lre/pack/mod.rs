//! The cold tier: a repacked expert pack file that holds every expert, always.
//!
//! The pack stores experts in the **repacked layout the kernels consume** — one
//! contiguous record per expert, gate/up/down at the offsets a VRAM slot uses —
//! rather than the original GGUF tensors. Three reasons, in order of weight:
//!
//! 1. **The repack is hot-path poison.** Startup repacks 6,144 experts in ~42 s,
//!    about 7 ms each, and a forward issues on the order of 1,150 expert loads.
//!    Repacking on load would cost seconds per forward.
//! 2. **A repacked expert is already one blob**, so a load is an offset and a
//!    copy. In the GGUF the same expert is a strided slice of three stacked
//!    per-tensor arrays, so loading one is a gather plus a dequantise.
//! 3. **It decouples the hot path from the checkpoint format**, so GGUF packing
//!    decisions cannot become cache performance regressions.
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
//! length and the destination pointer to be 4 KiB-aligned. So a record's stride
//! is the slot image padded up to a sector, and every record therefore starts on
//! one. The warm pool's slots are cut to the same stride, which is what lets a
//! cold read land *directly* in a pinned slot with no bounce buffer.

mod fingerprint;
mod header;

use ahash::{HashMap, HashMapExt};
use candle::direct_io::{round_up_sector, DirectFile, StripeRead};
use candle::fletcher::fletcher32;
use candle::quantized::{GgmlDType, Int8Mode};
use candle::Result;
use header::{LayerSpans, PackHeader, ProjectionSpan};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;

/// Bytes of the GGUF that go into the identity checksum.
///
/// The header and tensor table live at the front of the file and carry every
/// tensor's name, dtype, shape and offset — so a checkpoint that differs in any
/// way that matters to us differs inside this window. Hashing the whole 16.6 GiB
/// would cost a full sequential read at every startup to re-derive a fact the
/// length and this window already settle.
const IDENTITY_SAMPLE: usize = 4 * 1024 * 1024;

/// What a pack claims to have been built from. A pack whose identity does not
/// match the GGUF in front of us is rewritten, not adapted to.
///
/// Two halves, answering two different questions: the first three fields are
/// *which checkpoint*, and `repack_fp` is *which repack of it* — see
/// [`fingerprint`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackIdentity {
    pub source_len: u64,
    pub source_sum: u32,
    pub int8_mode: u32,
    /// Fingerprint of the repack formula, produced by running it.
    pub repack_fp: u64,
}

impl PackIdentity {
    /// The identity of the GGUF mapped at `gguf`, repacked for `int8mode`, by a
    /// build whose repack fingerprints as `repack_fp`.
    ///
    /// The caller supplies the fingerprint rather than this computing it,
    /// because the sweep needs a CUDA device — which this module deliberately
    /// knows nothing about.
    pub(crate) fn of(gguf: &[u8], int8mode: Int8Mode, repack_fp: u64) -> Self {
        let sample = &gguf[..gguf.len().min(IDENTITY_SAMPLE)];
        Self {
            source_len: gguf.len() as u64,
            source_sum: fletcher32(sample),
            // The repack targets a different dtype under int8 (the KO twin), so
            // two packs from one checkpoint are different files.
            int8_mode: int8mode as u32,
            repack_fp,
        }
    }
}

/// Where one of an expert's three projections sits inside a record.
///
/// The caller reads a record into a buffer and then issues three H2D copies
/// against these spans, so this is the only thing it needs to know about the
/// file's interior. No dtype: the header stores one per projection and checks it
/// against the checkpoint when the pack is opened, and after that the layer
/// geometry is the single source the kernels read.
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

/// Byte ceiling for [`ColdRecordCache`]: a QUARTER of physical RAM. The
/// pinned warm tier already claims up to half (`warm_slots_for`), so a
/// quarter for the pageable cold cache leaves at least a quarter for the
/// process, the KV warm arenas, and the OS. On the 284B dev box that is
/// ~47 GiB against a ~51 GiB cold universe (the ~3.9k experts in neither
/// the warm tier nor VRAM, at 13.2 MiB per record) — a 16 GiB flat cap was
/// measured to plateau at 1213 records and an 11% hit rate, leaving the
/// trailing prefill re-reading ~36 GB of NVMe per pass. Insertion simply
/// stops at the cap — no eviction — because the miss set is stable across
/// waves, so whichever records filled first keep paying off every wave.
fn cold_cache_cap() -> usize {
    match candle::vram::total_physical_ram() {
        Some(total) => (total / 4) as usize,
        // No probe on this platform: fall back to a fixed ceiling rather
        // than either unbounded growth or no cache at all.
        None => 16 << 30,
    }
}

/// See [`ExpertPack::cold_cache`]. Keyed by flat record index.
struct ColdRecordCache {
    records: RwLock<HashMap<usize, Box<[u8]>>>,
    bytes: AtomicUsize,
    cap: usize,
}

impl ColdRecordCache {
    fn new() -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
            bytes: AtomicUsize::new(0),
            cap: cold_cache_cap(),
        }
    }

    /// Whether record `idx` is cached. Records are never removed, so a `true`
    /// stays true — callers may partition on this and copy later.
    fn contains(&self, idx: usize) -> bool {
        self.records
            .read()
            .expect("cold cache lock poisoned")
            .contains_key(&idx)
    }

    /// Copy record `idx` into `dest` if cached. Returns whether it was.
    fn fill(&self, idx: usize, dest: &mut [u8]) -> bool {
        let map = self.records.read().expect("cold cache lock poisoned");
        match map.get(&idx) {
            Some(rec) if rec.len() == dest.len() => {
                dest.copy_from_slice(rec);
                true
            }
            _ => false,
        }
    }

    /// Remember `record` for `idx` (no-op once the cap is reached — the miss
    /// set is stable, so uncached records just keep reading from the pack).
    /// The 13 MB copy happens BEFORE the write lock, so concurrent stores
    /// serialize only on the map insert.
    fn store(&self, idx: usize, record: &[u8]) {
        if self.bytes.load(Ordering::Relaxed) + record.len() > self.cap {
            return;
        }
        if self.contains(idx) {
            return;
        }
        let copy = record.to_vec().into_boxed_slice();
        let mut map = self.records.write().expect("cold cache lock poisoned");
        if map.contains_key(&idx) {
            return;
        }
        self.bytes.fetch_add(copy.len(), Ordering::Relaxed);
        map.insert(idx, copy);
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

/// An open pack file: every expert, in kernel-ready form, readable at any time.
pub(crate) struct ExpertPack {
    path: PathBuf,
    reader: DirectFile,
    /// In-process cache of every record the RUNTIME miss path has read —
    /// plain pageable memory, filled lazily, never evicted (capacity-bounded
    /// by [`cold_cache_cap`]). The cold-eligible universe is bounded (the
    /// experts in neither the pinned warm tier nor VRAM — ~3.9k records
    /// ≈ 51 GiB on the 284B target) and re-reads on EVERY prefill wave for
    /// the process lifetime, so after first touch a cold miss is a memcpy
    /// instead of a physical NVMe round-trip.
    ///
    /// Why not the OS page cache: it was measured NOT to retain this set on
    /// the dev box — with the warm tier pinned the machine runs at free=0,
    /// and the continuous allocation churn repurposes the pack's standby
    /// pages between waves (a warm-standby rerun still read at physical
    /// speed, ~1.9s per 670-token prefill = the post-merge single-session
    /// prefill regression; the pre-pack GGUF mmap kept its pages because
    /// mapped views live in the process working set — which is exactly what
    /// this cache restores, deliberately). The startup fill (`read_many`,
    /// verified) bypasses the cache: it reads every record exactly once and
    /// most of it lands pinned in the warm tier.
    cold_cache: ColdRecordCache,
    /// Where the first record starts — the header padded to a sector.
    records_at: u64,
    stride: usize,
    slot_bytes: usize,
    experts_per_layer: usize,
    /// Leading MoE layers with no record in the file — permanently VRAM-resident
    /// and never reloaded. Record `0` is layer `pinned_layers`, expert `0`.
    pinned_layers: usize,
    layouts: Vec<RecordLayout>,
    /// One checksum per record, in index order, read from the trailer at open.
    ///
    /// **A trailer rather than a table between the header and the records**,
    /// because the writer never seeks: it accumulates these as it streams the
    /// 16.6 GiB out and appends them at the end. Putting them in front would
    /// mean either seeking back over the whole file or buffering it.
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

    /// Where the three projections sit inside `layer`'s records.
    pub(crate) fn layout(&self, layer: usize) -> RecordLayout {
        self.layouts[layer]
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Leading MoE layers this pack holds no records for — see
    /// [`Self::record_index`].
    pub(crate) fn pinned_layers(&self) -> usize {
        self.pinned_layers
    }

    /// Flat record index of `(layer, expert)`, or an error for a layer the file
    /// deliberately does not store.
    ///
    /// The pinned prefix is permanently VRAM-resident and never evicted, so a
    /// read against it is not a cache miss — it is a bug in whatever decided the
    /// expert was evictable, and it must say so here rather than silently
    /// serving the record `pinned_layers` further along.
    fn record_index(&self, layer: usize, expert: usize) -> Result<usize> {
        let Some(rel) = layer.checked_sub(self.pinned_layers) else {
            candle::bail!(
                "expert pack: layer {layer} is one of the {} permanently resident \
                 layers and has no record — it is never evicted, so nothing should \
                 be reloading it",
                self.pinned_layers,
            );
        };
        Ok(rel * self.experts_per_layer + expert)
    }

    /// Byte offset of `(layer, expert)`'s record.
    fn offset_of(&self, layer: usize, expert: usize) -> Result<u64> {
        Ok(self.records_at + (self.record_index(layer, expert)? * self.stride) as u64)
    }

    /// Check a record against the checksum written with it.
    ///
    /// **What this catches is the storage, not the writer.** A half-built pack
    /// cannot be published (the writer streams to a private temp file and only
    /// renames a complete one) and two writers cannot collide (the temp name
    /// carries the pid). What is left is the medium: bit rot, a bad sector, a
    /// truncating filesystem — on a file that is 16.6 GiB, lives beside the
    /// checkpoint for as long as the checkpoint does, and whose contents become
    /// weights with no further validation.
    ///
    /// # Why only the bulk path calls this
    ///
    /// It is checked on [`Self::read_many`] — the startup fill, where thousands
    /// of records move at once, the cores are idle waiting on the drive, and the
    /// work parallelises. It is **not** checked on [`Self::read_into`], the
    /// per-miss path, and that is a measured decision rather than an oversight:
    /// a `fletcher32` over 2.9 MB costs about as much as the read it follows, on
    /// the pipeline thread, in front of a forward that is waiting for it. With
    /// it there the gate lost **more than half its throughput** — 723 → 299 t/s
    /// on the narrowest config — for ~850 records per config.
    ///
    /// The residual exposure is bounded and small: an unverified cold read is
    /// one expert of 6,144 whose contribution is wrong for as long as it stays
    /// resident, in a mixture-of-experts layer that sums eight of them. The
    /// alternative was making every cold miss twice as slow to insure against a
    /// medium failure the checkpoint it derives from is not itself insured
    /// against — the GGUF has no per-tensor checksum either.
    fn verify(&self, layer: usize, expert: usize, record: &[u8]) -> Result<()> {
        let idx = self.record_index(layer, expert)?;
        let Some(&want) = self.sums.get(idx) else {
            return Ok(());
        };
        let got = fletcher32(record);
        if got != want {
            candle::bail!(
                "expert pack L{layer}E{expert} is corrupt in {}: checksum {got:#010x}, \
                 expected {want:#010x}. Delete the pack to rebuild it from the checkpoint.",
                self.path.display()
            );
        }
        Ok(())
    }

    /// Read one expert's record into `dest`, which must be exactly one stride
    /// long and 4 KiB-aligned.
    ///
    /// This is the miss path when an expert is in neither VRAM nor RAM. The
    /// first touch is a blocking positioned direct read; every later touch is
    /// a memcpy from [`Self::cold_cache`] (misses recur every wave). It does
    /// **not** verify the record's checksum — see [`Self::verify`] for the
    /// measurement behind that.
    pub(crate) fn read_into(&self, layer: usize, expert: usize, dest: &mut [u8]) -> Result<()> {
        if dest.len() != self.stride {
            candle::bail!(
                "expert pack read wants a {}-byte destination, got {}",
                self.stride,
                dest.len()
            );
        }
        let idx = self.record_index(layer, expert)?;
        if self.cold_cache.fill(idx, dest) {
            return Ok(());
        }
        self.reader
            .read_at(self.offset_of(layer, expert)?, dest)
            .map_err(|e| {
                candle::Error::Msg(format!(
                    "expert pack read L{layer}E{expert} from {}: {e}",
                    self.path.display()
                ))
            })?;
        self.cold_cache.store(idx, dest);
        Ok(())
    }

    /// Read many records at once, each into its own stride-long aligned buffer.
    ///
    /// The reads are spread across the file handles so the drive sees a full
    /// queue — this is the startup fill, where thousands of records move and
    /// per-read latency would otherwise dominate.
    pub(crate) fn read_many(&self, targets: Vec<PackRead<'_>>) -> Result<()> {
        self.read_many_impl(targets, true)
    }

    /// [`Self::read_many`] without the checksum pass — the RUNTIME miss path,
    /// which shares [`Self::read_into`]'s contract: hot-loop reads skip
    /// verification (see [`Self::verify`] for the measurement behind that)
    /// while the startup fill, which reads every record exactly once with idle
    /// cores, keeps it. Routing the hot loop through the verifying form put a
    /// full-record checksum on every cold miss and multiplied its latency.
    pub(crate) fn read_many_unverified(&self, targets: Vec<PackRead<'_>>) -> Result<()> {
        self.read_many_impl(targets, false)
    }

    fn read_many_impl(&self, targets: Vec<PackRead<'_>>, verify: bool) -> Result<()> {
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
        // Runtime (unverified) batches consult the cold cache first: the miss
        // set recurs every wave, so after first touch most of the batch is
        // memcpys and only the residue reads the drive. The cache memcpys run
        // on the rayon pool CONCURRENTLY with the residue's striped direct
        // reads — a serial 13 MB copy per record was barely faster than the
        // QD16 NVMe read it replaced. The verified startup fill skips the
        // cache both ways — it reads every record exactly once and most of
        // what it reads lands pinned in the warm tier.
        let mut hits: Vec<(usize, &mut [u8])> = Vec::new();
        // `(record index, (layer, expert))` — the index for the cold cache, the
        // pair for the error message a checksum failure prints.
        let mut ids: Vec<(usize, (usize, usize))> = Vec::with_capacity(targets.len());
        let mut stripes: Vec<StripeRead<'_>> = Vec::with_capacity(targets.len());
        for t in targets {
            let idx = self.record_index(t.layer, t.expert)?;
            if !verify && self.cold_cache.contains(idx) {
                hits.push((idx, t.dest));
                continue;
            }
            ids.push((idx, (t.layer, t.expert)));
            stripes.push(StripeRead {
                file_offset: self.records_at + (idx * self.stride) as u64,
                dest: t.dest,
            });
        }
        let (_, read) = rayon::join(
            || {
                hits.into_par_iter().for_each(|(idx, dest)| {
                    // Records are never removed, so the `contains` above
                    // guarantees this fills.
                    self.cold_cache.fill(idx, dest);
                });
            },
            || -> Result<()> {
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
                    })
            },
        );
        read?;
        if !verify {
            // Parallel stores: the per-record 13 MB copy happens outside the
            // map lock, so the copies spread across the pool.
            ids.par_iter()
                .zip(stripes.par_iter())
                .for_each(|(&(idx, _), stripe)| {
                    self.cold_cache.store(idx, stripe.dest);
                });
            return Ok(());
        }
        // Verified across the pool: this is the whole warm tier, ~14 GB, and a
        // checksum is memory-bound, so one thread would add seconds to startup
        // where the cores are otherwise idle waiting on the drive.
        ids.par_iter()
            .zip(stripes.par_iter())
            .try_for_each(|(&(_, (layer, expert)), stripe)| self.verify(layer, expert, stripe.dest))
    }

    /// Open `path` if it is a pack this build can use for this checkpoint.
    ///
    /// Every failure — missing, unreadable, wrong magic, wrong version, wrong
    /// checkpoint, geometry that no longer matches — has the same remedy, so
    /// they are all reported as `Err` and the caller rewrites.
    fn open(path: &Path, want: &PackHeader) -> Result<Self> {
        let mut f = File::open(path)
            .map_err(|e| candle::Error::Msg(format!("expert pack open {}: {e}", path.display())))?;
        let mut head = vec![0u8; want.encoded_len()];
        f.read_exact(&mut head).map_err(|e| {
            candle::Error::Msg(format!("expert pack read header {}: {e}", path.display()))
        })?;
        let got = PackHeader::decode(&head)?;
        if got != *want {
            candle::bail!(
                "expert pack {} was built for a different checkpoint or repack layout",
                path.display()
            );
        }
        let records_at = round_up_sector(got.encoded_len()) as u64;
        let total = got.total_experts();
        let trailer_at = records_at + total as u64 * got.stride;
        let need = trailer_at + (total * 4) as u64;
        let have = f
            .metadata()
            .map_err(|e| candle::Error::Msg(format!("expert pack stat: {e}")))?
            .len();
        if have < need {
            candle::bail!(
                "expert pack {} is {have} bytes, needs {need} — truncated",
                path.display()
            );
        }
        // The checksum trailer, read buffered: it is 24 KB and read once, so it
        // has none of the reasons the records bypass the page cache.
        let mut raw = vec![0u8; total * 4];
        f.seek(SeekFrom::Start(trailer_at))
            .and_then(|_| f.read_exact(&mut raw))
            .map_err(|e| {
                candle::Error::Msg(format!(
                    "expert pack read checksum trailer {}: {e}",
                    path.display()
                ))
            })?;
        let sums: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        drop(f);
        let reader = DirectFile::open(path).map_err(|e| {
            candle::Error::Msg(format!("expert pack direct open {}: {e}", path.display()))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            reader,
            cold_cache: ColdRecordCache::new(),
            records_at,
            stride: got.stride as usize,
            slot_bytes: got.slot_bytes as usize,
            experts_per_layer: got.experts_per_layer as usize,
            pinned_layers: got.pinned_layers as usize,
            layouts: got.layers.iter().copied().map(RecordLayout::from).collect(),
            sums,
        })
    }

    /// Where the pack is, how it is cut, and nothing about its contents — the
    /// read handles have no printable form and the layout table is per-layer
    /// noise in a log line.
    fn describe(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertPack")
            .field("path", &self.path)
            .field("stride", &self.stride)
            .field("slot_bytes", &self.slot_bytes)
            .field("layers", &self.layouts.len())
            .field("experts_per_layer", &self.experts_per_layer)
            .field("pinned_layers", &self.pinned_layers)
            .finish()
    }

    /// Unlink the file while keeping it readable through the open handles.
    ///
    /// This is how a pack with no persistent home cleans up after itself: the
    /// directory entry goes now, the bytes stay until the last handle closes,
    /// and the OS reclaims them at exit however the process ends — including a
    /// kill, which no drop handler survives. Unix has always allowed this;
    /// Windows does too, because `std::fs` opens with `FILE_SHARE_DELETE`.
    fn unlink_but_keep_open(&self) -> Result<()> {
        std::fs::remove_file(&self.path).map_err(|e| {
            candle::Error::Msg(format!("expert pack unlink {}: {e}", self.path.display()))
        })
    }
}

impl std::fmt::Debug for ExpertPack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.describe(f)
    }
}

/// Builds a pack file record by record, in ascending expert order.
///
/// Sequential by construction — the writer never seeks, so the 16.6 GiB write is
/// one streaming pass and the OS sees it as such.
pub(crate) struct PackWriter {
    /// `None` once [`PackWriter::finish`] has taken the handle to sync and
    /// publish it. An `Option` rather than a plain field because the writer has
    /// a `Drop` that removes the temp file, and a type with `Drop` cannot have
    /// its fields moved out.
    out: Option<BufWriter<File>>,
    tmp: PathBuf,
    final_path: PathBuf,
    header: PackHeader,
    /// One record, reused. Zeroed whenever the layer changes, because the gaps
    /// between projections are fixed within a layer and only move between them.
    record: Vec<u8>,
    current_layer: Option<usize>,
    written: usize,
    ephemeral: bool,
    /// One checksum per record written, appended as the trailer at `finish`.
    sums: Vec<u32>,
}

impl PackWriter {
    /// Create the pack at its final path via a sibling temp file.
    ///
    /// Writing through a temp and renaming at the end is what keeps a killed
    /// process from leaving a half-written pack that looks complete: the header
    /// is only reachable under the final name once every record is behind it.
    fn create(final_path: &Path, header: PackHeader, ephemeral: bool) -> Result<Self> {
        // **The temp name must be unique per writer, not per pack.** §5.2 puts
        // the pack beside the checkpoint precisely so several workspaces share
        // one, and the pack's name is a pure function of that checkpoint — so a
        // fixed `.partial` sibling is the same path for every process that
        // decides to build it. Two daemons starting together, or one restarting
        // over another, would interleave their writes into one file and then
        // both rename it into place. The header is written first and is
        // identical, so the result validates and serves garbage weights.
        //
        // With the pid and a nanosecond stamp in the name they build separately
        // and the rename picks a winner. Both files are complete; the loser is
        // simply replaced.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp =
            final_path.with_extension(format!("pack.{}.{stamp:x}.partial", std::process::id()));
        if let Some(dir) = final_path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| {
                candle::Error::Msg(format!("expert pack mkdir {}: {e}", dir.display()))
            })?;
        }
        let file = File::create(&tmp).map_err(|e| {
            candle::Error::Msg(format!("expert pack create {}: {e}", tmp.display()))
        })?;
        let mut out = BufWriter::with_capacity(8 << 20, file);
        let mut head = header.encode();
        head.resize(round_up_sector(head.len()), 0);
        out.write_all(&head)
            .map_err(|e| candle::Error::Msg(format!("expert pack write header: {e}")))?;
        let stride = header.stride as usize;
        let total = header.total_experts();
        Ok(Self {
            out: Some(out),
            tmp,
            final_path: final_path.to_path_buf(),
            header,
            record: vec![0u8; stride],
            current_layer: None,
            written: 0,
            ephemeral,
            sums: Vec::with_capacity(total),
        })
    }

    /// Append `(layer, expert)`'s record. Must be called in ascending index
    /// order, once per expert, for every expert **outside the pinned prefix**.
    ///
    /// Pinned-layer experts are permanently VRAM-resident and never reloaded, so
    /// the caller must not offer them: doing so is caught here rather than
    /// silently writing a record the reader's index arithmetic does not expect.
    pub(crate) fn write_expert(
        &mut self,
        layer: usize,
        expert: usize,
        gate: &[u8],
        up: &[u8],
        down: &[u8],
    ) -> Result<()> {
        let pinned = self.header.pinned_layers as usize;
        let Some(rel) = layer.checked_sub(pinned) else {
            candle::bail!(
                "expert pack writer: L{layer}E{expert} is inside the {pinned} permanently \
                 resident layers, which this pack deliberately holds no records for"
            );
        };
        let expect = rel * self.header.experts_per_layer as usize + expert;
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
        let Some(out) = self.out.as_mut() else {
            candle::bail!("expert pack writer L{layer}E{expert}: already published");
        };
        out.write_all(&self.record)
            .map_err(|e| candle::Error::Msg(format!("expert pack write L{layer}E{expert}: {e}")))?;
        // Over the whole record including its zero padding, which is what the
        // reader has in hand and so what it can check without knowing the
        // layer's geometry.
        self.sums.push(fletcher32(&self.record));
        self.written += 1;
        Ok(())
    }

    /// Flush, publish under the final name, and reopen for reading.
    pub(crate) fn finish(mut self) -> Result<ExpertPack> {
        let expect = self.header.total_experts();
        if self.written != expect {
            candle::bail!(
                "expert pack is short: {} of {expect} records written",
                self.written
            );
        }
        let Some(mut out) = self.out.take() else {
            candle::bail!("expert pack writer: finished twice");
        };
        // The checksum trailer, appended rather than placed in front: the
        // writer streams and never seeks, so the only place a per-record value
        // known *after* writing that record can go is the end.
        let mut trailer = Vec::with_capacity(self.sums.len() * 4);
        for s in &self.sums {
            trailer.extend_from_slice(&s.to_le_bytes());
        }
        out.write_all(&trailer)
            .map_err(|e| candle::Error::Msg(format!("expert pack write trailer: {e}")))?;
        out.flush()
            .map_err(|e| candle::Error::Msg(format!("expert pack flush: {e}")))?;
        let file = out
            .into_inner()
            .map_err(|e| candle::Error::Msg(format!("expert pack unwrap: {e}")))?;
        file.sync_all()
            .map_err(|e| candle::Error::Msg(format!("expert pack sync: {e}")))?;
        drop(file);
        // An existing pack under the final name is stale by construction — we
        // only got here because opening it failed — and Windows will not rename
        // over it.
        let _ = std::fs::remove_file(&self.final_path);
        std::fs::rename(&self.tmp, &self.final_path).map_err(|e| {
            candle::Error::Msg(format!(
                "expert pack publish {} → {}: {e}",
                self.tmp.display(),
                self.final_path.display()
            ))
        })?;
        let pack = ExpertPack::open(&self.final_path, &self.header)?;
        if self.ephemeral {
            pack.unlink_but_keep_open()?;
        }
        Ok(pack)
    }
}

impl Drop for PackWriter {
    fn drop(&mut self) {
        // A writer dropped without `finish` failed part way. The temp file is
        // the only trace and nothing will ever read it.
        let _ = std::fs::remove_file(&self.tmp);
    }
}

/// The pack for this checkpoint: opened if a valid one is already there,
/// otherwise a writer to build one.
pub(crate) enum PackSource {
    /// A pack that already exists and matches — the repack is skipped entirely.
    Ready(ExpertPack),
    /// No usable pack; the caller repacks from the GGUF and feeds this writer.
    Build(PackWriter),
}

/// Decide the pack's path and open or create it.
///
/// `dir` is where a persistent pack lives — the GGUF's own directory, so one
/// pack is shared by every workspace using that checkpoint and survives a
/// substrate wipe. `None` puts it in the system temp directory and deletes it
/// when the process is done, which is the default: an embedder, an example or a
/// test must never have a 16.6 GiB file appear beside its model unasked.
pub(crate) use fingerprint::repack_fingerprint;

pub(crate) fn open_or_create(spec: PackSpec<'_>) -> Result<PackSource> {
    let PackSpec {
        dir,
        gguf_path,
        identity,
        num_layers,
        experts_per_layer,
        pinned_layers,
        slot_bytes,
        layers,
    } = spec;
    let header = PackHeader {
        num_layers: num_layers as u32,
        experts_per_layer: experts_per_layer as u32,
        slot_bytes: slot_bytes as u32,
        stride: round_up_sector(slot_bytes) as u64,
        source_len: identity.source_len,
        source_sum: identity.source_sum,
        int8_mode: identity.int8_mode,
        repack_fp: identity.repack_fp,
        pinned_layers: pinned_layers as u32,
        layers: layers.into_iter().map(LayerSpans::from).collect(),
    };
    let stem = gguf_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());
    // The fingerprint deliberately does **not** appear in the name. It is a
    // validity check, not an identity: a build whose repack changed should
    // *replace* the pack for this checkpoint, not accumulate a second 16.6 GiB
    // file beside it under a different name.
    let name = format!(
        "{stem}.{:016x}.experts.pack",
        identity.source_len
            ^ ((identity.source_sum as u64) << 32)
            ^ ((identity.int8_mode as u64) << 16)
    );
    let (path, ephemeral) = match dir {
        Some(d) => (d.join(&name), false),
        None => (std::env::temp_dir().join(&name), true),
    };

    // An ephemeral pack is never reused: it is unlinked the moment its writer
    // publishes it, so the only way to find one under that name is a leftover
    // from a process that died before it could unlink. Rebuilding is both the
    // safe answer and the one that clears the leftover.
    match ExpertPack::open(&path, &header).and_then(|p| {
        if ephemeral {
            candle::bail!("an ephemeral pack is rebuilt every boot")
        } else {
            Ok(p)
        }
    }) {
        Ok(pack) => {
            tracing::info!(
                target: "candle_transformers::expert_lre",
                path = %path.display(),
                "expert pack: reusing the existing pack — no repack this boot"
            );
            Ok(PackSource::Ready(pack))
        }
        Err(why) => {
            tracing::info!(
                target: "candle_transformers::expert_lre",
                path = %path.display(),
                %why,
                "expert pack: building"
            );
            Ok(PackSource::Build(PackWriter::create(
                &path, header, ephemeral,
            )?))
        }
    }
}

/// Everything [`open_or_create`] needs to name the pack and check the one on
/// disk against the checkpoint in front of it.
pub(crate) struct PackSpec<'a> {
    /// Where a persistent pack lives, or `None` for a temp file.
    pub dir: Option<&'a Path>,
    /// The checkpoint the pack is derived from — its stem names the file.
    pub gguf_path: &'a Path,
    pub identity: PackIdentity,
    pub num_layers: usize,
    pub experts_per_layer: usize,
    /// Leading MoE layers to hold **no records** for, because the cache pins
    /// them in VRAM permanently and never reloads them.
    pub pinned_layers: usize,
    /// The slot image: three projections at their aligned offsets.
    pub slot_bytes: usize,
    pub layers: Vec<LayerSpansInput>,
}

/// One layer's projection geometry, as the caller knows it before a pack exists.
///
/// The same three triples the header stores, named from the caller's side so the
/// header's byte layout stays private to [`header`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerSpansInput {
    pub gate: (usize, usize, GgmlDType),
    pub up: (usize, usize, GgmlDType),
    pub down: (usize, usize, GgmlDType),
}

impl From<LayerSpansInput> for RecordLayout {
    fn from(i: LayerSpansInput) -> Self {
        let span = |(offset, bytes, _): (usize, usize, GgmlDType)| RecordSpan { offset, bytes };
        Self {
            gate: span(i.gate),
            up: span(i.up),
            down: span(i.down),
        }
    }
}

impl From<LayerSpansInput> for LayerSpans {
    fn from(i: LayerSpansInput) -> Self {
        let span = |(offset, bytes, dtype): (usize, usize, GgmlDType)| ProjectionSpan {
            offset: offset as u32,
            bytes: bytes as u32,
            dtype,
        };
        Self {
            gate: span(i.gate),
            up: span(i.up),
            down: span(i.down),
        }
    }
}

/// Encoded header length for a pack of `num_layers` layers — where the records
/// begin, before the sector padding is applied.
#[cfg(test)]
fn open_header_len(num_layers: usize) -> usize {
    PackHeader {
        num_layers: num_layers as u32,
        experts_per_layer: 0,
        slot_bytes: 0,
        stride: 0,
        source_len: 0,
        source_sum: 0,
        int8_mode: 0,
        repack_fp: 0,
        pinned_layers: 0,
        layers: vec![
            LayerSpans {
                gate: ProjectionSpan {
                    offset: 0,
                    bytes: 0,
                    dtype: GgmlDType::Q4_K
                },
                up: ProjectionSpan {
                    offset: 0,
                    bytes: 0,
                    dtype: GgmlDType::Q4_K
                },
                down: ProjectionSpan {
                    offset: 0,
                    bytes: 0,
                    dtype: GgmlDType::Q4_K
                },
            };
            num_layers
        ],
    }
    .encoded_len()
}

/// Read a pack's header without opening it for reads — used by tooling and
/// tests to assert what a file on disk claims to be.
#[cfg(test)]
fn peek_header(path: &Path, encoded_len: usize) -> Result<PackHeader> {
    let mut f = File::open(path).map_err(|e| candle::Error::Msg(format!("peek open: {e}")))?;
    let mut buf = vec![0u8; encoded_len];
    f.read_exact(&mut buf)
        .map_err(|e| candle::Error::Msg(format!("peek read: {e}")))?;
    PackHeader::decode(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::direct_io::{AlignedScratch, DIRECT_IO_SECTOR};

    fn tmp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let d = std::env::temp_dir().join(format!("candle_expert_pack_{tag}_{nanos}"));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Two layers, three projections each, sized so the record needs padding to
    /// reach a sector — the interesting case for both the writer and the reader.
    fn spans() -> Vec<LayerSpansInput> {
        vec![
            LayerSpansInput {
                gate: (0, 300, GgmlDType::Q4_K),
                up: (512, 300, GgmlDType::Q4_K),
                down: (1024, 200, GgmlDType::Q6_K),
            },
            LayerSpansInput {
                gate: (0, 256, GgmlDType::Q4_K),
                up: (512, 256, GgmlDType::Q4_K),
                down: (1024, 256, GgmlDType::Q6_K),
            },
        ]
    }

    const SLOT_BYTES: usize = 1024 + 256;

    fn identity() -> PackIdentity {
        PackIdentity {
            source_len: 123_456,
            source_sum: 0xAABB_CCDD,
            int8_mode: 0,
            repack_fp: 0x0F0F_0F0F_5A5A_5A5A,
        }
    }

    /// A spec over `dir` for the two-layer fixture, with the identity and
    /// geometry a test may then vary one field of.
    fn spec(dir: &Path) -> PackSpec<'_> {
        PackSpec {
            dir: Some(dir),
            gguf_path: Path::new("model-q4.gguf"),
            identity: identity(),
            num_layers: 2,
            experts_per_layer: 2,
            // The fixture stores every layer, so the record indices are the
            // plain flat ones and the pinned-prefix arithmetic is exercised
            // separately by `a_pinned_prefix_shifts_every_record_index`.
            pinned_layers: 0,
            slot_bytes: SLOT_BYTES,
            layers: spans(),
        }
    }

    fn build(dir: &Path) -> ExpertPack {
        let source = super::open_or_create(spec(dir)).unwrap();
        let mut w = match source {
            PackSource::Build(w) => w,
            PackSource::Ready(_) => panic!("an empty directory cannot hold a ready pack"),
        };
        for layer in 0..2 {
            let s = spans()[layer];
            for expert in 0..2 {
                let tag = (layer * 2 + expert) as u8;
                w.write_expert(
                    layer,
                    expert,
                    &vec![tag; s.gate.1],
                    &vec![tag.wrapping_add(0x40); s.up.1],
                    &vec![tag.wrapping_add(0x80); s.down.1],
                )
                .unwrap();
            }
        }
        w.finish().unwrap()
    }

    /// The record stride is the slot image padded to a direct-I/O sector, and
    /// the first record starts after an equally padded header — both required
    /// for a positioned read to be legal at every record.
    #[test]
    fn every_record_starts_on_a_sector() {
        let dir = tmp_dir("sectors");
        let pack = build(&dir);
        assert_eq!(pack.stride(), round_up_sector(SLOT_BYTES));
        assert_eq!(pack.stride() % DIRECT_IO_SECTOR, 0);
        for layer in 0..2 {
            for expert in 0..2 {
                assert_eq!(
                    pack.offset_of(layer, expert).unwrap() % DIRECT_IO_SECTOR as u64,
                    0
                );
            }
        }
        drop(pack);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A record read back carries exactly the bytes written, at exactly the
    /// offsets the layer's geometry names, with the gaps zeroed.
    #[test]
    fn a_record_round_trips_byte_for_byte() {
        let dir = tmp_dir("roundtrip");
        let pack = build(&dir);
        let mut scratch = AlignedScratch::new();
        scratch.ensure(pack.stride()).unwrap();
        let dest = scratch.as_mut_slice(pack.stride());
        pack.read_into(1, 1, dest).unwrap();

        let layout = pack.layout(1);
        let tag = 3u8;
        assert_eq!(
            &dest[layout.gate.offset..layout.gate.offset + layout.gate.bytes],
            vec![tag; 256].as_slice()
        );
        assert_eq!(
            &dest[layout.up.offset..layout.up.offset + layout.up.bytes],
            vec![tag + 0x40; 256].as_slice()
        );
        assert_eq!(
            &dest[layout.down.offset..layout.down.offset + layout.down.bytes],
            vec![tag + 0x80; 256].as_slice()
        );
        // The gap between gate's payload and up's offset is untouched space.
        assert!(dest[256..512].iter().all(|&b| b == 0), "gap not zeroed");
        drop(pack);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two layers with different projection sizes must not bleed into each
    /// other through the reused record buffer.
    #[test]
    fn a_layer_change_clears_the_previous_layers_tail() {
        let dir = tmp_dir("layerchange");
        let pack = build(&dir);
        let mut scratch = AlignedScratch::new();
        scratch.ensure(pack.stride()).unwrap();
        let dest = scratch.as_mut_slice(pack.stride());
        // Layer 0's gate is 300 bytes, layer 1's is 256. Bytes 256..300 of a
        // layer-1 record would still hold layer 0's tag if the buffer were not
        // cleared at the layer boundary.
        pack.read_into(1, 0, dest).unwrap();
        assert!(
            dest[256..300].iter().all(|&b| b == 0),
            "layer 0's tail survived into layer 1: {:?}",
            &dest[256..300]
        );
        drop(pack);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The batch path reads the same bytes as the single-record path.
    #[test]
    fn a_batch_read_agrees_with_the_single_reads() {
        let dir = tmp_dir("batch");
        let pack = build(&dir);
        let stride = pack.stride();
        let mut batch = AlignedScratch::new();
        batch.ensure(stride * 4).unwrap();
        let all = batch.as_mut_slice(stride * 4);
        let mut chunks: Vec<&mut [u8]> = Vec::new();
        let mut rest = all;
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
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **The runtime paths serve repeats from the in-process cold cache.**
    /// Proven observably: after a record is read once, clobbering it on disk
    /// must NOT change what the runtime paths return — the repeat is a memcpy
    /// from the cache, not a file read. (The verified startup path is exempt:
    /// it bypasses the cache by design.)
    #[test]
    fn a_runtime_reread_is_served_from_the_cold_cache() {
        let dir = tmp_dir("coldcache");
        let pack = build(&dir);
        let stride = pack.stride();
        let mut scratch = AlignedScratch::new();
        scratch.ensure(stride).unwrap();

        // First touch fills the cache.
        let first = {
            let dest = scratch.as_mut_slice(stride);
            pack.read_into(1, 1, dest).unwrap();
            dest.to_vec()
        };

        // Zero the record's whole region on disk (header/trailer intact).
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(pack.path())
                .unwrap();
            let stride64 = round_up_sector(SLOT_BYTES) as u64;
            let header_bytes = round_up_sector(super::open_header_len(2)) as u64;
            f.seek(SeekFrom::Start(header_bytes + 3 * stride64))
                .unwrap();
            f.write_all(&vec![0u8; stride]).unwrap();
            f.sync_all().unwrap();
        }

        // Single-record repeat: cache-served, original bytes.
        {
            let dest = scratch.as_mut_slice(stride);
            dest.fill(0xEE);
            pack.read_into(1, 1, dest).unwrap();
            assert_eq!(dest, first.as_slice(), "read_into re-read the file");
        }
        // Batch repeat: cache-served, original bytes.
        {
            let dest = scratch.as_mut_slice(stride);
            dest.fill(0xEE);
            pack.read_many_unverified(vec![PackRead {
                layer: 1,
                expert: 1,
                dest,
            }])
            .unwrap();
            let dest = scratch.as_slice(stride);
            assert_eq!(
                dest,
                first.as_slice(),
                "read_many_unverified re-read the file"
            );
        }
        drop(pack);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **Bit rot in a record is caught, not served.** The pack lives beside the
    /// checkpoint for as long as the checkpoint does, and its bytes become
    /// weights with no further validation — so a flipped bit on the medium
    /// would otherwise be silent.
    #[test]
    fn a_corrupted_record_is_refused() {
        let dir = tmp_dir("corrupt");
        let path = {
            let pack = build(&dir);
            pack.path().to_path_buf()
        };
        // Flip one bit inside L1E1's payload, leaving every offset, the header
        // and the trailer intact.
        {
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let stride = round_up_sector(SLOT_BYTES) as u64;
            let header_bytes = round_up_sector(
                super::open_header_len(2), // 2 layers
            ) as u64;
            f.seek(SeekFrom::Start(header_bytes + 3 * stride + 16))
                .unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            f.seek(SeekFrom::Start(header_bytes + 3 * stride + 16))
                .unwrap();
            f.write_all(&[b[0] ^ 0x01]).unwrap();
            f.sync_all().unwrap();
        }

        let PackSource::Ready(pack) = super::open_or_create(spec(&dir)).unwrap() else {
            panic!("the header and trailer are intact, so the pack still opens")
        };
        let mut scratch = AlignedScratch::new();
        scratch.ensure(pack.stride()).unwrap();
        let dest = scratch.as_mut_slice(pack.stride());
        // The bulk path is where the check lives: a fill that includes the
        // damaged record refuses rather than warming a wrong expert.
        let e = pack
            .read_many(vec![PackRead {
                layer: 1,
                expert: 1,
                dest,
            }])
            .unwrap_err()
            .to_string();
        assert!(e.contains("corrupt"), "{e}");
        assert!(e.contains("L1E1"), "{e}");
        // A neighbouring record is untouched and passes.
        let dest = scratch.as_mut_slice(pack.stride());
        pack.read_many(vec![PackRead {
            layer: 1,
            expert: 0,
            dest,
        }])
        .unwrap();
        drop(pack);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A second `open_or_create` over the same directory and identity finds the
    /// pack ready — this is what makes a restart skip the repack.
    #[test]
    fn a_matching_pack_is_reused() {
        let dir = tmp_dir("reuse");
        let pack = build(&dir);
        let path = pack.path().to_path_buf();
        drop(pack);
        let again = super::open_or_create(spec(&dir)).unwrap();
        match again {
            PackSource::Ready(p) => assert_eq!(p.path(), path),
            PackSource::Build(_) => panic!("a matching pack was not reused"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// With no directory the pack goes to the temp dir and is unlinked as soon
    /// as it is open — the name is gone, the bytes are still readable through
    /// the handles, and nothing is left behind however the process ends.
    #[test]
    fn an_ephemeral_pack_is_unlinked_but_still_readable() {
        let source = super::open_or_create(PackSpec {
            dir: None,
            ..spec(Path::new("."))
        })
        .unwrap();
        let PackSource::Build(mut w) = source else {
            panic!("a temp pack is always built")
        };
        for layer in 0..2 {
            let s = spans()[layer];
            for expert in 0..2 {
                let tag = (layer * 2 + expert) as u8;
                w.write_expert(
                    layer,
                    expert,
                    &vec![tag; s.gate.1],
                    &vec![tag; s.up.1],
                    &vec![tag; s.down.1],
                )
                .unwrap();
            }
        }
        let pack = w.finish().unwrap();
        assert!(
            !pack.path().exists(),
            "the ephemeral pack still has a directory entry: {}",
            pack.path().display()
        );
        let mut scratch = AlignedScratch::new();
        scratch.ensure(pack.stride()).unwrap();
        let dest = scratch.as_mut_slice(pack.stride());
        pack.read_into(1, 1, dest).unwrap();
        let layout = pack.layout(1);
        assert_eq!(
            &dest[layout.gate.offset..layout.gate.offset + layout.gate.bytes],
            vec![3u8; 256].as_slice(),
            "an unlinked pack must still read through its open handles"
        );
    }

    /// A different checkpoint under the same directory does not reuse the pack,
    /// even though the geometry is identical.
    #[test]
    fn a_different_checkpoint_rebuilds() {
        let dir = tmp_dir("identity");
        let pack = build(&dir);
        drop(pack);
        let again = super::open_or_create(PackSpec {
            identity: PackIdentity {
                source_len: 999,
                ..identity()
            },
            ..spec(&dir)
        })
        .unwrap();
        assert!(
            matches!(again, PackSource::Build(_)),
            "a pack from another checkpoint was reused"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **The case geometry cannot catch.** A repack that emits different bytes
    /// at identical sizes, offsets and dtypes — a changed permutation, moved
    /// rounding — leaves every other field of the identity equal. Without the
    /// fingerprint this pack would be reused and would serve wrong weights for
    /// the whole model, silently.
    #[test]
    fn a_changed_repack_formula_rebuilds() {
        let dir = tmp_dir("fingerprint");
        let pack = build(&dir);
        drop(pack);
        let again = super::open_or_create(PackSpec {
            identity: PackIdentity {
                repack_fp: identity().repack_fp ^ 1,
                ..identity()
            },
            ..spec(&dir)
        })
        .unwrap();
        assert!(
            matches!(again, PackSource::Build(_)),
            "a pack from a different repack was reused"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The fingerprint rides in the identity untouched, and moves nothing else.
    #[test]
    fn the_fingerprint_is_carried_independently() {
        let gguf = vec![7u8; 64];
        let a = PackIdentity::of(&gguf, Int8Mode::Off, 0xAAAA);
        let same = PackIdentity::of(&gguf, Int8Mode::Off, 0xAAAA);
        let moved = PackIdentity::of(&gguf, Int8Mode::Off, 0xBBBB);
        assert_eq!(a, same);
        assert_ne!(a.repack_fp, moved.repack_fp);
        assert_eq!(
            (a.source_len, a.source_sum, a.int8_mode),
            (moved.source_len, moved.source_sum, moved.int8_mode),
            "only the fingerprint differs"
        );
    }

    /// Two writers on one checkpoint must not share a temp path — the pack is
    /// deliberately shared between workspaces, so a fixed `.partial` sibling
    /// would let them interleave into one file and both publish it.
    #[test]
    fn concurrent_builders_do_not_share_a_temp_file() {
        let dir = tmp_dir("concurrent");
        let PackSource::Build(a) = super::open_or_create(spec(&dir)).unwrap() else {
            panic!("expected a build")
        };
        let PackSource::Build(b) = super::open_or_create(spec(&dir)).unwrap() else {
            panic!("expected a build")
        };
        assert_ne!(a.tmp, b.tmp, "two writers took the same partial path");
        assert_eq!(a.final_path, b.final_path, "but the same destination");
        drop(a);
        drop(b);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Changing a layer's repacked geometry invalidates the pack even when the
    /// checkpoint identity is unchanged — the case a version bump would miss.
    #[test]
    fn a_changed_repack_layout_rebuilds() {
        let dir = tmp_dir("geometry");
        let pack = build(&dir);
        drop(pack);
        let mut moved = spans();
        moved[0].down.0 += 256;
        let again = super::open_or_create(PackSpec {
            slot_bytes: SLOT_BYTES + 256,
            layers: moved,
            ..spec(&dir)
        })
        .unwrap();
        assert!(
            matches!(again, PackSource::Build(_)),
            "a pack with a stale record layout was reused"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A writer dropped before `finish` leaves nothing behind, and the header
    /// never appears under the final name.
    #[test]
    fn an_abandoned_build_leaves_no_file() {
        let dir = tmp_dir("abandoned");
        let source = super::open_or_create(spec(&dir)).unwrap();
        let PackSource::Build(mut w) = source else {
            panic!("expected a build")
        };
        let s = spans()[0];
        w.write_expert(
            0,
            0,
            &vec![1; s.gate.1],
            &vec![2; s.up.1],
            &vec![3; s.down.1],
        )
        .unwrap();
        let partial = w.tmp.clone();
        drop(w);
        assert!(!partial.exists(), "the partial file survived the drop");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "the directory is not empty"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Records must be written in index order, once each: a gap would leave a
    /// record of zeroes that reads back as a valid-looking expert.
    #[test]
    fn out_of_order_writes_are_refused() {
        let dir = tmp_dir("order");
        let source = super::open_or_create(spec(&dir)).unwrap();
        let PackSource::Build(mut w) = source else {
            panic!("expected a build")
        };
        let s = spans()[0];
        let e = w
            .write_expert(
                0,
                1,
                &vec![1; s.gate.1],
                &vec![2; s.up.1],
                &vec![3; s.down.1],
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("sequential"), "{e}");
        drop(w);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A projection whose length disagrees with the layer's geometry is a
    /// mismatch between the repack and the header, and must not be padded or
    /// truncated into place.
    #[test]
    fn a_wrong_sized_projection_is_refused() {
        let dir = tmp_dir("size");
        let source = super::open_or_create(spec(&dir)).unwrap();
        let PackSource::Build(mut w) = source else {
            panic!("expected a build")
        };
        let s = spans()[0];
        let e = w
            .write_expert(
                0,
                0,
                &vec![1; s.gate.1 - 1],
                &vec![2; s.up.1],
                &vec![3; s.down.1],
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("geometry"), "{e}");
        drop(w);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `finish` on a short build fails rather than publishing a pack whose tail
    /// records were never written.
    #[test]
    fn finishing_early_is_refused() {
        let dir = tmp_dir("short");
        let source = super::open_or_create(spec(&dir)).unwrap();
        let PackSource::Build(mut w) = source else {
            panic!("expected a build")
        };
        let s = spans()[0];
        w.write_expert(
            0,
            0,
            &vec![1; s.gate.1],
            &vec![2; s.up.1],
            &vec![3; s.down.1],
        )
        .unwrap();
        let e = w.finish().unwrap_err().to_string();
        assert!(e.contains("short"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The published header is byte-identical to what the caller asked for.
    #[test]
    fn the_published_header_matches_the_request() {
        let dir = tmp_dir("header");
        let pack = build(&dir);
        let path = pack.path().to_path_buf();
        let want = PackHeader {
            num_layers: 2,
            experts_per_layer: 2,
            slot_bytes: SLOT_BYTES as u32,
            stride: round_up_sector(SLOT_BYTES) as u64,
            source_len: identity().source_len,
            source_sum: identity().source_sum,
            int8_mode: identity().int8_mode,
            repack_fp: identity().repack_fp,
            pinned_layers: 0,
            layers: spans().into_iter().map(LayerSpans::from).collect(),
        };
        drop(pack);
        assert_eq!(peek_header(&path, want.encoded_len()).unwrap(), want);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A pinned prefix shifts every record index**, and the pack must refuse
    /// the prefix outright rather than serve the record `pinned_layers` further
    /// along — which would be a real expert, of the wrong layer, with no error.
    ///
    /// Written as bytes rather than a round trip: layer 1 expert 1 is the last
    /// record of a 3-layer model that pins 1, so it must land at flat index 3,
    /// not 5.
    #[test]
    fn a_pinned_prefix_shifts_every_record_index() {
        let dir = tmp_dir("pinned");
        let mut s = spec(&dir);
        s.num_layers = 3;
        s.pinned_layers = 1;
        let g = spans()[0];
        s.layers = vec![g, g, g];
        let PackSource::Build(mut w) = super::open_or_create(s).unwrap() else {
            panic!("a fresh directory must yield a writer");
        };
        // The pinned layer is refused, and refusing it does not consume an index.
        assert!(w
            .write_expert(
                0,
                0,
                &vec![0u8; g.gate.1],
                &vec![0u8; g.up.1],
                &vec![0u8; g.down.1]
            )
            .is_err());
        for layer in 1..3 {
            for expert in 0..2 {
                let tag = (layer * 2 + expert) as u8;
                w.write_expert(
                    layer,
                    expert,
                    &vec![tag; g.gate.1],
                    &vec![tag; g.up.1],
                    &vec![tag; g.down.1],
                )
                .unwrap();
            }
        }
        let pack = w.finish().unwrap();
        assert_eq!(pack.pinned_layers(), 1);
        // Four records, starting at layer 1 — not six.
        assert_eq!(pack.record_index(1, 0).unwrap(), 0);
        assert_eq!(pack.record_index(2, 1).unwrap(), 3);
        assert!(pack.record_index(0, 0).is_err(), "served a pinned layer");

        // And the bytes at that offset are the ones written for it.
        let mut got = vec![0u8; pack.stride()];
        pack.read_into(2, 1, &mut got).unwrap();
        assert_eq!(got[0], 5, "L2E1 read back another layer's record");

        drop(pack);
        std::fs::remove_dir_all(&dir).ok();
    }
}
