//! An index page resident on the span: its rows in QSA-index arena slots, in a
//! layout the scorer reads in place, held by every cache whose window contains it.
//!
//! # Why a page is resident rather than rebuilt per cache
//!
//! A turn's K/V lives once in the KV arenas and is Arc-injected into every slot
//! that selects it — no copy, however many slots. Its index used to be rebuilt per
//! slot per projection: decoded from the record on the host, uploaded into a
//! fresh device tensor, and copied again into the scorer's layout, both buffers
//! from the plain CUDA pool. Every slot carried its whole selected history's
//! index twice, outside the span, and every view carve placed it again. On a
//! card whose span took everything the driver had, that pool growth
//! over-committed the device and the display driver paged it.
//!
//! A page is now placed once, into slots of the tenant the live index already
//! uses, and handed around by `Arc`: a fork, a view carve and a second push of the
//! same page share the same rows. The span accounts for it, and compaction moves
//! it with the rest of the tenant.
//!
//! # Chunks
//!
//! A page is one or more slots of [`PAGE_BLOCKS`] rows — the stride the live
//! tail's key pages use, so the tenant stays on two strides and one compaction
//! pass covers both. Each chunk is one entry of the scorer's descriptor table;
//! the rows of a page are consecutive global rows, so every chunk of a page
//! shares the page's rotation offset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use candle::{Device, Result, Tensor};
use candle_kernels::simple::qsa_page_place::{run_qsa_page_place, PLACE_JOB_WORDS};
use candle_nn::kv_cache::{relocate_tensor, ArenaSlot};

use super::index_keys::{
    alloc_key_pages, bytes_ptr, scale_bytes, scale_plane_offset, scales_of, write_bytes,
    PAGE_BLOCKS,
};
use super::key_q8::{blocks, KeyPlanes, Q8Keys};
use super::place::PLACE_TILE_R;
use crate::models::piece_key::PieceKey;

/// Codes in one group of the scorer's blocked layout — one 16-byte load.
pub const CODE_GROUP: usize = 16;

/// How a resident page's codes sit in each of its chunks. The scales are
/// row-major behind the codes in both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkLayout {
    /// `[d/16, rows, 16]` per chunk: channel-blocked, so a warp scoring
    /// consecutive candidates reads one contiguous run. What a page made from a
    /// record is placed as.
    Blocked,
    /// `[rows, d]` per chunk: a live tail's key pages, closed where they stand
    /// rather than copied.
    RowMajor,
}

/// One chunk's entry in the scorer's descriptor table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkDesc {
    /// The chunk's codes.
    pub ptr: u64,
    /// The chunk's scales, `[rows, blocks(d)]` `f16` row-major.
    pub scales: u64,
    pub rows: usize,
    /// Bytes between code group `c` and `c + 1` of one row.
    pub group_stride: i64,
    /// Bytes between row `j` and `j + 1` of one code group.
    pub row_stride: i64,
}

/// One sealed piece's rows for one attention layer, resident on the span.
#[derive(Debug)]
pub struct ResidentPage {
    /// One key page (`index_keys::key_page_bytes`) per chunk; chunk `c` holds
    /// rows `[c·PAGE_BLOCKS, (c+1)·PAGE_BLOCKS)`. Behind a lock because a compaction
    /// moves them between forwards while every holder keeps the same `Arc`.
    chunks: Mutex<Vec<Tensor>>,
    rows: usize,
    /// Tokens the last row covers, `1..=ratio`.
    last_cells: usize,
    d: usize,
    layout: ChunkLayout,
    /// The record this page was placed from, once it is known — a seal of it
    /// reads that record back exactly (`seal_memo`). A page closed over a live
    /// tail was placed from none.
    source: OnceLock<PieceKey>,
}

/// Rows of chunk `c` of a page of `rows` rows.
fn chunk_rows(rows: usize, c: usize) -> usize {
    (rows - c * PAGE_BLOCKS).min(PAGE_BLOCKS)
}

impl ResidentPage {
    /// Pages made from host keys — one `(keys, last_cells)` page per entry of
    /// `pages`, each a record's Q8_0 rows — placed in ONE launch.
    ///
    /// The codes are uploaded into staging slots of the same tenant and placed
    /// from there into the page's own slots; the scales, row-major in both
    /// layouts, are uploaded straight into the page. The job table rides the
    /// device's held staging scratch, so nothing is taken from the CUDA pool; the
    /// staging goes back when this returns, behind the launch on the same stream.
    ///
    /// **Between forwards.** A slot claim that needs a new arena takes the arena
    /// window.
    pub fn place_host(
        pages: &[(&Q8Keys, usize)],
        d: usize,
        device: &Device,
    ) -> Result<Vec<Arc<Self>>> {
        let scales: Vec<Vec<u8>> = pages.iter().map(|(k, _)| scale_bytes(&k.scales)).collect();
        let planes: Vec<(KeyPlanes<'_>, usize)> = pages
            .iter()
            .zip(&scales)
            .map(|(&(k, last), s)| {
                (
                    KeyPlanes {
                        codes: &k.codes,
                        scales: s,
                        dim: k.dim,
                    },
                    last,
                )
            })
            .collect();
        Self::place_planes(&planes, d, device)
    }

    /// [`Self::place_host`] from borrowed planes — a page's own bytes, read in
    /// place. The uploads are queued on the device's stream; from pinned
    /// memory they run asynchronously, so a caller whose bytes may be pinned
    /// waits for the stream before it lets them go.
    pub fn place_planes(
        pages: &[(KeyPlanes<'_>, usize)],
        d: usize,
        device: &Device,
    ) -> Result<Vec<Arc<Self>>> {
        let Device::Cuda(cuda) = device else {
            candle::bail!("qsa resident page: placed on a CUDA device");
        };
        if !d.is_multiple_of(CODE_GROUP) {
            candle::bail!(
                "qsa resident page: head_dim {d} is not a multiple of {CODE_GROUP} — the \
                 scorer's channel-blocked layout is built in groups of {CODE_GROUP} codes"
            );
        }
        // Placement allocates and uploads: it runs eagerly for its whole extent,
        // behind whatever this thread has recorded.
        let _eager = cuda.pause_capture()?;
        let stream = cuda.cuda_stream();
        let mut out = Vec::with_capacity(pages.len());
        let mut staging: Vec<Tensor> = Vec::new();
        let mut jobs: Vec<i64> = Vec::new();
        let mut max_rows = 0usize;
        for &(host, last_cells) in pages {
            if host.dim != d && !host.is_empty() {
                candle::bail!(
                    "qsa resident page: {}-wide keys placed as a {d}-wide page",
                    host.dim
                );
            }
            let rows = host.n_rows();
            let n = rows.div_ceil(PAGE_BLOCKS);
            let chunks = alloc_key_pages(device, d, n)?;
            let stage = alloc_key_pages(device, d, n)?;
            for c in 0..n {
                let first = c * PAGE_BLOCKS;
                let used = chunk_rows(rows, c);
                let (codes, scales) = host.rows(first, used);
                // Both uploads are ordered on the stream the placement below
                // and every later reader use.
                write_bytes(&stage[c], 0, codes)?;
                write_bytes(&chunks[c], scale_plane_offset(d), scales)?;
                jobs.push(bytes_ptr(&stage[c])? as i64);
                jobs.push(bytes_ptr(&chunks[c])? as i64);
                jobs.push(used as i64);
                max_rows = max_rows.max(used);
            }
            staging.extend(stage);
            out.push(Arc::new(Self {
                chunks: Mutex::new(chunks),
                rows,
                last_cells,
                d,
                layout: ChunkLayout::Blocked,
                source: OnceLock::new(),
            }));
        }
        let n_jobs = jobs.len() / PLACE_JOB_WORDS;
        if n_jobs > 0 && max_rows > 0 {
            // The job table rides the device's held staging scratch: a few words
            // that live from this upload to the launch's read, which a fresh
            // tensor per placement took from the driver every time a page was
            // pushed.
            cuda.with_staged_upload(&jobs, |table| {
                candle::set_kernel_breadcrumb("run_qsa_page_place", file!(), line!());
                // SAFETY: every job names a live staging slot and a live chunk
                // slot of `used` rows, both held until after this launch is
                // queued; `table` holds the jobs until the launch has read them.
                unsafe {
                    run_qsa_page_place(
                        table as *const i64,
                        d as i32,
                        n_jobs as i32,
                        max_rows as i32,
                        PLACE_TILE_R as i32,
                        stream.cu_stream() as *mut c_void,
                    );
                }
                Ok(())
            })?;
        }
        // The staging slots go back here, behind the launch that reads them: a
        // slot's next holder works on the same stream.
        drop(staging);
        Ok(out)
    }

    /// A page over a live tail's key pages, taken as they stand — `chunks` each a
    /// key page holding its codes row-major.
    pub fn from_key_pages(
        chunks: Vec<Tensor>,
        rows: usize,
        last_cells: usize,
        d: usize,
    ) -> Result<Arc<Self>> {
        if chunks.len() != rows.div_ceil(PAGE_BLOCKS) {
            candle::bail!(
                "qsa resident page: {} key page(s) for {rows} rows of {PAGE_BLOCKS}",
                chunks.len()
            );
        }
        Ok(Arc::new(Self {
            chunks: Mutex::new(chunks),
            rows,
            last_cells,
            d,
            layout: ChunkLayout::RowMajor,
            source: OnceLock::new(),
        }))
    }

    /// Record that this page was placed from the record `key` names. Set once:
    /// a page's rows never change, so neither does what it was placed from.
    pub fn placed_from(&self, key: PieceKey) {
        let _ = self.source.set(key);
    }

    /// The record this page was placed from, if it was.
    pub fn source(&self) -> Option<&PieceKey> {
        self.source.get()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn last_cells(&self) -> usize {
        self.last_cells
    }

    pub fn layout(&self) -> ChunkLayout {
        self.layout
    }

    /// Tokens this page covers: full rows at `ratio`, plus the short last one.
    pub fn tokens(&self, ratio: usize) -> usize {
        if self.rows == 0 {
            0
        } else {
            (self.rows - 1) * ratio + self.last_cells
        }
    }

    /// One descriptor per chunk, in row order.
    pub fn descriptors(&self) -> Result<Vec<ChunkDesc>> {
        let chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        chunks
            .iter()
            .enumerate()
            .map(|(c, t)| {
                let rows = chunk_rows(self.rows, c);
                let (group_stride, row_stride) = match self.layout {
                    ChunkLayout::Blocked => ((rows * CODE_GROUP) as i64, CODE_GROUP as i64),
                    ChunkLayout::RowMajor => (CODE_GROUP as i64, self.d as i64),
                };
                let ptr = bytes_ptr(t)?;
                Ok(ChunkDesc {
                    ptr,
                    scales: ptr + scale_plane_offset(self.d) as u64,
                    rows,
                    group_stride,
                    row_stride,
                })
            })
            .collect()
    }

    /// The page's keys as they are stored, row-major, read back to the host —
    /// for a seal that hands the page on as a record.
    pub fn host_rows(&self) -> Result<Q8Keys> {
        let chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        let d = self.d;
        let nb = blocks(d);
        let mut out = Q8Keys {
            codes: vec![0u8; self.rows * d],
            scales: Vec::with_capacity(self.rows * nb),
            dim: d,
        };
        for (c, t) in chunks.iter().enumerate() {
            let rows = chunk_rows(self.rows, c);
            let first = c * PAGE_BLOCKS;
            // One readback per chunk, codes through the end of the rows' scales:
            // each `to_vec1` waits for the device, and the two planes sit in the
            // one tensor.
            let scales_at = scale_plane_offset(d);
            let scale_bytes = rows * nb * 2;
            let whole = t.narrow(0, 0, scales_at + scale_bytes)?.to_vec1::<u8>()?;
            let vals = &whole[..rows * d];
            let dst = &mut out.codes[first * d..(first + rows) * d];
            match self.layout {
                ChunkLayout::RowMajor => dst.copy_from_slice(vals),
                // `[d/16, rows, 16]`: group `g` of row `j` sits at
                // `(g·rows + j)·16`.
                ChunkLayout::Blocked => {
                    for g in 0..d / CODE_GROUP {
                        for j in 0..rows {
                            let s = (g * rows + j) * CODE_GROUP;
                            let o = j * d + g * CODE_GROUP;
                            dst[o..o + CODE_GROUP].copy_from_slice(&vals[s..s + CODE_GROUP]);
                        }
                    }
                }
            }
            out.scales
                .extend(scales_of(&whole[scales_at..scales_at + scale_bytes]));
        }
        Ok(out)
    }

    /// Move every chunk whose slot is a planned source. A page shared by several
    /// caches is moved by the first to reach it: the move leaves the map when it
    /// is taken, so the rest find nothing to do and read the new address through
    /// the same `Arc`.
    pub fn relocate(&self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        let mut chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        let mut moved = 0;
        for t in chunks.iter_mut() {
            moved += usize::from(relocate_tensor(t, moves)?);
        }
        Ok(moved)
    }
}

/// Resident pages by the record they were placed from, so every slot that
/// injects the same piece — a section every conversation borrows, a turn a
/// projection selects into many slots, a fork's inherited prefix — holds the
/// same rows.
///
/// Pages are held **per layer**: a piece in a sequence's pool holds only its
/// sampled layers' pages, and placing it in a slot later places the other
/// layers and shares the sampled ones (`docs/hit_driven_projection.md` §7.2).
///
/// The registry holds no page alive: its entries are weak, so a piece's rows go
/// back to the tenant when the last cache or pool holding them drops, and the
/// next push of that piece places it afresh. A dead entry is swept on the next
/// insert.
#[derive(Debug, Default)]
pub struct PageRegistry {
    /// One entry per KV layer; `None` where nothing of that layer is placed.
    map: Mutex<HashMap<PieceKey, Vec<Option<Weak<ResidentPage>>>>>,
}

impl PageRegistry {
    /// `key`'s resident page at KV layer `kv`, while something still holds it.
    pub fn layer(&self, key: &PieceKey, kv: usize) -> Option<Arc<ResidentPage>> {
        let map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        map.get(key)?.get(kv)?.as_ref()?.upgrade()
    }

    /// Record `placed` — `(kv layer, page)` — as resident pages of `key`, in
    /// an entry of `layers` layers, beside whatever of it is already resident;
    /// entries whose pages have all dropped are swept.
    pub fn insert(&self, key: PieceKey, layers: usize, placed: &[(usize, Arc<ResidentPage>)]) {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, layers| {
            layers
                .iter()
                .any(|w| w.as_ref().is_some_and(|w| w.strong_count() > 0))
        });
        let entry = map.entry(key).or_insert_with(|| vec![None; layers]);
        if entry.len() < layers {
            entry.resize(layers, None);
        }
        for (kv, page) in placed {
            entry[*kv] = Some(Arc::downgrade(page));
        }
    }

    /// Entries standing — for a test, and for the memory report.
    pub fn len(&self) -> usize {
        self.map.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use candle::DType;

    use super::super::paged_index::{aux_parts, decode_page, encode_aux, SealedIndex};
    use super::*;

    fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => Some(d),
            _ => {
                eprintln!("skipping: CUDA device required");
                None
            }
        }
    }

    /// `n` keys of width `d` whose codes are all distinct enough to tell a
    /// misplaced group from a placed one.
    fn keys(n: usize, d: usize, seed: usize) -> Q8Keys {
        let vals: Vec<f32> = (0..n * d)
            .map(|i| ((i * 7 + seed * 13) % 253) as f32 - 126.0)
            .collect();
        Q8Keys::quantize(&vals, d)
    }

    /// Pages placed in one launch hold exactly the keys they were given, each
    /// chunk's codes in the channel-blocked layout — `[d/16, rows, 16]` over the
    /// chunk's own row count — with its scales row-major behind them, and
    /// describe themselves to the scorer by that layout.
    #[test]
    fn placed_pages_hold_their_rows_channel_blocked() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let d = 32usize;
        let long = keys(PAGE_BLOCKS + 3, d, 0);
        let short = keys(2, d, 5);
        let pages = ResidentPage::place_host(&[(&long, 3), (&short, 4)], d, &device)?;
        assert_eq!(pages.len(), 2);

        assert_eq!(pages[0].rows(), PAGE_BLOCKS + 3);
        assert_eq!(pages[0].last_cells(), 3);
        assert_eq!(pages[0].host_rows()?, long);
        assert_eq!(pages[1].host_rows()?, short);

        // The short page's one chunk, raw: group 0 of rows 0 and 1, then group 1
        // of rows 0 and 1; then, at the scale plane, each row's one scale.
        let chunk = pages[1].chunks.lock().unwrap()[0].clone();
        let raw = chunk.narrow(0, 0, 2 * d)?.to_vec1::<u8>()?;
        let c = &short.codes;
        let want: Vec<u8> = [&c[0..16], &c[32..48], &c[16..32], &c[48..64]].concat();
        assert_eq!(raw, want);
        let scales = chunk.narrow(0, scale_plane_offset(d), 4)?.to_vec1::<u8>()?;
        assert_eq!(scales, scale_bytes(&short.scales));

        let desc = pages[0].descriptors()?;
        let shape: Vec<_> = desc
            .iter()
            .map(|c| (c.rows, c.group_stride, c.row_stride, c.scales - c.ptr))
            .collect();
        let plane = scale_plane_offset(d) as u64;
        assert_eq!(
            shape,
            vec![
                (PAGE_BLOCKS, (PAGE_BLOCKS * 16) as i64, 16, plane),
                (3, 48, 16, plane)
            ]
        );
        Ok(())
    }

    /// **A closed page's record, placed and sealed again, is the same record
    /// byte for byte** — what lets a tail seal hand back the record a page was
    /// placed from instead of reading the page off the card (`seal_memo`). The
    /// record is made the way the seal makes one: each layer's page as
    /// `(host_rows, last_cells, no open rows)`; it is placed the way a push
    /// places one, from its decoded sections.
    #[test]
    fn a_placed_records_seal_is_the_record() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let d = 32usize;
        let seal = |pages: &[Arc<ResidentPage>]| -> Result<Vec<u8>> {
            let layers = pages
                .iter()
                .map(|p| {
                    Ok(SealedIndex {
                        keys: p.host_rows()?,
                        last_cells: p.last_cells(),
                        open: Vec::new(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            encode_aux(&[], &layers)
        };
        let first = [keys(PAGE_BLOCKS + 5, d, 1), keys(PAGE_BLOCKS + 5, d, 2)];
        let placed = ResidentPage::place_host(&[(&first[0], 3), (&first[1], 3)], d, &device)?;
        let record = seal(&placed)?;
        let (_, sections) = aux_parts(&record)?;
        let decoded = sections
            .iter()
            .map(|s| decode_page(s))
            .collect::<Result<Vec<_>>>()?;
        let host: Vec<(&Q8Keys, usize)> = decoded.iter().map(|s| (&s.keys, s.last_cells)).collect();
        let again = ResidentPage::place_host(&host, d, &device)?;
        assert_eq!(seal(&again)?, record);
        // What it was placed from is set once and kept.
        let key = PieceKey::of(&record);
        assert_eq!(again[0].source(), None);
        again[0].placed_from(key);
        again[0].placed_from(PieceKey::of(b"another record"));
        assert_eq!(again[0].source(), Some(&key));
        Ok(())
    }

    /// A key page holding `keys` row-major, as the append leaves one.
    fn key_page(keys: &Q8Keys, device: &Device) -> Result<Tensor> {
        let d = keys.dim;
        let mut bytes = vec![0u8; super::super::index_keys::key_page_bytes(d)];
        bytes[..keys.codes.len()].copy_from_slice(&keys.codes);
        let at = scale_plane_offset(d);
        let scales = scale_bytes(&keys.scales);
        bytes[at..at + scales.len()].copy_from_slice(&scales);
        let n = bytes.len();
        Tensor::from_vec(bytes, (n,), device)
    }

    /// A closed tail's key pages are taken as they stand: row-major, read at a
    /// row stride of `d` bytes.
    #[test]
    fn key_pages_are_read_row_major_in_place() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let d = 16usize;
        let rows = PAGE_BLOCKS + 1;
        let all = keys(rows, d, 3);
        let (codes, scales) = all.rows(0, PAGE_BLOCKS);
        let first = Q8Keys {
            codes: codes.to_vec(),
            scales: scales.to_vec(),
            dim: d,
        };
        let chunks = vec![
            key_page(&first, &device)?,
            key_page(&all.from_row(PAGE_BLOCKS), &device)?,
        ];
        let page = ResidentPage::from_key_pages(chunks, rows, 2, d)?;
        assert_eq!(page.layout(), ChunkLayout::RowMajor);
        assert_eq!(page.host_rows()?, all);
        let shape: Vec<_> = page
            .descriptors()?
            .iter()
            .map(|c| (c.rows, c.group_stride, c.row_stride))
            .collect();
        assert_eq!(shape, vec![(PAGE_BLOCKS, 16, 16), (1, 16, 16)]);
        Ok(())
    }

    /// A chunk count that does not cover the rows is refused: the scorer would
    /// read a page's last rows through a chunk that is not there.
    #[test]
    fn key_pages_must_cover_the_rows() {
        let chunk = Tensor::zeros((PAGE_BLOCKS, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(ResidentPage::from_key_pages(vec![chunk], PAGE_BLOCKS + 1, 1, 4).is_err());
    }

    /// Tokens are whole rows at `ratio` plus the short last one, and an empty
    /// page covers none.
    #[test]
    fn a_page_covers_its_full_rows_and_its_short_last_one() {
        let chunk = Tensor::zeros((PAGE_BLOCKS, 4), DType::F32, &Device::Cpu).unwrap();
        let page = ResidentPage::from_key_pages(vec![chunk], 5, 3, 4).unwrap();
        assert_eq!(page.tokens(4), 19);
        let empty = ResidentPage::from_key_pages(Vec::new(), 0, 1, 4).unwrap();
        assert_eq!(empty.tokens(4), 0);
    }

    /// The registry hands back the pages every holder shares while one holds
    /// them, and forgets a piece once none does.
    #[test]
    fn the_registry_shares_pages_while_they_are_held() {
        let chunk = Tensor::zeros((PAGE_BLOCKS, 4), DType::F32, &Device::Cpu).unwrap();
        let page = ResidentPage::from_key_pages(vec![chunk], 2, 1, 4).unwrap();
        let key = PieceKey::of(b"a sealed piece");
        assert_eq!(
            key,
            PieceKey::of(b"a sealed piece"),
            "equal records, one key"
        );
        assert_ne!(key, PieceKey::of(b"another piece"));
        let reg = PageRegistry::default();
        reg.insert(key, 3, &[(1, Arc::clone(&page))]);

        assert!(reg.layer(&key, 0).is_none(), "an unplaced layer");
        let got = reg.layer(&key, 1).expect("a held page is found");
        assert!(Arc::ptr_eq(&got, &page));
        drop(got);

        // A later placement of another layer joins the entry.
        let other = ResidentPage::from_key_pages(
            vec![Tensor::zeros((PAGE_BLOCKS, 4), DType::F32, &Device::Cpu).unwrap()],
            2,
            1,
            4,
        )
        .unwrap();
        reg.insert(key, 3, &[(2, Arc::clone(&other))]);
        assert!(reg.layer(&key, 1).is_some(), "the earlier layer stays");
        assert!(reg.layer(&key, 2).is_some());

        drop(page);
        drop(other);
        assert!(
            reg.layer(&key, 1).is_none(),
            "a dropped page is not handed back"
        );
        reg.insert(PieceKey::of(b"another piece"), 1, &[]);
        assert_eq!(reg.len(), 1, "the dead entry is swept on the next insert");
    }
}
