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
use std::sync::{Arc, Mutex, Weak};

use candle::cuda_backend::cudarc::driver::result::memcpy_htod_async;
use candle::{Device, Result, Tensor};
use candle_kernels::simple::qsa_page_place::{run_qsa_page_place, PLACE_JOB_WORDS};
use candle_nn::kv_cache::{relocate_tensor, ArenaSlot};

use super::index_keys::{alloc_buffers, PAGE_BLOCKS};
use super::indexer::{i64_ptr, tensor_ptr};
use super::place::PLACE_TILE_R;
use crate::models::piece_key::PieceKey;
use crate::models::wave_buffers::wave_from_vec_ticketed;

/// How a resident page's rows sit in each of its chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkLayout {
    /// `[d/4, rows, 4]` per chunk: channel-blocked, so a warp scoring
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
    pub ptr: u64,
    pub rows: usize,
    /// Float4 groups between channel group `c` and `c + 1` of one row.
    pub group_stride: i64,
    /// Float4 groups between row `j` and `j + 1` of one channel group.
    pub row_stride: i64,
}

/// One sealed piece's rows for one attention layer, resident on the span.
#[derive(Debug)]
pub struct ResidentPage {
    /// One `[PAGE_BLOCKS, d]` F32 view of a slot per chunk; chunk `c` holds rows
    /// `[c·PAGE_BLOCKS, (c+1)·PAGE_BLOCKS)`. Behind a lock because a compaction
    /// moves them between forwards while every holder keeps the same `Arc`.
    chunks: Mutex<Vec<Tensor>>,
    rows: usize,
    /// Tokens the last row covers, `1..=ratio`.
    last_cells: usize,
    d: usize,
    layout: ChunkLayout,
}

/// Rows of chunk `c` of a page of `rows` rows.
fn chunk_rows(rows: usize, c: usize) -> usize {
    (rows - c * PAGE_BLOCKS).min(PAGE_BLOCKS)
}

impl ResidentPage {
    /// Pages made from host rows — one `(rows, last_cells)` page per entry of
    /// `pages`, each `[rows, d]` row-major F32 — placed in ONE launch.
    ///
    /// The rows are uploaded into staging slots of the same tenant and placed from
    /// there into the page's own slots, so nothing is taken from the CUDA pool but
    /// the few words of the job table; the staging goes back when this returns,
    /// behind the launch on the same stream.
    ///
    /// **Between forwards.** A slot claim that needs a new arena takes the arena
    /// window.
    pub fn place_host(
        pages: &[(&[f32], usize)],
        d: usize,
        device: &Device,
    ) -> Result<Vec<Arc<Self>>> {
        let Device::Cuda(cuda) = device else {
            candle::bail!("qsa resident page: placed on a CUDA device");
        };
        if !d.is_multiple_of(4) {
            candle::bail!(
                "qsa resident page: head_dim {d} is not a multiple of four — the scorer's \
                 channel-blocked layout is built in float4 groups"
            );
        }
        // Placement allocates and uploads: it runs eagerly for its whole extent,
        // behind whatever this thread has recorded.
        let _eager = cuda.pause_capture()?;
        let stream = cuda.compute_stream();
        let mut out = Vec::with_capacity(pages.len());
        let mut staging: Vec<Tensor> = Vec::new();
        let mut jobs: Vec<i64> = Vec::new();
        let mut max_rows = 0usize;
        for &(host, last_cells) in pages {
            if !host.len().is_multiple_of(d) {
                candle::bail!(
                    "qsa resident page: {} values is not a whole number of {d}-wide rows",
                    host.len()
                );
            }
            let rows = host.len() / d;
            let n = rows.div_ceil(PAGE_BLOCKS);
            let chunks = alloc_buffers(device, PAGE_BLOCKS, d, n)?;
            let stage = alloc_buffers(device, PAGE_BLOCKS, d, n)?;
            for c in 0..n {
                let first = c * PAGE_BLOCKS;
                let used = chunk_rows(rows, c);
                let src = &host[first * d..(first + used) * d];
                // SAFETY: the staging slot holds `PAGE_BLOCKS · d` floats and `src`
                // is at most that; the copy is ordered on the stream the placement
                // below reads it from. From pageable memory the driver has staged
                // the bytes before this returns, so `host` may go.
                unsafe { memcpy_htod_async(tensor_ptr(&stage[c])?, src, stream.cu_stream()) }
                    .map_err(|e| candle::Error::Msg(format!("qsa resident page upload: {e}")))?;
                jobs.push(tensor_ptr(&stage[c])? as i64);
                jobs.push(tensor_ptr(&chunks[c])? as i64);
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
            }));
        }
        let n_jobs = jobs.len() / PLACE_JOB_WORDS;
        if n_jobs > 0 && max_rows > 0 {
            let table = wave_from_vec_ticketed(jobs, (n_jobs * PLACE_JOB_WORDS,), device, None)?;
            candle::set_kernel_breadcrumb("run_qsa_page_place", file!(), line!());
            // SAFETY: every job names a live staging slot and a live chunk slot of
            // `used` rows, both held until after this launch is queued.
            unsafe {
                run_qsa_page_place(
                    i64_ptr(&table)? as *const i64,
                    d as i32,
                    n_jobs as i32,
                    max_rows as i32,
                    PLACE_TILE_R as i32,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        // The staging slots go back here, behind the launch that reads them: a
        // slot's next holder works on the same stream.
        drop(staging);
        Ok(out)
    }

    /// A page over a live tail's key pages, taken as they stand — `chunks` each a
    /// `[PAGE_BLOCKS, d]` slot view holding rows row-major.
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
        }))
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
                    ChunkLayout::Blocked => (rows as i64, 1),
                    ChunkLayout::RowMajor => (1, (self.d / 4) as i64),
                };
                Ok(ChunkDesc {
                    ptr: tensor_ptr(t)?,
                    rows,
                    group_stride,
                    row_stride,
                })
            })
            .collect()
    }

    /// The page's rows, `[rows, d]` row-major, read back to the host — for a seal
    /// that hands the page on as a record.
    pub fn host_rows(&self) -> Result<Vec<f32>> {
        let chunks = self.chunks.lock().unwrap_or_else(|e| e.into_inner());
        let d = self.d;
        let mut out = vec![0f32; self.rows * d];
        for (c, t) in chunks.iter().enumerate() {
            let rows = chunk_rows(self.rows, c);
            let first = c * PAGE_BLOCKS;
            let vals = t.flatten_all()?.narrow(0, 0, rows * d)?.to_vec1::<f32>()?;
            let dst = &mut out[first * d..(first + rows) * d];
            match self.layout {
                ChunkLayout::RowMajor => dst.copy_from_slice(&vals),
                // `[d/4, rows, 4]`: group `g` of row `j` sits at `(g·rows + j)·4`.
                ChunkLayout::Blocked => {
                    for g in 0..d / 4 {
                        for j in 0..rows {
                            let s = (g * rows + j) * 4;
                            let o = j * d + g * 4;
                            dst[o..o + 4].copy_from_slice(&vals[s..s + 4]);
                        }
                    }
                }
            }
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
/// The registry holds no page alive: its entries are weak, so a piece's rows go
/// back to the tenant when the last cache holding them drops, and the next push
/// of that piece places it afresh. A dead entry is swept on the next insert.
#[derive(Debug, Default)]
pub struct PageRegistry {
    /// One page per KV layer; `None` for a layer that indexes nothing.
    map: Mutex<HashMap<PieceKey, Vec<Option<Weak<ResidentPage>>>>>,
}

impl PageRegistry {
    /// The resident pages for `key`, when every layer's page is still held.
    pub fn get(&self, key: &PieceKey) -> Option<Vec<Option<Arc<ResidentPage>>>> {
        let map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        map.get(key)?
            .iter()
            .map(|w| match w {
                Some(w) => w.upgrade().map(Some),
                None => Some(None),
            })
            .collect()
    }

    /// Record `layers` as the resident pages of `key`, sweeping entries whose
    /// pages have all dropped.
    pub fn insert(&self, key: PieceKey, layers: &[Option<Arc<ResidentPage>>]) {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, layers| {
            layers
                .iter()
                .any(|w| w.as_ref().is_some_and(|w| w.strong_count() > 0))
        });
        map.insert(
            key,
            layers
                .iter()
                .map(|p| p.as_ref().map(Arc::downgrade))
                .collect(),
        );
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

    fn ramp(n: usize, from: f32) -> Vec<f32> {
        (0..n).map(|i| from + i as f32).collect()
    }

    /// Pages placed in one launch hold exactly the rows they were given, each
    /// chunk in the channel-blocked layout — `[d/4, rows, 4]` over the chunk's
    /// own row count — and describe themselves to the scorer by that layout.
    #[test]
    fn placed_pages_hold_their_rows_channel_blocked() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let d = 8usize;
        let long = ramp((PAGE_BLOCKS + 3) * d, 0.0);
        let short = ramp(2 * d, 10_000.0);
        let pages = ResidentPage::place_host(&[(&long, 3), (&short, 4)], d, &device)?;
        assert_eq!(pages.len(), 2);

        assert_eq!(pages[0].rows(), PAGE_BLOCKS + 3);
        assert_eq!(pages[0].last_cells(), 3);
        assert_eq!(pages[0].host_rows()?, long);
        assert_eq!(pages[1].host_rows()?, short);

        // The short page's one chunk, raw: group 0 of rows 0 and 1, then group 1.
        let raw = pages[1].chunks.lock().unwrap()[0]
            .flatten_all()?
            .narrow(0, 0, 2 * d)?
            .to_vec1::<f32>()?;
        assert_eq!(
            raw,
            vec![
                10_000.0, 10_001.0, 10_002.0, 10_003.0, 10_008.0, 10_009.0, 10_010.0, 10_011.0,
                10_004.0, 10_005.0, 10_006.0, 10_007.0, 10_012.0, 10_013.0, 10_014.0, 10_015.0,
            ]
        );

        let desc = pages[0].descriptors()?;
        let shape: Vec<_> = desc
            .iter()
            .map(|c| (c.rows, c.group_stride, c.row_stride))
            .collect();
        assert_eq!(shape, vec![(PAGE_BLOCKS, PAGE_BLOCKS as i64, 1), (3, 3, 1)]);
        Ok(())
    }

    /// A closed tail's key pages are taken as they stand: row-major, read at a
    /// row stride of `d/4` float4 groups.
    #[test]
    fn key_pages_are_read_row_major_in_place() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let d = 8usize;
        let rows = PAGE_BLOCKS + 1;
        let vals = ramp(2 * PAGE_BLOCKS * d, 0.0);
        let chunks = vec![
            Tensor::from_vec(vals[..PAGE_BLOCKS * d].to_vec(), (PAGE_BLOCKS, d), &device)?,
            Tensor::from_vec(vals[PAGE_BLOCKS * d..].to_vec(), (PAGE_BLOCKS, d), &device)?,
        ];
        let page = ResidentPage::from_key_pages(chunks, rows, 2, d)?;
        assert_eq!(page.layout(), ChunkLayout::RowMajor);
        assert_eq!(page.host_rows()?, vals[..rows * d].to_vec());
        let shape: Vec<_> = page
            .descriptors()?
            .iter()
            .map(|c| (c.rows, c.group_stride, c.row_stride))
            .collect();
        assert_eq!(shape, vec![(PAGE_BLOCKS, 1, 2), (1, 1, 2)]);
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
        reg.insert(key, &[None, Some(Arc::clone(&page))]);

        let got = reg.get(&key).expect("held pages are found");
        assert!(got[0].is_none());
        assert!(Arc::ptr_eq(got[1].as_ref().unwrap(), &page));
        drop(got);

        drop(page);
        assert!(reg.get(&key).is_none(), "a dropped page is not handed back");
        reg.insert(PieceKey::of(b"another piece"), &[None]);
        assert_eq!(reg.len(), 1, "the dead entry is swept on the next insert");
    }
}
