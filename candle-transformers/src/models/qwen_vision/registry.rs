//! Rows waiting to be read, named by virtual token ids.
//!
//! An image in a prompt is recorded everywhere as its `<|image_pad|>` ids. Only
//! where the scheduler turns a prompt into the engine's input does each pad
//! become a **virtual id**, `VIRTUAL_BASE + base + i`: the `i`-th row of the
//! image registered at `base`. The engine reads its wave's ids once anyway; one
//! past the vocabulary names a row here, which the wave uploads beside the
//! embedding table and the gather reads as a second table. Nothing recorded
//! ever holds a virtual id, and the rows themselves stay on the host until a
//! wave wants them.
//!
//! A range is held by its [`ImageHandle`] and returned when the last clone of
//! the handle drops: a prefill that holds the handle can always read its rows.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::geometry::Grid;

/// The first virtual id. Far above any vocabulary, so a real id is never read
/// as one, and below `u32::MAX` by half the id space, which is the room ranges
/// are handed out from.
pub const VIRTUAL_BASE: u32 = 0x8000_0000;

/// The id space ranges are cut from: everything from [`VIRTUAL_BASE`] up.
const SPACE: u64 = (u32::MAX - VIRTUAL_BASE) as u64 + 1;

/// One image's rows: `grid.tokens()` rows of `width` F32 values, in raster
/// order over the merged grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageRows {
    pub grid: Grid,
    pub width: usize,
    pub values: Vec<f32>,
}

impl ImageRows {
    /// Rows held.
    pub fn len(&self) -> usize {
        self.grid.tokens()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `i`'s values.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.width..(i + 1) * self.width]
    }
}

#[derive(Default)]
struct Registry {
    /// Ranges held, by their base offset from [`VIRTUAL_BASE`].
    held: BTreeMap<u64, (u64, Arc<ImageRows>)>,
    /// Where the next range is cut from.
    next: u64,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Registry::default()))
}

impl Registry {
    /// A free range of `len`, cut from `next` onward, wrapping once at the end
    /// of the space. `None` when no stretch that long is free.
    fn cut(&mut self, len: u64) -> Option<u64> {
        let mut at = self.next;
        for _ in 0..2 {
            loop {
                if at + len > SPACE {
                    break;
                }
                // The held range starting at or before `at + len`, if it
                // overlaps `at..at + len`, pushes the cut past it.
                let clash = self
                    .held
                    .range(..at + len)
                    .next_back()
                    .filter(|(b, (l, _))| **b + *l > at)
                    .map(|(b, (l, _))| b + l);
                match clash {
                    Some(end) => at = end,
                    None => return Some(at),
                }
            }
            at = 0;
        }
        None
    }
}

/// A registered image's id range. Dropping the last clone returns it.
#[derive(Debug)]
pub struct ImageHandle {
    base: u64,
    rows: Arc<ImageRows>,
}

impl ImageHandle {
    /// Hold `rows` under a fresh id range.
    pub fn register(rows: ImageRows) -> Result<Arc<Self>, String> {
        let rows = Arc::new(rows);
        let len = rows.len() as u64;
        if len == 0 {
            return Err("an image with no rows has no ids to hold".into());
        }
        let mut r = registry().lock().unwrap();
        let base = r
            .cut(len)
            .ok_or_else(|| format!("no free range of {len} virtual ids"))?;
        r.held.insert(base, (len, rows.clone()));
        r.next = base + len;
        Ok(Arc::new(Self { base, rows }))
    }

    /// The virtual id of row `i`.
    pub fn id(&self, i: usize) -> u32 {
        debug_assert!(i < self.rows.len());
        VIRTUAL_BASE + (self.base + i as u64) as u32
    }

    /// The rows held.
    pub fn rows(&self) -> &Arc<ImageRows> {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl Drop for ImageHandle {
    fn drop(&mut self) {
        if let Ok(mut r) = registry().lock() {
            r.held.remove(&self.base);
        }
    }
}

/// Whether `id` is virtual.
pub fn is_virtual(id: u32) -> bool {
    id >= VIRTUAL_BASE
}

/// The rows and the row index a virtual id names, while its handle is held.
pub fn lookup(id: u32) -> Option<(Arc<ImageRows>, usize)> {
    if !is_virtual(id) {
        return None;
    }
    let off = (id - VIRTUAL_BASE) as u64;
    let r = registry().lock().ok()?;
    let (base, (len, rows)) = r.held.range(..=off).next_back()?;
    (off < base + len).then(|| (rows.clone(), (off - base) as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen_vision::geometry::FACTOR;

    fn rows(merged_rows: usize, merged_cols: usize, seed: f32) -> ImageRows {
        let grid = Grid {
            height: merged_rows * FACTOR,
            width: merged_cols * FACTOR,
        };
        let n = grid.tokens();
        ImageRows {
            grid,
            width: 3,
            values: (0..n * 3).map(|i| seed + i as f32).collect(),
        }
    }

    /// A registered image's ids name its rows, and stop naming them once the
    /// handle drops.
    #[test]
    fn ids_name_rows_while_the_handle_is_held() {
        let h = ImageHandle::register(rows(1, 2, 100.0)).unwrap();
        assert_eq!(h.len(), 2);
        let (r, i) = lookup(h.id(1)).unwrap();
        assert_eq!(i, 1);
        assert_eq!(r.row(1), &[103.0, 104.0, 105.0]);
        assert!(is_virtual(h.id(0)));
        let id = h.id(0);
        drop(h);
        assert!(lookup(id).is_none());
    }

    /// Two live images take disjoint ranges.
    #[test]
    fn live_images_never_share_ids() {
        let a = ImageHandle::register(rows(2, 2, 0.0)).unwrap();
        let b = ImageHandle::register(rows(1, 3, 50.0)).unwrap();
        let ids_a: Vec<u32> = (0..a.len()).map(|i| a.id(i)).collect();
        for i in 0..b.len() {
            assert!(!ids_a.contains(&b.id(i)));
            assert_eq!(lookup(b.id(i)).unwrap().0.values[0], 50.0);
        }
    }

    /// A real id is never virtual.
    #[test]
    fn a_vocabulary_id_is_not_virtual() {
        assert!(!is_virtual(248_056));
        assert!(lookup(248_056).is_none());
    }

    /// The cut skips held ranges and wraps once at the end of the space.
    #[test]
    fn a_cut_skips_held_ranges_and_wraps() {
        let dummy = Arc::new(rows(1, 1, 0.0));
        let mut r = Registry::default();
        r.held.insert(0, (10, dummy.clone()));
        r.held.insert(20, (5, dummy.clone()));
        r.next = 5;
        assert_eq!(r.cut(4), Some(10));
        assert_eq!(r.cut(12), Some(25));
        r.next = SPACE - 3;
        assert_eq!(r.cut(4), Some(10), "wrapped past the end");
    }
}
