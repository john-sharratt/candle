//! Each sequence's pool: the pieces it holds as index rows with no K/V behind
//! them in its slot, and their resident pages at the sampled layers
//! (`docs/hit_driven_projection.md` §7.2).
//!
//! A pool holds its pages strongly; the [`PageRegistry`] holds them weakly. So
//! a reading two sequences pool is placed once — the second finds it in the
//! registry — and its rows stay resident while any pool or slot holds them,
//! going back to the index tenant with the last.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use candle::Result;

use super::resident_page::ResidentPage;
use crate::models::piece_key::PieceKey;

/// One sequence's pool: each pooled piece's resident pages.
type Pool = HashMap<PieceKey, Vec<Arc<ResidentPage>>>;

/// Every sequence's pool.
#[derive(Debug, Default)]
pub struct SequencePools {
    map: RwLock<HashMap<usize, Pool>>,
}

fn poisoned() -> candle::Error {
    candle::Error::Msg("qwen4exp: pool lock poisoned".into())
}

impl SequencePools {
    /// Hold `pages` as `key`'s in `seq`'s pool, replacing what it held of it.
    pub fn hold(&self, seq: usize, key: PieceKey, pages: Vec<Arc<ResidentPage>>) -> Result<()> {
        self.map
            .write()
            .map_err(|_| poisoned())?
            .entry(seq)
            .or_default()
            .insert(key, pages);
        Ok(())
    }

    /// Let go of `seq`'s pooled pieces not in `keep`.
    ///
    /// By set, not by scanning `keep` for each held piece: a pool holds a key
    /// per page of every turn it pools, and a refresh runs at every
    /// reprojection, so the scan was the pool's size squared on the scheduler
    /// thread each time.
    pub fn retain(&self, seq: usize, keep: &[PieceKey]) -> Result<()> {
        let keep: HashSet<&PieceKey> = keep.iter().collect();
        let mut map = self.map.write().map_err(|_| poisoned())?;
        if let Some(p) = map.get_mut(&seq) {
            p.retain(|k, _| keep.contains(k));
        }
        Ok(())
    }

    /// The pieces `seq`'s pool holds.
    pub fn keys(&self, seq: usize) -> Vec<PieceKey> {
        self.map
            .read()
            .map(|m| {
                m.get(&seq)
                    .map(|p| p.keys().copied().collect())
                    .unwrap_or_default()
            })
            .unwrap_or_default()
    }

    /// Let go of `seq`'s whole pool: the sequence is released.
    pub fn release(&self, seq: usize) {
        if let Ok(mut m) = self.map.write() {
            m.remove(&seq);
        }
    }
}

#[cfg(test)]
mod tests {
    use candle::{DType, Device, Tensor};

    use super::*;
    use crate::models::qwen4exp::index_keys::PAGE_BLOCKS;
    use crate::models::qwen4exp::resident_page::PageRegistry;

    /// A page of one chunk, as a placement would make it.
    fn page() -> Arc<ResidentPage> {
        let chunk = Tensor::zeros((PAGE_BLOCKS, 4), DType::F32, &Device::Cpu).unwrap();
        ResidentPage::from_key_pages(vec![chunk], 3, 1, 4).unwrap()
    }

    /// The sampled layers' pages of `key`, found in the registry where they
    /// stand and placed where they do not — as `resident_pages` does.
    fn pages_of(reg: &PageRegistry, key: &PieceKey, sampled: &[usize]) -> Vec<Arc<ResidentPage>> {
        let mut out = Vec::new();
        let mut placed = Vec::new();
        for &kv in sampled {
            match reg.layer(key, kv) {
                Some(p) => out.push(p),
                None => {
                    let p = page();
                    placed.push((kv, Arc::clone(&p)));
                    out.push(p);
                }
            }
        }
        reg.insert(*key, 4, &placed);
        out
    }

    /// **Two sequences pooling one reading share its pages; the pages outlive
    /// the first release and go with the last** (§13.2).
    #[test]
    fn two_pools_share_a_reading_and_the_last_release_frees_it() -> Result<()> {
        let reg = PageRegistry::default();
        let pools = SequencePools::default();
        let reading = PieceKey::of(b"a file's reading");
        let sampled = [1, 3];

        let first = pages_of(&reg, &reading, &sampled);
        pools.hold(7, reading, first.clone())?;
        let second = pages_of(&reg, &reading, &sampled);
        pools.hold(9, reading, second.clone())?;
        for (a, b) in first.iter().zip(&second) {
            assert!(Arc::ptr_eq(a, b), "the second pool places nothing again");
        }
        drop((first, second));
        assert_eq!(pools.keys(7), vec![reading]);
        assert_eq!(pools.keys(9), vec![reading]);

        // The first sequence is released: the second still holds the rows.
        pools.release(7);
        assert!(pools.keys(7).is_empty());
        for &kv in &sampled {
            assert!(
                reg.layer(&reading, kv).is_some(),
                "layer {kv} outlives the first release"
            );
        }

        // The second pool drops the reading: its rows go back.
        pools.retain(9, &[])?;
        for &kv in &sampled {
            assert!(
                reg.layer(&reading, kv).is_none(),
                "layer {kv} goes with the last"
            );
        }
        Ok(())
    }

    /// A pool keeps what `retain` names and lets the rest go, leaving other
    /// sequences' pools as they were.
    #[test]
    fn retain_keeps_only_what_it_names() -> Result<()> {
        let pools = SequencePools::default();
        let a = PieceKey::of(b"a");
        let b = PieceKey::of(b"b");
        pools.hold(1, a, vec![page()])?;
        pools.hold(1, b, vec![page()])?;
        pools.hold(2, a, vec![page()])?;
        pools.retain(1, &[b])?;
        assert_eq!(pools.keys(1), vec![b]);
        assert_eq!(pools.keys(2), vec![a]);
        assert!(pools.keys(3).is_empty(), "a sequence that pooled nothing");
        Ok(())
    }
}
