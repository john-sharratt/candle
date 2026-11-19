//! Many turns made resident and pinned for one scan at once — what a scan's
//! index build admits.
//!
//! One turn at a time, a miss decoded its stored window a signature a token,
//! transposed it, claimed its slots and copied its pages, all under the
//! residency lock: 177,647 turns took 4.4 s that way, with nothing resident
//! after a restart (BlackWorld, 2026-11-19), and a conversation's first
//! reprojection waited on it. Here the turns already resident are pinned under
//! one lock; the misses are claimed and transposed off it, in parallel, from
//! their stored words into one buffer per batch; and each batch is copied in
//! its slots' contiguous runs and installed under one lock more.
//!
//! What an upload holds is decided exactly as one turn's upload decides it
//! ([`Held::widened`]). A turn another thread changed between the plan and the
//! install is admitted the single-turn way instead, against what it holds now.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use candle::Result;
use candle_nn::kv_cache::{claim_arena_slots, slot_stride, ArenaSlot, SlotTenant};
use rayon::prelude::*;

use crate::persistence::streams::StreamId;

use super::dedup::{Keep, KeepMask};
use super::groups::GroupSet;
use super::holder::Holder;
use super::page_io;
use super::pages::{pages_for, transpose_into};
use super::turn_sigs::TurnSigs;
use super::{Ensured, GalleryArena, Held, PageRun, ResidentTurn};

/// Host words of pages one batch transposes before it is copied: 128 MiB, so
/// a corpus of millions of tokens is admitted without holding it on the host
/// whole.
const BATCH_WORDS: usize = 16 << 20;

/// One turn a scan reads: its window, its content, and what the read weighs.
#[derive(Clone, Copy)]
pub(super) struct Wanted<'a> {
    pub sid: StreamId,
    pub sigs: TurnSigs<'a>,
    pub fingerprint: u64,
    pub holder: Holder,
    pub groups: GroupSet,
    pub keep: Keep,
}

/// A miss, planned: which wanted turn, and what its upload holds.
struct Planned {
    at: usize,
    held: Held,
}

/// A planned miss with its mask and its pages' place in the batch's buffer.
struct Staged {
    at: usize,
    held: Held,
    mask: Arc<KeepMask>,
    pages: usize,
}

impl GalleryArena {
    /// Make every one of `wanted` resident and **pin** it for the duration of a
    /// scan, answering one [`Ensured`] each, in order. Every pin is balanced by
    /// an [`unpin_all`](Self::unpin_all). **Makes no room**: the index builder
    /// runs the keep eviction once, before. On an error every pin this call
    /// took is released before it is returned.
    pub(super) fn scan_ensure_all(&self, wanted: &[Wanted<'_>]) -> Result<Vec<Ensured>> {
        let mut out: Vec<Option<Ensured>> = (0..wanted.len()).map(|_| None).collect();
        let book = self.dedup.book();
        let book_id = book.as_deref().map(|b| b.id());
        let mut misses: Vec<Planned> = Vec::new();
        {
            let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
            for (at, w) in wanted.iter().enumerate() {
                let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
                match res.get_mut(&w.sid) {
                    Some(rt) if rt.serves(w.fingerprint, w.groups, w.keep) => {
                        out[at] = Some(rt.hit(lru, w.holder, true));
                    }
                    old => misses.push(Planned {
                        at,
                        held: Held::widened(
                            old.as_deref(),
                            w.fingerprint,
                            w.holder,
                            w.groups,
                            w.keep,
                            book.is_some(),
                        ),
                    }),
                }
            }
        }
        let admitted = self.admit_misses(wanted, &misses, book_id, &mut out);
        let pinned: Vec<StreamId> = wanted
            .iter()
            .zip(&out)
            .filter(|(_, e)| e.is_some())
            .map(|(w, _)| w.sid)
            .collect();
        if let Err(e) = admitted {
            self.unpin_all(&pinned);
            return Err(e);
        }
        Ok(out
            .into_iter()
            .map(|e| e.expect("every wanted turn was admitted"))
            .collect())
    }

    /// Upload `misses` batch by batch, filling their places in `out`.
    fn admit_misses(
        &self,
        wanted: &[Wanted<'_>],
        misses: &[Planned],
        book_id: Option<u64>,
        out: &mut [Option<Ensured>],
    ) -> Result<()> {
        let book = self.dedup.book();
        // Every miss's mask, in parallel: a keep by code reads the turn's
        // words through the codebook, and most were computed already
        // (`prepare_keeps`).
        let masks: Vec<Arc<KeepMask>> = misses
            .par_iter()
            .map(|m| {
                let w = &wanted[m.at];
                self.dedup
                    .mask(w.sid, w.fingerprint, w.sigs, m.held.keep, book.as_deref())
            })
            .collect();
        let mut batch: Vec<Staged> = Vec::new();
        let mut words = 0usize;
        for (m, mask) in misses.iter().zip(masks) {
            let pages = pages_for(mask.kept());
            let turn_words = pages * self.stride_words(m.held.groups);
            if !batch.is_empty() && words + turn_words > BATCH_WORDS {
                self.admit_batch(wanted, std::mem::take(&mut batch), book_id, out)?;
                words = 0;
            }
            words += turn_words;
            batch.push(Staged {
                at: m.at,
                held: m.held,
                mask,
                pages,
            });
        }
        if !batch.is_empty() {
            self.admit_batch(wanted, batch, book_id, out)?;
        }
        Ok(())
    }

    /// Words a slot holding pages of `groups` spans.
    fn stride_words(&self, groups: GroupSet) -> usize {
        slot_stride(self.page_bytes_for(groups) as usize) / std::mem::size_of::<u64>()
    }

    /// Claim, transpose, copy and install one batch of misses. A batch is
    /// grouped by the fold groups its turns' pages hold, each group one claim
    /// and one buffer, the slot stride being the page's.
    fn admit_batch(
        &self,
        wanted: &[Wanted<'_>],
        batch: Vec<Staged>,
        book_id: Option<u64>,
        out: &mut [Option<Ensured>],
    ) -> Result<()> {
        let mut classes: Vec<(GroupSet, Vec<Staged>)> = Vec::new();
        for s in batch {
            match classes.iter_mut().find(|(g, _)| *g == s.held.groups) {
                Some((_, turns)) => turns.push(s),
                None => classes.push((s.held.groups, vec![s])),
            }
        }
        for (groups, turns) in classes {
            self.admit_class(wanted, groups, turns, book_id, out)?;
        }
        Ok(())
    }

    /// One fold-group class of a batch: one claim, one buffer transposed in
    /// parallel, one copy per contiguous run of slots, and the installs under
    /// one lock.
    fn admit_class(
        &self,
        wanted: &[Wanted<'_>],
        groups: GroupSet,
        turns: Vec<Staged>,
        book_id: Option<u64>,
        out: &mut [Option<Ensured>],
    ) -> Result<()> {
        let total: usize = turns.iter().map(|t| t.pages).sum();
        let stride_words = self.stride_words(groups);
        let slots = if total == 0 {
            Vec::new()
        } else {
            claim_arena_slots(
                &self.device,
                SlotTenant::Gallery,
                self.page_bytes_for(groups) as usize,
                total,
            )?
        };
        let mut host = vec![0u64; total * stride_words];
        {
            let mut parts: Vec<(&Staged, &mut [u64])> = Vec::with_capacity(turns.len());
            let mut rest: &mut [u64] = &mut host;
            for t in &turns {
                let (part, tail) = rest.split_at_mut(t.pages * stride_words);
                parts.push((t, part));
                rest = tail;
            }
            let (gw, wpt) = (self.gw, self.wpt);
            parts.into_par_iter().for_each(|(t, part)| {
                transpose_turn(
                    wanted[t.at].sigs,
                    &t.mask,
                    gw,
                    wpt,
                    groups,
                    stride_words,
                    part,
                );
            });
        }
        let mut res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        if !slots.is_empty() {
            page_io::write_pages(&self.device, &slots, &host)?;
        }
        let current_book = self.dedup.book().as_deref().map(|b| b.id());
        let mut slots = slots.into_iter();
        for t in turns {
            let run: Vec<ArenaSlot> = slots.by_ref().take(t.pages).collect();
            let w = &wanted[t.at];
            let lru = self.lru_clock.fetch_add(1, Ordering::Relaxed);
            if let Some(rt) = res.get_mut(&w.sid) {
                if rt.serves(w.fingerprint, w.groups, w.keep) {
                    // Made resident by another thread since the plan: its pages
                    // serve, and the ones copied here go back unread.
                    out[t.at] = Some(rt.hit(lru, w.holder, true));
                    continue;
                }
            }
            let now = Held::widened(
                res.get(&w.sid),
                w.fingerprint,
                w.holder,
                w.groups,
                w.keep,
                current_book.is_some(),
            );
            if now != t.held || current_book != book_id {
                drop(run);
                let (addrs, run_id, groups, mask) = self.replace_locked(
                    &mut res,
                    w.sid,
                    w.sigs,
                    w.fingerprint,
                    true,
                    lru,
                    w.holder,
                    w.groups,
                    w.keep,
                )?;
                out[t.at] = Some(Ensured {
                    addrs,
                    run_id,
                    groups,
                    mask,
                    uploaded: true,
                });
                continue;
            }
            self.drop_replaced(&mut res, w.sid);
            let run = PageRun { slots: run };
            let addrs = run.addrs();
            let run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
            res.insert(
                w.sid,
                ResidentTurn {
                    fingerprint: w.fingerprint,
                    run,
                    run_id,
                    mask: Arc::clone(&t.mask),
                    keep: t.held.keep,
                    lru,
                    used: Instant::now(),
                    interval: None,
                    pinned: 1,
                    holder: t.held.holder,
                    groups: t.held.groups,
                },
            );
            out[t.at] = Some(Ensured {
                addrs,
                run_id,
                groups: t.held.groups,
                mask: t.mask,
                uploaded: true,
            });
        }
        Ok(())
    }
}

/// One turn's kept tokens into its slice of the batch's buffer: a stored
/// window of the arena's width straight from its words, anything else through
/// its decoded signatures.
fn transpose_turn(
    sigs: TurnSigs<'_>,
    mask: &KeepMask,
    gw: usize,
    wpt: usize,
    groups: GroupSet,
    stride_words: usize,
    out: &mut [u64],
) {
    match sigs {
        TurnSigs::Encoded(w) if w.width() == wpt => {
            let words = w.words();
            transpose_into(words.chunks_exact(wpt), mask, gw, groups, stride_words, out);
        }
        _ => sigs.with_decoded(|decoded| {
            transpose_into(
                decoded.iter().map(|s| s.words.as_slice()),
                mask,
                gw,
                groups,
                stride_words,
                out,
            );
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::pages::{page_words, transpose_to_pages};
    use super::super::tests::gallery_pool;
    use super::*;
    use crate::persistence::content_hash::turn_stream_id;
    use crate::provenance::{encode_wide_sigs, SigWindow, WideQSig};
    use candle::Device;

    fn sig(fill: u64) -> WideQSig {
        WideQSig {
            n_heads: 12,
            words: (0..24).map(|w| fill + w as u64).collect(),
        }
    }

    fn sid(n: u32) -> StreamId {
        turn_stream_id(7, n)
    }

    /// **One call admits a mixed set, every page holding exactly its turn's
    /// transpose.** Turn 0 is resident already and found, not uploaded; turn 1
    /// is decoded signatures of every group; turn 2 a stored window holding
    /// group 2 only — another slot stride, so another claim. Every turn comes
    /// back pinned, in order, with the run it holds, and its pages read back
    /// as raw words equal to the single-turn transpose.
    #[test]
    fn a_batch_admits_hits_and_misses_of_either_form() {
        let Ok(device) = Device::new_cuda(0) else {
            return;
        };
        let _pool = gallery_pool();
        let arena = GalleryArena::new(&device, 24, 3).unwrap();
        let all = GroupSet::all(3);
        let two = GroupSet::default().with(2);
        let t0: Vec<WideQSig> = (0..10).map(|t| sig(1_000 + 100 * t)).collect();
        let t1: Vec<WideQSig> = (0..70).map(|t| sig(50_000 + 100 * t)).collect();
        let t2: Vec<WideQSig> = (0..33).map(|t| sig(900_000 + 100 * t)).collect();
        let window = SigWindow::new(Arc::from(encode_wide_sigs(&t2))).unwrap();
        arena
            .ensure_resident(sid(0), &t0, 10, Holder::Shared, all, Keep::All)
            .unwrap();
        let want = |sid, sigs, fingerprint, groups| Wanted {
            sid,
            sigs,
            fingerprint,
            holder: Holder::Shared,
            groups,
            keep: Keep::All,
        };
        let wanted = [
            want(sid(0), TurnSigs::from(&t0), 10, all),
            want(sid(1), TurnSigs::from(&t1), 11, all),
            want(sid(2), TurnSigs::from(&window), 12, two),
        ];
        let got = arena.scan_ensure_all(&wanted).unwrap();
        assert_eq!(
            got.iter()
                .map(|e| (e.uploaded, e.addrs.len()))
                .collect::<Vec<_>>(),
            vec![(false, 1), (true, 3), (true, 2)]
        );
        assert_eq!(got[2].groups, two);

        let res = arena.residency.lock().unwrap();
        for ((sigs, groups), (n, e)) in [(&t0, all), (&t1, all), (&t2, two)]
            .into_iter()
            .zip(got.iter().enumerate())
        {
            let rt = res.get(&sid(n as u32)).unwrap();
            assert_eq!(rt.pinned, 1, "turn {n} pinned once");
            assert_eq!((rt.run.addrs(), rt.run_id), (e.addrs.clone(), e.run_id));
            let pw = page_words(8, groups);
            let stride = rt.run.slots[0].stride() / 8;
            let expect = transpose_to_pages(sigs, &KeepMask::all(sigs.len()), 8, groups, stride);
            for (p, slot) in rt.run.slots.iter().enumerate() {
                assert_eq!(
                    page_io::read_page(&device, slot, pw).unwrap(),
                    expect[p * stride..p * stride + pw],
                    "turn {n} page {p}"
                );
            }
        }
        drop(res);
        assert_eq!(
            arena.drop_turns(&[sid(0), sid(1), sid(2)]).0,
            0,
            "pinned turns stay"
        );
        arena.unpin_all(&[sid(0), sid(1), sid(2)]);
        assert_eq!(arena.drop_turns(&[sid(0), sid(1), sid(2)]).0, 3);
    }
}
