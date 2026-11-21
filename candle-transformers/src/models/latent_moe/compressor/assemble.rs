//! Assembling the compressor's groups for a whole fleet of sequences in a fixed
//! number of device operations.
//!
//! The per-sequence assemble costs about fifteen small launches per sequence, and
//! under WDDM submission pressure those launches dominate prefill. The batched
//! form builds every sequence's combined stream back to back, then pulls the group
//! rows, overlap prev rows, retained last-group rows and remainder tails for the
//! whole fleet with a few index selects. Its result is bit-identical per sequence
//! to the per-sequence assemble. The index tensors depend only on the shape of the
//! wave, so they are cached and shared by every layer of that wave.

use super::*;

/// One sequence's input to [`assemble_groups_batched`]: its carried
/// partial-group rows + its new projected rows (all `Arc` clones), and the
/// scalar state the group math needs. Built by
/// [`IncrementalCompressor::assemble_input`].
pub struct SeqAssemble {
    pub(super) kv_pieces: Vec<Tensor>,
    pub(super) sc_pieces: Vec<Tensor>,
    pub(super) l0: usize,
    pub(super) n: usize,
    pub(super) prev_kv: Option<Vec<Tensor>>,
    pub(super) prev_sc: Option<Vec<Tensor>>,
    pub(super) group_idx: usize,
}

/// One sequence's state advance out of [`assemble_groups_batched`], installed
/// by [`IncrementalCompressor::assemble_apply`]. `prev_*` are `None` when the
/// sequence completed no group (the carried prev is kept).
pub struct AssembleUpdate {
    pub(super) kv_rows: Vec<Tensor>,
    pub(super) score_rows: Vec<Tensor>,
    pub(super) prev_kv: Option<Vec<Tensor>>,
    pub(super) prev_sc: Option<Vec<Tensor>>,
    pub(super) group_add: usize,
}

/// The fleet's assemble GEOMETRY — every host-side quantity and device index
/// tensor derivable from `ratio` and the per-seq `(carried, new)` shape alone,
/// independent of the layer's actual data.
///
/// The same wave calls [`assemble_groups_batched`] once per LAYER, and the
/// geometry is layer-invariant by construction: every layer's compressor has
/// consumed the same token stream, so carried-tail lengths and group counts
/// match across the wave's layers. Building the indices per call cost 2
/// host-vec builds and 2 tiny H2D uploads × layers × compressor families; the
/// cache pays that once per wave shape and every other layer-call hits.
pub(super) struct AssembleGeom {
    pub(super) groups: Vec<usize>,
    pub(super) cutoff: Vec<usize>,
    pub(super) n_tot: Vec<usize>,
    pub(super) g_off: Vec<usize>,
    pub(super) total_rows: usize,
    pub(super) total_groups: usize,
    pub(super) tail_off: Vec<usize>,
    /// Combined-stream index of every remainder-tail row (`None`: no tails).
    pub(super) tail_it: Option<Tensor>,
    /// Combined-stream index of every group row (`None`: no completed groups).
    pub(super) gi_t: Option<Tensor>,
    /// Pool index of each group-completing seq's LAST group (its next prev).
    pub(super) li_t: Option<Tensor>,
}

/// Cache key: the template's `ratio` plus every seq's `(carried, new)` — equal
/// keys imply identical geometry. Neither `cd`, `overlap` nor prev-presence
/// appears, because the descriptor-table pool expresses the overlap shift as a
/// narrow offset and an absent prev as an absent segment, so none of the three
/// reaches the index tensors any more.
pub(super) type AssembleGeomKey = (usize, Vec<(usize, usize)>);

/// Process-wide geometry cache, bounded like `CudaDevice::info_table`
/// (cleared wholesale past the cap — entries are a handful of tiny device
/// tensors and rebuilding is one call's work). Single-accelerator assumption,
/// like `cached_sm_count` in candle-core: the cached index tensors live on
/// the one device this fork targets.
pub(super) fn assemble_geom_cache() -> &'static Mutex<HashMap<AssembleGeomKey, Arc<AssembleGeom>>> {
    static CACHE: OnceLock<Mutex<HashMap<AssembleGeomKey, Arc<AssembleGeom>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn build_assemble_geom(
    r: usize,
    seqs: &[(usize, usize)],
    dev: &Device,
) -> Result<AssembleGeom> {
    let n_seq = seqs.len();
    let mut c_off = Vec::with_capacity(n_seq);
    let mut groups = Vec::with_capacity(n_seq);
    let mut cutoff = Vec::with_capacity(n_seq);
    let mut n_tot = Vec::with_capacity(n_seq);
    let mut g_off = Vec::with_capacity(n_seq);
    let mut off = 0usize;
    let mut goff = 0usize;
    for &(l0, n) in seqs {
        let nt = l0 + n;
        let g = nt / r;
        c_off.push(off);
        n_tot.push(nt);
        groups.push(g);
        cutoff.push(g * r);
        g_off.push(goff);
        off += nt;
        goff += g;
    }
    let total_rows = off;
    let total_groups = goff;

    let mut tail_idx: Vec<u32> = Vec::with_capacity(total_rows - total_groups * r);
    let mut tail_off = Vec::with_capacity(n_seq);
    for i in 0..n_seq {
        tail_off.push(tail_idx.len());
        for t in cutoff[i]..n_tot[i] {
            tail_idx.push((c_off[i] + t) as u32);
        }
    }
    let tail_it = if tail_idx.is_empty() {
        None
    } else {
        let n_tail = tail_idx.len();
        Some(Tensor::from_vec(tail_idx, n_tail, dev)?)
    };

    // The prev-group shift needs no geometry: a group's previous half is the
    // group block read one group earlier, which the pool descriptor expresses as
    // a narrow offset, and a group 0 with no carried prev simply omits the
    // segment. That is why there is no prev-source index or pad here.
    let (gi_t, li_t) = if total_groups == 0 {
        (None, None)
    } else {
        let mut grp_idx: Vec<u32> = Vec::with_capacity(total_groups * r);
        for i in 0..n_seq {
            for t in 0..cutoff[i] {
                grp_idx.push((c_off[i] + t) as u32);
            }
        }
        let gi_t = Tensor::from_vec(grp_idx, total_groups * r, dev)?;

        let last_idx: Vec<u32> = (0..n_seq)
            .filter(|&i| groups[i] > 0)
            .map(|i| (g_off[i] + groups[i] - 1) as u32)
            .collect();
        let n_active = last_idx.len();
        let li_t = Tensor::from_vec(last_idx, n_active, dev)?;
        (Some(gi_t), Some(li_t))
    };

    Ok(AssembleGeom {
        groups,
        cutoff,
        n_tot,
        g_off,
        total_rows,
        total_groups,
        tail_off,
        tail_it,
        gi_t,
        li_t,
    })
}

/// The whole prompt fleet's compressor ASSEMBLE in ~a dozen device ops —
/// bit-identical per sequence to [`IncrementalCompressor::assemble_groups`]
/// called per seq, whose ~15 small launches × N sequences were HALF the
/// cfg20 prefill wall (15.7s of 29.8s — every enqueue in the per-seq loop
/// stalls under WDDM submission pressure, and the loop multiplies the stall
/// points by the fleet width).
///
/// Construction: one `cat` builds every sequence's `[carried ++ new]`
/// combined stream back-to-back; `index_select`s then pull the group rows,
/// the overlap-prev rows, the retained last-group rows and the remainder
/// tails for ALL sequences at once (pure row copies — byte-identical to the
/// per-seq narrows they replace); the one `broadcast_add(ape)` is
/// row-elementwise. The returned per-seq [`GroupPool`]s are dim-0 NARROW
/// VIEWS of one fleet-wide pool tensor, in sequence order — exactly the
/// concatenation `pool_prefill_across_seqs` builds from per-seq pools, so
/// everything downstream is unchanged. The index tensors + geometry come from
/// the per-shape [`AssembleGeom`] cache — layer-invariant within a wave, so
/// only the wave's first layer-call builds them.
///
/// `template` supplies the layer-shared [`Compressor`] (weights/ape/ratio —
/// identical across the fleet's per-seq states by construction).
pub fn assemble_groups_batched(
    template: &IncrementalCompressor,
    inputs: &[SeqAssemble],
) -> Result<Vec<(Option<GroupPool>, AssembleUpdate)>> {
    let c = &template.c;
    let r = c.ratio;
    let cd = c.cd();
    let dev = c.device().clone();
    let n_seq = inputs.len();

    // Geometry + index tensors from the per-shape cache (layer-invariant
    // within a wave — the key IS the verification that the shape matches).
    let key: AssembleGeomKey = (r, inputs.iter().map(|i| (i.l0, i.n)).collect());
    let geom = {
        let mut cache = assemble_geom_cache().lock().unwrap();
        if let Some(g) = cache.get(&key) {
            g.clone()
        } else {
            if cache.len() >= 4096 {
                cache.clear();
            }
            let g = Arc::new(build_assemble_geom(r, &key.1, &dev)?);
            cache.insert(key, g.clone());
            g
        }
    };
    let AssembleGeom {
        groups,
        n_tot,
        cutoff,
        g_off,
        total_rows,
        total_groups,
        tail_off,
        ..
    } = &*geom;
    let (total_rows, total_groups) = (*total_rows, *total_groups);

    // 1. Combined stream: every seq's [carried ++ new], back-to-back.
    let pieces_kv: Vec<&Tensor> = inputs.iter().flat_map(|i| i.kv_pieces.iter()).collect();
    let pieces_sc: Vec<&Tensor> = inputs.iter().flat_map(|i| i.sc_pieces.iter()).collect();
    let comb_kv = Tensor::cat(&pieces_kv, 0)?; // [total_rows, cd] owned
    let comb_sc = Tensor::cat(&pieces_sc, 0)?;
    debug_assert_eq!(comb_kv.dim(0)?, total_rows);

    // Remainder tails for ALL seqs (also the whole-stream buffer when a seq
    // completes no group): one select, per-seq row views into it.
    let (kv_tails, sc_tails) = match &geom.tail_it {
        None => (None, None),
        Some(it) => (
            Some(comb_kv.index_select(it, 0)?),
            Some(comb_sc.index_select(it, 0)?),
        ),
    };
    let tail_views = |src: &Option<Tensor>, i: usize| -> Result<Vec<Tensor>> {
        let rem = n_tot[i] - cutoff[i];
        let src = match src {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        (0..rem)
            .map(|t| src.narrow(0, tail_off[i] + t, 1))
            .collect()
    };

    if total_groups == 0 {
        // No sequence completes a group: everyone buffers its combined rows.
        let mut out = Vec::with_capacity(n_seq);
        for i in 0..n_seq {
            out.push((
                None,
                AssembleUpdate {
                    kv_rows: tail_views(&kv_tails, i)?,
                    score_rows: tail_views(&sc_tails, i)?,
                    prev_kv: None,
                    prev_sc: None,
                    group_add: 0,
                },
            ));
        }
        return Ok(out);
    }

    // 2. RAW group rows for the whole fleet: [total_groups, r, cd]. `ape` is not
    // applied here — it rides in the pool descriptor, so the retained rows stay
    // raw and match what the streaming path retains.
    let gi_t = geom
        .gi_t
        .as_ref()
        .expect("total_groups > 0 implies group indices");
    let kv_g = comb_kv
        .index_select(gi_t, 0)?
        .reshape((total_groups, r, cd))?;
    let sc_g = comb_sc
        .index_select(gi_t, 0)?
        .reshape((total_groups, r, cd))?;

    // 3. Retained last-group rows (the next call's overlap prev) for every seq
    // that completed a group: one select, then per-seq row views.
    let li_t = geom
        .li_t
        .as_ref()
        .expect("total_groups > 0 implies last-group indices");
    let last_kv = kv_g.index_select(li_t, 0)?; // [n_active, r, cd]
    let last_sc = sc_g.index_select(li_t, 0)?;

    // 4. Per-seq outputs: pool spans + state advance. Each sequence describes
    // its own groups against the shared `kv_g`/`sc_g` block — the prev-group
    // shift is that block read one group earlier, so the pad row, the prev
    // source concatenation and the `pi_t` gather the reference form needed are
    // all gone, and so is the join of the two halves.
    let mut out = Vec::with_capacity(n_seq);
    let mut a_slot = 0usize;
    for i in 0..n_seq {
        let gp = if groups[i] > 0 {
            let positions: Vec<u32> = (0..groups[i])
                .map(|j| ((inputs[i].group_idx + j) * r) as u32)
                .collect();
            let prev = match (&inputs[i].prev_kv, &inputs[i].prev_sc) {
                (Some(pk), Some(ps)) => Some((pk.as_slice(), ps.as_slice())),
                _ => None,
            };
            Some(GroupPool {
                spans: c.group_spans(&kv_g, &sc_g, g_off[i], groups[i], prev)?,
                positions,
            })
        } else {
            None
        };
        let (prev_kv, prev_sc) = if groups[i] > 0 {
            // Views into the small `last_*` select output, which is already off
            // the fleet-wide stream — no copy needed to break a pin.
            let pk = row_views(&last_kv.narrow(0, a_slot, 1)?.reshape((r, cd))?)?;
            let ps = row_views(&last_sc.narrow(0, a_slot, 1)?.reshape((r, cd))?)?;
            a_slot += 1;
            (Some(pk), Some(ps))
        } else {
            (None, None)
        };
        out.push((
            gp,
            AssembleUpdate {
                kv_rows: tail_views(&kv_tails, i)?,
                score_rows: tail_views(&sc_tails, i)?,
                prev_kv,
                prev_sc,
                group_add: groups[i],
            },
        ));
    }
    Ok(out)
}
