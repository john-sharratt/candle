//! `GET /v1/kvmap` — the KV span's arena map, for diagnosing fragmentation.
//!
//! The companion to [`super::memory`]'s class rows, which answer *how much* of
//! each size class is live. That total cannot see fragmentation: "one arena
//! half full" and "two arenas a quarter full each" produce the same class row,
//! and only the second strands ground. Measured on run 39, class 8192 read
//! `592 MiB reserved / 74 MiB live` — which turned out to be **37 arenas at 12%
//! apiece**, and no total could have said so.
//!
//! Fragmentation matters because the weight zone grows into
//! `weight_floor − live_end()` and `live_end` is the *highest* live region. An
//! arena holding one chunk pins its whole region as firmly as a full one, so a
//! class scattered across many arenas holds the boundary open and starves the
//! expert cache — the run-39 chain was sealing → arenas emptied but never
//! wholly → `release_empty_arenas` could take none → the frontier held → the
//! weight side asked for spare ground every forward and was told none existed.
//!
//! Served straight from the gid pool's per-arena atomics with no engine lock,
//! so it can be read while a run is mid-decode. The numbers are a snapshot of a
//! moving target, which is what a fragmentation map is.

use axum::Json;
use serde::Serialize;

/// One arena's occupancy.
#[derive(Serialize)]
pub struct ArenaRow {
    /// Index into the arena registry — the identity a `ChunkGid` encodes.
    pub arena_idx: usize,
    /// Position in **span-address order**: the arena's region index, monotonic
    /// in its base address. Reported because `arena_idx` is registry order and
    /// says nothing about where the arena sits, while the whole reclaim question
    /// — does emptying this return ground? — turns on span position. Reading
    /// fragmentation off `arena_idx` means guessing at the field the policy
    /// actually uses.
    pub rank: usize,
    /// Slot stride in bytes, i.e. which size class this arena belongs to.
    pub slot_bytes: usize,
    /// Slots the arena holds in total.
    pub capacity: usize,
    /// Slots currently occupied. Zero means it is reclaimable right now.
    pub live: usize,
    /// Occupied fraction, 0.0–1.0. The number fragmentation is read off.
    pub occupancy: f32,
    /// Bytes the arena's slab holds that nothing is using.
    pub stranded_bytes: usize,
}

/// One size class's rollup, so the map can be read without summing by hand.
#[derive(Serialize)]
pub struct ClassRow {
    pub slot_bytes: usize,
    pub arenas: usize,
    /// Arenas holding nothing — `release_empty_arenas` takes exactly these.
    pub empty_arenas: usize,
    pub capacity_slots: usize,
    pub live_slots: usize,
    pub occupancy: f32,
    pub stranded_bytes: usize,
    /// Arenas this class's live slots would need if packed perfectly. The gap
    /// between this and `arenas` is what defragmentation could return.
    pub packed_arenas: usize,
}

/// Response body for `GET /v1/kvmap`.
#[derive(Serialize)]
pub struct KvMapDump {
    pub arenas: Vec<ArenaRow>,
    pub classes: Vec<ClassRow>,
    /// Whole-span totals across every GPU class.
    pub total_arenas: usize,
    pub total_empty_arenas: usize,
    pub total_stranded_bytes: usize,
    /// Arenas the whole span would need if every class packed perfectly.
    ///
    /// `total_arenas − total_packed_arenas` is an upper bound on what
    /// relocation could return, not a promise: it assumes every live chunk is
    /// movable, and writer-owned ground and chunks a residence no longer owns
    /// are not. Read `total_stranded_bytes` alongside it and note that the
    /// stranded sum is *unused capacity* — maximal for an arena that is already
    /// empty and needs no copies at all — so the two answer different
    /// questions and neither is "bytes a pass would free".
    pub total_packed_arenas: usize,
}

pub async fn dump() -> Json<KvMapDump> {
    let map = candle_conversation::kv_arenas::global_arena_map();

    let arenas: Vec<ArenaRow> = map
        .iter()
        .map(|a| ArenaRow {
            arena_idx: a.arena_idx,
            rank: a.rank,
            slot_bytes: a.slot_bytes,
            capacity: a.capacity,
            live: a.live,
            occupancy: if a.capacity == 0 {
                0.0
            } else {
                a.live as f32 / a.capacity as f32
            },
            stranded_bytes: a.stranded_bytes(),
        })
        .collect();

    // Grouped by stride. The map arrives sorted by (class, arena), so a class's
    // rows are contiguous and one pass suffices.
    let mut classes: Vec<ClassRow> = Vec::new();
    for a in &map {
        match classes.last_mut() {
            Some(c) if c.slot_bytes == a.slot_bytes => {
                c.arenas += 1;
                c.empty_arenas += usize::from(a.live == 0);
                c.capacity_slots += a.capacity;
                c.live_slots += a.live;
                c.stranded_bytes += a.stranded_bytes();
            }
            _ => classes.push(ClassRow {
                slot_bytes: a.slot_bytes,
                arenas: 1,
                empty_arenas: usize::from(a.live == 0),
                capacity_slots: a.capacity,
                live_slots: a.live,
                occupancy: 0.0,
                stranded_bytes: a.stranded_bytes(),
                packed_arenas: 0,
            }),
        }
    }
    for c in &mut classes {
        c.occupancy = if c.capacity_slots == 0 {
            0.0
        } else {
            c.live_slots as f32 / c.capacity_slots as f32
        };
        // Arenas in this class are one size, so the packed count is the live
        // slots over one arena's capacity, rounded up.
        let per_arena = c.capacity_slots.checked_div(c.arenas).unwrap_or(0);
        c.packed_arenas = if per_arena == 0 {
            0
        } else {
            c.live_slots.div_ceil(per_arena)
        };
    }

    Json(KvMapDump {
        total_arenas: arenas.len(),
        total_empty_arenas: classes.iter().map(|c| c.empty_arenas).sum(),
        total_stranded_bytes: classes.iter().map(|c| c.stranded_bytes).sum(),
        total_packed_arenas: classes.iter().map(|c| c.packed_arenas).sum(),
        arenas,
        classes,
    })
}
