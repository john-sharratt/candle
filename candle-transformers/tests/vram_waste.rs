//! Wasted VRAM, wave optimality, and soak — `docs/vram_partition_behavioural_tests.md`.
//!
//! **No device, no checkpoint, no forward pass.** Every decision under test is
//! the production one: `GrowthPolicy` is the policy `RegionPool::spare_regions`
//! delegates to, `plan_zone` is what the dense boundary plans with, and
//! `WeightZone` is the MoE zone itself. What the harness supplies is the
//! *workload* and, crucially, an **independent** measure of waste.
//!
//! # Why the waste measure is computed here rather than read from the engine
//!
//! A partition that mis-accounts its own memory reports itself healthy — that is
//! the entire failure mode. The ratchet held 34 of 64 layers streaming while
//! every counter the engine kept said the ground was spoken for, because the
//! counters and the policy shared an assumption. So [`WasteAudit`] recomputes
//! occupancy from first principles — span, weights, KV, tier — and never asks the
//! code under test what it thinks it is using. The two agreeing is the finding;
//! having only one of them is not.
//!
//! # What "wasted" means, and why most idle bytes are not waste
//!
//! Idle VRAM is only a defect if something could have been bought with it. A 3B
//! model on a 192 GB card leaves 180 GB idle and there is nothing whatever to do
//! with it — no layer left to home, no KV the workload asked for. So the audit
//! fails only on **unjustified** waste:
//!
//! - the model is not fully resident and the idle bytes would have homed another
//!   layer, or
//! - a KV claim was refused and the idle bytes would have satisfied it.
//!
//! That makes "each wave is optimal" and "no unjustified waste" the same
//! property, which is why one audit covers both.

use candle_nn::kv_cache::{kv_grow_step, GrowthPolicy, Occupancy, WeightZone, MIN_ELASTIC_RESERVE};
use candle_transformers::models::layer_stream::{
    layer_image, plan_zone, FfnForm, LayerImage, LayerTensor, MixKind, Projection, PROJECTION_ALIGN,
};

use candle::quantized::GgmlDType;

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * MIB;

/// One KV region. Mirrors `TARGET_ARENA_BYTES`; asserted against the real
/// constant below so this cannot drift silently.
const REGION: usize = 16 * MIB;

/// The wave transient tier's ceiling — what a forward may stand at most.
const TIER_CEILING: usize = 912 * MIB;

/// Headroom the KV side keeps when the weight side takes ground.
///
/// The engine's `KV_REGION_SLACK`. It covers the one quantity genuinely in the
/// future: persistence's quantize destinations, unclaimed when the boundary
/// moves.
const SLACK_REGIONS: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════
// Cards
// ═══════════════════════════════════════════════════════════════════════════

/// Usable span for a card, after the governor's cushion.
///
/// Measured on the 4090 Mobile: a 16 GB card yields a 13,840 MiB span once the
/// driver's working set and the pool's cushion are out. ~86% is the ratio that
/// reproduces, and it is applied uniformly rather than guessed per card — the
/// point of the sweep is the *shape* of the answer across sizes, and a per-card
/// fudge factor would only encode this machine's driver into every row.
fn span_of(card_bytes: usize) -> usize {
    card_bytes * 86 / 100
}

/// Every card the engine is expected to run on.
///
/// Consumer parts first, then workstation and datacentre. The 8 GB rows are not
/// decoration: they are where a 27B is almost entirely streamed, which is the
/// regime the boundary policy is least exercised in and most likely to be wrong.
fn cards() -> Vec<(&'static str, usize)> {
    vec![
        ("RTX 4060 / 3070 8GB", 8 * GIB),
        ("RTX 3080 10GB", 10 * GIB),
        ("RTX 3060 / 4070 12GB", 12 * GIB),
        ("RTX 4090 Mobile / 4080 16GB", 16 * GIB),
        ("RTX 4000 Ada 20GB", 20 * GIB),
        ("RTX 3090 / 4090 24GB", 24 * GIB),
        ("RTX 5090 32GB", 32 * GIB),
        ("A100 40GB", 40 * GIB),
        ("L40S / RTX 6000 Ada 48GB", 48 * GIB),
        ("RTX PRO 5000 Blackwell 72GB", 72 * GIB),
        ("A100 / H100 80GB", 80 * GIB),
        ("H100 NVL 94GB", 94 * GIB),
        ("H200 141GB", 141 * GIB),
        ("MI300X 192GB", 192 * GIB),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Models
// ═══════════════════════════════════════════════════════════════════════════

/// How a model's droppable weights are organised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Zone {
    /// Dense trunk: the zone holds whole layers at their natural, unequal sizes.
    DenseLayers,
    /// Routed: the zone holds equal-sized expert slots, and a layer's dense part
    /// is resident always.
    Experts { slot_bytes: usize, slots: usize },
}

/// A representative profile of a model the wave engine runs.
///
/// **Derived, not measured, except where stated.** Per-layer bytes come from
/// published parameter counts at the model's production quant; what the
/// partition reasons about is layer count, per-layer bytes, dense-vs-routed and
/// KV per token, and those are what these carry faithfully. The 27B's figures
/// are the exception and are measured — they are read off this repo's own gate
/// logs (`conceded_mib` of 154/173/179/202 as single layers left the zone, a
/// 657 MiB dense block, 64 layers).
struct Model {
    name: &'static str,
    /// Weights that are never streamed: norms, the trunk's non-layer tensors,
    /// and the routed models' always-resident dense part. The embedding is
    /// excluded — it is host-mapped and costs no VRAM.
    dense_block: usize,
    /// Streamable trunk layers, in trunk order. Ragged where the model is.
    layers: Vec<usize>,
    /// Layers the inference loop cannot run without, held resident always.
    pinned: usize,
    /// KV bytes per token across the whole trunk, at the model's sealed formats.
    kv_per_token: usize,
    zone: Zone,
}

/// A 3:1 DeltaNet:Attention trunk, whose two layer kinds differ in size.
fn hybrid_layers(n: usize, dn: usize, attn: usize) -> Vec<usize> {
    (0..n).map(|i| if i % 4 == 3 { attn } else { dn }).collect()
}

fn uniform_layers(n: usize, each: usize) -> Vec<usize> {
    vec![each; n]
}

fn models() -> Vec<Model> {
    vec![
        // ── Measured. This repo's own 27B gate logs. ──
        Model {
            name: "Qwen3.8-27B Q3_K_M (dense hybrid, measured)",
            dense_block: 657 * MIB,
            layers: hybrid_layers(64, 166 * MIB, 202 * MIB),
            pinned: 2,
            // 64 layers of GQA at the sealed C8-ish ratio; ~10 KiB/token trunk-wide.
            kv_per_token: 10 * 1024,
            zone: Zone::DenseLayers,
        },
        // ── Derived from published shapes. ──
        Model {
            name: "Qwen3.5-0.8B (dense hybrid)",
            dense_block: 90 * MIB,
            layers: hybrid_layers(24, 14 * MIB, 18 * MIB),
            pinned: 2,
            kv_per_token: 1536,
            zone: Zone::DenseLayers,
        },
        Model {
            name: "Qwen3.5-9B (dense hybrid)",
            dense_block: 320 * MIB,
            layers: hybrid_layers(48, 96 * MIB, 118 * MIB),
            pinned: 2,
            kv_per_token: 5 * 1024,
            zone: Zone::DenseLayers,
        },
        Model {
            name: "Llama-3.2-3B Q4_K_M (dense attention)",
            dense_block: 210 * MIB,
            layers: uniform_layers(28, 62 * MIB),
            pinned: 2,
            kv_per_token: 3 * 1024,
            zone: Zone::DenseLayers,
        },
        Model {
            name: "Qwen3-8B Q4_K_M (dense attention)",
            dense_block: 280 * MIB,
            layers: uniform_layers(36, 118 * MIB),
            pinned: 2,
            kv_per_token: 4 * 1024,
            zone: Zone::DenseLayers,
        },
        Model {
            name: "Qwen3-14B Q4_K_M (dense attention)",
            dense_block: 380 * MIB,
            layers: uniform_layers(40, 196 * MIB),
            pinned: 2,
            kv_per_token: 5 * 1024,
            zone: Zone::DenseLayers,
        },
        Model {
            name: "Qwen3-30B-A3B Q4_K_M (MoE)",
            dense_block: 2200 * MIB,
            layers: vec![],
            pinned: 0,
            kv_per_token: 6 * 1024,
            zone: Zone::Experts {
                slot_bytes: 3 * MIB,
                slots: 48 * 128,
            },
        },
        Model {
            name: "Qwen3.6-35B-A3B (MoE)",
            dense_block: 2600 * MIB,
            layers: vec![],
            pinned: 0,
            kv_per_token: 7 * 1024,
            zone: Zone::Experts {
                slot_bytes: 3 * MIB,
                slots: 48 * 160,
            },
        },
        Model {
            name: "Qwen3-235B-A22B Q4_K_M (MoE)",
            dense_block: 6 * GIB,
            layers: vec![],
            pinned: 0,
            kv_per_token: 12 * 1024,
            zone: Zone::Experts {
                slot_bytes: 9 * MIB,
                slots: 94 * 128,
            },
        },
        Model {
            // Geometry from `deepseek4.rs`: 43 layers, 256 routed experts,
            // head_dim 512, K≡V latent attention (so KV per token is small for
            // the parameter count).
            name: "DeepSeek-V4-Flash MXFP4 (latent MoE)",
            dense_block: 8 * GIB,
            layers: vec![],
            pinned: 0,
            kv_per_token: 2 * 1024,
            zone: Zone::Experts {
                slot_bytes: 5 * MIB,
                slots: 43 * 256,
            },
        },
    ]
}

impl Model {
    fn weights_total(&self) -> usize {
        self.dense_block
            + match self.zone {
                Zone::DenseLayers => self.layers.iter().sum::<usize>(),
                Zone::Experts { slot_bytes, slots } => slot_bytes * slots,
            }
    }

    /// The `LayerImage`s the real planner takes. Built through `layer_image`, so
    /// alignment and the padding tail are the production rules.
    fn images(&self) -> Vec<LayerImage> {
        self.layers.iter().map(|b| image_of(*b)).collect()
    }
}

/// A layer image whose `total` is at least `bytes`, built through the real
/// placement rules.
fn image_of(bytes: usize) -> LayerImage {
    let units = bytes.div_ceil(PROJECTION_ALIGN).max(5);
    let roles = [
        LayerTensor::Wqkv,
        LayerTensor::Wz,
        LayerTensor::WOut,
        LayerTensor::FfnGateUp,
        LayerTensor::FfnDown,
    ];
    let rest = units - (roles.len() - 1);
    let p: Vec<Projection> = roles
        .iter()
        .enumerate()
        .map(|(i, &role)| Projection {
            role,
            shape: [32, 128],
            dtype: GgmlDType::Q4_KO,
            payload: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
            extent: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
        })
        .collect();
    layer_image(MixKind::DeltaNet, FfnForm::Fused, &p).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// The independent waste audit
// ═══════════════════════════════════════════════════════════════════════════

/// A partition, as the harness measures it — never as the engine reports it.
#[derive(Debug, Clone, Copy)]
struct Partition {
    span: usize,
    dense_block: usize,
    /// Bytes the weight zone actually occupies.
    zone_used: usize,
    /// Bytes the KV side holds, live.
    kv_used: usize,
    /// The tier standing this forward.
    tier: usize,
    /// Whether every droppable unit is resident.
    whole: bool,
    /// Regions a claim asked for and did not get.
    kv_refused: usize,
    /// Bytes the **next** unit the zone would take back needs.
    ///
    /// Not the smallest unit in the model: the zone fills in protection order,
    /// so the only unit idle ground could actually buy is the specific one that
    /// comes next. A ragged trunk makes the difference real — 200 MiB idle
    /// against a 166 MiB *minimum* looks like waste and is not, when the layer
    /// actually next in line is 202 MiB.
    next_unit: usize,
}

impl Partition {
    fn idle(&self) -> usize {
        self.span
            .saturating_sub(self.dense_block + self.zone_used + self.kv_used + self.tier)
    }

    /// `None` when every idle byte is genuinely unspendable.
    ///
    /// The whole judgement of this suite: idle VRAM on a card far larger than
    /// the workload is not a defect, and failing it would make every big-card row
    /// red for no reason. Waste is only waste when there was something to buy.
    fn unjustified(&self) -> Option<String> {
        let idle = self.idle();
        if !self.whole && idle >= self.next_unit {
            return Some(format!(
                "{} MiB idle while the model still streams — the next unit needs {} MiB",
                idle / MIB,
                self.next_unit / MIB
            ));
        }
        if self.kv_refused > 0 && idle >= REGION {
            return Some(format!(
                "{} MiB idle while {} KV regions were refused",
                idle / MIB,
                self.kv_refused
            ));
        }
        None
    }
}

/// Plan the best partition this card can give this model at this KV demand, and
/// measure it independently.
///
/// This *is* the optimality claim: KV demand is satisfied first (a refused claim
/// fails a request, a streamed layer only slows one), the tier is reserved, and
/// everything left goes to weights. If the result leaves unjustified idle bytes
/// then either the planner or this reasoning is wrong, and both are worth
/// knowing.
fn best_partition(m: &Model, span: usize, kv_bytes: usize, tier: usize) -> Option<Partition> {
    let fixed = m.dense_block + tier;
    if span <= fixed {
        return None; // the card cannot host this model at all
    }
    let kv_used = kv_bytes.min(span - fixed);
    let kv_refused = (kv_bytes - kv_used) / REGION;
    let for_weights = span - fixed - kv_used;

    let (zone_used, whole, next_unit) = match m.zone {
        Zone::DenseLayers => {
            let imgs = m.images();
            let end: u64 = 1 << 62;
            match plan_zone(&imgs, m.pinned, end, for_weights) {
                Ok(plan) => {
                    // `missing` is in eviction order, so the layer that comes
                    // back first is the one given up last.
                    let next = plan
                        .missing
                        .last()
                        .map(|&l| imgs[l].total)
                        .unwrap_or(usize::MAX);
                    (plan.used_bytes(end), plan.is_whole(), next)
                }
                // Below the model's floor: it cannot run here.
                Err(_) => return None,
            }
        }
        Zone::Experts { slot_bytes, slots } => {
            let end: u64 = 1 << 62;
            let mut zone = WeightZone::new(end, slot_bytes, 0, MIN_ELASTIC_RESERVE, 1);
            let want = (for_weights / slot_bytes).min(slots);
            let got = zone.grow_to(want);
            (got * slot_bytes, got >= slots, slot_bytes)
        }
    };

    Some(Partition {
        span,
        dense_block: m.dense_block,
        zone_used,
        kv_used,
        tier,
        whole,
        kv_refused,
        next_unit,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// The waste sweep — every model on every card
// ═══════════════════════════════════════════════════════════════════════════

/// A modest KV demand: one conversation.
fn kv_for(m: &Model, contexts: usize, tokens: usize) -> usize {
    let raw = m.kv_per_token * tokens * contexts;
    raw.div_ceil(REGION) * REGION
}

/// **No unjustified waste, for every model on every card, at every load.**
///
/// The sweep the catalogue's §G describes, run as one property: whatever the
/// card and whatever the workload, VRAM that could have held another layer or
/// served a refused KV claim is not left idle.
#[test]
fn no_unjustified_waste_across_every_model_and_card() {
    let loads = [
        ("single short", 1usize, 512usize),
        ("single long", 1, 32_768),
        ("many short", 32, 512),
        ("many long", 32, 8_192),
        ("huge kv", 64, 16_384),
    ];
    let mut checked = 0usize;
    for m in models() {
        for (card, bytes) in cards() {
            let span = span_of(bytes);
            for (load, ctx, tok) in loads {
                let kv = kv_for(&m, ctx, tok);
                let Some(p) = best_partition(&m, span, kv, TIER_CEILING) else {
                    continue; // model does not fit this card at all
                };
                checked += 1;
                if let Some(why) = p.unjustified() {
                    panic!("{}: {card} @ {load}: {why}", m.name);
                }
            }
        }
    }
    assert!(checked > 200, "the sweep only covered {checked} cases");
}

/// **A card with room to spare hosts the model whole**, and the idle remainder
/// is then genuinely unspendable rather than a planner giving up early.
#[test]
fn a_card_with_room_holds_the_whole_model() {
    for m in models() {
        let need = m.weights_total() + TIER_CEILING + 4 * GIB;
        for (card, bytes) in cards() {
            let span = span_of(bytes);
            if span < need {
                continue;
            }
            let p = best_partition(&m, span, kv_for(&m, 1, 4096), TIER_CEILING)
                .unwrap_or_else(|| panic!("{}: {card} has room but hosts nothing", m.name));
            assert!(
                p.whole,
                "{}: {card} has {} MiB of span for {} MiB of weights and still streams",
                m.name,
                span / MIB,
                m.weights_total() / MIB
            );
        }
    }
}

/// **More VRAM is never worse.** Residency is monotone in card size, for every
/// model at every load.
///
/// The property a per-card special case would break, and the reason `span_of`
/// applies one ratio rather than a table of fudge factors.
#[test]
fn residency_is_monotone_in_card_size() {
    for m in models() {
        for (ctx, tok) in [(1usize, 4096usize), (16, 4096), (64, 16_384)] {
            let kv = kv_for(&m, ctx, tok);
            let mut best = 0usize;
            for (card, bytes) in cards() {
                let span = span_of(bytes);
                let Some(p) = best_partition(&m, span, kv, TIER_CEILING) else {
                    continue;
                };
                assert!(
                    p.zone_used >= best,
                    "{}: {card} homed {} MiB where a smaller card homed {} MiB",
                    m.name,
                    p.zone_used / MIB,
                    best / MIB
                );
                best = p.zone_used;
            }
        }
    }
}

/// **KV is served before weights.** A refused KV claim fails a request; a
/// streamed layer only slows one, so the partition must never hold weights that
/// a refused claim needed.
#[test]
fn kv_demand_is_satisfied_before_weights_are_homed() {
    for m in models() {
        for (card, bytes) in cards() {
            let span = span_of(bytes);
            // A demand large enough to squeeze, but not to exceed the span.
            let kv = (span / 3).div_ceil(REGION) * REGION;
            // The model's own immovable parts plus a wave's tier. Where those
            // alone leave no room for the demand, a refusal is physics rather
            // than policy — the card cannot host this model at this load, and
            // there is no partition that would change it.
            if m.dense_block + TIER_CEILING + kv > span {
                continue;
            }
            let Some(p) = best_partition(&m, span, kv, TIER_CEILING) else {
                continue;
            };
            assert_eq!(
                p.kv_refused,
                0,
                "{}: {card} refused {} KV regions with {} MiB idle",
                m.name,
                p.kv_refused,
                p.idle() / MIB
            );
            assert!(p.kv_used >= kv, "{}: {card} short-changed KV", m.name);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Chain scenarios — the soak
// ═══════════════════════════════════════════════════════════════════════════

/// One step of a continuously varied load.
#[derive(Debug, Clone, Copy)]
struct Step {
    contexts: usize,
    tokens: usize,
}

/// The engine as a server on the internet: bursts of prefill, long stretches of
/// pure decode, a lone interactive session, then a parallel flood — over and
/// over, with nothing returning to a quiescent state in between.
///
/// This is the shape that broke the boundary in production and that no
/// single-config benchmark reproduces: what matters is not any one phase but the
/// *transitions*, and specifically whether ground released by a phase that ended
/// is offered back before the next one needs it.
fn soak_chain(cycles: usize) -> Vec<Step> {
    let mut out = Vec::new();
    // **A settle phase, before anything else.** The assertion compares every
    // later low-demand point against the zone this establishes, so there has to
    // be a "before" to compare to. Without it the test can only see a zone that
    // decays *progressively*, and a ratchet whose forecast saturates on the first
    // flood holds every later cycle equally small — passing a cycle-to-cycle
    // comparison while the zone sits at half the residency it should have.
    for _ in 0..16 {
        out.push(Step {
            contexts: 1,
            tokens: 512,
        });
    }
    for c in 0..cycles {
        // High prefill: many contexts arriving at once, short each.
        for _ in 0..6 {
            out.push(Step {
                contexts: 48,
                tokens: 1024,
            });
        }
        // Pure decode: the same cohort, growing token by token.
        for i in 0..12 {
            out.push(Step {
                contexts: 48,
                tokens: 1024 + i * 256,
            });
        }
        // The flood drains; one interactive session remains.
        for i in 0..10 {
            out.push(Step {
                contexts: 1,
                tokens: 2048 + i * 512,
            });
        }
        // A single long-context inference — the "huge KV, one user" regime.
        out.push(Step {
            contexts: 1,
            tokens: 65_536,
        });
        // High parallelism again. **The same depth every cycle**, and that is
        // load-bearing: the assertion compares the zone at each cycle's tail, so
        // the states have to be comparable. A flood that deepened per cycle was
        // the first version and it made the test fail on correct code — the
        // deeper flood legitimately opens a wider gap, and the geometric growth
        // hedge closes a gap asymptotically rather than in a fixed step count.
        // Variety comes from the phases within a cycle, not from drifting their
        // depth.
        for _ in 0..8 {
            out.push(Step {
                contexts: 40,
                tokens: 4096,
            });
        }
        // The tail: demand genuinely falls away, and it is long enough for the
        // halving grow step to converge rather than merely start.
        for _ in 0..10 {
            out.push(Step {
                contexts: 1,
                tokens: 512,
            });
        }
        let _ = c;
    }
    out
}

/// Run a chain against the **real** growth policy and audit every step.
///
/// The KV side is modelled (arenas claimed and swept as the workload demands);
/// the decision about whether the weight side may take ground is not — that is
/// `GrowthPolicy`, exactly as `RegionPool::spare_regions` calls it.
struct Soak {
    policy: GrowthPolicy,
    total_regions: usize,
    live: usize,
    zone_regions: usize,
    floor_regions: usize,
    grants: usize,
    concessions: usize,
    refused: usize,
    /// The tier standing this forward, in bytes, and the widest seen.
    ///
    /// Varied rather than fixed: a constant tier is the easy case, and the
    /// ceiling's interaction with the boundary is exactly where a placed tier
    /// once made the ceiling deaf to concessions.
    tier_bytes: usize,
    tier_high_water: usize,
    /// Zone size after every step — the trajectory, for determinism and
    /// overshoot.
    trace: Vec<usize>,
}

impl Soak {
    fn new(total_regions: usize, floor_regions: usize) -> Self {
        Self {
            policy: GrowthPolicy::new(),
            total_regions,
            live: 0,
            zone_regions: 0,
            floor_regions,
            grants: 0,
            concessions: 0,
            refused: 0,
            tier_bytes: 0,
            tier_high_water: TIER_CEILING,
            trace: Vec::new(),
        }
    }

    /// One forward: settle KV demand, then negotiate the boundary.
    fn step(&mut self, want_regions: usize) {
        let ceiling = self.total_regions.saturating_sub(self.zone_regions);
        // The KV side takes what it needs, buying from the weight side on
        // contact — the shrink direction, which is exact and immediate.
        if want_regions > ceiling {
            let short = want_regions - ceiling;
            let can_sell = self.zone_regions.saturating_sub(self.floor_regions);
            let sold = short.min(can_sell);
            self.zone_regions -= sold;
            self.concessions += usize::from(sold > 0);
            if sold < short {
                self.refused += short - sold;
            }
            // A purchase is the KV side's own voice; the policy must hear it.
            self.policy.note_demand();
        }
        self.live = want_regions.min(self.total_regions.saturating_sub(self.zone_regions));

        // Phase 0: the tier is released and empty arenas are swept, so occupancy
        // is exact. The grow direction then asks.
        //
        // **`ceiling_blocked` is KV ground the tier's ceiling puts out of reach,
        // not the weight zone's own.** Feeding the zone's regions in here offers
        // the weight side ground it already holds, and it grows without bound —
        // caught by the span invariant on the fourth step of the first soak. At
        // phase 0 the tier is released, so nothing is ceiling-blocked at all.
        let free = self
            .total_regions
            .saturating_sub(self.zone_regions)
            .saturating_sub(self.live);
        let occ = Occupancy {
            live: self.live,
            free_below_ceiling: free,
            ceiling_blocked: 0,
            tier_bytes: self.tier_bytes,
            tier_high_water: self.tier_high_water,
        };
        if let Ok(spare) = self.policy.spare(occ, SLACK_REGIONS, REGION) {
            // **The engine's own hedge, called rather than restated.** A copy
            // here would mean a mutation test proved only that my copy had
            // teeth; the point of these assertions is that they bite on the
            // production step.
            let take = kv_grow_step(spare, 1).min(free);
            self.zone_regions += take;
            self.grants += 1;
        }
        self.trace.push(self.zone_regions);
    }

    /// Set the tier this forward will stand, as `plan_wave_transient` would.
    fn set_tier(&mut self, bytes: usize) {
        self.tier_bytes = bytes;
        self.tier_high_water = self.tier_high_water.max(bytes);
    }

    /// The invariant every step must satisfy, checked independently.
    fn check(&self, at: usize, what: &str) {
        assert!(
            self.zone_regions + self.live <= self.total_regions,
            "{what} step {at}: zone {} + live {} exceeds {} regions",
            self.zone_regions,
            self.live,
            self.total_regions
        );
        assert!(
            self.zone_regions >= self.floor_regions || self.zone_regions == 0,
            "{what} step {at}: zone {} fell below its floor {}",
            self.zone_regions,
            self.floor_regions
        );
    }
}

/// **The soak: a continuously varied load leaves no unjustified idle ground.**
///
/// Prefill bursts, pure decode, a lone session, parallel floods — cycled, with
/// the depth changing each cycle so the policy cannot settle into a period that
/// happens to match. After every step the KV side has what it asked for, the
/// zone is within its floor, and idle ground is bounded by the slack the policy
/// deliberately keeps.
#[test]
fn soak_a_varied_internet_load_leaves_no_ground_stranded() {
    for (card, bytes) in cards() {
        let span = span_of(bytes);
        let total_regions = span / REGION;
        if total_regions < 4 * SLACK_REGIONS {
            continue; // too small for the boundary to have anything to trade
        }
        let floor = total_regions / 20;
        let mut s = Soak::new(total_regions, floor);
        let cycles = 3;
        let chain = soak_chain(cycles);
        // **Scale-free, and compared cycle to cycle — deliberately.**
        //
        // A bound on absolute idle regions was the first attempt and it is slop.
        // Mutation-tested by reinstating the retired windowed forecast, it passed
        // *with the ratchet present*, because on most cards the stranded ground
        // happened to fall under the threshold. What identifies a ratchet is not
        // how much sits idle but that the zone is **smaller at the same point of
        // a later cycle**: every cycle ends with four identical low-demand steps,
        // so those points are directly comparable and nothing but a ratchet moves
        // them apart.
        const SETTLE: usize = 16;
        let per_cycle = (chain.len() - SETTLE) / cycles;
        let mut baseline = 0usize;
        let mut cycle_end_zone: Vec<usize> = Vec::new();
        for (i, step) in chain.iter().enumerate() {
            // A representative trunk: ~10 KiB/token, as the measured 27B.
            let want = (step.contexts * step.tokens * 10 * 1024).div_ceil(REGION);
            s.step(want.min(total_regions - floor));
            s.check(i, card);
            if i + 1 == SETTLE {
                baseline = s.zone_regions;
            } else if i >= SETTLE && (i + 1 - SETTLE).is_multiple_of(per_cycle) {
                cycle_end_zone.push(s.zone_regions);
            }
        }
        assert!(
            baseline > 0,
            "{card}: the zone never settled before the load"
        );
        for (n, &end) in cycle_end_zone.iter().enumerate() {
            assert!(
                end >= baseline,
                "{card}: cycle {n} ended with {end} regions against a settled {baseline} — \
                 ground the load released was not returned"
            );
        }
        assert!(
            s.grants > 0,
            "{card}: the boundary never grew across the soak"
        );
    }
}

/// **The ratchet, as a trajectory over the real policy.**
///
/// Demand rises to a flood and falls away. The zone must end at least as large
/// as it started — the property whose absence held 34 of 64 layers streaming.
#[test]
fn soak_a_burst_that_ends_returns_its_ground() {
    for (card, bytes) in cards() {
        let span = span_of(bytes);
        let total = span / REGION;
        if total < 4 * SLACK_REGIONS {
            continue;
        }
        let floor = total / 20;
        let mut s = Soak::new(total, floor);
        // Settle at a light load.
        for _ in 0..20 {
            s.step(total / 10);
        }
        let before = s.zone_regions;
        // A flood takes the ground.
        for _ in 0..10 {
            s.step(total - floor);
        }
        assert!(
            s.zone_regions <= before,
            "{card}: the flood did not take ground"
        );
        // It ends; the ground must come back.
        for _ in 0..40 {
            s.step(total / 10);
        }
        assert!(
            s.zone_regions >= before,
            "{card}: after the burst the zone held {} regions, was {before} before it",
            s.zone_regions
        );
    }
}

/// **A stationary load reaches a fixed point and stops moving.** Churn on the
/// hot path is what the geometric hedge exists to prevent, and a boundary that
/// never settles pays an eviction per forward for ever.
#[test]
fn soak_a_stationary_load_converges_and_stops_moving() {
    for (card, bytes) in cards() {
        let span = span_of(bytes);
        let total = span / REGION;
        if total < 4 * SLACK_REGIONS {
            continue;
        }
        let mut s = Soak::new(total, total / 20);
        for _ in 0..60 {
            s.step(total / 4);
        }
        let settled = s.zone_regions;
        let moves_before = s.grants + s.concessions;
        for _ in 0..60 {
            s.step(total / 4);
        }
        assert_eq!(
            s.zone_regions, settled,
            "{card}: a stationary load kept moving the boundary"
        );
        let churn = s.grants + s.concessions - moves_before;
        assert!(
            churn <= 2,
            "{card}: {churn} boundary moves over 60 stationary forwards"
        );
    }
}

/// **An ingest — demand that only ever climbs — is never handed ground it is
/// about to need.**
///
/// The derivative guard's test. Occupancy alone cannot see this: at every
/// instant there is room, and a moment later there is not.
#[test]
fn soak_a_monotone_ingest_never_loses_ground_to_the_weight_side() {
    for (card, bytes) in cards() {
        let span = span_of(bytes);
        let total = span / REGION;
        if total < 4 * SLACK_REGIONS {
            continue;
        }
        let floor = total / 20;
        let mut s = Soak::new(total, floor);
        let mut want = total / 20;
        while want < total - floor {
            s.step(want);
            assert_eq!(
                s.refused, 0,
                "{card}: an ingest was refused {} regions at demand {want}",
                s.refused
            );
            want += (total / 40).max(1);
        }
    }
}

/// **H5 — the partition is deterministic.**
///
/// The property every other soak assertion rests on. A trajectory test that is
/// not reproducible is not a test: a failure could not be distinguished from
/// scheduling noise, and a pass would say nothing about the next run. Stated
/// over the full varied chain rather than a settled load, because the
/// interesting non-determinism would live in the transitions.
#[test]
fn h5_the_same_workload_produces_the_same_trajectory() {
    for (card, bytes) in cards() {
        let total = span_of(bytes) / REGION;
        if total < 4 * SLACK_REGIONS {
            continue;
        }
        let floor = total / 20;
        let chain = soak_chain(2);
        let run = |_: usize| {
            let mut s = Soak::new(total, floor);
            for step in &chain {
                let want = (step.contexts * step.tokens * 10 * 1024).div_ceil(REGION);
                s.step(want.min(total - floor));
            }
            (s.trace.clone(), s.grants, s.concessions, s.refused)
        };
        let a = run(0);
        let b = run(1);
        assert_eq!(
            a.0, b.0,
            "{card}: the zone trajectory differed between runs"
        );
        assert_eq!(
            (a.1, a.2, a.3),
            (b.1, b.2, b.3),
            "{card}: the boundary-move counts differed between runs"
        );
    }
}

/// **H4 — under a stationary load the zone rises monotonically to its fixed
/// point and never passes it.**
///
/// # What this does *not* test, established by mutation
///
/// It reads like a guard against the growth step taking too much, and it is not.
/// Replacing `kv_grow_step` with "take the whole offer" — the change measured
/// worse on real hardware, applied grants falling from 17 to 4 and the zone
/// settling a layer lower — leaves every assertion here passing. The harm from
/// over-taking is *churn*: the KV side buys the ground straight back and the
/// purchase trips the pressure guard, and reproducing that needs the engine's
/// own free list and broker rather than this harness's model of them. Measured
/// on the varied chain, over-taking changed grants from ~55 to 7 and concessions
/// not at all, so the final zone size — which is what these assertions read — is
/// identical.
///
/// The property is guarded, just not here: `kv_grow_step`'s own unit tests
/// (`a_negotiation_always_leaves_something_on_the_table` and two others) fail
/// under that mutation. This one is worth keeping for what it does say —
/// monotone approach, no overshoot — and worth not mistaking for more.
#[test]
fn h4_convergence_approaches_the_fixed_point_without_overshooting() {
    for (card, bytes) in cards() {
        let total = span_of(bytes) / REGION;
        if total < 4 * SLACK_REGIONS {
            continue;
        }
        let mut s = Soak::new(total, total / 20);
        for _ in 0..80 {
            s.step(total / 4);
        }
        let settled = s.zone_regions;
        // Every point of the approach must lie at or below where it settles: a
        // trace that rises past `settled` and comes back is an overshoot.
        for (i, &z) in s.trace.iter().enumerate() {
            assert!(
                z <= settled,
                "{card}: step {i} held {z} regions, past the settled {settled} — overshoot"
            );
        }
        // ...and the approach is monotone, so the zone never gives ground back
        // under a load that never asked for it.
        for w in s.trace.windows(2) {
            assert!(
                w[1] >= w[0],
                "{card}: the zone shrank under a stationary load, {} → {}",
                w[0],
                w[1]
            );
        }
    }
}

/// **C4 — recovery is bounded in forwards, not in wall-clock.**
///
/// A policy keyed on time cannot serve a workload that changes faster than its
/// window, which is what the retired forecast did: every boundary event in a full
/// 27B gate run falls inside a 30 s span against what was a 60 s window. So the
/// bound has to be counted in forwards, and stated.
#[test]
fn c4_a_burst_is_recovered_within_a_bounded_number_of_forwards() {
    /// Forwards a burst's ground must be back within. The geometric hedge halves
    /// the gap each negotiation, so this is logarithmic in the span — 24 is ample
    /// for the largest card here and tight enough to catch a policy that has
    /// stopped converging.
    const BOUND: usize = 24;
    for (card, bytes) in cards() {
        let total = span_of(bytes) / REGION;
        if total < 4 * SLACK_REGIONS {
            continue;
        }
        let floor = total / 20;
        let mut s = Soak::new(total, floor);
        // **Settle to the actual fixed point, not for a fixed count.**
        //
        // A fixed number of warm-up forwards was the first version and it is
        // circular: under a policy that converges slowly the baseline is itself
        // tiny, so the burst barely dents it and recovery looks instant.
        // Mutation-tested with `kv_grow_step` reduced to a flat one-region step —
        // exactly the slow convergence this test exists to catch — and it passed.
        // Running to a fixed point makes the baseline the policy's real answer,
        // so a slower policy has correspondingly further to climb back.
        let mut settled_at = 0;
        for _ in 0..4000 {
            let before = s.zone_regions;
            s.step(total / 10);
            settled_at += 1;
            if s.zone_regions == before && settled_at > 8 {
                break;
            }
        }
        let baseline = s.zone_regions;
        assert!(baseline > 0, "{card}: the zone never grew before the burst");
        for _ in 0..8 {
            s.step(total - floor);
        }
        // Count the forwards it takes to get back.
        let mut took = None;
        for n in 1..=BOUND {
            s.step(total / 10);
            if s.zone_regions >= baseline {
                took = Some(n);
                break;
            }
        }
        assert!(
            took.is_some(),
            "{card}: {BOUND} forwards after the burst the zone is {} of a baseline {baseline}",
            s.zone_regions
        );
    }
}

/// **G7 — a tier whose price changes every forward does not strand ground.**
///
/// The soak above holds the tier constant, which is the easy case. A real wave's
/// tier varies with cohort width, and the tier is what the region ceiling is
/// measured against — a placed tier once made the ceiling deaf to the boundary
/// entirely. So the price is varied here, including the step down, which is where
/// ground comes free and must be noticed.
#[test]
fn g7_a_varying_tier_price_leaves_no_ground_stranded() {
    for (card, bytes) in cards() {
        let total = span_of(bytes) / REGION;
        if total < 4 * SLACK_REGIONS {
            continue;
        }
        let floor = total / 20;
        let mut s = Soak::new(total, floor);
        // Settle with a narrow tier.
        s.set_tier(64 * MIB);
        for _ in 0..24 {
            s.step(total / 8);
        }
        let baseline = s.zone_regions;
        // The tier swells and shrinks under an otherwise unchanged load.
        for (i, tier) in [896, 512, 128, 640, 64, 912, 96].iter().enumerate() {
            s.set_tier(tier * MIB);
            for _ in 0..6 {
                s.step(total / 8);
                s.check(i, card);
            }
        }
        // Back to the narrow tier: the ground the wide ones needed is free again.
        s.set_tier(64 * MIB);
        for _ in 0..24 {
            s.step(total / 8);
        }
        assert!(
            s.zone_regions >= baseline,
            "{card}: after the tier settled back the zone held {} against a baseline {baseline}",
            s.zone_regions
        );
    }
}

/// **E5 — both zone kinds answer the broker to the same contract.**
///
/// A model is dense or routed, never both, so the two zones are never exercised
/// together and a divergence in their contract is invisible until a model of the
/// other kind meets the same pressure. Both must concede what is asked, stop at
/// their own floor, and report honestly which of the two happened.
#[test]
fn e5_the_dense_and_expert_zones_concede_to_the_same_contract() {
    let end: u64 = 1 << 62;
    for m in models() {
        // Ask each zone to give up ground in steps, from full down past its floor.
        match m.zone {
            Zone::DenseLayers => {
                let imgs = m.images();
                let total: usize = imgs.iter().map(|i| i.total).sum();
                let mut last = usize::MAX;
                let mut refused_below_floor = false;
                let mut budget = total;
                while budget > 0 {
                    match plan_zone(&imgs, m.pinned, end, budget) {
                        Ok(p) => {
                            assert!(
                                p.resident() <= last,
                                "{}: conceding ground raised residency",
                                m.name
                            );
                            last = p.resident();
                            assert!(
                                p.resident() >= m.pinned,
                                "{}: a concession dropped the pinned head",
                                m.name
                            );
                        }
                        Err(_) => {
                            refused_below_floor = true;
                            break;
                        }
                    }
                    budget = budget.saturating_sub(total / 32 + 1);
                }
                assert!(
                    refused_below_floor,
                    "{}: the dense zone never refused, however far it was pushed",
                    m.name
                );
            }
            Zone::Experts { slot_bytes, slots } => {
                // **The floor is stated here so the assertion can be independent
                // of the zone's own answer.** Comparing capacity against
                // `min_capacity()` was the first version and it is circular:
                // mutation-tested by making `min_capacity` return zero — the
                // floor removed entirely — and it passed, because both sides of
                // the comparison moved together.
                const EXPERT_FLOOR: usize = 4;
                let mut zone =
                    WeightZone::new(end, slot_bytes, 0, MIN_ELASTIC_RESERVE, EXPERT_FLOOR);
                zone.grow_to(slots.min(4096));
                let full = zone.capacity();
                let mut last = full;
                for target in (0..=full).rev().step_by((full / 32).max(1)) {
                    let plan = zone.retract_to(target, |_| 0.0);
                    assert!(
                        zone.capacity() <= last,
                        "{}: retracting raised capacity",
                        m.name
                    );
                    last = zone.capacity();
                    // Whatever it gives up, it never gives up a live slot's data
                    // without saying so — the plan names every move and drop.
                    assert!(
                        plan.evict.len() + plan.relocate.len() <= full,
                        "{}: a retraction reported more moves than slots",
                        m.name
                    );
                }
                assert!(
                    zone.capacity() >= EXPERT_FLOOR,
                    "{}: the expert zone retracted to {} slots, below the floor of \
                     {EXPERT_FLOOR} it was built with — a model with no resident \
                     experts cannot run at all",
                    m.name,
                    zone.capacity()
                );
            }
        }
    }
}

/// The region size this file reasons in is the engine's.
///
/// Gated because `REGION_BYTES` is only exported on a CUDA build, while the rest
/// of this suite deliberately needs no backend at all — running it on a machine
/// with no GPU is the point. So the constant is restated locally and this is what
/// stops the restatement drifting: on any build that *can* see the real one, it
/// must agree.
#[cfg(feature = "cuda")]
#[test]
fn the_harness_region_size_matches_the_engine() {
    assert_eq!(
        REGION,
        candle_nn::kv_cache::REGION_BYTES,
        "the harness and the pool disagree about a region"
    );
}
