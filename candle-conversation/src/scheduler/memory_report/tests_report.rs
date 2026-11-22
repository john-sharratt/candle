//! Checks the report's fixture, its JSON shape and the containment of every
//! part within its total.
//!
//! The fixture is shaped like a measured large MoE model so the sums mean
//! something. Parts are asserted as contained in totals rather than summed,
//! because summing a part into its total was a real bug that counted the model
//! twice. Expert occupancy must not exceed the zone that holds it, and the
//! residual must subtract the driver baseline.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::state_census::{
        GalleryCensus, GalleryRow, RecurrentBreakdown, SlotRow,
    };

    use super::super::super::turn_status::TurnStatus;

    use super::*;

    use crate::provenance::gallery_arena::HolderKind;

    pub(in super::super) fn sample() -> MemoryReport {
        MemoryReport {
            captured_unix_ms: 1_700_000_000_000,
            vram: Some(VramSection {
                driver_free_bytes: 1,
                driver_total_bytes: 2,
                pool_used_bytes: 3,
                pool_reserved_bytes: 4,
                budget_available_bytes: Some(5),
                governor: Some(GovernorSection {
                    driver_model: Some("tcc"),
                    probe: "cuda",
                    capacity_bytes: 6,
                    headroom_bytes: 7,
                    pool_cushion_bytes: 8,
                    reserved_weights_bytes: 11,
                    reserved_expert_bytes: 12,
                    reserved_scratch_bytes: 13,
                    reserved_kv_bytes: 14,
                }),
            }),
            kv: KvSection {
                classes: vec![KvClassRow {
                    slot_bytes: 1152,
                    arenas: 1,
                    reserved_bytes: 2,
                    live_bytes: 3,
                    packed_arenas: 1,
                    freeable_arenas: 0,
                }],
                arenas: vec![ArenaRow {
                    backing: 0,
                    format: "R16".into(),
                    arenas: 2,
                    bytes: 256,
                }],
                // One live arena at the frontier, so the fixture is a pool with
                // nothing stranded and nothing sparse — the shape a consumer dividing
                // `packed_arenas` by the frontier should read as fully efficient.
                frontier_regions: 1,
                frontier_holder: "the KV pools",
                live_regions: 1,
                span_regions: 0,
                record_regions: 0,
                // No span tenant in the fixture: the breakdown sums to
                // `span_regions`, so an empty list is the consistent pairing.
                span_tenants: Vec::new(),
            },
            warm: WarmSection {
                resident_count: 7,
                pending_warm_bytes: 8,
                pinned_undrainable_bytes: 9,
                resident_bytes: 10,
                held_payload_bytes: 11,
            },
            host: HostSection {
                total_bytes: 32,
                available_bytes: 2,
                buffer_bytes: 4,
                weights_reserved_bytes: 17,
                weights_capped: false,
                kv_warm_budget_bytes: 0,
                pipeline_slack_bytes: 1,
                warm_usage_bytes: 0,
                over_budget: false,
                pages_in_per_sec: Some(12.5),
                commit_total_bytes: Some(30),
                commit_limit_bytes: Some(40),
                index_cache: IndexCacheSection {
                    target_bytes: 8,
                    pinned_bytes: 4,
                    pageable_bytes: 4,
                    held_bytes: 3,
                    entries: 2,
                    hits: 5,
                    misses: 2,
                    uncached: 1,
                    evicted: 0,
                },
            },
            admission: AdmissionSection {
                setpoint_bytes: 10,
                ceiling_bytes: 11,
                quantum_bytes: 12,
                prefill_width: 1,
                section_width: 2,
                decode_width: 3,
                queued_prefills: 4,
            },
            experts: ExpertSection {
                host_pinned_bytes: 11_000_000_000,
                vram_reserved_bytes: 5_100_000_000,
            },
            weights: WeightSection {
                // OCCUPANCY, and so strictly inside the zone extent
                // (`accounting.expert_zone_bytes` below) that holds it. A
                // fixture with more resident experts than zone to put them in
                // describes a machine that cannot exist, and would let a
                // regression that wires both fields to the zone pass.
                base_bytes: Some(1_100_000_000),
                resident_expert_bytes: Some(5_100_000_000),
                // Below the resident experts above, as a floor must be: it is
                // the ground the zone may not fall under, not the ground it
                // holds.
                floor_bytes: Some(4_000_000_000),
            },
            gallery: GallerySection {
                resident_bytes: 268_435_456,
                resident_turns: 42,
            },
            state: StateSection {
                gallery: Some(GalleryCensus {
                    taken_unix_ms: 1_700_000_000_000,
                    page_bytes: 251_658_240,
                    region_bytes: 268_435_456,
                    keep_within_secs: Some(14_069),
                    idle_past_keep_turns: 0,
                    idle_past_keep_bytes: 0,
                    rows: vec![GalleryRow {
                        status: TurnStatus::Live,
                        holder: HolderKind::Shared,
                        layer: "code_reading".into(),
                        group: "files".into(),
                        turns: 42,
                        bytes: 251_658_240,
                    }],
                }),
                recurrent: RecurrentBreakdown {
                    held_bytes: 2_113_929_216,
                    store_bytes: 132_120_576,
                    live_stores: 16,
                    parked_stores: 0,
                    slots: vec![SlotRow {
                        layer: "code_reading".into(),
                        group: "files".into(),
                        slots: 16,
                        holders: (0..16).map(|i| format!("conv-{i:02}")).collect(),
                    }],
                },
            },
            // Shaped like the measured 3.6-35B so the sums mean something: a
            // 64 GiB span holding the model, a pool reduced to per-wave churn,
            // and ~2 GiB the driver's own context holds that we cannot name.
            //
            // The pool figure is the one that carries the intent. It was 4 GiB
            // when the weights loaded into it; the whole reservation-before-load
            // change is what took it to a few hundred MiB, and a fixture still
            // shaped around the old number would let the containment assertions
            // below pass while asserting the opposite of the design.
            accounting: AccountingSection {
                span_bytes: Some(68_719_476_736),
                // The pool IS the outside set — but the weights are no longer
                // in it.
                outside_span_bytes: 234_881_024,
                outside_pool_bytes: 234_881_024,
                // The model, inside the span.
                inside_dense_bytes: 1_914_699_776,
                // Also inside, so neither may appear in the outside sum.
                inside_gallery_bytes: 268_435_456,
                // 16 sequences of hybrid recurrent state.
                inside_recurrent_bytes: 2_113_929_216,
                // Launch tables, resident headers and attention partials.
                inside_held_bytes: 48_234_496,
                // The elastic weight zone — the largest span tenant on a MoE
                // model, and the one that shrinks as the boundary concedes.
                // The zone's EXTENT — deliberately larger than
                // `weights.resident_expert_bytes` above, which is what is
                // actually loaded into it.
                expert_zone_bytes: 5_939_691_520,
                // One forward's activation ground, released between waves.
                inside_transient_bytes: 67_108_864,
                // One wave's worth: provenance breaks dominating, the arena
                // itself never refusing — the shape measured on the 3.6-35B.
                decline_no_ticket_bytes: 205_959_852,
                decline_closed_bytes: 0,
                device_in_use_bytes: 71_015_942_144,
                // The context + driver working set, measured before the span was
                // reserved. Naming it is what leaves `unaccounted` meaning only
                // "an allocation the report does not know about".
                driver_baseline_bytes: Some(1_900_000_000),
                unaccounted_bytes: 71_015_942_144i64
                    - 68_719_476_736i64
                    - 234_881_024i64
                    - 1_900_000_000i64,
            },
            // The same 48,234,496 B as `inside_held_bytes`, split between its
            // holders, with one holder already turned away once.
            held: Some(HeldSection {
                used_bytes: 48_234_496,
                pending_bytes: 1_048_576,
                peak_bytes: 50_331_648,
                capacity_bytes: 402_653_184,
                largest_free_bytes: 354_418_688,
                sites: vec![
                    HeldSiteRow {
                        site: "candle-core/src/cuda_backend/kernel_scratch.rs:38:24".into(),
                        live_bytes: 33_554_432,
                        live_count: 3,
                        peak_bytes: 33_554_432,
                        takes: 3,
                        declines: 1,
                        declined_bytes: 67_108_864,
                        largest_decline_bytes: 67_108_864,
                    },
                    HeldSiteRow {
                        site: "candle-conversation/src/scheduler/decode.rs:500:66".into(),
                        live_bytes: 13_631_488,
                        live_count: 13,
                        peak_bytes: 16_777_216,
                        takes: 2_048,
                        declines: 0,
                        declined_bytes: 0,
                        largest_decline_bytes: 0,
                    },
                ],
            }),
        }
    }

    /// The held section's holders and fenced bytes account for exactly what the
    /// ground reports in use — the ledger and the extents are two books over
    /// the same ranges, and a holder missing from one is a range nobody owns.
    #[test]
    fn the_held_holders_and_fences_sum_to_the_ground_in_use() {
        let r = sample();
        let h = r.held.as_ref().expect("the fixture has a held section");
        let holders: u64 = h.sites.iter().map(|s| s.live_bytes).sum();
        assert_eq!(holders + h.pending_bytes, h.used_bytes);
        assert_eq!(h.used_bytes, r.accounting.inside_held_bytes);
    }

    /// The JSON shape is the API contract — every section and the fields other
    /// tooling greps for must be present by name.
    #[test]
    fn report_serializes_with_every_section() {
        let json = serde_json::to_string(&sample()).unwrap();
        for key in [
            "captured_unix_ms",
            "driver_free_bytes",
            "pool_reserved_bytes",
            "\"driver_model\":\"tcc\"",
            "\"probe\":\"cuda\"",
            "headroom_bytes",
            "\"arenas\"",
            "pending_warm_bytes",
            "pinned_undrainable_bytes",
            "available_bytes",
            "kv_warm_budget_bytes",
            "weights_reserved_bytes",
            "warm_usage_bytes",
            "over_budget",
            "setpoint_bytes",
            "ceiling_bytes",
            "host_pinned_bytes",
            "pages_in_per_sec",
            "commit_limit_bytes",
            "\"index_cache\"",
            "\"evicted\"",
            "resident_bytes",
            // The two consumers a whole-card sum could otherwise only reach by
            // subtraction: dense weights sit outside the KV pool, and the
            // gallery arena allocates outside the pool AND the class tallies.
            "base_bytes",
            "resident_expert_bytes",
            "resident_turns",
            // The reconciliation. `outside_span_bytes` is the meter for the
            // demotable set — the thing that cost 17x on decode while every
            // individual section read as healthy.
            "outside_span_bytes",
            "outside_pool_bytes",
            "inside_dense_bytes",
            "inside_gallery_bytes",
            "inside_recurrent_bytes",
            "inside_held_bytes",
            "decline_no_ticket_bytes",
            "decline_closed_bytes",
            "unaccounted_bytes",
            "span_bytes",
            // Who holds the held ground, and who it turned away.
            "largest_free_bytes",
            "\"sites\"",
            "declined_bytes",
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
    }

    /// The named parts must be **contained** in the total, not summed into it.
    ///
    /// Trivial arithmetic, and worth a gate precisely because the bug it guards
    /// was arithmetic somebody performed wrongly. The first version of this test
    /// asserted `outside_span == dense + pool` and passed for exactly as long as
    /// the code made the same mistake: the dense weights are allocated through
    /// `CudaDevice::alloc` → `cuMemAllocAsync`, so they are *in* the pool, and
    /// adding them to it counted the model twice — 4,746 MiB of demotable memory
    /// reported where 2,784 MiB existed, with the difference silently deducted
    /// from `unaccounted_bytes`.
    ///
    /// A sum test cannot catch that; a containment test can, which is why the
    /// assertion is now an inequality on each part.
    #[test]
    fn the_outside_span_parts_are_contained_in_the_whole() {
        let r = sample();
        let a = &r.accounting;
        assert_eq!(
            a.outside_span_bytes, a.outside_pool_bytes,
            "the CUDA pool is the whole outside set: every non-span device \
             allocation goes through `cuMemAllocAsync`",
        );
        assert!(
            a.inside_dense_bytes > a.outside_pool_bytes,
            "the span must hold the model, and hold more of it than the pool holds \
             of anything ({} in the span, {} in the whole pool) — this is the point \
             of claiming the reservation before the load, and a fixture that did \
             not exercise it would assert nothing",
            a.inside_dense_bytes,
            a.outside_pool_bytes,
        );
        assert!(
            a.inside_gallery_bytes > 0,
            "the fixture must exercise a gallery that holds something, or this \
             says nothing about where its bytes are counted",
        );
        assert!(
            a.inside_recurrent_bytes > 0,
            "likewise the recurrent state — it is the largest in-span tenant that \
             is neither KV nor weights, and a zero fixture would not exercise the \
             one thing this section is for",
        );
        // The in-span tenants are named PARTS of the span, never addends to it.
        // Same containment rule as the dense weights against the pool, in the
        // other half of the card: the two halves fail the same way, by summing
        // a component with the total that already includes it.
        let named_in_span = a.inside_gallery_bytes
            + a.inside_recurrent_bytes
            + a.inside_held_bytes
            + a.inside_dense_bytes
            + a.expert_zone_bytes
            + a.inside_transient_bytes;
        assert!(
            named_in_span < a.span_bytes.expect("the fixture reserves a span"),
            "the gallery, recurrent state, dense weights, expert zone and transient \
             tier are tenants OF the span ({named_in_span} of {:?}), so they must fit \
             inside it",
            a.span_bytes,
        );
        // **Every large span tenant is reported.** Deliberately not a coverage
        // ratio: the span is a *reservation*, and its unclaimed KV regions are
        // legitimately empty, so the tenants can sum to a fraction of it and
        // nothing is wrong. What can go wrong is a tenant having no field at
        // all — the expert zone was exactly that, the biggest one on a MoE model
        // and entirely absent, which left `dense + recurrent + gallery` several
        // gigabytes short of the truth with no way to tell that from a leak.
        //
        // So the check is presence, not proportion: each named tenant must
        // actually be populated, and a regression that quietly stops reading one
        // (a renamed `region_stats` field, say) trips here rather than showing up
        // as memory that vanished.
        for (name, v) in [
            ("dense", a.inside_dense_bytes),
            ("gallery", a.inside_gallery_bytes),
            ("recurrent", a.inside_recurrent_bytes),
            ("held", a.inside_held_bytes),
            ("expert zone", a.expert_zone_bytes),
            ("transient", a.inside_transient_bytes),
        ] {
            assert!(
                v > 0,
                "span tenant {name:?} reports zero — the fixture exercises every \
                 tenant, so a zero here means the field is no longer being read",
            );
        }
        // **The zone is capacity; the resident figure is occupancy.** Two
        // different questions, and the report answers both — the first version
        // of `expert_zone_bytes` read the zone extent while documenting itself
        // as "resident expert weights", which filed capacity under an occupancy
        // heading and over-counted the span's tenants by the zone's free slots.
        //
        // The invariant is containment, `resident <= zone`, and deliberately not
        // a strict `<`: a fully-packed zone reports the two as EQUAL, and that
        // is the healthy steady state, not a defect — measured 6,724 MiB for
        // both on a loaded 3.6-35B, because the elastic boundary grows the
        // weight zone to fit exactly what loads into it.
        //
        // The fixture models a partially-packed zone (5.1 GB resident in 5.94 GB
        // of ground) so the gap between the two is real here and a reading that
        // collapses them is visible.
        let resident = r
            .weights
            .resident_expert_bytes
            .expect("the fixture reports resident experts");
        assert!(
            resident <= a.expert_zone_bytes,
            "resident experts ({resident}) must fit in the zone that holds them \
             ({}) — occupancy above capacity means one of the two is reading the \
             wrong source",
            a.expert_zone_bytes,
        );
        assert!(
            resident < a.expert_zone_bytes,
            "this FIXTURE is meant to have free slots in the zone, so that the \
             containment check above is exercised rather than trivially met by \
             equality; if the fixture changed, fix the fixture",
        );
        // The residual is the card minus every total we can name — the span, the
        // pool, and the driver's own baseline — and specifically NOT minus the
        // parts *within* those totals, which would double-count.
        //
        // The baseline belongs in this subtraction and used to be missing, which
        // published the driver's footprint as if it were an unexplained
        // allocation: 1.4 GB of "unaccounted" on a 16 GB card that was almost
        // entirely context. A residual that large cannot be read as a signal —
        // a real leak moves it by a fraction of itself.
        //
        // `expect` rather than `unwrap_or(0)`: on a fixture that names a baseline
        // the two spellings agree, so a silent default here would let the field
        // regress to `None` — the state where the subtraction stops happening and
        // the residual quietly means the old thing again — without failing.
        let baseline = a
            .driver_baseline_bytes
            .expect("the fixture measures a driver baseline");
        assert_eq!(
            a.unaccounted_bytes,
            a.device_in_use_bytes as i64
                - a.span_bytes.unwrap_or(0) as i64
                - a.outside_span_bytes as i64
                - baseline as i64,
            "the residual must be what the card holds minus everything we can name",
        );
    }

    /// `latest` hands back what `publish` stored, with a sane age.
    #[test]
    fn publish_then_latest_round_trips() {
        publish(sample());
        let (got, age_ms) = latest().expect("just published");
        assert_eq!(got.captured_unix_ms, 1_700_000_000_000);
        assert_eq!(got.experts.host_pinned_bytes, 11_000_000_000);
        assert!(age_ms < 60_000, "freshly published, age {age_ms}ms");
    }
}
