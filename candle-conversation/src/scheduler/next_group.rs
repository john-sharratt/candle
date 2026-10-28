//! The prefill group that forms next, and an offered turn's place in it.
//!
//! One answer, read by the group's formation (`Scheduler::form_wave_group`)
//! and by admission (`admit_ground::AdmitPass`), so admission sizes a turn
//! against the group it will run in — see `docs/continuous_fair_waves.md` §7.
//! Sized against the standing cohort instead, the held group first at its fixed
//! rows, one long member starved everything behind it: on WhiteWorld
//! (2026-10-27 22:46–22:48) a single `High` member 158k tokens long held all
//! 3,072 rows of every forward, four `Urgent` content checks were never offered
//! for ninety seconds, and an 87-token `High` turn waited six minutes.

use std::cmp::Reverse;
use std::collections::HashSet;

use candle_nn::kv_cache::CHUNK_SIZE;

use super::admit::pass_budget::GroupCaps;
use super::band::rank;
use super::gang;
use super::interleave::effective_weight_zone_bytes;
use super::section_pack;
use super::{Scheduler, WaveMember};
use crate::projection::DecodePriority;

/// A prompt a formation considers: one in flight, or one offered to admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Candidate {
    pub(super) seq: usize,
    pub(super) band: DecodePriority,
    /// Rows it has left to prefill.
    pub(super) remaining: usize,
    /// The position its next row lands at.
    pub(super) depth: usize,
    /// A section ingest rather than a dialogue prefill.
    pub(super) section: bool,
}

/// What admission asks about: a candidate, and the most rows its pricing lets
/// it advance a forward — `None` for whatever its place in the group gives it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Offer {
    pub(super) who: Candidate,
    pub(super) cap: Option<usize>,
}

/// Where a formation's sections ride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Sections {
    /// Nowhere: the standalone section pass already ran this wave.
    Out,
    /// In what the prompts leave, and only beside one — the creep group's rule.
    Beside,
    /// As [`Self::Beside`]; and with no prompt in the group, packed as the
    /// standalone section pass packs them (`build_section_batch`) — where a
    /// section offered to admission runs when the prompts leave it no room.
    Alone,
}

/// One member of a group as it forms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Planned {
    /// The member, with the rows it advances.
    pub(super) member: WaveMember,
    /// Its rows left to prefill, and the position its next row lands at.
    pub(super) remaining: usize,
    pub(super) depth: usize,
}

/// A gang a formation forms anew: how wide it may run, and how many long
/// prompts it was formed from.
#[derive(Clone, Copy, Debug)]
pub(super) struct GangFormed {
    pub(super) width: usize,
    pub(super) longs: usize,
}

/// A group as it forms.
#[derive(Debug, Default)]
pub(super) struct NextGroup {
    /// Its members in group order — prefills, then sections.
    pub(super) members: Vec<Planned>,
    /// The gang it forms with (`docs/prefill_gangs.md`), and how, when it is
    /// formed here rather than carried on.
    pub(super) gang: Vec<usize>,
    pub(super) gang_formed: Option<GangFormed>,
    /// The rows its forward may carry.
    pub(super) budget: usize,
    /// The offer's place in [`Self::members`], when it is one.
    pub(super) offer_at: Option<usize>,
    /// Whether it is the standalone section pass rather than a creep group.
    pub(super) alone: bool,
}

impl NextGroup {
    /// The members as the wave carries them.
    pub(super) fn wave_members(&self) -> Vec<WaveMember> {
        self.members.iter().map(|p| p.member).collect()
    }

    /// The rows the group carries.
    pub(super) fn rows(&self) -> usize {
        self.members.iter().map(|p| p.member.advance()).sum()
    }

    /// The rows the offer advances in it — zero when it is no member.
    pub(super) fn offer_rows(&self) -> usize {
        self.offer_at
            .map_or(0, |i| self.members[i].member.advance())
    }

    /// Add `e` advancing `rows`.
    fn push(&mut self, e: &Entry, rows: usize) {
        if e.offer {
            self.offer_at = Some(self.members.len());
        }
        let (seq_id, advance) = (e.who.seq, rows);
        self.members.push(Planned {
            member: if e.who.section {
                WaveMember::Section { seq_id, advance }
            } else {
                WaveMember::Prefill { seq_id, advance }
            },
            remaining: e.who.remaining,
            depth: e.who.depth,
        });
    }
}

/// What the formation an offer joins draws from ([`Scheduler::prospect`]):
/// the same for every offer of one band until something is admitted, however
/// many rows each is priced at.
#[derive(Clone, Debug)]
pub(super) struct Prospect {
    /// The work in flight it forms from.
    candidates: Vec<Candidate>,
    /// The members of the creep group standing — in flight or formed — which
    /// no standalone section pass takes (`wave_group_section_seqs`); a paused
    /// group's are no candidates at all.
    creeping: HashSet<usize>,
    /// The highest band among the prompts and sections in flight that can
    /// advance — held, paused or neither — which holds a standalone section
    /// pass of any band below it (`section_batch_band`, `waiting_above`).
    top_in_flight: Option<DecodePriority>,
    /// The caps every group it forms is budgeted against.
    caps: GroupCaps,
    /// The band leading its prompts, when it is a live one: the group
    /// carries that band and above only.
    live_lead: Option<DecodePriority>,
}

impl Prospect {
    /// Whether every group it forms leaves out work at `band` — below the
    /// live band its prompts lead with — so no formation need be run to say so.
    pub(super) fn leaves_out(&self, band: DecodePriority) -> bool {
        self.live_lead.is_some_and(|lead| rank(band) < rank(lead))
    }
}

/// A candidate as the formation carries it: the offer marked, and the most
/// rows it may advance.
#[derive(Clone, Copy)]
struct Entry {
    who: Candidate,
    cap: usize,
    offer: bool,
}

impl Scheduler {
    /// The prompts and sections in flight a formation draws from, each
    /// `advanced(seq)` rows further on than it stands: every one that can
    /// advance and is not `paused`, prefills then sections, in arrival order.
    /// A prompt past the model's RoPE reach is no candidate: its headers would
    /// be refused, and the refusal would fail every member of the forward.
    pub(super) fn formation_candidates(
        &self,
        paused: &HashSet<usize>,
        advanced: impl Fn(usize) -> usize,
    ) -> Vec<Candidate> {
        let reach = self.session.rope_reach();
        let at = |seq: usize| self.session.sequence_offset(seq).unwrap_or(0);
        let prefills = self
            .active_prefills
            .iter()
            .filter(|p| p.can_advance() && !paused.contains(&p.work.sequence_id.0))
            .filter_map(|p| {
                let seq = p.work.sequence_id.0;
                let left = p.remaining();
                if at(seq) + left > reach {
                    return None;
                }
                let moved = advanced(seq).min(left);
                Some(Candidate {
                    seq,
                    band: self.decode_priority_or_high(p.work.sequence_id),
                    remaining: left - moved,
                    depth: at(seq) + moved,
                    section: false,
                })
            });
        let sections = self
            .active_section_ingests
            .iter()
            .filter(|s| s.can_advance() && !paused.contains(&s.sequence_id.0))
            .map(|s| {
                let seq = s.sequence_id.0;
                let left = s.remaining();
                let moved = advanced(seq).min(left);
                Candidate {
                    seq,
                    band: s.band,
                    remaining: left - moved,
                    depth: at(seq) + moved,
                    section: true,
                }
            });
        prefills
            .chain(sections)
            .filter(|c| c.remaining > 0)
            .collect()
    }

    /// The group `candidates` — and `offer`, when given — form at the
    /// composition `target`, budgeted against `caps`. Pure: the formation
    /// `form_wave_group` stores, and the one admission prices an offer
    /// against.
    ///
    /// **The highest band leads the group.** The group's adapter is its first
    /// member's, so leading with a lower band could leave a waiting dialogue
    /// prompt out of the group re-formed to carry it
    /// (`creep_outranked_by_a_waiting_prompt`) — and that restart would then
    /// fire again every step, the creep never completing. **Within a band, the
    /// prompt nearest its first token leads**, so the budget filled in order
    /// (`gang::gang_shares`) finishes it and it decodes while the next ones
    /// prefill. Both sorts are stable, so a band keeps its arrival order.
    ///
    /// **A `High` group carries `High` prompts only.** Every row in a group
    /// rides every layer of its sweep, so a person's prompt beside eight
    /// characters' and ingest chunks waits for all of their rows — measured on
    /// `.5`, 2026-10-17, a 10-member, 8,190-row group creeping at one layer a
    /// step held a dialogue's first token ~24 s. Below `High` throughput matters
    /// more than latency, and the bands mix. A `High` section counts: it is a
    /// dialogue's own (a journal entry its next turn reads), and a group of
    /// ingest rows would hold it as long. **An `Urgent` group carries `Urgent`
    /// prompts only**, for the same reason one band down: the check an image
    /// draw waits on, beside four Makers' table prompts, would wait for all of
    /// their rows.
    ///
    /// **One adapter per group**, the first member's: the projections run once
    /// over every row, so a group carries one adapter or none, and the rest wait
    /// for a group of their own. A section is an ordinary row of the forward,
    /// and an unadapted ingest riding an adapted group would be prefilled
    /// through the wrong projections and its K/V written that way permanently.
    ///
    /// **Bounded by what one forward can carry, filled in order.** Each prompt
    /// offers at most a pass; the group budget is filled in the order above —
    /// the nearest prompt takes all it has, the next what is left, a gang in
    /// proportion to its remainders — and the rest rides the next group;
    /// sections fill only what the prompts left. Unbounded, a burst of queued
    /// turns became one forward: npcd's world ingest on the routed
    /// Qwen3.6-35B-A3B asked for a 3.3 GB tier against a 3.0 GB gap between the
    /// KV frontier and the weight floor, and every turn in the wave failed with
    /// it.
    pub(super) fn form_group(
        &self,
        candidates: &[Candidate],
        offer: Option<Offer>,
        sections: Sections,
        target: usize,
        caps: GroupCaps,
    ) -> NextGroup {
        let pass = self.max_prefill_pass_tokens;
        let entry = |who: Candidate, cap: Option<usize>, offer: bool| {
            let most = if who.section { usize::MAX } else { pass };
            Entry {
                who,
                cap: cap.map_or(most, |c| c.min(most)),
                offer,
            }
        };
        let entries: Vec<Entry> = candidates
            .iter()
            .map(|&who| entry(who, None, false))
            .chain(offer.map(|o| entry(o.who, o.cap, true)))
            .filter(|e| e.who.remaining > 0)
            .collect();
        let mut prefills: Vec<Entry> = entries.iter().copied().filter(|e| !e.who.section).collect();
        prefills.sort_by_key(|e| (Reverse(rank(e.who.band)), e.who.remaining));
        let lead = prefills.first().map(|e| e.who.band);
        let high_section = sections != Sections::Out
            && entries
                .iter()
                .any(|e| e.who.section && e.who.band.is_live());
        let floor = match lead.filter(|b| b.is_live()) {
            Some(lead) => Some(lead),
            None => high_section.then_some(DecodePriority::High),
        };
        let background = floor.is_none();
        let below_floor = |b: DecodePriority| floor.is_some_and(|f| rank(b) < rank(f));
        prefills.retain(|e| !below_floor(e.who.band));
        // ── A background group prefills its long prompts as a gang ───────────
        //
        // With nothing live leading, the long prompts are prefilled together so
        // they reach their first tokens in the same forward and decode side by
        // side, rather than one at a time and one row wide. The shorts ride
        // ahead of the gang and the longs outside it behind; stable, so each
        // lane keeps its band and nearest-first order.
        let (gang, gang_formed) = self.gang_for(background, &entries);
        // An offer is not in flight, so no live gang holds it — whatever
        // sequence it is keyed by — and only a gang formed here can.
        let in_gang = |e: &Entry| gang.contains(&e.who.seq) && (!e.offer || gang_formed.is_some());
        if background && !gang.is_empty() {
            prefills.sort_by_key(|e| {
                if in_gang(e) {
                    1
                } else if e.who.remaining > pass {
                    2
                } else {
                    0
                }
            });
        }
        let adapter = prefills
            .first()
            .and_then(|e| self.session.sequence_adapter(e.who.seq))
            .map(str::to_owned);
        let same_adapter =
            |e: &Entry| self.session.sequence_adapter(e.who.seq) == adapter.as_deref();
        prefills.retain(same_adapter);

        // A queue that passes the cap by no more than the model's slack rides
        // whole, rather than leaving a straggler group a sweep of its own.
        let advances: Vec<usize> = prefills
            .iter()
            .map(|e| e.who.remaining.min(e.cap))
            .collect();
        let budget = caps.budget(advances.iter().sum(), target);
        let ganged: Vec<bool> = prefills.iter().map(in_gang).collect();
        let shares = gang::gang_shares(&advances, &ganged, budget, CHUNK_SIZE);
        let mut group = NextGroup {
            gang_formed,
            budget,
            ..NextGroup::default()
        };
        for (e, share) in prefills.iter().zip(shares) {
            if share > 0 {
                group.push(e, share);
            }
        }
        let section_entries = entries.iter().filter(|e| e.who.section);
        if sections == Sections::Alone && group.members.is_empty() {
            self.pack_sections_alone(&mut group, section_entries, target);
        } else if sections != Sections::Out && !group.members.is_empty() {
            let mut used = group.rows();
            for e in section_entries {
                if below_floor(e.who.band) || !same_adapter(e) {
                    continue;
                }
                let advance = e.who.remaining.min(budget).min(e.cap);
                // The prompts guarantee the group progresses, so no section
                // is forced in past the budget; the rest ride a later group or
                // the standalone section pass.
                if used + advance > budget {
                    break;
                }
                group.push(e, advance);
                used += advance;
            }
        }
        group.gang = gang;
        group
    }

    /// The sections `entries` as the standalone section pass would carry them
    /// at the composition `target`: the highest band waiting, and only live
    /// sections while one is; one pass of rows, packed in order
    /// ([`section_pack::pack`]). Their stores are priced by admission, so none
    /// is passed over for its store here.
    fn pack_sections_alone<'a>(
        &self,
        group: &mut NextGroup,
        entries: impl Iterator<Item = &'a Entry>,
        target: usize,
    ) {
        let entries: Vec<&Entry> = entries.collect();
        let Some(band) = entries.iter().map(|e| e.who.band).max_by_key(|&b| rank(b)) else {
            return;
        };
        let riding: Vec<&Entry> = entries
            .into_iter()
            .filter(|e| !band.is_live() || e.who.band.is_live())
            .collect();
        let cap = self.pass_budget_under(target);
        let packing: Vec<section_pack::Candidate> = riding
            .iter()
            .map(|e| section_pack::Candidate {
                remaining: e.who.remaining.min(e.cap),
                fresh: false,
            })
            .collect();
        for (k, advance) in section_pack::pack(&packing, cap, usize::MAX) {
            group.push(riding[k], advance);
        }
        group.budget = cap;
        group.alone = true;
    }

    /// What the formation an offer at `band` joins draws from — `None` when a
    /// paused creep resumes ahead of it instead, so the offer would wait
    /// behind a sweep it is not in.
    ///
    /// A creep is in flight once its sweep has run a layer (`cursor > 0`, or
    /// its residual held); a group formed and not yet run re-forms from
    /// scratch at the next step, so it is no creep here. One in flight is
    /// **paused** when `band`, or a prompt already waiting, outranks it and no
    /// creep is paused yet: its members are set aside and the formation runs
    /// without them. Otherwise it reaches its sweep boundary first — at once
    /// when it is outranked with a creep already paused (`finish_sweep`), at
    /// its own pace when it is not — and its members re-enter the formation
    /// the rows that sweep advances them further on.
    pub(super) fn prospect(&self, band: DecodePriority) -> Option<Prospect> {
        let in_flight = self.wave_prefill_cursor > 0 || self.wave_prefill_residual.is_some();
        let held: &[WaveMember] = if in_flight {
            &self.wave_prefill_members
        } else {
            &[]
        };
        let creep = self.creep_priority(held);
        let pauses = !held.is_empty()
            && self.paused_creep.is_none()
            && (rank(band) > rank(creep) || self.creep_outranked_by_a_waiting_prompt());
        let (paused, resumes_above) = if pauses {
            let seqs: HashSet<usize> = held.iter().map(WaveMember::seq_id).collect();
            (seqs, Some(creep))
        } else {
            match &self.paused_creep {
                Some(p) => (p.seq_ids(), Some(p.band)),
                None => (HashSet::new(), None),
            }
        };
        let advanced = |seq: usize| {
            if pauses {
                return 0;
            }
            held.iter()
                .find(|m| m.seq_id() == seq)
                .map_or(0, WaveMember::advance)
        };
        let candidates = self.formation_candidates(&paused, advanced);
        // **A paused creep resumes once nothing above its band waits**
        // (`resume_creep`), ahead of any formation.
        if let Some(b) = resumes_above {
            let above = |p: DecodePriority| rank(p) > rank(b);
            if !candidates.iter().any(|c| above(c.band)) && !above(band) {
                return None;
            }
        }
        let prompts = self
            .active_prefills
            .iter()
            .filter(|p| p.can_advance())
            .map(|p| self.decode_priority_or_high(p.work.sequence_id));
        let sections = self
            .active_section_ingests
            .iter()
            .filter(|s| s.can_advance())
            .map(|s| s.band);
        Some(Prospect {
            creeping: self
                .wave_prefill_members
                .iter()
                .map(WaveMember::seq_id)
                .collect(),
            top_in_flight: prompts.chain(sections).max_by_key(|&b| rank(b)),
            caps: self.group_caps(),
            live_lead: candidates
                .iter()
                .filter(|c| !c.section)
                .map(|c| c.band)
                .max_by_key(|&b| rank(b))
                .filter(|b| b.is_live()),
            candidates,
        })
    }

    /// The group `offer` — or, given none, the formation without it — runs
    /// in, drawn from `prospect` and composed at `target`.
    ///
    /// **A section the prompts leave no room for runs in the standalone
    /// section pass** (`build_section_batch`) — unless a prompt in flight is
    /// above the band that pass would run at, which holds it
    /// (`section_batch_band`). Sections already creeping in the group in
    /// flight are not in it.
    pub(super) fn group_in(
        &self,
        prospect: &Prospect,
        offer: Option<Offer>,
        target: usize,
    ) -> NextGroup {
        // **An offer on a slot already prefilling has no place beside it**: a
        // slot runs one prefill at a time, and a submission forking a slot
        // mid-prefill would read a state its turn has yet to finish changing.
        let offer = offer.filter(|o| !self.prefilling(o.who.seq));
        let group = self.form_group(
            &prospect.candidates,
            offer,
            Sections::Beside,
            target,
            prospect.caps,
        );
        match offer.filter(|o| o.who.section && group.offer_at.is_none()) {
            Some(section) if self.runs_alone(prospect, section.who.band) => {
                self.alone_in(prospect, offer, target)
            }
            _ => group,
        }
    }

    /// Whether a prefill or section in flight on `seq` can still advance —
    /// held, paused or neither.
    fn prefilling(&self, seq: usize) -> bool {
        self.active_prefills
            .iter()
            .any(|p| p.work.sequence_id.0 == seq && p.can_advance())
            || self
                .active_section_ingests
                .iter()
                .any(|s| s.sequence_id.0 == seq && s.can_advance())
    }

    /// The formation `like` came from — the creep group, or the standalone
    /// section pass — drawn from `prospect` without any offer: what its rows
    /// are measured against to say how far an offer widens it.
    pub(super) fn without_in(
        &self,
        prospect: &Prospect,
        like: &NextGroup,
        target: usize,
    ) -> NextGroup {
        if like.alone {
            self.alone_in(prospect, None, target)
        } else {
            self.form_group(
                &prospect.candidates,
                None,
                Sections::Beside,
                target,
                prospect.caps,
            )
        }
    }

    /// The standalone section pass `prospect`'s sections — and `offer`, when
    /// given — form, the sections creeping in the group in flight left out.
    fn alone_in(&self, prospect: &Prospect, offer: Option<Offer>, target: usize) -> NextGroup {
        let alone: Vec<Candidate> = prospect
            .candidates
            .iter()
            .copied()
            .filter(|c| c.section && !prospect.creeping.contains(&c.seq))
            .collect();
        self.form_group(&alone, offer, Sections::Alone, target, prospect.caps)
    }

    /// Whether a standalone section pass runs beside `prospect` for a section
    /// at `band`: none does while anything in flight — a prompt, or a section
    /// creeping in the group in flight — is above the highest band of the
    /// sections it would carry.
    fn runs_alone(&self, prospect: &Prospect, band: DecodePriority) -> bool {
        let top = prospect
            .candidates
            .iter()
            .filter(|c| c.section && !prospect.creeping.contains(&c.seq))
            .map(|c| c.band)
            .chain([band])
            .max_by_key(|&b| rank(b))
            .unwrap_or(band);
        prospect
            .top_in_flight
            .is_none_or(|above| rank(above) <= rank(top))
    }

    /// The gang a group forms with (`docs/prefill_gangs.md`): the live one —
    /// those of its members still prefilling — or, in a `background` group with
    /// none live, a new one from the long prompts among `entries`, the highest
    /// band's in arrival order, as wide as [`Self::gang_width`] allows. A long
    /// prompt is one a pass cannot finish. How it was formed, beside it, when
    /// it is formed here.
    fn gang_for(&self, background: bool, entries: &[Entry]) -> (Vec<usize>, Option<GangFormed>) {
        let live: Vec<usize> = self
            .prefill_gang
            .iter()
            .copied()
            .filter(|&s| {
                self.active_prefills
                    .iter()
                    .any(|p| p.work.sequence_id.0 == s && p.can_advance())
            })
            .collect();
        if !live.is_empty() || !background {
            return (live, None);
        }
        let pass = self.max_prefill_pass_tokens;
        let longs: Vec<(usize, usize)> = entries
            .iter()
            .filter(|e| !e.who.section && e.who.remaining > pass)
            .map(|e| (e.who.seq, rank(e.who.band)))
            .collect();
        let width = self.gang_width(longs.len());
        let formed = gang::form_gang(&longs, width);
        let how = (!formed.is_empty()).then_some(GangFormed {
            width,
            longs: longs.len(),
        });
        (formed, how)
    }

    /// How wide a gang of up to `longs` prompts may run: the widest whose decode
    /// step, by the learned decode model at the weight zone's residency, costs
    /// no more than twice a lone decode's ([`gang::gang_width`]). `1` — no gang
    /// — while the decode model is unlearned.
    fn gang_width(&self, longs: usize) -> usize {
        let Some(rate) = self.wave_rate.as_ref().filter(|r| r.decode_learned()) else {
            return 1;
        };
        let Some(zone) = effective_weight_zone_bytes(0) else {
            return 1;
        };
        let per = rate.experts_per_decode(0);
        gang::gang_width(longs, |n| {
            rate.decode_step_secs(n.saturating_mul(per), zone)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::prefill::wave_chunk_tests::dialogue_prefill;
    use super::super::tests::make_test_scheduler;
    use super::super::{SequenceId, WaveMember};
    use super::{Candidate, Offer, Prospect, Sections};
    use crate::projection::DecodePriority::{self, High, Low, Normal};

    /// A prompt of `remaining` rows on `seq` at `band`, at position 0.
    fn prompt(seq: usize, band: DecodePriority, remaining: usize) -> Candidate {
        Candidate {
            seq,
            band,
            remaining,
            depth: 0,
            section: false,
        }
    }

    fn prefill(seq_id: usize, advance: usize) -> WaveMember {
        WaveMember::Prefill { seq_id, advance }
    }

    /// **The highest band leads, then the prompt nearest its first token, and
    /// a live band's group carries its own band only.**
    #[test]
    fn the_highest_band_leads_nearest_first_and_carries_its_own() {
        let (s, _tx) = make_test_scheduler();
        let group = s.form_group(
            &[
                prompt(1, High, 300),
                prompt(2, Low, 50),
                prompt(3, High, 100),
            ],
            None,
            Sections::Out,
            s.composition_target(),
            s.group_caps(),
        );
        assert_eq!(group.wave_members(), vec![prefill(3, 100), prefill(1, 300)]);
    }

    /// **Below `High` the bands mix**, the higher still leading.
    #[test]
    fn below_high_the_bands_mix() {
        let (s, _tx) = make_test_scheduler();
        let group = s.form_group(
            &[prompt(1, Low, 50), prompt(2, Normal, 200)],
            None,
            Sections::Out,
            s.composition_target(),
            s.group_caps(),
        );
        assert_eq!(group.wave_members(), vec![prefill(2, 200), prefill(1, 50)]);
    }

    /// **An offer takes its place by the same rules, at no more than its cap.**
    /// Nearer its first token than the member beside it, it leads; capped, it
    /// advances by the cap and the member takes what it leaves.
    #[test]
    fn an_offer_takes_its_place_at_no_more_than_its_cap() {
        let (s, _tx) = make_test_scheduler();
        let target = s.composition_target();
        let budget = s.prefill_group_budget_at(400 + 300, target);
        let offer = Offer {
            who: prompt(9, High, 300),
            cap: Some(64),
        };
        let group = s.form_group(
            &[prompt(1, High, 400)],
            Some(offer),
            Sections::Out,
            target,
            s.group_caps(),
        );
        assert_eq!(group.offer_at, Some(0), "it leads");
        assert_eq!(
            group.wave_members(),
            vec![prefill(9, 64), prefill(1, 400.min(budget - 64))]
        );
        assert_eq!(group.members.len(), 2);
    }

    /// **A standalone section pass is held by anything in flight above it** —
    /// a prompt or a creeping section, as `section_batch_band` holds it.
    #[test]
    fn a_standalone_section_pass_is_held_by_work_above_it() {
        let (s, _tx) = make_test_scheduler();
        let prospect = |top| Prospect {
            candidates: Vec::new(),
            creeping: HashSet::new(),
            top_in_flight: top,
            caps: s.group_caps(),
            live_lead: None,
        };
        assert!(!s.runs_alone(&prospect(Some(High)), Low));
        assert!(s.runs_alone(&prospect(Some(High)), High));
        assert!(s.runs_alone(&prospect(None), Low));
    }

    /// **A member that re-enters after its sweep is staged where the sweep
    /// left it**: its rows left and its depth both move by its advance.
    #[test]
    fn a_member_reenters_where_its_sweep_left_it() {
        let (mut s, _tx) = make_test_scheduler();
        let seq = SequenceId(s.session.create_sequence().expect("create"));
        s.active_prefills
            .push(dialogue_prefill(seq, vec![1; 5_000]));
        let at = s.session.sequence_offset(seq.0).unwrap_or(0);
        let moved = s.formation_candidates(&HashSet::new(), |_| 1_200);
        assert_eq!(moved.len(), 1);
        assert_eq!(moved[0].remaining, 3_800);
        assert_eq!(moved[0].depth, at + 1_200);
        // A member whose sweep finishes it is no candidate.
        assert!(s
            .formation_candidates(&HashSet::new(), |_| 5_000)
            .is_empty());
    }

    /// **An offer on a slot already prefilling has no place beside it.**
    #[test]
    fn an_offer_on_a_prefilling_slot_has_no_place() {
        let (mut s, _tx) = make_test_scheduler();
        let seq = SequenceId(s.session.create_sequence().expect("create"));
        s.slot_bands.insert(seq, High);
        s.active_prefills.push(dialogue_prefill(seq, vec![1; 100]));
        let prospect = s.prospect(High).expect("nothing is paused");
        let offer = Offer {
            who: prompt(seq.0, High, 64),
            cap: None,
        };
        let group = s.group_in(&prospect, Some(offer), s.composition_target());
        assert!(group.offer_at.is_none());
        assert_eq!(group.wave_members(), vec![prefill(seq.0, 100)]);
    }
}
