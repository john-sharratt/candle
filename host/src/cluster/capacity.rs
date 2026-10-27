//! A member's `CAPACITY` (`docs/replicated_substrate.md` §5.3): what `Best`
//! ranks it by, measured every second.
//!
//! - **`power`** is the hour's peak aggregate decode rate. Before the engine
//!   has decoded anything it is the card's memory in MiB over 100 — a ranking
//!   of last resort, replaced by the first measurement.
//! - **`idle`** says the scheduler can admit now: its newest wave left no
//!   prefill backlog and used less than its budget. A scheduler that has run no
//!   wave for [`QUIET`] has nothing in hand, and is idle.
//! - **`slots_free`** is how many more residences of the newest wave's average
//!   size fit in what its budget left.
//! - **`prefill_tok_s`**, **`decode_step_next_us`**, **`decode_step_wide_us`**
//!   and **`prefill_tok_s_wide`** are what the scheduler has learned a turn and
//!   a burst of them cost it —
//!   the gateway places a turn where it would finish soonest
//!   (`docs/cooperative_admission.md` §2.1); zero until it has learned them.
//! - **`guest_reach`** is the side of the largest square its card draws, as
//!   its engine stated it: a gateway places a draw only where it is reached.

use cluster::frame::{Capacity, MemberState};
use cluster::machine::Machine;

use crate::telemetry::CardReading;

/// How long without a wave reads as a scheduler with nothing in hand.
pub const QUIET: f64 = 5.0;

/// The scheduler's newest wave, as capacity reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wave {
    /// Seconds since it ran.
    pub age_s: f64,
    pub backlog: u64,
    pub decodes: u32,
    pub slots: u32,
    pub used_mib: u64,
    pub budget_mib: u64,
}

/// What the scheduler has learned a turn costs it now, as capacity reads it
/// (`docs/cooperative_admission.md` §2.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Prefill rate at the width the scheduler runs, tokens a second.
    pub prefill_tok_s: f64,
    /// The decode step with one row more than it decodes, milliseconds.
    pub decode_step_next_ms: f64,
    /// The decode step with `PLACEMENT_WIDE` rows more than it decodes,
    /// milliseconds: how the step grows as a burst lands.
    pub decode_step_wide_ms: f64,
    /// The prefill rate with `PLACEMENT_WIDE` rows more than it decodes,
    /// tokens a second: how prefill slows as a burst lands.
    pub prefill_tok_s_wide: f64,
    /// The mean prompt of the turns the scheduler has begun, and the mean reply
    /// such a turn decodes, in tokens; zero until learned.
    pub prompt_tokens: f64,
    pub reply_tokens: f64,
    /// Tokens a decoding sequence commits a step, above 1 where it speculates;
    /// zero until learned.
    pub tokens_per_step: f64,
    /// Prompt tokens a placed turn would queue behind: those waiting at its
    /// own priority. Background ingest is not among them — a placed turn goes
    /// ahead of it.
    pub high_backlog_tokens: u64,
}

/// What `CAPACITY`'s state says: faulted while the device cannot take a turn
/// (`candle::gpu_poison::device_failing`) — a poisoned context, or every
/// operation out of memory — whatever the tenant reports, so the gateways
/// route it nothing; otherwise ready once the tenant is.
pub fn member_state(tenant_ready: bool, device_failing: bool) -> MemberState {
    if device_failing {
        MemberState::Fault
    } else if tenant_ready {
        MemberState::Ready
    } else {
        MemberState::Loading
    }
}

pub fn capacity(
    state: MemberState,
    card: CardReading,
    wave: Option<Wave>,
    rates: Option<Rates>,
    machine: Machine,
    guest_reach: u32,
) -> Capacity {
    // What the scheduler learned stands while it is quiet: a rate is the
    // card's, not the window's. Only a rate it has actually measured is said.
    let rates = rates.filter(|r| r.prefill_tok_s >= 1.0 && r.decode_step_next_ms > 0.0);
    let power = match card.peak_decode_tps {
        Some(tps) if tps >= 1.0 => tps.round() as u32,
        _ => card.vram_total_mib.map_or(0, |m| (m / 100) as u32),
    };
    let recent = wave.filter(|w| w.age_s < QUIET);
    let slots_free = wave.map_or(0, |w| {
        let room = w.budget_mib.saturating_sub(w.used_mib);
        match w.slots {
            0 => u32::from(room > 0),
            n => {
                let each = (w.used_mib / u64::from(n)).max(1);
                (room / each).min(u64::from(u32::MAX)) as u32
            }
        }
    });
    Capacity {
        state,
        power,
        idle: recent.is_none_or(|w| w.backlog == 0 && w.used_mib < w.budget_mib),
        slots_free,
        decodes: recent.map_or(0, |w| w.decodes),
        // What a placed turn would wait behind, which is what `Best` divides
        // by the prefill rate — not the whole pending total, whose background
        // ingest a placed turn goes ahead of. `idle` reads the total: it says
        // whether the scheduler can admit now.
        prefill_backlog: rates.map_or(0, |r| r.high_backlog_tokens.min(u64::from(u32::MAX)) as u32),
        vram_free_mib: card
            .vram_free_mib
            .map_or(0, |m| m.min(u64::from(u32::MAX)) as u32),
        cores: machine.cores,
        load_milli: machine.load_milli,
        os: machine.os,
        mem_free_mib: machine.mem_free_mib,
        jobs: 0,
        toolchains: Vec::new(),
        // A model member runs no jobs; its host's agent, when it has one,
        // reports its own capacity.
        low_priority: false,
        prefill_tok_s: rates.map_or(0, |r| {
            r.prefill_tok_s.round().min(f64::from(u32::MAX)) as u32
        }),
        decode_step_next_us: rates.map_or(0, |r| micros(r.decode_step_next_ms)),
        // Never below the one-more-row step: a wide step no slower is flat,
        // and the gateway reads it so.
        decode_step_wide_us: rates.map_or(0, |r| {
            micros(r.decode_step_wide_ms.max(r.decode_step_next_ms))
        }),
        // Never above the one-more rate: a wide rate no slower is flat, and the
        // gateway reads it so.
        prefill_tok_s_wide: rates.map_or(0, |r| {
            r.prefill_tok_s_wide
                .min(r.prefill_tok_s)
                .max(1.0)
                .round()
                .min(f64::from(u32::MAX)) as u32
        }),
        prompt_tokens: rates.map_or(0, |r| tokens(r.prompt_tokens)),
        reply_tokens: rates.map_or(0, |r| tokens(r.reply_tokens)),
        tokens_per_step_milli: rates.map_or(0, |r| tokens(r.tokens_per_step * 1000.0)),
        guest_reach,
    }
}

/// A token count as the frame's whole tokens; zero for a figure not learned.
fn tokens(n: f64) -> u32 {
    if n.is_finite() && n > 0.0 {
        n.round().min(f64::from(u32::MAX)) as u32
    } else {
        0
    }
}

/// Milliseconds as the frame's whole microseconds.
fn micros(ms: f64) -> u32 {
    (ms * 1000.0).round().min(f64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> Machine {
        Machine {
            os: "linux".into(),
            cores: 16,
            load_milli: 1_250,
            mem_free_mib: 20_000,
        }
    }

    fn card(peak: Option<f64>) -> CardReading {
        CardReading {
            peak_decode_tps: peak,
            vram_total_mib: Some(73_728),
            vram_free_mib: Some(4_096),
        }
    }

    fn wave(age_s: f64, backlog: u64, used_mib: u64) -> Wave {
        Wave {
            age_s,
            backlog,
            decodes: 6,
            slots: 4,
            used_mib,
            budget_mib: 10_000,
        }
    }

    /// A failing device is a fault whatever the tenant says; a healthy one is
    /// ready once its tenant is.
    #[test]
    fn a_failing_device_is_a_fault() {
        assert_eq!(member_state(true, true), MemberState::Fault);
        assert_eq!(member_state(false, true), MemberState::Fault);
        assert_eq!(member_state(true, false), MemberState::Ready);
        assert_eq!(member_state(false, false), MemberState::Loading);
    }

    /// Power is the measured peak once there is one, and the card's memory over
    /// 100 before.
    #[test]
    fn power_is_the_peak_decode_rate_or_the_card() {
        let c = capacity(
            MemberState::Ready,
            card(Some(171.6)),
            None,
            None,
            machine(),
            0,
        );
        assert_eq!(c.power, 172);
        let c = capacity(MemberState::Loading, card(None), None, None, machine(), 0);
        assert_eq!((c.power, c.state), (737, MemberState::Loading));
        assert_eq!((c.vram_free_mib, c.cores, c.load_milli), (4_096, 16, 1_250));
        assert_eq!((c.os.as_str(), c.mem_free_mib), ("linux", 20_000));
    }

    /// **What its card draws is said as its engine stated it**, and nothing
    /// before the engine has.
    #[test]
    fn the_reach_is_said_as_the_engine_stated_it() {
        let c = capacity(MemberState::Ready, card(None), None, None, machine(), 2048);
        assert_eq!(c.guest_reach, 2048);
        let c = capacity(MemberState::Loading, card(None), None, None, machine(), 0);
        assert_eq!(c.guest_reach, 0);
    }

    /// A recent wave with a backlog or a full budget is not idle; one with
    /// neither is; a scheduler quiet for 5 s is idle whatever its last wave
    /// said. Free slots are the room left over the average residence. The
    /// wave's backlog is everything pending, background ingest included: it
    /// decides idleness, never the backlog a placed turn is said to wait on.
    #[test]
    fn idle_reads_the_newest_wave() {
        let at = |w: Wave| capacity(MemberState::Ready, card(None), Some(w), None, machine(), 0);
        let busy = at(wave(1.0, 3, 8_000));
        assert_eq!(
            (busy.idle, busy.prefill_backlog, busy.decodes),
            (false, 0, 6)
        );
        // 8,000 MiB over 4 residences is 2,000 each; 2,000 left is one more.
        assert_eq!(busy.slots_free, 1);
        let full = at(wave(1.0, 0, 10_000));
        assert_eq!((full.idle, full.slots_free), (false, 0));
        let free = at(wave(1.0, 0, 4_000));
        assert_eq!((free.idle, free.slots_free), (true, 6));
        let quiet = at(wave(9.0, 3, 10_000));
        assert_eq!(
            (quiet.idle, quiet.decodes, quiet.prefill_backlog),
            (true, 0, 0)
        );
        let never = capacity(MemberState::Ready, card(None), None, None, machine(), 0);
        assert_eq!((never.idle, never.slots_free), (true, 0));
    }

    /// **What a turn costs is said once it is learned**, in the frame's units —
    /// tokens a second, the one-more-row decode step in microseconds, and the
    /// prompt tokens a placed turn would queue behind — and stays said while
    /// the scheduler is quiet. Background ingest waiting in the wave is not
    /// what a placed turn waits on. Nothing learned, or a rate of nothing,
    /// says zero: the gateway does not place by it.
    #[test]
    fn learned_rates_are_said_in_the_frames_units() {
        let learned = Rates {
            prefill_tok_s: 2_302.6,
            decode_step_next_ms: 41.53,
            decode_step_wide_ms: 96.25,
            prefill_tok_s_wide: 640.4,
            prompt_tokens: 3_100.4,
            reply_tokens: 409.6,
            tokens_per_step: 2.6504,
            high_backlog_tokens: 6_144,
        };
        let c = capacity(
            MemberState::Ready,
            card(None),
            Some(wave(1.0, 91_818, 0)),
            Some(learned),
            machine(),
            0,
        );
        assert_eq!(
            (c.prefill_tok_s, c.decode_step_next_us, c.prefill_backlog),
            (2_303, 41_530, 6_144)
        );
        assert_eq!((c.decode_step_wide_us, c.prefill_tok_s_wide), (96_250, 640));
        assert_eq!((c.prompt_tokens, c.reply_tokens), (3_100, 410));
        assert_eq!(c.tokens_per_step_milli, 2_650);
        assert!(!c.idle, "the ingest still keeps it from admitting now");
        // A wide step learned below the one-more step, and a wide rate above
        // the one-more rate, are said as flat.
        let flat = capacity(
            MemberState::Ready,
            card(None),
            None,
            Some(Rates {
                decode_step_wide_ms: 30.0,
                prefill_tok_s_wide: 3_000.0,
                ..learned
            }),
            machine(),
            0,
        );
        assert_eq!(
            (flat.decode_step_next_us, flat.decode_step_wide_us),
            (41_530, 41_530)
        );
        assert_eq!(
            (flat.prefill_tok_s, flat.prefill_tok_s_wide),
            (2_303, 2_303)
        );
        let none = capacity(MemberState::Ready, card(None), None, None, machine(), 0);
        assert_eq!(
            (
                none.prefill_tok_s,
                none.decode_step_next_us,
                none.decode_step_wide_us,
                none.prefill_tok_s_wide
            ),
            (0, 0, 0, 0)
        );
        let unmeasured = Rates {
            prefill_tok_s: 0.0,
            decode_step_next_ms: 41.5,
            decode_step_wide_ms: 80.0,
            prefill_tok_s_wide: 0.0,
            prompt_tokens: 0.0,
            reply_tokens: 0.0,
            tokens_per_step: 0.0,
            high_backlog_tokens: 0,
        };
        let c = capacity(
            MemberState::Ready,
            card(None),
            None,
            Some(unmeasured),
            machine(),
            0,
        );
        assert_eq!((c.prefill_tok_s, c.decode_step_next_us), (0, 0));
    }

    /// A turn size is said in whole tokens, and one not learned — zero, or not
    /// a number — is said as zero, which the gateway reads as unknown.
    #[test]
    fn turn_sizes_are_whole_tokens_or_zero() {
        assert_eq!(tokens(2_047.6), 2_048);
        assert_eq!(tokens(0.0), 0);
        assert_eq!(tokens(-3.0), 0);
        assert_eq!(tokens(f64::NAN), 0);
    }
}
