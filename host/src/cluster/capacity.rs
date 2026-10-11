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

pub fn capacity(
    state: MemberState,
    card: CardReading,
    wave: Option<Wave>,
    machine: Machine,
) -> Capacity {
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
        prefill_backlog: recent.map_or(0, |w| w.backlog.min(u64::from(u32::MAX)) as u32),
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
    }
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

    /// Power is the measured peak once there is one, and the card's memory over
    /// 100 before.
    #[test]
    fn power_is_the_peak_decode_rate_or_the_card() {
        let c = capacity(MemberState::Ready, card(Some(171.6)), None, machine());
        assert_eq!(c.power, 172);
        let c = capacity(MemberState::Loading, card(None), None, machine());
        assert_eq!((c.power, c.state), (737, MemberState::Loading));
        assert_eq!((c.vram_free_mib, c.cores, c.load_milli), (4_096, 16, 1_250));
        assert_eq!((c.os.as_str(), c.mem_free_mib), ("linux", 20_000));
    }

    /// A recent wave with a backlog or a full budget is not idle; one with
    /// neither is; a scheduler quiet for 5 s is idle whatever its last wave
    /// said. Free slots are the room left over the average residence.
    #[test]
    fn idle_reads_the_newest_wave() {
        let at = |w: Wave| capacity(MemberState::Ready, card(None), Some(w), machine());
        let busy = at(wave(1.0, 3, 8_000));
        assert_eq!(
            (busy.idle, busy.prefill_backlog, busy.decodes),
            (false, 3, 6)
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
        let never = capacity(MemberState::Ready, card(None), None, machine());
        assert_eq!((never.idle, never.slots_free), (true, 0));
    }
}
