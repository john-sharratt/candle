//! What a window's `allocation` band was spent on: the admission passes'
//! parts, from the VRAM-pressure relief they open with to the frontier pass a
//! pass held on its floor asks for.
//!
//! The phase chart draws admission as one band; which part costs it is what
//! making it faster needs, so the window's close logs this beside it.

use std::time::Duration;

/// The parts of an admission pass, in the order [`AdmissionSplit::part`] takes
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AdmitPart {
    /// The pressure check and, under pressure, the relief it runs.
    Relief,
    /// Arming the planner and pricing the ground (`AdmitPass::new`).
    Setup,
    /// The rate model judging each offer (`admit::fill`), whole — `price` and
    /// `admit` below are parts of it.
    Fill,
    /// Of `fill`: composing and pricing the cohort for each offer
    /// (`AdmitPass::offer_tier`).
    Price,
    /// Of `fill`: beginning each admitted offer — its view and store carved —
    /// and recomposing the cohort after it (`Ground::admit`).
    Admit,
    /// Of `admit`: the begin alone — the case carved, the view's state forked,
    /// the carried state claimed, the turn sealed under its conversation and
    /// its prefill started (`Scheduler::begin_admitted`).
    Begin,
    /// Of `fill`: finding each offer and pricing it (`Ground::peek`) — `price`
    /// is part of it.
    Peek,
    /// Of `fill`: reading the residency each offer is judged against.
    Residency,
    /// Of `fill`: the rate model's verdict on each offer.
    Judge,
    /// Handing back what the pass did not offer (`AdmitPass::conclude`).
    Conclude,
    /// Lowering the frontier for a pass held on its floor over holes.
    Frontier,
}

const PARTS: [AdmitPart; 11] = [
    AdmitPart::Relief,
    AdmitPart::Setup,
    AdmitPart::Fill,
    AdmitPart::Peek,
    AdmitPart::Price,
    AdmitPart::Residency,
    AdmitPart::Judge,
    AdmitPart::Admit,
    AdmitPart::Begin,
    AdmitPart::Conclude,
    AdmitPart::Frontier,
];

impl AdmitPart {
    fn label(self) -> &'static str {
        match self {
            AdmitPart::Relief => "relief",
            AdmitPart::Setup => "setup",
            AdmitPart::Fill => "fill",
            AdmitPart::Price => "price",
            AdmitPart::Admit => "admit",
            AdmitPart::Begin => "begin",
            AdmitPart::Peek => "peek",
            AdmitPart::Residency => "residency",
            AdmitPart::Judge => "judge",
            AdmitPart::Conclude => "conclude",
            AdmitPart::Frontier => "frontier",
        }
    }

    fn index(self) -> usize {
        PARTS
            .iter()
            .position(|&p| p == self)
            .expect("every part is listed")
    }
}

/// One window's admission, by part, in microseconds, and how many passes ran.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct AdmissionSplit {
    pub part_us: [u64; PARTS.len()],
    pub passes: u32,
}

impl AdmissionSplit {
    /// File `took` of a pass as `part`.
    pub fn part(&mut self, part: AdmitPart, took: Duration) {
        self.part_us[part.index()] += took.as_micros() as u64;
    }

    /// One admission pass ran.
    pub fn pass(&mut self) {
        self.passes += 1;
    }

    /// The parts as `name=ms` pairs, for the log.
    pub fn parts_ms(&self) -> String {
        PARTS
            .iter()
            .zip(&self.part_us)
            .map(|(p, us)| format!("{}={}", p.label(), us / 1000))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each part sums its own time and reads back by name, in order.
    #[test]
    fn each_part_sums_its_time_and_reads_back_by_name() {
        let ms = Duration::from_millis;
        let mut s = AdmissionSplit::default();
        s.part(AdmitPart::Relief, ms(12));
        s.part(AdmitPart::Fill, ms(3));
        s.part(AdmitPart::Fill, ms(4));
        s.part(AdmitPart::Frontier, ms(40));
        s.pass();
        s.pass();
        assert_eq!(s.passes, 2);
        assert_eq!(
            s.parts_ms(),
            "relief=12 setup=0 fill=7 peek=0 price=0 residency=0 judge=0 admit=0 begin=0 \
             conclude=0 frontier=40"
        );
    }
}
