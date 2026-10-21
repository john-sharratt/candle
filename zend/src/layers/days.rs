//! The day units a day layer reads: each closed UTC day of dialogue, keyed by
//! exactly the turns it holds.
//!
//! A day is closed once it is over — today's dialogue is still being written,
//! and a history of half a day would be stale by evening. Its key covers every
//! turn written that day, by conversation and index, so a conversation deleted
//! later, or turns replicated in late from another member, read the day
//! again; a day nothing changed keeps its history.

use sha2::{Digest, Sha256};

use candle_conversation::projection::{TimelineId, TurnIndex};

/// One closed day of dialogue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaySource {
    /// `YYYY-MM-DD`, UTC.
    pub day: String,
    /// What the day holds, digested: its key.
    pub key: String,
    /// Each conversation with turns that day, and those turns, in order.
    pub turns: Vec<(TimelineId, Vec<TurnIndex>)>,
}

const MS_PER_DAY: u64 = 86_400_000;

/// The UTC date of the wall-clock instant `ms`, as `YYYY-MM-DD`.
pub fn day_of(ms: u64) -> String {
    let (y, m, d) = civil_from_days((ms / MS_PER_DAY) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Every closed day in `convs` — each conversation's turns with the instant
/// each was written — before `today`, oldest first.
pub fn closed_days(convs: &[(TimelineId, Vec<(TurnIndex, u64)>)], today: &str) -> Vec<DaySource> {
    let mut by_day: std::collections::BTreeMap<String, Vec<(TimelineId, Vec<TurnIndex>)>> =
        std::collections::BTreeMap::new();
    for (timeline, turns) in convs {
        let mut per_day: std::collections::BTreeMap<String, Vec<TurnIndex>> =
            std::collections::BTreeMap::new();
        for (turn, ms) in turns {
            per_day.entry(day_of(*ms)).or_default().push(*turn);
        }
        for (day, mut turns) in per_day {
            if day.as_str() >= today {
                continue;
            }
            turns.sort_by_key(|t| t.0);
            by_day.entry(day).or_default().push((*timeline, turns));
        }
    }
    by_day
        .into_iter()
        .map(|(day, mut turns)| {
            turns.sort_by_key(|(tl, _)| tl.raw());
            let mut h = Sha256::new();
            h.update(day.as_bytes());
            for (tl, idx) in &turns {
                h.update(format!("\n{}:", tl.raw()).as_bytes());
                for i in idx {
                    h.update(format!("{},", i.0).as_bytes());
                }
            }
            let hex: String = h.finalize()[..16]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            DaySource {
                key: format!("day1:{day}:{hex}"),
                day,
                turns,
            }
        })
        .collect()
}

/// Days since 1970-01-01 to `(year, month, day)` — the proleptic Gregorian
/// calendar, exact for every date (Howard Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    const DAY: u64 = MS_PER_DAY;
    /// 2026-10-20T00:00:00Z.
    const OCT_20: u64 = 1_792_454_400_000;

    #[test]
    fn instants_fall_on_their_utc_dates() {
        assert_eq!(day_of(0), "1970-01-01");
        assert_eq!(day_of(OCT_20), "2026-10-20");
        assert_eq!(day_of(OCT_20 - 1), "2026-10-19");
        assert_eq!(day_of(951_782_400_000), "2000-02-29");
    }

    /// Today is not a unit; a conversation spanning two days is in both, with
    /// the turns it wrote on each.
    #[test]
    fn closed_days_hold_the_turns_written_on_them() {
        let convs = vec![
            (
                tl(2),
                vec![(TurnIndex(0), OCT_20 + 5), (TurnIndex(1), OCT_20 + DAY + 5)],
            ),
            (
                tl(1),
                vec![(TurnIndex(0), OCT_20 + 9), (TurnIndex(1), OCT_20 + 2 * DAY)],
            ),
        ];
        let days = closed_days(&convs, "2026-10-22");
        assert_eq!(days.len(), 2);
        assert_eq!(days[0].day, "2026-10-20");
        assert_eq!(
            days[0].turns,
            vec![(tl(1), vec![TurnIndex(0)]), (tl(2), vec![TurnIndex(0)])]
        );
        assert_eq!(days[1].day, "2026-10-21");
        assert_eq!(days[1].turns, vec![(tl(2), vec![TurnIndex(1)])]);
    }

    /// A day's key moves with the turns it holds, and only with them.
    #[test]
    fn a_days_key_moves_with_its_turns() {
        let one = vec![(tl(1), vec![(TurnIndex(0), OCT_20)])];
        let two = vec![(
            tl(1),
            vec![(TurnIndex(0), OCT_20), (TurnIndex(1), OCT_20 + 7)],
        )];
        let a = &closed_days(&one, "2026-10-21")[0];
        let b = &closed_days(&two, "2026-10-21")[0];
        assert_ne!(a.key, b.key);
        assert_eq!(a.key, closed_days(&one, "2026-10-21")[0].key);
        assert!(a.key.starts_with("day1:2026-10-20:"));
    }
}
