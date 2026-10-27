//! The performance series on one wall-clock grid (`docs/performance_series.md`
//! §2): every row of `/v1/telemetry`'s engine half and of `/v1/phases` is one
//! [`SLOT_MS`] slot, stamped with the slot's start in unix milliseconds — so
//! every member's rows line up, and a cluster fuses slot by slot.

use std::time::{SystemTime, UNIX_EPOCH};

/// A slot's width.
pub const SLOT_MS: u64 = 5_000;

/// Now, in unix milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// `items` — oldest first, each `age_s(item)` seconds before `now_ms` — by the
/// slot each ran in, oldest first. The slot still filling at `now_ms` is left
/// out, so a row once answered never changes.
pub fn by_slot<T>(items: &[T], now_ms: u64, age_s: impl Fn(&T) -> f64) -> Vec<(u64, Vec<&T>)> {
    let current = now_ms / SLOT_MS * SLOT_MS;
    let mut out: Vec<(u64, Vec<&T>)> = Vec::new();
    for item in items {
        let at = (now_ms as f64 - age_s(item) * 1000.0).max(0.0) as u64;
        let slot = at / SLOT_MS * SLOT_MS;
        if slot >= current {
            continue;
        }
        match out.last_mut() {
            Some((s, group)) if *s == slot => group.push(item),
            _ => out.push((slot, vec![item])),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows land in the slot they ran in, oldest first, and the slot still
    /// filling is left out.
    #[test]
    fn rows_group_by_the_slot_they_ran_in_and_the_filling_one_waits() {
        let ages = [17.5, 16.0, 12.9, 2.0];
        let slots: Vec<(u64, Vec<f64>)> = by_slot(&ages, 23_000, |a| *a)
            .into_iter()
            .map(|(s, g)| (s, g.into_iter().copied().collect()))
            .collect();
        assert_eq!(slots, vec![(5_000, vec![17.5, 16.0]), (10_000, vec![12.9])]);
    }
}
