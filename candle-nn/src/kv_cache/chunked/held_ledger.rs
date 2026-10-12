//! Who holds the held ground: a per-call-site ledger of what each holder has
//! taken, given back and been refused.
//!
//! The ground's own totals — used, peak, largest free — say *that* it is full,
//! never *who* filled it, and a buffer the ground refuses goes to the CUDA pool
//! and from there to the driver. On a card with no free memory that refusal is
//! a `CUDA_ERROR_OUT_OF_MEMORY` in the middle of a forward, named by nothing but
//! the forward it killed. Every range is filed here under the source location
//! that asked for it (`Held` and the `held_tensor` helpers track their caller),
//! so the question "what is the ground holding" has an answer by line.
//!
//! Pure host bookkeeping over byte counts: every rule — live bytes per site, a
//! site's own high-water mark, refusals counted per site, the first refusal
//! flagged so it can be reported once — is tested without a device.

use std::collections::HashMap;
use std::panic::Location;

/// One holder's account.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Tally {
    live_bytes: usize,
    live_count: usize,
    peak_bytes: usize,
    takes: u64,
    declines: u64,
    declined_bytes: u64,
    largest_decline: usize,
}

/// One holder's account, as a report reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeldSite {
    /// `file:line:column` of the code that asked.
    pub site: String,
    /// Bytes it holds now, alignment padding included.
    pub live_bytes: usize,
    /// Ranges it holds now.
    pub live_count: usize,
    /// The most it has held at once.
    pub peak_bytes: usize,
    /// Ranges it has been given, in all.
    pub takes: u64,
    /// Requests the ground could not hold — each one went to the CUDA pool.
    pub declines: u64,
    pub declined_bytes: u64,
    /// The widest request refused.
    pub largest_decline: usize,
}

/// Per-site accounts for one ground.
#[derive(Debug, Default)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) struct HeldLedger {
    sites: HashMap<&'static Location<'static>, Tally>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl HeldLedger {
    /// `site` was given a range of `bytes`.
    pub(crate) fn take(&mut self, site: &'static Location<'static>, bytes: usize) {
        let t = self.sites.entry(site).or_default();
        t.live_bytes += bytes;
        t.live_count += 1;
        t.takes += 1;
        t.peak_bytes = t.peak_bytes.max(t.live_bytes);
    }

    /// `site` gave back a range of `bytes`.
    ///
    /// # Panics
    ///
    /// When `site` holds less than that — a range given back under a site
    /// other than the one it was taken for, which would leave every account
    /// after it wrong.
    pub(crate) fn release(&mut self, site: &'static Location<'static>, bytes: usize) {
        let t = self
            .sites
            .get_mut(site)
            .unwrap_or_else(|| panic!("held ledger: {site} gave back a range it never took"));
        assert!(
            t.live_bytes >= bytes && t.live_count > 0,
            "held ledger: {site} gave back {bytes} B holding {} B in {} ranges",
            t.live_bytes,
            t.live_count
        );
        t.live_bytes -= bytes;
        t.live_count -= 1;
    }

    /// The ground refused `site` a range of `bytes`. Answers whether this is
    /// the site's first refusal — the one worth reporting in full.
    pub(crate) fn decline(&mut self, site: &'static Location<'static>, bytes: usize) -> bool {
        let t = self.sites.entry(site).or_default();
        t.declines += 1;
        t.declined_bytes += bytes as u64;
        t.largest_decline = t.largest_decline.max(bytes);
        t.declines == 1
    }

    /// Every site that has taken or been refused anything, the largest live
    /// holder first; ties by site, so the order is stable between readings.
    pub(crate) fn sites(&self) -> Vec<HeldSite> {
        let mut rows: Vec<HeldSite> = self
            .sites
            .iter()
            .map(|(site, t)| HeldSite {
                site: site.to_string(),
                live_bytes: t.live_bytes,
                live_count: t.live_count,
                peak_bytes: t.peak_bytes,
                takes: t.takes,
                declines: t.declines,
                declined_bytes: t.declined_bytes,
                largest_decline: t.largest_decline,
            })
            .collect();
        rows.sort_by(|a, b| {
            b.live_bytes
                .cmp(&a.live_bytes)
                .then_with(|| a.site.cmp(&b.site))
        });
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn here() -> &'static Location<'static> {
        Location::caller()
    }

    /// A site's live bytes rise and fall with its ranges, and its peak keeps
    /// the high-water mark.
    #[test]
    fn a_site_accounts_its_live_ranges_and_its_peak() {
        let a = here();
        let mut l = HeldLedger::default();
        l.take(a, 4096);
        l.take(a, 1024);
        l.release(a, 4096);
        let row = &l.sites()[0];
        assert_eq!(row.site, a.to_string());
        assert_eq!(
            (row.live_bytes, row.live_count, row.peak_bytes, row.takes),
            (1024, 1, 5120, 2)
        );
    }

    /// The largest live holder comes first, whoever took first.
    #[test]
    fn sites_are_listed_largest_holder_first() {
        let small = here();
        let big = here();
        let mut l = HeldLedger::default();
        l.take(small, 256);
        l.take(big, 8192);
        let rows = l.sites();
        assert_eq!(
            rows.iter().map(|r| r.live_bytes).collect::<Vec<_>>(),
            vec![8192, 256]
        );
        assert_eq!(rows[0].site, big.to_string());
    }

    /// A refusal is counted on the refused site — including one that never
    /// held anything — and only its first is flagged for a full report.
    #[test]
    fn a_refusal_is_counted_and_its_first_is_flagged() {
        let a = here();
        let mut l = HeldLedger::default();
        assert!(l.decline(a, 64 << 20));
        assert!(!l.decline(a, 128 << 20));
        let row = &l.sites()[0];
        assert_eq!(
            (
                row.live_bytes,
                row.declines,
                row.declined_bytes,
                row.largest_decline
            ),
            (0, 2, 192 << 20, 128 << 20)
        );
    }

    /// Giving back more than a site holds is two holders' books crossed, and
    /// is refused loudly.
    #[test]
    #[should_panic(expected = "gave back")]
    fn giving_back_more_than_held_is_refused() {
        let a = here();
        let mut l = HeldLedger::default();
        l.take(a, 256);
        l.release(a, 512);
    }
}
