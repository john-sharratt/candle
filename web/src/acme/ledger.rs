//! Every request this gateway has made of the CA, and the rules that refuse the
//! next one before the CA ever sees it.
//!
//! Let's Encrypt's limits are generous for a gateway that behaves, and
//! unforgiving of one in a loop: five duplicate certificates per name set per
//! week, five failed validations per name per hour, fifty certificates per
//! registered domain per week, ten accounts per IP per three hours. A bug that
//! re-orders on every restart reaches one of those within the hour, and the
//! limit then outlives the fix by days.
//!
//! So the ledger is on disk — a restart cannot forget it — and its caps sit far
//! below the CA's:
//!
//! | rule | cap | the CA's limit it keeps clear of |
//! |---|---|---|
//! | gap between one site's orders | 1 hour | failed validations, 5 / name / hour |
//! | after a site's order fails | that site, 1 h, doubling, at most 24 h | failed validations; consecutive failures |
//! | issued per site | 2 / 7 days | duplicate certificates, 5 / set / week; 50 / domain / week |
//! | orders, all sites | 10 / 3 hours | new orders, 300 / account / 3 hours |
//! | after the CA says `rateLimited` | every site, 1 h, doubling, at most 24 h | whichever limit it was |
//! | accounts created | 1 / 24 hours | 10 / IP / 3 hours |
//!
//! A `rateLimited` answer holds every site back, not only the one that asked,
//! because the limit may be the account's. It backs off exponentially rather
//! than for a fixed day: Let's Encrypt answers a load-shedding "service busy"
//! with the same problem type as a real limit, and the first costs an hour
//! where the second keeps doubling until the CA stops refusing.
//!
//! Renewals the CA's renewal-information endpoint asked for are sent with
//! `replaces`, which Let's Encrypt exempts from its limits — the caps above
//! still apply to them, so a misread window cannot turn into a loop either.

use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const HOUR: u64 = 3600;
pub const DAY: u64 = 24 * HOUR;
pub const WEEK: u64 = 7 * DAY;

/// The least time between two orders for one site.
pub const SITE_GAP: u64 = HOUR;
/// The longest a run of failures holds a site back.
pub const FAILURE_BACKOFF_MAX: u64 = DAY;
/// Certificates issued per site per [`WEEK`].
pub const ISSUED_PER_SITE_PER_WEEK: usize = 2;
/// Orders across every site per [`ORDER_WINDOW`].
pub const ORDERS_PER_WINDOW: usize = 10;
pub const ORDER_WINDOW: u64 = 3 * HOUR;
/// Accounts created per [`DAY`].
pub const ACCOUNTS_PER_DAY: usize = 1;

/// How one order ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Issued,
    Failed,
    /// The CA refused with `rateLimited`.
    RateLimited,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub site: String,
    pub names: Vec<String>,
    /// Unix seconds.
    pub at: u64,
    pub outcome: Outcome,
}

/// Why an order may not be placed yet, and when it may.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wait {
    pub until: u64,
    pub why: &'static str,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    pub attempts: Vec<Attempt>,
    /// Unix seconds of each account created.
    pub accounts: Vec<u64>,
}

impl Ledger {
    /// Read the ledger, or start an empty one where there is none yet. A file
    /// that is present and unreadable is an error: an empty ledger in its place
    /// would forget exactly the history the caps are made of.
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", path.display())),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        super::store::write_atomic(path, &serde_json::to_vec_pretty(self)?)
    }

    /// Record an order's end, and drop history older than any rule reads.
    pub fn record(&mut self, attempt: Attempt) {
        let now = attempt.at;
        self.attempts.push(attempt);
        self.attempts
            .retain(|a| now.saturating_sub(a.at) <= WEEK + DAY);
        self.accounts.retain(|&t| now.saturating_sub(t) <= WEEK);
    }

    pub fn record_account(&mut self, now: u64) {
        self.accounts.push(now);
    }

    /// Whether an account may be created now.
    pub fn may_create_account(&self, now: u64) -> Result<(), Wait> {
        let recent: Vec<u64> = self
            .accounts
            .iter()
            .copied()
            .filter(|&t| now.saturating_sub(t) < DAY)
            .collect();
        if recent.len() >= ACCOUNTS_PER_DAY {
            let oldest = recent.into_iter().min().unwrap_or(now);
            return Err(Wait {
                until: oldest + DAY,
                why: "an account was already created today",
            });
        }
        Ok(())
    }

    /// Whether `site` may place an order now. The latest of every rule that
    /// applies is the answer, so the wait it reports is one the next check will
    /// not refuse for a different reason.
    pub fn may_order(&self, site: &str, now: u64) -> Result<(), Wait> {
        let mut waits: Vec<Wait> = Vec::new();

        // The run of refusals since the CA last took an order, from any site:
        // any other outcome means it was accepting orders again.
        let refusals: Vec<u64> = self
            .attempts
            .iter()
            .rev()
            .take_while(|a| a.outcome == Outcome::RateLimited)
            .map(|a| a.at)
            .collect();
        if let Some(&last) = refusals.first() {
            waits.push(Wait {
                until: last + failure_backoff(refusals.len()),
                why: "the CA reported a rate limit",
            });
        }

        let window: Vec<u64> = self
            .attempts
            .iter()
            .filter(|a| now.saturating_sub(a.at) < ORDER_WINDOW)
            .map(|a| a.at)
            .collect();
        if window.len() >= ORDERS_PER_WINDOW {
            waits.push(Wait {
                until: window.iter().min().copied().unwrap_or(now) + ORDER_WINDOW,
                why: "the gateway-wide order cap is reached",
            });
        }

        let mine: Vec<&Attempt> = self.attempts.iter().filter(|a| a.site == site).collect();
        if let Some(last) = mine.iter().map(|a| a.at).max() {
            waits.push(Wait {
                until: last + SITE_GAP,
                why: "this site ordered within the hour",
            });
        }

        // The run of failures since this site last succeeded.
        let failures: Vec<u64> = mine
            .iter()
            .rev()
            .take_while(|a| a.outcome != Outcome::Issued)
            .map(|a| a.at)
            .collect();
        if let Some(&last) = failures.first() {
            waits.push(Wait {
                until: last + failure_backoff(failures.len()),
                why: "this site's last order failed",
            });
        }

        let issued: Vec<u64> = mine
            .iter()
            .filter(|a| a.outcome == Outcome::Issued && now.saturating_sub(a.at) < WEEK)
            .map(|a| a.at)
            .collect();
        if issued.len() >= ISSUED_PER_SITE_PER_WEEK {
            waits.push(Wait {
                until: issued.iter().min().copied().unwrap_or(now) + WEEK,
                why: "this site was issued its weekly allowance",
            });
        }

        match waits
            .into_iter()
            .filter(|w| w.until > now)
            .max_by_key(|w| w.until)
        {
            Some(w) => Err(w),
            None => Ok(()),
        }
    }
}

/// Hold-off after `n` consecutive failures: an hour, doubling, at most a day.
pub fn failure_backoff(n: usize) -> u64 {
    let shift = n.saturating_sub(1).min(16) as u32;
    (HOUR << shift).min(FAILURE_BACKOFF_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_800_000_000;

    fn attempt(site: &str, at: u64, outcome: Outcome) -> Attempt {
        Attempt {
            site: site.into(),
            names: vec![format!("{site}.example.net")],
            at,
            outcome,
        }
    }

    #[test]
    fn a_fresh_ledger_allows_one_order_and_one_account() {
        let l = Ledger::default();
        assert_eq!(l.may_order("tokera", T0), Ok(()));
        assert_eq!(l.may_create_account(T0), Ok(()));
    }

    #[test]
    fn a_site_waits_an_hour_between_orders() {
        let mut l = Ledger::default();
        l.record(attempt("tokera", T0, Outcome::Issued));
        assert_eq!(
            l.may_order("tokera", T0 + HOUR - 1),
            Err(Wait {
                until: T0 + HOUR,
                why: "this site ordered within the hour"
            })
        );
        assert_eq!(l.may_order("tokera", T0 + HOUR), Ok(()));
        assert_eq!(
            l.may_order("npcd", T0 + 1),
            Ok(()),
            "another site is not held"
        );
    }

    #[test]
    fn failures_back_off_from_an_hour_doubling_to_a_day() {
        assert_eq!(failure_backoff(1), HOUR);
        assert_eq!(failure_backoff(2), 2 * HOUR);
        assert_eq!(failure_backoff(3), 4 * HOUR);
        assert_eq!(failure_backoff(5), 16 * HOUR);
        assert_eq!(failure_backoff(6), DAY);
        assert_eq!(failure_backoff(60), DAY);

        let mut l = Ledger::default();
        l.record(attempt("tokera", T0, Outcome::Failed));
        l.record(attempt("tokera", T0 + HOUR, Outcome::Failed));
        l.record(attempt("tokera", T0 + 3 * HOUR, Outcome::Failed));
        let w = l.may_order("tokera", T0 + 3 * HOUR + 1).unwrap_err();
        assert_eq!(w.until, T0 + 3 * HOUR + 4 * HOUR);
        assert_eq!(w.why, "this site's last order failed");
        assert_eq!(l.may_order("tokera", T0 + 7 * HOUR), Ok(()));
    }

    /// A success ends the run: the next failure starts the back-off again at
    /// an hour.
    #[test]
    fn a_success_resets_the_failure_run() {
        let mut l = Ledger::default();
        for k in 0..4 {
            l.record(attempt("tokera", T0 + k * DAY, Outcome::Failed));
        }
        l.record(attempt("tokera", T0 + 4 * DAY, Outcome::Issued));
        l.record(attempt("tokera", T0 + 5 * DAY, Outcome::Failed));
        assert_eq!(
            l.may_order("tokera", T0 + 5 * DAY + HOUR)
                .map_err(|w| w.until),
            Ok(())
        );
    }

    #[test]
    fn a_site_is_issued_at_most_twice_a_week() {
        let mut l = Ledger::default();
        l.record(attempt("tokera", T0, Outcome::Issued));
        l.record(attempt("tokera", T0 + DAY, Outcome::Issued));
        assert_eq!(
            l.may_order("tokera", T0 + 2 * DAY),
            Err(Wait {
                until: T0 + WEEK,
                why: "this site was issued its weekly allowance"
            })
        );
        assert_eq!(l.may_order("tokera", T0 + WEEK), Ok(()));
    }

    #[test]
    fn the_whole_gateway_orders_at_most_ten_times_in_three_hours() {
        let mut l = Ledger::default();
        for k in 0..10 {
            l.record(attempt(&format!("site{k}"), T0 + k * 60, Outcome::Issued));
        }
        assert_eq!(
            l.may_order("fresh", T0 + 3600),
            Err(Wait {
                until: T0 + ORDER_WINDOW,
                why: "the gateway-wide order cap is reached"
            })
        );
        assert_eq!(l.may_order("fresh", T0 + ORDER_WINDOW + 540), Ok(()));
    }

    /// One refusal holds every site back an hour — the limit may be the
    /// account's, which no other site's own history would show.
    #[test]
    fn a_rate_limit_holds_every_site_back_an_hour() {
        let mut l = Ledger::default();
        l.record(attempt("tokera", T0, Outcome::RateLimited));
        assert_eq!(
            l.may_order("npcd", T0 + HOUR - 1),
            Err(Wait {
                until: T0 + HOUR,
                why: "the CA reported a rate limit"
            })
        );
        assert_eq!(l.may_order("npcd", T0 + HOUR), Ok(()));
    }

    /// Refusals in a row double the hold, whichever sites they came from, up to
    /// a day.
    #[test]
    fn consecutive_rate_limits_back_off_exponentially_across_sites() {
        let mut l = Ledger::default();
        l.record(attempt("tokera", T0, Outcome::RateLimited));
        l.record(attempt("npcd", T0 + HOUR, Outcome::RateLimited));
        l.record(attempt("zend", T0 + 3 * HOUR, Outcome::RateLimited));
        assert_eq!(
            l.may_order("fresh", T0 + 3 * HOUR + 1),
            Err(Wait {
                until: T0 + 3 * HOUR + 4 * HOUR,
                why: "the CA reported a rate limit"
            })
        );
        assert_eq!(l.may_order("fresh", T0 + 7 * HOUR), Ok(()));

        let mut l = Ledger::default();
        for k in 0..8 {
            l.record(attempt("tokera", T0 + k * DAY, Outcome::RateLimited));
        }
        assert_eq!(
            l.may_order("fresh", T0 + 7 * DAY + 1).unwrap_err().until,
            T0 + 7 * DAY + DAY,
            "capped at a day"
        );
    }

    /// An order the CA took — issued or not — ends the run of refusals: it was
    /// accepting orders again.
    #[test]
    fn an_accepted_order_ends_the_rate_limit_run() {
        for accepted in [Outcome::Issued, Outcome::Failed] {
            let mut l = Ledger::default();
            l.record(attempt("tokera", T0, Outcome::RateLimited));
            l.record(attempt("npcd", T0 + 10, accepted));
            assert_eq!(l.may_order("fresh", T0 + 11), Ok(()), "{accepted:?}");
        }
    }

    #[test]
    fn one_account_a_day() {
        let mut l = Ledger::default();
        l.record_account(T0);
        assert_eq!(
            l.may_create_account(T0 + DAY - 1),
            Err(Wait {
                until: T0 + DAY,
                why: "an account was already created today"
            })
        );
        assert_eq!(l.may_create_account(T0 + DAY), Ok(()));
    }

    /// History no rule reads any more is dropped, so the file stays small; what
    /// a rule still reads is kept.
    #[test]
    fn history_past_every_window_is_pruned() {
        let mut l = Ledger::default();
        l.record(attempt("old", T0, Outcome::Issued));
        l.record(attempt("kept", T0 + 2 * DAY, Outcome::Issued));
        l.record(attempt("now", T0 + WEEK + DAY + 1, Outcome::Issued));
        let sites: Vec<&str> = l.attempts.iter().map(|a| a.site.as_str()).collect();
        assert_eq!(sites, vec!["kept", "now"]);
    }

    #[test]
    fn the_ledger_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ledger.json");
        assert_eq!(
            Ledger::load(&path).unwrap(),
            Ledger::default(),
            "none yet is empty"
        );
        let mut l = Ledger::default();
        l.record(attempt("tokera", T0, Outcome::Failed));
        l.record_account(T0);
        l.save(&path).unwrap();
        assert_eq!(Ledger::load(&path).unwrap(), l);
    }

    /// A ledger that cannot be read refuses to load rather than reading as
    /// empty — an empty one would forget the very history that holds orders
    /// back.
    #[test]
    fn a_corrupt_ledger_is_an_error_not_a_fresh_start() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ledger.json");
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(Ledger::load(&path).is_err());
    }
}
