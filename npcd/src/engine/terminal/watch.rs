//! The watch the engine keeps without a mind (`docs/zen_everywhere.md` §4):
//! what of the cluster's standing and of a machine's health check is news,
//! so a mind is told when something changed and not asked to look on the hour.
//!
//! **Looking needs no mind; deciding what a change means does.** A round that
//! sent a mind to read the cluster every two hours spent a turn — its prompt,
//! a connection, a reading — on finding nothing new, which is what it found
//! nearly every time. The engine reads on its own clock and raises the
//! terminal only for a change; the mind is woken by the change, with it in
//! hand.

/// The words of the cluster's standing that move on every reading and mean
/// nothing by moving: how long ago a member started, what it has yet to send,
/// how far a download has got.
const PASSING: &[&str] = &[" ago", "unsent", "%"];

/// The cluster's standing as it is compared from one reading to the next:
/// each line of `status`, with the parts that move on every reading taken
/// out, and the empty ones dropped.
pub fn standing(status: &str) -> Vec<String> {
    status
        .lines()
        .map(|line| {
            line.split([';', ','])
                .map(str::trim)
                .filter(|part| !part.is_empty() && !PASSING.iter().any(|p| part.contains(p)))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|line| !line.is_empty())
        .collect()
}

/// What changed between two readings of the cluster's standing, as the lines
/// a mind is told — `None` when nothing did.
pub fn cluster_news(before: &[String], now: &[String]) -> Option<String> {
    let came: Vec<&String> = now.iter().filter(|l| !before.contains(l)).collect();
    let went: Vec<&String> = before.iter().filter(|l| !now.contains(l)).collect();
    if came.is_empty() && went.is_empty() {
        return None;
    }
    let mut out = vec!["The cluster's standing has changed.".to_string()];
    if !came.is_empty() {
        out.push("Now:".to_string());
        out.extend(came.iter().map(|l| format!("  {l}")));
    }
    if !went.is_empty() {
        out.push("No longer:".to_string());
        out.extend(went.iter().map(|l| format!("  {l}")));
    }
    Some(out.join("\n"))
}

/// How a check's report opens when something changed since it last ran
/// (`agent::checks::changed`).
const CHANGED: &str = "Changed since the last check";

/// What a machine's check found changed, from its report — the part it leads
/// with, to its first blank line — or `None` when it reports no change, or is
/// its first run there.
pub fn machine_news(report: &str) -> Option<&str> {
    let report = report.trim_start();
    report
        .starts_with(CHANGED)
        .then(|| report.split("\n\n").next().unwrap_or(report).trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BEFORE: &str = "\
Members answering /v1/status: 2
- BlackWorld 223ebb209828d84c build 5102-572e1a22c: ready, started 11.5 h ago; tenants: zend ready; replication: zend joined gen 0, 1 unsent
- WhiteWorld 09dd3987369d9cd8 build 5110-e8060efde: ready, started 2.0 h ago; tenants: npcd ready

  npcd WhiteWorld 09dd3987369d9cd8 build 5110-e8060efde: ready
";

    /// What moves on every reading is not the standing: two readings an hour
    /// apart of a cluster nothing happened to compare equal.
    #[test]
    fn a_reading_an_hour_later_is_the_same_standing() {
        let later = BEFORE
            .replace("11.5 h ago", "12.5 h ago")
            .replace("2.0 h ago", "3.0 h ago")
            .replace(", 1 unsent", "");
        assert_eq!(standing(BEFORE), standing(&later));
        assert_eq!(
            standing(BEFORE)[1],
            "- BlackWorld 223ebb209828d84c build 5102-572e1a22c: ready, tenants: zend ready, \
             replication: zend joined gen 0"
        );
        assert_eq!(cluster_news(&standing(BEFORE), &standing(&later)), None);
    }

    /// A member that fell out, and one on a new build, are told as what is so
    /// now and what is so no longer.
    #[test]
    fn a_member_gone_or_on_a_new_build_is_news() {
        let now = BEFORE
            .replace("Members answering /v1/status: 2", "Members answering /v1/status: 1")
            .replace(
                "- WhiteWorld 09dd3987369d9cd8 build 5110-e8060efde: ready, started 2.0 h ago; tenants: npcd ready\n",
                "",
            )
            .replace(
                "  npcd WhiteWorld 09dd3987369d9cd8 build 5110-e8060efde: ready",
                "  npcd WhiteWorld 09dd3987369d9cd8 build 5111-81ee2d07b: loading",
            );
        assert_eq!(
            cluster_news(&standing(BEFORE), &standing(&now)).as_deref(),
            Some(
                "The cluster's standing has changed.\n\
                 Now:\n  \
                 Members answering /v1/status: 1\n  \
                 npcd WhiteWorld 09dd3987369d9cd8 build 5111-81ee2d07b: loading\n\
                 No longer:\n  \
                 Members answering /v1/status: 2\n  \
                 - WhiteWorld 09dd3987369d9cd8 build 5110-e8060efde: ready, tenants: npcd ready\n  \
                 npcd WhiteWorld 09dd3987369d9cd8 build 5110-e8060efde: ready"
            )
        );
    }

    /// A check's report is news only when it leads with a change, and the
    /// news is that leading part alone.
    #[test]
    fn a_check_is_news_only_when_it_leads_with_a_change() {
        assert_eq!(
            machine_news("Nothing changed since the last check.\n\ncpu 4%"),
            None
        );
        assert_eq!(
            machine_news("This is the first time this check has run here: there is nothing to compare it with.\n\ndisk C: 40% free"),
            None
        );
        assert_eq!(
            machine_news(
                "Changed since the last check — 1 new, 0 gone:\n+ disk D: 3% free\n\ncpu 4%\n\ndisk C: 40% free"
            ),
            Some("Changed since the last check — 1 new, 0 gone:\n+ disk D: 3% free")
        );
    }
}
