//! Jobs (`docs/jobs.md`): everything that runs longer than a call — a task, a
//! build or test run, a deploy, a maintenance job — watched, read, searched,
//! listed and stopped with the same five tools.
//!
//! | Module | What |
//! |---|---|
//! | [`handle`] | a job's handle, `<kind>:<id>` |
//! | [`source`] | each kind's job: where it stands, its output, stopping it, listing them |
//! | [`list`] | `job_list` |
//! | [`status`] | `job_status` |
//! | [`read`] | `job_read` |
//! | [`grep`] | `job_grep` |
//! | [`cancel`] | `job_cancel` |
//!
//! Every job has an output of numbered lines (§3), paged 200 to a page: page
//! `P` is lines `200(P-1)+1 … 200P`, and a line keeps its number as the job
//! writes more.

use std::time::Duration;

use cluster::log_read::{LogAsk, LogRead};

use crate::mcp::reported::Mark;

pub mod cancel;
pub mod grep;
pub mod handle;
pub mod list;
pub mod read;
pub mod source;
pub mod status;

pub use handle::JobHandle;

/// The lines a page holds.
pub const PAGE_LINES: u64 = 200;

/// How often a call that waits looks at its jobs again.
pub const LOOK_EVERY: Duration = Duration::from_secs(2);

/// How many pages `total` lines make: one, for an output still empty.
pub fn pages(total: u64) -> u64 {
    total.div_ceil(PAGE_LINES).max(1)
}

/// The ask that reads page `page` of an output of `total` lines: the page
/// itself, or the last page for one past the end.
pub fn page_ask(page: u64, total: u64) -> (u64, LogAsk) {
    let page = page.max(1).min(pages(total));
    (
        page,
        LogAsk::Lines {
            from: (page - 1) * PAGE_LINES + 1,
            count: PAGE_LINES as u32,
        },
    )
}

/// Where a job stands, whatever its kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Standing {
    /// It has ended, one way or another: its output is all written.
    pub ended: bool,
    /// Where it is, in a line: what `job_list` shows of it.
    pub line: String,
    /// What a status answer says beneath the line: how it went, once ended —
    /// its results, its failures by line, its answer — or what it is doing.
    pub detail: String,
    /// Lines in its output.
    pub total: u64,
    /// The journal entries `detail` hands over, kept as reported once the
    /// answer is delivered, so the next status gives only newer ones: a
    /// task's, and none for any other kind.
    pub marks: Vec<Mark>,
}

/// `read`, an answer to `ask` of a job's output, shifted `by` lines: what one
/// part of an output made of several reads as within the whole.
pub fn shifted(mut read: LogRead, by: u64) -> LogRead {
    for line in &mut read.lines {
        line.n += by;
    }
    if let Some(s) = &mut read.summary {
        for list in [&mut s.results, &mut s.failures, &mut s.errors, &mut s.last] {
            for line in list.iter_mut() {
                line.n += by;
            }
        }
    }
    read.next = read.next.map(|n| n + by);
    read.total += by;
    read
}

#[cfg(test)]
mod tests {
    use cluster::log_read::Line;

    use super::*;

    #[test]
    fn a_page_is_two_hundred_lines_and_past_the_end_is_the_last() {
        assert_eq!(pages(0), 1);
        assert_eq!(pages(200), 1);
        assert_eq!(pages(201), 2);
        assert_eq!(
            page_ask(2, 450),
            (
                2,
                LogAsk::Lines {
                    from: 201,
                    count: 200
                }
            )
        );
        assert_eq!(page_ask(99, 450).0, 3, "past the end reads the last page");
        assert_eq!(page_ask(0, 10).0, 1);
    }

    #[test]
    fn a_part_of_an_output_is_shifted_into_the_whole() {
        let read = LogRead {
            total: 3,
            lines: vec![Line {
                n: 2,
                text: "b".into(),
            }],
            next: Some(3),
            ..LogRead::default()
        };
        let moved = shifted(read, 100);
        assert_eq!(
            (moved.lines[0].n, moved.next, moved.total),
            (102, Some(103), 103)
        );
    }
}
