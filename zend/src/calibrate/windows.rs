//! The windows a long reading is drafted over.
//!
//! A draft holds its unit's reading in front of it, laid down as its lineage
//! (`conversation::Frame`). An inherited chain is capped
//! (`Substrate::INHERITED_CHAIN_TOKEN_CAP`) and a parent past the cap is
//! dropped whole, so a reading longer than the cap left its draft holding
//! nothing: on BlackWorld (2026-11-19) the six largest of the dependency
//! battery's answer files, and some fifty of the workspace's most central, had
//! no calibration at all. A reading past the cap is drafted instead over
//! windows of whole turns, each its request, a run of its pages and its answer
//! — the first and last turns every window holds, so each reads as the file's
//! reading — within a budget the trained window holds with the draft's own
//! turn beside it.

/// The most a window lays down, in tokens: the trained window (262,144) less
/// room for the system prompt, the draft's request and its answer.
pub const WINDOW_TOKENS: usize = 196_608;

/// The windows to draft a reading of `turn_tokens` over, each as the indices
/// of the turns it lays down, in order. A reading inside `budget` is one
/// window of every turn. Past it, every window holds the first and last turns
/// and as many of the turns between, in order, as fit beside them; a turn too
/// large to fit beside them is a window on its own with them.
pub fn windows(turn_tokens: &[usize], budget: usize) -> Vec<Vec<u32>> {
    let n = turn_tokens.len();
    if n == 0 {
        return Vec::new();
    }
    if turn_tokens.iter().sum::<usize>() <= budget || n <= 2 {
        return vec![(0..n as u32).collect()];
    }
    let last = n - 1;
    let floor = turn_tokens[0] + turn_tokens[last];
    let room = budget.saturating_sub(floor);
    let window = |middle: &[u32]| -> Vec<u32> {
        std::iter::once(0)
            .chain(middle.iter().copied())
            .chain(std::iter::once(last as u32))
            .collect()
    };
    let mut out = Vec::new();
    let mut middle: Vec<u32> = Vec::new();
    let mut used = 0usize;
    for (i, &t) in turn_tokens.iter().enumerate().take(last).skip(1) {
        if !middle.is_empty() && used + t > room {
            out.push(window(&middle));
            middle.clear();
            used = 0;
        }
        middle.push(i as u32);
        used += t;
    }
    if !middle.is_empty() {
        out.push(window(&middle));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A reading inside the budget is one window of every turn.**
    #[test]
    fn a_reading_inside_the_budget_is_one_window() {
        assert_eq!(windows(&[100, 300, 50], 1_000), vec![vec![0, 1, 2]]);
        assert_eq!(windows(&[], 1_000), Vec::<Vec<u32>>::new());
        assert_eq!(windows(&[900, 900], 1_000), vec![vec![0, 1]]);
    }

    /// **Past the budget every window is the request, a run of pages and the
    /// answer**, the pages in order and each in exactly one window.
    #[test]
    fn a_long_reading_is_windows_of_its_pages_between_its_ends() {
        // Request 100, answer 50: 850 of room for pages.
        let turns = [100, 400, 400, 400, 300, 50];
        assert_eq!(
            windows(&turns, 1_000),
            vec![vec![0, 1, 2, 5], vec![0, 3, 4, 5]]
        );
    }

    /// **A page too large for the room is a window of its own** with the
    /// request and the answer, never dropped.
    #[test]
    fn an_oversized_page_is_its_own_window() {
        let turns = [100, 2_000, 200, 50];
        assert_eq!(windows(&turns, 1_000), vec![vec![0, 1, 3], vec![0, 2, 3]]);
    }
}
