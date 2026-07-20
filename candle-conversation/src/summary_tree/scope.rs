//! Deriving a summary's **user half** — its *scope* — from its children's
//! (`docs/immutable_summary_forest.md`, *Scope derivation*).
//!
//! A summary turn is a compressed *exchange*: a user half (the question it
//! answers — its scope) and an assistant half (the answer — its content). The
//! content may be decoded by the model or built deterministically, but the
//! scope never is: **a model decode always speaks as the assistant**, so asking
//! it for the user half is asking it to invent a question that was never asked.
//! Given the thin input a summary node carries it fabricates — placeholder
//! paths, shell snippets, rambling — a model ceiling no prompt fixes, because
//! the *input* is degenerate. The children's scopes, meanwhile, already say
//! exactly what the node covers. So the scope is always derived, never decoded;
//! only the derivation varies by layer:
//!
//! - [`Scope::Union`] — the generic default: deduplicate the children's scopes,
//!   keep the most recent, elide the remainder by count.
//! - [`Scope::LineSpans`] — `path:a-b` references coalesced per path, for layers
//!   whose turns are line ranges of the same files (`code_read`).
//!
//! This drives [`Content::Decode`](crate::projection::Content) levels. A
//! `Content::Structural` level derives *both* halves from its children's
//! skeletons (`super::structural`) instead: for a directory tree the skeleton is
//! the authoritative statement of what the node covers, whereas the children's
//! scopes are not — a `repo_map` scan turn's user half is prose (``Repository
//! index — `candle-nn/src`:``), so deriving a directory scope from it would
//! parse the prose as a path.
//!
//! Every variant is **monotonically coarsening**: a node's scope is never larger
//! than the union of its children's, and shrinks with tree height. That is the
//! property a naive union lacks — unioning eight children's scopes at every
//! level grows the scope toward the root until it is the whole conversation,
//! which is the same failure the structural roll-up exists to avoid on the
//! content side.

use std::collections::BTreeMap;

use super::tree::MERGE_FANOUT;

/// How a summary node's user half (its scope) is derived from its children's.
///
/// Deliberately has no `Decode` variant — see the module docs: the user half is
/// never the model's to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Deduplicated union of the children's scopes, capped by height.
    #[default]
    Union,
    /// `path:a-b` references, coalesced into merged spans per path.
    LineSpans,
}

impl Scope {
    /// Derive the scope for a node at `height` from its `children`'s scopes
    /// (their user halves, in chronological order — the last is the newest).
    ///
    /// `height` is the node's tree height: a `SummaryOfTurns` leaf is 1 (and has
    /// exactly one child, whose scope it therefore keeps whole); a
    /// `SummaryOfSummaries` over level-`h` children is `h + 1`.
    pub fn combine(&self, children: &[String], height: u8) -> String {
        match self {
            Scope::Union => union_scope(children, height),
            Scope::LineSpans => line_spans_scope(children, height),
        }
    }
}

/// How many distinct child scopes a node at `height` keeps verbatim before
/// eliding the rest. A leaf (`h = 1`, exactly one child) keeps its child's scope
/// whole; each level up halves the budget, so the scope provably coarsens toward
/// the root. Never below 1 — a node always names *something*.
fn entries_for_height(height: u8) -> usize {
    let steps = u32::from(height.saturating_sub(1)).min(3);
    (MERGE_FANOUT >> steps).max(1)
}

/// Deduplicate `children` (order-preserving, first occurrence wins) and drop
/// empties.
fn distinct(children: &[String]) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    for child in children {
        let trimmed = child.trim();
        if !trimmed.is_empty() && !out.contains(&trimmed) {
            out.push(trimmed);
        }
    }
    out
}

/// The generic derivation: the distinct children's scopes, newest first, capped
/// at the height's budget. An elided remainder is *counted* rather than dropped
/// silently, so the node still admits the history it covers.
fn union_scope(children: &[String], height: u8) -> String {
    let distinct = distinct(children);
    if distinct.is_empty() {
        return String::new();
    }
    let cap = entries_for_height(height);
    if distinct.len() <= cap {
        return distinct.join("; ");
    }
    // Children arrive chronologically, so the tail is the most recent.
    let elided = distinct.len() - cap;
    let kept = &distinct[distinct.len() - cap..];
    format!("{}; (+{elided} earlier)", kept.join("; "))
}

/// The `code_read` derivation: parse `path:a-b` references out of the children's
/// scopes and coalesce them into merged spans per path — the children are reads
/// of the *same files* at different line ranges, so the honest roll-up of
/// `foo.rs:1-40` and `foo.rs:41-90` is `foo.rs:1-90`, not a list of both.
///
/// Beyond `height 2` the line numbers themselves stop earning their tokens and
/// only the paths are kept: a node that high covers so much of each file that
/// naming the range says little.
fn line_spans_scope(children: &[String], height: u8) -> String {
    let mut files: BTreeMap<&str, Vec<(u32, u32)>> = BTreeMap::new();
    let mut bare: Vec<&str> = Vec::new();
    for child in children {
        for token in child.split([',', ';']).map(str::trim) {
            if token.is_empty() {
                continue;
            }
            match parse_ref(token) {
                Some((path, span)) => files.entry(path).or_default().push(span),
                None => {
                    if !bare.contains(&token) {
                        bare.push(token);
                    }
                }
            }
        }
    }
    if files.is_empty() {
        // Nothing parsed as a reference — fall back to the generic derivation
        // rather than emitting an empty scope.
        return union_scope(children, height);
    }
    let keep_lines = height <= 2;
    let mut out: Vec<String> = files
        .into_iter()
        .map(|(path, mut spans)| {
            if !keep_lines {
                return path.to_string();
            }
            spans.sort_unstable();
            let merged = coalesce(&spans);
            let ranges = merged
                .iter()
                .map(|(a, b)| {
                    if a == b {
                        a.to_string()
                    } else {
                        format!("{a}-{b}")
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{path}:{ranges}")
        })
        .collect();
    out.extend(bare.into_iter().map(str::to_string));
    let cap = entries_for_height(height);
    if out.len() <= cap {
        return out.join("; ");
    }
    let elided = out.len() - cap;
    out.truncate(cap);
    format!("{}; (+{elided} more)", out.join("; "))
}

/// Parse a line reference out of one token, in either form it occurs in:
///
/// - the **summarise-request header** a `code_reading` turn's user half actually
///   carries — ``Summarize `src/auth/handler.rs` (lines 47-93) in no more than
///   two sentences.`` (see `zend::code_read::header::render_part_user_prompt`);
/// - the **compact** `path:a-b` this derivation itself emits, so a roll-up can
///   re-parse its children's scopes and merge again one level up.
///
/// Returns `None` when the token carries no parsable reference (a bare path, or
/// prose) — the caller then keeps it verbatim rather than inventing a span.
fn parse_ref(token: &str) -> Option<(&str, (u32, u32))> {
    parse_excerpt_ref(token).or_else(|| parse_compact_ref(token))
}

/// ``… `path` (lines A-B) …`` (and the older ``… `path` lines A-B …``) — the
/// request-header form. The path is the backticked segment; requiring the
/// backticks keeps the surrounding prose ("Summarize ", the trailing sentence
/// count) out of it. The span is the leading `A-B` immediately after `lines `,
/// so trailing prose (`) in no more than two sentences.`) never bleeds into it.
fn parse_excerpt_ref(token: &str) -> Option<(&str, (u32, u32))> {
    // Split on `lines ` (no leading space) so both `(lines 47-93)` and the older
    // ` lines 47-93:` head forms leave the backticked path in `head`.
    let (head, tail) = token.rsplit_once("lines ")?;
    let mut backticked = head.rsplit('`');
    let _after = backticked.next()?;
    let path = backticked.next().filter(|p| !p.is_empty())?;
    // Take the leading span token only — digits, '-', spaces — up to the first
    // other char (`)`, `:`, or the start of the trailing prose).
    let span_str: String = tail
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == ' ')
        .collect();
    let span = parse_span(span_str.trim())?;
    Some((path, span))
}

/// `path:a-b` / `path:a` — split on the LAST colon so a Windows drive letter
/// (`C:/src/a.rs:1-40`) leaves the path intact.
fn parse_compact_ref(token: &str) -> Option<(&str, (u32, u32))> {
    let (path, suffix) = token.rsplit_once(':')?;
    if path.is_empty() {
        return None;
    }
    Some((path, parse_span(suffix.trim())?))
}

/// `a-b` or a lone `a`. Rejects a reversed span rather than silently swapping —
/// that token is not a reference we understand.
fn parse_span(text: &str) -> Option<(u32, u32)> {
    let span = match text.split_once('-') {
        Some((a, b)) => (a.trim().parse().ok()?, b.trim().parse().ok()?),
        None => {
            let n = text.trim().parse().ok()?;
            (n, n)
        }
    };
    (span.0 <= span.1).then_some(span)
}

/// Merge overlapping and adjacent spans. `spans` must be sorted.
fn coalesce(spans: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &(start, end) in spans {
        match out.last_mut() {
            // Adjacent counts as overlapping: 1-40 and 41-90 are one read of
            // 1-90, and saying so is both shorter and more honest.
            Some(last) if start <= last.1.saturating_add(1) => last.1 = last.1.max(end),
            _ => out.push((start, end)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|x| x.to_string()).collect()
    }

    /// The property every variant owes the forest: a leaf keeps its one child's
    /// scope verbatim, so an SoT is a faithful `(question, compressed answer)`.
    #[test]
    fn leaf_keeps_its_single_child_scope_verbatim() {
        let child = s(&["how does the paged KV cache evict?"]);
        assert_eq!(
            Scope::Union.combine(&child, 1),
            "how does the paged KV cache evict?"
        );
    }

    #[test]
    fn union_deduplicates_and_preserves_order() {
        let kids = s(&["alpha", "beta", "alpha"]);
        assert_eq!(Scope::Union.combine(&kids, 1), "alpha; beta");
    }

    #[test]
    fn union_drops_empty_children() {
        let kids = s(&["alpha", "   ", "", "beta"]);
        assert_eq!(Scope::Union.combine(&kids, 1), "alpha; beta");
    }

    #[test]
    fn union_of_nothing_is_empty() {
        assert_eq!(Scope::Union.combine(&[], 1), "");
        assert_eq!(Scope::Union.combine(&s(&["", "  "]), 2), "");
    }

    /// The budget halves each level, so the scope coarsens toward the root.
    #[test]
    fn entries_budget_halves_per_level() {
        assert_eq!(entries_for_height(1), 8);
        assert_eq!(entries_for_height(2), 4);
        assert_eq!(entries_for_height(3), 2);
        assert_eq!(entries_for_height(4), 1);
        // Never below one — a node always names something.
        assert_eq!(entries_for_height(5), 1);
        assert_eq!(entries_for_height(200), 1);
    }

    /// Over budget: the newest survive verbatim and the rest are counted, not
    /// silently dropped.
    #[test]
    fn union_keeps_newest_and_counts_the_elided() {
        let kids = s(&["a", "b", "c", "d", "e", "f"]);
        // height 2 → budget 4 → keep the last four, count the two elided.
        assert_eq!(Scope::Union.combine(&kids, 2), "c; d; e; f; (+2 earlier)");
        // height 3 → budget 2.
        assert_eq!(Scope::Union.combine(&kids, 3), "e; f; (+4 earlier)");
    }

    /// A roll-up is never longer than the naive union of its children, and never
    /// grows with height — the property the naive merge lacked.
    #[test]
    fn union_is_monotonically_coarsening() {
        let kids = s(&[
            "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh",
        ]);
        let naive = kids.join("; ");
        let mut prev = usize::MAX;
        for height in 2..=6u8 {
            let combined = Scope::Union.combine(&kids, height);
            assert!(
                combined.len() < naive.len(),
                "height {height} scope {combined:?} is no shorter than the naive union"
            );
            assert!(
                combined.len() <= prev,
                "height {height} scope {combined:?} grew against the level below it"
            );
            prev = combined.len();
        }
    }

    #[test]
    fn line_spans_merge_adjacent_reads_of_one_file() {
        let kids = s(&["src/lib.rs:1-40", "src/lib.rs:41-90"]);
        assert_eq!(Scope::LineSpans.combine(&kids, 1), "src/lib.rs:1-90");
    }

    #[test]
    fn line_spans_merge_overlapping_reads() {
        let kids = s(&["src/lib.rs:1-50", "src/lib.rs:30-90"]);
        assert_eq!(Scope::LineSpans.combine(&kids, 1), "src/lib.rs:1-90");
    }

    /// Disjoint ranges stay disjoint — merging them would claim a read that
    /// never happened.
    #[test]
    fn line_spans_keep_disjoint_ranges_apart() {
        let kids = s(&["src/lib.rs:1-10", "src/lib.rs:80-90"]);
        assert_eq!(Scope::LineSpans.combine(&kids, 1), "src/lib.rs:1-10,80-90");
    }

    #[test]
    fn line_spans_group_by_path() {
        let kids = s(&["a.rs:1-10", "b.rs:5-9", "a.rs:11-20"]);
        assert_eq!(Scope::LineSpans.combine(&kids, 1), "a.rs:1-20; b.rs:5-9");
    }

    #[test]
    fn line_spans_accept_a_single_line_ref() {
        let kids = s(&["a.rs:42", "a.rs:43"]);
        assert_eq!(Scope::LineSpans.combine(&kids, 1), "a.rs:42-43");
        let lone = s(&["a.rs:42"]);
        assert_eq!(Scope::LineSpans.combine(&lone, 1), "a.rs:42");
    }

    #[test]
    fn line_spans_split_on_commas_within_a_child() {
        let kids = s(&["a.rs:1-10, b.rs:1-5", "a.rs:11-15"]);
        assert_eq!(Scope::LineSpans.combine(&kids, 1), "a.rs:1-15; b.rs:1-5");
    }

    /// High in the tree the ranges stop earning their tokens — only paths.
    #[test]
    fn line_spans_drop_line_numbers_above_height_two() {
        let kids = s(&["a.rs:1-10", "b.rs:5-9"]);
        assert_eq!(Scope::LineSpans.combine(&kids, 2), "a.rs:1-10; b.rs:5-9");
        assert_eq!(Scope::LineSpans.combine(&kids, 3), "a.rs; b.rs");
    }

    /// Prose that carries no reference must not vanish into an empty scope.
    #[test]
    fn line_spans_fall_back_to_union_without_refs() {
        let kids = s(&["what does the scheduler do?"]);
        assert_eq!(
            Scope::LineSpans.combine(&kids, 1),
            "what does the scheduler do?"
        );
    }

    /// A malformed or reversed span is not a reference — it must not be parsed
    /// into a bogus range.
    #[test]
    fn malformed_refs_are_not_spans() {
        assert_eq!(parse_ref("a.rs:90-10"), None);
        assert_eq!(parse_ref("a.rs:abc"), None);
        assert_eq!(parse_ref("a.rs"), None);
        assert_eq!(parse_ref(":10-20"), None);
        assert_eq!(parse_ref("a.rs:1-40"), Some(("a.rs", (1, 40))));
    }

    /// The form a `code_reading` turn's user half ACTUALLY carries
    /// (`zend::code_read::header::render_part_user_prompt`). Parsing the real
    /// header — not just the compact form this module emits — is what makes
    /// `line_spans` do anything at all on a live substrate.
    #[test]
    fn parses_the_production_excerpt_header() {
        assert_eq!(
            parse_ref(
                "Summarize `src/auth/handler.rs` (lines 47-93) in no more than two sentences."
            ),
            Some(("src/auth/handler.rs", (47, 93)))
        );
        // The older ``… lines A-B:`` header form still parses (robustness).
        assert_eq!(
            parse_ref("Source excerpt — `src/auth/handler.rs` lines 47-93:"),
            Some(("src/auth/handler.rs", (47, 93)))
        );
    }

    /// The leaf normalises the real header into the compact form, and the level
    /// above re-parses that and merges — the recursion the roll-up depends on.
    #[test]
    fn excerpt_headers_normalise_then_merge_one_level_up() {
        let leaves = s(&[
            "Source excerpt — `src/auth/handler.rs` lines 1-40:",
            "Source excerpt — `src/auth/handler.rs` lines 41-93:",
        ]);
        // Each leaf keeps its own single child verbatim…
        let a = Scope::LineSpans.combine(&leaves[..1], 1);
        let b = Scope::LineSpans.combine(&leaves[1..], 1);
        assert_eq!(a, "src/auth/handler.rs:1-40");
        assert_eq!(b, "src/auth/handler.rs:41-93");
        // …and the SoS over them coalesces into one honest range.
        assert_eq!(
            Scope::LineSpans.combine(&[a, b], 2),
            "src/auth/handler.rs:1-93"
        );
    }

    /// Prose without backticks must not be mined for a path.
    #[test]
    fn excerpt_header_requires_a_backticked_path() {
        assert_eq!(parse_ref("some prose lines 4-8"), None);
    }

    /// Windows-style paths carry a drive colon; the *last* colon is the span
    /// separator, so the path survives intact.
    #[test]
    fn drive_letter_paths_split_on_the_last_colon() {
        assert_eq!(
            parse_ref("C:/src/a.rs:1-40"),
            Some(("C:/src/a.rs", (1, 40)))
        );
    }

    /// `union` is the default so a layer that names no derivation still gets a
    /// real user half rather than the empty one summaries carry today.
    #[test]
    fn default_is_union() {
        assert_eq!(Scope::default(), Scope::Union);
    }
}
