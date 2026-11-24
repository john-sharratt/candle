//! The terms BM25 indexes and queries by. A term is an identifier-like word —
//! a run of letters, digits and `_` — lowercased, kept whole and, when it is
//! compound, split into its parts as well: `snake_case` at its underscores,
//! `CamelCase` at each lower-to-upper step and at the end of an acronym run
//! (`HTTPServer` is `http` and `server`). A question that says "tier fits"
//! thus meets a file that spells `tier_fits`, and one that says `tier_fits`
//! meets it whole.

use crate::score::words;

/// `piece`'s CamelCase parts, in order.
pub fn camel_parts(piece: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = piece.char_indices().collect();
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..chars.len() {
        let (at, c) = chars[i];
        let prev = chars[i - 1].1;
        let next_lower = chars.get(i + 1).is_some_and(|(_, n)| n.is_lowercase());
        let step_up = (prev.is_lowercase() || prev.is_ascii_digit()) && c.is_uppercase();
        let acronym_ends = prev.is_uppercase() && c.is_uppercase() && next_lower;
        if step_up || acronym_ends {
            out.push(&piece[start..at]);
            start = at;
        }
    }
    if start < piece.len() {
        out.push(&piece[start..]);
    }
    out
}

/// `word`'s parts: split at its underscores, then each piece at its case
/// steps, empty pieces dropped.
pub fn parts(word: &str) -> Vec<&str> {
    word.split('_')
        .filter(|p| !p.is_empty())
        .flat_map(camel_parts)
        .collect()
}

/// Every term of `text`, in order: each word lowercased, then its parts
/// lowercased when it has any other than itself.
pub fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in words(text) {
        let whole = w.to_lowercase();
        let ps = parts(w);
        let compound = !(ps.len() == 1 && ps[0].to_lowercase() == whole);
        out.push(whole);
        if compound {
            out.extend(ps.into_iter().map(str::to_lowercase));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_case_splits_at_its_steps_and_its_acronyms() {
        assert_eq!(camel_parts("PagedKvArenas"), vec!["Paged", "Kv", "Arenas"]);
        assert_eq!(camel_parts("HTTPServer"), vec!["HTTP", "Server"]);
        assert_eq!(
            camel_parts("bdpRecall2Batched"),
            vec!["bdp", "Recall2", "Batched"]
        );
        assert_eq!(camel_parts("scan"), vec!["scan"]);
        assert_eq!(camel_parts("Q8KS"), vec!["Q8", "KS"]);
    }

    /// **A compound word is kept whole and in its parts; a plain one once.**
    #[test]
    fn a_word_is_indexed_whole_and_by_its_parts() {
        assert_eq!(
            terms("`tier_fits` in PagedKvArenas, scan.rs"),
            vec![
                "tier_fits",
                "tier",
                "fits",
                "in",
                "pagedkvarenas",
                "paged",
                "kv",
                "arenas",
                "scan",
                "rs",
            ]
        );
        assert_eq!(
            terms("__init__ Q4_0"),
            vec!["__init__", "init", "q4_0", "q4", "0"]
        );
        assert!(terms(" — ").is_empty());
    }
}
