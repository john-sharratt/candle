//! Okapi BM25 (Robertson and Zaragoza, 2009) over the corpus's documents, at
//! k1 = [`K1`] and b = [`B`]. A document's score for a query is, over the
//! query's distinct terms t,
//!
//! ```text
//! idf(t) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · dl / avgdl))
//! idf(t) = ln(1 + (N − n(t) + 0.5) / (n(t) + 0.5))
//! ```
//!
//! with tf the term's count in the document, dl the document's length in
//! terms, avgdl the mean length, N the documents and n(t) those holding t.
//! The idf is the non-negative form, so a term in most documents still adds a
//! little rather than subtracting.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub const K1: f64 = 1.2;
pub const B: f64 = 0.75;

/// The inverted index: each term's postings, `(document, count)`.
#[derive(Default)]
pub struct Index {
    paths: Vec<String>,
    lengths: Vec<u32>,
    postings: HashMap<String, Vec<(u32, u32)>>,
}

impl Index {
    /// Index the document at `path` holding `terms`.
    pub fn add(&mut self, path: String, terms: &[String]) {
        let doc = self.paths.len() as u32;
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for t in terms {
            *counts.entry(t.as_str()).or_default() += 1;
        }
        for (t, n) in counts {
            self.postings
                .entry(t.to_string())
                .or_default()
                .push((doc, n));
        }
        self.paths.push(path);
        self.lengths.push(terms.len() as u32);
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn path(&self, doc: usize) -> &str {
        &self.paths[doc]
    }

    /// `term`'s idf.
    pub fn idf(&self, term: &str) -> f64 {
        let n = self.postings.get(term).map_or(0, Vec::len) as f64;
        (1.0 + (self.len() as f64 - n + 0.5) / (n + 0.5)).ln()
    }

    /// Every document scoring above zero for `query`, best first, a tie
    /// broken by path so a ranking reproduces exactly.
    pub fn rank(&self, query: &[String]) -> Vec<(usize, f64)> {
        if self.is_empty() {
            return Vec::new();
        }
        let avgdl = self.lengths.iter().map(|&l| l as f64).sum::<f64>() / self.len() as f64;
        let mut scores = vec![0.0f64; self.len()];
        let distinct: HashSet<&str> = query.iter().map(String::as_str).collect();
        for term in distinct {
            let Some(posts) = self.postings.get(term) else {
                continue;
            };
            let idf = self.idf(term);
            for &(doc, tf) in posts {
                let tf = tf as f64;
                let dl = self.lengths[doc as usize] as f64;
                scores[doc as usize] +=
                    idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / avgdl));
            }
        }
        let mut ranked: Vec<(usize, f64)> = scores
            .into_iter()
            .enumerate()
            .filter(|(_, s)| *s > 0.0)
            .collect();
        ranked.sort_by(|a, b| match b.1.total_cmp(&a.1) {
            Ordering::Equal => self.paths[a.0].cmp(&self.paths[b.0]),
            o => o,
        });
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(s: &str) -> Vec<String> {
        s.split(' ').map(str::to_string).collect()
    }

    fn index() -> Index {
        let mut ix = Index::default();
        ix.add("a.rs".into(), &terms("alpha beta"));
        ix.add("b.rs".into(), &terms("beta gamma gamma"));
        ix
    }

    /// N = 2, avgdl = 2.5. `alpha` is in one document: idf = ln(1 + 1.5/1.5)
    /// = ln 2, and a.rs (tf 1, dl 2) scores ln 2 · 2.2 / (1 + 1.2 · (0.25 +
    /// 0.6)) = ln 2 · 2.2 / 2.02. `gamma` twice in b.rs (dl 3): ln 2 · 4.4 /
    /// (2 + 1.2 · (0.25 + 0.9)) = ln 2 · 4.4 / 3.38.
    #[test]
    fn a_score_is_the_okapi_sum_by_hand() {
        let ix = index();
        assert_eq!(ix.idf("alpha"), 2f64.ln());
        assert_eq!(ix.idf("beta"), 1.2f64.ln());
        let alpha = ix.rank(&terms("alpha"));
        assert_eq!(alpha.len(), 1);
        assert_eq!(alpha[0].0, 0);
        assert!((alpha[0].1 - 2f64.ln() * 2.2 / 2.02).abs() < 1e-12);
        let gamma = ix.rank(&terms("gamma"));
        assert_eq!(gamma[0].0, 1);
        assert!((gamma[0].1 - 2f64.ln() * 4.4 / 3.38).abs() < 1e-12);
    }

    /// **The shorter document ranks first on a shared term**, a repeated query
    /// term counts once, and a term no document holds ranks nothing.
    #[test]
    fn ranking_prefers_the_denser_document() {
        let ix = index();
        let beta: Vec<usize> = ix
            .rank(&terms("beta"))
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        assert_eq!(beta, vec![0, 1]);
        assert_eq!(ix.rank(&terms("alpha alpha")), ix.rank(&terms("alpha")));
        assert!(ix.rank(&terms("delta")).is_empty());
        assert!(Index::default().rank(&terms("alpha")).is_empty());
        assert_eq!(ix.path(1), "b.rs");
    }

    #[test]
    fn a_tie_is_broken_by_path() {
        let mut ix = Index::default();
        ix.add("z.rs".into(), &terms("x y"));
        ix.add("m.rs".into(), &terms("x y"));
        let order: Vec<&str> = ix
            .rank(&terms("x"))
            .iter()
            .map(|(d, _)| ix.path(*d))
            .collect();
        assert_eq!(order, vec!["m.rs", "z.rs"]);
    }
}
