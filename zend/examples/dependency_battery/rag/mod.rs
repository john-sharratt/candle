//! Arm R1's retrieval: the retrieval-augmented baseline of
//! `docs/unbounded_agents.md` §9.12. The repository's text files
//! (`corpus.rs`) are indexed by BM25 (`bm25.rs`) over identifier terms
//! (`tokens.rs`); a question's ranking is read back from disk best first and
//! packed under the token budget (`context.rs`) into the prompt ahead of the
//! question. A document's terms are its path's and its text's, so a question
//! that names a file meets it by name too.

pub mod bm25;
pub mod context;
pub mod corpus;
pub mod tokens;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context as _;

use bm25::Index;
use context::{pack, Packed, RetrievedFile, BUDGET_CHARS};
use corpus::{text_of, walk};
use tokens::terms;

/// The repository's index, and where its files are read back from.
pub struct Retriever {
    repo: PathBuf,
    index: Index,
}

impl Retriever {
    /// Index every admitted file of `repo` not under one of `exclude`.
    pub fn build(repo: &Path, exclude: &[PathBuf]) -> anyhow::Result<Self> {
        let repo = fs::canonicalize(repo).with_context(|| format!("finding {}", repo.display()))?;
        let exclude: Vec<PathBuf> = exclude
            .iter()
            .filter_map(|x| fs::canonicalize(x).ok())
            .collect();
        let mut index = Index::default();
        walk(&repo, &exclude, |rel, text| {
            let mut t = terms(&rel);
            t.extend(terms(text));
            index.add(rel, &t);
        })?;
        Ok(Self { repo, index })
    }

    /// The files indexed.
    pub fn files(&self) -> usize {
        self.index.len()
    }

    /// `question`'s files block: its BM25 ranking read back best first and
    /// packed under [`BUDGET_CHARS`].
    pub fn retrieve(&self, question: &str) -> Packed {
        let ranked = self.index.rank(&terms(question));
        let read = ranked
            .into_iter()
            .enumerate()
            .filter_map(|(i, (doc, score))| {
                let rel = self.index.path(doc);
                let raw = fs::read(self.repo.join(rel)).ok()?;
                let text = text_of(&raw)?.to_string();
                Some((i as u32 + 1, rel.to_string(), score, text))
            });
        pack(read, BUDGET_CHARS)
    }
}

/// `files` with each answer file of `expected` marked; whether any was.
pub fn mark(files: &mut [RetrievedFile], expected: &[String]) -> bool {
    for f in files.iter_mut() {
        f.answer_file = expected.iter().any(|e| e.replace('\\', "/") == f.path);
    }
    files.iter().any(|f| f.answer_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A question retrieves the file that spells what it asks about**, by
    /// a compound word's parts, and the answer file is marked as the hit.
    #[test]
    fn a_question_retrieves_and_marks_its_answer_file() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/span.rs"),
            "/// Whether the tier fits below the weight floor.\nfn tier_fits() {}\n",
        )
        .unwrap();
        fs::write(root.join("src/other.rs"), "fn unrelated() {}\n").unwrap();
        fs::write(root.join("battery.yaml"), "expect: [tier_fits, span.rs]\n").unwrap();
        let r = Retriever::build(root, &[root.join("battery.yaml")]).unwrap();
        assert_eq!(r.files(), 2);
        let mut p = r.retrieve("Which function checks that the tier fits?");
        assert_eq!(p.files.len(), 1);
        assert_eq!(
            (p.files[0].path.as_str(), p.files[0].rank),
            ("src/span.rs", 1)
        );
        assert!(p
            .text
            .starts_with("=== src/span.rs ===\n/// Whether the tier fits"));
        assert!(mark(&mut p.files, &["src/span.rs".to_string()]));
        assert!(p.files[0].answer_file);
        assert!(!mark(&mut p.files, &["src/other.rs".to_string()]));
    }
}
