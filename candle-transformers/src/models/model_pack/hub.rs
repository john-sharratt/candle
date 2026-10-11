//! Model pack sources from the HuggingFace hub.

use super::request::SourceRef;
use super::resolve::{Fetched, SourceFetch};
use crate::models::hub_download::repo_file;
use candle::Result;
use hf_hub::{Repo, RepoType};

/// A [`SourceFetch`] over the HuggingFace hub, through `hub_download`: the hub
/// cache first, then one timeout-protected download.
///
/// A source released after a build is removed from whichever cache answered —
/// a hub snapshot entry and the blob it links to, or the download cache's file —
/// because the pack now holds everything it carried.
pub struct HubFetch;

fn repo(repo: &str, rev: &str) -> Repo {
    let rev = if rev.is_empty() { "main" } else { rev };
    Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string())
}

impl SourceFetch for HubFetch {
    fn fetch(&self, s: &SourceRef) -> Result<Fetched> {
        Ok(Fetched {
            path: repo_file(&repo(&s.repo, &s.rev), &s.file)?,
            cached: true,
        })
    }

    fn tokenizer_json(&self, repo_id: &str, rev: &str) -> Result<String> {
        let p = repo_file(&repo(repo_id, rev), "tokenizer.json")?;
        std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
    }

    fn release(&self, f: &Fetched) -> Result<()> {
        // A hub snapshot entry is a link to its blob; the blob is the bytes.
        let target = std::fs::read_link(&f.path)
            .ok()
            .map(|t| match f.path.parent() {
                Some(dir) if t.is_relative() => dir.join(t),
                _ => t,
            });
        std::fs::remove_file(&f.path)
            .map_err(|e| candle::Error::Msg(format!("release {:?}: {e}", f.path)))?;
        if let Some(blob) = target {
            std::fs::remove_file(&blob)
                .map_err(|e| candle::Error::Msg(format!("release {blob:?}: {e}")))?;
        }
        Ok(())
    }
}
