//! The prepared engine artifact as a model pack's source.
//!
//! A Flash-Next pack is built from the artifact this module's recipe prepares,
//! not from a published file: [`EngineFetch`] answers a pack build's request
//! for its checkpoint by preparing the artifact (or finding it prepared), and
//! deletes it once the pack holds everything it carried.

use std::path::PathBuf;

use candle::{Device, Result};

use super::build::prepare_engine;
use super::recipe::Recipe;
use super::store::SourceStore;
use crate::models::model_pack::{Fetched, SourceFetch, SourceRef};

/// A [`SourceFetch`] whose checkpoint is the prepared engine artifact.
///
/// `store` decides whether a missing artifact is built: a store that fetches
/// the recipe's pinned sources builds it, one that refuses leaves the request
/// failing with the store's own reason. The gate passes the first; a daemon,
/// which must never start a 190 GB download as a side effect of opening a
/// model, passes the second.
pub struct EngineFetch<'a> {
    pub recipe: Recipe,
    /// Where the artifact is prepared — the model cache's repo directory.
    pub dir: PathBuf,
    pub store: &'a dyn SourceStore,
    /// `tokenizer.json`'s text from a repo at a revision.
    pub tokenizer: &'a dyn Fn(&str, &str) -> Result<String>,
    pub device: &'a Device,
}

impl SourceFetch for EngineFetch<'_> {
    fn fetch(&self, source: &SourceRef) -> Result<Fetched> {
        let name = self.recipe.artifact_name();
        if source.file != name {
            candle::bail!(
                "engine fetch: asked for {}, this recipe prepares {name}",
                source.file
            );
        }
        Ok(Fetched {
            path: prepare_engine(&self.recipe, &self.dir, self.store, self.device)?,
            cached: true,
        })
    }

    fn tokenizer_json(&self, repo: &str, rev: &str) -> Result<String> {
        (self.tokenizer)(repo, rev)
    }

    fn release(&self, fetched: &Fetched) -> Result<()> {
        std::fs::remove_file(&fetched.path).map_err(|e| {
            candle::Error::Msg(format!(
                "engine fetch: release {}: {e}",
                fetched.path.display()
            ))
        })
    }
}
