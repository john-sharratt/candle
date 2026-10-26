//! The guide library (`docs/mcp_guides.md`): the methods for doing work well
//! with the backends' tools — research, review, deploy, bots — served as tools,
//! the one channel every client delivers.
//!
//! Each guide is a Markdown file beside this one, with YAML frontmatter
//! (`name`, `description`, `tools`), embedded at build time, so the guides ship
//! in the same binary as the code that serves them. A `.md` dropped in this
//! folder is picked up by the next build.
//!
//! | Module | What |
//! |---|---|
//! | [`guide`] | One guide, read from its file |
//! | [`library`] | Every guide: the catalogue, one guide's body, the line each tool lists its guides on |
//! | [`tools`] | `guide_list` and `guide_read`, the aggregator's own tools |

use include_dir::{include_dir, Dir};

pub mod guide;
pub mod library;
pub mod tools;

pub use guide::Guide;
pub use library::Library;

/// The guides, as written beside this file.
static GUIDES: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/src/guides");

/// Every embedded guide, read and checked.
pub fn embedded() -> anyhow::Result<Library> {
    let mut guides = Vec::new();
    for file in GUIDES.files() {
        let path = file.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let text = file
            .contents_utf8()
            .ok_or_else(|| anyhow::anyhow!("guide {} is not UTF-8", path.display()))?;
        let guide = Guide::parse(text).map_err(|e| anyhow::anyhow!("guide {stem}: {e}"))?;
        if guide.name != stem {
            anyhow::bail!("guide {stem}: its name is `{}`, not its file's", guide.name);
        }
        guides.push(guide);
    }
    Library::new(guides)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every guide written beside this file reads**, its name its file's,
    /// and the library they make holds each once.
    #[test]
    fn every_embedded_guide_reads() {
        let library = embedded().unwrap();
        assert!(library.count() >= 40, "{} guides", library.count());
        for name in ["thread", "delegate", "research", "verify", "deploy", "bots"] {
            assert!(library.get(name).is_some(), "{name}");
        }
    }
}
