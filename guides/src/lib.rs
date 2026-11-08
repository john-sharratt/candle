//! The guide library (`docs/mcp_guides.md`): the methods for doing work well
//! with the tools — research, review, deploy, bots — read by the MCP connector,
//! which serves them as `guide_list` and `guide_read`, and by Zen's own guide
//! tools, so Claude and Zen read one set.
//!
//! Each guide is a Markdown file in `mcp/src/guides/`, with YAML frontmatter
//! (`name`, `description`, `tools`), embedded at build time, so the guides ship
//! in the binaries that serve them. A `.md` dropped in that folder is picked up
//! by the next build.
//!
//! | Module | What |
//! |---|---|
//! | [`guide`] | One guide, read from its file |
//! | [`library`] | Every guide: the catalogue, one guide's body, the line each tool lists its guides on |

use include_dir::{include_dir, Dir};

pub mod guide;
pub mod library;

pub use guide::Guide;
pub use library::Library;

/// The guides, as written in `mcp/src/guides/`.
static GUIDES: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../mcp/src/guides");

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

    /// **Every guide written in `mcp/src/guides/` reads**, its name its
    /// file's, and the library they make holds each once.
    #[test]
    fn every_embedded_guide_reads() {
        let library = embedded().unwrap();
        assert!(library.count() >= 40, "{} guides", library.count());
        for name in [
            "thread",
            "delegate",
            "research",
            "verify",
            "deploy",
            "test",
            "bots",
            "guide-change",
            "journal-restructure",
            "yen-apps",
            "claude",
            "memory",
            "drive",
            "approvals",
            "groups",
            "files",
        ] {
            assert!(library.get(name).is_some(), "{name}");
        }
    }

    /// **Each client is given its own short guide first**, written on a
    /// budget since it rides the first call of every conversation
    /// (`docs/mcp_user_context.md` §2.4): Grok its `grok`, every Claude client
    /// `claude`. Both point to the full `memory` guide, read on demand, which
    /// says how a topic is added and who restructures the tree.
    #[test]
    fn each_client_is_given_its_short_guide_and_memory_is_read_on_demand() {
        let library = embedded().unwrap();
        let grok = library.for_client("grok").unwrap();
        assert_eq!(grok.name, "grok");
        let claude = library.for_client("claude-ai").unwrap();
        assert_eq!(claude.name, "claude");
        for client in ["claude-code", "other"] {
            assert_eq!(library.for_client(client).unwrap().name, "claude");
        }
        for short in [grok, claude] {
            assert!(
                short.body.len() < 2_400,
                "{}: {} bytes",
                short.name,
                short.body.len()
            );
            assert!(short.body.contains("`memory`"), "{}", short.name);
            assert!(short.body.contains("journal_focus"), "{}", short.name);
        }
        assert!(claude.body.contains("`yen-apps`"));
        let memory = library.get("memory").unwrap();
        assert!(memory.clients.is_empty(), "memory is read on demand");
        for says in [
            "journal_focus",
            "journal_create",
            "journal_accept",
            "`journals` checkout",
            "replaces",
        ] {
            assert!(memory.body.contains(says), "memory: {says}");
        }
        assert!(library.get("journal").is_none() && library.get("knowledge").is_none());
    }

    /// **Every guide a guide sends the reader to exists**: each "`<name>`
    /// guide" in any guide's text names one in the library.
    #[test]
    fn every_guide_named_in_a_guide_exists() {
        let library = embedded().unwrap();
        let mut missing = Vec::new();
        for name in library.names() {
            let body = &library.get(name).unwrap().body;
            let mut rest = body.as_str();
            while let Some(at) = rest.find("` guide") {
                let before = &rest[..at];
                if let Some(open) = before.rfind('`') {
                    let named = &before[open + 1..];
                    let is_name = !named.is_empty()
                        && named
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
                    if is_name && library.get(named).is_none() {
                        missing.push(format!("{name} → {named}"));
                    }
                }
                rest = &rest[at + 1..];
            }
        }
        assert!(missing.is_empty(), "guides naming no guide: {missing:?}");
    }
}
