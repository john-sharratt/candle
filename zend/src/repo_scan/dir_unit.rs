//! A `repo_map` unit ready to render: a folder as a branch lists it
//! ([`FolderUnit`], `docs/zend_branch_ingest.md` §6.2), with the anchor
//! excerpt and manifest hint its turns show read from the commit it was
//! found on.

use super::anchor::{self, Anchor};
use super::types::ModuleHint;
use crate::branch_ingest::manifest;
use crate::branch_ingest::units::FolderUnit;

/// One directory's ingest unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirUnit {
    /// Workspace-relative directory path with a trailing slash
    /// (`candle/zend/src/`), or `"."` for the workspace — the gather-scope
    /// tag and the conversation's `dir`.
    pub dir: String,
    /// The files its listing's first page shows, as the key saw them.
    pub listed: Vec<String>,
    /// The excerpt describing the folder, when one of its files provides it.
    pub anchor: Option<Anchor>,
    /// The hint its manifest gives, when it has one that parses.
    pub module_hint: Option<ModuleHint>,
    /// The unit's content key.
    pub content_key: String,
}

impl DirUnit {
    /// `unit` with its anchor excerpt and manifest hint read through `read`,
    /// which returns a workspace-relative file's bytes. A file `read` cannot
    /// return gives no excerpt or hint, as a file that yields none does.
    pub fn read(unit: &FolderUnit, read: impl Fn(&str) -> Option<Vec<u8>>) -> Self {
        let anchor = unit
            .anchor
            .as_ref()
            .and_then(|a| anchor::excerpt(&a.path, a.language, &read(&a.path)?));
        let module_hint = unit.manifests.iter().find_map(|m| {
            let name = m.path.rsplit('/').next().unwrap_or(&m.path);
            manifest::hint(name, &read(&m.path)?)
        });
        Self {
            dir: unit.dir.clone(),
            listed: unit.listed.clone(),
            anchor,
            module_hint,
            content_key: unit.key.clone(),
        }
    }

    /// The directory the folder's `file_list` call uses. The workspace root
    /// lists with an empty path, matching how the live tool addresses it.
    pub fn list_path(&self) -> &str {
        if self.dir == "." {
            ""
        } else {
            &self.dir
        }
    }

    /// Human-facing directory label for the summarise request.
    pub fn label(&self) -> &str {
        &self.dir
    }

    /// The hint rendered into the summarise request, so a crate or package
    /// root announces itself rather than leaving the model to infer it from
    /// a filename in the listing.
    pub fn module_hint(&self) -> Option<&ModuleHint> {
        self.module_hint.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use zend_tools::ToolContext;
    use zend_vfs::{Oid, RepoSpec, Workspace};

    use super::*;
    use crate::branch_ingest::filter::IngestScope;
    use crate::branch_ingest::units::{
        folder_units, test_tree, test_units, workspace_unit, TreeFile,
    };
    use crate::repo_scan::types::Language;

    const BLOB: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

    fn unit(files: &[(&str, Language)]) -> FolderUnit {
        test_units(files).remove(0)
    }

    /// **The anchor and the manifest hint come from the bytes read**, and
    /// the key is the unit's.
    #[test]
    fn the_anchor_and_hint_are_read_from_the_units_files() {
        let u = unit(&[
            ("a/Cargo.toml", Language::Toml),
            ("a/lib.rs", Language::Rust),
            ("a/x.rs", Language::Rust),
        ]);
        let bytes: HashMap<&str, &[u8]> = HashMap::from([
            ("a/Cargo.toml", &b"[package]\nname = \"demo\"\n"[..]),
            ("a/lib.rs", b"//! The demo crate.\npub fn x() {}\n"),
        ]);
        let d = DirUnit::read(&u, |path| bytes.get(path).map(|b| b.to_vec()));
        assert_eq!(d.dir, "a/");
        assert_eq!(d.content_key, u.key);
        assert_eq!(
            d.module_hint(),
            Some(&ModuleHint::CargoPackage {
                name: "demo".into()
            })
        );
        let anchor = d.anchor.expect("lib.rs describes the folder");
        assert_eq!(anchor.path, "a/lib.rs");
        assert!(anchor.body.contains("The demo crate."));

        let unread = DirUnit::read(&u, |_| None);
        assert_eq!((unread.anchor, unread.module_hint), (None, None));
    }

    #[test]
    fn the_workspace_lists_with_an_empty_path_and_a_folder_with_its_own() {
        let root = DirUnit::read(&workspace_unit(&["a".into()]), |_| None);
        assert_eq!(root.list_path(), "");
        let nested = DirUnit::read(&unit(&[("a/zend/src/x.rs", Language::Rust)]), |_| None);
        assert_eq!(nested.list_path(), "a/zend/src/");
        assert_eq!(nested.label(), "a/zend/src/");
    }

    /// **`listed` is the listing the turn shows, not the files the layer
    /// reads**: the extension allowlist and the size cap are the layer's, and
    /// `file_list` shows everything — so a file the layer never reads is in
    /// the listing, and in the key, all the same.
    #[test]
    fn listed_is_the_listing_the_turn_shows() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(d.path().join("k")).unwrap();
        let names = ["api.rs", "decode.cu", "LICENSE"];
        for (name, body) in
            names
                .iter()
                .zip(["//! Kernel wrapper.\n", "__global__ void d() {}\n", "MIT\n"])
        {
            std::fs::write(d.path().join("k").join(name), body).unwrap();
        }
        let scope = IngestScope::new("", None);
        let read: Vec<TreeFile> = names
            .iter()
            .filter_map(|name| {
                Some(TreeFile {
                    path: format!("k/{name}"),
                    blob: Oid::parse(BLOB).unwrap(),
                    size: 1,
                    language: scope.admits("k", name, 1)?,
                })
            })
            .collect();
        assert_eq!(read.len(), 2, "the layer does not read LICENSE");
        let tree = test_tree(&names.map(|name| (name, BLOB)));
        let u = folder_units("k", &tree, &read).remove(0);

        let workspace = Workspace::new(d.path(), vec![RepoSpec::named("k")]).unwrap();
        let ctx = ToolContext::with_workspace(workspace);
        let shown = zend_tools::run("file_list", "test", &serde_json::json!({"repo": "k"}), &ctx);
        let listed_by_tool: Vec<String> = shown["entries"]
            .as_array()
            .expect("entries array")
            .iter()
            .map(|f| format!("k/{}", f["path"].as_str().expect("path")))
            .collect();
        assert_eq!(listed_by_tool, ["k/LICENSE", "k/api.rs", "k/decode.cu"]);
        assert_eq!(u.listed, listed_by_tool);
    }
}
