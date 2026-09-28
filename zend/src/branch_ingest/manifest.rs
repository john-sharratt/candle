//! A folder's manifest — `Cargo.toml`, `package.json`, `pyproject.toml`,
//! `go.mod` — and the hint it gives the folder's summary request
//! (`(crate: candle-nn)`). Read from the manifest's bytes, best effort: a
//! manifest that does not parse gives no hint.

use std::collections::HashMap;

use zend_vfs::Oid;

use crate::repo_scan::types::ModuleHint;

/// Each manifest version's hint, read once: `(file name, blob id)` → the
/// hint, `None` for a manifest that gives none or could not be read. A blob
/// id names bytes, so an entry never goes stale and a manifest shared by any
/// number of branches or trees is parsed once.
#[derive(Debug, Default)]
pub struct Hints {
    by_blob: HashMap<(String, Oid), Option<ModuleHint>>,
}

impl Hints {
    /// The hint the manifest `basename` at `blob` gives, reading its bytes
    /// through `read` only the first time that version is asked for.
    pub fn of(
        &mut self,
        basename: &str,
        blob: &Oid,
        read: impl FnOnce(&Oid) -> Option<Vec<u8>>,
    ) -> Option<ModuleHint> {
        self.by_blob
            .entry((basename.to_string(), blob.clone()))
            .or_insert_with(|| hint(basename, &read(blob)?))
            .clone()
    }
}

/// The file names a folder's manifest can have.
const MANIFESTS: [&str; 4] = ["Cargo.toml", "package.json", "pyproject.toml", "go.mod"];

/// Whether a file called `basename` is a manifest.
pub fn is_manifest(basename: &str) -> bool {
    MANIFESTS.contains(&basename)
}

/// The hint a manifest called `basename` gives, from its `bytes`.
pub fn hint(basename: &str, bytes: &[u8]) -> Option<ModuleHint> {
    let body = std::str::from_utf8(bytes).ok()?;
    match basename {
        "Cargo.toml" => cargo_hint(body),
        "package.json" => node_hint(body),
        "pyproject.toml" => pyproject_hint(body),
        "go.mod" => go_mod_hint(body),
        _ => None,
    }
}

fn cargo_hint(body: &str) -> Option<ModuleHint> {
    // A hand-roll rather than a TOML parser: the shapes Cargo writes are few,
    // and a hint that fails to parse is simply absent.
    if has_workspace_members(body) {
        return Some(ModuleHint::CargoWorkspace);
    }
    table_name(body, "[package]").map(|name| ModuleHint::CargoPackage { name })
}

/// Whether `body` is a workspace manifest: a `[workspace]` table followed by a
/// `members` array. The array is only ever *detected*, never counted — see
/// [`ModuleHint::CargoWorkspace`] for why the count does not travel.
fn has_workspace_members(body: &str) -> bool {
    let Some(ws_start) = body.find("[workspace]") else {
        return false;
    };
    let after = &body[ws_start + "[workspace]".len()..];
    let Some(members_idx) = after.find("members") else {
        return false;
    };
    let after_members = &after[members_idx..];
    let Some(array_start) = after_members.find('[') else {
        return false;
    };
    after_members[array_start..].contains(']')
}

/// The `name = "…"` inside the TOML table `header`, before the next table.
fn table_name(body: &str, header: &str) -> Option<String> {
    let start = body.find(header)?;
    let after = &body[start + header.len()..];
    let section = &after[..after.find('[').unwrap_or(after.len())];
    section.lines().find_map(|line| {
        let rest = line.trim_start().strip_prefix("name")?.trim_start();
        let value = rest.strip_prefix('=')?.trim();
        value
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .map(str::to_string)
    })
}

fn node_hint(body: &str) -> Option<ModuleHint> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    Some(ModuleHint::NodePackage { name })
}

fn pyproject_hint(body: &str) -> Option<ModuleHint> {
    table_name(body, "[project]").map(|name| ModuleHint::PythonProject { name })
}

fn go_mod_hint(body: &str) -> Option<ModuleHint> {
    body.lines().find_map(|line| {
        let name = line.trim_start().strip_prefix("module")?.trim();
        (!name.is_empty()).then(|| ModuleHint::GoModule {
            name: name.to_string(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_four_manifests_are_manifests() {
        for name in ["Cargo.toml", "package.json", "pyproject.toml", "go.mod"] {
            assert!(is_manifest(name), "{name}");
        }
        for name in ["cargo.toml", "Cargo.lock", "go.sum", "README.md"] {
            assert!(!is_manifest(name), "{name}");
        }
    }

    /// The member array is detected, never counted: the rendered hint states the
    /// folder's role and carries nothing specific to this checkout.
    #[test]
    fn a_cargo_workspace_is_named_without_a_member_count() {
        let body = b"[workspace]\nmembers = [\"a\", \"b\", \"c\"]\n";
        assert_eq!(hint("Cargo.toml", body), Some(ModuleHint::CargoWorkspace));
        assert_eq!(
            ModuleHint::CargoWorkspace.render(),
            "Cargo workspace root",
            "a workspace hint must not carry a member count — it does not \
             generalize and it leaked into unrelated dialogue",
        );
    }

    #[test]
    fn a_cargo_package_names_itself() {
        let body = b"[package]\nname = \"my-crate\"\nversion = \"0.1.0\"\n";
        assert_eq!(
            hint("Cargo.toml", body),
            Some(ModuleHint::CargoPackage {
                name: "my-crate".into()
            })
        );
    }

    #[test]
    fn node_python_and_go_manifests_name_their_module() {
        assert_eq!(
            hint("package.json", br#"{"name":"my-app","version":"1.0.0"}"#),
            Some(ModuleHint::NodePackage {
                name: "my-app".into()
            })
        );
        assert_eq!(
            hint("pyproject.toml", b"[project]\nname = \"tool\"\n"),
            Some(ModuleHint::PythonProject {
                name: "tool".into()
            })
        );
        assert_eq!(
            hint("go.mod", b"module example.com/me/widget\ngo 1.22\n"),
            Some(ModuleHint::GoModule {
                name: "example.com/me/widget".into()
            })
        );
    }

    /// **A manifest version is read once**, whoever asks: the second ask for
    /// the same blob never reads, and a different blob does.
    #[test]
    fn each_manifest_version_is_read_once() {
        let a = Oid::parse("ce013625030ba8dba906f756967f9e9ca394464a").unwrap();
        let b = Oid::parse("4b825dc642cb6eb9a060e54bf8d69288fbee4904").unwrap();
        let mut hints = Hints::default();
        let reads = std::cell::Cell::new(0);
        let read = |bytes: &'static [u8]| {
            let reads = &reads;
            move |_: &Oid| {
                reads.set(reads.get() + 1);
                Some(bytes.to_vec())
            }
        };
        let package = Some(ModuleHint::CargoPackage { name: "x".into() });
        assert_eq!(
            hints.of("Cargo.toml", &a, read(b"[package]\nname = \"x\"\n")),
            package
        );
        assert_eq!(hints.of("Cargo.toml", &a, read(b"unread")), package);
        assert_eq!(reads.get(), 1, "the same version is not read again");
        assert_eq!(
            hints.of("Cargo.toml", &b, read(b"[workspace]\nmembers = [\"a\"]\n")),
            Some(ModuleHint::CargoWorkspace)
        );
        assert_eq!(reads.get(), 2);
        assert_eq!(
            hints.of("Cargo.toml", &b, |_| None),
            Some(ModuleHint::CargoWorkspace)
        );
    }

    #[test]
    fn a_manifest_that_does_not_parse_gives_no_hint() {
        assert_eq!(hint("package.json", b"{not json"), None);
        assert_eq!(hint("Cargo.toml", b"[dependencies]\nx = 1\n"), None);
        assert_eq!(hint("Cargo.toml", b"\xff\xfe"), None);
        assert_eq!(hint("README.md", b"# x\n"), None);
    }
}
