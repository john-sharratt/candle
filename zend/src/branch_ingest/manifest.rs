//! A folder's manifest — `Cargo.toml`, `package.json`, `pyproject.toml`,
//! `go.mod` — and the hint it gives the folder's summary request
//! (`(crate: candle-nn)`). Read from the manifest's bytes, best effort: a
//! manifest that does not parse gives no hint.

use crate::repo_scan::types::ModuleHint;

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
    if let Some(members) = workspace_members(body) {
        return Some(ModuleHint::CargoWorkspace { members });
    }
    table_name(body, "[package]").map(|name| ModuleHint::CargoPackage { name })
}

fn workspace_members(body: &str) -> Option<usize> {
    let ws_start = body.find("[workspace]")?;
    let after = &body[ws_start + "[workspace]".len()..];
    let members_idx = after.find("members")?;
    let after_members = &after[members_idx..];
    let array_start = after_members.find('[')?;
    let array_end = after_members[array_start..].find(']')?;
    let inside = &after_members[array_start + 1..array_start + array_end];
    Some(
        inside
            .split(',')
            .map(|s| s.trim().trim_matches('"'))
            .filter(|s| !s.is_empty())
            .count(),
    )
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

    #[test]
    fn a_cargo_workspace_counts_its_members() {
        let body = b"[workspace]\nmembers = [\"a\", \"b\", \"c\"]\n";
        assert_eq!(
            hint("Cargo.toml", body),
            Some(ModuleHint::CargoWorkspace { members: 3 })
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

    #[test]
    fn a_manifest_that_does_not_parse_gives_no_hint() {
        assert_eq!(hint("package.json", b"{not json"), None);
        assert_eq!(hint("Cargo.toml", b"[dependencies]\nx = 1\n"), None);
        assert_eq!(hint("Cargo.toml", b"\xff\xfe"), None);
        assert_eq!(hint("README.md", b"# x\n"), None);
    }
}
