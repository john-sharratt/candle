//! The builds a member never tries again (§3.9): each one an install of which
//! failed and was restored here, kept in `<data>/upgrade/refused.json` so a
//! restart remembers them. Only a forced rollout naming the member retries one
//! (§3.11).

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::Path;

pub const REFUSED: &str = "refused.json";

/// The refused builds, `count-hash`; none when the file is absent.
pub fn read(upgrade_dir: &Path) -> BTreeSet<String> {
    fs::read(upgrade_dir.join(REFUSED))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Add `build` to the refused builds.
pub fn refuse(upgrade_dir: &Path, build: &str) -> io::Result<BTreeSet<String>> {
    let mut refused = read(upgrade_dir);
    refused.insert(build.to_string());
    fs::create_dir_all(upgrade_dir)?;
    let tmp = upgrade_dir.join(format!("{REFUSED}.tmp"));
    fs::write(
        &tmp,
        serde_json::to_vec(&refused).expect("a set of names serializes"),
    )?;
    fs::rename(tmp, upgrade_dir.join(REFUSED))?;
    Ok(refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_build_is_remembered() {
        let t = tempfile::TempDir::new().unwrap();
        assert!(read(t.path()).is_empty());
        refuse(t.path(), "12-h12").unwrap();
        let both = refuse(t.path(), "13-h13").unwrap();
        assert_eq!(both, read(t.path()));
        assert_eq!(
            fs::read_to_string(t.path().join(REFUSED)).unwrap(),
            r#"["12-h12","13-h13"]"#
        );
    }
}
