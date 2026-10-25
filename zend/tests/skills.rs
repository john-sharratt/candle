//! The skills Claude follows to use zend (`skills/`,
//! `docs/claude_zend_gateway.md` §9, §13.4) meet the rules an upload to
//! claude.ai enforces: each folder holds a `SKILL.md` whose frontmatter is
//! `name` and `description` alone, `name` matches the folder — lowercase
//! letters, digits and hyphens, at most 64 characters — the description is at
//! most 1,024 characters, and the file is under 500 lines.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn skills_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../skills")
}

/// The skills the design names (§7, §9), and the thread skill they follow.
const DESIGNED: [&str; 13] = [
    "zend-thread",
    "zend-implement",
    "zend-review",
    "zend-research",
    "zend-ask",
    "cluster-status",
    "cluster-logs",
    "zend-test",
    "zend-deploy",
    "zend-release",
    "zend-rollback",
    "cluster-care",
    "zend-watch",
];

/// `text`'s frontmatter as its keys and values, or why it has none.
fn frontmatter(text: &str) -> Result<Vec<(String, String)>, String> {
    let rest = text
        .strip_prefix("---\n")
        .ok_or("the file does not open with `---`")?;
    let end = rest
        .find("\n---\n")
        .ok_or("the frontmatter does not close")?;
    rest[..end]
        .lines()
        .map(|line| {
            let (k, v) = line
                .split_once(": ")
                .ok_or_else(|| format!("`{line}` is not `key: value`"))?;
            Ok((k.to_string(), v.to_string()))
        })
        .collect()
}

#[test]
fn every_designed_skill_is_there_and_nothing_else_is() {
    let found: BTreeSet<String> = fs::read_dir(skills_dir())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let designed: BTreeSet<String> = DESIGNED.iter().map(|s| s.to_string()).collect();
    assert_eq!(found, designed);
}

#[test]
fn every_skill_meets_the_upload_rules() {
    for name in DESIGNED {
        let path = skills_dir().join(name).join("SKILL.md");
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .replace("\r\n", "\n");
        let fm = frontmatter(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        let keys: Vec<&str> = fm.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["name", "description"], "{name}");
        assert_eq!(fm[0].1, name, "the name is the folder's");
        assert!(
            name.len() <= 64
                && name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "{name}"
        );
        let description = &fm[1].1;
        assert!(
            !description.is_empty() && description.chars().count() <= 1_024,
            "{name}: a description of {} characters",
            description.chars().count()
        );
        assert!(text.lines().count() < 500, "{name} is too long");
    }
}

/// **Every operation that changes what the cluster runs names the user's
/// approval**, in the endpoint's own words, wherever a skill sends one.
#[test]
fn a_skill_that_sends_a_decision_says_how_it_is_approved() {
    for name in ["zend-release", "zend-rollback", "cluster-care"] {
        let text = fs::read_to_string(skills_dir().join(name).join("SKILL.md")).unwrap();
        assert!(
            text.contains("confirm: \"approved by the user\""),
            "{name} sends a decision without the approval gate"
        );
    }
}
