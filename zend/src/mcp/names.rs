//! Names and paths (`docs/claude_zend_gateway.md` §5.2).
//!
//! The call that creates a conversation names it: a thread by `thread_attach` or
//! `task_start`, a worker by `task_start`, a part by the worker's `sub_run`. A
//! name is a kebab-case slug saying what the conversation is for. A path is
//! the names down the lineage: `T:<thread>` for a thread,
//! `T:<thread>/<worker>` for a worker under it, and a part continues its
//! parent's, `T:<thread>/<worker>/<part>`. Below the thread, the names joined
//! by `/` are the conversation's *lineage name*, which its record is kept
//! under.

/// The longest name: long enough to say what the work is, short enough to sit
/// in a sealed header and a grep hit.
pub const MAX_NAME: usize = 40;

/// Refuse `name` unless it is a kebab-case slug of at most [`MAX_NAME`]
/// characters: lowercase letters, digits and single hyphens between them. A
/// numbered sibling's suffix (`verify-expiry.2`) is the server's to add, never
/// the caller's.
pub fn check(name: &str) -> Result<(), String> {
    let refuse = |why: &str| {
        Err(format!(
            "`{name}` is not a name: {why}. A name is a kebab-case slug of at most \
             {MAX_NAME} characters saying what the work is for, such as `verify-expiry`"
        ))
    };
    if name.is_empty() {
        return refuse("it is empty");
    }
    if name.len() > MAX_NAME {
        return refuse("it is too long");
    }
    if !name
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return refuse("only lowercase letters, digits and hyphens may appear in one");
    }
    if name.starts_with('-') || name.ends_with('-') || name.contains("--") {
        return refuse("a hyphen goes between words, one at a time");
    }
    Ok(())
}

/// A journal or conversation `/mcp` addresses by path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Path {
    /// `P:<repo>`: a project's standing rules, which every task inherits.
    Project(String),
    /// `T:<thread>`.
    Thread(String),
    /// `T:<thread>/<worker>[/<part>…]`: `worker` is the lineage name below the
    /// thread, each of its names possibly numbered (`name.2`).
    Worker { thread: String, worker: String },
}

impl Path {
    /// Read `text` as a path, refusing anything that is not one.
    pub fn parse(text: &str) -> Result<Self, String> {
        if let Some(repo) = text.strip_prefix("P:") {
            let named = !repo.is_empty()
                && repo
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
            return if named {
                Ok(Self::Project(repo.to_string()))
            } else {
                Err(format!(
                    "`{text}` is not a path: a project's journal is `P:<repository>`"
                ))
            };
        }
        let Some(rest) = text.strip_prefix("T:") else {
            return Err(format!(
                "`{text}` is not a path: a project's journal is `P:<repository>`, a thread \
                 `T:<name>`, a worker under it `T:<thread>/<name>`, and a part under a worker \
                 `T:<thread>/<worker>/<name>`"
            ));
        };
        match rest.split_once('/') {
            None => check(rest).map(|()| Self::Thread(rest.to_string())),
            Some((thread, worker)) => {
                check(thread)?;
                for name in worker.split('/') {
                    let (base, n) = split_numbered(name);
                    check(base)?;
                    if n == Some(0) || n == Some(1) {
                        return Err(format!(
                            "`{text}` is not a path: a numbered sibling starts at .2"
                        ));
                    }
                }
                Ok(Self::Worker {
                    thread: thread.to_string(),
                    worker: worker.to_string(),
                })
            }
        }
    }

    /// The path as it is printed.
    pub fn text(&self) -> String {
        match self {
            Self::Project(repo) => format!("P:{repo}"),
            Self::Thread(t) => format!("T:{t}"),
            Self::Worker { thread, worker } => format!("T:{thread}/{worker}"),
        }
    }
}

/// `name` split into its base and its sibling number: `verify.2` is
/// `("verify", Some(2))`, `verify` is `("verify", None)`.
pub fn split_numbered(name: &str) -> (&str, Option<u32>) {
    match name.rsplit_once('.') {
        Some((base, n)) => match n.parse::<u32>() {
            Ok(n) => (base, Some(n)),
            Err(_) => (name, None),
        },
        None => (name, None),
    }
}

/// The lineage name of the conversation `lineage` is a child of: `""` for a
/// worker directly under its thread, `verify` for `verify/check-plan`.
pub fn parent_of(lineage: &str) -> &str {
    lineage.rsplit_once('/').map_or("", |(parent, _)| parent)
}

/// The last name of a lineage name: `check-plan` for `verify/check-plan`.
pub fn leaf(lineage: &str) -> &str {
    lineage.rsplit_once('/').map_or(lineage, |(_, leaf)| leaf)
}

/// The lineage name a new child of `parent` (`""` for the thread itself) under
/// `thread` takes, asked for as `wanted`, beside the thread's conversations
/// `taken` (each `(lineage name, live)`).
///
/// - `wanted` equal to any ancestor's name — the thread's, or one of
///   `parent`'s — is refused: a name may not repeat within its lineage, so a
///   bare name always resolves to one conversation up the chain.
/// - A live sibling of the same base name is refused, the clash named.
/// - Siblings of that name that have ended leave the next number free:
///   `verify`, then `verify.2`, `verify.3`. The numbered name is checked
///   against the lineage too.
/// - Cousins — children of another parent — may share a name: their paths
///   differ and they never read one another's journals.
pub fn child_name(
    thread: &str,
    parent: &str,
    wanted: &str,
    taken: &[(String, bool)],
) -> Result<String, String> {
    check(wanted)?;
    let mut ancestor = format!("T:{thread}");
    let clash = |at: &str| {
        Err(format!(
            "`{wanted}` is already the name of an ancestor (`{at}`); choose another"
        ))
    };
    if wanted == thread {
        return clash(&ancestor);
    }
    for name in parent.split('/').filter(|n| !n.is_empty()) {
        ancestor = format!("{ancestor}/{name}");
        if split_numbered(name).0 == wanted {
            return clash(&ancestor);
        }
    }
    let same: Vec<(Option<u32>, bool)> = taken
        .iter()
        .filter(|(name, _)| parent_of(name) == parent)
        .filter_map(|(name, live)| {
            let (base, n) = split_numbered(leaf(name));
            (base == wanted).then_some((n, *live))
        })
        .collect();
    let under = |name: &str| match parent {
        "" => format!("T:{thread}/{name}"),
        p => format!("T:{thread}/{p}/{name}"),
    };
    if let Some((n, _)) = same.iter().find(|(_, live)| *live) {
        let name = match n {
            Some(n) => format!("{wanted}.{n}"),
            None => wanted.to_string(),
        };
        return Err(format!(
            "`{wanted}` is the name of a conversation still live (`{}`); choose another",
            under(&name)
        ));
    }
    let leaf = match same.iter().map(|(n, _)| n.unwrap_or(1)).max() {
        None => wanted.to_string(),
        Some(n) => format!("{wanted}.{}", n + 1),
    };
    Ok(match parent {
        "" => leaf,
        p => format!("{p}/{leaf}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_a_short_kebab_slug() {
        assert!(check("verify-expiry").is_ok());
        assert!(check("r2-d2").is_ok());
        for bad in [
            "",
            "Verify",
            "verify_expiry",
            "-a",
            "a-",
            "a--b",
            "a.2",
            "a b",
        ] {
            assert!(check(bad).is_err(), "{bad}");
        }
        assert!(check(&"a".repeat(MAX_NAME)).is_ok());
        assert!(check(&"a".repeat(MAX_NAME + 1)).is_err());
    }

    #[test]
    fn a_path_names_a_thread_or_a_worker_under_it() {
        assert_eq!(
            Path::parse("T:review").unwrap(),
            Path::Thread("review".into())
        );
        assert_eq!(
            Path::parse("T:review/verify.2").unwrap(),
            Path::Worker {
                thread: "review".into(),
                worker: "verify.2".into()
            }
        );
        assert_eq!(
            Path::parse("T:review/verify.2").unwrap().text(),
            "T:review/verify.2"
        );
        for bad in [
            "review",
            "T:",
            "T:Review",
            "T:a/B",
            "T:a/b.1",
            "P:",
            "P:a/b",
            "P:a b",
            "T:a/b//c",
            "T:a/b/c.0",
        ] {
            assert!(Path::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_projects_journal_is_named_by_its_repository() {
        assert_eq!(
            Path::parse("P:candle").unwrap(),
            Path::Project("candle".into())
        );
        assert_eq!(
            Path::parse("P:battle_cities.v2").unwrap().text(),
            "P:battle_cities.v2"
        );
    }

    #[test]
    fn a_part_continues_its_workers_path() {
        assert_eq!(
            Path::parse("T:review/verify.2/check-plan").unwrap(),
            Path::Worker {
                thread: "review".into(),
                worker: "verify.2/check-plan".into()
            }
        );
        assert_eq!(parent_of("verify.2/check-plan"), "verify.2");
        assert_eq!(parent_of("verify"), "");
        assert_eq!(leaf("verify.2/check-plan"), "check-plan");
        assert_eq!(leaf("verify"), "verify");
    }

    fn live(names: &[(&str, bool)]) -> Vec<(String, bool)> {
        names.iter().map(|(n, l)| (n.to_string(), *l)).collect()
    }

    #[test]
    fn an_ancestors_name_and_a_live_siblings_are_refused() {
        let err = child_name("verify", "", "verify", &[]).unwrap_err();
        assert!(err.contains("ancestor (`T:verify`)"), "{err}");
        let err = child_name("t", "", "check", &live(&[("check", true)])).unwrap_err();
        assert_eq!(
            err,
            "`check` is the name of a conversation still live (`T:t/check`); choose another"
        );
        let taken = live(&[("check", false), ("check.2", true)]);
        let err = child_name("t", "", "check", &taken).unwrap_err();
        assert!(err.contains("`T:t/check.2`"), "{err}");
    }

    /// **A part may not take any ancestor's name**: the thread's, its worker's,
    /// or a numbered worker's base name.
    #[test]
    fn a_parts_name_is_checked_against_every_ancestor() {
        let err = child_name("review", "verify.2", "verify", &[]).unwrap_err();
        assert_eq!(
            err,
            "`verify` is already the name of an ancestor (`T:review/verify.2`); choose another"
        );
        let err = child_name("review", "verify/check-plan", "review", &[]).unwrap_err();
        assert!(err.contains("(`T:review`)"), "{err}");
        assert_eq!(
            child_name("review", "verify", "check-plan", &[]).unwrap(),
            "verify/check-plan"
        );
    }

    #[test]
    fn an_ended_siblings_name_is_numbered_on() {
        assert_eq!(child_name("t", "", "check", &[]).unwrap(), "check");
        let one = live(&[("check", false)]);
        assert_eq!(child_name("t", "", "check", &one).unwrap(), "check.2");
        let two = live(&[("check", false), ("check.2", false)]);
        assert_eq!(child_name("t", "", "check", &two).unwrap(), "check.3");
        let cousin = live(&[("checker", true)]);
        assert_eq!(child_name("t", "", "check", &cousin).unwrap(), "check");
    }

    /// Siblings are the children of one parent: a live part under another
    /// worker — a cousin — never blocks the name, and a part's own number
    /// counts only its siblings.
    #[test]
    fn cousins_share_a_name_and_siblings_number_on() {
        let taken = live(&[
            ("verify", true),
            ("verify/read", true),
            ("audit", true),
            ("audit/read", false),
        ]);
        assert_eq!(
            child_name("t", "audit", "read", &taken).unwrap(),
            "audit/read.2"
        );
        let err = child_name("t", "verify", "read", &taken).unwrap_err();
        assert!(err.contains("(`T:t/verify/read`)"), "{err}");
        assert_eq!(child_name("t", "", "read", &taken).unwrap(), "read");
    }
}
