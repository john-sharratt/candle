//! Names and paths (`docs/claude_zend_gateway.md` §5.2).
//!
//! The call that creates a conversation names it: a thread by `zend_attach` or
//! `zend_task`, a worker by `zend_task`. A name is a kebab-case slug saying
//! what the conversation is for. A path is the names down the lineage:
//! `T:<thread>` for a thread, `T:<thread>/<worker>` for a worker under it.

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

/// A conversation `/mcp` addresses by path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Path {
    /// `T:<thread>`.
    Thread(String),
    /// `T:<thread>/<worker>`, the worker's name possibly numbered (`name.2`).
    Worker { thread: String, worker: String },
}

impl Path {
    /// Read `text` as a path, refusing anything that is not one.
    pub fn parse(text: &str) -> Result<Self, String> {
        let Some(rest) = text.strip_prefix("T:") else {
            return Err(format!(
                "`{text}` is not a path: a thread is `T:<name>`, a worker under it \
                 `T:<thread>/<name>`"
            ));
        };
        match rest.split_once('/') {
            None => check(rest).map(|()| Self::Thread(rest.to_string())),
            Some((thread, worker)) => {
                check(thread)?;
                let (base, n) = split_numbered(worker);
                check(base)?;
                if n == Some(0) || n == Some(1) {
                    return Err(format!(
                        "`{text}` is not a path: a numbered sibling starts at .2"
                    ));
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

/// The name a new worker under a thread takes, asked for as `wanted`, beside
/// siblings named `taken` (each `(name, live)`).
///
/// - `wanted` equal to the thread's own name is refused: a name may not repeat
///   within its lineage.
/// - A live sibling of the same base name is refused, the clash named.
/// - Siblings of that name that have ended leave the next number free:
///   `verify`, then `verify.2`, `verify.3`.
pub fn worker_name(thread: &str, wanted: &str, taken: &[(String, bool)]) -> Result<String, String> {
    check(wanted)?;
    if wanted == thread {
        return Err(format!(
            "`{wanted}` is already the name of an ancestor (`T:{thread}`); choose another"
        ));
    }
    let same: Vec<(Option<u32>, bool)> = taken
        .iter()
        .filter_map(|(name, live)| {
            let (base, n) = split_numbered(name);
            (base == wanted).then_some((n, *live))
        })
        .collect();
    if let Some((n, _)) = same.iter().find(|(_, live)| *live) {
        let name = match n {
            Some(n) => format!("{wanted}.{n}"),
            None => wanted.to_string(),
        };
        return Err(format!(
            "`{wanted}` is the name of a worker still running (`T:{thread}/{name}`); choose another"
        ));
    }
    let highest = same.iter().map(|(n, _)| n.unwrap_or(1)).max();
    Ok(match highest {
        None => wanted.to_string(),
        Some(n) => format!("{wanted}.{}", n + 1),
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
        for bad in ["review", "T:", "T:Review", "T:a/B", "T:a/b.1", "P:candle"] {
            assert!(Path::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn an_ancestors_name_and_a_live_siblings_are_refused() {
        let err = worker_name("verify", "verify", &[]).unwrap_err();
        assert!(err.contains("ancestor (`T:verify`)"), "{err}");
        let taken = [("check".to_string(), true)];
        let err = worker_name("t", "check", &taken).unwrap_err();
        assert!(err.contains("still running (`T:t/check`)"), "{err}");
        let taken = [("check".to_string(), false), ("check.2".to_string(), true)];
        let err = worker_name("t", "check", &taken).unwrap_err();
        assert!(err.contains("`T:t/check.2`"), "{err}");
    }

    #[test]
    fn an_ended_siblings_name_is_numbered_on() {
        assert_eq!(worker_name("t", "check", &[]).unwrap(), "check");
        let one = [("check".to_string(), false)];
        assert_eq!(worker_name("t", "check", &one).unwrap(), "check.2");
        let two = [("check".to_string(), false), ("check.2".to_string(), false)];
        assert_eq!(worker_name("t", "check", &two).unwrap(), "check.3");
        let cousin = [("checker".to_string(), true)];
        assert_eq!(worker_name("t", "check", &cousin).unwrap(), "check");
    }
}
