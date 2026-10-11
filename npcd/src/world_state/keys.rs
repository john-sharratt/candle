//! Where a world's state records live (`docs/replicated_substrate.md` §9.1):
//! one keyed record per entity, `world/<wid>/<entity>`. An entity's own path
//! is made of segments, and a segment that is a name from the world — a body,
//! a document path — is escaped so it never reads as two.

/// The prefix every record of world `wid` is under.
pub fn world_prefix(wid: &str) -> String {
    format!("world/{}/", escape(wid))
}

/// The key of entity `entity` of world `wid`.
pub fn key(wid: &str, entity: &str) -> String {
    format!("{}{entity}", world_prefix(wid))
}

/// The entity a key under `wid`'s prefix names, or `None` for another
/// world's key.
pub fn entity<'a>(wid: &str, key: &'a str) -> Option<&'a str> {
    key.strip_prefix(&world_prefix(wid))
}

/// `name` as one segment of an entity's path: `%` and `/` written `%25` and
/// `%2F`.
pub fn escape(name: &str) -> String {
    name.replace('%', "%25").replace('/', "%2F")
}

/// A segment back as the name it was.
pub fn unescape(segment: &str) -> String {
    segment.replace("%2F", "/").replace("%25", "%")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_is_the_worlds_prefix_and_the_entity() {
        assert_eq!(key("vault", "table"), "world/vault/table");
        assert_eq!(
            entity("vault", "world/vault/body/maker-1"),
            Some("body/maker-1")
        );
        assert_eq!(entity("vault", "world/vaults/table"), None);
        assert_eq!(world_prefix("a/b"), "world/a%2Fb/");
    }

    /// A document path is one segment however many slashes it has, and a
    /// name with a literal `%2F` in it comes back as itself.
    #[test]
    fn a_segment_escapes_and_comes_back() {
        for name in ["notes/a.md", "100%", "odd%2Fname", "plain"] {
            let s = escape(name);
            assert!(!s.contains('/'), "{s}");
            assert_eq!(unescape(&s), name);
        }
        assert_eq!(escape("notes/a.md"), "notes%2Fa.md");
        assert_eq!(escape("100%"), "100%25");
    }
}
