//! What a chat and Zen's side of it are called (`docs/zen_chat.md` §2).
//!
//! A chat's thread is `~<id>`. A name starting `~` is no name a person can give
//! a thread (`names::check` refuses it), so no `T:` path reaches a chat's
//! thread or its worker, and no thread a person attaches can collide with one.

use sha2::{Digest, Sha256};
use web::auth::oidc::random_token;

/// What starts a chat's thread name.
const MARK: char = '~';

/// The name Zen's side of a chat is asked for: `zen`, and `zen.2` after a
/// failed turn ended the first.
pub const ZEN: &str = "zen";

/// A new chat's id: 16 hex digits, from the OS's randomness.
pub fn new_id() -> String {
    let digest = Sha256::digest(random_token().as_bytes());
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// The thread the chat `id`'s workers run under.
pub fn thread_of(id: &str) -> String {
    format!("{MARK}{id}")
}

/// Whether `thread` is a chat's.
pub fn is_chat_thread(thread: &str) -> bool {
    thread.starts_with(MARK)
}

/// The chat a chat's thread is for.
pub fn id_of(thread: &str) -> Option<&str> {
    thread.strip_prefix(MARK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::names::{self, Path};

    /// **A chat's thread is no thread a person can name**: `~` is refused as a
    /// name and as a path, so neither a task nor a reader reaches it.
    #[test]
    fn a_chats_thread_is_no_name_a_person_can_give() {
        let thread = thread_of("0123456789abcdef");
        assert_eq!(thread, "~0123456789abcdef");
        assert!(is_chat_thread(&thread) && !is_chat_thread("review"));
        assert_eq!(id_of(&thread), Some("0123456789abcdef"));
        assert_eq!(id_of("review"), None);
        assert!(names::check(&thread).is_err());
        assert!(Path::parse(&format!("T:{thread}")).is_err());
        assert!(Path::parse(&format!("T:{thread}/{ZEN}")).is_err());
    }

    #[test]
    fn a_chats_id_is_sixteen_hex_digits_and_new_each_time() {
        let (a, b) = (new_id(), new_id());
        assert_eq!(a.len(), 16);
        assert!(a
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(a, b);
    }
}
