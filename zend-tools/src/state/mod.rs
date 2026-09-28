//! In-memory state stores owned by a [`crate::ToolContext`].
//!
//! All stores use interior locking (`RwLock` or `Mutex`) so they can be shared
//! across tool invocations via `Arc` without the context itself needing to be
//! mutable.  None of these stores persist to disk — everything lives in process
//! memory for the lifetime of the conversation.
//!
//! | Module | Store | Used by |
//! |--------|-------|---------|
//! | [`credentials`] | [`CredentialStore`] | `credential_*` tools, session opens |
//! | [`notes`] | [`NotesStore`] | `notes_*` tools — cross-conversation KV store |
//! | [`sessions`] | [`SessionRegistry`] | All session tool groups |
//! | [`hash_state`] | [`HashStateStore`] | `hash_state_*` streaming hash tools |
//! | [`secrets`] | [`Secrets`] | `web_search`, git — the daemon's API keys and tokens |
//!
//! The workspace's repositories and the per-conversation file overlays the
//! `file_*` tools work through live in `zend_vfs`.
//!
//! [`Secrets`] is the one store that is not in-memory-only and not
//! conversation-scoped: it is read once from a per-user file on disk, outside
//! the workspace, and is the same for every conversation the daemon serves.

pub mod credentials;
pub mod hash_state;
pub mod notes;
pub mod secrets;
mod secrets_exposure;
pub mod sessions;

pub use credentials::CredentialStore;
pub use hash_state::HashStateStore;
pub use notes::NotesStore;
pub use secrets::{Secrets, SecretsError};
pub use sessions::SessionRegistry;
