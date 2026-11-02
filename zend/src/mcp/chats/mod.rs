//! Zen chat (`docs/zen_chat.md`): a private conversation with Zen for the life
//! of one MCP session, disposed of — its conversation tombstoned, every record
//! it made ceased — when the session is done.
//!
//! | Module | What |
//! |---|---|
//! | [`naming`] | A chat's id, its thread `~<id>`, and Zen's worker on it |
//! | [`record`] | The chat and its messages, as records |
//! | [`talk`] | A send, a read, and Zen's answer said as a message |
//! | [`dispose`] | Disposing of a chat: tombstone its conversations, cease its records |
//! | [`sweep`] | The sweep that disposes of the chats whose session is done |

pub mod dispose;
pub mod naming;
pub mod record;
pub mod sweep;
pub mod talk;
