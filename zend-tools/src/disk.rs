//! Every way a tool opens host storage outside the file store, refused without
//! [`Capability::DiskWrite`](crate::Capability::DiskWrite).
//!
//! The workspace files go through [`VfsStore`](crate::state::VfsStore), which
//! writes the disk only when built with a [`DiskWriteGrant`]. A SQLite
//! connection is a second route to the disk that the store never sees — a path
//! opens or creates a file, and even `:memory:` can `ATTACH` or `VACUUM INTO`
//! one — so opening any connection takes the same proof.

use rusqlite::{Connection, Result};

use crate::grants::DiskWriteGrant;

/// A SQLite connection to `path` (or `:memory:`).
pub fn sqlite_open(_grant: &DiskWriteGrant, path: &str) -> Result<Connection> {
    Connection::open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grants::{Capability, Grants, NotPermitted};
    use crate::source_scan::tool_sources_containing;

    #[test]
    fn a_connection_needs_the_disk_write_capability() {
        assert_eq!(
            Grants::NONE.disk_write().unwrap_err(),
            NotPermitted(Capability::DiskWrite)
        );
        let grant = Grants::NONE
            .with(Capability::DiskWrite)
            .disk_write()
            .unwrap();
        assert!(sqlite_open(&grant, ":memory:").is_ok());
    }

    /// **Tool code changes the disk only through the file store.** The one
    /// exception is a key credential handed to an SSH/SFTP library that takes
    /// a key *file*: it is written to the temp directory and removed after the
    /// handshake, and both tools are declared `network` + `secrets`.
    #[test]
    fn tools_write_the_disk_only_through_the_file_store() {
        const KEY_FILE_WRITERS: [&str; 2] = ["ssh/open.rs", "remote_fs/open.rs"];
        let offenders: Vec<_> = tool_sources_containing(&[
            "fs::write",
            "File::create",
            "OpenOptions",
            "remove_file",
            "remove_dir",
            "create_dir",
            "fs::rename",
            "fs::copy",
        ])
        .into_iter()
        .filter(|(file, _, _)| {
            !KEY_FILE_WRITERS
                .iter()
                .any(|k| file.replace('\\', "/").ends_with(k))
        })
        .collect();
        assert!(
            offenders.is_empty(),
            "tool code writing the disk directly: {offenders:#?}"
        );
    }

    /// **Tool code opens a database only through this module.**
    #[test]
    fn tools_open_databases_only_through_this_module() {
        let offenders = tool_sources_containing(&["Connection::open"]);
        assert!(
            offenders.is_empty(),
            "tool code opening a database directly: {offenders:#?}"
        );
    }
}
