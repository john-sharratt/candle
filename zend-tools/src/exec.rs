//! Every way a tool starts a program, refused without [`Capability::Exec`].
//!
//! The JS VM checks the same capability itself (`tools::code::engine`); this
//! module covers local subprocesses. Remote execution over SSH or telnet runs
//! on the remote host, and reaches it only through [`crate::net`].

use std::process::Command;

use crate::grants::{Capability, Grants, NotPermitted};

/// A [`Command`] for `program`, when the context may run programs.
pub fn command(grants: Grants, program: &str) -> Result<Command, NotPermitted> {
    grants.require(Capability::Exec)?;
    Ok(Command::new(program))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_scan::tool_sources_containing;

    #[test]
    fn no_program_is_started_without_the_exec_capability() {
        assert_eq!(
            command(Grants::NONE, "ping").unwrap_err(),
            NotPermitted(Capability::Exec)
        );
        assert!(command(Grants::NONE.with(Capability::Exec), "ping").is_ok());
    }

    /// **Tool code starts programs only through this module.**
    #[test]
    fn tools_start_programs_only_through_this_module() {
        let offenders = tool_sources_containing(&["Command::new", "process::Command"]);
        assert!(
            offenders.is_empty(),
            "tool code starting a program directly: {offenders:#?}"
        );
    }
}
