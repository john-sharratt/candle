//! What a tool call is allowed to do to the world outside the daemon.
//!
//! A [`ToolContext`](crate::ToolContext) carries [`Grants`], set by whoever
//! built it and never by the tool call it runs. Four capabilities cover the
//! actions with effects beyond the conversation:
//!
//! | Capability | Covers |
//! |---|---|
//! | [`Capability::DiskWrite`] | changing files on the host's disk |
//! | [`Capability::Network`] | any outbound connection — HTTP, sockets, DNS, ICMP |
//! | [`Capability::Exec`] | running code or programs — the JS VM, subprocesses, remote shells, sub-agents |
//! | [`Capability::Secrets`] | reading or changing stored credentials |
//!
//! # Deny by default, checked where the action happens
//!
//! A context grants nothing unless its builder grants it. The check is made
//! twice, independently:
//!
//! 1. **At dispatch** — each registered tool declares what it needs, and
//!    [`RegisteredTool::call`](crate::RegisteredTool::call) refuses a call whose
//!    context lacks it before the tool's code runs at all.
//! 2. **At the primitive** — opening a socket, resolving a name, building an
//!    HTTP client, spawning a process, starting the JS VM, opening a database,
//!    reading a stored credential, and writing the disk each go through a
//!    function that takes the grants and refuses without the capability
//!    ([`crate::net`], [`crate::exec`], [`crate::disk`],
//!    [`ToolContext::credentials`](crate::ToolContext::credentials),
//!    [`ToolContext::http`](crate::ToolContext::http),
//!    [`VfsStore::direct`](crate::state::VfsStore::direct)). A tool whose
//!    declaration is wrong or missing still cannot perform the action.
//!
//! Source-scanning tests hold the second layer in place: no tool module may
//! reach a raw socket, process, HTTP client, database, or disk-writing
//! constructor except through those functions, and the stores behind the
//! accessors are private fields.
//!
//! Which caller gets which grants is the daemon's decision (in `zend`, from the
//! caller's role and tools mode); this crate only enforces them.

use std::fmt;

/// One class of action with effects outside the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    DiskWrite,
    Network,
    Exec,
    Secrets,
}

impl Capability {
    pub const ALL: [Capability; 4] = [
        Capability::DiskWrite,
        Capability::Network,
        Capability::Exec,
        Capability::Secrets,
    ];

    const fn bit(self) -> u8 {
        match self {
            Capability::DiskWrite => 1,
            Capability::Network => 2,
            Capability::Exec => 4,
            Capability::Secrets => 8,
        }
    }

    /// The spelling a refusal names.
    pub fn as_str(self) -> &'static str {
        match self {
            Capability::DiskWrite => "disk_write",
            Capability::Network => "network",
            Capability::Exec => "exec",
            Capability::Secrets => "secrets",
        }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The capabilities a context holds. [`Grants::NONE`] unless granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Grants {
    bits: u8,
}

impl Grants {
    /// Nothing outside the conversation.
    pub const NONE: Grants = Grants { bits: 0 };

    /// Every capability.
    pub const ALL: Grants = Grants { bits: 0b1111 };

    /// These grants and `cap`.
    pub const fn with(self, cap: Capability) -> Grants {
        Grants {
            bits: self.bits | cap.bit(),
        }
    }

    /// Whether `cap` is held.
    pub const fn has(self, cap: Capability) -> bool {
        self.bits & cap.bit() != 0
    }

    /// `Ok` when `cap` is held; otherwise the refusal naming it.
    pub fn require(self, cap: Capability) -> Result<(), NotPermitted> {
        if self.has(cap) {
            Ok(())
        } else {
            Err(NotPermitted(cap))
        }
    }

    /// `Ok` when every one of `caps` is held; otherwise the refusal naming the
    /// first missing one.
    pub fn require_all(self, caps: &[Capability]) -> Result<(), NotPermitted> {
        caps.iter().try_for_each(|c| self.require(*c))
    }

    /// The proof of [`Capability::DiskWrite`] a disk-writing file store is
    /// built from — see [`VfsStore::direct`](crate::state::VfsStore::direct).
    pub fn disk_write(self) -> Result<DiskWriteGrant, NotPermitted> {
        self.require(Capability::DiskWrite)?;
        Ok(DiskWriteGrant { _private: () })
    }
}

/// Proof that [`Capability::DiskWrite`] was granted. Only
/// [`Grants::disk_write`] can make one, so a store that writes the disk cannot
/// be built by code that was not granted the capability.
#[derive(Debug)]
pub struct DiskWriteGrant {
    _private: (),
}

/// A call refused for want of a capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotPermitted(pub Capability);

impl NotPermitted {
    /// The stable error code every refusal carries.
    pub const CODE: &'static str = "not_permitted";
}

impl fmt::Display for NotPermitted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this action needs the `{}` permission, which this conversation does not have; \
             nothing was done",
            self.0
        )
    }
}

impl std::error::Error for NotPermitted {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_granted_by_default() {
        let g = Grants::default();
        assert_eq!(g, Grants::NONE);
        for cap in Capability::ALL {
            assert!(!g.has(cap));
            assert_eq!(g.require(cap), Err(NotPermitted(cap)));
        }
        assert!(g.disk_write().is_err());
    }

    #[test]
    fn each_capability_is_granted_alone() {
        for cap in Capability::ALL {
            let g = Grants::NONE.with(cap);
            for other in Capability::ALL {
                assert_eq!(g.has(other), other == cap, "{cap} granted {other}");
            }
        }
        assert!(Grants::NONE
            .with(Capability::DiskWrite)
            .disk_write()
            .is_ok());
        for cap in Capability::ALL {
            assert!(Grants::ALL.has(cap));
        }
    }

    #[test]
    fn require_all_names_the_first_missing_capability() {
        let g = Grants::NONE.with(Capability::Network);
        assert_eq!(
            g.require_all(&[Capability::Network, Capability::Exec]),
            Err(NotPermitted(Capability::Exec))
        );
        assert!(g.require_all(&[Capability::Network]).is_ok());
        assert!(g.require_all(&[]).is_ok());
    }
}
