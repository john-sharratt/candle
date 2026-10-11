//! Close codes (§3.4) and what a member's dial loop does after each.

use std::fmt;

/// Why a socket was closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloseCode {
    /// The gateway is shutting down.
    GoingAway,
    /// A malformed frame or a protocol violation.
    Malformed,
    /// A proof failed, in either direction.
    Proof,
    /// A newer session of the same node was admitted.
    Replaced,
    /// Below the app's floor with no way to upgrade, or left behind by a
    /// rollout.
    Superseded,
    /// The app's group is locked to another model.
    Model,
    /// `HELLO.now_ms` is more than 60 s from the gateway's clock.
    Skew,
    /// Protocol version unsupported.
    Protocol,
}

impl CloseCode {
    pub fn code(self) -> u16 {
        match self {
            CloseCode::GoingAway => 1001,
            CloseCode::Malformed => 4400,
            CloseCode::Proof => 4401,
            CloseCode::Replaced => 4409,
            CloseCode::Superseded => 4410,
            CloseCode::Model => 4412,
            CloseCode::Skew => 4417,
            CloseCode::Protocol => 4426,
        }
    }

    /// The code a peer sent; one this protocol does not define is `None`, and
    /// the dial loop backs off from it as from any other failure.
    pub fn from_code(code: u16) -> Option<CloseCode> {
        Some(match code {
            1001 => CloseCode::GoingAway,
            4400 => CloseCode::Malformed,
            4401 => CloseCode::Proof,
            4409 => CloseCode::Replaced,
            4410 => CloseCode::Superseded,
            4412 => CloseCode::Model,
            4417 => CloseCode::Skew,
            4426 => CloseCode::Protocol,
            _ => return None,
        })
    }

    /// What the member's dial loop does next.
    pub fn after(self) -> AfterClose {
        match self {
            CloseCode::GoingAway | CloseCode::Malformed | CloseCode::Proof => AfterClose::Backoff,
            CloseCode::Model | CloseCode::Skew => AfterClose::AtCap,
            CloseCode::Replaced => AfterClose::StopSocket,
            CloseCode::Superseded => AfterClose::StopApp,
            CloseCode::Protocol => AfterClose::StopSocket,
        }
    }
}

/// The dial loop's response to a close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AfterClose {
    /// Wait the next backoff delay and dial again.
    Backoff,
    /// Wait at the 8 s cap and dial again: the cause will not clear sooner.
    AtCap,
    /// This socket stops dialling — it is stale (4409) or speaks a protocol the
    /// gateway refuses (4426). The member's other sockets carry on.
    StopSocket,
    /// The app's sockets to every cluster stop until the binary is replaced
    /// (4410): a build any cluster has superseded serves on none.
    StopApp,
}

/// A close: its code and the reason naming the values involved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Close {
    pub code: u16,
    pub reason: String,
}

impl Close {
    pub fn new(code: CloseCode, reason: impl Into<String>) -> Self {
        Self {
            code: code.code(),
            reason: reason.into(),
        }
    }

    /// What the dial loop does after this close.
    pub fn after(&self) -> AfterClose {
        CloseCode::from_code(self.code).map_or(AfterClose::Backoff, CloseCode::after)
    }
}

impl fmt::Display for Close {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_round_trips_and_names_its_response() {
        let table = [
            (CloseCode::GoingAway, 1001, AfterClose::Backoff),
            (CloseCode::Malformed, 4400, AfterClose::Backoff),
            (CloseCode::Proof, 4401, AfterClose::Backoff),
            (CloseCode::Replaced, 4409, AfterClose::StopSocket),
            (CloseCode::Superseded, 4410, AfterClose::StopApp),
            (CloseCode::Model, 4412, AfterClose::AtCap),
            (CloseCode::Skew, 4417, AfterClose::AtCap),
            (CloseCode::Protocol, 4426, AfterClose::StopSocket),
        ];
        for (code, number, after) in table {
            assert_eq!(code.code(), number);
            assert_eq!(CloseCode::from_code(number), Some(code));
            assert_eq!(Close::new(code, "r").after(), after);
        }
        assert_eq!(
            Close {
                code: 1006,
                reason: String::new()
            }
            .after(),
            AfterClose::Backoff
        );
    }

    #[test]
    fn a_close_shows_its_code_and_reason() {
        let c = Close::new(
            CloseCode::Superseded,
            "build 4812 (a1b2c3d) is below the npcd floor 4815 (9f8e7d6)",
        );
        assert_eq!(
            c.to_string(),
            "4410: build 4812 (a1b2c3d) is below the npcd floor 4815 (9f8e7d6)"
        );
    }
}
