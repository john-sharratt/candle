//! The bus of a daemon in no cluster: there is no gateway to carry a message,
//! so a publish reaches nobody and a gate cannot be asked. A gated
//! repository's push is then refused, as a gate that is not connected refuses
//! it (`docs/mcp_file_tools.md` §5.4), never let through unchecked.

use std::time::Duration;

use serde_json::Value;

use super::{Bus, Unasked};

/// Why a gate cannot be asked here.
pub const NO_BUS: &str = "this daemon is in no cluster, so there is no bus to ask it on";

pub struct Offline;

impl Bus for Offline {
    fn publish(&self, _topic: &str, _payload: Value) {}

    fn ask(
        &self,
        _topic: &str,
        _payload: Value,
        _reply: &str,
        _wait: Duration,
    ) -> Result<Value, Unasked> {
        Err(Unasked::Failed(NO_BUS.into()))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A daemon in no cluster asks no gate**, so a gated push cannot pass.
    #[test]
    fn no_gate_is_asked_without_a_bus() {
        assert_eq!(
            Offline.ask(
                "gate/yenbot",
                json!({}),
                "gate/yenbot/result/1",
                Duration::ZERO
            ),
            Err(Unasked::Failed(NO_BUS.into()))
        );
    }
}
