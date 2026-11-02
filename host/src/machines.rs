//! The estate's machine table, `host/machines.yaml`
//! (`docs/static_placement.md` §2). Compiled in, as the role table is: the
//! gateway reads the same file, so the machine a daemon sizes itself as and the
//! machine the gateway places work on are one entry.

use cluster::machines::MachineTable;

/// The machine table every daemon carries.
pub const MACHINES: &str = include_str!("../machines.yaml");

/// [`MACHINES`], parsed.
pub fn machines() -> MachineTable {
    MachineTable::from_yaml(MACHINES).expect("host/machines.yaml is valid")
}

/// Tell the engine how much host RAM the expert cache may page-lock on
/// `machine`, when the table says (§7); otherwise the engine derives it.
/// Called once, before any tenant loads the model.
pub fn state_expert_pin(machine: &str) {
    if let Some(bytes) = machines().pinned_experts_bytes(machine) {
        tracing::info!(
            machine,
            pinned_experts_gib = bytes >> 30,
            "the machine table states the expert pin"
        );
        candle::vram::state_expert_pin(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compiled-in table parses, and states WhiteWorld's pin.
    #[test]
    fn the_table_states_whiteworlds_pin() {
        assert_eq!(
            machines().pinned_experts_bytes("WhiteWorld"),
            Some(28 << 30)
        );
        assert_eq!(machines().pinned_experts_bytes("BlackWorld"), None);
    }
}
