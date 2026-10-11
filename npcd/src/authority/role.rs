//! What this member is to each world it hosts (`docs/replicated_substrate.md`
//! §9.2–§9.4).

use std::collections::BTreeMap;
use std::sync::Mutex;

/// This member's part in one world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// It runs the world: the metronome, the generator, every act that
    /// changes it. It writes the world's state, stamped with `epoch`, and
    /// lends its characters. A daemon with no cluster is the authority of
    /// every world it hosts, at epoch zero.
    Authority { epoch: u64 },
    /// It follows the authority: the state records as they replicate, the
    /// pulses as they arrive. Its characters' acts that change the world go to
    /// the authority.
    Replica,
    /// It is cut off from the world's authority — its sockets are gone, or
    /// the lease it held lapsed — and runs the world locally for the characters
    /// it holds, persisting nothing of it (§9.4).
    Disconnected,
}

impl Role {
    pub fn name(&self) -> &'static str {
        match self {
            Role::Authority { .. } => "authority",
            Role::Replica => "replica",
            Role::Disconnected => "disconnected",
        }
    }

    pub fn is_authority(&self) -> bool {
        matches!(self, Role::Authority { .. })
    }
}

/// Every hosted world's role. A world not named is the default given at
/// construction: the authority for a daemon with no cluster, a replica for a
/// member, until its lease says otherwise.
#[derive(Debug)]
pub struct Roles {
    default: Mutex<Role>,
    roles: Mutex<BTreeMap<String, Role>>,
}

impl Roles {
    pub fn new(default: Role) -> Self {
        Self {
            default: Mutex::new(default),
            roles: Mutex::new(BTreeMap::new()),
        }
    }

    /// The role of every world not given one.
    pub fn set_default(&self, role: Role) {
        *self.default.lock().unwrap() = role;
    }

    fn default(&self) -> Role {
        *self.default.lock().unwrap()
    }

    pub fn get(&self, world: &str) -> Role {
        let given = self.roles.lock().unwrap().get(world).copied();
        given.unwrap_or_else(|| self.default())
    }

    /// Set `world`'s role; the role it had.
    pub fn set(&self, world: &str, role: Role) -> Role {
        let was = self.roles.lock().unwrap().insert(world.to_string(), role);
        was.unwrap_or_else(|| self.default())
    }

    /// Every world given a role, with it.
    pub fn all(&self) -> BTreeMap<String, Role> {
        self.roles.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_has_the_default_until_given_one() {
        let roles = Roles::new(Role::Replica);
        assert_eq!(roles.get("w1"), Role::Replica);
        assert_eq!(roles.set("w1", Role::Authority { epoch: 9 }), Role::Replica);
        assert_eq!(roles.get("w1"), Role::Authority { epoch: 9 });
        assert!(roles.get("w1").is_authority());
        assert_eq!(roles.get("w2").name(), "replica");
    }
}
