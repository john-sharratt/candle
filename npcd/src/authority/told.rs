//! What a world's authority tells characters that run elsewhere
//! (`docs/replicated_substrate.md` §9.1, §9.3): a message off a handset, a
//! mission withdrawn, a standing task the table sets. Each would go into the
//! character's inbox were it here; for one lent to another member it goes as a
//! `told` pulse, which that member delivers to the characters it runs. Like any
//! pulse it is never persisted: what it says is a moment of the world, and the
//! world's records hold the state behind it.

use serde::{Deserialize, Serialize};

use crate::engine::event::{EventKind, Salience};

/// A pulse's kind: deliveries to characters.
pub const TOLD: &str = "told";

/// One delivery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Told {
    pub npc: u64,
    /// The world time it is stamped with, the character's own.
    pub at_ms: u64,
    pub salience: f32,
    pub kind: EventKind,
}

impl Told {
    pub fn new(npc: u64, at_ms: u64, salience: Salience, kind: EventKind) -> Self {
        Self {
            npc,
            at_ms,
            salience: salience.get(),
            kind,
        }
    }

    /// The salience it arrives with, through the one clamp every wire value
    /// goes through.
    pub fn salience(&self) -> Salience {
        Salience::new(self.salience)
    }
}

pub fn encode(told: &[Told]) -> Vec<u8> {
    serde_json::to_vec(told).expect("a delivery always encodes")
}

pub fn decode(body: &[u8]) -> Result<Vec<Told>, serde_json::Error> {
    serde_json::from_slice(body)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_delivery_travels_with_its_kind_and_salience() {
        let told = vec![Told::new(
            7,
            1_000,
            Salience::URGENT,
            EventKind::Message {
                thread: "t1".into(),
                from: "Maker-02".into(),
                text: "come up".into(),
            },
        )];
        let wire: serde_json::Value = serde_json::from_slice(&encode(&told)).unwrap();
        assert_eq!(wire[0]["npc"], json!(7));
        assert_eq!(wire[0]["kind"]["kind"], json!("message"));
        let back = decode(&encode(&told)).unwrap();
        assert_eq!(back, told);
        assert_eq!(back[0].salience(), Salience::URGENT);
    }
}
