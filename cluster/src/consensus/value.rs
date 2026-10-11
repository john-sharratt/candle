//! What a Paxos round decides for a key (§5.9): a **decision**, one value for
//! good, or a **lease**, a holder for a term.
//!
//! Encoded as the `value` of `PROPOSE` and of an accepted value: `u8` tag,
//! then for a decision its bytes, for a lease `u64 holder · u64 ttl_ms`.

use std::time::Duration;

use crate::wire::{Reader, WireError, Writer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    Decision(Vec<u8>),
    Lease { holder: u64, ttl: Duration },
}

impl Value {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer::new();
        match self {
            Value::Decision(bytes) => {
                w.u8(0).fixed(bytes);
            }
            Value::Lease { holder, ttl } => {
                w.u8(1).u64(*holder).u64(ttl.as_millis() as u64);
            }
        }
        w.into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Result<Value, WireError> {
        let mut r = Reader::new(bytes, "consensus value");
        match r.u8()? {
            0 => Ok(Value::Decision(r.rest().to_vec())),
            1 => {
                let holder = r.u64()?;
                let ttl = Duration::from_millis(r.u64()?);
                r.finish()?;
                Ok(Value::Lease { holder, ttl })
            }
            tag => Err(WireError::Invalid {
                what: "consensus value",
                value: tag as u64,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_encode_to_exact_bytes() {
        assert_eq!(Value::Decision(b"gen".to_vec()).encode(), b"\0gen");
        let lease = Value::Lease {
            holder: 0x0102,
            ttl: Duration::from_secs(30),
        };
        assert_eq!(
            lease.encode(),
            [1, 2, 1, 0, 0, 0, 0, 0, 0, 0x30, 0x75, 0, 0, 0, 0, 0, 0]
        );
        for v in [Value::Decision(vec![]), lease] {
            assert_eq!(Value::decode(&v.encode()).unwrap(), v);
        }
        assert!(Value::decode(&[2]).is_err());
        assert!(Value::decode(&[1, 0]).is_err());
    }
}
