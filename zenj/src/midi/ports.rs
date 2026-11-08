//! The two MIDI ports: the FLX4's, observed — Windows MIDI Services lets any
//! number of applications open it beside rekordbox (§6.3) — and the virtual
//! controller's, which rekordbox has learned and ZenJ drives.
//!
//! Each opens by a part of its name, so a port renumbered after a replug is
//! still found.

use std::time::Instant;

use midir::{MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use rtrb::Producer;

use super::map::{Map, Reading};
use super::wire::{Decoder, Message};
use crate::control::Control;

/// One control a person touched, as the FLX4 said it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Touch {
    pub control: Control,
    pub reading: Reading,
    pub at: Instant,
}

/// The first port whose name contains `part`, ignoring case, among `names`.
pub fn find(names: &[String], part: &str) -> Option<usize> {
    let part = part.to_ascii_lowercase();
    names
        .iter()
        .position(|n| n.to_ascii_lowercase().contains(&part))
}

/// Every MIDI input port's name, in the order midir lists them.
pub fn input_names() -> Result<Vec<String>, String> {
    let input = MidiInput::new("zenj-list").map_err(|e| e.to_string())?;
    Ok(input
        .ports()
        .iter()
        .map(|p| input.port_name(p).unwrap_or_default())
        .collect())
}

/// Every MIDI output port's name.
pub fn output_names() -> Result<Vec<String>, String> {
    let output = MidiOutput::new("zenj-list").map_err(|e| e.to_string())?;
    Ok(output
        .ports()
        .iter()
        .map(|p| output.port_name(p).unwrap_or_default())
        .collect())
}

/// The FLX4, observed: every message decoded through `map` and each control
/// it names pushed to `touches`. A full queue drops the touch: the hands'
/// reader is behind, and the next touch says where the control is.
pub fn observe(
    mut map: Map,
    mut touches: Producer<Touch>,
) -> Result<MidiInputConnection<()>, String> {
    let mut input = MidiInput::new("zenj-flx4").map_err(|e| e.to_string())?;
    input.ignore(midir::Ignore::None);
    let ports = input.ports();
    let names: Vec<String> = ports
        .iter()
        .map(|p| input.port_name(p).unwrap_or_default())
        .collect();
    let i = find(&names, &map.port)
        .ok_or_else(|| format!("no MIDI input named like `{}` among {names:?}", map.port))?;
    let mut decoder = Decoder::new();
    let mut out = Vec::new();
    input
        .connect(
            &ports[i],
            "zenj-observe",
            move |_stamp, bytes, _| {
                out.clear();
                for &b in bytes {
                    decoder.push(b, &mut out);
                }
                for m in &out {
                    if let Some((control, reading)) = map.decode(m) {
                        let _ = touches.push(Touch {
                            control,
                            reading,
                            at: Instant::now(),
                        });
                    }
                }
            },
            (),
        )
        .map_err(|e| e.to_string())
}

/// The virtual controller's port, open for sending.
pub struct Out {
    conn: MidiOutputConnection,
}

impl Out {
    /// The output port whose name contains `part`.
    pub fn open(part: &str) -> Result<Out, String> {
        let output = MidiOutput::new("zenj-hands").map_err(|e| e.to_string())?;
        let ports = output.ports();
        let names: Vec<String> = ports
            .iter()
            .map(|p| output.port_name(p).unwrap_or_default())
            .collect();
        let i = find(&names, part)
            .ok_or_else(|| format!("no MIDI output named like `{part}` among {names:?}"))?;
        let conn = output
            .connect(&ports[i], "zenj-hands")
            .map_err(|e| e.to_string())?;
        Ok(Out { conn })
    }

    pub fn send(&mut self, messages: &[Message]) -> Result<(), String> {
        for m in messages {
            self.conn.send(&m.encode()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_port_is_found_by_a_part_of_its_name() {
        let names = vec![
            "Microsoft GS Wavetable Synth".to_string(),
            "DDJ-FLX4".to_string(),
            "ZenJ Loopback A".to_string(),
        ];
        assert_eq!(find(&names, "ddj-flx4"), Some(1));
        assert_eq!(find(&names, "zenj"), Some(2));
        assert_eq!(find(&names, "launchpad"), None);
    }
}
