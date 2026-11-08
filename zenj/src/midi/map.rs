//! A controller's MIDI map: which message is which control. One format for
//! both ends — the FLX4's own messages, observed (`maps/ddj-flx4.yaml`), and
//! the virtual controller's, which rekordbox learns (`maps/rekordbox-7.yaml`).
//!
//! ```yaml
//! port: ZenJ               # the MIDI port's name, or a part of it
//! controls:
//!   deck1.fader: {channel: 0, cc: 0x13, lsb: 0x33}   # 14-bit: MSB and LSB
//!   mixer.crossfader: {channel: 6, cc: 0x1F}         # 7-bit
//!   deck1.play: {channel: 0, note: 0x0B}
//! ```
//!
//! **A map is refused at load when it is ambiguous**: a control named twice,
//! or one message (a note, a CC or an LSB) used by two controls. The refusal
//! names the file and both claimants, so a map that loads is a map that
//! means one thing.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use super::wire::Message;
use crate::control::{Control, Knob};

/// One control's message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Binding {
    /// A button: note-on 127 pressed, note-off (or note-on 0) released.
    Note { channel: u8, note: u8 },
    /// A knob or fader: a CC, and the CC that carries its low seven bits
    /// when it is 14-bit.
    Cc {
        channel: u8,
        cc: u8,
        #[serde(default)]
        lsb: Option<u8>,
    },
}

/// What one message used, for the ambiguity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Slot {
    Note(u8, u8),
    Cc(u8, u8),
}

impl Binding {
    fn slots(self) -> Vec<Slot> {
        match self {
            Binding::Note { channel, note } => vec![Slot::Note(channel, note)],
            Binding::Cc { channel, cc, lsb } => {
                let mut s = vec![Slot::Cc(channel, cc)];
                s.extend(lsb.map(|l| Slot::Cc(channel, l)));
                s
            }
        }
    }

    fn valid(self) -> bool {
        match self {
            Binding::Note { channel, note } => channel < 16 && note < 128,
            Binding::Cc { channel, cc, lsb } => {
                channel < 16 && cc < 128 && lsb.is_none_or(|l| l < 128)
            }
        }
    }
}

/// What a decoded message said of its control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reading {
    /// A knob's position, `0..=1`.
    At(f32),
    Pressed,
    Released,
}

/// What the file says, before it is checked.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    port: String,
    controls: serde_yaml::Mapping,
}

/// A loaded, unambiguous map.
#[derive(Debug, Clone)]
pub struct Map {
    pub port: String,
    controls: BTreeMap<Control, Binding>,
    by_slot: HashMap<Slot, (Control, bool)>,
    /// The last MSB of each 14-bit CC, for pairing its LSB.
    msb: HashMap<(u8, u8), u8>,
}

impl Map {
    /// The map in `path`.
    pub fn load(path: &Path) -> Result<Map, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Map::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The map `text` says.
    pub fn parse(text: &str) -> Result<Map, String> {
        let file: File = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
        let mut controls = BTreeMap::new();
        let mut by_slot: HashMap<Slot, (Control, bool)> = HashMap::new();
        for (key, value) in file.controls {
            let name = key
                .as_str()
                .ok_or_else(|| format!("a control's name is text: {key:?}"))?;
            let control: Control = name.parse()?;
            let binding: Binding = serde_yaml::from_value(value).map_err(|e| {
                format!("{name}: a note {{channel, note}} or a CC {{channel, cc, lsb?}}: {e}")
            })?;
            if !binding.valid() {
                return Err(format!("{name}: a channel is 0–15 and a note or CC 0–127"));
            }
            match (control, binding) {
                (Control::Knob(_), Binding::Note { .. }) => {
                    return Err(format!("{name} is a knob: it takes a CC, not a note"))
                }
                (Control::Button(_), Binding::Cc { .. }) => {
                    return Err(format!("{name} is a button: it takes a note, not a CC"))
                }
                _ => {}
            }
            if controls.insert(control, binding).is_some() {
                return Err(format!("{name} is named twice"));
            }
            for (i, slot) in binding.slots().into_iter().enumerate() {
                if let Some((other, _)) = by_slot.insert(slot, (control, i == 1)) {
                    return Err(format!(
                        "{name} and {other} use the same message ({slot:?})"
                    ));
                }
            }
        }
        Ok(Map {
            port: file.port,
            controls,
            by_slot,
            msb: HashMap::new(),
        })
    }

    /// Every control the map binds, in name order.
    pub fn controls(&self) -> impl Iterator<Item = (&Control, &Binding)> {
        self.controls.iter()
    }

    pub fn binding(&self, control: Control) -> Option<Binding> {
        self.controls.get(&control).copied()
    }

    /// What `message` says of which control, if the map binds it. A 14-bit
    /// CC reads its MSB at once, with the low bits it last had cleared, and
    /// its LSB as the refinement of the MSB before it.
    pub fn decode(&mut self, message: &Message) -> Option<(Control, Reading)> {
        match *message {
            Message::NoteOn {
                channel,
                note,
                velocity,
            } => {
                let (c, _) = self.by_slot.get(&Slot::Note(channel, note))?;
                Some((
                    *c,
                    if velocity > 0 {
                        Reading::Pressed
                    } else {
                        Reading::Released
                    },
                ))
            }
            Message::NoteOff { channel, note, .. } => {
                let (c, _) = self.by_slot.get(&Slot::Note(channel, note))?;
                Some((*c, Reading::Released))
            }
            Message::Control {
                channel,
                control,
                value,
            } => {
                let (c, is_lsb) = *self.by_slot.get(&Slot::Cc(channel, control))?;
                let Some(Binding::Cc { cc, lsb, .. }) = self.controls.get(&c).copied() else {
                    return None;
                };
                let at = match (lsb, is_lsb) {
                    (None, _) => value as f32 / 127.0,
                    (Some(_), false) => {
                        self.msb.insert((channel, cc), value);
                        ((value as u16) << 7) as f32 / 16383.0
                    }
                    (Some(_), true) => {
                        let msb = *self.msb.get(&(channel, cc))?;
                        ((msb as u16) << 7 | value as u16) as f32 / 16383.0
                    }
                };
                Some((c, Reading::At(at)))
            }
            _ => None,
        }
    }

    /// The messages that set `knob` to `at` (`0..=1`): its CC, then its LSB
    /// when it is 14-bit. `None` when the map does not bind it.
    pub fn set(&self, knob: Knob, at: f32) -> Option<Vec<Message>> {
        let Binding::Cc { channel, cc, lsb } = self.binding(Control::Knob(knob))? else {
            return None;
        };
        let at = at.clamp(0.0, 1.0);
        Some(match lsb {
            None => vec![Message::Control {
                channel,
                control: cc,
                value: (at * 127.0).round() as u8,
            }],
            Some(l) => {
                let v = (at * 16383.0).round() as u16;
                vec![
                    Message::Control {
                        channel,
                        control: cc,
                        value: (v >> 7) as u8,
                    },
                    Message::Control {
                        channel,
                        control: l,
                        value: (v & 0x7F) as u8,
                    },
                ]
            }
        })
    }

    /// The messages of a press of `control`'s button: down, then up.
    pub fn press(&self, control: Control) -> Option<Vec<Message>> {
        let Binding::Note { channel, note } = self.binding(control)? else {
            return None;
        };
        Some(vec![
            Message::NoteOn {
                channel,
                note,
                velocity: 0x7F,
            },
            Message::NoteOff {
                channel,
                note,
                velocity: 0,
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{Band, Button};

    const MAP: &str = "port: ZenJ\ncontrols:\n  \
        deck1.fader: {channel: 0, cc: 0x13, lsb: 0x33}\n  \
        mixer.crossfader: {channel: 6, cc: 0x1F}\n  \
        deck1.play: {channel: 0, note: 0x0B}\n  \
        deck2.eq_low: {channel: 1, cc: 0x0F}\n";

    fn map() -> Map {
        Map::parse(MAP).unwrap()
    }

    #[test]
    fn a_map_reads_its_controls() {
        let m = map();
        assert_eq!(m.port, "ZenJ");
        assert_eq!(m.controls().count(), 4);
        assert_eq!(
            m.binding(Knob::Fader(1).into()),
            Some(Binding::Cc {
                channel: 0,
                cc: 0x13,
                lsb: Some(0x33)
            })
        );
    }

    /// **An ambiguous map is refused, naming both claimants**: a control
    /// twice, a message twice, an LSB on another control's CC, a knob on a
    /// note.
    #[test]
    fn an_ambiguous_map_is_refused() {
        let with = |extra: &str| Map::parse(&format!("{MAP}  {extra}\n")).unwrap_err();
        let twice = with("deck1.fader: {channel: 0, cc: 0x14}");
        assert!(twice.contains("deck1.fader"), "a control twice: {twice}");
        assert_eq!(
            with("deck2.play: {channel: 0, note: 0x0B}"),
            "deck2.play and deck1.play use the same message (Note(0, 11))"
        );
        assert_eq!(
            with("deck1.trim: {channel: 0, cc: 0x04, lsb: 0x13}"),
            "deck1.trim and deck1.fader use the same message (Cc(0, 19))"
        );
        assert_eq!(
            with("deck1.trim: {channel: 0, note: 4}"),
            "deck1.trim is a knob: it takes a CC, not a note"
        );
        assert_eq!(
            with("deck1.cue: {channel: 16, note: 4}"),
            "deck1.cue: a channel is 0–15 and a note or CC 0–127"
        );
        assert_eq!(
            with("deck9.cue: {channel: 0, note: 4}"),
            "`deck9.cue` names no control"
        );
    }

    /// **The shipped maps load**, and the virtual controller binds every
    /// control a plan can send to decks 1 and 2.
    #[test]
    fn the_shipped_maps_load() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("maps");
        let flx4 = Map::load(&dir.join("ddj-flx4.yaml")).unwrap();
        assert_eq!(flx4.port, "DDJ-FLX4");
        assert!(flx4.binding(Button::Jog(2).into()).is_some());
        let virt = Map::load(&dir.join("rekordbox-7.yaml")).unwrap();
        assert_eq!(virt.port, "ZenJ");
        for d in 1..=2 {
            for k in [
                Knob::Fader(d),
                Knob::Trim(d),
                Knob::Eq(d, Band::High),
                Knob::Eq(d, Band::Mid),
                Knob::Eq(d, Band::Low),
                Knob::Filter(d),
            ] {
                assert!(virt.set(k, 0.5).is_some(), "{k:?}");
            }
            for b in [
                Button::Play(d),
                Button::Cue(d),
                Button::Sync(d),
                Button::Load(d),
                Button::AutoLoop(d),
                Button::Fx(d),
                Button::HotCue(d, 8),
            ] {
                assert!(virt.press(b.into()).is_some(), "{b:?}");
            }
        }
        for b in [
            Button::BrowseUp,
            Button::BrowseDown,
            Button::Automix,
            Button::Sampler(16),
        ] {
            assert!(virt.press(b.into()).is_some(), "{b:?}");
        }
        assert!(virt.set(Knob::Crossfader, 0.5).is_some());
    }

    #[test]
    fn a_load_names_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.yaml");
        fs::write(
            &path,
            "port: x\ncontrols:\n  deck1.volume: {channel: 0, cc: 1}\n",
        )
        .unwrap();
        assert_eq!(
            Map::load(&path).unwrap_err(),
            format!("{}: `deck1.volume` names no control", path.display())
        );
    }

    /// **Decoding**: a button's press and release, a 7-bit CC, and a 14-bit
    /// pair — the MSB at once, the LSB refining it.
    #[test]
    fn messages_decode_to_their_controls() {
        let mut m = map();
        let play = Control::Button(Button::Play(1));
        assert_eq!(
            m.decode(&Message::NoteOn {
                channel: 0,
                note: 0x0B,
                velocity: 0x7F
            }),
            Some((play, Reading::Pressed))
        );
        assert_eq!(
            m.decode(&Message::NoteOff {
                channel: 0,
                note: 0x0B,
                velocity: 0
            }),
            Some((play, Reading::Released))
        );
        let cc = |channel, control, value| Message::Control {
            channel,
            control,
            value,
        };
        assert_eq!(
            m.decode(&cc(6, 0x1F, 127)),
            Some((Control::Knob(Knob::Crossfader), Reading::At(1.0)))
        );
        let fader = Control::Knob(Knob::Fader(1));
        assert_eq!(m.decode(&cc(0, 0x33, 5)), None, "an LSB with no MSB yet");
        assert_eq!(
            m.decode(&cc(0, 0x13, 0x40)),
            Some((fader, Reading::At(8192.0 / 16383.0)))
        );
        assert_eq!(
            m.decode(&cc(0, 0x33, 0x7F)),
            Some((fader, Reading::At(8319.0 / 16383.0)))
        );
        assert_eq!(m.decode(&cc(1, 0x13, 0x40)), None, "another channel");
        assert_eq!(
            m.decode(&cc(1, 0x0F, 0)),
            Some((Control::Knob(Knob::Eq(2, Band::Low)), Reading::At(0.0)))
        );
    }

    /// **Encoding**: a 14-bit knob sends MSB then LSB, a 7-bit one its CC,
    /// a press down then up — exact bytes.
    #[test]
    fn controls_encode_to_exact_bytes() {
        let m = map();
        let bytes = |ms: Vec<Message>| ms.iter().flat_map(Message::encode).collect::<Vec<u8>>();
        assert_eq!(
            bytes(m.set(Knob::Fader(1), 0.5).unwrap()),
            vec![0xB0, 0x13, 0x40, 0xB0, 0x33, 0x00]
        );
        assert_eq!(
            bytes(m.set(Knob::Fader(1), 1.0).unwrap()),
            vec![0xB0, 0x13, 0x7F, 0xB0, 0x33, 0x7F]
        );
        assert_eq!(
            bytes(m.set(Knob::Crossfader, 2.0).unwrap()),
            vec![0xB6, 0x1F, 0x7F],
            "clamped"
        );
        assert_eq!(
            bytes(m.press(Button::Play(1).into()).unwrap()),
            vec![0x90, 0x0B, 0x7F, 0x80, 0x0B, 0x00]
        );
        assert_eq!(m.set(Knob::Trim(1), 0.5), None, "not bound");
        assert_eq!(m.press(Knob::Fader(1).into()), None, "not a button");
    }
}
