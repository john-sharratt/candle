//! MIDI 1.0 on the wire, decoded from bytes and encoded back: channel voice
//! messages under running status, SysEx with its terminator, and real-time
//! bytes interleaved anywhere — even inside another message — without
//! disturbing it.
//!
//! The decoder is a pure state machine over bytes, so what it makes of a
//! stream is asserted against the raw bytes themselves.

/// One message. Channels are 0–15.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    NoteOff { channel: u8, note: u8, velocity: u8 },
    /// A note-on with velocity 0 is decoded as the note-off it means.
    NoteOn { channel: u8, note: u8, velocity: u8 },
    PolyPressure { channel: u8, note: u8, pressure: u8 },
    Control { channel: u8, control: u8, value: u8 },
    Program { channel: u8, program: u8 },
    ChannelPressure { channel: u8, pressure: u8 },
    /// 14 bits, centre 8192.
    PitchBend { channel: u8, value: u16 },
    /// The bytes between `F0` and `F7`, neither included.
    SysEx(Vec<u8>),
    /// A system common message (`F1`–`F6`), with its data.
    Common(u8, Vec<u8>),
    /// A real-time byte, `F8`–`FF`.
    RealTime(u8),
}

/// How many data bytes a channel status takes.
fn data_len(status: u8) -> usize {
    match status & 0xF0 {
        0xC0 | 0xD0 => 1,
        _ => 2,
    }
}

/// How many data bytes a system common status takes.
fn common_len(status: u8) -> usize {
    match status {
        0xF1 | 0xF3 => 1,
        0xF2 => 2,
        _ => 0,
    }
}

fn channel_message(status: u8, d: &[u8]) -> Message {
    let channel = status & 0x0F;
    match status & 0xF0 {
        0x80 => Message::NoteOff {
            channel,
            note: d[0],
            velocity: d[1],
        },
        0x90 if d[1] == 0 => Message::NoteOff {
            channel,
            note: d[0],
            velocity: 0,
        },
        0x90 => Message::NoteOn {
            channel,
            note: d[0],
            velocity: d[1],
        },
        0xA0 => Message::PolyPressure {
            channel,
            note: d[0],
            pressure: d[1],
        },
        0xB0 => Message::Control {
            channel,
            control: d[0],
            value: d[1],
        },
        0xC0 => Message::Program {
            channel,
            program: d[0],
        },
        0xD0 => Message::ChannelPressure {
            channel,
            pressure: d[0],
        },
        _ => Message::PitchBend {
            channel,
            value: d[0] as u16 | (d[1] as u16) << 7,
        },
    }
}

/// The decoder's state between bytes.
#[derive(Debug, Default)]
pub struct Decoder {
    /// The running status: the last channel status seen.
    running: Option<u8>,
    /// A system common message being read.
    common: Option<u8>,
    data: Vec<u8>,
    /// Bytes of a SysEx being read.
    sysex: Option<Vec<u8>>,
}

impl Decoder {
    pub fn new() -> Decoder {
        Decoder::default()
    }

    /// Feed one byte; the message it completes, if any. A stray data byte
    /// with no status to run under is dropped.
    pub fn push(&mut self, byte: u8, out: &mut Vec<Message>) {
        if byte >= 0xF8 {
            out.push(Message::RealTime(byte));
            return;
        }
        if byte & 0x80 != 0 {
            // A status byte ends a SysEx, terminated or not.
            if let Some(body) = self.sysex.take() {
                out.push(Message::SysEx(body));
                if byte == 0xF7 {
                    return;
                }
            }
            self.data.clear();
            match byte {
                0xF0 => {
                    self.running = None;
                    self.common = None;
                    self.sysex = Some(Vec::new());
                }
                0xF7 => {
                    // An end with no start.
                    self.running = None;
                    self.common = None;
                }
                0xF1..=0xF6 => {
                    self.running = None;
                    if common_len(byte) == 0 {
                        self.common = None;
                        out.push(Message::Common(byte, Vec::new()));
                    } else {
                        self.common = Some(byte);
                    }
                }
                _ => {
                    self.common = None;
                    self.running = Some(byte);
                }
            }
            return;
        }
        if let Some(body) = &mut self.sysex {
            body.push(byte);
            return;
        }
        if let Some(status) = self.common {
            self.data.push(byte);
            if self.data.len() == common_len(status) {
                out.push(Message::Common(status, std::mem::take(&mut self.data)));
                self.common = None;
            }
            return;
        }
        let Some(status) = self.running else {
            return;
        };
        self.data.push(byte);
        if self.data.len() == data_len(status) {
            out.push(channel_message(status, &self.data));
            self.data.clear();
        }
    }

    /// Every message `bytes` completes.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<Message> {
        let mut out = Vec::new();
        for &b in bytes {
            self.push(b, &mut out);
        }
        out
    }
}

impl Message {
    /// The message as bytes, with its own status byte (no running status).
    pub fn encode(&self) -> Vec<u8> {
        match *self {
            Message::NoteOff {
                channel,
                note,
                velocity,
            } => vec![0x80 | channel, note & 0x7F, velocity & 0x7F],
            Message::NoteOn {
                channel,
                note,
                velocity,
            } => vec![0x90 | channel, note & 0x7F, velocity & 0x7F],
            Message::PolyPressure {
                channel,
                note,
                pressure,
            } => vec![0xA0 | channel, note & 0x7F, pressure & 0x7F],
            Message::Control {
                channel,
                control,
                value,
            } => vec![0xB0 | channel, control & 0x7F, value & 0x7F],
            Message::Program { channel, program } => vec![0xC0 | channel, program & 0x7F],
            Message::ChannelPressure { channel, pressure } => {
                vec![0xD0 | channel, pressure & 0x7F]
            }
            Message::PitchBend { channel, value } => vec![
                0xE0 | channel,
                (value & 0x7F) as u8,
                ((value >> 7) & 0x7F) as u8,
            ],
            Message::SysEx(ref body) => {
                let mut v = Vec::with_capacity(body.len() + 2);
                v.push(0xF0);
                v.extend(body.iter().map(|b| b & 0x7F));
                v.push(0xF7);
                v
            }
            Message::Common(status, ref data) => {
                let mut v = vec![status];
                v.extend(data);
                v
            }
            Message::RealTime(b) => vec![b],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(channel: u8, control: u8, value: u8) -> Message {
        Message::Control {
            channel,
            control,
            value,
        }
    }

    #[test]
    fn channel_messages_decode_from_their_bytes() {
        let mut d = Decoder::new();
        assert_eq!(
            d.feed(&[0x90, 0x0B, 0x7F, 0x80, 0x0B, 0x00, 0xB6, 0x1F, 0x40]),
            vec![
                Message::NoteOn {
                    channel: 0,
                    note: 0x0B,
                    velocity: 0x7F
                },
                Message::NoteOff {
                    channel: 0,
                    note: 0x0B,
                    velocity: 0
                },
                cc(6, 0x1F, 0x40),
            ]
        );
        assert_eq!(
            d.feed(&[0xE1, 0x00, 0x40, 0xC2, 0x05, 0xD3, 0x10]),
            vec![
                Message::PitchBend {
                    channel: 1,
                    value: 8192
                },
                Message::Program {
                    channel: 2,
                    program: 5
                },
                Message::ChannelPressure {
                    channel: 3,
                    pressure: 0x10
                },
            ]
        );
    }

    /// **A note-on at velocity 0 is the note-off it means.**
    #[test]
    fn a_silent_note_on_is_a_note_off() {
        assert_eq!(
            Decoder::new().feed(&[0x91, 0x0C, 0x00]),
            vec![Message::NoteOff {
                channel: 1,
                note: 0x0C,
                velocity: 0
            }]
        );
    }

    /// **Running status**: data bytes with no status run under the last
    /// channel status, a pair at a time.
    #[test]
    fn data_runs_under_the_last_status() {
        assert_eq!(
            Decoder::new().feed(&[0xB0, 0x13, 0x10, 0x33, 0x05, 0x13, 0x11]),
            vec![cc(0, 0x13, 0x10), cc(0, 0x33, 0x05), cc(0, 0x13, 0x11)]
        );
    }

    /// **A real-time byte anywhere** — between messages, or between a
    /// message's data bytes — comes out on its own and leaves the message
    /// it interrupted whole.
    #[test]
    fn real_time_bytes_interleave_without_breaking_a_message() {
        assert_eq!(
            Decoder::new().feed(&[0xF8, 0xB0, 0x13, 0xF8, 0x7F, 0xFE, 0x13, 0x00]),
            vec![
                Message::RealTime(0xF8),
                Message::RealTime(0xF8),
                cc(0, 0x13, 0x7F),
                Message::RealTime(0xFE),
                cc(0, 0x13, 0x00),
            ]
        );
    }

    /// **SysEx ends at `F7`, or at any status byte that cuts it short**, and
    /// clears the running status either way.
    #[test]
    fn sysex_ends_at_its_terminator_or_the_next_status() {
        let mut d = Decoder::new();
        assert_eq!(
            d.feed(&[0xB0, 0x01, 0x02, 0xF0, 0x00, 0x40, 0x05, 0xF7, 0x03, 0x04]),
            vec![cc(0, 1, 2), Message::SysEx(vec![0x00, 0x40, 0x05])],
            "data after the SysEx has no status to run under"
        );
        assert_eq!(
            d.feed(&[0xF0, 0x01, 0xF8, 0x02, 0x90, 0x10, 0x20]),
            vec![
                Message::RealTime(0xF8),
                Message::SysEx(vec![0x01, 0x02]),
                Message::NoteOn {
                    channel: 0,
                    note: 0x10,
                    velocity: 0x20
                },
            ]
        );
    }

    #[test]
    fn system_common_messages_take_their_own_data() {
        assert_eq!(
            Decoder::new().feed(&[0xF2, 0x10, 0x20, 0xF3, 0x05, 0xF6, 0x40]),
            vec![
                Message::Common(0xF2, vec![0x10, 0x20]),
                Message::Common(0xF3, vec![0x05]),
                Message::Common(0xF6, vec![]),
            ],
            "the trailing data byte has no status"
        );
    }

    /// **Every message encodes to its own bytes**, which decode back to it.
    #[test]
    fn messages_encode_to_bytes_that_decode_back() {
        let all = vec![
            Message::NoteOn {
                channel: 7,
                note: 0x00,
                velocity: 0x7F,
            },
            Message::NoteOff {
                channel: 7,
                note: 0x00,
                velocity: 0,
            },
            cc(15, 0x7F, 0x7F),
            Message::PitchBend {
                channel: 0,
                value: 16383,
            },
            Message::SysEx(vec![0x7E, 0x7F, 0x06, 0x01]),
            Message::RealTime(0xFA),
        ];
        let bytes: Vec<u8> = all.iter().flat_map(Message::encode).collect();
        assert_eq!(
            bytes,
            vec![
                0x97, 0x00, 0x7F, 0x87, 0x00, 0x00, 0xBF, 0x7F, 0x7F, 0xE0, 0x7F, 0x7F, 0xF0, 0x7E,
                0x7F, 0x06, 0x01, 0xF7, 0xFA
            ]
        );
        assert_eq!(Decoder::new().feed(&bytes), all);
    }
}
