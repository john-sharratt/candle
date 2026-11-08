//! The FLX4's output, captured (§4.4): WASAPI loopback of its render
//! device, which is what rekordbox plays the master through. Each callback's
//! frames are mixed to mono and handed on, without waiting, to every reader
//! — the room's analysis and each screen's scenes — through its own queue.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamError};
use rtrb::Producer;

/// The open capture: dropping it stops it.
pub struct Capture {
    _stream: Stream,
    pub rate: u32,
    pub device: String,
}

/// The output devices' names, as Windows lists them.
pub fn outputs() -> Result<Vec<String>, String> {
    let host = cpal::default_host();
    let devices = host.output_devices().map_err(|e| e.to_string())?;
    Ok(devices.filter_map(|d| d.name().ok()).collect())
}

/// Every frame of `data`, `channels` wide, mixed to mono.
pub fn mono(data: &[f32], channels: usize) -> impl Iterator<Item = f32> + Clone + '_ {
    data.chunks_exact(channels.max(1))
        .map(move |f| f.iter().sum::<f32>() / channels.max(1) as f32)
}

fn hand_on(sinks: &mut [Producer<f32>], frames: impl Iterator<Item = f32> + Clone) {
    for s in sinks.iter_mut() {
        for x in frames.clone() {
            // A full queue is a reader behind: it loses the oldest-to-come,
            // never holds up the capture.
            if s.push(x).is_err() {
                break;
            }
        }
    }
}

/// Loopback of the output device whose name contains `part`, handing mono
/// samples to `sinks`.
pub fn open(part: &str, mut sinks: Vec<Producer<f32>>) -> Result<Capture, String> {
    let host = cpal::default_host();
    let want = part.to_ascii_lowercase();
    let device = host
        .output_devices()
        .map_err(|e| e.to_string())?
        .find(|d| d.name().is_ok_and(|n| n.to_ascii_lowercase().contains(&want)))
        .ok_or_else(|| format!("no output device named like `{part}` among {:?}", outputs().unwrap_or_default()))?;
    let name = device.name().unwrap_or_default();
    let config = device.default_output_config().map_err(|e| e.to_string())?;
    let channels = config.channels() as usize;
    let rate = config.sample_rate().0;
    let err = |e: StreamError| tracing::warn!("capture: {e}");
    // An input stream on an output device is WASAPI loopback.
    let stream = match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _| hand_on(&mut sinks, mono(data, channels)),
            err,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &config.into(),
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|x| *x as f32 / 32768.0).collect();
                let m: Vec<f32> = mono(&f, channels).collect();
                hand_on(&mut sinks, m.into_iter());
            },
            err,
            None,
        ),
        other => return Err(format!("{name} plays {other:?}, which this does not read")),
    }
    .map_err(|e| format!("{name}: {e}"))?;
    stream.play().map_err(|e| format!("{name}: {e}"))?;
    Ok(Capture {
        _stream: stream,
        rate,
        device: name,
    })
}

#[cfg(test)]
mod tests {
    use rtrb::RingBuffer;

    use super::*;

    #[test]
    fn frames_mix_to_mono_and_reach_every_reader() {
        let m: Vec<f32> = mono(&[1.0, 0.0, 0.5, 0.5, -1.0, 1.0], 2).collect();
        assert_eq!(m, vec![0.5, 0.5, 0.0]);
        let (p1, mut c1) = RingBuffer::new(2);
        let (p2, c2) = RingBuffer::new(8);
        let mut sinks = vec![p1, p2];
        hand_on(&mut sinks, m.iter().copied());
        assert_eq!((c1.pop(), c1.pop(), c1.pop().is_err()), (Ok(0.5), Ok(0.5), true), "full: the rest dropped");
        assert_eq!(c2.slots(), 3, "all three for the roomier reader");
    }
}
