//! One CSV line a second: what the balloon holds, how it and a watched process
//! page, and the machine's free RAM.

use std::time::Instant;

use candle::vram::process_paging::{own_paging, paging_of_named, ProcessPaging};

const GIB: f64 = (1u64 << 30) as f64;

/// The columns, in order. Rates are a second, over the interval since the
/// line before; `sweeps_s` is the active RAM's full sweeps — one falling is the
/// balloon itself being paged.
pub const HEADER: &str = "t_s,phase,active_gib,idle_gib,available_gib,sweeps_s,\
own_ws_gib,own_private_gib,own_faults_s,\
watch_pid,watch_ws_gib,watch_private_gib,watch_faults_s";

/// The counts and the instant a line's rates are taken against.
#[derive(Default)]
pub struct Sampler {
    at: Option<Instant>,
    own: Option<u64>,
    sweeps: Option<u64>,
    watched: Option<(u32, u64)>,
}

impl Sampler {
    /// One line at `t_s` into the run.
    pub fn line(
        &mut self,
        t_s: u64,
        phase: usize,
        held: (u64, u64),
        sweeps: u64,
        watch: Option<&str>,
    ) -> String {
        let now = Instant::now();
        let dt = self.at.map_or(0.0, |t| now.duration_since(t).as_secs_f64());
        self.at = Some(now);
        let available = candle::vram::available_physical_ram().unwrap_or(0);
        let own = own_paging();
        let own_rate = rate(&mut self.own, own.as_ref().map(|p| p.hard_faults), dt);
        let sweep_rate = rate(&mut self.sweeps, Some(sweeps), dt);
        let watched: Option<ProcessPaging> = watch.and_then(|name| {
            paging_of_named(name)
                .into_iter()
                .max_by_key(|p| p.private_bytes)
        });
        let watch_rate = match (&watched, self.watched) {
            (Some(w), Some((pid, prev))) if pid == w.pid && dt > 0.0 => {
                Some(w.hard_faults.saturating_sub(prev) as f64 / dt)
            }
            _ => None,
        };
        self.watched = watched.as_ref().map(|w| (w.pid, w.hard_faults));
        let gib = |b: u64| format!("{:.2}", b as f64 / GIB);
        let opt = |v: Option<f64>| v.map_or(String::new(), |r| format!("{r:.0}"));
        format!(
            "{t_s},{phase},{},{},{},{},{},{},{},{},{},{},{}",
            gib(held.0),
            gib(held.1),
            gib(available),
            sweep_rate.map_or(String::new(), |r| format!("{r:.1}")),
            own.as_ref().map_or(String::new(), |p| gib(p.working_set)),
            own.as_ref().map_or(String::new(), |p| gib(p.private_bytes)),
            opt(own_rate),
            watched
                .as_ref()
                .map_or(String::new(), |w| w.pid.to_string()),
            watched
                .as_ref()
                .map_or(String::new(), |w| gib(w.working_set)),
            watched
                .as_ref()
                .map_or(String::new(), |w| gib(w.private_bytes)),
            opt(watch_rate),
        )
    }
}

/// The change in a cumulative count since the last call, a second over `dt`
/// seconds.
fn rate(prev: &mut Option<u64>, now: Option<u64>, dt: f64) -> Option<f64> {
    let r = match (*prev, now) {
        (Some(p), Some(n)) if dt > 0.0 => Some(n.saturating_sub(p) as f64 / dt),
        _ => None,
    };
    *prev = now;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rate_needs_two_readings_is_a_second_and_never_goes_negative() {
        let mut prev = None;
        assert_eq!(rate(&mut prev, Some(10), 1.0), None);
        assert_eq!(rate(&mut prev, Some(40), 2.0), Some(15.0));
        assert_eq!(rate(&mut prev, Some(20), 1.0), Some(0.0));
    }

    #[test]
    fn a_line_has_every_column() {
        let mut s = Sampler::default();
        let line = s.line(3, 1, (0, 0), 0, Some("no-such-process"));
        assert_eq!(line.split(',').count(), HEADER.split(',').count(), "{line}");
        assert!(line.starts_with("3,1,0.00,0.00,"), "{line}");
    }
}
