//! `zenj_moment`: book something for a time — a track ("Yen's song at
//! 22:30"), a dedication on the screens, or a sample.

use chrono::{DateTime, Local, NaiveTime, TimeDelta, TimeZone};
use serde_json::{json, Value};

use super::{text, Answer, Dj, Tool};
use crate::brain::run::{Ask, BookWhat};

pub struct MomentTool;

impl Tool for MomentTool {
    const NAME: &'static str = "zenj_moment";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Book a moment at the party for a time (HH:MM, the next one to come): a `track` to play then, a `dedication` shown on every screen, or a `sample` slot fired on the bar. Give exactly one of the three.",
            "inputSchema": {"type": "object", "properties": {
                "at": {"type": "string", "description": "HH:MM, the next time it comes round."},
                "track": {"type": "string", "description": "Artist and title, or ISRC."},
                "dedication": {"type": "string"},
                "sample": {"type": "integer", "minimum": 1, "maximum": 16}
            }, "required": ["at"]},
        })
    }
}

/// The next `hh:mm` at or after `now`.
pub fn next_at(hh_mm: &str, now: DateTime<Local>) -> Result<DateTime<Local>, String> {
    let t =
        NaiveTime::parse_from_str(hh_mm, "%H:%M").map_err(|_| format!("`{hh_mm}` is not HH:MM"))?;
    let today = now.date_naive().and_time(t);
    let at = Local
        .from_local_datetime(&today)
        .earliest()
        .ok_or("no such time today")?;
    Ok(if at < now {
        at + TimeDelta::days(1)
    } else {
        at
    })
}

pub async fn call(dj: &Dj, args: &Value) -> Answer {
    let at = match text(args, "at").map(|a| next_at(a, Local::now())) {
        Ok(Ok(at)) => at,
        Ok(Err(e)) => return Answer::error(e),
        Err(a) => return a,
    };
    let s = |k: &str| {
        args.get(k)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|x| !x.is_empty())
    };
    let what = match (
        s("track"),
        s("dedication"),
        args.get("sample").and_then(Value::as_u64),
    ) {
        (Some(t), None, None) => BookWhat::Track(t.to_string()),
        (None, Some(d), None) => BookWhat::Dedication(d.to_string()),
        (None, None, Some(n)) if (1..=16).contains(&n) => BookWhat::Sample(n as u8),
        _ => return Answer::error("give exactly one of `track`, `dedication` or `sample` (1–16)"),
    };
    dj.ask(|reply| Ask::Book { at, what, reply }).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_time_is_the_next_to_come() {
        let now = Local.with_ymd_and_hms(2026, 12, 31, 22, 0, 0).unwrap();
        assert_eq!(
            next_at("22:30", now).unwrap(),
            Local.with_ymd_and_hms(2026, 12, 31, 22, 30, 0).unwrap()
        );
        assert_eq!(
            next_at("00:15", now).unwrap(),
            Local.with_ymd_and_hms(2027, 1, 1, 0, 15, 0).unwrap()
        );
        assert_eq!(next_at("25:00", now).unwrap_err(), "`25:00` is not HH:MM");
    }
}
