//! A bot's trace as text (`docs/npcd_discord.md` §5.5): what
//! `bot_discord_read` answers, a line per fact, every time in UTC.

use journal_core::period::civil_from_days;

use super::report::Report;
use crate::discord::state::State;

/// `at_ms` as `2026-11-01 12:00:05 UTC`.
pub fn utc(at_ms: u64) -> String {
    let secs = at_ms / 1_000;
    let (y, mo, d) = civil_from_days((secs / 86_400) as i64);
    let s = secs % 86_400;
    format!(
        "{y:04}-{mo:02}-{d:02} {:02}:{:02}:{:02} UTC",
        s / 3_600,
        s % 3_600 / 60,
        s % 60
    )
}

fn or_never(at_ms: Option<u64>) -> String {
    at_ms.map_or_else(|| "never".into(), utc)
}

/// Bot `bot`'s trace `r`, as text.
pub fn render(bot: &str, r: &Report) -> String {
    let mut out = vec![format!("Bot {bot} on Discord.")];
    out.push(match &r.state {
        None => "State: no record (offline).".into(),
        Some(s) => {
            let state = match s.state {
                State::Online => "online",
                State::NeedsContentIntent => "needs its Message Content intent",
                State::Refused => "refused",
            };
            format!(
                "State: {state} in {} since {} (token connected {}).",
                s.world,
                utc(s.since_ms),
                utc(s.connected_ms)
            )
        }
    });
    let member = r
        .shard
        .member
        .map_or_else(|| "member unknown".into(), |m| format!("member {m}"));
    let sh = &r.shard.link;
    out.push(match sh.connected {
        true => format!(
            "Shard: connected, session {}, since {}, last event {}, last heartbeat ack {}, {member}.",
            sh.session.as_deref().unwrap_or("none"),
            or_never(sh.since_ms),
            or_never(sh.last_event_ms),
            or_never(sh.last_heartbeat_ack_ms),
        ),
        false => format!("Shard: not connected here, {member}."),
    });

    out.push(section(
        "DMs",
        r.dms.iter().map(|d| {
            format!(
                "account {}: channel {}, user {}, thread {}, last seq {}, mirrored through {}, \
             unposted {}, last seen {}",
                d.account,
                d.channel_id,
                d.user_id,
                d.thread_id.as_deref().unwrap_or("none"),
                d.thread_last_seq
                    .map_or_else(|| "none".into(), |s| s.to_string()),
                d.mirrored_through,
                d.unposted,
                d.last_seen_discord
            )
        }),
    ));
    out.push(section(
        "Channels",
        r.channels.iter().map(|c| {
            format!(
                "{} ({}): channel {} in server {}, thread {}, last seq {}, mirrored through {}, \
             unposted {}, last seen {}",
                c.name.as_deref().unwrap_or("no thread"),
                if c.open { "open" } else { "closed" },
                c.channel_id,
                c.guild_id,
                c.thread,
                c.thread_last_seq
                    .map_or_else(|| "none".into(), |s| s.to_string()),
                c.mirrored_through,
                c.unposted,
                c.last_seen_discord
            )
        }),
    ));
    out.push(section(
        "Servers",
        r.servers
            .iter()
            .map(|s| format!("{} ({})", s.name, s.guild_id)),
    ));
    out.push(section(
        "Events, newest first",
        r.events
            .iter()
            .map(|n| format!("{} {}", utc(n.at_ms), n.line())),
    ));
    out.join("\n")
}

/// A titled list, a line per item, or `none`.
fn section(title: &str, items: impl Iterator<Item = String>) -> String {
    let lines: Vec<String> = items.map(|i| format!("  {i}")).collect();
    match lines.is_empty() {
        true => format!("{title}: none"),
        false => format!("{title}:\n{}", lines.join("\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discord::state::BotState;
    use crate::discord::trace::link::Link;
    use crate::discord::trace::note::{Direction, Note};
    use crate::discord::trace::report::{ChannelRow, DmRow, ServerRow, Shard};

    /// Noon UTC, 1 November 2026.
    const NOON: u64 = 1_793_534_400_000;

    #[test]
    fn a_time_is_written_in_utc() {
        assert_eq!(utc(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(utc(NOON + 5_250), "2026-11-01 12:00:05 UTC");
    }

    /// **The trace reads as these lines, exactly.**
    #[test]
    fn a_trace_renders_as_these_lines() {
        let r = Report {
            state: Some(BotState {
                state: State::Online,
                since_ms: NOON,
                world: "life".into(),
                connected_ms: NOON - 60_000,
                member: Some(3),
            }),
            shard: Shard {
                link: Link {
                    connected: true,
                    session: Some("s-1".into()),
                    since_ms: Some(NOON),
                    last_event_ms: Some(NOON + 2_000),
                    last_heartbeat_ack_ms: None,
                },
                member: Some(3),
            },
            dms: vec![DmRow {
                account: "0a1b".into(),
                channel_id: 31,
                user_id: 77,
                mirrored_through: 2,
                last_seen_discord: 901,
                thread_id: Some("th1".into()),
                thread_last_seq: Some(3),
                unposted: 1,
            }],
            channels: vec![ChannelRow {
                guild_id: 600,
                channel_id: 610,
                thread: "th2".into(),
                name: Some("zenworld #general".into()),
                mirrored_through: 4,
                last_seen_discord: 905,
                thread_last_seq: Some(4),
                unposted: 0,
                open: true,
            }],
            servers: vec![ServerRow {
                guild_id: 600,
                name: "zenworld".into(),
            }],
            events: vec![Note::new(NOON + 2_000, Direction::In, "dm", "applied")
                .message(901)
                .channel(31)
                .author(77)
                .thread("th1")
                .seq(3)],
        };
        assert_eq!(
            render("1z", &r),
            "Bot 1z on Discord.\n\
             State: online in life since 2026-11-01 12:00:00 UTC (token connected 2026-11-01 11:59:00 UTC).\n\
             Shard: connected, session s-1, since 2026-11-01 12:00:00 UTC, last event 2026-11-01 12:00:02 UTC, \
             last heartbeat ack never, member 3.\n\
             DMs:\n  account 0a1b: channel 31, user 77, thread th1, last seq 3, mirrored through 2, unposted 1, \
             last seen 901\n\
             Channels:\n  zenworld #general (open): channel 610 in server 600, thread th2, last seq 4, \
             mirrored through 4, unposted 0, last seen 905\n\
             Servers:\n  zenworld (600)\n\
             Events, newest first:\n  2026-11-01 12:00:02 UTC in dm message=901 channel=31 author=77 thread=th1 \
             seq=3: applied"
        );
        let empty = Report {
            state: None,
            shard: Shard {
                link: Link::default(),
                member: None,
            },
            dms: vec![],
            channels: vec![],
            servers: vec![],
            events: vec![],
        };
        assert_eq!(
            render("1z", &empty),
            "Bot 1z on Discord.\nState: no record (offline).\nShard: not connected here, member unknown.\n\
             DMs: none\nChannels: none\nServers: none\nEvents, newest first: none"
        );
    }
}
