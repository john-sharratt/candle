//! ping_icmp tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{host_argument, DiagError};
use crate::{exec, net};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PingRequest {
    /// Hostname or IP address to ping.
    #[validate(length(min = 1, max = 253))]
    pub host: String,
    /// Number of echo requests to send (1-10). Default: 4.
    #[validate(range(min = 1, max = 10))]
    pub count: Option<u32>,
    /// Per-reply timeout in seconds (1-60). Default: 5.
    #[validate(range(min = 1, max = 60))]
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct PingResponse {
    pub host: String,
    pub resolved_ip: String,
    pub packets_sent: u32,
    pub packets_received: u32,
    pub packet_loss_pct: f64,
    pub rtt_min_ms: f64,
    pub rtt_avg_ms: f64,
    pub rtt_max_ms: f64,
    pub raw_output: String,
}

pub struct PingIcmp;

impl Tool for PingIcmp {
    const NAME: &'static str = "ping_icmp";
    const DESCRIPTION: &'static str =
        "Ping a host using ICMP echo and return round-trip statistics. Use for: checking \
         whether a host is reachable, measuring network latency, confirming a server is up, \
         diagnosing connectivity problems. Triggered by \"ping\", \"is X reachable\", \"can \
         you reach\", \"check if the server is up\", \"latency to\", \"is the host alive\", \
         \"test connectivity to\". Returns resolved IP, packets sent/received, packet loss \
         percentage, and RTT min/avg/max in milliseconds. Use port_scan to check specific \
         service ports; use trace_route for path diagnostics; use dns_lookup for name \
         resolution without connectivity test.";

    type Request = PingRequest;
    type Response = PingResponse;
    type Error = DiagError;

    fn run(ctx: &ToolContext, req: PingRequest) -> Result<PingResponse, DiagError> {
        let count = req.count.unwrap_or(4);
        let timeout = req.timeout_sec.unwrap_or(5);
        let grants = ctx.grants();
        let host = host_argument(&req.host)?;

        let resolved_ip = net::lookup_host(grants, host)
            .map_err(|e| DiagError::HostNotFound(format!("{host}: {e}")))?
            .into_iter()
            .next()
            .map(|ip| ip.to_string())
            .unwrap_or_else(|| host.to_string());

        let mut cmd =
            exec::command(grants, "ping").map_err(|e| DiagError::Failed(e.to_string()))?;
        #[cfg(target_os = "windows")]
        cmd.args([
            "-n",
            &count.to_string(),
            "-w",
            &(timeout * 1000).to_string(),
            host,
        ]);
        #[cfg(not(target_os = "windows"))]
        cmd.args(["-c", &count.to_string(), "-W", &timeout.to_string(), host]);

        let output = cmd.output().map_err(|e| DiagError::Failed(e.to_string()))?;
        // `ping` exits non-zero when nothing answered, which is a result (100%
        // loss), not a failure — but whatever it said on stderr is the reason,
        // so it rides along instead of being dropped.
        // The statistics are read from stdout alone: a stderr line that
        // happened to carry `received` or `minimum` would otherwise be parsed
        // as one.
        let mut raw = String::from_utf8_lossy(&output.stdout).into_owned();
        let (packets_received, rtt_min, rtt_avg, rtt_max) = parse_ping_output(&raw);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            raw.push_str("\n[stderr]\n");
            raw.push_str(stderr.trim());
        }

        Ok(PingResponse {
            host: req.host,
            resolved_ip,
            packets_sent: count,
            packets_received,
            packet_loss_pct: packet_loss_pct(count, packets_received),
            rtt_min_ms: rtt_min,
            rtt_avg_ms: rtt_avg,
            rtt_max_ms: rtt_max,
            raw_output: raw,
        })
    }
}

/// Percentage of `sent` echo requests that got no reply. A parse that reads
/// more replies than requests (a malformed or localised line) counts as none
/// lost rather than underflowing.
fn packet_loss_pct(sent: u32, received: u32) -> f64 {
    if sent == 0 {
        return 100.0;
    }
    f64::from(sent.saturating_sub(received)) / f64::from(sent) * 100.0
}

/// `(received, rtt_min, rtt_avg, rtt_max)` from `ping`'s output, Linux or
/// Windows.
///
/// The two print their round-trip line in different orders: Linux
/// `rtt min/avg/max/mdev = a/b/c/d ms`, Windows `Minimum = a, Maximum = b,
/// Average = c`, so each is read by its own labels.
fn parse_ping_output(output: &str) -> (u32, f64, f64, f64) {
    let mut received = 0u32;
    let mut min = 0.0f64;
    let mut avg = 0.0f64;
    let mut max = 0.0f64;

    for line in output.lines() {
        let l = line.to_lowercase();
        // Windows: "Packets: Sent = 4, Received = 4, Lost = 0 (0% loss),"
        if l.contains("received") && l.contains("sent") {
            if let Some(r) = extract_number_after(&l, "received = ") {
                received = r as u32;
            }
        }
        // Linux: "4 packets transmitted, 4 received, 0% packet loss, ..."
        if l.contains("packets transmitted") {
            let parts: Vec<&str> = l.split(',').collect();
            if let Some(recv_part) = parts.get(1) {
                if let Some(n) = recv_part.split_whitespace().next() {
                    received = n.parse().unwrap_or(0);
                }
            }
        }
        if l.contains("min/avg/max") {
            let nums: Vec<f64> = l
                .split(&['/', '=', ' ', ','][..])
                .filter_map(|s| s.trim().trim_end_matches("ms").parse::<f64>().ok())
                .collect();
            if nums.len() >= 3 {
                min = nums[0];
                avg = nums[1];
                max = nums[2];
            }
        }
        if l.contains("minimum") {
            let labelled = |label: &str| extract_number_after(&l, label);
            min = labelled("minimum = ").unwrap_or(min);
            max = labelled("maximum = ").unwrap_or(max);
            avg = labelled("average = ").unwrap_or(avg);
        }
    }

    (received, min, avg, max)
}

fn extract_number_after(s: &str, after: &str) -> Option<f64> {
    let pos = s.find(after)?;
    let rest = &s[pos + after.len()..];
    rest.split(|c: char| !c.is_numeric() && c != '.')
        .next()?
        .parse()
        .ok()
}

pub const PING_ICMP: RegisteredTool = RegisteredTool::new::<PingIcmp>();

#[cfg(test)]
mod tests {
    use validator::Validate;

    use super::*;

    const WINDOWS: &str = "\
Pinging 1.1.1.1 with 32 bytes of data:
Reply from 1.1.1.1: bytes=32 time=9ms TTL=57
Reply from 1.1.1.1: bytes=32 time=12ms TTL=57

Ping statistics for 1.1.1.1:
    Packets: Sent = 2, Received = 2, Lost = 0 (0% loss),
Approximate round trip times in milli-seconds:
    Minimum = 9ms, Maximum = 12ms, Average = 10ms
";

    const LINUX: &str = "\
PING 1.1.1.1 (1.1.1.1) 56(84) bytes of data.
64 bytes from 1.1.1.1: icmp_seq=1 ttl=57 time=9.12 ms

--- 1.1.1.1 ping statistics ---
2 packets transmitted, 1 received, 50% packet loss, time 1001ms
rtt min/avg/max/mdev = 9.120/10.500/11.880/1.380 ms
";

    /// **Windows prints Minimum, Maximum, Average — each lands in its own
    /// field**, not in Linux's min/avg/max order.
    #[test]
    fn windows_round_trip_times_are_read_by_label() {
        assert_eq!(parse_ping_output(WINDOWS), (2, 9.0, 10.0, 12.0));
    }

    #[test]
    fn linux_round_trip_times_are_read_in_order() {
        assert_eq!(parse_ping_output(LINUX), (1, 9.12, 10.5, 11.88));
    }

    #[test]
    fn loss_cannot_underflow() {
        assert_eq!(packet_loss_pct(4, 4), 0.0);
        assert_eq!(packet_loss_pct(4, 1), 75.0);
        assert_eq!(packet_loss_pct(4, 9), 0.0, "more replies than requests");
        assert_eq!(packet_loss_pct(0, 0), 100.0);
    }

    /// A timeout that would overflow `ms = s × 1000` is refused.
    #[test]
    fn the_timeout_is_bounded() {
        let req = |t: u32| PingRequest {
            host: "1.1.1.1".to_string(),
            count: None,
            timeout_sec: Some(t),
        };
        assert!(req(60).validate().is_ok());
        assert!(req(61).validate().is_err());
        assert!(req(0).validate().is_err());
        assert!(req(4_294_968).validate().is_err());
    }

    /// A host longer than any DNS name (253 bytes) is refused before it
    /// reaches the command line.
    #[test]
    fn the_host_is_bounded_by_the_longest_dns_name() {
        let req = |host: String| PingRequest {
            host,
            count: None,
            timeout_sec: None,
        };
        assert!(req("a".repeat(253)).validate().is_ok());
        assert!(req("a".repeat(254)).validate().is_err());
    }
}
