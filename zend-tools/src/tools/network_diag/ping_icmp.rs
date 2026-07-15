//! ping_icmp tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::DiagError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PingRequest {
    /// Hostname or IP address to ping.
    #[validate(length(min = 1))]
    pub host: String,
    /// Number of echo requests to send (1-10). Default: 4.
    #[validate(range(min = 1, max = 10))]
    pub count: Option<u32>,
    /// Per-reply timeout in seconds. Default: 5.
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

    fn run(_ctx: &ToolContext, req: PingRequest) -> Result<PingResponse, DiagError> {
        let count = req.count.unwrap_or(4);
        let timeout = req.timeout_sec.unwrap_or(5);

        let resolved_ip = dns_lookup::lookup_host(&req.host)
            .ok()
            .and_then(|ips| ips.into_iter().next())
            .map(|ip| ip.to_string())
            .unwrap_or_else(|| req.host.clone());

        #[cfg(target_os = "windows")]
        let output = std::process::Command::new("ping")
            .args([
                "-n",
                &count.to_string(),
                "-w",
                &(timeout * 1000).to_string(),
                &req.host,
            ])
            .output();

        #[cfg(not(target_os = "windows"))]
        let output = std::process::Command::new("ping")
            .args([
                "-c",
                &count.to_string(),
                "-W",
                &timeout.to_string(),
                &req.host,
            ])
            .output();

        let output = output.map_err(|e| DiagError::Failed(e.to_string()))?;
        let raw = String::from_utf8_lossy(&output.stdout).into_owned();

        let (packets_received, rtt_min, rtt_avg, rtt_max) = parse_ping_output(&raw);

        let packet_loss = if count > 0 {
            ((count - packets_received) as f64 / count as f64) * 100.0
        } else {
            100.0
        };

        Ok(PingResponse {
            host: req.host,
            resolved_ip,
            packets_sent: count,
            packets_received,
            packet_loss_pct: packet_loss,
            rtt_min_ms: rtt_min,
            rtt_avg_ms: rtt_avg,
            rtt_max_ms: rtt_max,
            raw_output: raw,
        })
    }
}

fn parse_ping_output(output: &str) -> (u32, f64, f64, f64) {
    let mut received = 0u32;
    let mut min = 0.0f64;
    let mut avg = 0.0f64;
    let mut max = 0.0f64;

    for line in output.lines() {
        let l = line.to_lowercase();
        if l.contains("received") && l.contains("sent") {
            if let Some(r) = extract_number_after(&l, "received = ") {
                received = r as u32;
            } else if let Some(r) = extract_number_after(&l, ", received = ") {
                received = r as u32;
            }
        }
        if l.contains("packets transmitted") {
            let parts: Vec<&str> = l.split(',').collect();
            if let Some(recv_part) = parts.get(1) {
                if let Some(n) = recv_part.trim().split_whitespace().next() {
                    received = n.parse().unwrap_or(0);
                }
            }
        }
        if l.contains("min/avg/max") || l.contains("minimum") {
            let nums: Vec<f64> = l
                .split(&['/', '=', ' ', ','][..])
                .filter_map(|s| s.trim().trim_end_matches("ms").parse::<f64>().ok())
                .collect();
            if nums.len() >= 3 {
                min = nums[0];
                avg = nums[1];
                max = nums[2];
            } else if nums.len() >= 1 {
                if l.contains("minimum") {
                    min = nums[0];
                }
                if l.contains("maximum") {
                    max = *nums.last().unwrap_or(&0.0);
                }
                if l.contains("average") {
                    avg = *nums.last().unwrap_or(&0.0);
                }
            }
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
