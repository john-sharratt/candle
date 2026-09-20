//! trace_route tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{extract_ip, extract_rtt, host_argument, DiagError};
use crate::exec;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct TraceRequest {
    /// Hostname or IP address to trace the route to.
    #[validate(length(min = 1, max = 253))]
    pub host: String,
    /// Maximum number of hops to probe (1-64). Defaults to 30.
    #[validate(range(min = 1, max = 64))]
    pub max_hops: Option<u32>,
    /// Per-hop probe timeout in seconds (1-60). Uses the OS default if omitted.
    #[validate(range(min = 1, max = 60))]
    pub timeout_sec: Option<u32>,
}

#[derive(Serialize)]
pub struct HopInfo {
    pub hop: u32,
    pub ip: String,
    pub hostname: String,
    pub rtt_ms: f64,
}

#[derive(Serialize)]
pub struct TraceResponse {
    pub host: String,
    pub hops: Vec<HopInfo>,
    pub raw_output: String,
}

pub struct TraceRoute;

impl Tool for TraceRoute {
    const NAME: &'static str = "trace_route";
    const DESCRIPTION: &'static str =
        "Map the router-by-router path packets take to reach a host, \
         reporting every intermediate hop and its latency. Use to locate \
         where along the route traffic fails or slows.";

    type Request = TraceRequest;
    type Response = TraceResponse;
    type Error = DiagError;

    fn run(ctx: &ToolContext, req: TraceRequest) -> Result<TraceResponse, DiagError> {
        let max_hops = req.max_hops.unwrap_or(30);
        let host = host_argument(&req.host)?;

        // tracert takes the per-hop wait in milliseconds (-w); traceroute takes it
        // in seconds (-w). Only pass it when the caller specified one.
        #[cfg(target_os = "windows")]
        let cmd = {
            let mut cmd = exec::command(ctx.grants(), "tracert")
                .map_err(|e| DiagError::Failed(e.to_string()))?;
            cmd.args(["-h", &max_hops.to_string()]);
            if let Some(t) = req.timeout_sec {
                cmd.args(["-w", &(t * 1000).to_string()]);
            }
            cmd
        };

        #[cfg(not(target_os = "windows"))]
        let cmd = {
            let mut cmd = exec::command(ctx.grants(), "traceroute")
                .map_err(|e| DiagError::Failed(e.to_string()))?;
            cmd.args(["-m", &max_hops.to_string()]);
            if let Some(t) = req.timeout_sec {
                cmd.args(["-w", &t.to_string()]);
            }
            cmd
        };

        let mut cmd = cmd;
        let output = cmd
            .arg(host)
            .output()
            .map_err(|e| DiagError::Failed(e.to_string()))?;
        let raw = String::from_utf8_lossy(&output.stdout).into_owned();

        let hops = parse_traceroute(&raw);

        Ok(TraceResponse {
            host: req.host,
            hops,
            raw_output: raw,
        })
    }
}

fn parse_traceroute(output: &str) -> Vec<HopInfo> {
    let mut hops = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();
        if let Ok(hop_num) = parts.first().unwrap_or(&"").trim().parse::<u32>() {
            let rest = parts.get(1..).map(|p| p.join(" ")).unwrap_or_default();
            let ip = extract_ip(&rest).unwrap_or_else(|| "*".to_string());
            let rtt = extract_rtt(&rest).unwrap_or(0.0);
            hops.push(HopInfo {
                hop: hop_num,
                ip: ip.clone(),
                hostname: ip,
                rtt_ms: rtt,
            });
        }
    }
    hops
}

pub const TRACE_ROUTE: RegisteredTool = RegisteredTool::new::<TraceRoute>();

#[cfg(test)]
mod tests {
    use super::*;

    fn req(host: &str, max_hops: Option<u32>, timeout_sec: Option<u32>) -> TraceRequest {
        TraceRequest {
            host: host.to_string(),
            max_hops,
            timeout_sec,
        }
    }

    /// The hop count and per-hop wait stay inside what the OS tools accept, and
    /// `timeout_sec × 1000` (tracert's milliseconds) cannot overflow.
    #[test]
    fn hops_and_timeout_are_bounded() {
        assert!(req("1.1.1.1", Some(64), Some(60)).validate().is_ok());
        assert!(req("1.1.1.1", Some(65), None).validate().is_err());
        assert!(req("1.1.1.1", Some(0), None).validate().is_err());
        assert!(req("1.1.1.1", None, Some(61)).validate().is_err());
        assert!(req("1.1.1.1", None, Some(0)).validate().is_err());
        assert!(req("1.1.1.1", None, Some(4_294_968)).validate().is_err());
        assert!(req(&"a".repeat(254), None, None).validate().is_err());
    }
}
