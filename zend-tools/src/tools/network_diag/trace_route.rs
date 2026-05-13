//! trace_route tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::{extract_ip, extract_rtt, DiagError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct TraceRequest {
    #[validate(length(min = 1))]
    pub host: String,
    pub max_hops: Option<u32>,
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
        "Trace the network path to a host showing each hop. \
         Use for: diagnosing routing issues, measuring per-hop latency.";

    type Request = TraceRequest;
    type Response = TraceResponse;
    type Error = DiagError;

    fn run(_ctx: &ToolContext, req: TraceRequest) -> Result<TraceResponse, DiagError> {
        let max_hops = req.max_hops.unwrap_or(30);

        #[cfg(target_os = "windows")]
        let output = std::process::Command::new("tracert")
            .args(["-h", &max_hops.to_string(), &req.host])
            .output();

        #[cfg(not(target_os = "windows"))]
        let output = std::process::Command::new("traceroute")
            .args(["-m", &max_hops.to_string(), &req.host])
            .output();

        let output = output.map_err(|e| DiagError::Failed(e.to_string()))?;
        let raw = String::from_utf8_lossy(&output.stdout).into_owned();

        let hops = parse_traceroute(&raw);

        Ok(TraceResponse { host: req.host, hops, raw_output: raw })
    }
}

fn parse_traceroute(output: &str) -> Vec<HopInfo> {
    let mut hops = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
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
