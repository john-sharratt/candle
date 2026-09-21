//! host_info tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::DiagError;
use crate::net;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct HostInfoRequest {
    /// Hostname or IP address to profile.
    #[validate(length(min = 1, max = 253))]
    pub host: String,
}

#[derive(Serialize)]
pub struct HostInfoResponse {
    pub host: String,
    pub resolved_ips: Vec<String>,
    pub reverse_dns: Vec<String>,
}

pub struct HostInfo;

impl Tool for HostInfo {
    const NAME: &'static str = "host_info";
    const DESCRIPTION: &'static str =
        "Profile a host: resolve its name to IP addresses, then run a \
         reverse-DNS lookup back from those IPs. Use for host identification \
         and DNS-record auditing — no connectivity probe is sent.";

    type Request = HostInfoRequest;
    type Response = HostInfoResponse;
    type Error = DiagError;

    /// Reports this host's own configuration; writes nothing and reaches no
    /// peer.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: HostInfoRequest) -> Result<HostInfoResponse, DiagError> {
        let grants = ctx.grants();
        let ips = net::lookup_host(grants, &req.host)
            .map_err(|e| DiagError::HostNotFound(e.to_string()))?;

        let ip_strings: Vec<String> = ips.iter().map(|ip| ip.to_string()).collect();
        let mut reverse_dns = Vec::new();
        for ip in &ips {
            if let Ok(name) = net::lookup_addr(grants, ip) {
                reverse_dns.push(name);
            }
        }

        Ok(HostInfoResponse {
            host: req.host,
            resolved_ips: ip_strings,
            reverse_dns,
        })
    }
}

pub const HOST_INFO: RegisteredTool = RegisteredTool::new::<HostInfo>();
