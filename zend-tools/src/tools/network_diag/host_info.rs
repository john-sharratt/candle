//! host_info tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::DiagError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct HostInfoRequest {
    #[validate(length(min = 1))]
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
        "Resolve a hostname to IPs and perform reverse DNS lookup. \
         Use for: host investigation, checking DNS records.";

    type Request = HostInfoRequest;
    type Response = HostInfoResponse;
    type Error = DiagError;

    fn run(_ctx: &ToolContext, req: HostInfoRequest) -> Result<HostInfoResponse, DiagError> {
        let ips = dns_lookup::lookup_host(&req.host)
            .map_err(|e| DiagError::HostNotFound(e.to_string()))?;

        let ip_strings: Vec<String> = ips.iter().map(|ip| ip.to_string()).collect();
        let mut reverse_dns = Vec::new();
        for ip in &ips {
            if let Ok(name) = dns_lookup::lookup_addr(ip) {
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
