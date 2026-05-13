//! dns_lookup tool.

use std::net::IpAddr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::DiagError;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct DnsRequest {
    #[validate(length(min = 1))]
    pub host: String,
    pub record_type: Option<String>,
}

#[derive(Serialize)]
pub struct DnsResponse {
    pub host: String,
    pub record_type: String,
    pub records: Vec<String>,
}

pub struct DnsLookup;

impl Tool for DnsLookup {
    const NAME: &'static str = "dns_lookup";
    const DESCRIPTION: &'static str =
        "Perform a DNS lookup for a hostname and return its IP address records. Use for: \
         resolving a hostname to its IP, verifying DNS is set up correctly for a domain, \
         checking what address a name points to, confirming a DNS change has propagated. \
         Triggered by \"what is the IP of\", \"resolve this hostname\", \"DNS lookup for\", \
         \"what does X resolve to\", \"dig\", \"nslookup\". Returns record type and resolved \
         addresses. A/AAAA records use the OS resolver. MX/TXT/NS records return \
         not_supported. Use web_search for information about a domain; use ping_icmp to test \
         connectivity to the resolved address.";

    type Request = DnsRequest;
    type Response = DnsResponse;
    type Error = DiagError;

    fn run(_ctx: &ToolContext, req: DnsRequest) -> Result<DnsResponse, DiagError> {
        let record_type = req.record_type.as_deref().unwrap_or("A").to_uppercase();

        match record_type.as_str() {
            "A" | "AAAA" => {
                let ips = dns_lookup::lookup_host(&req.host)
                    .map_err(|e| DiagError::HostNotFound(format!("{}: {}", req.host, e)))?;
                let records: Vec<String> = ips.into_iter()
                    .filter(|ip| {
                        if record_type == "A" { ip.is_ipv4() }
                        else { ip.is_ipv6() }
                    })
                    .map(|ip| ip.to_string())
                    .collect();
                Ok(DnsResponse { host: req.host, record_type, records })
            }
            "PTR" => {
                let ip: IpAddr = req.host.parse()
                    .map_err(|_| DiagError::HostNotFound(format!("{} is not an IP address", req.host)))?;
                let name = dns_lookup::lookup_addr(&ip)
                    .map_err(|e| DiagError::HostNotFound(e.to_string()))?;
                Ok(DnsResponse { host: req.host, record_type, records: vec![name] })
            }
            other => {
                Ok(DnsResponse {
                    host: req.host,
                    record_type: other.to_string(),
                    records: vec!["not_supported: Only A/AAAA/PTR lookups supported via OS resolver".to_string()],
                })
            }
        }
    }
}

pub const DNS_LOOKUP: RegisteredTool = RegisteredTool::new::<DnsLookup>();
