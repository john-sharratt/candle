//! ip_scan tool.

use std::net::TcpStream;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::DiagError;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct IpScanRequest {
    /// IPv4 subnet in CIDR notation (e.g. "192.168.1.0/24"). At most 254 hosts are
    /// probed; only TCP port 80 is tested per address. 169.254.x.x link-local is blocked.
    #[validate(length(min = 1))]
    pub subnet: String,
    /// Per-host TCP connect timeout in milliseconds. Default: 200.
    pub timeout_ms: Option<u64>,
}

#[derive(Serialize)]
pub struct HostAlive {
    pub ip: String,
    pub alive: bool,
}

#[derive(Serialize)]
pub struct IpScanResponse {
    pub subnet: String,
    pub hosts: Vec<HostAlive>,
}

pub struct IpScan;

impl Tool for IpScan {
    const NAME: &'static str = "ip_scan";
    const DESCRIPTION: &'static str =
        "Sweep an entire CIDR subnet for live hosts, probing TCP port 80 on \
         each address in the range. Use for LAN discovery and network \
         mapping; link-local 169.254.x.x ranges are blocked.";

    type Request = IpScanRequest;
    type Response = IpScanResponse;
    type Error = DiagError;

    fn run(_ctx: &ToolContext, req: IpScanRequest) -> Result<IpScanResponse, DiagError> {
        let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(200));

        let parts: Vec<&str> = req.subnet.split('/').collect();
        if parts.len() != 2 {
            return Err(DiagError::Failed("invalid CIDR notation".to_string()));
        }
        let base_ip: std::net::Ipv4Addr = parts[0]
            .parse()
            .map_err(|e| DiagError::Failed(format!("invalid IP: {e}")))?;
        let prefix_len: u8 = parts[1]
            .parse()
            .map_err(|_| DiagError::Failed("invalid prefix length".to_string()))?;

        if base_ip.octets()[0] == 169 && base_ip.octets()[1] == 254 {
            return Err(DiagError::Failed(
                "169.254.x.x link-local subnet blocked".to_string(),
            ));
        }

        let host_bits = 32u32.saturating_sub(prefix_len as u32);
        let num_hosts = (1u32 << host_bits).min(254);
        let base_u32 = u32::from(base_ip) & !((1u32 << host_bits) - 1);

        let mut hosts = Vec::new();
        for i in 1..num_hosts {
            let ip = std::net::Ipv4Addr::from(base_u32 + i);
            let addr: std::net::SocketAddr = format!("{}:80", ip).parse().unwrap();
            let alive = TcpStream::connect_timeout(&addr, timeout).is_ok();
            hosts.push(HostAlive {
                ip: ip.to_string(),
                alive,
            });
        }

        Ok(IpScanResponse {
            subnet: req.subnet,
            hosts,
        })
    }
}

pub const IP_SCAN: RegisteredTool = RegisteredTool::new::<IpScan>();
