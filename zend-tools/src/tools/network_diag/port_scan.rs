//! port_scan tool.

use std::net::SocketAddr;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::DiagError;
use crate::net;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct PortScanRequest {
    /// Hostname or IP address to scan.
    #[validate(length(min = 1))]
    pub host: String,
    /// TCP ports to check (1-100 ports per call).
    #[validate(length(min = 1, max = 100))]
    pub ports: Vec<u16>,
    /// Per-port TCP connect timeout in milliseconds. Default: 500.
    pub timeout_ms: Option<u64>,
}

#[derive(Serialize)]
pub struct PortResult {
    pub port: u16,
    pub open: bool,
}

#[derive(Serialize)]
pub struct PortScanResponse {
    pub host: String,
    pub results: Vec<PortResult>,
}

pub struct PortScan;

impl Tool for PortScan {
    const NAME: &'static str = "port_scan";
    const DESCRIPTION: &'static str =
        "Check whether specific TCP ports are open on a host by attempting connections. Use \
         for: verifying a service is listening, checking whether a firewall is blocking a port, \
         service discovery on known-candidate ports, confirming a deployment started correctly. \
         Triggered by \"is port X open on\", \"check if the service is running on\", \"scan \
         these ports\", \"is SSH/HTTP/database accessible on\". Returns open/closed status per \
         port with a configurable timeout per attempt. Provide up to 100 ports per call. Use \
         ping_icmp first to confirm the host is reachable.";

    type Request = PortScanRequest;
    type Response = PortScanResponse;
    type Error = DiagError;

    fn run(ctx: &ToolContext, req: PortScanRequest) -> Result<PortScanResponse, DiagError> {
        let timeout = Duration::from_millis(req.timeout_ms.unwrap_or(500));
        let grants = ctx.grants();
        // Resolved once, up front: a refusal or an unknown host is an error the
        // model reads, not a column of "closed" ports it would believe.
        let target = net::resolve(grants, &format!("{}:0", req.host))
            .map_err(|e| DiagError::HostNotFound(format!("{}: {e}", req.host)))?;
        let mut results = Vec::new();

        for port in &req.ports {
            let sa = SocketAddr::new(target.ip(), *port);
            let open = net::tcp_connect(grants, &sa, Some(timeout)).is_ok();
            results.push(PortResult { port: *port, open });
        }

        Ok(PortScanResponse {
            host: req.host,
            results,
        })
    }
}

pub const PORT_SCAN: RegisteredTool = RegisteredTool::new::<PortScan>();
