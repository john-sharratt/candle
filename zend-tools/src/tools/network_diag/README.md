# network_diag — dns_lookup, ping_icmp, trace_route, port_scan, ip_scan, host_info

Thin wrappers around OS networking utilities that invoke a subprocess and parse structured output.

## Files

| File | Tool | Underlying command | Output |
|------|------|--------------------|--------|
| `dns_lookup.rs` | `dns_lookup` | `dns-lookup` crate | A/AAAA/MX/TXT/PTR records |
| `ping_icmp.rs` | `ping_icmp` | system `ping` | RTT stats (min/avg/max/loss) |
| `trace_route.rs` | `trace_route` | system `traceroute`/`tracert` | Ordered hop list with RTTs |
| `port_scan.rs` | `port_scan` | TCP connect | Open/closed/filtered per port |
| `ip_scan.rs` | `ip_scan` | system `ping` sweep | Live hosts in a CIDR range |
| `host_info.rs` | `host_info` | `dns-lookup` crate | Reverse DNS, IP list |
| `mod.rs` | — | — | `DiagError`; `extract_ip`; `extract_rtt` |

## Shared error codes

All six tools share `DiagError` with three codes:

| Code | When |
|------|------|
| `host_not_found` | DNS resolution returned no records |
| `operation_failed` | Subprocess error, parse failure, or OS-level denial |
| `not_supported` | Operation not available on this platform |

## Shared helpers

- `extract_ip(s)` — pulls the first IPv4 address from a subprocess output string
- `extract_rtt(s)` — extracts the first RTT value (ms) from a string like `"time=1.23 ms"`

## Platform notes

- `ping_icmp` and `trace_route` call platform system commands (`ping`/`traceroute` on
  Linux/macOS, `ping`/`tracert` on Windows).  ICMP may require elevated privileges.
- `port_scan` uses plain TCP connect; no raw sockets needed.
- `ip_scan` sends ICMP pings to each address in the range; may be slow for large CIDRs
  and may require privileges.
