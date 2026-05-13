mod harness;

use serde_json::json;

#[test]
fn dns_lookup_localhost() {
    let resp = harness::invoke("dns_lookup", json!({"host": "localhost", "record_type": "A"}));
    // Either succeeds with 127.0.0.1 or gives some response
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert_eq!(r["host"], "localhost");
    }
}

#[test]
fn dns_lookup_invalid_host() {
    // A nonsense host that should fail resolution
    let resp = harness::invoke(
        "dns_lookup",
        json!({"host": "this.host.definitely.does.not.exist.invalid", "record_type": "A"}),
    );
    // Either host_not_found error OR success with empty records
    if let Some(code) = resp.get("error").and_then(|e| e.as_str()) {
        assert!(
            code == "host_not_found" || code == "lookup_failed",
            "unexpected error code: {code}"
        );
    } else {
        let r = harness::expect_success(resp);
        // Records may be empty
        let records = r["records"].as_array().unwrap();
        assert!(records.is_empty());
    }
}

#[test]
fn port_scan_range() {
    // Scan some ports on localhost — just check structure
    let resp = harness::expect_success(harness::invoke(
        "port_scan",
        json!({
            "host": "127.0.0.1",
            "ports": [80, 443, 65535],
            "timeout_ms": 100
        }),
    ));
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 3);
    for r in results {
        assert!(r["port"].as_u64().is_some());
        assert!(r["open"].is_boolean());
    }
}

#[test]
fn host_info_localhost() {
    let resp = harness::invoke("host_info", json!({"host": "localhost"}));
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert_eq!(r["host"], "localhost");
        assert!(r["resolved_ips"].is_array());
    }
}

#[test]
fn ping_icmp_structure() {
    // ping_icmp may or may not be available (requires privileges on some OSes)
    let resp = harness::invoke("ping_icmp", json!({"host": "127.0.0.1"}));
    // Just check it returns some response — success or a known error
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert!(r.get("host").is_some() || r.get("reachable").is_some());
    }
    // If it errors, that's acceptable too (e.g. permission_denied)
}

#[test]
fn dns_lookup_localhost_a() {
    let resp = harness::invoke("dns_lookup", json!({
        "host": "localhost",
        "record_type": "A"
    }));
    // Either succeeds with 127.0.0.1 or no A records (IPv6 only hosts)
    if resp.get("error").is_none() {
        let r = harness::expect_success(resp);
        assert_eq!(r["host"], "localhost");
        assert_eq!(r["record_type"], "A");
    }
}

#[test]
fn dns_lookup_mx_not_supported() {
    let resp = harness::expect_success(harness::invoke("dns_lookup", json!({
        "host": "example.com",
        "record_type": "MX"
    })));
    // Returns records with "not_supported" message
    let records = resp["records"].as_array().unwrap();
    if !records.is_empty() {
        assert!(records[0].as_str().unwrap().contains("not_supported"));
    }
}

#[test]
fn port_scan_known_closed_port() {
    // Port 1 should be closed on most systems
    let resp = harness::expect_success(harness::invoke("port_scan", json!({
        "host": "127.0.0.1",
        "ports": [1],
        "timeout_ms": 200
    })));
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["port"], 1);
    // closed is expected but we just check the structure
    assert!(results[0]["open"].is_boolean());
}

#[test]
fn port_scan_too_many_ports() {
    let ports: Vec<u16> = (1..=101).collect();
    let resp = harness::invoke("port_scan", json!({
        "host": "127.0.0.1",
        "ports": ports,
        "timeout_ms": 100
    }));
    harness::expect_error(&resp, "invalid_arguments");
}

