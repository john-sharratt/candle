//! Tests that the configuration parses, resolves its paths and refuses settings
//! it cannot honour.
//!
//! Hosts, upstreams, backoff and cluster routes are checked as they are read, and
//! each secret path resolves against the config file. An unknown key is refused,
//! because a typo that silently does nothing is worse than a refusal. A cluster
//! route needs an app, an upstream and a key. Two defaults, or none, is an error.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) const YAML: &str = r#"
server:
  bind: "0.0.0.0:80"
sites:
  - name: npcd
    hosts: ["npcd.test", "*.npcd.dev"]
    roots: ["."]
    default: true
    api:
      - prefix: /v1
        upstream: local
      - prefix: /v1/stream
        upstream: "http://10.0.0.9:8081"
"#;

    pub(in super::super) fn cfg() -> Config {
        Config::from_yaml(YAML, Path::new(".")).expect("valid")
    }

    #[test]
    fn yaml_parses_and_resolves_hosts() {
        let c = cfg();
        assert_eq!(c.server.bind.port(), 80);
        assert_eq!(c.site_for(Some("NPCD.test:8080")).name, "npcd");
        assert_eq!(c.site_for(Some("a.npcd.dev")).name, "npcd");
        assert_eq!(c.site_for(Some("nothing.example")).name, "npcd"); // default
    }

    #[test]
    fn local_and_url_upstreams_both_parse() {
        let c = cfg();
        let s = &c.sites[0];
        assert_eq!(s.route_for("/v1/status").unwrap().upstream, Upstream::Local);
        assert_eq!(
            s.route_for("/v1/stream/x").unwrap().upstream,
            Upstream::Url("http://10.0.0.9:8081".into())
        );
    }

    #[test]
    fn wildcard_excludes_the_apex() {
        let c = cfg();
        assert!(c.sites[0].matches_host("x.npcd.dev"));
        assert!(!c.sites[0].matches_host("npcd.dev"));
    }

    #[test]
    fn prefix_respects_segment_boundaries() {
        assert!(segment_prefix("/v1", "/v1"));
        assert!(segment_prefix("/v1/x", "/v1"));
        assert!(!segment_prefix("/v1x", "/v1"));
    }

    #[test]
    fn backoff_doubles_then_caps_at_ten_seconds() {
        let b = Backoff {
            initial_ms: 250,
            max_ms: 10_000,
        };
        assert_eq!(b.delay(1).as_millis(), 250);
        assert_eq!(b.delay(2).as_millis(), 500);
        assert_eq!(b.delay(3).as_millis(), 1_000);
        assert_eq!(b.delay(6).as_millis(), 8_000);
        assert_eq!(b.delay(7).as_millis(), 10_000); // capped
        assert_eq!(b.delay(99).as_millis(), 10_000); // and stays capped
    }

    #[test]
    fn self_signed_belongs_to_an_https_upstream_only() {
        let ok = "sites:\n  - name: a\n    default: true\n    api:\n      - {prefix: /, upstream: \"https://10.0.0.5:8444\", self_signed: true}\n";
        let c = Config::from_yaml(ok, Path::new(".")).expect("valid");
        assert!(c.sites[0].api[0].self_signed);
        for bad in [
            "sites:\n  - name: a\n    default: true\n    api:\n      - {prefix: /, upstream: \"http://10.0.0.5:8081\", self_signed: true}\n",
            "sites:\n  - name: a\n    default: true\n    api:\n      - {prefix: /, upstream: local, self_signed: true}\n",
        ] {
            assert!(Config::from_yaml(bad, Path::new(".")).is_err(), "{bad}");
        }
    }

    /// A route's `cluster:` names one of the three apps, keeps an upstream URL
    /// for while no member is ready, and needs a `cluster:` block naming the
    /// group key; the key's path resolves against the config.
    #[test]
    fn a_cluster_route_needs_an_app_an_upstream_and_a_key() {
        let ok = &format!(
            "cluster: {{group_key_file: \"secrets/group.key\", {}}}\nsites:\n  - name: a\n    default: true\n    api:\n      - {{prefix: /, upstream: \"http://10.0.0.5:8081\", cluster: zend}}\n",
            machines_file()
        );
        let c = Config::from_yaml(ok, Path::new("/srv/web")).expect("valid");
        assert_eq!(c.sites[0].api[0].cluster.as_deref(), Some("zend"));
        let cluster = c.cluster.clone().unwrap();
        assert_eq!(
            cluster.group_key_file,
            Path::new("/srv/web").join("secrets/group.key")
        );
        // Certification's defaults: a 30-minute grace, a 10% live share.
        assert_eq!(
            (cluster.floor_grace_minutes, cluster.canary_share),
            (30, 10)
        );
        // No status key unless one is named; a named one resolves like the
        // group key.
        assert_eq!(cluster.status_key_file, None);
        let keyed = Config::from_yaml(
            &ok.replace(
                "{group_key_file:",
                "{status_key_file: \"secrets/status.key\", group_key_file:",
            ),
            Path::new("/srv/web"),
        )
        .unwrap();
        assert_eq!(
            keyed.cluster.unwrap().status_key_file,
            Some(Path::new("/srv/web").join("secrets/status.key"))
        );
        // No ops key unless one is named; its file resolves the same way, and
        // it names the admin it acts as.
        assert_eq!(cluster.ops_key, None);
        let ops = Config::from_yaml(
            &ok.replace(
                "{group_key_file:",
                "{ops_key: {file: \"secrets/ops.key\", admin: \"op@tokera.com\"}, group_key_file:",
            ),
            Path::new("/srv/web"),
        )
        .unwrap();
        assert_eq!(
            ops.cluster.unwrap().ops_key,
            Some(OpsKeyConfig {
                file: Path::new("/srv/web").join("secrets/ops.key"),
                admin: "op@tokera.com".into(),
            })
        );
        // Service keys: none unless named; each file resolves the same way,
        // and a name given twice is refused.
        assert!(cluster.service_keys.is_empty());
        let one = "{name: yenbot, key_file: \"secrets/yenbot.key\", routes: [/v1/models], \
                   topics: {subscribe: [repo/yenbot], publish: [\"gate/yenbot/result/*\"]}}";
        let services = Config::from_yaml(
            &ok.replace(
                "{group_key_file:",
                &format!("{{service_keys: [{one}], group_key_file:"),
            ),
            Path::new("/srv/web"),
        )
        .unwrap();
        assert_eq!(
            services.cluster.unwrap().service_keys,
            vec![ServiceKeyConfig {
                name: "yenbot".into(),
                key_file: Path::new("/srv/web").join("secrets/yenbot.key"),
                routes: vec!["/v1/models".into()],
                topics: ServiceTopics {
                    subscribe: vec!["repo/yenbot".into()],
                    publish: vec!["gate/yenbot/result/*".into()],
                },
            }]
        );
        let twice = Config::from_yaml(
            &ok.replace(
                "{group_key_file:",
                &format!("{{service_keys: [{one}, {one}], group_key_file:"),
            ),
            Path::new("/srv/web"),
        )
        .unwrap_err();
        assert!(
            format!("{twice:#}").contains("names `yenbot` twice"),
            "{twice:#}"
        );
        // Pushes deployed at once and forwarded ports: none unless named; the
        // webhook's secret resolves like every key file, and a forward names
        // an app and its ports.
        assert_eq!(cluster.deploy_on_push, None);
        assert!(cluster.udp_forward.is_empty());
        let game = Config::from_yaml(
            &ok.replace(
                "{group_key_file:",
                "{deploy_on_push: {repos: [battle-cities], github_hook_key_file: \
                 \"secrets/github-hook.key\"}, udp_forward: [{app: battlecities, ports: \
                 [{listen: 4300, member: 14300}, {listen: 4123, member: 14123}]}], \
                 group_key_file:",
            ),
            Path::new("/srv/web"),
        )
        .unwrap()
        .cluster
        .unwrap();
        assert_eq!(
            game.deploy_on_push,
            Some(DeployOnPush {
                repos: vec!["battle-cities".into()],
                github_hook_key_file: Some(Path::new("/srv/web").join("secrets/github-hook.key")),
            })
        );
        assert_eq!(
            game.udp_forward,
            vec![UdpForward {
                app: "battlecities".into(),
                ports: vec![
                    UdpPort {
                        listen: 4300,
                        member: 14300,
                    },
                    UdpPort {
                        listen: 4123,
                        member: 14123,
                    },
                ],
            }]
        );
        for (bad, why) in [
            (
                "[{app: chess, ports: [{listen: 4300, member: 4300}]}]",
                "which is no app",
            ),
            ("[{app: battlecities, ports: []}]", "names no port"),
            (
                "[{app: battlecities, ports: [{listen: 4300, member: 0}]}]",
                "names no port",
            ),
        ] {
            let refused = Config::from_yaml(
                &ok.replace(
                    "{group_key_file:",
                    &format!("{{udp_forward: {bad}, group_key_file:"),
                ),
                Path::new("/srv/web"),
            )
            .unwrap_err();
            assert!(format!("{refused:#}").contains(why), "{refused:#}");
        }
        let tuned = Config::from_yaml(
            &ok.replace(
                "{group_key_file:",
                "{floor_grace_minutes: 5, canary_share: 25, group_key_file:",
            ),
            Path::new("."),
        )
        .unwrap()
        .cluster
        .unwrap();
        assert_eq!((tuned.floor_grace_minutes, tuned.canary_share), (5, 25));
        assert_eq!(
            c.sites[0].route_for("/x").unwrap().cluster.as_deref(),
            Some("zend")
        );
        for bad in [
            // No such app.
            format!("cluster: {{group_key_file: k, {}}}\nsites:\n  - name: a\n    default: true\n    api:\n      - {{prefix: /, upstream: \"http://10.0.0.5:8081\", cluster: other}}\n", machines_file()),
            // Nowhere to go while no member is ready.
            format!("cluster: {{group_key_file: k, {}}}\nsites:\n  - name: a\n    default: true\n    api:\n      - {{prefix: /, upstream: local, cluster: zend}}\n", machines_file()),
            // No key for a member to prove.
            "sites:\n  - name: a\n    default: true\n    api:\n      - {prefix: /, upstream: \"http://10.0.0.5:8081\", cluster: zend}\n".to_string(),
        ] {
            assert!(Config::from_yaml(&bad, Path::new(".")).is_err(), "{bad}");
        }
    }

    /// `machines_file: '<the shipped machine table>'`, for a config read
    /// against a base where no table lies.
    pub(in super::super) fn machines_file() -> String {
        let shipped = Path::new(env!("CARGO_MANIFEST_DIR")).join("../host/machines.yaml");
        format!("machines_file: '{}'", shipped.display())
    }

    /// Peers are listed by URL, with the name a LAN address is verified
    /// against; a gateway without a TLS entrance has no certificates to share
    /// and is refused them.
    #[test]
    fn peers_need_the_tls_entrance() {
        let site = "sites:\n  - {name: a, roots: ['.'], default: true}\n";
        let peers = &format!(
            "cluster: {{group_key_file: k, {}, peers: [{{url: \"https://192.168.0.6:9443\", name: tokera.com}}, {{url: \"https://gw2.tokera.com\"}}]}}\n",
            machines_file()
        );
        let tls = "server:\n  tls: {bind: \"0.0.0.0:9443\", acme: {store: acme}}\n";
        let c = Config::from_yaml(&format!("{tls}{peers}{site}"), Path::new(".")).unwrap();
        assert_eq!(
            c.cluster.unwrap().peers,
            [
                Peer {
                    url: "https://192.168.0.6:9443".into(),
                    name: Some("tokera.com".into()),
                },
                Peer {
                    url: "https://gw2.tokera.com".into(),
                    name: None,
                },
            ]
        );
        let err = Config::from_yaml(&format!("{peers}{site}"), Path::new("."))
            .unwrap_err()
            .to_string();
        assert!(err.contains("there is no TLS entrance"), "{err}");
    }

    #[test]
    fn two_defaults_is_an_error() {
        let y = "sites:\n  - {name: a, roots: ['.'], default: true}\n  - {name: b, roots: ['.'], default: true}\n";
        assert!(Config::from_yaml(y, Path::new(".")).is_err());
    }

    #[test]
    fn no_default_is_an_error() {
        let y = "sites:\n  - {name: a, roots: ['.']}\n";
        assert!(Config::from_yaml(y, Path::new(".")).is_err());
    }

    #[test]
    fn unknown_key_is_rejected_rather_than_ignored() {
        // A typo in a config that silently does nothing is worse than a refusal.
        let y = "sites:\n  - {name: a, roots: ['.'], default: true, hsots: []}\n";
        assert!(Config::from_yaml(y, Path::new(".")).is_err());
    }
}
