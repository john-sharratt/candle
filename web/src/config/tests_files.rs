//! Tests of the file-backed parts of the configuration: sign-in, TLS, the machine
//! table and the role table.
//!
//! A file that is absent turns its feature off, so a public site keeps serving.
//! A file that is present and broken is refused, since the deployment has asked
//! for the feature. The same block cannot be given both inline and as a file. The
//! shipped table names its credential files rather than carrying them, so it is
//! the same bytes on every machine.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_parse::tests::{machines_file, YAML};

    /// **A cluster names its machine table, and the table is read**: a block
    /// without one, a table missing, or one that does not parse is refused,
    /// since a gateway with no table gives every machine no work. The shipped
    /// config names the table every daemon compiles in.
    #[test]
    fn a_cluster_reads_its_machine_table() {
        let site = "sites:\n  - {name: a, roots: ['.'], default: true}\n";
        let c = Config::from_yaml(
            &format!(
                "cluster: {{group_key_file: k, {}}}\n{site}",
                machines_file()
            ),
            Path::new("."),
        )
        .unwrap();
        let table = c.cluster.unwrap().machines;
        assert!(table.conversations("BlackWorld").is_some(), "{table:?}");

        for bad in [
            "cluster: {group_key_file: k}\n".to_string(),
            "cluster: {group_key_file: k, machines_file: absent.yaml}\n".to_string(),
        ] {
            assert!(
                Config::from_yaml(&format!("{bad}{site}"), Path::new(".")).is_err(),
                "{bad}"
            );
        }
        let dir = tmpdir();
        std::fs::write(dir.join("machines.yaml"), "M:\n  power: 3\n").unwrap();
        assert!(Config::from_yaml(
            &format!("cluster: {{group_key_file: k, machines_file: machines.yaml}}\n{site}"),
            &dir
        )
        .is_err());

        let text = include_str!("../../web.yaml");
        let line = text
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with("machines_file:"))
            .expect("web.yaml names a machine table");
        let rel = line["machines_file:".len()..].trim().trim_matches('"');
        let shipped =
            std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).unwrap();
        assert_eq!(shipped, include_str!("../../../host/machines.yaml"));
    }

    // ── auth_file ───────────────────────────────────────────────────────────
    //
    // The site table has to be identical on every machine, so the one piece
    // that is not — a real `client_id` — lives in a file beside it. These pin
    // the three states that split creates.

    /// A directory of this test's own. Counted rather than timestamped: the
    /// platform clock is coarse enough that tests starting together get the
    /// same nanosecond, and then the same directory.
    pub(in super::super) fn tmpdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "web-authfile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    pub(in super::super) fn with_auth_file(dir: &Path, body: Option<&str>) -> Result<Config> {
        if let Some(b) = body {
            std::fs::create_dir_all(dir.join("secrets")).unwrap();
            std::fs::write(dir.join("secrets").join("auth.yaml"), b).unwrap();
        }
        let y = format!(
            "auth_file: \"secrets/auth.yaml\"\n{}",
            YAML.trim_start_matches('\n')
        );
        Config::from_yaml(&y, dir)
    }

    pub(in super::super) const AUTH: &str = r#"
cookie_domain: ".tokera.com"
session_ttl_hours: 720
session_secret_file: "secrets/session.key"
google:
  client_id: "abc.apps.googleusercontent.com"
  client_secret_file: "secrets/google.secret"
  redirect_uri: "https://tokera.com/auth/callback"
"#;

    /// Absent file, sign-in off — the same meaning the commented-out block had.
    /// A public site must not stop serving because a key it does not need is
    /// missing.
    #[test]
    fn a_missing_auth_file_means_no_sign_in_rather_than_a_failure() {
        let dir = tmpdir();
        let c = with_auth_file(&dir, None).expect("a missing auth file is not an error");
        assert!(c.auth.is_none());
        // And the resolved path is kept, so `--check` can say what it looked for.
        assert_eq!(c.auth_file.unwrap(), dir.join("secrets").join("auth.yaml"));
    }

    #[test]
    fn a_present_auth_file_configures_sign_in() {
        let dir = tmpdir();
        let c = with_auth_file(&dir, Some(AUTH)).expect("valid auth file");
        let a = c.auth.expect("sign-in is configured");
        assert_eq!(a.cookie_domain, ".tokera.com");
        assert_eq!(a.google.client_id, "abc.apps.googleusercontent.com");
        // Its paths resolve against the config, exactly like `roots`.
        assert_eq!(
            a.session_secret_file,
            dir.join("secrets").join("session.key")
        );
    }

    /// **A host signs in on the estate whose cookie domain covers it**, port
    /// and case aside; any other host on the first.
    #[test]
    fn a_host_signs_in_on_the_estate_that_covers_it() {
        let dir = tmpdir();
        let mut a = with_auth_file(&dir, Some(AUTH)).unwrap().auth.unwrap();
        a.domains = vec![CookieDomain {
            cookie_domain: ".yenbot.com.au".into(),
            redirect_uri: "https://yenbot.com.au/auth/callback".into(),
        }];
        let yen = CookieDomain {
            cookie_domain: ".yenbot.com.au".into(),
            redirect_uri: "https://yenbot.com.au/auth/callback".into(),
        };
        let tokera = CookieDomain {
            cookie_domain: ".tokera.com".into(),
            redirect_uri: a.google.redirect_uri.clone(),
        };
        assert_eq!(a.domain_for("yenbot.com.au"), yen);
        assert_eq!(a.domain_for("WWW.Yenbot.com.au:443"), yen);
        assert_eq!(a.domain_for("bot.tokera.com"), tokera);
        assert_eq!(a.domain_for("notyenbot.com.au"), tokera);
        assert_eq!(a.domain_for(""), tokera);
    }

    /// Present but broken is fatal. By then the deployment has said it wants
    /// sign-in, and a gateway that quietly has none is the worse outcome —
    /// which is the whole reason a *missing* file is treated differently.
    #[test]
    fn a_broken_auth_file_is_a_hard_failure() {
        for bad in [
            "cookie_domain: [not a string]\n",
            "google: {}\n",
            "cookie_domain: \".x\"\nsession_secret_file: \"k\"\ngoogle:\n  client_id: a\n  client_secret_file: s\n  redirect_uri: r\n  typo: 1\n",
            ": : :\n",
        ] {
            let dir = tmpdir();
            assert!(
                with_auth_file(&dir, Some(bad)).is_err(),
                "{bad:?} was accepted"
            );
        }
    }

    /// Two sources for one block would make whichever lost look ignored.
    #[test]
    fn inline_auth_and_an_auth_file_together_are_refused() {
        let dir = tmpdir();
        std::fs::create_dir_all(dir.join("secrets")).unwrap();
        std::fs::write(dir.join("secrets").join("auth.yaml"), AUTH).unwrap();
        let y = format!(
            "auth_file: \"secrets/auth.yaml\"\nauth:\n{}\n{}",
            AUTH.trim_start_matches('\n')
                .lines()
                .map(|l| format!("  {l}"))
                .collect::<Vec<_>>()
                .join("\n"),
            YAML.trim_start_matches('\n')
        );
        assert!(Config::from_yaml(&y, &dir).is_err());
    }

    // ── tls ─────────────────────────────────────────────────────────────────

    pub(in super::super) fn with_tls_file(dir: &Path, body: Option<&str>) -> Result<Config> {
        if let Some(b) = body {
            std::fs::create_dir_all(dir.join("secrets")).unwrap();
            std::fs::write(dir.join("secrets").join("tls.yaml"), b).unwrap();
        }
        let y = format!(
            "server:\n  bind: \"0.0.0.0:80\"\n  tls_file: \"secrets/tls.yaml\"\n{}",
            YAML.trim_start_matches('\n')
                .lines()
                .skip(2)
                .collect::<Vec<_>>()
                .join("\n")
        );
        Config::from_yaml(&y, dir)
    }

    pub(in super::super) const TLS: &str =
        "bind: \"0.0.0.0:443\"\nacme:\n  store: \"secrets/acme\"\n";

    /// A TLS file opens the entrance, its store resolved against the config
    /// like every other path in the table.
    #[test]
    fn a_tls_file_opens_the_tls_entrance() {
        let dir = tmpdir();
        let c = with_tls_file(&dir, Some(TLS)).expect("valid");
        let tls = c.server.tls.expect("the TLS entrance is configured");
        assert_eq!(tls.bind.port(), 443);
        assert_eq!(tls.acme.store, dir.join("secrets").join("acme"));
        assert_eq!(c.server.bind.port(), 80, "the plain entrance is untouched");
    }

    /// No file: no TLS entrance, and the plain one still serves — the meaning
    /// an absent auth file has, so one table runs everywhere.
    #[test]
    fn no_tls_file_means_no_tls_entrance_rather_than_a_failure() {
        let c = with_tls_file(&tmpdir(), None).expect("an absent file is not an error");
        assert!(c.server.tls.is_none());
    }

    /// Present and wrong is a deployment that meant to have TLS: refused — a
    /// field nothing reads included, so a leftover setting cannot look live.
    #[test]
    fn a_broken_tls_file_is_a_hard_failure() {
        for bad in [
            "bind: \"0.0.0.0:443\"\n",
            "bind: \"0.0.0.0:443\"\nacme:\n  store: s\n  typo: 1\n",
            "bind: \"0.0.0.0:443\"\nacme:\n  directory: staging\n  store: s\n",
        ] {
            assert!(
                with_tls_file(&tmpdir(), Some(bad)).is_err(),
                "{bad:?} was accepted"
            );
        }
    }

    /// The shipped table names an auth file and does not carry the block, so
    /// it is the same bytes on every machine and can always be pulled.
    #[test]
    fn the_shipped_table_keeps_its_credentials_out_of_line() {
        let text = include_str!("../../web.yaml");
        assert!(
            text.contains("auth_file:"),
            "web.yaml does not name an auth file"
        );
        // Only as a comment. An uncommented `auth:` here is the thing that made
        // the file locally-modified on one box and unpullable ever after.
        for line in text.lines() {
            assert!(
                !line.starts_with("auth:"),
                "web.yaml carries an inline `auth:` block"
            );
        }
    }

    /// **The gateway reads the estate's role table**, the one every daemon
    /// compiles in: with none, it took its own admin for a user and offered
    /// neither the status page nor the Cluster link.
    #[test]
    fn the_shipped_table_reads_the_estates_roles() {
        let text = include_str!("../../web.yaml");
        let line = text
            .lines()
            .find(|l| l.starts_with("roles_file:"))
            .expect("web.yaml names a role table");
        let rel = line["roles_file:".len()..].trim().trim_matches('"');
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let shipped = std::fs::read_to_string(base.join(rel)).unwrap();
        assert_eq!(shipped, include_str!("../../../host/roles.yaml"));

        let dir = tmpdir();
        std::fs::write(
            dir.join("roles.yaml"),
            "people:\n  - email: a@x.com\n    role: admin\n",
        )
        .unwrap();
        let y = format!(
            "roles_file: \"roles.yaml\"\n{}",
            YAML.trim_start_matches('\n')
        );
        let c = Config::from_yaml(&y, &dir).unwrap();
        assert!(!c.roles.names_no_admin(), "the table's admin was read");
        assert_eq!(c.roles_file.unwrap(), dir.join("roles.yaml"));

        let missing = format!(
            "roles_file: \"absent.yaml\"\n{}",
            YAML.trim_start_matches('\n')
        );
        assert!(
            Config::from_yaml(&missing, &dir).is_err(),
            "a named table must exist"
        );
        assert!(
            Config::from_yaml(&format!("roles:\n  people: []\n{YAML}"), &dir).is_err(),
            "an inline table is not read"
        );
    }
}
