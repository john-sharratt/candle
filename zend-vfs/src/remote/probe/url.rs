//! Where an origin answers over HTTPS.
//!
//! An `https://` origin is probed as it is. An SSH origin is probed at the
//! same host and path over HTTPS, which GitHub, GitLab and Gitea all serve —
//! and a host that does not refuses the probe, which then falls back to the
//! remote's own transport. Credentials written into a URL are never carried:
//! the probe presents its own, per host.

/// An origin's HTTPS form: the repository URL requests are made under, and
/// the host that decides which credentials go with them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeUrl {
    /// `https://host/path`, no credentials, no trailing `/`.
    pub url: String,
    /// The host, lowercase, without a port.
    pub host: String,
}

impl ProbeUrl {
    /// Where the advertisement is read.
    pub fn advertisement(&self) -> String {
        format!("{}/info/refs?service=git-upload-pack", self.url)
    }

    /// Where `ls-refs` is posted.
    pub fn upload_pack(&self) -> String {
        format!("{}/git-upload-pack", self.url)
    }
}

/// The HTTPS form of the origin `url`, as `git remote` reports it; `None`
/// for a local path, a `file://` URL, plain `http://` (a credential would
/// cross the network in the clear) or a path relative to a home folder.
pub fn https_url(url: &str) -> Option<ProbeUrl> {
    let url = url.trim();
    if let Some((scheme, rest)) = url.split_once("://") {
        let (authority, path) = rest.split_once('/')?;
        let (host, port) = host_and_port(authority);
        return match scheme.to_ascii_lowercase().as_str() {
            // An HTTPS origin is served where it says, port included.
            "https" => probe_url(host, port, path),
            // An SSH or git-protocol port is not the HTTPS one.
            "ssh" | "git" | "git+ssh" | "ssh+git" => probe_url(host, None, path),
            _ => None,
        };
    }
    // scp-like `[user@]host:path`. A Windows drive (`C:\x`, `C:/x`) has a
    // one-letter "host" and is a local path.
    let (authority, path) = url.split_once(':')?;
    if authority.len() < 2 || authority.contains('/') || authority.contains('\\') {
        return None;
    }
    probe_url(host_and_port(authority).0, None, path)
}

/// `[user[:password]@]host[:port]` → the host, lowercase, and the port.
fn host_and_port(authority: &str) -> (String, Option<&str>) {
    let hostport = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    // An IPv6 literal is bracketed; its colons are not a port.
    let (host, port) = match hostport.strip_prefix('[') {
        Some(v6) => match v6.split_once(']') {
            Some((addr, rest)) => (addr, rest.strip_prefix(':')),
            None => (v6, None),
        },
        None => match hostport.split_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (hostport, None),
        },
    };
    (host.to_ascii_lowercase(), port.filter(|p| !p.is_empty()))
}

fn probe_url(host: String, port: Option<&str>, path: &str) -> Option<ProbeUrl> {
    let path = path.trim_start_matches('/').trim_end_matches('/');
    if host.is_empty() || path.is_empty() || path.starts_with('~') {
        return None;
    }
    // An IPv6 address goes back between brackets in a URL.
    let at = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.clone()
    };
    let url = match port {
        Some(port) => format!("https://{at}:{port}/{path}"),
        None => format!("https://{at}/{path}"),
    };
    Some(ProbeUrl { url, host })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn url(u: &str) -> Option<String> {
        https_url(u).map(|p| p.url)
    }

    #[test]
    fn every_ssh_form_maps_to_https_on_the_same_host() {
        let want = Some("https://github.com/john-sharratt/candle.git".to_string());
        for origin in [
            "git@github.com:john-sharratt/candle.git",
            "github.com:john-sharratt/candle.git",
            "ssh://git@github.com/john-sharratt/candle.git",
            "ssh://git@github.com:22/john-sharratt/candle.git",
            "git+ssh://git@GitHub.com/john-sharratt/candle.git",
            "git://github.com/john-sharratt/candle.git",
        ] {
            assert_eq!(url(origin), want, "{origin}");
        }
    }

    /// Credentials in the URL — or `git remote`'s redaction of them — are
    /// dropped; the probe presents its own. An HTTPS origin's port is where
    /// it is served, so it stays; the host names the credentials without it.
    #[test]
    fn an_https_origin_is_probed_as_it_is_without_credentials() {
        assert_eq!(
            url("https://github.com/x/y.git"),
            Some("https://github.com/x/y.git".into())
        );
        let p = https_url("https://***@gitlab.example.com:8443/group/sub/y/").unwrap();
        assert_eq!(p.url, "https://gitlab.example.com:8443/group/sub/y");
        assert_eq!(p.host, "gitlab.example.com");
        let p = https_url("https://user:secret@Example.COM/y").unwrap();
        assert_eq!(p.host, "example.com");
        assert_eq!(p.url, "https://example.com/y");
        assert_eq!(
            p.advertisement(),
            "https://example.com/y/info/refs?service=git-upload-pack"
        );
        assert_eq!(p.upload_pack(), "https://example.com/y/git-upload-pack");
    }

    #[test]
    fn local_paths_files_plain_http_and_home_relative_paths_have_none() {
        for origin in [
            "C:\\Users\\x\\origin",
            "C:/Users/x/origin",
            "/srv/git/origin.git",
            "../origin",
            "file:///C:/Users/x/origin",
            "http://example.com/y.git",
            "git@example.com:~user/y.git",
            "ssh://git@example.com/",
        ] {
            assert_eq!(url(origin), None, "{origin}");
        }
    }

    #[test]
    fn an_ipv6_host_keeps_its_address() {
        let p = https_url("ssh://git@[::1]:2222/y.git").unwrap();
        assert_eq!(p.host, "::1");
        assert_eq!(p.url, "https://[::1]/y.git", "an SSH port is not HTTPS's");
        let p = https_url("https://[::1]:8443/y.git").unwrap();
        assert_eq!(p.url, "https://[::1]:8443/y.git");
    }
}
