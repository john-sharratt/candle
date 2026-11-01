//! UPnP SOAP over HTTP: the envelope a control request is, the one-tag read of
//! an answer, and the POST that carries it — shared by every profile that
//! speaks UPnP (the Sonos player, a generic media renderer).

use std::net::Ipv4Addr;
use std::time::Duration;

use anyhow::{bail, Result};
use reqwest::redirect::Policy;
use reqwest::Client;

/// A UPnP service: its type and where its control is on the device.
#[derive(Clone, Copy)]
pub struct Service {
    pub urn: &'static str,
    pub path: &'static str,
}

/// The SOAP envelope for `action` on `service`, its arguments in order.
pub fn envelope(service: Service, action: &str, args: &[(&str, &str)]) -> String {
    let args: String = args
        .iter()
        .map(|(k, v)| format!("<{k}>{v}</{k}>"))
        .collect();
    format!(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?><s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body><u:{action} xmlns:u=\"{urn}\">{args}</u:{action}></s:Body></s:Envelope>",
        urn = service.urn
    )
}

/// The text of the first `<tag>` in `xml`, unescaped once.
pub fn tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&format!("</{tag}>"))?;
    Some(unescape(xml[start..start + end].trim()))
}

/// XML's five escapes, undone.
pub fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// An argument value made safe to place in element text: the three characters
/// that would otherwise be read as markup, escaped. `&` goes first so the
/// escapes it introduces are not escaped again.
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// A client that waits `answer` and is never redirected off the device.
pub fn client(answer: Duration) -> Result<Client> {
    Ok(Client::builder()
        .timeout(answer)
        .redirect(Policy::none())
        .build()?)
}

/// POST `action` on `service` at the device, returning the answer's body.
pub async fn call(
    client: &Client,
    address: Ipv4Addr,
    port: u16,
    service: Service,
    action: &str,
    args: &[(&str, &str)],
) -> Result<String> {
    let url = format!("http://{address}:{port}{}", service.path);
    let answer = client
        .post(url)
        .header("Content-Type", "text/xml; charset=\"utf-8\"")
        .header("SOAPACTION", format!("\"{}#{action}\"", service.urn))
        .body(envelope(service, action, args))
        .send()
        .await?;
    let status = answer.status();
    let text = answer.text().await?;
    if !status.is_success() {
        bail!("the device refused {action}: {status}");
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RENDERING: Service = Service {
        urn: "urn:schemas-upnp-org:service:RenderingControl:1",
        path: "/MediaRenderer/RenderingControl/Control",
    };

    /// **A call is the SOAP envelope the device expects, byte for byte.**
    #[test]
    fn a_call_is_the_envelope_the_device_expects() {
        assert_eq!(
            envelope(RENDERING, "SetVolume", &[("InstanceID", "0"), ("Channel", "Master"), ("DesiredVolume", "20")]),
            "<?xml version=\"1.0\" encoding=\"utf-8\"?><s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body><u:SetVolume xmlns:u=\"urn:schemas-upnp-org:service:RenderingControl:1\"><InstanceID>0</InstanceID><Channel>Master</Channel><DesiredVolume>20</DesiredVolume></u:SetVolume></s:Body></s:Envelope>"
        );
    }

    /// **A tag is read out of the answer and unescaped once** — the DIDL-Lite
    /// a media answer nests comes back as plain text.
    #[test]
    fn a_tag_is_read_and_unescaped() {
        let xml = "<s:Body><u:R><CurrentVolume>23</CurrentVolume><Meta>&lt;x&gt;a &amp;amp; b&lt;/x&gt;</Meta></u:R></s:Body>";
        assert_eq!(tag(xml, "CurrentVolume").as_deref(), Some("23"));
        assert_eq!(tag(xml, "Meta").as_deref(), Some("<x>a &amp; b</x>"));
        assert_eq!(tag(xml, "Absent"), None);
    }

    /// **An argument's markup characters are escaped, and read back whole** —
    /// a stream URL with a query string survives the envelope.
    #[test]
    fn an_argument_is_escaped_for_element_text() {
        assert_eq!(escape("a & b <c>"), "a &amp; b &lt;c&gt;");
        let url = "http://host/stream?a=1&b=2";
        let body = format!("<CurrentURI>{}</CurrentURI>", escape(url));
        assert_eq!(tag(&body, "CurrentURI").as_deref(), Some(url));
    }
}
