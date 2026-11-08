//! A response's bytes, read only up to a cap: a body past it is abandoned as
//! it arrives, never read whole and then refused.

use std::io::Read;

use reqwest::blocking::Response;

use super::error::FetchError;

/// `response`'s body, at most `cap` bytes; [`FetchError::TooLarge`] for one
/// longer, read no further than one byte past the cap.
pub fn read_within(response: Response, cap: usize) -> Result<Vec<u8>, FetchError> {
    let mut bytes = Vec::new();
    response
        .take(cap as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| FetchError::FetchFailed(e.to_string()))?;
    if bytes.len() > cap {
        return Err(FetchError::TooLarge { cap });
    }
    Ok(bytes)
}

/// The first `n` bytes of `response`'s body, as text, and whether there was
/// more: what an upload's answer shows of what the server said back.
pub fn read_head(response: Response, n: usize) -> (String, bool) {
    let mut bytes = Vec::new();
    let _ = response.take(n as u64 + 1).read_to_end(&mut bytes);
    let more = bytes.len() > n;
    bytes.truncate(n);
    (String::from_utf8_lossy(&bytes).into_owned(), more)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    use super::*;
    use crate::grants::{Capability, Grants};
    use crate::net;

    /// A server on loopback answering every request with `body`, once per
    /// connection, for `connections` connections; its address.
    fn serving(body: &'static [u8], connections: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming().take(connections) {
                let mut s = stream.unwrap();
                let mut request = [0u8; 1024];
                let _ = s.read(&mut request);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(body);
            }
        });
        format!("http://{addr}/")
    }

    fn get(url: &str) -> Response {
        net::http_client_builder(Grants::NONE.with(Capability::Network))
            .unwrap()
            .build()
            .unwrap()
            .get(url)
            .send()
            .unwrap()
    }

    /// **A body within its cap is its exact bytes**, binary included; one a
    /// byte past it is refused.
    #[test]
    fn a_body_is_read_to_its_cap() {
        let png: &'static [u8] = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR";
        let url = serving(png, 3);
        assert_eq!(read_within(get(&url), png.len()).unwrap(), png);
        assert_eq!(read_within(get(&url), 64).unwrap(), png);
        assert!(matches!(
            read_within(get(&url), png.len() - 1),
            Err(FetchError::TooLarge { cap: 15 })
        ));
    }

    /// **The head of an answer is its first bytes**, and says whether more
    /// followed.
    #[test]
    fn the_head_of_an_answer_is_its_first_bytes() {
        let url = serving(b"created: library/moon.png", 2);
        assert_eq!(read_head(get(&url), 7), ("created".to_string(), true));
        assert_eq!(
            read_head(get(&url), 64),
            ("created: library/moon.png".to_string(), false)
        );
    }
}
