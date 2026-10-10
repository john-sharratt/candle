//! Before a name goes to the CA, prove it reaches this gateway.
//!
//! A name whose DNS points somewhere else, or whose tunnel route is missing,
//! fails HTTP-01 — and every failure spends one of the five failed
//! validations the CA allows that name an hour, and counts toward the run of
//! consecutive failures that gets an account paused. So the gateway asks first:
//! it publishes a random token of its own, fetches it back through the public
//! name exactly as the CA will (plain `http://`, following redirects), and only
//! a name that answers with the token goes into the order.

use std::time::Duration;

use rand::distr::Alphanumeric;
use rand::Rng;
use reqwest::redirect::Policy;
use reqwest::Client;

use super::challenges::{Challenges, PREFIX};

/// How long one name has to answer.
const TIMEOUT: Duration = Duration::from_secs(15);

/// 43 characters — the length of a real ACME token, so the probe looks to every
/// hop in between exactly like the request that follows it.
fn random_token() -> String {
    rand::rng()
        .sample_iter(&Alphanumeric)
        .take(43)
        .map(char::from)
        .collect()
}

/// The names among `names` that served this gateway's own probe back. Every
/// name is asked, concurrently, and a failure is logged with its reason.
pub async fn reachable(challenges: &Challenges, names: &[String]) -> Vec<String> {
    let token = random_token();
    let answer = random_token();
    challenges.set(&token, &answer);
    let client = match Client::builder()
        .timeout(TIMEOUT)
        .redirect(Policy::limited(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "acme: the preflight client could not be built");
            challenges.clear(&token);
            return Vec::new();
        }
    };
    let checks = names.iter().map(|name| {
        let client = client.clone();
        let url = format!("http://{name}{PREFIX}{token}");
        let answer = answer.clone();
        async move {
            let ok = match client.get(&url).send().await {
                Ok(res) if res.status().is_success() => match res.text().await {
                    Ok(body) if body == answer => true,
                    Ok(_) => {
                        tracing::warn!(%name, "acme: preflight answered, but not with this gateway's token");
                        false
                    }
                    Err(e) => {
                        tracing::warn!(%name, error = %e, "acme: preflight body failed");
                        false
                    }
                },
                Ok(res) => {
                    tracing::warn!(%name, status = %res.status(), "acme: preflight refused");
                    false
                }
                Err(e) => {
                    tracing::warn!(%name, error = %e, "acme: preflight could not reach it");
                    false
                }
            };
            (name.clone(), ok)
        }
    });
    let results = futures::future::join_all(checks).await;
    challenges.clear(&token);
    results
        .into_iter()
        .filter_map(|(n, ok)| ok.then_some(n))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use axum::Router;
    use tokio::net::TcpListener;

    async fn listen(router: Router) -> String {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, router).await.unwrap() });
        addr.to_string()
    }

    /// Only a name that hands back this gateway's own token passes: one that
    /// reaches some other server — answering, but not with the token — and one
    /// that reaches nothing are both left out of the order.
    #[tokio::test]
    async fn only_a_name_that_serves_this_gateways_token_passes() {
        let challenges = Challenges::default();
        let here = listen(challenges.router()).await;
        let elsewhere = listen(Router::new().route(
            "/.well-known/acme-challenge/:token",
            get(|| async { "somebody else's answer" }),
        ))
        .await;
        let nowhere = {
            let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let a = l.local_addr().unwrap().to_string();
            drop(l);
            a
        };
        let names = vec![here.clone(), elsewhere, nowhere];
        assert_eq!(reachable(&challenges, &names).await, vec![here]);
    }

    /// The probe is withdrawn once asked, so it never lingers as an answer.
    #[tokio::test]
    async fn the_probe_token_is_withdrawn_afterwards() {
        let challenges = Challenges::default();
        let here = listen(challenges.router()).await;
        reachable(&challenges, &[here]).await;
        assert!(challenges.is_empty());
    }
}
