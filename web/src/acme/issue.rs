//! One ACME order, start to finish: authorizations by HTTP-01, finalization
//! with a fresh key, and the chain.
//!
//! Nothing here decides *whether* to order — that is the ledger's — and
//! nothing here retries. An order either issues or fails once, and the caller
//! records which, so a failure is always followed by the ledger's back-off
//! rather than by another attempt.

use std::fmt::{self, Display, Formatter};
use std::time::Duration;

use instant_acme::{
    Account, AuthorizationStatus, CertificateIdentifier, ChallengeType, Error as AcmeError,
    Identifier, NewOrder, OrderStatus, RetryPolicy,
};

use super::challenges::Challenges;

/// The problem type the CA answers with when a limit has been reached.
const RATE_LIMITED: &str = "urn:ietf:params:acme:error:rateLimited";

/// How long the CA has to validate and to sign. Let's Encrypt usually takes a
/// few seconds; past this the order is abandoned and counted as failed.
const POLL: RetryPolicy = RetryPolicy::new()
    .initial_delay(Duration::from_secs(2))
    .backoff(1.5)
    .timeout(Duration::from_secs(120));

/// An issued certificate: the chain (leaf first) and its key, both PEM.
pub struct Issued {
    pub chain_pem: String,
    pub key_pem: String,
}

#[derive(Debug)]
pub enum IssueError {
    /// The CA refused with `rateLimited` — a real limit or its "service busy"
    /// load-shedding, which it reports the same way. Every site backs off.
    RateLimited(String),
    Failed(String),
}

impl Display for IssueError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            IssueError::RateLimited(m) => write!(f, "rate limited: {m}"),
            IssueError::Failed(m) => f.write_str(m),
        }
    }
}

/// Sort a client error into the two kinds the ledger tells apart.
pub fn classify(e: AcmeError) -> IssueError {
    match &e {
        AcmeError::Api(p) if p.r#type.as_deref() == Some(RATE_LIMITED) => {
            IssueError::RateLimited(p.detail.clone().unwrap_or_else(|| e.to_string()))
        }
        _ => IssueError::Failed(e.to_string()),
    }
}

/// Order a certificate for `names`. `replaces` names the certificate this one
/// renews, when the CA's renewal window asked for it.
pub async fn issue(
    account: &Account,
    challenges: &Challenges,
    names: &[String],
    replaces: Option<CertificateIdentifier<'_>>,
) -> Result<Issued, IssueError> {
    let identifiers: Vec<Identifier> = names.iter().map(|n| Identifier::Dns(n.clone())).collect();
    let mut new = NewOrder::new(&identifiers);
    if let Some(r) = replaces {
        new = new.replaces(r);
    }
    let mut order = account.new_order(&new).await.map_err(classify)?;

    // Every token this order publishes, so every one is withdrawn however the
    // order ends.
    let mut published: Vec<String> = Vec::new();
    let result = async {
        let mut authorizations = order.authorizations();
        while let Some(authz) = authorizations.next().await {
            let mut authz = authz.map_err(classify)?;
            match authz.status {
                AuthorizationStatus::Pending => {}
                AuthorizationStatus::Valid => continue,
                other => {
                    return Err(IssueError::Failed(format!(
                        "an authorization was {other:?} before it was attempted"
                    )))
                }
            }
            let mut challenge = authz
                .challenge(ChallengeType::Http01)
                .ok_or_else(|| IssueError::Failed("the CA offered no http-01 challenge".into()))?;
            challenges.set(&challenge.token, challenge.key_authorization().as_str());
            published.push(challenge.token.clone());
            challenge.set_ready().await.map_err(classify)?;
        }

        match order.poll_ready(&POLL).await.map_err(classify)? {
            OrderStatus::Ready => {}
            other => return Err(IssueError::Failed(format!("the order became {other:?}"))),
        }
        let key_pem = order.finalize().await.map_err(classify)?;
        let chain_pem = order.poll_certificate(&POLL).await.map_err(classify)?;
        Ok(Issued { chain_pem, key_pem })
    }
    .await;

    for token in &published {
        challenges.clear(token);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use instant_acme::Problem;
    use serde_json::json;

    fn problem(kind: &str) -> AcmeError {
        let p: Problem = serde_json::from_value(json!({
            "type": kind,
            "detail": "too many certificates already issued",
            "status": 429,
        }))
        .unwrap();
        AcmeError::Api(p)
    }

    #[test]
    fn a_rate_limit_problem_is_told_apart_from_any_other_failure() {
        match classify(problem("urn:ietf:params:acme:error:rateLimited")) {
            IssueError::RateLimited(m) => assert_eq!(m, "too many certificates already issued"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            classify(problem("urn:ietf:params:acme:error:unauthorized")),
            IssueError::Failed(_)
        ));
        assert!(matches!(
            classify(AcmeError::Str("no certificate URL found")),
            IssueError::Failed(_)
        ));
    }
}
