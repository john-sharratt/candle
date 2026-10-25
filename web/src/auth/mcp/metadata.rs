//! The two discovery documents a client reads before it signs in
//! (`docs/claude_zend_gateway.md` §12, §13.1).
//!
//! - The protected-resource document (RFC 9728) at
//!   `/.well-known/oauth-protected-resource/mcp`, naming `/mcp` exactly and
//!   this origin as its one authorization server — claude.ai reads only the
//!   first of `authorization_servers`.
//! - The authorization-server document (RFC 8414) at
//!   `/.well-known/oauth-authorization-server`. It advertises Client ID
//!   Metadata Documents and the `none` client authentication, which is what
//!   makes claude.ai and Claude Code both use CIMD, so no registration endpoint
//!   is offered.

use serde_json::{json, Value};

/// The path the protected resource is served at, on every site.
pub const MCP_PATH: &str = "/mcp";
/// Where the protected-resource document is served, for `/mcp`.
pub const RESOURCE_DOC_PATH: &str = "/.well-known/oauth-protected-resource/mcp";
/// Where the authorization-server document is served.
pub const SERVER_DOC_PATH: &str = "/.well-known/oauth-authorization-server";
pub const AUTHORIZE_PATH: &str = "/oauth/authorize";
pub const TOKEN_PATH: &str = "/oauth/token";
/// The one scope a token carries.
pub const SCOPE: &str = "zend";

/// The protected-resource document for `resource`, issued by `issuer`.
pub fn protected_resource(resource: &str, issuer: &str) -> Value {
    json!({
        "resource": resource,
        "authorization_servers": [issuer],
        "bearer_methods_supported": ["header"],
        "scopes_supported": [SCOPE],
        "resource_name": "zend",
    })
}

/// The authorization-server document for `issuer`.
pub fn authorization_server(issuer: &str) -> Value {
    json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{issuer}{AUTHORIZE_PATH}"),
        "token_endpoint": format!("{issuer}{TOKEN_PATH}"),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "client_id_metadata_document_supported": true,
        "scopes_supported": [SCOPE],
        "authorization_response_iss_parameter_supported": true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_documents_are_exact() {
        assert_eq!(
            protected_resource("https://code.tokera.com/mcp", "https://code.tokera.com")
                .to_string(),
            r#"{"resource":"https://code.tokera.com/mcp","authorization_servers":["https://code.tokera.com"],"bearer_methods_supported":["header"],"scopes_supported":["zend"],"resource_name":"zend"}"#
        );
        assert_eq!(
            authorization_server("https://code.tokera.com").to_string(),
            r#"{"issuer":"https://code.tokera.com","authorization_endpoint":"https://code.tokera.com/oauth/authorize","token_endpoint":"https://code.tokera.com/oauth/token","response_types_supported":["code"],"grant_types_supported":["authorization_code","refresh_token"],"code_challenge_methods_supported":["S256"],"token_endpoint_auth_methods_supported":["none"],"client_id_metadata_document_supported":true,"scopes_supported":["zend"],"authorization_response_iss_parameter_supported":true}"#
        );
    }
}
