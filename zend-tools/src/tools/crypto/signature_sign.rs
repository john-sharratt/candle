//! signature_sign tool.

use p256::ecdsa::signature::Signer;
use p256::ecdsa::{Signature as P256Signature, SigningKey as P256SigningKey};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{decode_data, CryptoError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct SigSignRequest {
    pub data: String,
    pub data_encoding: Option<String>,
    #[validate(length(min = 1))]
    pub credential_name: String,
    #[validate(length(min = 1))]
    pub algorithm: String,
}

#[derive(Serialize)]
pub struct SigSignResponse {
    pub signature_hex: String,
    pub algorithm: String,
}

pub struct SignatureSign;

impl Tool for SignatureSign {
    const NAME: &'static str = "signature_sign";
    const DESCRIPTION: &'static str =
        "Produce a digital signature over a message using a private signing \
         key from the credential store (ed25519 or p256_sha256). Proves \
         authorship; the counterpart signature_verify checks it.";

    type Request = SigSignRequest;
    type Response = SigSignResponse;
    type Error = CryptoError;

    fn run(ctx: &ToolContext, req: SigSignRequest) -> Result<SigSignResponse, CryptoError> {
        let cred = ctx
            .credentials
            .get_by_name(&req.credential_name)
            .ok_or_else(|| CryptoError::CredentialNotFound(req.credential_name.clone()))?;

        if cred.cred_type != "signing_key" {
            return Err(CryptoError::InvalidCredentialType(cred.cred_type));
        }

        let enc = req.data_encoding.as_deref().unwrap_or("text");
        let data = decode_data(&req.data, enc).map_err(CryptoError::InvalidDataEncoding)?;

        let sig_hex = match req.algorithm.as_str() {
            "ed25519" => {
                use ed25519_dalek::pkcs8::DecodePrivateKey;
                use ed25519_dalek::Signer;
                let sk = ed25519_dalek::SigningKey::from_pkcs8_pem(&cred.secret)
                    .map_err(|e| CryptoError::SigningFailed(e.to_string()))?;
                let sig = sk.sign(&data);
                hex::encode(sig.to_bytes())
            }
            "p256_sha256" => {
                use p256::pkcs8::DecodePrivateKey;
                let sk = P256SigningKey::from_pkcs8_pem(&cred.secret)
                    .map_err(|e| CryptoError::SigningFailed(e.to_string()))?;
                let sig: P256Signature = sk.sign(&data);
                hex::encode(sig.to_bytes())
            }
            other => return Err(CryptoError::UnknownAlgorithm(other.to_string())),
        };

        Ok(SigSignResponse {
            signature_hex: sig_hex,
            algorithm: req.algorithm,
        })
    }
}

pub const SIGNATURE_SIGN: RegisteredTool = RegisteredTool::new::<SignatureSign>();
