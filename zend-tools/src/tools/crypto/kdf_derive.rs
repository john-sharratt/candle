//! kdf_derive tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext};
use super::{decode_data, encode_output, CryptoError};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct KdfRequest {
    #[validate(length(min = 1))]
    pub password: String,
    #[validate(length(min = 1))]
    pub salt: String,
    pub salt_encoding: Option<String>,
    #[validate(length(min = 1))]
    pub algorithm: String,
    #[validate(range(min = 1, max = 64))]
    pub length: Option<u32>,
    pub iterations: Option<u32>,
    pub memory_kib: Option<u32>,
    pub output_encoding: Option<String>,
}

#[derive(Serialize)]
pub struct KdfResponse {
    pub derived_key: String,
    pub algorithm: String,
    pub length: u32,
    pub output_encoding: String,
}

pub struct KdfDerive;

impl Tool for KdfDerive {
    const NAME: &'static str = "kdf_derive";
    const DESCRIPTION: &'static str =
        "Derive a key from a password using argon2id, pbkdf2_sha256, or scrypt. \
         Use for: password hashing, key stretching.";

    type Request = KdfRequest;
    type Response = KdfResponse;
    type Error = CryptoError;

    fn run(_ctx: &ToolContext, req: KdfRequest) -> Result<KdfResponse, CryptoError> {
        let salt_enc = req.salt_encoding.as_deref().unwrap_or("text");
        let salt = decode_data(&req.salt, salt_enc)
            .map_err(CryptoError::InvalidDataEncoding)?;
        let length = req.length.unwrap_or(32) as usize;
        let out_enc = req.output_encoding.as_deref().unwrap_or("hex");
        let mut output = vec![0u8; length];

        match req.algorithm.as_str() {
            "argon2id" => {
                let iterations = req.iterations.unwrap_or(3);
                let memory = req.memory_kib.unwrap_or(65536);
                let params = argon2::Params::new(memory, iterations, 1, Some(length))
                    .map_err(|e| CryptoError::DerivationFailed(e.to_string()))?;
                let argon = argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
                argon.hash_password_into(req.password.as_bytes(), &salt, &mut output)
                    .map_err(|e| CryptoError::DerivationFailed(e.to_string()))?;
            }
            "pbkdf2_sha256" => {
                let iters = req.iterations.unwrap_or(100000);
                pbkdf2::pbkdf2_hmac::<sha2::Sha256>(req.password.as_bytes(), &salt, iters, &mut output);
            }
            "scrypt" => {
                let log_n = 15u8;
                let params = scrypt::Params::new(log_n, 8, 1, length)
                    .map_err(|e| CryptoError::DerivationFailed(e.to_string()))?;
                scrypt::scrypt(req.password.as_bytes(), &salt, &params, &mut output)
                    .map_err(|e| CryptoError::DerivationFailed(e.to_string()))?;
            }
            other => return Err(CryptoError::UnknownAlgorithm(other.to_string())),
        }

        Ok(KdfResponse {
            derived_key: encode_output(&output, out_enc),
            algorithm: req.algorithm,
            length: length as u32,
            output_encoding: out_enc.to_string(),
        })
    }
}

pub const KDF_DERIVE: RegisteredTool = RegisteredTool::new::<KdfDerive>();
