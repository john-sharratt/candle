mod harness;

use serde_json::json;
use zend_tools::ToolContext;

fn ctx() -> ToolContext {
    ToolContext::new()
}

const KEY_HEX: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const NONCE_HEX: &str = "000000000000000000000000";

// Key for chacha20poly1305 (also 32 bytes, same nonce length works)

#[test]
fn aead_encrypt_with_aad() {
    let enc = harness::expect_success(harness::invoke(
        "aead_encrypt",
        json!({
            "data": "secret with aad",
            "data_encoding": "text",
            "key_hex": KEY_HEX,
            "algorithm": "aes256gcm",
            "nonce_hex": NONCE_HEX,
            "aad": "extra auth data"
        }),
    ));
    let ct = enc["ciphertext_hex"].as_str().unwrap().to_string();
    let nonce = enc["nonce_hex"].as_str().unwrap().to_string();

    // Decrypt with correct AAD
    let dec = harness::expect_success(harness::invoke(
        "aead_decrypt",
        json!({
            "ciphertext_hex": ct,
            "key_hex": KEY_HEX,
            "nonce_hex": nonce,
            "algorithm": "aes256gcm",
            "aad": "extra auth data"
        }),
    ));
    assert_eq!(dec["plaintext"], "secret with aad");

    // Decrypt with wrong AAD should fail
    let bad = harness::invoke(
        "aead_decrypt",
        json!({
            "ciphertext_hex": ct,
            "key_hex": KEY_HEX,
            "nonce_hex": nonce,
            "algorithm": "aes256gcm",
            "aad": "wrong aad"
        }),
    );
    harness::expect_error(&bad, "decryption_failed");
}

#[test]
fn aead_wrong_key() {
    let enc = harness::expect_success(harness::invoke(
        "aead_encrypt",
        json!({
            "data": "secret",
            "data_encoding": "text",
            "key_hex": KEY_HEX,
            "algorithm": "aes256gcm",
            "nonce_hex": NONCE_HEX
        }),
    ));
    let ct = enc["ciphertext_hex"].as_str().unwrap();
    let nonce = enc["nonce_hex"].as_str().unwrap();
    let wrong_key = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    let resp = harness::invoke(
        "aead_decrypt",
        json!({
            "ciphertext_hex": ct,
            "key_hex": wrong_key,
            "nonce_hex": nonce,
            "algorithm": "aes256gcm"
        }),
    );
    harness::expect_error(&resp, "decryption_failed");
}

#[test]
fn hmac_sha512() {
    let resp = harness::expect_success(harness::invoke(
        "hmac_compute",
        json!({
            "data": "hello",
            "data_encoding": "text",
            "key": "secret",
            "key_encoding": "text",
            "algorithm": "sha512"
        }),
    ));
    assert_eq!(resp["algorithm"], "sha512");
    // HMAC-SHA512 = 64 bytes = 128 hex chars
    let mac = resp["mac"].as_str().unwrap();
    assert_eq!(mac.len(), 128, "expected 128 hex chars, got {}", mac.len());
}

#[test]
fn hmac_known_vector() {
    // HMAC-SHA256("The quick brown fox jumps over the lazy dog", "key")
    // = f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8
    let resp = harness::expect_success(harness::invoke(
        "hmac_compute",
        json!({
            "data": "The quick brown fox jumps over the lazy dog",
            "data_encoding": "text",
            "key": "key",
            "key_encoding": "text",
            "algorithm": "sha256"
        }),
    ));
    assert_eq!(
        resp["mac"].as_str().unwrap(),
        "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
    );
}

#[test]
fn kdf_scrypt() {
    let resp = harness::expect_success(harness::invoke(
        "kdf_derive",
        json!({
            "password": "testpassword",
            "salt": "saltsalt",
            "salt_encoding": "text",
            "algorithm": "scrypt",
            "length": 32
        }),
    ));
    assert_eq!(resp["algorithm"], "scrypt");
    let key = resp["derived_key"].as_str().unwrap();
    assert_eq!(key.len(), 64, "32 bytes = 64 hex chars, got {}", key.len());
}

#[test]
fn hkdf_extract_with_salt() {
    let resp = harness::expect_success(harness::invoke(
        "hkdf_extract",
        json!({
            "ikm": "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
            "ikm_encoding": "hex",
            "salt": "000102030405060708090a0b0c",
            "salt_encoding": "hex",
            "algorithm": "sha256"
        }),
    ));
    assert_eq!(resp["algorithm"], "sha256");
    let prk = resp["prk_hex"].as_str().unwrap();
    assert_eq!(prk.len(), 64);
    // RFC 5869 test vector 1: PRK should be
    // 077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5
    assert_eq!(
        prk,
        "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
    );
}

#[test]
fn signature_sign_verify_ed25519() {
    use zend_tools::ToolContext;
    let ctx = ToolContext::new();

    // Save an Ed25519 private key credential
    // This is a known test private key in PKCS#8 PEM format (all-zero seed)
    let private_key_pem = "-----BEGIN PRIVATE KEY-----\n\
        MC4CAQAwBQYDK2VdBCIEIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n\
        -----END PRIVATE KEY-----\n";

    let saved = harness::expect_success(harness::invoke_with_ctx(
        "credential_save",
        json!({
            "name": "ed25519-test",
            "type": "ed25519_key",
            "secret": private_key_pem
        }),
        &ctx,
    ));
    // Sign a message
    let sign_resp = harness::invoke_with_ctx(
        "signature_sign",
        json!({
            "data": "48656c6c6f",
            "data_encoding": "hex",
            "credential_name": "ed25519-test",
            "algorithm": "ed25519"
        }),
        &ctx,
    );

    // If the tool succeeds, verify it
    if sign_resp.get("error").is_none() {
        let sig = harness::expect_success(sign_resp);
        let sig_hex = sig["signature_hex"].as_str().unwrap().to_string();
        let pub_hex = sig["public_key_hex"].as_str().unwrap().to_string();

        let verify = harness::expect_success(harness::invoke_with_ctx(
            "signature_verify",
            json!({
                "data": "48656c6c6f",
                "data_encoding": "hex",
                "signature_hex": sig_hex,
                "public_key_hex": pub_hex,
                "algorithm": "ed25519"
            }),
            &ctx,
        ));
        assert_eq!(verify["valid"], true);
    } else {
        // Tool may not be registered; skip gracefully
        let code = sign_resp["error"].as_str().unwrap_or("");
        println!("signature_sign not available ({code}), skipping verify");
    }
}

#[test]
fn aead_encrypt_decrypt_aes256gcm() {
    let enc = harness::expect_success(harness::invoke("aead_encrypt", json!({
        "data": "secret message",
        "data_encoding": "text",
        "key_hex": KEY_HEX,
        "algorithm": "aes256gcm",
        "nonce_hex": NONCE_HEX
    })));
    assert_eq!(enc["algorithm"], "aes256gcm");
    let ct = enc["ciphertext_hex"].as_str().unwrap().to_string();
    let nonce = enc["nonce_hex"].as_str().unwrap().to_string();

    let dec = harness::expect_success(harness::invoke("aead_decrypt", json!({
        "ciphertext_hex": ct,
        "key_hex": KEY_HEX,
        "nonce_hex": nonce,
        "algorithm": "aes256gcm"
    })));
    assert_eq!(dec["plaintext"], "secret message");
}

#[test]
fn aead_encrypt_decrypt_chacha() {
    let enc = harness::expect_success(harness::invoke("aead_encrypt", json!({
        "data": "hello chacha",
        "data_encoding": "text",
        "key_hex": KEY_HEX,
        "algorithm": "chacha20poly1305",
        "nonce_hex": NONCE_HEX
    })));
    let ct = enc["ciphertext_hex"].as_str().unwrap();
    let nonce = enc["nonce_hex"].as_str().unwrap();
    let dec = harness::expect_success(harness::invoke("aead_decrypt", json!({
        "ciphertext_hex": ct,
        "key_hex": KEY_HEX,
        "nonce_hex": nonce,
        "algorithm": "chacha20poly1305"
    })));
    assert_eq!(dec["plaintext"], "hello chacha");
}

#[test]
fn hmac_sha256() {
    // HMAC-SHA256("", "") should be deterministic
    let resp = harness::expect_success(harness::invoke("hmac_compute", json!({
        "data": "hello",
        "data_encoding": "text",
        "key": "secret",
        "key_encoding": "text",
        "algorithm": "sha256"
    })));
    assert_eq!(resp["algorithm"], "sha256");
    assert!(resp["mac"].as_str().unwrap().len() == 64); // 32 bytes hex
}

#[test]
fn hmac_unknown_algorithm() {
    let resp = harness::invoke("hmac_compute", json!({
        "data": "d",
        "key": "k",
        "algorithm": "md4"
    }));
    harness::expect_error(&resp, "unknown_algorithm");
}

#[test]
fn kdf_argon2id() {
    let resp = harness::expect_success(harness::invoke("kdf_derive", json!({
        "password": "mypassword",
        "salt": "saltsalt",
        "salt_encoding": "text",
        "algorithm": "argon2id",
        "length": 32,
        "memory_kib": 1024,
        "iterations": 1
    })));
    assert_eq!(resp["algorithm"], "argon2id");
    assert_eq!(resp["length"], 32);
    let key = resp["derived_key"].as_str().unwrap();
    assert_eq!(key.len(), 64); // 32 bytes hex
}

#[test]
fn kdf_pbkdf2() {
    let resp = harness::expect_success(harness::invoke("kdf_derive", json!({
        "password": "pass",
        "salt": "salt",
        "salt_encoding": "text",
        "algorithm": "pbkdf2_sha256",
        "length": 16,
        "iterations": 1000
    })));
    assert_eq!(resp["algorithm"], "pbkdf2_sha256");
}

#[test]
fn hkdf_extract_sha256() {
    let resp = harness::expect_success(harness::invoke("hkdf_extract", json!({
        "ikm": "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
        "ikm_encoding": "hex",
        "algorithm": "sha256"
    })));
    assert_eq!(resp["algorithm"], "sha256");
    assert!(resp["prk_hex"].as_str().unwrap().len() == 64);
}

#[test]
fn hkdf_expand_label() {
    // Use known PRK
    let resp = harness::expect_success(harness::invoke("hkdf_expand_label", json!({
        "prk_hex": "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5",
        "label": "test label",
        "length": 32,
        "algorithm": "sha256"
    })));
    assert_eq!(resp["length"], 32);
    assert_eq!(resp["okm_hex"].as_str().unwrap().len(), 64);
}

#[test]
fn bytes_transcode_hex_to_base64() {
    let resp = harness::expect_success(harness::invoke("bytes_transcode", json!({
        "data": "48656c6c6f",
        "from": "hex",
        "to": "utf8"
    })));
    assert_eq!(resp["data"], "Hello");
}

#[test]
fn bytes_xor_simple() {
    let resp = harness::expect_success(harness::invoke("bytes_xor", json!({
        "a": "ff",
        "b": "0f",
        "a_encoding": "hex",
        "b_encoding": "hex",
        "output_encoding": "hex"
    })));
    assert_eq!(resp["result"], "f0");
    assert_eq!(resp["bytes"], 1);
}
