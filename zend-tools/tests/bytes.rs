mod harness;

use serde_json::json;

#[test]
fn bytes_transcode_base64url() {
    // base64url of "Hello" = "SGVsbG8" (no padding, URL-safe)
    let resp = harness::expect_success(harness::invoke(
        "bytes_transcode",
        json!({"data": "Hello", "from": "utf8", "to": "base64url"}),
    ));
    let out = resp["data"].as_str().unwrap();
    // base64url doesn't use + or / and has no = padding
    assert!(!out.contains('+'));
    assert!(!out.contains('/'));
    assert!(!out.contains('='));
}

#[test]
fn bytes_transcode_invalid_hex() {
    let resp = harness::invoke(
        "bytes_transcode",
        json!({"data": "ZZZZ", "from": "hex", "to": "utf8"}),
    );
    harness::expect_error(&resp, "decode_failed");
}

#[test]
fn bytes_pack_u32() {
    let resp = harness::expect_success(harness::invoke(
        "bytes_pack",
        json!({"values": [0xdeadbeef_u64], "format": ">I", "output_encoding": "hex"}),
    ));
    assert_eq!(resp["data"].as_str().unwrap(), "deadbeef");
    assert_eq!(resp["bytes_packed"], 4);
}

#[test]
fn bytes_pack_float() {
    let resp = harness::expect_success(harness::invoke(
        "bytes_pack",
        json!({"values": [1.0], "format": ">f", "output_encoding": "hex"}),
    ));
    // IEEE 754 big-endian 1.0f32 = 3f800000
    assert_eq!(resp["data"].as_str().unwrap(), "3f800000");
}

#[test]
fn bytes_xor_same_value() {
    let resp = harness::expect_success(harness::invoke(
        "bytes_xor",
        json!({"a": "aabbcc", "b": "aabbcc", "output_encoding": "hex"}),
    ));
    assert_eq!(resp["result"].as_str().unwrap(), "000000");
}

#[test]
fn bytes_unpack_little_endian_u32() {
    // Little-endian 0x01000000 = 1 in u32 LE
    let resp = harness::expect_success(harness::invoke(
        "bytes_unpack",
        json!({"data": "01000000", "data_encoding": "hex", "format": "<I"}),
    ));
    let values = resp["values"].as_array().unwrap();
    assert_eq!(values[0].as_u64().unwrap(), 1u64);
}

#[test]
fn bytes_transcode_hex_to_utf8() {
    let resp = harness::expect_success(harness::invoke("bytes_transcode", json!({
        "data": "48656c6c6f",
        "from": "hex",
        "to": "utf8"
    })));
    assert_eq!(resp["data"], "Hello");
    assert_eq!(resp["bytes"], 5);
}

#[test]
fn bytes_transcode_utf8_to_base64() {
    let resp = harness::expect_success(harness::invoke("bytes_transcode", json!({
        "data": "Hello",
        "from": "utf8",
        "to": "base64"
    })));
    assert_eq!(resp["data"], "SGVsbG8=");
}

#[test]
fn bytes_transcode_invalid_encoding() {
    let resp = harness::invoke("bytes_transcode", json!({
        "data": "abc",
        "from": "binary",
        "to": "hex"
    }));
    harness::expect_error(&resp, "invalid_encoding");
}

#[test]
fn bytes_pack_unpack_big_endian() {
    let pack = harness::expect_success(harness::invoke("bytes_pack", json!({
        "values": [1, 2],
        "format": ">HH",
        "output_encoding": "hex"
    })));
    let hex = pack["data"].as_str().unwrap();
    assert_eq!(hex, "00010002");

    let unpack = harness::expect_success(harness::invoke("bytes_unpack", json!({
        "data": hex,
        "data_encoding": "hex",
        "format": ">HH"
    })));
    let values = unpack["values"].as_array().unwrap();
    assert_eq!(values[0], 1);
    assert_eq!(values[1], 2);
}

#[test]
fn bytes_pack_little_endian() {
    let pack = harness::expect_success(harness::invoke("bytes_pack", json!({
        "values": [256],
        "format": "<H",
        "output_encoding": "hex"
    })));
    // 256 in LE = 0x0100
    assert_eq!(pack["data"].as_str().unwrap(), "0001");
}

#[test]
fn bytes_xor_different_lengths() {
    let resp = harness::expect_success(harness::invoke("bytes_xor", json!({
        "a": "ffff",
        "b": "ff",
        "output_encoding": "hex"
    })));
    // ff ^ ff = 00, ff ^ 00 = ff
    assert_eq!(resp["result"].as_str().unwrap(), "00ff");
    assert_eq!(resp["bytes"], 2);
}

#[test]
fn bytes_xor_all_zeros() {
    let resp = harness::expect_success(harness::invoke("bytes_xor", json!({
        "a": "aabbcc",
        "b": "aabbcc",
        "output_encoding": "hex"
    })));
    assert_eq!(resp["result"].as_str().unwrap(), "000000");
}
