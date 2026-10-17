// Names no Rust source in this workspace may use, checked by each CUDA crate's
// build script over its own sources before anything compiles.
//
// Every one reaches the CUDA legacy null stream. A device here has one created
// stream that every thread issues on and a wave records on
// (`candle-core/src/cuda_backend/stream_lease.rs`): work on the null stream is
// neither recorded into a wave nor ordered against the device's stream, so it
// races everything else on the device. The kernel sources are held to the same
// rule by `candle-kernels/build_lints.rs`.
//
// Included with `include!` by the build scripts, and by candle-core's tests,
// which exercise the matching below.

/// Each banned name, and what to use instead.
const NULL_STREAM_NAMES: [(&str, &str); 15] = [
    (
        "default_stream",
        "the legacy null stream: take the device's stream with `CudaDevice::cuda_stream`",
    ),
    (
        "memcpy_htod_sync",
        "a copy ordered on the null stream: `memcpy_htod_async` on the device's stream",
    ),
    (
        "memcpy_dtoh_sync",
        "a copy ordered on the null stream: `memcpy_dtoh_async` on the device's stream, then synchronise it",
    ),
    (
        "memcpy_dtod_sync",
        "a copy ordered on the null stream: `memcpy_dtod_async` on the device's stream",
    ),
    (
        "memset_d8_sync",
        "a fill ordered on the null stream: `memset_d8_async` on the device's stream",
    ),
    (
        "memset_d32_sync",
        "a fill ordered on the null stream: `memset_d32_async` on the device's stream",
    ),
    (
        "cuMemcpyHtoD_v2",
        "a copy ordered on the null stream: `cuMemcpyHtoDAsync_v2` on the device's stream",
    ),
    (
        "cuMemcpyDtoH_v2",
        "a copy ordered on the null stream: `cuMemcpyDtoHAsync_v2` on the device's stream, then synchronise it",
    ),
    (
        "cuMemcpyDtoD_v2",
        "a copy ordered on the null stream: `cuMemcpyDtoDAsync_v2` on the device's stream",
    ),
    (
        "cuMemsetD8_v2",
        "a fill ordered on the null stream: `cuMemsetD8Async` on the device's stream",
    ),
    (
        "cuMemsetD32_v2",
        "a fill ordered on the null stream: `cuMemsetD32Async` on the device's stream",
    ),
    (
        "cuMemcpy2D_v2",
        "a copy ordered on the null stream: `cuMemcpy2DAsync_v2` on the device's stream",
    ),
    (
        "cuMemcpy3D_v2",
        "a copy ordered on the null stream: `cuMemcpy3DAsync_v2` on the device's stream",
    ),
    (
        "CU_STREAM_LEGACY",
        "the legacy null stream by handle: take the device's stream with `CudaDevice::cuda_stream`",
    ),
    (
        "CU_STREAM_PER_THREAD",
        "a per-thread default stream, unordered against the device's stream",
    ),
];

/// What a null pointer passed as a launcher's last argument is reported as. A
/// kernel launcher here is named `run_*` or `launch_*` and takes its stream
/// last, and a null one is the legacy null stream — a call no name above
/// catches.
const TRAILING_NULL: &str = "null pointer as a launcher's last argument";
const TRAILING_NULL_FIX: &str =
    "a launcher's stream position: pass the device's stream, `stream.cu_stream()`";

/// `(line number, name)` for every banned name used as code in `src`, and every
/// null pointer passed as a call's last argument — comments, string literals and
/// character literals are not code.
fn null_stream_uses_in(src: &str) -> Vec<(usize, &'static str)> {
    let code = rust_code_only(src);
    let mut out = Vec::new();
    for (i, line) in code.lines().enumerate() {
        for (name, _) in NULL_STREAM_NAMES {
            if names(line, name) {
                out.push((i + 1, name));
            }
        }
    }
    out.extend(trailing_null_lines(&code).into_iter().map(|l| (l, TRAILING_NULL)));
    out.sort_by_key(|&(line, _)| line);
    out
}

/// Whether the call whose argument list encloses byte `at` of `code` is to a
/// function named `run_*` or `launch_*`.
fn inside_launcher_call(b: &[u8], at: usize) -> bool {
    let mut depth = 0usize;
    let mut i = at;
    while i > 0 {
        i -= 1;
        match b[i] {
            b')' | b']' | b'}' => depth += 1,
            b'(' | b'[' | b'{' if depth > 0 => depth -= 1,
            b'(' => {
                let mut end = i;
                while end > 0 && b[end - 1].is_ascii_whitespace() {
                    end -= 1;
                }
                let mut start = end;
                while start > 0 && (b[start - 1].is_ascii_alphanumeric() || b[start - 1] == b'_') {
                    start -= 1;
                }
                let callee = &b[start..end];
                return callee.starts_with(b"run_") || callee.starts_with(b"launch_");
            }
            b'[' | b'{' => return false,
            _ => {}
        }
    }
    false
}

/// The line of every `null()` / `null_mut()` in `code` that is the last
/// argument of a launcher call: `)`, an optional `,` and then the call's `)`,
/// across lines.
fn trailing_null_lines(code: &str) -> Vec<usize> {
    let b = code.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let skip_ws = |mut j: usize| {
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        j
    };
    let mut out = Vec::new();
    for pat in ["null_mut", "null"] {
        let mut from = 0;
        while let Some(at) = code[from..].find(pat) {
            let start = from + at;
            let end = start + pat.len();
            from = end;
            if ident_at(b, start) || b.get(end).copied().is_some_and(ident) {
                continue;
            }
            let mut j = skip_ws(end);
            if b.get(j) != Some(&b'(') {
                continue;
            }
            j = skip_ws(j + 1);
            if b.get(j) != Some(&b')') {
                continue;
            }
            j = skip_ws(j + 1);
            if b.get(j) == Some(&b',') {
                j = skip_ws(j + 1);
            }
            if b.get(j) == Some(&b')') && inside_launcher_call(b, start) {
                out.push(b[..start].iter().filter(|&&c| c == b'\n').count() + 1);
            }
        }
    }
    out
}

/// `src` with every comment, string literal and character literal blanked to
/// spaces, line breaks kept so line numbers still match.
fn rust_code_only(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let blank = |out: &mut Vec<u8>, c: u8| out.push(if c == b'\n' { b'\n' } else { b' ' });
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();
        if c == b'/' && next == Some(b'/') {
            while i < b.len() && b[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
        } else if c == b'/' && next == Some(b'*') {
            // Block comments nest in Rust.
            let mut depth = 0usize;
            while i < b.len() {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    out.extend_from_slice(b"  ");
                    i += 2;
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    out.extend_from_slice(b"  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    blank(&mut out, b[i]);
                    i += 1;
                }
            }
        } else if c == b'r' && matches!(next, Some(b'"') | Some(b'#')) && !ident_at(b, i) {
            // A raw string: `r"…"` or `r#"…"#`, with as many `#` closing it.
            let mut j = i + 1;
            let mut hashes = 0usize;
            while b.get(j) == Some(&b'#') {
                hashes += 1;
                j += 1;
            }
            if b.get(j) != Some(&b'"') {
                out.push(c);
                i += 1;
                continue;
            }
            out.resize(out.len() + (j + 1 - i), b' ');
            i = j + 1;
            while i < b.len() {
                if b[i] == b'"' && (1..=hashes).all(|h| b.get(i + h) == Some(&b'#')) {
                    out.resize(out.len() + 1 + hashes, b' ');
                    i += 1 + hashes;
                    break;
                }
                blank(&mut out, b[i]);
                i += 1;
            }
        } else if c == b'"' {
            out.push(b' ');
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < b.len() {
                    blank(&mut out, b[i]);
                    i += 1;
                }
                blank(&mut out, b[i]);
                i += 1;
            }
            if i < b.len() {
                out.push(b' ');
                i += 1;
            }
        } else if c == b'\'' && char_literal_len(b, i).is_some() {
            let len = char_literal_len(b, i).expect("checked");
            out.resize(out.len() + len, b' ');
            i += len;
        } else {
            out.push(c);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Whether the byte before `i` continues an identifier — `r` in `bar"` is not a
/// raw-string prefix.
fn ident_at(b: &[u8], i: usize) -> bool {
    i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_')
}

/// The length of the character literal starting at `i`, or `None` when the `'`
/// there opens a lifetime or a label instead.
fn char_literal_len(b: &[u8], i: usize) -> Option<usize> {
    match b.get(i + 1)? {
        b'\\' => {
            // Past the escaped character, which may itself be a `'`.
            let mut j = i + 3;
            while j < b.len() && b[j] != b'\'' && j - i < 12 {
                j += 1;
            }
            (b.get(j) == Some(&b'\'')).then_some(j + 1 - i)
        }
        _ => {
            // One character, possibly several bytes of UTF-8, then `'`.
            let mut j = i + 2;
            while j < b.len() && (b[j] & 0xC0) == 0x80 {
                j += 1;
            }
            (b.get(j) == Some(&b'\'')).then_some(j + 1 - i)
        }
    }
}

/// Whether `line` uses `name` as a whole identifier.
fn names(line: &str, name: &str) -> bool {
    let bytes = line.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut from = 0;
    while let Some(at) = line[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before_ok = start == 0 || !ident(bytes[start - 1]);
        let after_ok = end == bytes.len() || !ident(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// Refuse the build when any `.rs` file under `dirs` uses a banned name, and
/// have cargo run the check again whenever one of them changes.
fn forbid_null_stream(dirs: &[&std::path::Path]) {
    let mut found = Vec::new();
    for dir in dirs {
        if dir.exists() {
            println!("cargo:rerun-if-changed={}", dir.display());
            scan_for_null_stream(dir, &mut found);
        }
    }
    if found.is_empty() {
        return;
    }
    let mut msg = String::from(
        "sources reach the CUDA legacy null stream, which is neither recorded into a wave nor \
         ordered against the device's stream (see lints/null_stream.rs):\n",
    );
    for (path, line, name) in &found {
        let fix = NULL_STREAM_NAMES
            .iter()
            .chain([(TRAILING_NULL, TRAILING_NULL_FIX)].iter())
            .find(|(n, _)| n == name)
            .map_or("", |(_, fix)| *fix);
        msg.push_str(&format!("  {}:{line}: {name} — {fix}\n", path.display()));
    }
    panic!("{msg}");
}

fn scan_for_null_stream(
    dir: &std::path::Path,
    found: &mut Vec<(std::path::PathBuf, usize, &'static str)>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_for_null_stream(&path, found);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (line, name) in null_stream_uses_in(&src) {
                found.push((path.clone(), line, name));
            }
        }
    }
}

#[cfg(test)]
mod null_stream_lint_tests {
    use super::*;

    #[test]
    fn a_null_stream_use_is_found_with_its_line() {
        let src = "fn f(ctx: &Ctx) {\n    let s = ctx.default_stream();\n}\n";
        assert_eq!(null_stream_uses_in(src), vec![(2, "default_stream")]);
        assert_eq!(
            null_stream_uses_in("unsafe { memset_d8_sync(p, 0, n) }"),
            vec![(1, "memset_d8_sync")]
        );
    }

    #[test]
    fn a_name_is_matched_whole() {
        assert!(null_stream_uses_in("memcpy_htod_async(p, h, s);").is_empty());
        assert!(null_stream_uses_in("my_default_stream_rule();").is_empty());
        assert!(null_stream_uses_in("cuMemcpyHtoDAsync_v2(d, h, n, s)").is_empty());
        assert_eq!(
            null_stream_uses_in("sys::cuMemcpyHtoD_v2(d, h, n)"),
            vec![(1, "cuMemcpyHtoD_v2")]
        );
    }

    #[test]
    fn comments_and_strings_are_not_code() {
        let src = "// was ctx.default_stream()\n\
                   let m = \"no default_stream here\";\n\
                   /* a /* nested */ memset_d8_sync( */ let x = 1;\n\
                   let r = r#\"default_stream \"quoted\"\"#;\n\
                   let c = '\"'; let q = '\\''; let l: &'static str = \"\";\n";
        assert!(null_stream_uses_in(src).is_empty(), "{:?}", null_stream_uses_in(src));
    }

    #[test]
    fn code_after_a_comment_or_string_is_still_read() {
        let src = "let s = \"x\"; let t = ctx.default_stream(); /* c */ memset_d32_sync(p, 0, n);";
        assert_eq!(
            null_stream_uses_in(src),
            vec![(1, "default_stream"), (1, "memset_d32_sync")]
        );
        let lifetime = "fn f<'a>(s: &'a Ctx) { s.default_stream() }";
        assert_eq!(null_stream_uses_in(lifetime), vec![(1, "default_stream")]);
    }

    #[test]
    fn a_null_last_argument_is_found_across_lines() {
        let src = "unsafe {\n    run_sort(\n        n,\n        std::ptr::null_mut(), // stream\n    );\n}\n";
        assert_eq!(null_stream_uses_in(src), vec![(4, TRAILING_NULL)]);
        assert_eq!(
            null_stream_uses_in("launch_gemm(a, f(b), ptr::null())"),
            vec![(1, TRAILING_NULL)]
        );
    }

    #[test]
    fn a_null_last_argument_of_another_call_is_not_found() {
        assert!(null_stream_uses_in("x.unwrap_or(std::ptr::null())").is_empty());
        assert!(null_stream_uses_in("CreateEventW(std::ptr::null(), 1, 0, std::ptr::null())").is_empty());
        assert!(null_stream_uses_in("AtomicPtr::new(std::ptr::null_mut())").is_empty());
        assert!(null_stream_uses_in("run_x(a, [b, ptr::null()])").is_empty());
    }

    #[test]
    fn a_null_that_is_not_the_last_argument_is_not_found() {
        assert!(null_stream_uses_in("run(ptr::null_mut(), n, stream)").is_empty());
        assert!(null_stream_uses_in("let p: *mut u8 = std::ptr::null_mut();").is_empty());
        assert!(null_stream_uses_in("Self { base: std::ptr::null_mut() }").is_empty());
        assert!(null_stream_uses_in("is_null(p)").is_empty());
        assert!(null_stream_uses_in("nullable(x)").is_empty());
    }
}
