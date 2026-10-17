// Calls no kernel source may make, checked over every .cu/.cuh before anything
// compiles.
//
// A device-wide synchronise — or a call the runtime makes one of — is refused
// while any stream of the context captures a graph, and the refusal invalidates
// that capture on whichever thread is recording. The engine records every wave,
// and other threads launch kernels beside it, so one of these inside a launcher
// is a wave lost the first time it runs beside one. `repack_gemx` ended with
// `cudaDeviceSynchronize`; `candle-core`'s graph tests failed whenever a repack
// ran beside them. Rust code is held to the same rule by `clippy.toml`; this is
// the same list for the code clippy cannot read.
//
// Kept apart from `build_utils.rs` because it is a rule rather than a utility,
// and because its matching is a pure function the tests below exercise.

/// What a launcher uses instead of a call that runs on the legacy null stream.
const ON_ITS_STREAM: &str =
    "runs on the legacy null stream: use the Async form, on the stream the launcher was given";

/// Each banned call, and what to do instead.
///
/// The second half are the synchronous copies and fills: each runs on the
/// legacy null stream. A device here has one created stream that every thread
/// issues on and a wave records on, and work on the null stream is neither
/// recorded nor ordered against it — the same rule `lints/null_stream.rs` holds
/// the Rust sources to.
const FORBIDDEN_CALLS: [(&str, &str); 25] = [
    ("cudaDeviceSynchronize", "synchronise the stream the work is on"),
    ("cudaThreadSynchronize", "synchronise the stream the work is on"),
    ("cuCtxSynchronize", "synchronise the stream the work is on"),
    ("cudaFree", "take the memory from the host's allocator, or free it stream-ordered (cudaFreeAsync)"),
    ("cudaFreeHost", "leave pinned memory to the host, which frees it outside any capture"),
    ("cudaFreeArray", "free it stream-ordered"),
    ("cudaDeviceReset", "never reset the device a daemon serves on"),
    ("cudaDeviceSetLimit", "set limits once, from the host, before the first capture"),
    ("cuModuleLoad", "load modules through the host's record gate"),
    ("cudaMemcpy", ON_ITS_STREAM),
    ("cudaMemcpy2D", ON_ITS_STREAM),
    ("cudaMemcpy3D", ON_ITS_STREAM),
    ("cudaMemcpyPeer", ON_ITS_STREAM),
    ("cudaMemcpyToSymbol", ON_ITS_STREAM),
    ("cudaMemcpyFromSymbol", ON_ITS_STREAM),
    ("cudaMemset", ON_ITS_STREAM),
    ("cudaMemset2D", ON_ITS_STREAM),
    ("cuMemcpy", ON_ITS_STREAM),
    ("cuMemcpyHtoD", ON_ITS_STREAM),
    ("cuMemcpyDtoH", ON_ITS_STREAM),
    ("cuMemcpyDtoD", ON_ITS_STREAM),
    ("cuMemcpy2D", ON_ITS_STREAM),
    ("cuMemsetD8", ON_ITS_STREAM),
    ("cuMemsetD32", ON_ITS_STREAM),
    ("cuMemsetD16", ON_ITS_STREAM),
];

/// Names of the null streams themselves, banned wherever they appear as code.
const NULL_STREAM_HANDLES: [&str; 4] = [
    "cudaStreamLegacy",
    "cudaStreamPerThread",
    "CU_STREAM_LEGACY",
    "CU_STREAM_PER_THREAD",
];

/// `src` with every comment blanked to spaces, line breaks kept.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut in_block = false;
    for raw in src.lines() {
        let mut rest = raw;
        loop {
            if in_block {
                match rest.find("*/") {
                    Some(end) => {
                        out.extend(std::iter::repeat_n(' ', end + 2));
                        rest = &rest[end + 2..];
                        in_block = false;
                    }
                    None => {
                        out.extend(std::iter::repeat_n(' ', rest.len()));
                        break;
                    }
                }
            } else {
                let block = rest.find("/*");
                let line_comment = rest.find("//");
                match (block, line_comment) {
                    (Some(b), Some(l)) if l < b => {
                        out.push_str(&rest[..l]);
                        break;
                    }
                    (Some(b), _) => {
                        out.push_str(&rest[..b]);
                        out.push_str("  ");
                        rest = &rest[b + 2..];
                        in_block = true;
                    }
                    (None, Some(l)) => {
                        out.push_str(&rest[..l]);
                        break;
                    }
                    (None, None) => {
                        out.push_str(rest);
                        break;
                    }
                }
            }
        }
        out.push('\n');
    }
    out
}

/// What a launch or a declaration does to reach the null stream, by kind — the
/// message each violation carries.
const LAUNCH_WITHOUT_STREAM: &str =
    "a `<<<grid, block>>>` launch with no stream runs on the legacy null stream: pass the launcher's stream as the fourth argument";
const LAUNCH_ON_NULL: &str =
    "a launch on stream 0 runs on the legacy null stream: pass the launcher's stream";
const DEFAULTED_STREAM: &str =
    "a stream parameter that defaults to the null stream lets a caller launch there by leaving it out: take it without a default";
const NULL_HANDLE: &str =
    "a null stream by name: launch on the stream the launcher was given";

/// `(line number, what)` for every way `src` reaches the null stream other than
/// a banned call: a launch configuration with no stream or stream 0, a stream
/// parameter defaulted to 0, and the null streams' own names. Comments ignored;
/// a launch configuration may span lines.
fn null_stream_uses_in(src: &str) -> Vec<(usize, &'static str)> {
    let code = strip_comments(src);
    let line_of = |at: usize| code[..at].matches('\n').count() + 1;
    let mut out = Vec::new();

    let mut from = 0;
    while let Some(open) = code[from..].find("<<<") {
        let start = from + open + 3;
        let Some(close) = code[start..].find(">>>") else {
            break;
        };
        let config = &code[start..start + close];
        let args = split_top_level(config);
        if args.len() < 4 {
            out.push((line_of(from + open), LAUNCH_WITHOUT_STREAM));
        } else if matches!(args[3].trim(), "0" | "nullptr" | "NULL" | "(cudaStream_t)0") {
            out.push((line_of(from + open), LAUNCH_ON_NULL));
        }
        from = start + close + 3;
    }

    for (i, line) in code.lines().enumerate() {
        for ty in ["cudaStream_t", "CUstream"] {
            if defaults_to_null(line, ty) {
                out.push((i + 1, DEFAULTED_STREAM));
            }
        }
        for handle in NULL_STREAM_HANDLES {
            if names(line, handle) {
                out.push((i + 1, NULL_HANDLE));
            }
        }
    }
    out.sort_by_key(|&(line, _)| line);
    out
}

/// `config`'s comma-separated arguments, commas inside brackets not splitting.
/// Angle brackets are not brackets here: a comparison inside a launch
/// configuration would unbalance them.
fn split_top_level(config: &str) -> Vec<&str> {
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in config.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                args.push(&config[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    args.push(&config[start..]);
    args
}

/// Whether `line` declares a `ty` parameter or variable defaulted to the null
/// stream: `ty name = 0`, `= nullptr` or `= NULL`.
fn defaults_to_null(line: &str, ty: &str) -> bool {
    let mut from = 0;
    while let Some(at) = line[from..].find(ty) {
        let start = from + at;
        let end = start + ty.len();
        from = end;
        let before_ok = start == 0 || !is_ident(line.as_bytes()[start - 1]);
        if !before_ok {
            continue;
        }
        let rest = line[end..].trim_start();
        let name_len = rest.bytes().take_while(|&c| is_ident(c)).count();
        if name_len == 0 {
            continue;
        }
        let after = rest[name_len..].trim_start();
        let Some(value) = after.strip_prefix('=') else {
            continue;
        };
        let value = value.trim_start();
        let token: String = value.chars().take_while(|&c| c.is_ascii_alphanumeric() || c == '_').collect();
        if matches!(token.as_str(), "0" | "nullptr" | "NULL") {
            return true;
        }
    }
    false
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Whether `line` uses `name` as a whole identifier.
fn names(line: &str, name: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(at) = line[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before_ok = start == 0 || !is_ident(bytes[start - 1]);
        let after_ok = end == bytes.len() || !is_ident(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// `(line number, call)` for every forbidden call in `src`, comments ignored.
fn forbidden_calls_in(src: &str) -> Vec<(usize, &'static str)> {
    let mut out = Vec::new();
    let mut in_block = false;
    for (i, raw) in src.lines().enumerate() {
        let mut line = String::with_capacity(raw.len());
        let mut rest = raw;
        loop {
            if in_block {
                match rest.find("*/") {
                    Some(end) => {
                        rest = &rest[end + 2..];
                        in_block = false;
                    }
                    None => break,
                }
            } else {
                let block = rest.find("/*");
                let line_comment = rest.find("//");
                match (block, line_comment) {
                    (Some(b), Some(l)) if l < b => {
                        line.push_str(&rest[..l]);
                        break;
                    }
                    (Some(b), _) => {
                        line.push_str(&rest[..b]);
                        rest = &rest[b + 2..];
                        in_block = true;
                    }
                    (None, Some(l)) => {
                        line.push_str(&rest[..l]);
                        break;
                    }
                    (None, None) => {
                        line.push_str(rest);
                        break;
                    }
                }
            }
        }
        for (call, _) in FORBIDDEN_CALLS {
            if calls(&line, call) {
                out.push((i + 1, call));
            }
        }
    }
    out
}

/// Whether `line` calls `name` — the identifier, whole, followed by `(`.
fn calls(line: &str, name: &str) -> bool {
    let bytes = line.as_bytes();
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut from = 0;
    while let Some(at) = line[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before_ok = start == 0 || !ident(bytes[start - 1]);
        let after = line[end..].trim_start();
        if before_ok && after.starts_with('(') {
            return true;
        }
        from = end;
    }
    false
}

/// Refuse the build if any kernel source under `dir` makes a forbidden call or
/// reaches the legacy null stream.
fn forbid_device_wide_calls(dir: &Path) -> Result<()> {
    let mut found = Vec::new();
    scan_for_forbidden_calls(dir, &mut found)?;
    if found.is_empty() {
        return Ok(());
    }
    let mut msg = String::from(
        "kernel sources make calls a graph capture on any thread refuses, or reach the legacy \
         null stream — each would invalidate a recording wave or run unordered against the \
         device's stream (see build_lints.rs):\n",
    );
    for (path, line, what) in &found {
        msg.push_str(&format!("  {}:{line}: {what}\n", path.display()));
    }
    anyhow::bail!(msg)
}

fn scan_for_forbidden_calls(dir: &Path, found: &mut Vec<(PathBuf, usize, String)>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            scan_for_forbidden_calls(&path, found)?;
        } else if path
            .extension()
            .is_some_and(|ext| ext == "cu" || ext == "cuh")
        {
            let src = fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            for (line, call) in forbidden_calls_in(&src) {
                let fix = FORBIDDEN_CALLS
                    .iter()
                    .find(|(c, _)| *c == call)
                    .map_or("", |(_, fix)| *fix);
                found.push((path.clone(), line, format!("{call} — {fix}")));
            }
            for (line, what) in null_stream_uses_in(&src) {
                found.push((path.clone(), line, what.to_string()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod build_lint_tests {
    use super::*;

    #[test]
    fn a_device_synchronise_is_found_with_its_line() {
        let src = "int f() {\n  k<<<1,1>>>();\n  cudaDeviceSynchronize();\n}\n";
        assert_eq!(forbidden_calls_in(src), vec![(3, "cudaDeviceSynchronize")]);
    }

    #[test]
    fn a_stream_ordered_free_is_not_a_free() {
        assert!(forbidden_calls_in("cudaFreeAsync(p, s);").is_empty());
        assert_eq!(forbidden_calls_in("cudaFree (p);"), vec![(1, "cudaFree")]);
        assert!(forbidden_calls_in("my_cudaFree(p);").is_empty());
    }

    #[test]
    fn comments_are_not_calls() {
        let src = "// ends with cudaDeviceSynchronize()\n\
                   x = 1; /* cudaFree(p) */ y = 2;\n\
                   /* a block\n   cuCtxSynchronize()\n*/ z = 3;\n";
        assert!(forbidden_calls_in(src).is_empty());
    }

    #[test]
    fn code_after_a_block_comment_is_still_read() {
        assert_eq!(
            forbidden_calls_in("/* note */ cudaDeviceReset();"),
            vec![(1, "cudaDeviceReset")]
        );
    }

    #[test]
    fn a_synchronous_copy_is_banned_and_its_async_form_is_not() {
        assert_eq!(
            forbidden_calls_in("cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);"),
            vec![(1, "cudaMemcpy")]
        );
        assert!(forbidden_calls_in("cudaMemcpyAsync(d, h, n, kind, stream);").is_empty());
        assert_eq!(
            forbidden_calls_in("cudaMemcpyToSymbol(lut, src, bytes);"),
            vec![(1, "cudaMemcpyToSymbol")]
        );
        assert!(forbidden_calls_in("cudaMemcpyToSymbolAsync(lut, src, b, 0, k, s);").is_empty());
    }

    #[test]
    fn a_launch_with_no_stream_is_found_with_its_line() {
        let src = "void f(cudaStream_t s) {\n  k<<<grid, block>>>(a);\n  k<<<g, b, smem>>>(a);\n}\n";
        assert_eq!(
            null_stream_uses_in(src),
            vec![(2, LAUNCH_WITHOUT_STREAM), (3, LAUNCH_WITHOUT_STREAM)]
        );
    }

    #[test]
    fn a_launch_on_its_stream_passes_and_one_on_zero_does_not() {
        assert!(null_stream_uses_in("k<<<g, b, 0, stream>>>(a);").is_empty());
        assert!(null_stream_uses_in("k<<<dim3(x, y), b, sizeof(T) * n, s>>>(a);").is_empty());
        assert_eq!(null_stream_uses_in("k<<<g, b, 0, 0>>>(a);"), vec![(1, LAUNCH_ON_NULL)]);
        assert_eq!(
            null_stream_uses_in("k<<<g, b, 0, nullptr>>>(a);"),
            vec![(1, LAUNCH_ON_NULL)]
        );
    }

    #[test]
    fn a_launch_configuration_may_span_lines() {
        let src = "k<<<grid,\n    block,\n    smem>>>(a);\nj<<<g,\n  b, 0,\n  s>>>(a);\n";
        assert_eq!(null_stream_uses_in(src), vec![(1, LAUNCH_WITHOUT_STREAM)]);
    }

    #[test]
    fn a_stream_parameter_defaulted_to_null_is_found() {
        assert_eq!(
            null_stream_uses_in("void run(int n, cudaStream_t stream = 0) {}"),
            vec![(1, DEFAULTED_STREAM)]
        );
        assert_eq!(
            null_stream_uses_in("void run(cudaStream_t s = nullptr);"),
            vec![(1, DEFAULTED_STREAM)]
        );
        assert!(null_stream_uses_in("void run(cudaStream_t stream) {}").is_empty());
        assert!(null_stream_uses_in("cudaStream_t s = caller;").is_empty());
    }

    #[test]
    fn the_null_streams_names_are_found() {
        assert_eq!(
            null_stream_uses_in("cudaMemsetAsync(p, 0, n, cudaStreamLegacy);"),
            vec![(1, NULL_HANDLE)]
        );
        assert!(null_stream_uses_in("// not cudaStreamPerThread here").is_empty());
    }
}
