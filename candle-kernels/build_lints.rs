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

/// Each banned call, and what to do instead.
const FORBIDDEN_CALLS: [(&str, &str); 9] = [
    ("cudaDeviceSynchronize", "synchronise the stream the work is on"),
    ("cudaThreadSynchronize", "synchronise the stream the work is on"),
    ("cuCtxSynchronize", "synchronise the stream the work is on"),
    ("cudaFree", "take the memory from the host's allocator, or free it stream-ordered (cudaFreeAsync)"),
    ("cudaFreeHost", "leave pinned memory to the host, which frees it outside any capture"),
    ("cudaFreeArray", "free it stream-ordered"),
    ("cudaDeviceReset", "never reset the device a daemon serves on"),
    ("cudaDeviceSetLimit", "set limits once, from the host, before the first capture"),
    ("cuModuleLoad", "load modules through the host's record gate"),
];

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

/// Refuse the build if any kernel source under `dir` makes a forbidden call.
fn forbid_device_wide_calls(dir: &Path) -> Result<()> {
    let mut found = Vec::new();
    scan_for_forbidden_calls(dir, &mut found)?;
    if found.is_empty() {
        return Ok(());
    }
    let mut msg = String::from(
        "kernel sources make calls a graph capture on any thread refuses — each would \
         invalidate a recording wave (see build_lints.rs):\n",
    );
    for (path, line, call) in &found {
        let fix = FORBIDDEN_CALLS
            .iter()
            .find(|(c, _)| c == call)
            .map_or("", |(_, fix)| *fix);
        msg.push_str(&format!("  {}:{line}: {call} — {fix}\n", path.display()));
    }
    anyhow::bail!(msg)
}

fn scan_for_forbidden_calls(
    dir: &Path,
    found: &mut Vec<(PathBuf, usize, &'static str)>,
) -> Result<()> {
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
                found.push((path.clone(), line, call));
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
}
