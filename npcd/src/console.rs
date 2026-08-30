//! The console's two backends, held to the contract they claim.
//!
//! `web/content/npcd/lib/api.js` picks between `api.live.js` and `api.mock.js`
//! at load time and says of them: *"Both satisfy the identical method
//! contract."* Nothing enforced that. A page written against the live client
//! and never opened with `?mock=1` breaks the mock silently — not at build, not
//! in any test, but the first time somebody demonstrates the console without a
//! daemon, as `API.mindList is not a function` in a console log nobody has
//! open.
//!
//! The mind pages were exactly that: five methods on the live client and none
//! on the mock. So the claim is a test now.
//!
//! It reads the two files as text rather than running them, because they are ES
//! modules and this is a Rust binary. That is enough for the failure it exists
//! to catch, which is a *missing name* — the mock's fixtures are already free
//! to answer with whatever they like.

/// Every method name an API object exposes, from the source of one of the two
/// client modules.
///
/// Both files are a single `export const X = { ... }` of methods, in two
/// spellings: `name: (args) => …` and `async name(args) { … }`. A line whose
/// first token is an identifier followed by one of those is a method; anything
/// indented deeper belongs to a method body and is not.
///
/// The scan starts at the `export` and ends at the object's closing brace.
/// Bounding it that way is what keeps a `for (…)` inside a module-level helper
/// out of the list — a two-space indent means "member of this object" only
/// while you are inside the object.
#[cfg(test)]
fn methods(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in source.lines() {
        if !inside {
            inside = line.starts_with("export const ") && line.trim_end().ends_with('{');
            continue;
        }
        if line.starts_with("};") {
            break;
        }
        // Two spaces exactly: the members of the exported object, and nothing
        // nested inside one of them.
        let Some(rest) = line.strip_prefix("  ") else {
            continue;
        };
        if rest.starts_with(' ') || rest.starts_with("//") || rest.starts_with('*') {
            continue;
        }
        let rest = rest.strip_prefix("async ").unwrap_or(rest);
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let after = rest[name.len()..].trim_start();
        // `name(` is a method; `name:` is a property, which is a method here
        // only when an arrow or a function follows.
        let is_method = after.starts_with('(')
            || after
                .strip_prefix(':')
                .map(|v| {
                    let v = v.trim_start();
                    v.starts_with('(') || v.starts_with("async") || v.starts_with("function")
                })
                .unwrap_or(false);
        if is_method {
            out.push(name);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Every `{}`, `()` and `[]` in a JavaScript source, outside its strings,
/// template literals, comments and regular expressions.
///
/// Returns the first place the nesting goes wrong: a closer with no opener, a
/// mismatched pair, or the opener left dangling at the end.
///
/// There is no JavaScript toolchain in this repository — the console is
/// hand-written ES modules served straight off disk, with no build step and no
/// linter — so a missing brace reaches the browser, and the page it is on
/// simply stops working. This is the cheapest check that catches it, and it
/// runs with `cargo test` like everything else.
#[cfg(test)]
fn unbalanced(source: &str) -> Option<String> {
    #[derive(Clone, Copy)]
    enum In {
        Code,
        Line,
        Block,
        Str(char),
        Template,
        Regex,
    }
    let bytes: Vec<char> = source.chars().collect();
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut state = In::Code;
    let mut line = 1usize;
    // What the previous non-space code character was, which is the only way to
    // tell a regular expression from a division.
    let mut prev = '\0';
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == '\n' {
            line += 1;
        }
        match state {
            In::Line => {
                if c == '\n' {
                    state = In::Code;
                }
            }
            In::Block => {
                if c == '*' && bytes.get(i + 1) == Some(&'/') {
                    state = In::Code;
                    i += 1;
                }
            }
            In::Str(quote) => match c {
                '\\' => i += 1,
                q if q == quote => state = In::Code,
                _ => {}
            },
            In::Template => match c {
                '\\' => i += 1,
                '`' => state = In::Code,
                _ => {}
            },
            In::Regex => match c {
                '\\' => i += 1,
                // A character class may hold an unescaped `/`, so it has to be
                // stepped over rather than read as the end of the pattern.
                '[' => {
                    while i < bytes.len() && bytes[i] != ']' {
                        if bytes[i] == '\\' {
                            i += 1;
                        }
                        i += 1;
                    }
                }
                '/' => state = In::Code,
                _ => {}
            },
            In::Code => {
                match c {
                    '/' => match bytes.get(i + 1) {
                        Some('/') => state = In::Line,
                        Some('*') => state = In::Block,
                        // A `/` where a value cannot be is the start of a
                        // pattern; anywhere else it is a division.
                        _ if matches!(
                            prev,
                            '\0' | '('
                                | ','
                                | '='
                                | ':'
                                | '['
                                | '!'
                                | '&'
                                | '|'
                                | '?'
                                | '{'
                                | '}'
                                | ';'
                                | '+'
                                | '-'
                                | '*'
                                | '%'
                                | '<'
                                | '>'
                                | '~'
                                | '^'
                        ) =>
                        {
                            state = In::Regex
                        }
                        _ => {}
                    },
                    '\'' | '"' => state = In::Str(c),
                    '`' => state = In::Template,
                    '{' | '(' | '[' => stack.push((c, line)),
                    '}' | ')' | ']' => {
                        let want = match c {
                            '}' => '{',
                            ')' => '(',
                            _ => '[',
                        };
                        match stack.pop() {
                            Some((open, _)) if open == want => {}
                            Some((open, at)) => {
                                return Some(format!(
                                    "line {line}: `{c}` closes `{open}` from line {at}"
                                ))
                            }
                            None => return Some(format!("line {line}: `{c}` closes nothing")),
                        }
                    }
                    _ => {}
                }
                if !c.is_whitespace() {
                    prev = c;
                }
            }
        }
        i += 1;
    }
    stack
        .pop()
        .map(|(open, at)| format!("line {at}: `{open}` is never closed"))
}

#[cfg(test)]
mod tests {
    use super::{methods, unbalanced};
    use std::path::PathBuf;

    fn read(name: &str) -> String {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../web/content/npcd/lib")
            .join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// **The contract `api.js` claims.** Every method the live client offers has
    /// to exist on the mock, or a page that uses it dies under `?mock=1`.
    ///
    /// One direction only. The mock may hold extras — a fixture-only affordance
    /// is not a broken promise — but a name the console can call and the mock
    /// cannot answer is.
    #[test]
    fn the_mock_answers_everything_the_live_client_does() {
        let live = methods(&read("api.live.js"));
        let mock = methods(&read("api.mock.js"));
        assert!(
            live.len() > 40,
            "only found {} methods — the parse is wrong, not the file: {live:?}",
            live.len()
        );
        let missing: Vec<&String> = live.iter().filter(|m| !mock.contains(m)).collect();
        assert!(
            missing.is_empty(),
            "the mock cannot answer {missing:?} — a page using one of these breaks under ?mock=1"
        );
    }

    /// **Every console source still parses as far as its brackets go.**
    ///
    /// The console is hand-written ES modules with no build step, so nothing
    /// between the editor and the browser looks at them. A dropped brace is a
    /// page that silently does nothing, found by opening it. This finds it in
    /// `cargo test` instead, and names the line.
    #[test]
    fn every_console_script_is_balanced() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../web/content");
        let mut checked = 0;
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{dir:?}: {e}")) {
                let path = entry.expect("readable").path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "js") {
                    let source = std::fs::read_to_string(&path).expect("utf-8");
                    if let Some(where_) = unbalanced(&source) {
                        let rel = path.strip_prefix(&root).unwrap_or(&path);
                        panic!("{}: {where_}", rel.display());
                    }
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 20,
            "only checked {checked} scripts — the walk is wrong"
        );
    }

    /// The checker itself, since a checker that says yes to everything is worse
    /// than none: it would report the console healthy for the rest of its life.
    #[test]
    fn the_balance_check_finds_what_it_is_for_and_ignores_what_it_is_not() {
        // Brackets that are not code: in strings, in comments, in templates,
        // and in a regular expression's character class.
        for fine in [
            "const a = { b: [1, 2] };",
            "const s = '} not code';",
            "const s = \"] not code\";",
            "// } not code\nconst a = (1);",
            "/* ] not code */\nconst a = [1];",
            "const t = `a ${ b } }`;",
            "const r = /[)]/g;\nconst a = { b: 1 };",
            "const r = x.replace(/[_-]/g, ' ');",
            "const d = (a) / (b);",
            "const s = 'it\\'s } fine';",
        ] {
            assert_eq!(unbalanced(fine), None, "rejected `{fine}`");
        }
        // And the thing it exists for.
        assert!(unbalanced("function f() {\n  return 1;\n").is_some());
        assert!(unbalanced("const a = [1, 2);").is_some());
        assert!(unbalanced("const a = 1;\n}").is_some());
    }

    /// The value of `key: '…'` on a line, if it is there.
    fn quoted(line: &str, key: &str) -> Option<String> {
        let rest = line.split_once(&format!("{key}: '"))?.1;
        rest.split_once('\'').map(|(v, _)| v.to_owned())
    }

    /// **Every `under:` names a nav entry that exists.**
    ///
    /// `under` is what keeps the top bar lit when you open a world or a
    /// character — the page says which section it belongs to, because
    /// `/world/:wid` under `/worlds` is not a rule anything could derive. A
    /// typo in one degrades silently back to a blank bar, which is the bug it
    /// was added to fix.
    #[test]
    fn every_page_sits_under_a_nav_entry_that_exists() {
        let app = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../web/content/npcd/app.js"),
        )
        .expect("app.js");

        let mut nav_paths = Vec::new();
        let mut unders = Vec::new();
        for line in app.lines() {
            if !line.contains("definePage({") {
                continue;
            }
            if line.contains("nav: {") {
                nav_paths.extend(quoted(line, "path"));
            }
            if let Some(u) = quoted(line, "under") {
                unders.push((quoted(line, "path").unwrap_or_default(), u));
            }
        }

        assert!(
            nav_paths.len() > 5,
            "found {} nav entries — the scan is wrong, not the file",
            nav_paths.len()
        );
        assert!(!unders.is_empty(), "no page declares `under` any more");
        for (page, under) in &unders {
            assert!(
                nav_paths.contains(under),
                "`{page}` sits under `{under}`, which is not a nav entry: {nav_paths:?}"
            );
        }
        // The pages that would otherwise blank the bar. Named, so removing one
        // of these is a decision rather than an omission.
        for page in ["/world/:wid", "/npc/:id", "/npc/new"] {
            assert!(
                unders.iter().any(|(p, _)| p == page),
                "`{page}` declares no `under`, so opening it clears the top bar"
            );
        }
    }

    /// The shared library is mirrored into `zend/web/common/`, and the two
    /// copies must not drift.
    ///
    /// They have before: a fix went into one and the other kept the bug, which
    /// is invisible until somebody opens the console that has the stale copy.
    /// Both are checked in, so this is a comparison rather than a build step.
    ///
    /// Line endings are not part of it. The two trees are checked out with
    /// different ones — the estate's copies are CRLF and zend's are LF — and
    /// failing on that would report drift on every file while hiding the one
    /// that actually said something different.
    #[test]
    fn the_shared_console_library_is_the_same_in_both_trees() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let web = root.join("../web/content/common");
        let zend = root.join("../zend/web/common");
        // The whole tree, not just `lib/`. `base.css` sits at the top of it and
        // was outside the walk when this only looked one directory down — which
        // is exactly where a fix landed in one copy and not the other.
        let mut queue = vec![PathBuf::new()];
        let mut compared = 0;
        while let Some(rel) = queue.pop() {
            let dir = web.join(&rel);
            for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{dir:?}: {e}")) {
                let path = entry.expect("readable").path();
                let name = path.file_name().expect("named").to_owned();
                let here = rel.join(&name);
                if path.is_dir() {
                    queue.push(here);
                    continue;
                }
                let name = here.to_string_lossy().replace('\\', "/");
                let mirror = zend.join(&here);
                if !mirror.exists() {
                    // A file only the estate serves is not drift.
                    continue;
                }
                let a = std::fs::read_to_string(&path).expect("utf-8");
                let b = std::fs::read_to_string(&mirror).expect("utf-8");
                // The first line that differs, rather than both files: a
                // whole-file `assert_eq!` on a two-hundred-line module prints
                // both and names neither.
                let (mut al, mut bl) = (a.lines(), b.lines());
                let mut n = 0;
                loop {
                    n += 1;
                    match (al.next(), bl.next()) {
                        (None, None) => break,
                        (x, y) if x == y => continue,
                        (x, y) => panic!(
                            "common/{name} differs at line {n} — the two trees have drifted\n\
                             web:  {x:?}\n zend: {y:?}"
                        ),
                    }
                }
                compared += 1;
            }
        }
        assert!(
            compared > 9,
            "only compared {compared} files — the walk is wrong"
        );
    }

    /// **`hidden` has to actually hide.**
    ///
    /// The shell hides three elements with the attribute alone, and the
    /// attribute is enforced by a single UA rule that *any* author `display`
    /// beats. `.icon-btn { display: flex }` beat it, so the rail toggle was
    /// visible on every narrow screen including the pages with no rail: pressing
    /// it drew the scrim over an empty drawer, which reads as a button that does
    /// nothing.
    ///
    /// The guard is one line in `base.css`, and removing it puts all three back.
    /// So: find what the shell hides, and check the guard is there.
    #[test]
    fn the_shell_can_hide_the_things_it_hides() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let html =
            std::fs::read_to_string(root.join("../web/content/npcd/index.html")).expect("shell");
        let hidden: Vec<&str> = html
            .lines()
            .filter(|l| l.contains(" hidden") || l.contains(" hidden>"))
            .collect();
        assert!(
            hidden.len() >= 3,
            "expected the shell to hide several elements, found {hidden:?}"
        );

        for css in [
            "../web/content/common/base.css",
            "../zend/web/common/base.css",
        ] {
            let text =
                std::fs::read_to_string(root.join(css)).unwrap_or_else(|e| panic!("{css}: {e}"));
            let guarded = text.lines().any(|l| {
                let l = l.trim();
                l.starts_with("[hidden]") && l.contains("display") && l.contains("!important")
            });
            assert!(
                guarded,
                "{css} has no `[hidden] {{ display: none !important }}` — every element the \
                 shell hides reappears the moment a rule sets `display` on it"
            );
        }
    }

    /// **The preload list is the shell's module graph, and stays it.**
    ///
    /// A browser discovers a module's imports only after parsing it, so the
    /// shell's nine modules arrive in three waves unless the HTML names them up
    /// front. A conditional request costs a round trip even when the answer is
    /// `304` and no bytes, so those waves cost as much on a refresh as on a
    /// first load.
    ///
    /// Both directions matter. An href that no longer exists is a 404 on every
    /// page load; a module added to the graph and not listed here quietly puts
    /// the extra wave back.
    #[test]
    fn the_preloads_are_exactly_the_shells_module_graph() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../web/content");
        let npcd = root.join("npcd");
        let html = std::fs::read_to_string(npcd.join("index.html")).expect("shell");

        let preloaded: Vec<String> = html
            .lines()
            .filter(|l| l.contains("rel=\"modulepreload\""))
            .filter_map(|l| quoted_attr(l, "href"))
            .collect();
        assert!(
            preloaded.len() > 5,
            "found {} preloads — the scan is wrong, not the file",
            preloaded.len()
        );

        // Every one of them is a file this console serves.
        for href in &preloaded {
            let rel = href.trim_start_matches('/');
            assert!(
                npcd.join(rel).exists() || root.join("common").join(rel).exists(),
                "index.html preloads `{href}`, which no site root has"
            );
        }

        // And every module the shell reaches statically is one of them. Walked
        // from `app.js`, because the graph is what matters and not the list.
        let mut seen: Vec<String> = Vec::new();
        let mut queue = vec!["app.js".to_string()];
        while let Some(rel) = queue.pop() {
            if seen.contains(&rel) {
                continue;
            }
            seen.push(rel.clone());
            let path = if npcd.join(&rel).exists() {
                npcd.join(&rel)
            } else {
                root.join("common").join(&rel)
            };
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            for line in source.lines() {
                // Static imports only: a dynamic `await import(…)` is deferred
                // on purpose, and preloading it would undo that.
                if !line.starts_with("import ") {
                    continue;
                }
                let Some(spec) = quoted_attr(line, "from").or_else(|| single_quoted(line)) else {
                    continue;
                };
                let child = spec.trim_start_matches("./").trim_start_matches('/');
                let child = if rel.starts_with("lib/") && !child.starts_with("lib/") {
                    format!("lib/{child}")
                } else {
                    child.to_string()
                };
                queue.push(child);
            }
        }

        for module in seen.iter().filter(|m| *m != "app.js") {
            let href = format!("/{module}");
            assert!(
                preloaded.contains(&href),
                "the shell imports `{href}` but index.html does not preload it — \
                 it arrives a round trip later than it needs to"
            );
        }
    }

    /// The value of `attr="…"` or `attr '…'` on a line.
    fn quoted_attr(line: &str, attr: &str) -> Option<String> {
        let rest = line.split_once(&format!("{attr}=\""))?.1;
        rest.split_once('"').map(|(v, _)| v.to_owned())
    }

    /// The first `'…'` on a line — an ES import specifier.
    fn single_quoted(line: &str) -> Option<String> {
        let rest = line.split_once('\'')?.1;
        rest.split_once('\'').map(|(v, _)| v.to_owned())
    }

    /// **The daemon does not fall back to the console's fixture.**
    ///
    /// It used to. `main.rs` ended in
    /// `fallback_service(guard::behind(…, web::mock::npcd::router()))`, so every
    /// path the real routes had not claimed was answered by invented data —
    /// correctly shaped, plausible, and wrong. A character's beliefs came back
    /// for ids that did not exist.
    ///
    /// This is the one-line check that it stays gone. The fixture is still
    /// built and still served by `web --authoritative`, which is what it is
    /// for; what it must not do is stand behind a daemon that means it.
    #[test]
    fn the_daemon_has_no_fixture_behind_it() {
        let main =
            std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/main.rs"))
                .expect("main.rs");
        for line in main.lines() {
            // Comments explaining the removal are the point; code is not.
            let code = line.trim_start();
            if code.starts_with("//") || code.starts_with('*') || code.starts_with("/*") {
                continue;
            }
            assert!(
                !code.contains("fallback_service"),
                "main.rs installs a fallback again: {line}"
            );
            assert!(
                !code.contains("mock::npcd"),
                "main.rs reaches for the console's fixture again: {line}"
            );
        }
    }

    /// **Every hand-written shell says who it is, and agrees with itself.**
    ///
    /// These pages render themselves, so a crawler — or a chat client building a
    /// preview card, which is how a developer tool actually spreads — sees the
    /// `<head>` and an empty body. The head is therefore the entire message, and
    /// it is the easiest thing in the tree to lose: nothing renders differently
    /// when a `<meta>` goes missing, and nothing fails when it is wrong.
    ///
    /// The agreement half is what makes this more than a presence check. A
    /// canonical URL, an `og:url` and the `Sitemap:` line in `robots.txt` are
    /// three statements of the same fact, written in three files; when they
    /// disagree the visible symptom is a page that will not index, weeks later,
    /// with nothing to point at.
    ///
    /// tokera.com is absent because it generates its head in Rust
    /// (`web/src/site/tokera/page.rs`) and is checked there, per page.
    #[test]
    fn every_shell_says_who_it_is() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // (the shell, the directory its robots.txt sits in, its one address)
        let shells = [
            (
                "../web/content/npcd/index.html",
                "../web/content/npcd",
                "https://bot.tokera.com/",
            ),
            (
                "../zend/web/index.html",
                "../zend/web",
                "https://code.tokera.com/",
            ),
            (
                "../web/content/battlecities/index.html",
                "../web/content/battlecities",
                "https://battlecities.net/",
            ),
        ];

        for (shell, dir, origin) in shells {
            let html = std::fs::read_to_string(root.join(shell))
                .unwrap_or_else(|e| panic!("{shell}: {e}"));

            // A card needs all four to render; a missing description is the one
            // that degrades silently, into whatever text the client scrapes.
            for needle in [
                "name=\"description\"",
                "property=\"og:title\"",
                "property=\"og:description\"",
                "name=\"twitter:card\"",
            ] {
                assert!(html.contains(needle), "{shell} has no {needle}");
            }

            // The title is the link text everywhere the page is listed. A bare
            // product name says nothing about what it is.
            let title = html
                .lines()
                .find_map(|l| l.split_once("<title>").and_then(|(_, r)| r.split_once("</title>")))
                .map(|(t, _)| t.trim().to_owned())
                .unwrap_or_else(|| panic!("{shell} has no <title>"));
            assert!(
                title.len() > 12,
                "{shell}'s title is {title:?} — too short to say what the page is"
            );

            // The three statements of the same address.
            let canonical = html
                .lines()
                .find(|l| l.contains("rel=\"canonical\""))
                .and_then(|l| quoted_attr(l, "href"))
                .unwrap_or_else(|| panic!("{shell} declares no canonical URL"));
            assert_eq!(canonical, origin, "{shell}'s canonical is not its address");

            let og_url = html
                .lines()
                .find(|l| l.contains("property=\"og:url\""))
                .and_then(|l| quoted_attr(l, "content"))
                .unwrap_or_else(|| panic!("{shell} has no og:url"));
            assert_eq!(og_url, origin, "{shell}'s og:url disagrees with its canonical");

            let robots = std::fs::read_to_string(root.join(dir).join("robots.txt"))
                .unwrap_or_else(|e| panic!("{dir}/robots.txt: {e}"));
            let sitemap = robots
                .lines()
                .find_map(|l| l.trim().strip_prefix("Sitemap:"))
                .map(str::trim)
                .unwrap_or_else(|| panic!("{dir}/robots.txt names no sitemap"));
            assert_eq!(
                sitemap,
                format!("{origin}sitemap.xml"),
                "{dir}/robots.txt points at a sitemap on another host"
            );

            // And that file exists and lists the address it was named for. A
            // sitemap referenced but absent is a 404 a crawler retries.
            let xml = std::fs::read_to_string(root.join(dir).join("sitemap.xml"))
                .unwrap_or_else(|e| panic!("{dir}/sitemap.xml: {e}"));
            assert!(
                xml.contains(&format!("<loc>{origin}</loc>")),
                "{dir}/sitemap.xml does not list {origin}"
            );
        }
    }

    /// The parse itself, pinned against both spellings the files use. Without
    /// this a change of style silently empties the list above and the contract
    /// stops being checked while the test keeps passing.
    #[test]
    fn both_spellings_of_a_method_are_found() {
        let found = methods(
            "const qs = (o) => {\n  \
             for (const k of o) {}\n\
             };\n\
             export const X = {\n  \
             getStatus:    () => j('/v1/status'),\n  \
             async getNpc(id) {\n    \
             notAMethod: 1,\n  \
             },\n  \
             // a comment\n  \
             plain: 3,\n  \
             subscribeLogs(a, b) {\n  \
             },\n\
             };\n",
        );
        assert_eq!(found, ["getNpc", "getStatus", "subscribeLogs"]);
    }
}
