//! LaTeX → MathML, at render time.
//!
//! The alternative was KaTeX or MathJax, which means shipping a JavaScript
//! bundle and sixty-odd font files, running the typesetter in every reader's
//! browser, and reflowing the page after first paint. MathML is laid out
//! natively by every current browser, so converting once on the server costs
//! the reader nothing and the repo no build step — which is the same reason
//! the consoles have no bundler.
//!
//! Conversion that fails does not fail the page. A paper with one construct
//! this converter does not know should render with that one expression shown
//! as literal LaTeX, not 404 — a reader can still read `\alpha=2.0`, and the
//! author can see exactly which expression needs attention.

use latex2mathml::{latex_to_mathml, DisplayStyle};

/// `$…$` — flows with the sentence around it.
pub fn inline(latex: &str) -> String {
    convert(latex, DisplayStyle::Inline, "math-inline")
}

/// `$$…$$` — its own centred block.
pub fn display(latex: &str) -> String {
    format!(
        "<div class=\"math-block\">{}</div>",
        convert(latex, DisplayStyle::Block, "math-display")
    )
}

/// `latex2mathml` reports a construct it does not know **inside** its output,
/// as `<mtext>[PARSE ERROR: …]</mtext>`, rather than as an `Err` — so this
/// marker is the only signal that a conversion went wrong. Coupling to it is
/// unattractive but the alternative is publishing the words "PARSE ERROR" in
/// the middle of a paper.
const PARSE_ERROR: &str = "[PARSE ERROR";

/// Rewrite `\mathcal{X}` to the Unicode mathematical script letter it denotes.
///
/// `latex2mathml` 0.2 does not know `\mathcal`, and the theorem this site
/// exists to publish states its bound over `\mathcal{W}` — so without this the
/// paper's central equations are the ones that fall back to raw LaTeX.
///
/// Substituting the character is not a workaround for a missing feature; it is
/// what the command means. `\mathcal{W}` *is* U+1D4B2, and the converter emits
/// it as an `<mi>` exactly as it would any other letter.
fn expand_script_letters(latex: &str) -> String {
    const CMD: &str = "\\mathcal{";
    if !latex.contains(CMD) {
        return latex.to_string();
    }
    let mut out = String::with_capacity(latex.len());
    let mut rest = latex;
    while let Some(at) = rest.find(CMD) {
        let after = &rest[at + CMD.len()..];
        let mut chars = after.chars();
        match (chars.next(), chars.next()) {
            (Some(letter), Some('}')) if letter.is_ascii_alphabetic() => {
                out.push_str(&rest[..at]);
                out.push(script_letter(letter));
                rest = &after[letter.len_utf8() + 1..];
            }
            // `\mathcal{ABC}` or anything else is left alone rather than
            // guessed at: the converter's fallback will show it as LaTeX, which
            // is the correct outcome for something this does not understand.
            _ => {
                let upto = at + CMD.len();
                out.push_str(&rest[..upto]);
                rest = &rest[upto..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The script letter for `c`. Unicode's Mathematical Alphanumeric Symbols block
/// has holes where the character already existed in Letterlike Symbols, so the
/// arithmetic needs those eleven exceptions.
fn script_letter(c: char) -> char {
    match c {
        'B' => 'ℬ',
        'E' => 'ℰ',
        'F' => 'ℱ',
        'H' => 'ℋ',
        'I' => 'ℐ',
        'L' => 'ℒ',
        'M' => 'ℳ',
        'R' => 'ℛ',
        'e' => 'ℯ',
        'g' => 'ℊ',
        'o' => 'ℴ',
        'A'..='Z' => char::from_u32(0x1D49C + (c as u32 - 'A' as u32)).unwrap_or(c),
        'a'..='z' => char::from_u32(0x1D4B6 + (c as u32 - 'a' as u32)).unwrap_or(c),
        _ => c,
    }
}

/// Manual delimiter sizing — `\big[`, `\Bigl(`, `\biggr\}` and the rest.
///
/// These are typesetting instructions for a system that does not size
/// delimiters on its own. MathML does: an `<mo>` fence stretches to its
/// content by default, so dropping the size command and keeping the delimiter
/// is not an approximation — it is asking the renderer to do the thing the
/// command was compensating for.
///
/// Without this, PalQuant's K-side error metric — a definition the paper turns
/// on — renders as raw LaTeX, because it wraps its numerator in `\big[…\big]`.
fn strip_delimiter_sizing(latex: &str) -> String {
    // Longest first: matching `\big` inside `\bigg[` would leave a stray `g`.
    const SIZERS: [&str; 16] = [
        "\\Biggl", "\\Biggr", "\\Biggm", "\\biggl", "\\biggr", "\\biggm", "\\Bigg", "\\bigg",
        "\\Bigl", "\\Bigr", "\\Bigm", "\\bigl", "\\bigr", "\\bigm", "\\Big", "\\big",
    ];
    if !latex.contains("\\big") && !latex.contains("\\Big") {
        return latex.to_string();
    }
    let mut out = String::with_capacity(latex.len());
    let mut rest = latex;
    'outer: while !rest.is_empty() {
        for s in SIZERS {
            if let Some(after) = rest.strip_prefix(s) {
                // Only a size command if a delimiter follows. `\bigcup` starts
                // with `\big` and is a symbol in its own right; eating the
                // prefix would turn it into `cup`.
                if after
                    .chars()
                    .next()
                    .is_some_and(|c| !c.is_ascii_alphanumeric())
                {
                    rest = after;
                    continue 'outer;
                }
            }
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

/// `\text{…}` whose body holds anything but letters, rewritten to `\mathrm{…}`.
///
/// `latex2mathml`'s `\text` tokenises its body as maths and gives up on the
/// first character that is not a letter, so `\text{top-4}` and `\text{Q4}`
/// abort the whole expression. `\mathrm` takes the same body without
/// complaint, and for a short upright label the two are visually the same
/// thing.
///
/// Bodies that are only letters keep `\text`, which produces `<mtext>` — the
/// semantically correct element, and what already works. So this changes the
/// output of exactly the expressions that were failing.
///
/// The ASCII hyphen becomes U+2010 on the way. Inside text a `-` *is* a
/// hyphen; leaving it would make `\mathrm` read it as a minus sign and render
/// `top - 4`, spaced as an operator.
fn upright_text(latex: &str) -> String {
    const CMD: &str = "\\text{";
    if !latex.contains(CMD) {
        return latex.to_string();
    }
    let mut out = String::with_capacity(latex.len());
    let mut rest = latex;
    while let Some(at) = rest.find(CMD) {
        out.push_str(&rest[..at]);
        let after = &rest[at + CMD.len()..];
        let Some(end) = matching_brace(after) else {
            // Unbalanced: hand the remainder over untouched and let the
            // converter report it, rather than inventing a closing brace.
            out.push_str(&rest[at..]);
            return out;
        };
        let body = &after[..end];
        if plain_letters(body) {
            out.push_str(CMD);
            out.push_str(body);
            out.push('}');
        } else {
            out.push_str("\\mathrm{");
            out.push_str(&body.replace('-', "\u{2010}"));
            out.push('}');
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    out
}

/// Offset of the `}` closing a body that started just past its `{`.
fn matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '{' => depth += 1,
            '}' if depth == 0 => return Some(i),
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Is this body something `\text` handles? Letters and spaces only —
/// backslash escapes such as `\_` are counted as letters because the converter
/// does accept them.
fn plain_letters(body: &str) -> bool {
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                chars.next();
            }
            c if c.is_alphabetic() || c == ' ' => {}
            _ => return false,
        }
    }
    true
}

/// Rewrites that make a command mean to the converter what it means on the
/// page. Each one is documented where it is defined; none of them guesses, and
/// each leaves untouched every expression that already converts.
fn preprocess(latex: &str) -> String {
    upright_text(&strip_delimiter_sizing(&expand_script_letters(latex)))
}

fn convert(latex: &str, style: DisplayStyle, class: &str) -> String {
    let latex = &preprocess(latex);
    let failure = match latex_to_mathml(latex, style) {
        Ok(mathml) if !mathml.contains(PARSE_ERROR) => return mathml,
        Ok(_) => "unsupported construct".to_string(),
        Err(e) => e.to_string(),
    };
    // Worth a log line: it is the author's cue that an expression in a
    // published document is not rendering, and nothing else surfaces it.
    tracing::debug!(error = %failure, latex, "math: falling back to literal LaTeX");
    format!("<code class=\"{class} math-raw\">{}</code>", escape(latex))
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_math_becomes_mathml() {
        let html = inline("\\alpha = 2.0");
        assert!(html.contains("<math"), "{html}");
        assert!(
            !html.contains("math-raw"),
            "should not have fallen back: {html}"
        );
    }

    #[test]
    fn display_math_is_wrapped_as_a_block() {
        let html = display("S = \\frac{1}{3}\\sum_b L_b^2");
        assert!(html.starts_with("<div class=\"math-block\">"), "{html}");
        assert!(html.contains("<math"), "{html}");
    }

    #[test]
    fn the_papers_own_expressions_convert() {
        // Taken verbatim from docs/unbounded_agents.md — the constructs that
        // actually have to work, rather than ones chosen to be easy.
        for src in [
            "\\text{TokenSignature} = \\bigoplus_{i=0}^{3} \\text{sign}(Q^{\\ell_0}_i)",
            "Q^{\\ell}_i \\in \\mathbb{R}^{128}",
            "\\{-1,+1\\}^{128}",
            "S(\\text{section}) = \\frac{1}{3}\\sum_{b}\\sum_{\\text{runs}} L_b^2",
        ] {
            let html = inline(src);
            assert!(
                !html.contains("math-raw"),
                "did not convert: {src} -> {html}"
            );
        }
    }

    #[test]
    fn the_theorems_own_statement_converts() {
        // The O(1) bound, verbatim from docs/unbounded_agents.md §11.2. It is
        // the reason the site exists; falling back here would be the single
        // most visible rendering failure on it.
        let src = r"E\left[\sum_{t \in \mathcal{W}} \varepsilon(t)\right] \leq \varepsilon_{\text{hot}} + W_{\text{warm\_max}} \cdot \varepsilon_{\text{warm}} + O\!\left(\frac{1}{N}\right) = O(1)";
        let html = display(src);
        assert!(!html.contains("math-raw"), "{html}");
        assert!(
            html.contains('\u{1D4B2}'),
            "\\mathcal{{W}} did not become script W"
        );
    }

    #[test]
    fn the_k_side_error_metric_converts() {
        // Verbatim from docs/palquant.md §3.3 — the definition the paper turns
        // on, and the one `\big[` used to send to the fallback.
        let src = r"\varepsilon_K(\mathbf{k}, \hat{\mathbf{k}}) = \frac{\text{mean}_4 \big[|k_i - \hat{k}_i| \cdot w_i\big]}{\text{head amax}}";
        let html = display(src);
        assert!(!html.contains("math-raw"), "{html}");
    }

    #[test]
    fn a_text_label_with_digits_or_a_hyphen_converts() {
        // Verbatim from docs/palquant.md Appendix G. `\text{top-4}` aborted
        // the whole expression before `\mathrm` took over for these bodies.
        let src = r"\varepsilon_K^{\text{top4}} = \text{mean}_{\text{top-4}}\big[|k_i - \hat{k}_i| \cdot w_i\big]";
        let html = inline(src);
        assert!(!html.contains("math-raw"), "{html}");
        // The label's hyphen is a hyphen, not a minus sign — an `<mo>` there
        // would be spaced as an operator and read "top - 4". The `<mo>-</mo>`
        // elsewhere in this expression is the real subtraction in
        // `k_i - \hat{k}_i`, which is why this asserts on the character.
        assert!(
            html.contains("<mi mathvariant=\"normal\">\u{2010}</mi>"),
            "{html}"
        );
    }

    #[test]
    fn a_letters_only_text_label_still_uses_mtext() {
        // The semantically right element, and it already worked — this rewrite
        // must not touch expressions that were fine.
        assert_eq!(upright_text(r"\text{mean}_4"), r"\text{mean}_4");
        assert!(inline(r"\text{sign}(x)").contains("<mtext>sign</mtext>"));
        // An escaped underscore is accepted by `\text`, so it stays too.
        assert_eq!(upright_text(r"\text{warm\_max}"), r"\text{warm\_max}");
    }

    #[test]
    fn only_bodies_the_converter_rejects_are_rewritten() {
        assert_eq!(upright_text(r"\text{top-4}"), "\\mathrm{top\u{2010}4}");
        assert_eq!(upright_text(r"\text{Q4}"), r"\mathrm{Q4}");
        // `\textbf` is a different command and must not be caught by the
        // `\text` prefix.
        assert_eq!(upright_text(r"\textbf{Q4}"), r"\textbf{Q4}");
    }

    #[test]
    fn an_unbalanced_brace_is_handed_over_rather_than_repaired() {
        let src = r"\text{oops";
        assert_eq!(upright_text(src), src);
    }

    #[test]
    fn delimiter_sizing_is_dropped_and_the_delimiter_kept() {
        assert_eq!(strip_delimiter_sizing(r"\big[x\big]"), "[x]");
        assert_eq!(strip_delimiter_sizing(r"\Biggl(y\Biggr)"), "(y)");
        // Longest-match: `\bigg` must not be read as `\big` + `g`.
        assert_eq!(strip_delimiter_sizing(r"\bigg\{z\bigg\}"), r"\{z\}");
    }

    #[test]
    fn a_symbol_that_merely_starts_with_big_is_left_alone() {
        // `\bigcup` and `\bigoplus` are operators, not sizing commands.
        assert_eq!(strip_delimiter_sizing(r"\bigcup_i A_i"), r"\bigcup_i A_i");
        let html = inline(r"\bigoplus_{i=0}^{3} x_i");
        assert!(!html.contains("math-raw"), "{html}");
    }

    #[test]
    fn script_letters_use_the_letterlike_characters_where_unicode_has_holes() {
        assert_eq!(script_letter('W'), '\u{1D4B2}');
        assert_eq!(script_letter('A'), '\u{1D49C}');
        assert_eq!(script_letter('L'), 'ℒ'); // U+2112, not U+1D4AB
        assert_eq!(script_letter('e'), 'ℯ'); // U+212F
        assert_eq!(script_letter('z'), '\u{1D4CF}');
    }

    #[test]
    fn a_mathcal_of_more_than_one_letter_is_left_for_the_converter() {
        // Guessing at it would be worse than the visible fallback.
        let src = r"\mathcal{ABC} + \mathcal{W}";
        let out = expand_script_letters(src);
        assert!(out.contains(r"\mathcal{ABC}"), "{out}");
        assert!(out.contains('\u{1D4B2}'), "{out}");
    }

    #[test]
    fn text_without_mathcal_is_untouched() {
        let src = r"\frac{1}{N} + \mathbb{R}";
        assert_eq!(expand_script_letters(src), src);
    }

    #[test]
    fn a_broken_expression_degrades_to_readable_latex_instead_of_failing() {
        let html = inline("\\notacommand{<x>}");
        assert!(html.contains("math-raw"), "{html}");
        assert!(
            html.contains("&lt;x&gt;"),
            "the fallback must still escape: {html}"
        );
    }
}
