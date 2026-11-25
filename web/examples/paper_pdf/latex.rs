//! The LaTeX the paper is typeset with beyond pandoc's own template: a running
//! head with the paper's short title and status, the page number at the foot,
//! section sizes for a 10 pt paper, and smaller type in tables.

/// The header pandoc includes after its own preamble.
pub fn header(head_left: &str, head_right: &str) -> String {
    format!(
        "\\usepackage{{fancyhdr}}\n\
         \\newcommand{{\\HeadLeft}}{{{left}}}\n\
         \\newcommand{{\\HeadRight}}{{{right}}}\n\
         \\pagestyle{{fancy}}\n\
         \\fancyhf{{}}\n\
         \\fancyhead[L]{{\\small\\itshape\\HeadLeft}}\n\
         \\fancyhead[R]{{\\small\\HeadRight}}\n\
         \\fancyfoot[C]{{\\thepage}}\n\
         \\renewcommand{{\\headrulewidth}}{{0.4pt}}\n\
         \\makeatletter\\let\\ps@plain\\ps@fancy\\makeatother\n\
         \\usepackage{{titlesec}}\n\
         \\titleformat*{{\\section}}{{\\large\\bfseries}}\n\
         \\titleformat*{{\\subsection}}{{\\normalsize\\bfseries}}\n\
         \\titleformat*{{\\subsubsection}}{{\\normalsize\\bfseries\\itshape}}\n\
         \\titlespacing*{{\\section}}{{0pt}}{{12pt plus 3pt minus 2pt}}{{5pt}}\n\
         \\titlespacing*{{\\subsection}}{{0pt}}{{9pt plus 2pt minus 2pt}}{{3pt}}\n\
         \\titlespacing*{{\\subsubsection}}{{0pt}}{{7pt plus 2pt minus 1pt}}{{2pt}}\n\
         \\usepackage{{titling}}\n\
         \\pretitle{{\\begin{{center}}\\LARGE\\bfseries}}\n\
         \\posttitle{{\\par\\end{{center}}\\vskip 0.4em}}\n\
         \\preauthor{{\\begin{{center}}\\normalsize}}\n\
         \\postauthor{{\\par\\end{{center}}}}\n\
         \\predate{{\\begin{{center}}\\normalsize}}\n\
         \\postdate{{\\par\\end{{center}}\\vskip 0.6em}}\n\
         \\usepackage{{xurl}}\n\
         \\usepackage{{newunicodechar}}\n\
         \\newunicodechar{{∈}}{{\\ensuremath{{\\in}}}}\n\
         \\newunicodechar{{𝒲}}{{\\ensuremath{{\\mathcal{{W}}}}}}\n\
         \\XeTeXinterchartokenstate=0\n\
         \\newXeTeXintercharclass\\PathBreak\n\
         \\XeTeXcharclass`\\/=\\PathBreak\n\
         \\XeTeXcharclass`\\.=\\PathBreak\n\
         \\XeTeXinterchartoks\\PathBreak 0={{\\penalty100\\relax}}\n\
         \\newcommand{{\\PaperUnderscore}}{{\\textunderscore\\penalty100\\relax}}\n\
         \\let\\PaperTT\\texttt\n\
         \\renewcommand{{\\texttt}}[1]{{\\PaperTT{{\\XeTeXinterchartokenstate=1 \\let\\_\\PaperUnderscore #1}}}}\n\
         \\usepackage{{etoolbox}}\n\
         \\AtBeginEnvironment{{longtable}}{{\\small}}\n\
         \\AtBeginEnvironment{{Shaded}}{{\\small}}\n\
         \\AtBeginEnvironment{{verbatim}}{{\\small}}\n\
         \\setlength{{\\emergencystretch}}{{3em}}\n",
        left = escape(head_left),
        right = escape(head_right),
    )
}

/// The characters the typeset log reports no glyph for, each once, in the
/// order first reported. XeTeX drops such a character from the page and goes
/// on, so a print that has any is refused rather than shipped with a gap.
pub fn missing_glyphs(log: &str) -> Vec<String> {
    const KEY: &str = "Missing character: There is no ";
    let mut out: Vec<String> = Vec::new();
    for line in log.lines() {
        if let Some(rest) = line.split(KEY).nth(1) {
            let what = rest.split(" in font").next().unwrap_or(rest).to_string();
            if !out.contains(&what) {
                out.push(what);
            }
        }
    }
    out
}

/// `s` as LaTeX text: every character LaTeX treats as markup, written so it
/// prints as itself.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' | '}' | '$' | '&' | '#' | '%' | '_' => {
                out.push('\\');
                out.push(c);
            }
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markup_characters_print_as_themselves() {
        assert_eq!(escape("One Card, One Stack"), "One Card, One Stack");
        assert_eq!(
            escape(r"a\b{c}$d&e#f%g_h~i^j"),
            r"a\textbackslash{}b\{c\}\$d\&e\#f\%g\_h\textasciitilde{}i\textasciicircum{}j"
        );
    }

    #[test]
    fn a_missing_glyph_is_named_once() {
        let log = "Missing character: There is no ∈ (U+2208) in font Consolas/OT:script=latn;\n\
                   ok line\n\
                   Missing character: There is no ∈ (U+2208) in font Times New Roman/OT:x\n\
                   Missing character: There is no 𝒲 (U+1D4B2) in font Times New Roman Italic/OT\n";
        assert_eq!(missing_glyphs(log), vec!["∈ (U+2208)", "𝒲 (U+1D4B2)"]);
        assert!(missing_glyphs("all set\n").is_empty());
    }

    #[test]
    fn the_running_head_carries_the_escaped_title_and_status() {
        let h = header("A & B", "v2");
        assert!(h.starts_with(
            "\\usepackage{fancyhdr}\n\\newcommand{\\HeadLeft}{A \\& B}\n\\newcommand{\\HeadRight}{v2}\n"
        ));
        assert!(h.contains("\\fancyhead[L]{\\small\\itshape\\HeadLeft}\n"));
        assert!(h.contains("\\fancyfoot[C]{\\thepage}\n"));
    }
}
