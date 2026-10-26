//! A fetched page as the text a reader takes from it.
//!
//! Every block of the body is kept — paragraphs, list items, table rows,
//! figure captions, preformatted text — each on lines of its own, so a fact in
//! a list or a table reaches the reader as one in a paragraph does. What is
//! written to change a number's meaning keeps it: a superscript is `^`, a
//! subscript `_` (`10^-3`, not `10-3`), and math is its TeX `alttext`
//! (`$1\frac{1}{2}$`) rather than the digits of its MathML nodes run
//! together. Chrome — scripts, styles, navigation, footers, buttons — is left
//! out; a `<header>` or a `<form>` is kept, since an article's title is in
//! its header and a whole page may sit in one form.

use scraper::{ElementRef, Html, Selector};

/// Elements whose content is never the page's text.
const SKIPPED: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "nav", "footer", "button", "iframe", "select",
];

/// Elements set apart by a blank line.
const SECTIONS: &[&str] = &[
    "p",
    "blockquote",
    "figure",
    "figcaption",
    "ul",
    "ol",
    "dl",
    "table",
    "caption",
    "details",
    "address",
    "section",
    "article",
    "main",
    "aside",
];

/// Elements that start a line of their own.
const LINES: &[&str] = &[
    "div", "dt", "dd", "summary", "hr", "thead", "tbody", "tfoot",
];

/// The page's title, and the text of its body.
pub fn html_to_text(html: &str) -> (String, String) {
    let document = Html::parse_document(html);
    let title = Selector::parse("title")
        .ok()
        .and_then(|s| document.select(&s).next())
        .map(|t| t.text().collect::<Vec<_>>().join(" "))
        .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    let mut text = Text::default();
    if let Some(body) = Selector::parse("body")
        .ok()
        .and_then(|s| document.select(&s).next())
    {
        text.children(body);
    }
    (title, text.finish())
}

/// The text being written, whether a space is owed before the next word, and
/// the cells of the table row being written so far.
#[derive(Default)]
struct Text {
    out: String,
    space: bool,
    cells: usize,
}

impl Text {
    fn walk(&mut self, el: ElementRef<'_>) {
        let name = el.value().name();
        if SKIPPED.contains(&name) {
            return;
        }
        match name {
            "br" => self.line(),
            "math" => {
                let tex = el
                    .value()
                    .attr("alttext")
                    .map(str::to_string)
                    .unwrap_or_else(|| el.text().collect());
                self.word(&format!("${}$", tex.trim()));
            }
            "pre" => {
                self.gap();
                self.out.push_str(el.text().collect::<String>().trim_end());
                self.gap();
            }
            "sup" | "sub" => {
                self.space = false;
                self.out.push(if name == "sup" { '^' } else { '_' });
                self.children(el);
            }
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.gap();
                let level = name[1..].parse().unwrap_or(1);
                self.out.push_str(&"#".repeat(level));
                self.out.push(' ');
                self.children(el);
                self.gap();
            }
            "li" => {
                self.line();
                self.out.push_str("- ");
                self.children(el);
                self.line();
            }
            "tr" => {
                self.line();
                self.cells = 0;
                self.children(el);
                self.line();
            }
            // Every cell after the row's first is set off by ` | `, an empty
            // one too, so each value stays in its header's column.
            "td" | "th" => {
                if self.cells > 0 {
                    let kept = self.out.trim_end_matches(' ').len();
                    self.out.truncate(kept);
                    self.out.push_str(" | ");
                }
                self.cells += 1;
                self.space = false;
                self.children(el);
            }
            _ if SECTIONS.contains(&name) => {
                self.gap();
                self.children(el);
                self.gap();
            }
            _ if LINES.contains(&name) => {
                self.line();
                self.children(el);
                self.line();
            }
            _ => self.children(el),
        }
    }

    fn children(&mut self, el: ElementRef<'_>) {
        for child in el.children() {
            if let Some(e) = ElementRef::wrap(child) {
                self.walk(e);
            } else if let Some(t) = child.value().as_text() {
                self.words(t);
            }
        }
    }

    /// Text as it reads: its whitespace one space, and a space owed across
    /// the edge of an element where the text had one.
    fn words(&mut self, s: &str) {
        if s.starts_with(char::is_whitespace) {
            self.space = true;
        }
        for w in s.split_whitespace() {
            self.word(w);
            self.space = true;
        }
        if !s.ends_with(char::is_whitespace) {
            self.space = false;
        }
    }

    fn word(&mut self, w: &str) {
        if self.space && !self.at_line_start() && !self.out.ends_with(' ') {
            self.out.push(' ');
        }
        self.out.push_str(w);
        self.space = false;
    }

    fn at_line_start(&self) -> bool {
        self.out.is_empty() || self.out.ends_with('\n')
    }

    /// End the line written so far.
    fn line(&mut self) {
        let kept = self.out.trim_end_matches(' ').len();
        self.out.truncate(kept);
        if !self.at_line_start() {
            self.out.push('\n');
        }
        self.space = false;
    }

    /// End the line, and leave a blank one before what follows.
    fn gap(&mut self) {
        self.line();
        if !self.out.is_empty() && !self.out.ends_with("\n\n") {
            self.out.push('\n');
        }
    }

    /// The text, without the line breaks around it. Inside it nothing is
    /// collapsed: blocks leave at most one blank line between them as they are
    /// written, and preformatted text keeps every line and blank line it has.
    fn finish(self) -> String {
        self.out.trim_start_matches('\n').trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(html: &str) -> String {
        html_to_text(&format!("<html><body>{html}</body></html>")).1
    }

    #[test]
    fn the_title_and_paragraphs_are_read_with_their_whitespace_collapsed() {
        let (title, text) = html_to_text(
            "<html><head><title>  KV   cache </title></head>\
             <body><h1>QuaRot</h1><p>Rotates  the\n keys.</p><p>Online.</p></body></html>",
        );
        assert_eq!(title, "KV cache");
        assert_eq!(text, "# QuaRot\n\nRotates the keys.\n\nOnline.");
    }

    /// **Lists and tables are text too**: a number in a table cell reaches
    /// the reader, row by row.
    #[test]
    fn lists_and_tables_are_kept_row_by_row() {
        assert_eq!(
            body(
                "<ul><li>4-bit keys</li><li>8-bit values</li></ul>\
                 <table><tr><th>Method</th><th>PPL</th></tr>\
                 <tr><td>QuaRot</td><td>6.10</td></tr></table>"
            ),
            "- 4-bit keys\n- 8-bit values\n\nMethod | PPL\nQuaRot | 6.10"
        );
    }

    /// **A number keeps its meaning**: an exponent is not run into its base,
    /// and math reads as its TeX rather than its nodes' digits.
    #[test]
    fn superscripts_subscripts_and_math_keep_what_the_number_means() {
        assert_eq!(
            body("<p>Error 10<sup>-3</sup> at K<sub>i</sub>.</p>"),
            "Error 10^-3 at K_i."
        );
        assert_eq!(
            body(
                "<p>About <math alttext=\"1\\frac{1}{2}\"><mn>1</mn><mfrac><mn>1</mn><mn>2\
                 </mn></mfrac></math> bits.</p>"
            ),
            "About $1\\frac{1}{2}$ bits."
        );
        assert_eq!(body("<p>About 1½ bits.</p>"), "About 1½ bits.");
    }

    #[test]
    fn chrome_is_left_out_and_preformatted_text_keeps_its_lines() {
        assert_eq!(
            body(
                "<nav>Home | About</nav>\
                 <script>var x = 1;</script><style>p{}</style>\
                 <pre>fn main() {\n    x\n}</pre><footer>(c)</footer>"
            ),
            "fn main() {\n    x\n}"
        );
        assert_eq!(
            body("<p>Before.</p><pre>import os\n\n\ndef main():\n    pass</pre>"),
            "Before.\n\nimport os\n\n\ndef main():\n    pass",
            "two blank lines in a listing are two"
        );
    }

    /// **An article's title and a page inside a form are its text.**
    #[test]
    fn a_header_and_a_form_are_kept() {
        assert_eq!(
            body("<article><header><h1>QuaRot</h1></header><p>Rotates keys.</p></article>"),
            "# QuaRot\n\nRotates keys."
        );
        assert_eq!(
            body("<form id=\"aspnetForm\"><p>The whole page.</p></form>"),
            "The whole page."
        );
    }

    /// **An empty cell keeps its column**: the values under a blank corner
    /// stay under their headers.
    #[test]
    fn an_empty_cell_keeps_every_value_in_its_column() {
        assert_eq!(
            body(
                "<table><tr><th></th><th>2023</th><th>2024</th></tr>\
                 <tr><td>PPL</td><td>6.1</td><td>5.9</td></tr></table>"
            ),
            " | 2023 | 2024\nPPL | 6.1 | 5.9"
        );
    }

    #[test]
    fn inline_elements_join_words_as_the_text_does() {
        assert_eq!(
            body("<p>The <b>key</b> rotation is <a href=\"#\">online</a>, not offline.</p>"),
            "The key rotation is online, not offline."
        );
        assert_eq!(
            body("<div>one</div><div>two<br>three</div>"),
            "one\ntwo\nthree"
        );
    }
}
