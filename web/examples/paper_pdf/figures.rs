//! A paper's figures, lifted out of its markdown. Each figure is an inline SVG
//! styled by the site's stylesheet, which LaTeX cannot read; it is cut out, put
//! on a sheet of its own exactly its size, and printed to a vector PDF that the
//! typeset paper includes in its place.

use std::path::Path;

use anyhow::{anyhow, Context};

const OPEN: &str = "<figure class=\"fig\">";
const CLOSE: &str = "</figure>";

/// One figure: its HTML as it stands in the paper, and its SVG's view box.
#[derive(Debug, Clone, PartialEq)]
pub struct Figure {
    pub html: String,
    pub width: f64,
    pub height: f64,
}

/// The paper with every figure replaced by an image of the file `name` gives
/// its 1-based number, and the figures in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Lifted {
    pub markdown: String,
    pub figures: Vec<Figure>,
}

pub fn lift(md: &str, name: impl Fn(usize) -> String) -> anyhow::Result<Lifted> {
    let mut markdown = String::with_capacity(md.len());
    let mut figures = Vec::new();
    let mut rest = md;
    while let Some(start) = rest.find(OPEN) {
        let number = figures.len() + 1;
        let end = rest[start..]
            .find(CLOSE)
            .ok_or_else(|| anyhow!("figure {number} never closes"))?
            + start
            + CLOSE.len();
        let html = &rest[start..end];
        let (width, height) = view_box(html).with_context(|| format!("figure {number}"))?;
        markdown.push_str(&rest[..start]);
        markdown.push_str(&format!("![]({}){{width=100%}}", name(number)));
        figures.push(Figure {
            html: html.to_string(),
            width,
            height,
        });
        rest = &rest[end..];
    }
    markdown.push_str(rest);
    Ok(Lifted { markdown, figures })
}

/// The width and height of the first `viewBox` in `html`.
pub fn view_box(html: &str) -> anyhow::Result<(f64, f64)> {
    const KEY: &str = "viewBox=\"";
    let at = html.find(KEY).ok_or_else(|| anyhow!("no viewBox"))? + KEY.len();
    let len = html[at..]
        .find('"')
        .ok_or_else(|| anyhow!("viewBox unclosed"))?;
    let value = &html[at..at + len];
    let numbers: Vec<f64> = value
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .with_context(|| format!("viewBox \"{value}\""))?;
    match numbers.as_slice() {
        [_, _, w, h] if *w > 0.0 && *h > 0.0 => Ok((*w, *h)),
        _ => Err(anyhow!("viewBox \"{value}\" is not four sizes")),
    }
}

/// A sheet holding one figure and nothing else, its page exactly the figure's
/// size, styled by the site's own stylesheets so it prints as the page shows it.
pub fn sheet(fig: &Figure, stylesheets: &[String]) -> String {
    let links: String = stylesheets
        .iter()
        .map(|href| format!("<link rel=\"stylesheet\" href=\"{href}\">"))
        .collect();
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\">{links}<style>\
         @page{{size:{w}px {h}px;margin:0}}\
         html,body{{margin:0;padding:0;width:{w}px;height:{h}px;overflow:hidden;background:#fff}}\
         .prose.paper{{margin:0;padding:0;border:none;max-width:none}}\
         .prose .fig{{margin:0}}\
         .prose.paper .fig svg{{border:none}}\
         .prose.paper>figure::before{{display:none}}\
         </style></head><body class=\"tokera\"><article class=\"prose paper\">{html}</article></body></html>",
        w = fig.width,
        h = fig.height,
        html = fig.html,
    )
}

/// `path` as a `file:` URL Chrome opens, with Windows' verbatim prefix dropped.
pub fn file_url(path: &Path) -> String {
    let s = path.to_string_lossy().replace('\\', "/");
    let s = s.strip_prefix("//?/").unwrap_or(&s);
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const PAPER: &str = "Intro.\n\n<figure class=\"fig\">\n<svg viewBox=\"0 0 640 292\" role=\"img\">\n  <text>A</text>\n</svg>\n</figure>\n\nMiddle.\n\n<figure class=\"fig\">\n<svg viewBox=\"0 0 640 214.5\"></svg>\n</figure>\n\nEnd.\n";

    #[test]
    fn each_figure_becomes_an_image_of_its_own_file() {
        let l = lift(PAPER, |n| format!("fig-{n:02}.pdf")).unwrap();
        assert_eq!(
            l.markdown,
            "Intro.\n\n![](fig-01.pdf){width=100%}\n\nMiddle.\n\n![](fig-02.pdf){width=100%}\n\nEnd.\n"
        );
        assert_eq!(l.figures.len(), 2);
        assert_eq!(
            l.figures[0].html,
            "<figure class=\"fig\">\n<svg viewBox=\"0 0 640 292\" role=\"img\">\n  <text>A</text>\n</svg>\n</figure>"
        );
        assert_eq!((l.figures[1].width, l.figures[1].height), (640.0, 214.5));
    }

    #[test]
    fn a_paper_with_no_figures_passes_through() {
        let l = lift("Just text.\n", |_| unreachable!()).unwrap();
        assert_eq!(l.markdown, "Just text.\n");
        assert!(l.figures.is_empty());
    }

    #[test]
    fn a_figure_that_never_closes_or_has_no_size_is_refused() {
        let none = |_: usize| String::new();
        assert!(lift("<figure class=\"fig\"><svg viewBox=\"0 0 1 1\">", none).is_err());
        assert!(lift("<figure class=\"fig\"><svg></svg></figure>", none).is_err());
        assert!(view_box("<svg viewBox=\"0 0 640\">").is_err());
        assert!(view_box("<svg viewBox=\"0 0 0 10\">").is_err());
    }

    #[test]
    fn the_sheet_is_the_figures_size_and_links_the_site_styles() {
        let fig = Figure {
            html: "<figure class=\"fig\"><svg/></figure>".into(),
            width: 640.0,
            height: 292.0,
        };
        let styles = ["file:///D:/a/base.css", "file:///D:/a/site.css"].map(String::from);
        let s = sheet(&fig, &styles);
        assert!(s.starts_with(
            "<!doctype html><html><head><meta charset=\"utf-8\">\
             <link rel=\"stylesheet\" href=\"file:///D:/a/base.css\">\
             <link rel=\"stylesheet\" href=\"file:///D:/a/site.css\"><style>\
             @page{size:640px 292px;margin:0}\
             html,body{margin:0;padding:0;width:640px;height:292px;"
        ));
        assert!(s.ends_with(
            "<body class=\"tokera\"><article class=\"prose paper\">\
             <figure class=\"fig\"><svg/></figure></article></body></html>"
        ));
    }

    #[test]
    fn a_windows_path_becomes_a_file_url() {
        assert_eq!(
            file_url(&PathBuf::from(r"\\?\D:\prog\candle\web\site.css")),
            "file:///D:/prog/candle/web/site.css"
        );
        let url = |p: &str| file_url(&PathBuf::from(p));
        assert_eq!(url(r"D:\x\y.html"), "file:///D:/x/y.html");
        assert_eq!(url("/home/x.html"), "file:///home/x.html");
    }
}
