//! The inline SVG every figure is drawn as: a `<figure class="fig">` holding
//! one `<svg>` 640 wide, its colours only the site's chart classes
//! (`web/content/tokera/site.css`, *Charts of measured results*), so a figure
//! follows the page's theme and carries none of its own.
//!
//! A snippet has no blank line anywhere inside it: markdown ends an HTML block
//! at the first blank line, so one would cut the figure off mid-SVG.

use crate::figures::text::{esc, n};

/// The viewBox width of every figure.
pub const WIDTH: f64 = 640.0;

/// A linear map from a domain onto a range of SVG coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Linear {
    d0: f64,
    d1: f64,
    r0: f64,
    r1: f64,
}

impl Linear {
    pub fn new((d0, d1): (f64, f64), (r0, r1): (f64, f64)) -> Self {
        Self { d0, d1, r0, r1 }
    }

    pub fn at(&self, x: f64) -> f64 {
        self.r0 + (x - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)
    }
}

/// A logarithmic map: linear in `ln x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Log(Linear);

impl Log {
    pub fn new((d0, d1): (f64, f64), range: (f64, f64)) -> Self {
        Self(Linear::new((d0.ln(), d1.ln()), range))
    }

    pub fn at(&self, x: f64) -> f64 {
        self.0.at(x.ln())
    }
}

/// Where a text's `x` sits on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    Start,
    /// Through the site's `mid` class.
    Middle,
    End,
}

/// A legend entry's mark: a filled square or a short stroke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swatch {
    Fill(&'static str),
    Line(&'static str),
}

/// One figure being drawn.
pub struct Svg {
    height: f64,
    aria: String,
    body: Vec<String>,
}

fn faded(opacity: Option<f64>) -> String {
    opacity
        .map(|o| format!(" opacity=\"{o}\""))
        .unwrap_or_default()
}

impl Svg {
    /// A figure `height` tall, described to a screen reader by `aria`, with
    /// its title and subtitle drawn at its top left.
    pub fn new(height: f64, aria: &str, title: &str, subtitle: &str) -> Self {
        let mut svg = Self {
            height,
            aria: aria.to_string(),
            body: Vec::new(),
        };
        svg.text("ttl", 16.0, 20.0, Anchor::Start, title);
        svg.text("ttl-sub", 16.0, 38.0, Anchor::Start, subtitle);
        svg
    }

    pub fn text(&mut self, class: &str, x: f64, y: f64, anchor: Anchor, s: &str) {
        let (class, attr) = match anchor {
            Anchor::Start => (class.to_string(), ""),
            Anchor::Middle => (format!("{class} mid").trim().to_string(), ""),
            Anchor::End => (class.to_string(), " text-anchor=\"end\""),
        };
        let class = if class.is_empty() {
            String::new()
        } else {
            format!(" class=\"{class}\"")
        };
        self.body.push(format!(
            "<text{class} x=\"{}\" y=\"{}\"{attr}>{}</text>",
            n(x),
            n(y),
            esc(s)
        ));
    }

    pub fn path(&mut self, class: &str, d: &str, opacity: Option<f64>) {
        self.body.push(format!(
            "<path class=\"{class}\"{} d=\"{d}\"/>",
            faded(opacity)
        ));
    }

    pub fn circle(&mut self, class: &str, cx: f64, cy: f64, r: f64, opacity: Option<f64>) {
        self.body.push(format!(
            "<circle class=\"{class}\"{} cx=\"{}\" cy=\"{}\" r=\"{}\"/>",
            faded(opacity),
            n(cx),
            n(cy),
            n(r)
        ));
    }

    pub fn rect(&mut self, class: &str, x: f64, y: f64, w: f64, h: f64, opacity: Option<f64>) {
        self.body.push(format!(
            "<rect class=\"{class}\"{} x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/>",
            faded(opacity),
            n(x),
            n(y),
            n(w),
            n(h)
        ));
    }

    /// Horizontal grid lines across `x0..x1` at each tick of `y`, labelled
    /// at the left; the line at `base` drawn as the axis.
    pub fn y_grid(&mut self, y: &Linear, ticks: &[(f64, String)], (x0, x1): (f64, f64), base: f64) {
        let d: Vec<String> = ticks
            .iter()
            .filter(|(v, _)| *v != base)
            .map(|(v, _)| format!("M{} {} H{}", n(x0), n(y.at(*v)), n(x1)))
            .collect();
        if !d.is_empty() {
            self.path("grid", &d.join(" "), None);
        }
        for (v, label) in ticks {
            self.text("tick", x0 - 8.0, y.at(*v) + 3.5, Anchor::End, label);
        }
        self.path(
            "grid base",
            &format!("M{} {} H{}", n(x0), n(y.at(base)), n(x1)),
            None,
        );
    }

    /// Vertical grid lines across `y0..y1` at each `(x, label)`, labelled
    /// beneath at `label_y`.
    pub fn x_grid(&mut self, xs: &[(f64, String)], (y0, y1): (f64, f64), label_y: f64) {
        let d: Vec<String> = xs
            .iter()
            .map(|(x, _)| format!("M{} {} V{}", n(*x), n(y0), n(y1)))
            .collect();
        self.path("grid", &d.join(" "), None);
        for (x, label) in xs {
            self.text("tick", *x, label_y, Anchor::Middle, label);
        }
    }

    /// A legend in one row from `(x, y)`, each entry its swatch then its
    /// label, spaced by the label's length.
    pub fn legend(&mut self, x: f64, y: f64, items: &[(Swatch, &str)]) {
        self.body.push("<g class=\"legend\">".into());
        let mut at = x;
        for (swatch, label) in items {
            match swatch {
                Swatch::Fill(class) => self.rect(class, at, y - 8.0, 10.0, 10.0, None),
                Swatch::Line(class) => self.path(
                    class,
                    &format!("M{} {} H{}", n(at), n(y - 3.0), n(at + 14.0)),
                    None,
                ),
            }
            self.text("", at + 18.0, y, Anchor::Start, label);
            at += 18.0 + 6.2 * label.chars().count() as f64 + 16.0;
        }
        self.body.push("</g>".into());
    }

    /// The snippet: no blank line inside it, ending in a newline.
    pub fn finish(self) -> String {
        let mut out = format!(
            "<figure class=\"fig\">\n<svg viewBox=\"0 0 {} {}\" role=\"img\" aria-label=\"{}\">\n",
            n(WIDTH),
            n(self.height),
            esc(&self.aria)
        );
        for line in self.body {
            out.push_str("  ");
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("</svg>\n</figure>\n");
        out
    }
}

/// How finely a path's coordinates are rounded: a tenth of a unit, or a
/// whole one — the dense series over time take whole units, which a 640-wide
/// figure cannot tell apart from tenths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Tenth,
    Whole,
}

impl Precision {
    fn scale(self) -> f64 {
        match self {
            Precision::Tenth => 10.0,
            Precision::Whole => 1.0,
        }
    }
}

/// The path data through `points` in the shortest absolute form: one `M`,
/// the rest implicit line-tos, each coordinate rounded to `precision`, and
/// every point dropped that lies on the straight line between its
/// neighbours once rounded — which loses nothing the rounding kept.
fn path_data(points: &[(f64, f64)], precision: Precision) -> String {
    let scale = precision.scale();
    let mut units: Vec<(i64, i64)> = Vec::with_capacity(points.len());
    for (x, y) in points {
        let p = ((x * scale).round() as i64, (y * scale).round() as i64);
        if units.last() == Some(&p) {
            continue;
        }
        if units.len() >= 2 {
            let (a, b) = (units[units.len() - 2], units[units.len() - 1]);
            let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            let onward = (b.0 - a.0) * (p.0 - b.0) + (b.1 - a.1) * (p.1 - b.1) >= 0;
            if cross == 0 && onward {
                units.pop();
            }
        }
        units.push(p);
    }
    let coords: Vec<String> = units
        .iter()
        .map(|(x, y)| format!("{} {}", n(*x as f64 / scale), n(*y as f64 / scale)))
        .collect();
    format!("M{}", coords.join(" "))
}

/// A polyline through `points`.
pub fn line_path(points: &[(f64, f64)], precision: Precision) -> String {
    path_data(points, precision)
}

/// The closed region between `top`, left to right, and `bottom`, drawn back
/// right to left.
pub fn area_path(top: &[(f64, f64)], bottom: &[(f64, f64)], precision: Precision) -> String {
    let mut ring: Vec<(f64, f64)> = top.to_vec();
    ring.extend(bottom.iter().rev());
    format!("{} Z", path_data(&ring, precision))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_map_their_domain_onto_their_range() {
        let y = Linear::new((0.0, 0.5), (250.0, 50.0));
        assert_eq!((y.at(0.0), y.at(0.25), y.at(0.5)), (250.0, 150.0, 50.0));
        let x = Log::new((1.0, 100.0), (70.0, 610.0));
        assert_eq!(
            (n(x.at(1.0)), n(x.at(10.0)), n(x.at(100.0))),
            ("70".into(), "340".into(), "610".into())
        );
    }

    #[test]
    fn paths_join_their_points() {
        assert_eq!(
            line_path(&[(1.0, 2.0), (3.25, 4.0)], Precision::Tenth),
            "M1 2 3.3 4"
        );
        assert_eq!(
            area_path(
                &[(0.0, 1.0), (2.0, 1.0)],
                &[(0.0, 5.0), (2.0, 5.0)],
                Precision::Tenth
            ),
            "M0 1 2 1 2 5 0 5 Z"
        );
    }

    /// Rounded to whole units, 2.4 and 2.6 land on 2 and 3; the points of a
    /// straight run and a repeated point are dropped, a turn back is kept.
    #[test]
    fn whole_paths_round_and_drop_the_points_a_straight_line_passes() {
        assert_eq!(
            line_path(
                &[
                    (0.0, 10.0),
                    (5.0, 10.4),
                    (10.0, 9.6),
                    (15.0, 12.6),
                    (15.2, 12.6),
                    (20.0, 14.0),
                    (25.0, 16.0),
                ],
                Precision::Whole
            ),
            "M0 10 10 10 15 13 20 14 25 16"
        );
        assert_eq!(
            line_path(&[(0.0, 0.0), (10.0, 0.0), (5.0, 0.0)], Precision::Whole),
            "M0 0 10 0 5 0"
        );
    }

    #[test]
    fn a_figure_is_its_frame_and_its_elements_with_no_blank_line() {
        let mut s = Svg::new(100.0, "A \"test\" & more", "Title", "sub <1>");
        s.circle("dot-us", 10.0, 20.04, 6.0, None);
        s.rect("them", 1.0, 2.0, 3.0, 4.0, Some(0.35));
        s.path("parity", "M0 0 V10", None);
        s.text("v-us", 5.0, 6.0, Anchor::Middle, "37.2%");
        s.text("tick", 5.0, 6.0, Anchor::End, "0%");
        assert_eq!(
            s.finish(),
            "<figure class=\"fig\">\n\
             <svg viewBox=\"0 0 640 100\" role=\"img\" aria-label=\"A &quot;test&quot; &amp; more\">\n  \
             <text class=\"ttl\" x=\"16\" y=\"20\">Title</text>\n  \
             <text class=\"ttl-sub\" x=\"16\" y=\"38\">sub &lt;1&gt;</text>\n  \
             <circle class=\"dot-us\" cx=\"10\" cy=\"20\" r=\"6\"/>\n  \
             <rect class=\"them\" opacity=\"0.35\" x=\"1\" y=\"2\" width=\"3\" height=\"4\"/>\n  \
             <path class=\"parity\" d=\"M0 0 V10\"/>\n  \
             <text class=\"v-us mid\" x=\"5\" y=\"6\">37.2%</text>\n  \
             <text class=\"tick\" x=\"5\" y=\"6\" text-anchor=\"end\">0%</text>\n\
             </svg>\n</figure>\n"
        );
    }

    #[test]
    fn grids_and_legends_draw_their_ticks_and_entries() {
        let mut s = Svg::new(100.0, "", "", "");
        let y = Linear::new((0.0, 1.0), (90.0, 50.0));
        s.y_grid(
            &y,
            &[(0.0, "0".into()), (1.0, "1".into())],
            (60.0, 600.0),
            0.0,
        );
        s.x_grid(&[(100.0, "a".into())], (50.0, 90.0), 104.0);
        s.legend(
            400.0,
            30.0,
            &[(Swatch::Fill("q1"), "ab"), (Swatch::Line("line-us"), "c")],
        );
        let out = s.finish();
        let body: Vec<&str> = out.lines().skip(4).map(str::trim).collect();
        assert_eq!(
            body,
            vec![
                "<path class=\"grid\" d=\"M60 50 H600\"/>",
                "<text class=\"tick\" x=\"52\" y=\"93.5\" text-anchor=\"end\">0</text>",
                "<text class=\"tick\" x=\"52\" y=\"53.5\" text-anchor=\"end\">1</text>",
                "<path class=\"grid base\" d=\"M60 90 H600\"/>",
                "<path class=\"grid\" d=\"M100 50 V90\"/>",
                "<text class=\"tick mid\" x=\"100\" y=\"104\">a</text>",
                "<g class=\"legend\">",
                "<rect class=\"q1\" x=\"400\" y=\"22\" width=\"10\" height=\"10\"/>",
                "<text x=\"418\" y=\"30\">ab</text>",
                "<path class=\"line-us\" d=\"M446.4 27 H460.4\"/>",
                "<text x=\"464.4\" y=\"30\">c</text>",
                "</g>",
                "</svg>",
                "</figure>",
            ]
        );
    }
}
