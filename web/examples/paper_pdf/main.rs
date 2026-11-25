//! Typesets a published paper as a LaTeX PDF.
//!
//! The paper's markdown is the source the site renders; this prints the same
//! source the way a paper is printed. Each figure is lifted out and printed by
//! Chrome to a vector PDF of its own, styled by the site's stylesheets
//! (`figures`); pandoc turns the rest into LaTeX with pandoc's template and a
//! header of the paper's own (`latex`); tectonic sets it.
//!
//! ```text
//! cargo run -p web --example paper_pdf -- --slug one-card-v2 \
//!     --out web/content/tokera/pdf/one-card-v2.pdf \
//!     --pandoc C:\tools\pandoc.exe --tectonic C:\tools\tectonic.exe
//! ```

mod figures;
mod latex;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context};
use clap::Parser;
use web::site::tokera::papers::Manifest;

use figures::{file_url, lift, sheet};

const CHROME: &str = r"C:\Program Files\Google\Chrome\Application\chrome.exe";

#[derive(Parser)]
struct Args {
    /// The paper's slug in the manifest.
    #[arg(long)]
    slug: String,
    /// Where the typeset PDF is written.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "web/content/tokera/papers.yaml")]
    manifest: PathBuf,
    /// The directory the manifest's `source` names are read from.
    #[arg(long, default_value = "docs")]
    papers: PathBuf,
    /// The site's stylesheets the figures are drawn with, in the page's order.
    #[arg(long, default_values = ["web/content/common/base.css", "web/content/tokera/site.css"])]
    stylesheet: Vec<PathBuf>,
    #[arg(long, default_value = "target/paper_pdf")]
    work: PathBuf,
    #[arg(long, default_value = "pandoc")]
    pandoc: PathBuf,
    #[arg(long, default_value = "tectonic")]
    tectonic: PathBuf,
    #[arg(long, default_value = CHROME)]
    chrome: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let manifest = Manifest::parse(&fs::read_to_string(&args.manifest).context("manifest")?)?;
    let paper = manifest
        .papers
        .iter()
        .find(|p| p.slug == args.slug)
        .ok_or_else(|| anyhow!("no paper `{}` in {}", args.slug, args.manifest.display()))?;
    let source = fs::read_to_string(args.papers.join(&paper.source))
        .with_context(|| format!("reading {}", paper.source))?;

    fs::create_dir_all(&args.work)?;
    let work = absolute(&args.work)?;
    let lifted = lift(&source, |n| format!("fig-{n:02}.pdf"))?;

    let styles: Vec<String> = args
        .stylesheet
        .iter()
        .map(|p| absolute(p).map(|p| file_url(&p)))
        .collect::<anyhow::Result<_>>()?;
    for (i, fig) in lifted.figures.iter().enumerate() {
        let html = work.join(format!("fig-{:02}.html", i + 1));
        let pdf = work.join(format!("fig-{:02}.pdf", i + 1));
        fs::write(&html, sheet(fig, &styles))?;
        run(Command::new(&args.chrome)
            .arg("--headless=new")
            .arg("--disable-gpu")
            .arg("--no-pdf-header-footer")
            .arg("--allow-file-access-from-files")
            .arg(format!("--print-to-pdf={}", pdf.display()))
            .arg(file_url(&html)))
        .with_context(|| format!("printing figure {}", i + 1))?;
    }
    println!("{} figures printed", lifted.figures.len());

    fs::write(work.join("paper.md"), &lifted.markdown)?;
    let header = latex::header(&paper.title, &paper.status);
    fs::write(work.join("header.tex"), header)?;
    let mut pandoc = Command::new(&args.pandoc);
    pandoc.current_dir(&work).args(PANDOC);
    for v in VARIABLES {
        pandoc.args(["-V", v]);
    }
    if !paper.authors.is_empty() {
        pandoc.args(["-M", &format!("author={}", paper.authors)]);
    }
    if !paper.date.is_empty() {
        pandoc.args(["-M", &format!("date={}", paper.date)]);
    }
    run(&mut pandoc).context("pandoc")?;

    run(Command::new(&args.tectonic)
        .current_dir(&work)
        .args(["--keep-logs", "paper.tex"]))
    .context("tectonic")?;
    let missing = latex::missing_glyphs(&fs::read_to_string(work.join("paper.log"))?);
    if !missing.is_empty() {
        bail!(
            "the fonts have no glyph for {}; map each in `latex::header`",
            missing.join(", ")
        );
    }

    if let Some(dir) = args.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    fs::copy(work.join("paper.pdf"), &args.out)?;
    println!("{}", args.out.display());
    Ok(())
}

/// The markdown in, the LaTeX out. The document's `# H1` is its title, so every
/// heading moves up a level: `##` is a section.
const PANDOC: [&str; 10] = [
    "paper.md",
    "--from",
    "markdown+autolink_bare_uris",
    "--to",
    "latex",
    "--standalone",
    "--shift-heading-level-by=-1",
    "--include-in-header=header.tex",
    "-o",
    "paper.tex",
];

/// Pandoc's template variables: a 10 pt article on US Letter, set in Times.
const VARIABLES: [&str; 8] = [
    "documentclass=article",
    "fontsize=10pt",
    "papersize=letter",
    "geometry=top=1in,bottom=1in,left=1.2in,right=1.2in,headheight=13pt,headsep=14pt",
    "mainfont=Times New Roman",
    "mathfont=Cambria Math",
    "monofont=Consolas",
    "monofontoptions=Scale=MatchLowercase",
];

fn run(cmd: &mut Command) -> anyhow::Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("starting {:?}", cmd.get_program()))?;
    if !status.success() {
        bail!("{:?} exited with {status}", cmd.get_program());
    }
    Ok(())
}

fn absolute(p: &Path) -> anyhow::Result<PathBuf> {
    fs::canonicalize(p).with_context(|| format!("resolving {}", p.display()))
}
