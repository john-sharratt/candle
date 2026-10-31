//! Every problem a loaded tree holds, in its files and its journals: what
//! `journals check` and the `journal_check` tool report.

use super::Tree;

impl Tree {
    /// Everything wrong with the tree, as one line each; empty when nothing is.
    pub fn all_problems(&self) -> Vec<String> {
        let mut out: Vec<String> = self.problems.iter().map(ToString::to_string).collect();
        for p in self.persons.values() {
            if let Err(e) = &p.file {
                out.push(e.to_string());
            }
            out.extend(p.problems.iter().map(ToString::to_string));
        }
        for f in self.groups.values() {
            if let Err(e) = &f.file {
                out.push(e.to_string());
            }
            out.extend(f.problems.iter().map(ToString::to_string));
        }
        for (_, j) in &self.journals() {
            out.extend(j.problems.iter().map(ToString::to_string));
        }
        out
    }
}
