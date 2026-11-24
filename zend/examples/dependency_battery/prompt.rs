//! What each arm's question is asked as. The memory arms and A0/A1 carry the
//! pilot's instruction to answer from memory alone; R1 carries the files its
//! retrieval packed ahead of the question and is told to answer from them;
//! T1 is told to read the repository with its tools.

use crate::arms::Arm;

/// The memory arms' instruction: the pilot's.
pub const PREFIX: &str = "Answer from your memory of the repository only — do not call any \
     tool, do not read or search any file. If you do not remember, say so.";

/// R1's instruction, after the files its retrieval packed.
pub const RETRIEVED_PREFIX: &str = "Answer from the files above and what you know of the \
     repository — call no tool. If you do not know, say so.";

/// T1's instruction.
pub const TOOLS_PREFIX: &str = "Use the tools to read the repository as needed, then answer. \
     If you cannot find it, say so.";

/// `question` as `arm` asks it, R1's after `files`, its packed files block.
pub fn prompt(arm: Arm, question: &str, files: &str) -> String {
    match arm {
        Arm::R1 => format!("{files}{RETRIEVED_PREFIX}\n\n{question}"),
        Arm::T1 => format!("{TOOLS_PREFIX}\n\n{question}"),
        _ => format!("{PREFIX}\n\n{question}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Each arm asks the same question under its own instruction**, and only
    /// R1 carries files.
    #[test]
    fn each_arm_asks_under_its_instruction() {
        let files = "=== a.rs ===\nfn x() {}\n\n";
        assert_eq!(
            prompt(Arm::C1, "Where?", files),
            "Answer from your memory of the repository only — do not call any tool, do not \
             read or search any file. If you do not remember, say so.\n\nWhere?"
        );
        assert_eq!(
            prompt(Arm::R1, "Where?", files),
            "=== a.rs ===\nfn x() {}\n\nAnswer from the files above and what you know of the \
             repository — call no tool. If you do not know, say so.\n\nWhere?"
        );
        assert_eq!(
            prompt(Arm::T1, "Where?", files),
            "Use the tools to read the repository as needed, then answer. If you cannot find \
             it, say so.\n\nWhere?"
        );
    }
}
