//! The degradation script: five questions, the last of which has the model read
//! a ~44k-token paper and give its view of it, run as fresh conversations on a
//! real model — with a verdict on whether each conversation's final turn
//! answers that last question.
//!
//! Live, Qwen3.8-Flash-Next answered an EARLIER question once the paper was
//! read: the square root (085044ec, 515f45ab) or the repo (a40740b1). By then
//! the context is ~52k tokens and QSA reads 2,051 of those cells per query, so
//! the arms separate the selection, the model and the pipeline:
//!
//! - `flash_next_selected` — production: Flash-Next on the checkpoint's own QSA
//!   budget.
//! - `flash_next_dense` — the same engine with the budget past the depth, so the
//!   selection is the identity and every cell is read. Passing where `selected`
//!   fails puts the fault in the selection.
//! - `qwen36` — Qwen3.6-35B-A3B: the same hybrid lineage with full attention and
//!   no QSA. Passing says zend delivers the question to the model.
//!
//! Each arm runs on a workspace of its own model under `target/tmp` — the
//! substrate's K/V is that model's, and the tool catalog's calibration is paid
//! once per model — with `repo_map` and `code_reading` out of service, and the
//! two files the script reads (`README.md`, `docs/unbounded_agents.md`) copied
//! in. The live repo's substrate is never opened.
//!
//! An arm boots a daemon, needs the whole card, and runs eight conversations
//! (~30–40 min). Run one by name, zend stopped:
//!
//! ```text
//! cargo test -p zend --features cuda --test degradation_script -- --ignored --nocapture --exact arms::flash_next_selected
//! ```

mod verdict;

#[cfg(feature = "cuda")]
#[path = "../common/mod.rs"]
mod common;
#[cfg(feature = "cuda")]
mod rig;

#[cfg(feature = "cuda")]
mod arms {
    use candle_conversation::models::Model;

    use crate::rig::{run_arm, Arm, DENSE_BUDGET};

    /// Flash-Next's workspace, shared by its two arms: they run one model.
    const FLASH_NEXT_WS: &str = "degradation_flash_next_ws";

    #[test]
    #[ignore = "boots Qwen3.8-Flash-Next and runs eight ~52k-token conversations on the whole \
                card (~40 min); run by name with zend stopped"]
    fn flash_next_selected() {
        run_arm(&Arm {
            name: "flash_next_selected",
            model: Model::Qwen38_FlashNext_Q4KO,
            workspace: FLASH_NEXT_WS,
            qsa_selection_budget: None,
        });
    }

    #[test]
    #[ignore = "boots Qwen3.8-Flash-Next reading every cell and runs eight ~52k-token \
                conversations on the whole card (~40 min); run by name with zend stopped"]
    fn flash_next_dense() {
        run_arm(&Arm {
            name: "flash_next_dense",
            model: Model::Qwen38_FlashNext_Q4KO,
            workspace: FLASH_NEXT_WS,
            qsa_selection_budget: Some(DENSE_BUDGET),
        });
    }

    #[test]
    #[ignore = "boots Qwen3.6-35B-A3B and runs eight ~52k-token conversations on the whole card \
                (~40 min); run by name with zend stopped"]
    fn qwen36() {
        run_arm(&Arm {
            name: "qwen36",
            model: Model::Qwen36_35B_A3B_Q4,
            workspace: "degradation_qwen36_ws",
            qsa_selection_budget: None,
        });
    }
}
