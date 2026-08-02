//! CLI entry point for three classic RL algorithms trained against a Python
//! Gym environment (via `gym_env`/`vec_gym_env` PyO3 bindings, not an LLM):
//! policy gradient (`pg`), DDPG (`ddpg`), and DQN (`dqn`), each in its own
//! sibling module.
//!
//! `Command` subcommand selects which algorithm's `run()` to invoke; no other
//! CLI flags. Requires a working Python/Gym install (`pyo3` feature paths in
//! `gym_env.rs`), unlike the rest of `candle-examples` which are pure Rust.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::Result;
use clap::{Parser, Subcommand};

mod gym_env;
mod vec_gym_env;

mod ddpg;
mod dqn;
mod policy_gradient;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Pg,
    Ddpg,
    Dqn,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Pg => policy_gradient::run()?,
        Command::Ddpg => ddpg::run()?,
        Command::Dqn => dqn::run()?,
    }
    Ok(())
}
