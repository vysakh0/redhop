//! `neorag` — a thin evaluation & observability CLI for NeoRAG.
//!
//! NeoRAG is a reasoning-preserving context optimization library; this CLI is
//! a thin, Unix-like shell over its public API (`build_context`,
//! `analyze_context`, `context_economics`). It exists for evaluation,
//! observability, benchmarking, reproducibility, and context inspection — not
//! as a serving runtime, workflow engine, or orchestration layer.

mod analyze;
mod benchmark;
mod compare;
mod io;
mod report;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "neorag",
    about = "Reasoning-preserving context optimization — eval & observability CLI",
    long_about = "Thin CLI over the NeoRAG context API. Compare strategies, inspect \
                  context economics, run reproducible benchmarks, and render reports."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compare context strategies side-by-side on one retrieval set.
    Compare(compare::Args),
    /// Inspect one retrieved/assembled context (observability & debugging).
    AnalyzeContext(analyze::Args),
    /// Run a reproducible strategy benchmark over a labeled dataset.
    Benchmark(benchmark::Args),
    /// Render a benchmark/compare JSON artifact to markdown / HTML.
    Report(report::Args),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Compare(a) => compare::run(a),
        Command::AnalyzeContext(a) => analyze::run(a),
        Command::Benchmark(a) => benchmark::run(a),
        Command::Report(a) => report::run(a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }
}
