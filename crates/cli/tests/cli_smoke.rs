//! CLI integration smoke. Exercises the actual `redhop` binary end-to-end
//! for each subcommand, asserting it exits cleanly and emits the expected
//! shape. Caught nothing on its first run but it's the safety net against
//! "subcommand silently broke on the upgrade and only the user noticed".
//!
//! `cargo test` picks this up automatically (it's a `tests/` integration
//! test); the test asks cargo for the path to the built binary so a fresh
//! checkout works without the user installing the binary first.

use std::io::Write;
use std::process::{Command, Stdio};

/// Path to the freshly-built `redhop` binary from cargo's CARGO_BIN_EXE
/// magic. Set automatically by cargo for `tests/` files; no toolchain
/// dance.
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_redhop")
}

#[test]
fn cli_help_works() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("run redhop --help");
    assert!(out.status.success(), "exit code: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Each subcommand should be listed in the top-level help.
    for sub in &["compare", "analyze-context", "benchmark", "report"] {
        assert!(
            stdout.contains(sub),
            "redhop --help should list `{sub}` subcommand; got:\n{stdout}"
        );
    }
}

#[test]
fn cli_subcommand_helps_work() {
    // Every subcommand must have its own --help (clap autogenerates this,
    // but a regression in how the subcommand is wired would break it).
    for sub in &["compare", "analyze-context", "benchmark", "report"] {
        let out = Command::new(bin())
            .args([sub, "--help"])
            .output()
            .unwrap_or_else(|e| panic!("run redhop {sub} --help: {e}"));
        assert!(
            out.status.success(),
            "redhop {sub} --help exited non-zero: {:?}",
            out.status
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.to_lowercase().contains("usage"),
            "redhop {sub} --help should print 'Usage:' header; got:\n{stdout}"
        );
    }
}

#[test]
fn cli_analyze_context_reads_stdin_json() {
    // `redhop analyze-context -` reads a JSON `{query, chunks}` from
    // stdin and prints the Decision Report. This is the primary
    // observability entry point the CLI ships.
    let input = r#"{
        "query": "refund window",
        "chunks": [
            {"id": "g1", "text": "The refund window is thirty days from purchase."},
            {"id": "d1", "text": "Photosynthesis converts sunlight into glucose."}
        ]
    }"#;

    let mut child = Command::new(bin())
        .args(["analyze-context", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn redhop analyze-context");

    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(input.as_bytes())
        .expect("write stdin");

    let out = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "analyze-context exited non-zero ({:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status
    );
    // Report shape: the rendered Decision Report carries these labels.
    for marker in &["Decision Report", "Retrieved tokens", "Final tokens"] {
        assert!(
            stdout.contains(marker),
            "analyze-context stdout should contain `{marker}`; got:\n{stdout}"
        );
    }
}
