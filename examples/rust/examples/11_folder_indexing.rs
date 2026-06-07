//! 11 · Folder indexing — `redhop::read_folder_with(path, &fo)` with
//!      .gitignore, ignore globs, and incremental persistent on-disk
//!      index.
//!
//! Real-world scenario:
//!     An engineering team has a `docs/` directory with mixed Markdown,
//!     code samples, and the occasional vendored upstream file they don't
//!     want indexed. They want:
//!       - One combined index over all readable files.
//!       - `.gitignore` honored automatically.
//!       - Custom `ignore` globs for the vendored-but-not-gitignored files.
//!       - `persist=true` so the second invocation skips re-indexing
//!         unchanged files (incremental on-disk cache).
//!
//! What this demonstrates:
//!     - `redhop::read_folder_with(path, &FolderOptions { ... })` —
//!       recursive indexing.
//!     - `recursive: Some(false)` for flat indexing.
//!     - `gitignore: Some(true)` (default).
//!     - `ignore: vec!["pattern", ...]` — extra globs.
//!     - `persist: true` — incremental on-disk index.
//!     - `doc.n_files()` / `doc.skipped_files()` — observability.
//!     - `redhop::read_bytes_with(data, source, &opts)` — for blobs.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 11_folder_indexing --release

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use redhop::{citations, read_bytes_with, read_folder_with, FolderOptions, LoadOptions};
use tempfile::TempDir;

fn setup_demo_docs(root: &Path) -> anyhow::Result<()> {
    fs::write(
        root.join("README.md"),
        "# Acme Inc Engineering Handbook\n\nWelcome. Start with onboarding.md for new hires.\n",
    )?;
    fs::write(
        root.join("onboarding.md"),
        "# Onboarding\n\nNew hires get a laptop on day 1 and access provisioned in 24 hours.\nTalk to it@acme.com if something is missing.\n",
    )?;

    fs::create_dir(root.join("policies"))?;
    fs::write(
        root.join("policies/refunds.md"),
        "# Refund Policy\n\nCustomers get a full refund within 30 days of delivery.\n",
    )?;
    fs::write(
        root.join("policies/shipping.md"),
        "# Shipping Policy\n\nStandard ships in 3-5 business days. Express in 1-2.\n",
    )?;

    // Vendored upstream file we want to ignore.
    fs::create_dir(root.join("vendored"))?;
    let mut vendored = fs::File::create(root.join("vendored/third_party_license.md"))?;
    writeln!(vendored, "# Apache 2.0 license text\n")?;
    for _ in 0..30 {
        writeln!(vendored, "IRRELEVANT BOILERPLATE")?;
    }

    // .gitignore that excludes build/.
    fs::create_dir(root.join("build"))?;
    fs::write(root.join("build/generated.md"), "# generated, ignore me\n")?;
    fs::write(root.join(".gitignore"), "build/\n")?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let tmp = TempDir::with_prefix("redhop-folder-demo-")?;
    let root = tmp.path();
    setup_demo_docs(root)?;
    println!("Demo directory: {}\n", root.display());

    // ── Arm A: vanilla recursive index ─────────────────────────
    // `.gitignore` is honored by default — `build/` is skipped.
    // `vendored/third_party_license.md` IS indexed (no rule for it).
    println!("─── Arm A · read_folder_with(path, &default) ──");
    let mut doc_a = read_folder_with(root, &FolderOptions::default())?;
    println!("  files indexed   : {}", doc_a.n_files());
    println!("  total chunks    : {}", doc_a.chunks().len());
    println!("  files skipped   : {}", doc_a.skipped_files().len());
    for (path, reason) in doc_a.skipped_files().iter().take(3) {
        println!("    - {}: {}", path, reason);
    }
    println!();

    let ctx = doc_a.context("how long do I have to get a refund?")?;
    let cites = citations(&ctx);
    if let Some(top) = cites.first() {
        println!("  top hit source : {}", top.source);
        println!("  top hit heading: {:?}", top.heading);
    }
    println!();

    // ── Arm B: add custom ignore globs ────────────────────────
    println!("─── Arm B · ignore: [\"vendored/**\"] ───────────");
    let fo_b = FolderOptions {
        ignore: vec!["vendored/**".to_string()],
        ..FolderOptions::default()
    };
    let doc_b = read_folder_with(root, &fo_b)?;
    println!(
        "  files indexed   : {}  (vs Arm A: {})",
        doc_b.n_files(),
        doc_a.n_files()
    );
    println!("  total chunks    : {}", doc_b.chunks().len());
    println!();

    // ── Arm C: persist=true (incremental on-disk index) ───────
    println!("─── Arm C · persist: true ─────────────────────");
    let fo_persist = FolderOptions {
        persist: true,
        ..FolderOptions::default()
    };
    let t0 = Instant::now();
    let _ = read_folder_with(root, &fo_persist)?;
    let first_run_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let doc_c2 = read_folder_with(root, &fo_persist)?;
    let second_run_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let cache_path = root.join(".redhop/index.json");
    println!("  cache written   : {}", cache_path.exists());
    println!("  first  run      : {:>5.1} ms (cold)", first_run_ms);
    println!(
        "  second run      : {:>5.1} ms (warm — re-read cache)",
        second_run_ms
    );
    println!("  same n_files    : {}", doc_c2.n_files());
    println!();

    // ── Arm D: read_bytes_with (for S3 / GCS / DB blobs) ──────
    println!("─── Arm D · read_bytes_with (for S3 / GCS / blobs) ──");
    let data = fs::read(root.join("policies/refunds.md"))?;
    let mut doc_d = read_bytes_with(&data, "refunds.md", &LoadOptions::default())?;
    println!(
        "  indexed         : {} file, {} chunks",
        doc_d.n_files(),
        doc_d.chunks().len()
    );
    let ctx_d = doc_d.context("refund window")?;
    let cites_d = citations(&ctx_d);
    if let Some(c) = cites_d.first() {
        println!("  citation source : {}", c.source);
    }
    println!();

    println!("─── When to use what ─────────────────────────────");
    println!("- read_folder_with(path, &default)        : one combined");
    println!("  index over a directory. Default `recursive=true`,");
    println!("  `gitignore=true`.");
    println!("- ignore: vec![...]                       : add gitignore-");
    println!("  style globs.");
    println!("- persist: true                           : incremental cache.");
    println!("- read_bytes_with(data, \"source.pdf\", ...) : bytes from S3 /");
    println!("  GCS / a DB column.");
    Ok(())
}
