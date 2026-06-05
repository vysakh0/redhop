//! Incremental persisted-folder loading: the cache hit/miss contract.
//!
//! `read_folder_with(FolderOptions { persist: true, .. })` writes an
//! `index.json` and on the next reload uses per-file `(mtime, size)` to
//! skip re-chunking unchanged files. The contract is load-bearing — it's
//! the difference between a cold rebuild (10s+ on a real codebase) and a
//! fast warm restart.
//!
//! There's no direct observable signal from the public API ("file X was
//! cache-hit"), so these tests inspect `index.json` on disk before and
//! after each reload to verify:
//!
//! 1. Per-file `(mtime, size)` for **unmodified** files is identical
//!    across reloads (proves the cache lookup matched).
//! 2. Per-file `(mtime, size)` for **modified** files changes (proves the
//!    re-chunk path ran).
//! 3. Persisted chunk content (text/source/metadata) for unmodified files
//!    is identical (proves chunks come from cache, not a fresh chunk
//!    pass). The chunk `id` field is excluded because the loader
//!    re-numbers IDs sequentially across the whole index on every rewrite
//!    — an implementation detail, not a cache-correctness signal.
//! 4. A no-op reload (nothing touched) leaves `index.json` itself
//!    untouched — the `changed = true` write gate works.
//! 5. A `LoadOptions` change flips the fingerprint and invalidates the
//!    whole cache, even with unchanged files (correct: chunking config
//!    drives chunk boundaries).
//! 6. A file deleted from disk drops from the cache on the next reload.

#![cfg(feature = "files")]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use redhop::{read_folder_with, FolderOptions, LoadOptions};

// ── Index.json inspection helpers ──────────────────────────────────────────
//
// `CachedFile` and `PersistedIndex` are private to the loader module, so we
// parse `index.json` as `serde_json::Value` and pluck out the fields we need
// to observe.

#[derive(Debug, Clone, PartialEq)]
struct FileEntry {
    mtime: u64,
    size: u64,
    /// Canonical JSON of the chunks array, with the `id` field stripped
    /// from each chunk. The loader re-numbers chunk IDs across the whole
    /// index every time it rewrites — so even a perfectly cache-hit
    /// chunk's `id` changes if neighboring files were added/removed.
    /// Stripping `id` lets us still assert that the load-bearing chunk
    /// content (text, source, metadata, token_count, embedding) survived
    /// unchanged.
    chunks_no_id: String,
}

/// Strip the `id` field from each chunk in a `chunks` array JSON value,
/// then re-serialize. Equality on the result means everything EXCEPT
/// re-numbered IDs matches — the tightest cache-content check the
/// implementation actually guarantees.
fn strip_chunk_ids(chunks: &serde_json::Value) -> String {
    let mut arr = chunks.as_array().cloned().unwrap_or_default();
    for c in &mut arr {
        if let Some(obj) = c.as_object_mut() {
            obj.remove("id");
        }
    }
    serde_json::to_string(&serde_json::Value::Array(arr)).unwrap()
}

fn read_index(dir: &Path) -> serde_json::Value {
    let path = dir.join(".redhop/index.json");
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("index.json missing at {}: {e}", path.display()));
    serde_json::from_str(&raw).expect("index.json must be valid JSON")
}

fn entries(idx: &serde_json::Value) -> std::collections::HashMap<String, FileEntry> {
    idx["files"]
        .as_array()
        .expect("index.json must have a `files` array")
        .iter()
        .map(|f| {
            let source = f["source"].as_str().unwrap().to_string();
            let mtime = f["mtime"].as_u64().unwrap();
            let size = f["size"].as_u64().unwrap();
            let chunks_no_id = strip_chunk_ids(&f["chunks"]);
            (
                source,
                FileEntry {
                    mtime,
                    size,
                    chunks_no_id,
                },
            )
        })
        .collect()
}

fn fingerprint(idx: &serde_json::Value) -> String {
    idx["fingerprint"].as_str().unwrap().to_string()
}

/// Find the cache entry whose source path ends with `suffix`. Sources are
/// stored as absolute paths so we can't compare by filename alone, but
/// suffix-match is unambiguous given unique test filenames.
fn entry_for<'a>(
    map: &'a std::collections::HashMap<String, FileEntry>,
    suffix: &str,
) -> &'a FileEntry {
    map.iter()
        .find(|(k, _)| k.ends_with(suffix))
        .map(|(_, v)| v)
        .unwrap_or_else(|| panic!("no cache entry ending in {suffix} (have: {:?})", map.keys()))
}

// ── Test scaffolding ───────────────────────────────────────────────────────

/// Stage 3 distinct text files in a fresh temp dir and return the dir
/// path. Caller cleans up. Filenames are unique-per-test to keep parallel
/// test runs isolated.
fn stage_three_files(prefix: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rh-persist-{prefix}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    // Three files, each with distinct content so chunks differ between
    // files (cache mismatch by accident would be very visible).
    fs::write(
        dir.join("alpha.txt"),
        "the quick brown fox jumps over the lazy dog",
    )
    .unwrap();
    fs::write(
        dir.join("beta.txt"),
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
    )
    .unwrap();
    fs::write(
        dir.join("gamma.txt"),
        "the rain in spain falls mainly on the plain",
    )
    .unwrap();
    dir
}

fn persist_opts() -> FolderOptions {
    FolderOptions {
        persist: true,
        ..Default::default()
    }
}

// ── 1. Incremental reload: only modified file is re-chunked ────────────────

#[test]
fn incremental_reload_only_re_chunks_modified_files() {
    let dir = stage_three_files("incr");
    let opts = persist_opts();

    // First load — populates the cache.
    let _doc1 = read_folder_with(&dir, &opts).expect("first read_folder_with");
    let idx1 = read_index(&dir);
    let map1 = entries(&idx1);
    assert_eq!(map1.len(), 3, "all 3 files cached on first load");

    // Sleep just enough that the filesystem mtime advances. APFS mtime
    // resolution is nanosecond on paper but the OS may round; 20ms is
    // belt-and-braces. Without this, "rewrite same byte length" could
    // produce a mtime collision and silently look like a cache hit even
    // though we changed the file.
    std::thread::sleep(Duration::from_millis(20));

    // Modify ONLY beta.txt with content of clearly different length
    // (the original was 54 bytes; this is much longer). Different byte
    // length means size differs → guaranteed cache miss regardless of
    // mtime resolution. Different content also gives different chunks,
    // so the post-reload chunks for beta will visibly differ.
    fs::write(
        dir.join("beta.txt"),
        "beta has been entirely rewritten with a much longer body that contains \
         many more words than before and clearly different content so the \
         re-chunk pass produces a visibly different chunk set than what was cached",
    )
    .unwrap();

    // Second load.
    let _doc2 = read_folder_with(&dir, &opts).expect("second read_folder_with");
    let idx2 = read_index(&dir);
    let map2 = entries(&idx2);
    assert_eq!(map2.len(), 3, "still 3 files after reload");

    // The two untouched files: every byte of their cache entry — including
    // mtime, size, and the serialized chunks — must match. This is the
    // tightest possible cache-hit assertion: a re-chunk pass would have
    // produced different chunks (different IDs at minimum) and would have
    // updated mtime+size.
    let alpha_before = entry_for(&map1, "alpha.txt");
    let alpha_after = entry_for(&map2, "alpha.txt");
    assert_eq!(
        alpha_before, alpha_after,
        "alpha.txt was untouched — its cache entry must be byte-identical \
         across the reload (proves cache-hit path; a re-chunk would have \
         changed mtime/size/chunks)"
    );

    let gamma_before = entry_for(&map1, "gamma.txt");
    let gamma_after = entry_for(&map2, "gamma.txt");
    assert_eq!(
        gamma_before, gamma_after,
        "gamma.txt was untouched — its cache entry must be byte-identical \
         across the reload"
    );

    // beta.txt MUST have a different cache entry. mtime, size, and chunks
    // should all have changed (we wrote different content with different
    // length). If beta's entry is identical, the loader missed our write.
    let beta_before = entry_for(&map1, "beta.txt");
    let beta_after = entry_for(&map2, "beta.txt");
    assert_ne!(
        beta_before.size, beta_after.size,
        "beta.txt was rewritten with a different byte length — size must \
         change. If size still matches, the cache didn't detect the modification."
    );
    assert_ne!(
        beta_before, beta_after,
        "beta.txt's full cache entry must differ — re-chunk path ran"
    );

    let _ = fs::remove_dir_all(&dir);
}

// ── 2. No-op reload: index.json itself is NOT rewritten ────────────────────

#[test]
fn unmodified_reload_does_not_rewrite_index_json() {
    let dir = stage_three_files("noop");
    let opts = persist_opts();

    let _ = read_folder_with(&dir, &opts).expect("first read_folder_with");
    let index_path = dir.join(".redhop/index.json");
    let mtime_after_first = fs::metadata(&index_path).unwrap().modified().unwrap();

    // Make sure enough wall-clock time has passed that any rewrite WOULD
    // show up as a different mtime. Otherwise a same-millisecond rewrite
    // could mask the bug we're trying to catch.
    std::thread::sleep(Duration::from_millis(20));

    // Reload with no on-disk changes. Loader hits `changed = false` and
    // skips the index.json write entirely. If a future refactor moves the
    // write outside the `if changed { ... }` guard, this trips.
    let _ = read_folder_with(&dir, &opts).expect("second read_folder_with");
    let mtime_after_second = fs::metadata(&index_path).unwrap().modified().unwrap();

    assert_eq!(
        mtime_after_first, mtime_after_second,
        "no-op reload must NOT rewrite index.json — the `changed = true` \
         write gate exists to keep idle reloads cheap. If this fails, every \
         restart re-serializes the whole index for no reason."
    );

    let _ = fs::remove_dir_all(&dir);
}

// ── 3. Fingerprint change invalidates the cache ────────────────────────────

#[test]
fn changing_load_options_invalidates_the_whole_cache() {
    let dir = stage_three_files("fp");

    // First load with default chunk size.
    let opts1 = FolderOptions {
        persist: true,
        load: LoadOptions {
            chunk_size: Some(64),
            ..Default::default()
        },
        ..Default::default()
    };
    let _ = read_folder_with(&dir, &opts1).expect("first load");
    let fp1 = fingerprint(&read_index(&dir));

    // Reload with a DIFFERENT chunk size — same files on disk but a
    // different chunking config. Fingerprint changes; cache is invalidated;
    // every file gets re-chunked even though none was touched. This is
    // correct behavior — chunks computed with chunk_size=64 are not
    // interchangeable with chunks computed at chunk_size=32.
    let opts2 = FolderOptions {
        persist: true,
        load: LoadOptions {
            chunk_size: Some(32),
            ..Default::default()
        },
        ..Default::default()
    };
    let _ = read_folder_with(&dir, &opts2).expect("second load with new config");
    let fp2 = fingerprint(&read_index(&dir));

    assert_ne!(
        fp1, fp2,
        "fingerprint must change when LoadOptions changes — otherwise an \
         old cache produced under config A would silently be served under \
         config B, giving chunks that don't match the requested chunk_size"
    );

    let _ = fs::remove_dir_all(&dir);
}

// ── 4. Deleted file drops from the cache ───────────────────────────────────

#[test]
fn deleted_file_drops_from_persisted_cache() {
    let dir = stage_three_files("del");
    let opts = persist_opts();

    let _ = read_folder_with(&dir, &opts).expect("first load");
    let map1 = entries(&read_index(&dir));
    assert_eq!(map1.len(), 3, "3 files cached initially");

    // Delete beta.txt and reload.
    fs::remove_file(dir.join("beta.txt")).unwrap();
    let _ = read_folder_with(&dir, &opts).expect("second load after delete");
    let map2 = entries(&read_index(&dir));

    assert_eq!(
        map2.len(),
        2,
        "deleted file must drop from the cache — otherwise the index grows \
         unboundedly across deletes and stale chunks leak into retrieval"
    );
    assert!(
        !map2.keys().any(|k| k.ends_with("beta.txt")),
        "beta.txt specifically must be gone from the cache"
    );
    // The surviving files must still be cache-hit (untouched, byte-identical).
    assert_eq!(entry_for(&map1, "alpha.txt"), entry_for(&map2, "alpha.txt"));
    assert_eq!(entry_for(&map1, "gamma.txt"), entry_for(&map2, "gamma.txt"));

    let _ = fs::remove_dir_all(&dir);
}
