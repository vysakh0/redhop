//! Loader orchestration — the ergonomic constructors over the `redhop-document`
//! core: text/chunks/file/bytes/folder loading, retrieval/chunking config, the
//! semantic embedder, folder persistence, and a citations accessor. This is the
//! same surface the Python and Node bindings expose, in Rust.

use crate::{
    BuiltContext, Chunk, ContextConfig, ContextStrategy, Document, DocumentConfig, Error, Result,
    RetrievalMode, Section,
};

/// Chunking / assembly / retrieval options for the loaders. Every field is
/// optional; defaults match the rest of RedHop.
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Source label for [`text`] (default `"document"`).
    pub source: Option<String>,
    /// Target tokens per chunk (index-time). Default 128.
    pub chunk_size: Option<usize>,
    /// Sentences of overlap between adjacent chunks. Default 1.
    pub chunk_overlap: Option<usize>,
    /// Default assembly token budget. Default 8192.
    pub token_budget: Option<usize>,
    /// Candidates retrieved before assembly. Default 20.
    pub candidate_k: Option<usize>,
    /// Assembly strategy: `auto` (default), `reasoning_preserving`,
    /// `distractor_filtered`, `redundancy_pruned`, `max_density`, `raw_topk`.
    pub strategy: Option<String>,
    /// `"lexical"` (default), `"hybrid"`, or `"semantic"`.
    pub retrieval: Option<String>,
    /// Built-in embedding model to download (`"bge-small"` / `"bge-base"`).
    pub model: Option<String>,
    /// Advanced: path to a local ONNX embedding model.
    pub embedder_model: Option<String>,
    /// Advanced: path to its tokenizer.
    pub embedder_tokenizer: Option<String>,
    /// Advanced: the model's output dimension. Default 384.
    pub embedder_dim: Option<usize>,
    /// Advanced: `"cls"` (default) or `"mean"`.
    pub embedder_pooling: Option<String>,
    /// Advanced: query prefix for asymmetric models (E5: `"query: "`).
    pub embedder_query_prefix: Option<String>,
    /// Advanced: passage prefix for asymmetric models (E5: `"passage: "`).
    pub embedder_passage_prefix: Option<String>,
    /// Hybrid BM25 prune depth. Default 50.
    pub candidate_pool: Option<usize>,
    /// Optional second-stage cross-encoder reranker, by name (`"cross-encoder"`,
    /// the MS-MARCO MiniLM model, auto-downloaded). Reorders the first-stage
    /// candidate pool by jointly scoring each `(query, passage)` pair — more
    /// accurate, at a model call per candidate. `None` (default) leaves the
    /// first-stage ranking as-is. Works under any `retrieval` tier and needs the
    /// `semantic` feature.
    pub rerank: Option<String>,
    /// Minimum candidates to deliver to the assembler. Under `hybrid` /
    /// `semantic` retrieval, if the primary tier returns fewer, a BM25
    /// fallback over the same chunks tops the result up to this floor.
    /// `None` (default) means no floor — primary results stand as-is. Has no
    /// effect under `lexical`. Pair with `report.low_confidence_retrieval` to
    /// detect when the fallback fired with weak chunks (issue #1).
    pub min_candidates: Option<usize>,
    /// Lexical analyzer language (`"english"` default; also `"french"`,
    /// `"german"`, `"spanish"`, `"italian"`, `"portuguese"`, `"dutch"`,
    /// `"russian"`, `"swedish"`, `"norwegian"`, `"danish"`, `"finnish"`,
    /// `"romanian"`, `"hungarian"`, `"turkish"`, `"arabic"`, `"greek"`,
    /// `"tamil"` — the 18 Snowball Porter2 languages). Drives both BM25
    /// retrieval and the grounding scorer's term extraction so the two
    /// layers agree on what counts as "the same term".
    ///
    /// Routes to [`crate::analyzer::SnowballAnalyzer::by_name`]; unknown
    /// language strings return an error (we don't silently fall back to
    /// English — a typo'd `"germann"` should surface, not silently degrade
    /// to English ranking). For a custom analyzer (e.g. a CJK tokenizer),
    /// drop the string config and call `Document::with_analyzer` directly
    /// with your `Arc<dyn Analyzer>`.
    pub language: Option<String>,
    /// Preserve source-document order of selected chunks instead of the
    /// strategy's relevance order. Useful for chat histories / transcripts
    /// where chronology matters. Default `false`.
    /// See `crates/examples/examples/chat_rag.rs` for the worked pattern.
    pub preserve_order: Option<bool>,
}

/// Options for `read_folder_with` (plus the chunking/retrieval [`LoadOptions`]).
/// `read_folder_with` itself is gated behind the `files` feature.
#[derive(Debug, Clone, Default)]
pub struct FolderOptions {
    /// Recurse into subdirectories. Default true.
    pub recursive: Option<bool>,
    /// Honor `.gitignore`. Default true.
    pub gitignore: Option<bool>,
    /// Extra gitignore-style globs to exclude (e.g. `["*.lock", "tests/**"]`).
    pub ignore: Vec<String>,
    /// Persist the index to disk and reload incrementally on the next run.
    pub persist: bool,
    /// Where the persisted index lives (default `<folder>/.redhop`).
    pub index_dir: Option<String>,
    /// Chunking / retrieval options.
    pub load: LoadOptions,
}

/// Where one selected chunk came from — for citing the evidence.
#[derive(Debug, Clone)]
pub struct Citation {
    /// File path / source the chunk came from.
    pub source: String,
    /// 1-based page or slide number, when the format has one.
    pub page: Option<u64>,
    /// Nearest heading / symbol / sheet name, when known.
    pub heading: Option<String>,
    /// 1-based start line, for text & code.
    pub line: Option<u64>,
    /// The chunk text.
    pub text: String,
}

/// Provenance of each selected chunk, in order — `source` plus whichever of
/// `page`/`heading`/`line` the format provides. Read off a [`BuiltContext`].
pub fn citations(ctx: &BuiltContext) -> Vec<Citation> {
    ctx.chunks
        .iter()
        .map(|c| Citation {
            source: c.source.clone(),
            page: c.metadata.get("page").and_then(|v| v.as_u64()),
            heading: c
                .metadata
                .get("heading")
                .and_then(|v| v.as_str())
                .map(String::from),
            line: c.metadata.get("line").and_then(|v| v.as_u64()),
            text: c.text.clone(),
        })
        .collect()
}

/// Parse a string into a [`ContextStrategy`]. The canonical mapping used by
/// every binding (Python `strategy=`, Node `options.strategy`, the CLI), so
/// each language exposes the same strings and the same error message for an
/// unknown name. See [`ContextStrategy`] for what each strategy does.
pub fn strategy_from_str(s: &str) -> Result<ContextStrategy> {
    use ContextStrategy as S;
    Ok(match s {
        "raw_topk" => S::RawTopK,
        "distractor_filtered" => S::DistractorFiltered,
        "redundancy_pruned" => S::RedundancyPruned,
        "max_density" => S::MaxDensity,
        "reasoning_preserving" => S::ReasoningPreserving,
        "auto" => S::Auto,
        other => {
            return Err(Error::Other(format!(
                "unknown strategy '{other}' (expected: raw_topk, distractor_filtered, \
                 redundancy_pruned, max_density, reasoning_preserving, auto)"
            )))
        }
    })
}

/// Parse a string into a [`RetrievalMode`]. `None` and `Some("lexical")` both
/// resolve to BM25 (the default); `Some("hybrid")` produces
/// [`RetrievalMode::Hybrid`] with the given candidate-pool depth (clamped to
/// at least 1); `Some("semantic")` produces [`RetrievalMode::Dense`]. Same
/// canonical mapping as [`strategy_from_str`] — every binding shares it so
/// the strings and the unknown-mode error message stay in sync.
pub fn retrieval_from_str(retrieval: Option<&str>, candidate_pool: usize) -> Result<RetrievalMode> {
    Ok(match retrieval {
        None | Some("lexical") => RetrievalMode::Lexical,
        Some("hybrid") => RetrievalMode::Hybrid {
            candidate_pool: candidate_pool.max(1),
        },
        Some("semantic") => RetrievalMode::Dense,
        Some(other) => {
            return Err(Error::Other(format!(
                "unknown retrieval mode '{other}'; use 'lexical' (default, BM25), 'hybrid' \
                 (BM25 prune → dense rerank — large corpora, no vector DB), or 'semantic' \
                 (global dense over every chunk — small/bounded corpora)"
            )))
        }
    })
}

fn doc_config(o: &LoadOptions, mode: RetrievalMode) -> Result<DocumentConfig> {
    let base = DocumentConfig::default();
    let strategy = match &o.strategy {
        Some(s) => strategy_from_str(s)?,
        None => base.context.strategy,
    };
    let chunk_size = o.chunk_size.unwrap_or(128);
    // Route LoadOptions::language → SnowballAnalyzer builtin, or `"raw"` /
    // `"none"` → RawAnalyzer (minimal pipeline; trades recall for warm-query
    // latency, see docs/findings/FRAMEWORK_MULTIQUERY.md). Unknown names
    // error out so a typo can't silently fall back to English (which would
    // hide the misconfiguration).
    let analyzer: std::sync::Arc<dyn crate::analyzer::Analyzer> = match o.language.as_deref() {
        None => base.context.analyzer.clone(),
        Some("raw") | Some("none") => std::sync::Arc::new(crate::analyzer::RawAnalyzer::new()),
        Some(name) => match crate::analyzer::SnowballAnalyzer::by_name(name) {
            Some(a) => std::sync::Arc::new(a),
            None => {
                return Err(Error::Other(format!(
                    "unknown language {name:?}; supported: arabic, danish, dutch, \
                     english, finnish, french, german, greek, hungarian, italian, \
                     norwegian, portuguese, romanian, russian, spanish, swedish, \
                     tamil, turkish — or 'raw' / 'none' for the minimal \
                     no-stemming pipeline"
                )));
            }
        },
    };
    let context = ContextConfig {
        token_budget: o.token_budget.unwrap_or(8192),
        strategy,
        analyzer,
        preserve_order: o.preserve_order.unwrap_or(base.context.preserve_order),
        ..base.context
    };
    Ok(DocumentConfig {
        target_tokens: chunk_size,
        max_tokens: chunk_size * 2,
        overlap_sentences: o.chunk_overlap.unwrap_or(1),
        candidate_k: o.candidate_k.unwrap_or(20),
        retrieval_mode: mode,
        rerank_pool: base.rerank_pool,
        context,
        min_candidates: o.min_candidates.unwrap_or(base.min_candidates),
        // Inherits the Rust-side default (1 = on). Not exposed as a loader
        // kwarg yet — the auto-expansion fires only on code-classified
        // chunks anyway, so plain text/prose loaders see no change.
        code_neighbors_default: base.code_neighbors_default,
        // Inherits the Rust-side default (true). Fires only on chunks that
        // carry a section heading (markdown / DOCX / PPTX / XLSX / PDF).
        prose_heading_default: base.prose_heading_default,
    })
}

#[cfg(feature = "semantic")]
fn apply_embedder(doc: Document, o: &LoadOptions) -> Result<Document> {
    use crate::embeddings::{EmbedderConfig, OnnxEmbedder, Pooling};
    use std::sync::Arc;

    // Path A — explicit local model files (advanced); `model` is ignored.
    if let (Some(m), Some(t)) = (o.embedder_model.as_ref(), o.embedder_tokenizer.as_ref()) {
        let pooling = match o.embedder_pooling.as_deref() {
            None | Some("cls") => Pooling::Cls,
            Some("mean") => Pooling::Mean,
            Some(other) => {
                return Err(Error::Other(format!(
                    "unknown embedder pooling '{other}' (expected: 'cls' or 'mean')"
                )))
            }
        };
        let dim = o.embedder_dim.unwrap_or(384);
        let load = |prefix: &str| -> Result<OnnxEmbedder> {
            let mut config = EmbedderConfig::bge(dim);
            config.pooling = pooling;
            config.prefix = prefix.to_string();
            OnnxEmbedder::load(m, t, config)
        };
        return match (&o.embedder_query_prefix, &o.embedder_passage_prefix) {
            (q, p) if q.is_some() || p.is_some() => Ok(doc
                .with_embedder(Arc::new(load(p.as_deref().unwrap_or(""))?))
                .with_query_embedder(Arc::new(load(q.as_deref().unwrap_or(""))?))),
            _ => Ok(doc.with_embedder(Arc::new(load("")?))),
        };
    }
    // Path B — model-by-name (auto-downloaded, cached), or the default.
    let name = o
        .model
        .as_deref()
        .unwrap_or(crate::embeddings::DEFAULT_MODEL);
    let resolved = crate::embeddings::resolve_model(name)?;
    let load = |prefix: &str| -> Result<OnnxEmbedder> {
        OnnxEmbedder::load(
            &resolved.model_path,
            &resolved.tokenizer_path,
            resolved.config(prefix),
        )
    };
    if resolved.is_asymmetric() {
        Ok(doc
            .with_embedder(Arc::new(load(&resolved.passage_prefix)?))
            .with_query_embedder(Arc::new(load(&resolved.query_prefix)?)))
    } else {
        Ok(doc.with_embedder(Arc::new(load(&resolved.passage_prefix)?)))
    }
}

#[cfg(not(feature = "semantic"))]
fn apply_embedder(_doc: Document, _o: &LoadOptions) -> Result<Document> {
    Err(Error::Other(
        "retrieval='semantic'/'hybrid' needs the `semantic` feature (the embedding engine)".into(),
    ))
}

/// Attach a second-stage cross-encoder reranker named by `o.rerank`.
#[cfg(feature = "semantic")]
fn apply_reranker(doc: Document, name: &str) -> Result<Document> {
    use crate::reranking::OnnxCrossEncoder;
    use std::sync::Arc;
    let r = crate::embeddings::resolve_reranker(name)?;
    let ce = OnnxCrossEncoder::load(&r.model_path, &r.tokenizer_path, r.max_seq_len)?;
    Ok(doc.with_reranker(Arc::new(ce)))
}

#[cfg(not(feature = "semantic"))]
fn apply_reranker(_doc: Document, _name: &str) -> Result<Document> {
    Err(Error::Other(
        "rerank='cross-encoder' needs the `semantic` feature (the model engine)".into(),
    ))
}

fn needs_embedder(mode: RetrievalMode) -> bool {
    matches!(mode, RetrievalMode::Hybrid { .. } | RetrievalMode::Dense)
}

/// Attach the optional dense embedder (for hybrid/semantic) and cross-encoder
/// reranker (`o.rerank`) a built [`Document`] needs. Shared by every loader.
fn finish_models(doc: Document, o: &LoadOptions, mode: RetrievalMode) -> Result<Document> {
    let doc = if needs_embedder(mode) {
        apply_embedder(doc, o)?
    } else {
        doc
    };
    match o.rerank.as_deref() {
        Some(name) => apply_reranker(doc, name),
        None => Ok(doc),
    }
}

fn build(files: Vec<(String, Vec<Section>)>, o: &LoadOptions) -> Result<Document> {
    let mode = retrieval_from_str(o.retrieval.as_deref(), o.candidate_pool.unwrap_or(50))?;
    let cfg = doc_config(o, mode)?;
    let doc = Document::from_sources_with(files, cfg)?;
    finish_models(doc, o, mode)
}

/// Build a [`Document`] from raw text with options (retrieval, chunking, …).
pub fn text(text: impl Into<String>, o: &LoadOptions) -> Result<Document> {
    let source = o.source.clone().unwrap_or_else(|| "document".to_string());
    build(
        vec![(
            source,
            vec![Section {
                text: text.into(),
                ..Default::default()
            }],
        )],
        o,
    )
}

/// Build a [`Document`] from chunks you already produced (strings).
pub fn chunks(chunks: Vec<String>, o: &LoadOptions) -> Result<Document> {
    let sections = chunks
        .into_iter()
        .map(|t| Section {
            text: t,
            ..Default::default()
        })
        .collect();
    build(vec![("chunks".to_string(), sections)], o)
}

/// Build a [`Document`] from pre-formed [`Chunk`]s with the supplied
/// retrieval/embedder/reranker options applied.
///
/// Bypasses the chunker — the caller's chunks are preserved 1-to-1
/// (no resplitting), with their `source` / `id` / `metadata` all
/// surviving into the index. Used by the typed-binding paths
/// (`redhop.Chunk` in Python, `new redhop.Chunk(...)` in Node) where
/// the user has already chunked their corpus elsewhere and wants the
/// retrieval/diagnostics layer over the result.
pub fn chunks_typed(chunks: Vec<Chunk>, o: &LoadOptions) -> Result<Document> {
    let mode = retrieval_from_str(o.retrieval.as_deref(), o.candidate_pool.unwrap_or(50))?;
    let cfg = doc_config(o, mode)?;
    let doc = Document::from_chunks_with(chunks, cfg)?;
    finish_models(doc, o, mode)
}

// ── File / bytes / folder loaders (require the `files` feature) ──────────────
#[cfg(feature = "files")]
mod files_loaders {
    use super::*;
    use crate::{Chunk, ChunkId};
    use std::path::{Path, PathBuf};

    fn convert(sections: Vec<crate::files::Section>) -> Vec<Section> {
        sections
            .into_iter()
            .map(|s| Section {
                text: s.text,
                page: s.page,
                heading: s.heading,
                line: s.line,
            })
            .collect()
    }

    fn extract_path(path: &Path) -> Result<(String, Vec<Section>)> {
        let d = crate::files::extract(path).map_err(|e| Error::Other(e.to_string()))?;
        Ok((d.source, convert(d.sections)))
    }

    /// Parse a file on disk into a [`Document`] (default config).
    pub fn read_file(path: impl AsRef<Path>) -> Result<Document> {
        read_file_with(path, &LoadOptions::default())
    }

    /// Parse a file on disk into a [`Document`] with options.
    pub fn read_file_with(path: impl AsRef<Path>, o: &LoadOptions) -> Result<Document> {
        build(vec![extract_path(path.as_ref())?], o)
    }

    /// Parse already-fetched bytes into a [`Document`] (default config). `name`
    /// (e.g. `"contract.pdf"`) picks the parser and is the citation source.
    pub fn read_bytes(data: &[u8], name: &str) -> Result<Document> {
        read_bytes_with(data, name, &LoadOptions::default())
    }

    /// Parse bytes into a [`Document`] with options — the on-ramp for S3 / GCS /
    /// Azure Blob / HTTP / DB blobs (fetch with your own client).
    pub fn read_bytes_with(data: &[u8], name: &str, o: &LoadOptions) -> Result<Document> {
        let d = crate::files::extract_bytes(data, name).map_err(|e| Error::Other(e.to_string()))?;
        build(vec![(d.source, convert(d.sections))], o)
    }

    const SKIP: &[&str] = &[
        "node_modules",
        "target",
        "__pycache__",
        "venv",
        "dist",
        "build",
    ];

    fn collect_files(root: &Path, fo: &FolderOptions) -> Result<Vec<PathBuf>> {
        let mut ob = ignore::overrides::OverrideBuilder::new(root);
        for d in SKIP {
            let _ = ob.add(&format!("!{d}"));
        }
        for g in &fo.ignore {
            ob.add(&format!("!{g}"))
                .map_err(|e| Error::Other(format!("bad ignore '{g}': {e}")))?;
        }
        let overrides = ob.build().map_err(|e| Error::Other(e.to_string()))?;
        let gi = fo.gitignore.unwrap_or(true);
        let mut wb = ignore::WalkBuilder::new(root);
        wb.hidden(true)
            .git_ignore(gi)
            .git_global(gi)
            .git_exclude(gi)
            .ignore(gi)
            .parents(gi)
            .require_git(false)
            .overrides(overrides);
        if !fo.recursive.unwrap_or(true) {
            wb.max_depth(Some(1));
        }
        let mut out = Vec::new();
        for entry in wb.build() {
            let Ok(entry) = entry else { continue };
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                out.push(entry.into_path());
            }
        }
        out.sort();
        Ok(out)
    }

    /// Index every readable file under a folder into one [`Document`] (default).
    pub fn read_folder(path: impl AsRef<Path>) -> Result<Document> {
        read_folder_with(path, &FolderOptions::default())
    }

    /// Index a folder into one [`Document`] with options — recursion,
    /// `.gitignore` (and custom `ignore` globs), retrieval/chunking config,
    /// and `persist` for an incremental on-disk index.
    pub fn read_folder_with(path: impl AsRef<Path>, fo: &FolderOptions) -> Result<Document> {
        let root = path.as_ref();
        if !root.is_dir() {
            return Err(Error::Other(format!(
                "read_folder: '{}' is not a directory",
                root.display()
            )));
        }
        if fo.persist || fo.index_dir.is_some() {
            return persisted(root, fo);
        }
        let mut files = Vec::new();
        let mut skipped: Vec<(String, String)> = Vec::new();
        for p in collect_files(root, fo)? {
            match extract_path(&p) {
                Ok(f) => files.push(f),
                Err(e) => skipped.push((p.to_string_lossy().into_owned(), e.to_string())),
            }
        }
        if files.is_empty() {
            return Err(Error::Other(format!(
                "read_folder: no readable files under '{}'",
                root.display()
            )));
        }
        let n_files = files.len();
        let mut doc = build(files, &fo.load)?;
        doc.set_folder_provenance(n_files, skipped);
        Ok(doc)
    }

    // ── persistence ─────────────────────────────────────────────────────────
    #[derive(serde::Serialize, serde::Deserialize)]
    struct CachedFile {
        source: String,
        mtime: u64,
        size: u64,
        chunks: Vec<Chunk>,
    }
    #[derive(serde::Serialize, serde::Deserialize)]
    struct PersistedIndex {
        version: u32,
        fingerprint: String,
        files: Vec<CachedFile>,
    }
    const INDEX_VERSION: u32 = 1;

    fn fingerprint(o: &LoadOptions) -> String {
        let s = |x: &Option<String>| x.clone().unwrap_or_default();
        format!(
            "v{INDEX_VERSION}|cs={}|co={}|r={}|m={}|em={}|ed={}",
            o.chunk_size.unwrap_or(128),
            o.chunk_overlap.unwrap_or(1),
            s(&o.retrieval),
            s(&o.model),
            s(&o.embedder_model),
            o.embedder_dim.unwrap_or(384),
        )
    }
    fn file_stat(p: &Path) -> Option<(u64, u64)> {
        let m = std::fs::metadata(p).ok()?;
        let mtime = m
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos() as u64;
        Some((mtime, m.len()))
    }
    fn index_dir(root: &Path, fo: &FolderOptions) -> PathBuf {
        fo.index_dir
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(".redhop"))
    }

    fn persisted(root: &Path, fo: &FolderOptions) -> Result<Document> {
        use std::collections::{HashMap, HashSet};
        let dir = index_dir(root, fo);
        let fp = fingerprint(&fo.load);
        let mut cache: HashMap<String, CachedFile> = HashMap::new();
        let had_cache = std::fs::read_to_string(dir.join("index.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<PersistedIndex>(&s).ok())
            .filter(|idx| idx.version == INDEX_VERSION && idx.fingerprint == fp)
            .map(|idx| {
                for f in idx.files {
                    cache.insert(f.source.clone(), f);
                }
            })
            .is_some();

        let mode = retrieval_from_str(
            fo.load.retrieval.as_deref(),
            fo.load.candidate_pool.unwrap_or(50),
        )?;
        let cfg = doc_config(&fo.load, mode)?;
        let mut entries: Vec<CachedFile> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut changed = !had_cache;
        let mut skipped: Vec<(String, String)> = Vec::new();
        for p in collect_files(root, fo)? {
            let source = p.to_string_lossy().to_string();
            let Some((mtime, size)) = file_stat(&p) else {
                skipped.push((source, "could not stat file".to_string()));
                continue;
            };
            seen.insert(source.clone());
            if let Some(prev) = cache.get(&source) {
                if prev.mtime == mtime && prev.size == size {
                    entries.push(CachedFile {
                        source,
                        mtime,
                        size,
                        chunks: prev.chunks.clone(),
                    });
                    continue;
                }
            }
            match extract_path(&p) {
                Ok((_, sections)) => {
                    let chunks = Document::chunk_sections(&source, &sections, &cfg)?;
                    entries.push(CachedFile {
                        source,
                        mtime,
                        size,
                        chunks,
                    });
                    changed = true;
                }
                Err(e) => {
                    skipped.push((source, e.to_string()));
                }
            }
        }
        if cache.keys().any(|s| !seen.contains(s)) {
            changed = true;
        }
        let mut all: Vec<Chunk> = entries
            .iter()
            .flat_map(|f| f.chunks.iter().cloned())
            .collect();
        if all.is_empty() {
            return Err(Error::Other(format!(
                "read_folder: no readable files under '{}'",
                root.display()
            )));
        }
        for (i, c) in all.iter_mut().enumerate() {
            c.id = ChunkId::new(i.to_string());
        }
        let mut doc = Document::from_chunks_with(all, cfg)?;
        doc = finish_models(doc, &fo.load, mode)?;
        doc.set_folder_provenance(entries.len(), skipped);
        if changed {
            let embedded = doc.embedded_chunks()?;
            let mut by_source: HashMap<String, Vec<Chunk>> = HashMap::new();
            for c in embedded {
                by_source.entry(c.source.clone()).or_default().push(c);
            }
            let files = entries
                .into_iter()
                .map(|f| CachedFile {
                    chunks: by_source.remove(&f.source).unwrap_or_default(),
                    source: f.source,
                    mtime: f.mtime,
                    size: f.size,
                })
                .collect();
            let idx = PersistedIndex {
                version: INDEX_VERSION,
                fingerprint: fp,
                files,
            };
            std::fs::create_dir_all(&dir).map_err(Error::from)?;
            let json = serde_json::to_string(&idx).map_err(|e| Error::Other(e.to_string()))?;
            std::fs::write(dir.join("index.json"), json).map_err(Error::from)?;
        }
        Ok(doc)
    }
}

#[cfg(feature = "files")]
pub use files_loaders::{
    read_bytes, read_bytes_with, read_file, read_file_with, read_folder, read_folder_with,
};
