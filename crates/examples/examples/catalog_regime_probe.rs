//! Catalog-regime retrieval probe — re-deriving the external short-query /
//! high-cardinality / near-duplicate evidence on a redhop rig.
//!
//! The external evidence (an unmeasured regime for redhop's HotpotQA / MuSiQue
//! / CUAD suite) claimed three things this probe tries to reproduce on a
//! synthetic catalog built from generic public goods:
//!
//!   1. **Retriever inverts with corpus size.** A char-ngram (subword) lexical
//!      retriever wins *early precision* on short noisy queries at small scale,
//!      but its dense gram vocabulary collides at large scale and its recall
//!      floor collapses, while word-BM25 holds.
//!   2. **Field weighting (BM25F-lite) lifts strict set-coverage, then a cliff.**
//!      A modest boost on the discriminative structured field (here the brand,
//!      indexed as `heading`) lifts AmbCov; over-boosting starves recall.
//!   3. **set-coverage hides behind recall@k.** A query that maps to a SET (all
//!      variants of a product) needs the WHOLE family retrieved; flat recall@k
//!      against a single gold reads healthy while a family is un-offerable.
//!
//! Everything here is deterministic (a seeded LCG drives typo injection and the
//! bootstrap) and model-free. Treat every number as a *synthetic, single-domain*
//! re-derivation — a hypothesis re-tested on a redhop rig, not a fact. See
//! `docs/findings/CATALOG_REGIME.md`.
//!
//! Run: cargo run -p redhop-examples --example catalog_regime_probe --release

use std::sync::Arc;

use redhop::analyzer::CharNgramAnalyzer;
use redhop::retrieval::{Bm25Retriever, FieldWeights};
use redhop::traits::Retriever;
use redhop::{Chunk, Query, TokenCount};

// ─── Deterministic RNG (no `rand` dep; reproducible) ────────────────────────

/// Knuth MMIX linear congruential generator. Deterministic given a seed, so
/// the typo injection and the bootstrap are bit-reproducible across runs.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use the high bits (better-distributed than the low bits of an LCG).
        self.0 >> 1
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

// ─── Synthetic catalog (generic public goods, no real brands) ───────────────

const BRANDS: &[&str] = &[
    "acme",
    "brightline",
    "northwind",
    "summit",
    "cascade",
    "meadow",
    "harbor",
    "vista",
    "cobalt",
    "maple",
    "orchard",
    "pioneer",
    "sterling",
];
const PRODUCTS: &[&str] = &[
    "chips", "cola", "biscuits", "noodles", "soap", "shampoo", "coffee", "tea", "juice", "candy",
];
const FLAVORS: &[&str] = &[
    "salted",
    "masala",
    "tomato",
    "classic",
    "mint",
    "lemon",
    "mango",
    "chocolate",
    "spicy",
    "plain",
    "cream",
    "honey",
    "orange",
    "berry",
    "vanilla",
    "caramel",
    "pepper",
    "ginger",
    "coconut",
    "almond",
];
const SIZES: &[u32] = &[25, 52, 90, 150, 200];

/// Number of (brand, product) families the queries target. Their full variant
/// sets are always present in the catalog (the "base") regardless of size.
const PROBE_COMBOS: usize = 24;
/// Variants per probe family: FAMILY_FLAVORS flavors x FAMILY_SIZES sizes.
const FAMILY_FLAVORS: usize = 3;
const FAMILY_SIZES: usize = 2;

fn mrp_for(size: u32) -> u32 {
    match size {
        25 => 5,
        52 => 10,
        90 => 20,
        150 => 30,
        200 => 50,
        _ => 10,
    }
}

#[derive(Clone)]
struct Sku {
    id: String,
    brand: &'static str,
    product: &'static str,
    flavor: &'static str,
    size: u32,
}

impl Sku {
    /// The full enriched document text (brand + name + flavor + size + price).
    /// The brand and product tokens repeat in `heading`, which is the
    /// field-weight lever.
    fn enriched(&self) -> String {
        format!(
            "{} {} {} {} gram mrp {}",
            self.brand,
            self.product,
            self.flavor,
            self.size,
            mrp_for(self.size),
        )
    }
    /// The heading token(s) a field-weight boost amplifies. `Brand` is a
    /// too-broad field (shared across all of a brand's products); `BrandProduct`
    /// is the family-discriminating field a clarify query ("brand product")
    /// targets exactly.
    fn heading(&self, hk: HeadingKind) -> String {
        match hk {
            HeadingKind::Brand => self.brand.to_string(),
            HeadingKind::BrandProduct => format!("{} {}", self.brand, self.product),
        }
    }
    fn chunk(&self, hk: HeadingKind) -> Chunk {
        let text = self.enriched();
        let tokens = text.split_whitespace().count();
        let mut c = Chunk::new(self.id.clone(), text, self.id.clone(), TokenCount(tokens));
        c.metadata.insert(
            "heading".into(),
            serde_json::Value::String(self.heading(hk)),
        );
        c
    }
}

/// Which structured token(s) the `heading` field carries (the field a boost
/// amplifies). The choice is the lesson: boosting the family-discriminating
/// field lifts set-coverage; boosting a too-broad one hurts it.
#[derive(Clone, Copy)]
enum HeadingKind {
    /// Brand only — shared across every product of the brand (too broad).
    Brand,
    /// Brand + product — the exact key a clarify query discriminates on.
    BrandProduct,
}

/// The PROBE_COMBOS distinct (brand_idx, product_idx) pairs the queries target.
fn probe_combos() -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    'outer: for p in 0..PRODUCTS.len() {
        for b in 0..BRANDS.len() {
            v.push((b, p));
            if v.len() == PROBE_COMBOS {
                break 'outer;
            }
        }
    }
    v
}

/// `(base, filler)`. `base` = every variant of every probe family (always in
/// the catalog so query gold is valid at all sizes). `filler` = near-duplicate
/// siblings (same brands / flavors / sizes, other combos) in a deterministic
/// order, sliced to pad the catalog to a target size.
fn base_and_filler() -> (Vec<Sku>, Vec<Sku>) {
    let combos = probe_combos();
    let mut base = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut id = 0usize;
    for &(b, p) in &combos {
        for &fl in FLAVORS.iter().take(FAMILY_FLAVORS) {
            for &sz in SIZES.iter().take(FAMILY_SIZES) {
                let sku = Sku {
                    id: format!("sku{id:05}"),
                    brand: BRANDS[b],
                    product: PRODUCTS[p],
                    flavor: fl,
                    size: sz,
                };
                seen.insert((b, p, fl, sz));
                base.push(sku);
                id += 1;
            }
        }
    }
    // Filler: every other (brand, product, flavor, size) combo, deterministic
    // order. Includes more variants of the probe families (sibling pressure on
    // the exact families the clarify queries target) AND entirely different
    // combos. All reuse the same brand / flavor / size vocabulary, so tokens
    // collide across SKUs — the near-duplicate stressor.
    let mut filler = Vec::new();
    for (b, _brand) in BRANDS.iter().enumerate() {
        for (p, _product) in PRODUCTS.iter().enumerate() {
            for &fl in FLAVORS.iter() {
                for &sz in SIZES.iter() {
                    let key = (b, p, fl, sz);
                    if seen.contains(&key) {
                        continue;
                    }
                    filler.push(Sku {
                        id: format!("sku{id:05}"),
                        brand: BRANDS[b],
                        product: PRODUCTS[p],
                        flavor: fl,
                        size: sz,
                    });
                    id += 1;
                }
            }
        }
    }
    (base, filler)
}

fn catalog(base: &[Sku], filler: &[Sku], target: usize) -> Vec<Sku> {
    let mut out = base.to_vec();
    let need = target.saturating_sub(out.len());
    out.extend(filler.iter().take(need).cloned());
    out
}

// ─── Queries (derived from the probe families; stable across sizes) ─────────

struct ResolveQ {
    text: String,
    gold: String,
}
struct ClarifyQ {
    text: String,
    family: Vec<String>,
}

/// OCR / transcription-style corruption of a single token: one deterministic
/// edit (a confusable-character swap where one exists, else a neighbor swap or
/// a drop). Mirrors `lays -> 1ays`, `kurkure -> kurkur`.
fn corrupt(token: &str, rng: &mut Lcg) -> String {
    let mut chars: Vec<char> = token.chars().collect();
    if chars.len() < 2 {
        return token.to_string();
    }
    let pos = rng.below(chars.len());
    let confuse = |c: char| -> Option<char> {
        match c {
            'l' => Some('1'),
            'o' => Some('0'),
            's' => Some('5'),
            'i' => Some('1'),
            'e' => Some('3'),
            'a' => Some('4'),
            'b' => Some('8'),
            'g' => Some('9'),
            't' => Some('7'),
            _ => None,
        }
    };
    match confuse(chars[pos]) {
        Some(r) => chars[pos] = r,
        None => {
            if pos + 1 < chars.len() {
                chars.swap(pos, pos + 1);
            } else {
                chars.remove(pos);
            }
        }
    }
    chars.into_iter().collect()
}

/// Build the resolve + clarify query sets for the probe families. `noisy`
/// corrupts the brand token (the deep-failure case: a brand typo buries the
/// SKU for token-exact BM25).
fn make_queries(base: &[Sku], noisy: bool) -> (Vec<ResolveQ>, Vec<ClarifyQ>) {
    let mut rng = Lcg::new(if noisy { 0xC0FFEE } else { 0xDEC0DE });
    let combos = probe_combos();
    let mut resolve = Vec::new();
    let mut clarify = Vec::new();

    for &(b, p) in &combos {
        let brand = BRANDS[b];
        let product = PRODUCTS[p];
        // Family = every base variant of this (brand, product).
        let family: Vec<String> = base
            .iter()
            .filter(|s| s.brand == brand && s.product == product)
            .map(|s| s.id.clone())
            .collect();

        // Clarify: "brand product" -> the whole family must be retrieved.
        let brand_tok = if noisy {
            corrupt(brand, &mut rng)
        } else {
            brand.to_string()
        };
        clarify.push(ClarifyQ {
            text: format!("{brand_tok} {product}"),
            family: family.clone(),
        });

        // Resolve: one query per base variant -> the specific SKU is gold.
        for &fl in FLAVORS.iter().take(FAMILY_FLAVORS) {
            for &sz in SIZES.iter().take(FAMILY_SIZES) {
                let gold = base
                    .iter()
                    .find(|s| {
                        s.brand == brand && s.product == product && s.flavor == fl && s.size == sz
                    })
                    .map(|s| s.id.clone());
                let Some(gold) = gold else { continue };
                let brand_tok = if noisy {
                    corrupt(brand, &mut rng)
                } else {
                    brand.to_string()
                };
                resolve.push(ResolveQ {
                    text: format!("{brand_tok} {product} {fl} {sz}"),
                    gold,
                });
            }
        }
    }
    (resolve, clarify)
}

// ─── Retrieval arms ─────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Arm {
    /// Word-BM25 over the enriched blob (the redhop default analyzer).
    Word,
    /// Char-ngram subword analyzer (the typo / short-token tier).
    CharNgram,
    /// Word-BM25 boosting the family-discriminating field (brand+product) — the
    /// BM25F-lite lever applied to the RIGHT field.
    BoostKey(f32),
    /// Word-BM25 boosting a too-broad field (brand only) — the wrong-field
    /// failure: it lifts every same-brand SKU and crowds the family out.
    BoostBrandOnly(f32),
}

impl Arm {
    fn label(&self) -> String {
        match self {
            Arm::Word => "word-bm25".into(),
            Arm::CharNgram => "char-ngram".into(),
            Arm::BoostKey(w) => format!("boost key x{w:.0}"),
            Arm::BoostBrandOnly(w) => format!("boost brand x{w:.0}"),
        }
    }
    fn heading_kind(&self) -> HeadingKind {
        match self {
            Arm::BoostBrandOnly(_) => HeadingKind::Brand,
            _ => HeadingKind::BrandProduct,
        }
    }
    async fn build(&self, skus: &[Sku]) -> anyhow::Result<Bm25Retriever> {
        let hk = self.heading_kind();
        let chunks: Vec<Chunk> = skus.iter().map(|s| s.chunk(hk)).collect();
        let mut r = match self {
            Arm::CharNgram => Bm25Retriever::with_analyzer(Arc::new(CharNgramAnalyzer::default()))?,
            _ => Bm25Retriever::new()?,
        };
        let w = match self {
            Arm::BoostKey(w) | Arm::BoostBrandOnly(w) => *w,
            _ => 1.0,
        };
        if w != 1.0 {
            r = r.with_field_weights(FieldWeights {
                text: 1.0,
                source: 1.0,
                heading: w,
            });
        }
        r.index(&chunks).await?;
        Ok(r)
    }
}

// ─── Metrics + bootstrap CI ─────────────────────────────────────────────────

fn recall_at_k(hits: &[String], gold: &str, k: usize) -> f64 {
    f64::from(hits.iter().take(k).any(|h| h == gold))
}

/// Strict set-coverage: 1.0 iff EVERY family member is in the top-k.
fn set_coverage_at_k(hits: &[String], family: &[String], k: usize) -> f64 {
    let top: std::collections::HashSet<&str> = hits.iter().take(k).map(String::as_str).collect();
    f64::from(family.iter().all(|f| top.contains(f.as_str())))
}

/// Seeded bootstrap 95% CI over a vector of per-query metric values.
fn bootstrap_ci(vals: &[f64], iters: usize, seed: u64) -> (f64, f64, f64) {
    let n = vals.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let mut rng = Lcg::new(seed);
    let mut means = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut s = 0.0;
        for _ in 0..n {
            s += vals[rng.below(n)];
        }
        means.push(s / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[((0.025 * iters as f64) as usize).min(iters - 1)];
    let hi = means[((0.975 * iters as f64) as usize).min(iters - 1)];
    (mean, lo, hi)
}

struct ArmResult {
    r1: f64,
    r5: f64,
    r10: f64,
    r20: (f64, f64, f64),
    setcov: (f64, f64, f64),
    setcov_count: (usize, usize),
}

const TOP_K: usize = 20;
const BOOTSTRAP_ITERS: usize = 2000;

/// Retrieve a deterministically-ordered list of chunk ids.
///
/// Tantivy's multi-threaded indexing assigns internal DocIds non-deterministically,
/// so among **equal-score near-duplicates** (this regime is full of them) the
/// membership of the top-k would be a coin flip run-to-run. We retrieve the full
/// matching pool (`pool` >= corpus size, so Tantivy never truncates at a tie
/// boundary) and re-break ties deterministically by `(score desc, id asc)`. BM25
/// score *values* are deterministic; only the tie order is not, so this makes the
/// metric reproducible without changing what "top-k" means.
async fn retrieve_ids(r: &Bm25Retriever, text: &str, pool: usize, k: usize) -> Vec<String> {
    let mut hits = r.retrieve(&Query::new(text), pool).await.expect("retrieve");
    // Near-duplicate SKUs get mathematically-equal BM25 scores, but
    // multi-segment indexing accumulates them in a non-deterministic order, so
    // the raw f32 values differ in their last bits run-to-run. Quantize to 1e-4
    // (far below any genuine score gap, far above FP noise) so true ties collapse
    // to the same bucket and break deterministically by id.
    let q = |s: f32| (s as f64 * 10_000.0).round() as i64;
    hits.sort_by(|a, b| {
        q(b.score.value)
            .cmp(&q(a.score.value))
            .then_with(|| a.chunk.id.as_str().cmp(b.chunk.id.as_str()))
    });
    hits.into_iter()
        .take(k)
        .map(|h| h.chunk.id.as_str().to_string())
        .collect()
}

async fn run_arm(arm: Arm, skus: &[Sku], resolve: &[ResolveQ], clarify: &[ClarifyQ]) -> ArmResult {
    let r = arm.build(skus).await.expect("build retriever");
    let pool = skus.len();

    let mut r1 = Vec::new();
    let mut r5 = Vec::new();
    let mut r10 = Vec::new();
    let mut r20 = Vec::new();
    for q in resolve {
        let hits = retrieve_ids(&r, &q.text, pool, TOP_K).await;
        r1.push(recall_at_k(&hits, &q.gold, 1));
        r5.push(recall_at_k(&hits, &q.gold, 5));
        r10.push(recall_at_k(&hits, &q.gold, 10));
        r20.push(recall_at_k(&hits, &q.gold, 20));
    }

    let mut sc = Vec::new();
    for q in clarify {
        let hits = retrieve_ids(&r, &q.text, pool, TOP_K).await;
        sc.push(set_coverage_at_k(&hits, &q.family, TOP_K));
    }
    let covered = sc.iter().filter(|&&v| v > 0.5).count();

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    ArmResult {
        r1: mean(&r1),
        r5: mean(&r5),
        r10: mean(&r10),
        r20: bootstrap_ci(&r20, BOOTSTRAP_ITERS, 0xA11CE),
        setcov: bootstrap_ci(&sc, BOOTSTRAP_ITERS, 0xB0B),
        setcov_count: (covered, sc.len()),
    }
}

fn print_table(size: usize, noisy: bool, rows: &[(String, ArmResult)]) {
    let mode = if noisy {
        "noisy (brand typos)"
    } else {
        "clean"
    };
    println!("\n=== catalog n={size}  queries: {mode} ===");
    println!(
        "| {:<16} | {:>6} | {:>6} | {:>6} | {:>20} | {:>20} | {:>9} |",
        "arm", "R@1", "R@5", "R@10", "R@20 [95% CI]", "AmbCov@20 [95% CI]", "AmbCov n"
    );
    println!(
        "|{:-<18}|{:-<8}|{:-<8}|{:-<8}|{:-<22}|{:-<22}|{:-<11}|",
        "", "", "", "", "", "", ""
    );
    for (label, r) in rows {
        println!(
            "| {:<16} | {:>6.3} | {:>6.3} | {:>6.3} | {:>6.3} [{:.3},{:.3}] | {:>6.3} [{:.3},{:.3}] | {:>4}/{:<4} |",
            label,
            r.r1,
            r.r5,
            r.r10,
            r.r20.0,
            r.r20.1,
            r.r20.2,
            r.setcov.0,
            r.setcov.1,
            r.setcov.2,
            r.setcov_count.0,
            r.setcov_count.1,
        );
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let (base, filler) = base_and_filler();
    let sizes = [base.len(), 600, 2500];
    let arms = [
        Arm::Word,
        Arm::CharNgram,
        Arm::BoostKey(3.0),
        Arm::BoostKey(8.0),
        Arm::BoostBrandOnly(3.0),
    ];

    println!("Catalog-regime retrieval probe (synthetic, deterministic, model-free)");
    println!(
        "base families: {PROBE_COMBOS} (brand,product) x {} variants = {} base SKUs",
        FAMILY_FLAVORS * FAMILY_SIZES,
        base.len()
    );
    println!(
        "sizes: {sizes:?}   arms: word-bm25, char-ngram, boost key (brand+product) x3/x8, boost brand-only x3"
    );
    println!(
        "metric: R@k vs single gold (resolve); strict AmbCov@20 = whole family in top-20 (clarify)"
    );

    for &size in &sizes {
        let skus = catalog(&base, &filler, size);
        let actual = skus.len();
        for noisy in [false, true] {
            let (resolve, clarify) = make_queries(&base, noisy);
            let mut rows = Vec::new();
            for &arm in &arms {
                let res = run_arm(arm, &skus, &resolve, &clarify).await;
                rows.push((arm.label(), res));
            }
            print_table(actual, noisy, &rows);
        }
    }

    println!(
        "\nNote: synthetic, single-domain re-derivation of an external regime. \
         Read the inversion (char-ngram early precision at small n vs word-BM25's \
         recall floor at large n) and the brand-boost AmbCov lift/cliff as \
         suggestive mechanism, not a portable number."
    );
    Ok(())
}
