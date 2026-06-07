//! 09 · Multilingual analyzer — `language = "german"` / `"french"` / ...
//!
//! Real-world scenario:
//!     A pharmaceutical company has internal policy documents in
//!     English, German, and French. They want BM25-quality retrieval on
//!     each — which means the tokenizer needs to understand each
//!     language's morphology. German `Bücher` should find a chunk that
//!     only contains `Buch`; French `manger` should find a chunk with
//!     `mange`. The default English analyzer would miss both.
//!
//! What this demonstrates:
//!     - `LoadOptions { language: Some("german"), .. }` routes the
//!       whole pipeline through the right Snowball stemmer.
//!     - The 18 supported languages: arabic, danish, dutch, english,
//!       finnish, french, german, greek, hungarian, italian, norwegian,
//!       portuguese, romanian, russian, spanish, swedish, tamil, turkish.
//!     - An unknown language string ERRORS rather than silently falling
//!       back to English — caught by docs/findings/MULTILINGUAL_ANALYZER.md.
//!
//! Run:
//!     cargo run -p redhop-rust-examples --example 09_multilingual --release

use redhop::{citations, text as load_text, LoadOptions};

const GERMAN_CORPUS: &str = "Ich habe viele Bücher im Regal stehen.\n\nEin Kind spielt fröhlich im Garten.\n\nDer Hund läuft schnell durch den Park.";
const FRENCH_CORPUS: &str = "Il aime manger des pommes chaque matin.\n\nLe chien court dans la rue très vite.\n\nLes enfants jouent au parc le weekend.";

fn demo(label: &str, corpus: &str, query: &str, language: &str) -> anyhow::Result<()> {
    let opts = LoadOptions {
        language: Some(language.to_string()),
        ..LoadOptions::default()
    };
    let mut doc = load_text(corpus, &opts)?;
    let ctx = doc.context(query)?;
    println!("─── {} ────────────────────────────────", label);
    println!("  language={:?}, query={:?}", language, query);
    let cites = citations(&ctx);
    if let Some(c) = cites.first() {
        println!("  top hit: {}", c.text);
    } else {
        println!("  (no hits)");
    }
    println!();
    Ok(())
}

fn main() -> anyhow::Result<()> {
    // German Snowball: "Buch" → reach "Bücher".
    demo("Arm A · German morphology", GERMAN_CORPUS, "Buch", "german")?;

    // French Snowball: "manger" → reach "mange".
    demo(
        "Arm B · French morphology",
        FRENCH_CORPUS,
        "manger",
        "french",
    )?;

    // Counter-example: same German corpus, default English analyzer.
    demo(
        "Arm C · German corpus + English analyzer (the bug it prevents)",
        GERMAN_CORPUS,
        "Buch",
        "english",
    )?;

    // Unknown language: deliberate error rather than silent English fallback.
    println!("─── Arm D · Unknown language string ──────────────");
    let bad_opts = LoadOptions {
        language: Some("germann".into()),
        ..LoadOptions::default()
    };
    match load_text(GERMAN_CORPUS, &bad_opts) {
        Ok(_) => println!("  (oops — should have raised)"),
        Err(e) => {
            let msg = e.to_string();
            let preview: String = msg.chars().take(140).collect();
            println!("  Error: {}…", preview);
        }
    }
    println!();

    println!("─── How to read this ─────────────────────────────");
    println!("Arm A and B: language=… routes the whole pipeline through");
    println!("  the right Snowball stemmer.");
    println!("Arm C: same German corpus + default English analyzer → miss.");
    println!("  Picking the wrong language SILENTLY is the real failure mode.");
    println!("Arm D: unknown language strings ERROR so a typo'd 'germann'");
    println!("  is caught at construction.");
    println!();
    println!("Validated cross-language by docs/findings/MULTILINGUAL_ANALYZER.md.");
    Ok(())
}
