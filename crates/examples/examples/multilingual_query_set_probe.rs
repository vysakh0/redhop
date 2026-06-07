//! Multilingual cross-script probe for `redhop::analyze_query_set`.
//!
//! The CUAD probe in QUERY_SET_ANALYZER.md confirmed the heuristic on three
//! English workloads but explicitly listed "English-default analyzer
//! assumed" as a limit. This probe pins what the implementation *actually*
//! does on non-English query sets, so the API contract reflects measured
//! coverage, not assumed coverage.
//!
//! Method:
//!   - Synthetic templated query sets in five languages: French, German,
//!     Spanish, Chinese, Japanese. Each is a faithful translation of the
//!     CUAD-shape template ("Highlight the parts of this contract related
//!     to X that should be reviewed by a lawyer") with the placeholder X
//!     varied across queries.
//!   - Synthetic diverse natural-language query sets in the same five
//!     languages (no shared template, mixed topics).
//!   - For each language × shape, run `analyze_query_set` and check whether
//!     it flagged correctly (templated → true, diverse → false).
//!
//! Headline metric:
//!   - per language: did the templated probe fire? did the diverse probe
//!     stay quiet? Together those are the two-failure-mode test we already
//!     ran on English in `query_set_analyzer_probe.rs`.
//!
//! Honest scope: this probe uses **synthetic** translations, not real
//! workload data. The mechanism conclusion (whitespace-separated scripts
//! work; CJK does not under the current tokenizer) is robust regardless;
//! the precise share / cost magnitudes on a real non-English workload
//! could differ.
//!
//! Run: cargo run -p redhop-examples --example multilingual_query_set_probe --release

use redhop::analyzer::{analyze_query_set, DilutionCost, QuerySetReport};

fn cost_label(c: DilutionCost) -> &'static str {
    match c {
        DilutionCost::High => "high",
        DilutionCost::Medium => "medium",
        DilutionCost::Low => "low",
        DilutionCost::None => "none",
    }
}

fn report_workload(label: &str, queries: &[&str], report: &QuerySetReport, expect_templated: bool) {
    let pass = report.is_templated == expect_templated;
    let symbol = if pass { "✓" } else { "✗" };
    println!("  {symbol} {label}");
    println!("    n={}, share={:.3}, cost={}, is_templated={}", report.n_queries, report.template_word_share, cost_label(report.estimated_dilution_cost), report.is_templated);
    println!("    boilerplate (top 10): {:?}", report.boilerplate_terms.iter().take(10).collect::<Vec<_>>());
    if !pass {
        let want = if expect_templated { "templated" } else { "NOT templated" };
        println!("    → mismatch: expected {want}");
    }
    if !queries.is_empty() {
        let snippet: String = queries[0].chars().take(110).collect();
        println!("    first query: \"{snippet}{}\"", if queries[0].len() > snippet.len() { "…" } else { "" });
    }
}

// ── French ─────────────────────────────────────────────────────────────────

fn french_templated() -> Vec<&'static str> {
    vec![
        "Mettez en évidence les parties (le cas échéant) de ce contrat liées à \"Nom du Document\" qui devraient être examinées par un avocat.",
        "Mettez en évidence les parties (le cas échéant) de ce contrat liées à \"Parties\" qui devraient être examinées par un avocat.",
        "Mettez en évidence les parties (le cas échéant) de ce contrat liées à \"Date de l'Accord\" qui devraient être examinées par un avocat.",
        "Mettez en évidence les parties (le cas échéant) de ce contrat liées à \"Date d'Effet\" qui devraient être examinées par un avocat.",
        "Mettez en évidence les parties (le cas échéant) de ce contrat liées à \"Date d'Expiration\" qui devraient être examinées par un avocat.",
        "Mettez en évidence les parties (le cas échéant) de ce contrat liées à \"Terme de Renouvellement\" qui devraient être examinées par un avocat.",
    ]
}
fn french_diverse() -> Vec<&'static str> {
    vec![
        "Quelle est la capitale de la France ?",
        "Quand la Tour Eiffel a-t-elle été construite ?",
        "Qui a écrit Les Misérables ?",
        "Quelle est la plus haute montagne du monde ?",
        "À quelle vitesse vole un avion commercial ?",
        "Combien de planètes y a-t-il dans le système solaire ?",
        "Qui a peint la Joconde ?",
        "Quelle langue parle-t-on au Brésil ?",
    ]
}

// ── German ─────────────────────────────────────────────────────────────────

fn german_templated() -> Vec<&'static str> {
    vec![
        "Markieren Sie die Teile (falls vorhanden) dieses Vertrags, die sich auf \"Dokumentname\" beziehen und von einem Anwalt geprüft werden sollten.",
        "Markieren Sie die Teile (falls vorhanden) dieses Vertrags, die sich auf \"Parteien\" beziehen und von einem Anwalt geprüft werden sollten.",
        "Markieren Sie die Teile (falls vorhanden) dieses Vertrags, die sich auf \"Vertragsdatum\" beziehen und von einem Anwalt geprüft werden sollten.",
        "Markieren Sie die Teile (falls vorhanden) dieses Vertrags, die sich auf \"Wirksamkeitsdatum\" beziehen und von einem Anwalt geprüft werden sollten.",
        "Markieren Sie die Teile (falls vorhanden) dieses Vertrags, die sich auf \"Ablaufdatum\" beziehen und von einem Anwalt geprüft werden sollten.",
        "Markieren Sie die Teile (falls vorhanden) dieses Vertrags, die sich auf \"Verlängerungsfrist\" beziehen und von einem Anwalt geprüft werden sollten.",
    ]
}
fn german_diverse() -> Vec<&'static str> {
    vec![
        "Wer ist der aktuelle Bundeskanzler von Deutschland?",
        "Wann wurde die Berliner Mauer gebaut?",
        "Welche Sprache spricht man in Brasilien?",
        "Wie hoch ist der Mount Everest?",
        "Welcher Planet ist der Sonne am nächsten?",
        "Wann endete der Zweite Weltkrieg?",
        "Wer hat den Faust geschrieben?",
        "Was ist die Hauptstadt Japans?",
    ]
}

// ── Spanish ────────────────────────────────────────────────────────────────

fn spanish_templated() -> Vec<&'static str> {
    vec![
        "Marca las partes (si las hay) de este contrato relacionadas con \"Nombre del Documento\" que deberían ser revisadas por un abogado.",
        "Marca las partes (si las hay) de este contrato relacionadas con \"Partes\" que deberían ser revisadas por un abogado.",
        "Marca las partes (si las hay) de este contrato relacionadas con \"Fecha del Acuerdo\" que deberían ser revisadas por un abogado.",
        "Marca las partes (si las hay) de este contrato relacionadas con \"Fecha de Vigencia\" que deberían ser revisadas por un abogado.",
        "Marca las partes (si las hay) de este contrato relacionadas con \"Fecha de Expiración\" que deberían ser revisadas por un abogado.",
        "Marca las partes (si las hay) de este contrato relacionadas con \"Plazo de Renovación\" que deberían ser revisadas por un abogado.",
    ]
}
fn spanish_diverse() -> Vec<&'static str> {
    vec![
        "¿Cuál es la capital de España?",
        "¿Cuándo se construyó el Coliseo Romano?",
        "¿Quién escribió Don Quijote?",
        "¿Cuál es el río más largo del mundo?",
        "¿A qué velocidad vuela un avión comercial?",
        "¿Cuántos continentes hay?",
        "¿Quién pintó La Última Cena?",
        "¿Qué idioma se habla en Brasil?",
    ]
}

// ── Chinese ────────────────────────────────────────────────────────────────

fn chinese_templated() -> Vec<&'static str> {
    vec![
        "请标注本合同中与「文档名称」相关的、应由律师审核的部分（如有）。",
        "请标注本合同中与「当事人」相关的、应由律师审核的部分（如有）。",
        "请标注本合同中与「协议日期」相关的、应由律师审核的部分（如有）。",
        "请标注本合同中与「生效日期」相关的、应由律师审核的部分（如有）。",
        "请标注本合同中与「到期日期」相关的、应由律师审核的部分（如有）。",
        "请标注本合同中与「续期条款」相关的、应由律师审核的部分（如有）。",
    ]
}
fn chinese_diverse() -> Vec<&'static str> {
    vec![
        "法国的首都是什么？",
        "埃菲尔铁塔是什么时候建成的？",
        "谁写了悲惨世界？",
        "世界上最高的山是哪座？",
        "客机飞行速度是多少？",
        "太阳系有多少行星？",
        "蒙娜丽莎是谁画的？",
        "巴西人说什么语言？",
    ]
}

// ── Japanese ───────────────────────────────────────────────────────────────

fn japanese_templated() -> Vec<&'static str> {
    vec![
        "本契約のうち「文書名」に関連する、弁護士の確認が必要な部分（もしあれば）を示してください。",
        "本契約のうち「当事者」に関連する、弁護士の確認が必要な部分（もしあれば）を示してください。",
        "本契約のうち「契約日」に関連する、弁護士の確認が必要な部分（もしあれば）を示してください。",
        "本契約のうち「発効日」に関連する、弁護士の確認が必要な部分（もしあれば）を示してください。",
        "本契約のうち「満了日」に関連する、弁護士の確認が必要な部分（もしあれば）を示してください。",
        "本契約のうち「更新期間」に関連する、弁護士の確認が必要な部分（もしあれば）を示してください。",
    ]
}
fn japanese_diverse() -> Vec<&'static str> {
    vec![
        "フランスの首都は何ですか？",
        "エッフェル塔はいつ建てられましたか？",
        "誰がレ・ミゼラブルを書きましたか？",
        "世界で一番高い山は何ですか？",
        "旅客機の飛行速度はどれくらいですか？",
        "太陽系には惑星がいくつありますか？",
        "モナリザを描いたのは誰ですか？",
        "ブラジルでは何語が話されていますか？",
    ]
}

fn run_language(label: &str, templated: Vec<&str>, diverse: Vec<&str>) -> (bool, bool) {
    println!("── {label} ──");
    let r_t = analyze_query_set(&templated);
    report_workload("templated", &templated, &r_t, true);
    let r_d = analyze_query_set(&diverse);
    report_workload("diverse", &diverse, &r_d, false);
    println!();
    (r_t.is_templated, !r_d.is_templated)
}

fn main() {
    println!("redhop::analyze_query_set — multilingual probe (Latin scripts + CJK)\n");
    let mut results: Vec<(&str, bool, bool)> = Vec::new();

    let (tp, fp) = run_language("French", french_templated(), french_diverse());
    results.push(("French", tp, fp));
    let (tp, fp) = run_language("German", german_templated(), german_diverse());
    results.push(("German", tp, fp));
    let (tp, fp) = run_language("Spanish", spanish_templated(), spanish_diverse());
    results.push(("Spanish", tp, fp));
    let (tp, fp) = run_language("Chinese", chinese_templated(), chinese_diverse());
    results.push(("Chinese", tp, fp));
    let (tp, fp) = run_language("Japanese", japanese_templated(), japanese_diverse());
    results.push(("Japanese", tp, fp));

    println!("══ summary ══");
    println!("  {:<10} {:>14} {:>14}", "language", "templated→true?", "diverse→false?");
    for (lang, tp, fp) in &results {
        println!(
            "  {:<10} {:>14} {:>14}",
            lang,
            if *tp { "✓ pass" } else { "✗ fail" },
            if *fp { "✓ pass" } else { "✗ fail" },
        );
    }

    let all_pass = results.iter().all(|(_, tp, fp)| *tp && *fp);
    let cjk_broken =
        results.iter().any(|(lang, tp, fp)| (*lang == "Chinese" || *lang == "Japanese") && !(*tp && *fp));
    let latin_pass = results
        .iter()
        .filter(|(lang, _, _)| *lang == "French" || *lang == "German" || *lang == "Spanish")
        .all(|(_, tp, fp)| *tp && *fp);

    println!();
    if all_pass {
        println!("  ✓ Heuristic works across all five languages, including CJK.");
    } else if latin_pass && cjk_broken {
        println!("  ~ Latin-script multilingual is supported (French/German/Spanish all pass).");
        println!("  ✗ CJK fails — the current tokenizer (split on non-alphanumeric) does not");
        println!("    word-segment Chinese/Japanese because both scripts have no whitespace");
        println!("    between words and every character is is_alphanumeric=true. Each query");
        println!("    collapses to a single token, so no shared-token detection is possible.");
        println!("    Documenting as known limitation. The tokenizer would need a CJK-aware");
        println!("    segmenter (jieba / mecab / unicode_segmentation::unicode_words) to fix.");
    } else {
        println!("  ✗ Unexpected failures — investigate before claiming multilingual support.");
    }
}
