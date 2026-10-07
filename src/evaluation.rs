use super::*;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub(super) struct Sample {
    pub label: String,
    pub text: String,
}

#[derive(Default, Serialize)]
pub(super) struct CorpusSummary {
    pub rows: usize,
    pub empty_rows: usize,
    pub invalid_labels: usize,
    pub duplicate_rows: usize,
    pub conflicting_groups: usize,
    pub conflicting_rows: usize,
    pub usable_groups: usize,
}

#[derive(Serialize)]
pub(super) struct ThresholdResult {
    threshold: f64,
    true_positive: usize,
    false_positive: usize,
    false_negative: usize,
    true_negative: usize,
    precision: Option<f64>,
    recall: Option<f64>,
}

#[derive(Serialize)]
pub(super) struct Report {
    pub live_threshold: f64,
    pub corpus: CorpusSummary,
    pub train_spam: usize,
    pub train_ham: usize,
    pub test_spam: usize,
    pub test_ham: usize,
    pub thresholds: Vec<ThresholdResult>,
    pub limitation: &'static str,
}

struct Group {
    tokens: Vec<String>,
    spam: bool,
}

fn groups(samples: &[Sample]) -> (CorpusSummary, Vec<Group>) {
    let mut summary = CorpusSummary {
        rows: samples.len(),
        ..Default::default()
    };
    let mut counts: BTreeMap<Vec<String>, (usize, usize)> = BTreeMap::new();
    for sample in samples {
        if !matches!(sample.label.as_str(), "spam" | "ham") {
            summary.invalid_labels += 1;
            continue;
        }
        let mut tokens = tokenize(&sample.text);
        if tokens.is_empty() {
            summary.empty_rows += 1;
            continue;
        }
        // The classifier ignores order and case. Identical feature bags must
        // stay together even if their source text differs in punctuation.
        tokens.sort();
        let (spam, ham) = counts.entry(tokens).or_default();
        if sample.label == "spam" {
            *spam += 1;
        } else {
            *ham += 1;
        }
    }
    let mut groups = Vec::new();
    for (tokens, (spam, ham)) in counts {
        if spam > 0 && ham > 0 {
            summary.conflicting_groups += 1;
            summary.conflicting_rows += spam + ham;
            continue;
        }
        summary.duplicate_rows += spam + ham - 1;
        groups.push(Group {
            tokens,
            spam: spam > 0,
        });
    }
    summary.usable_groups = groups.len();
    (summary, groups)
}

fn split_key(tokens: &[String]) -> u64 {
    // A fixed FNV-1a ordering keeps the split reproducible across processes.
    let mut hash = 0xcbf29ce484222325u64;
    for token in tokens {
        for byte in token.as_bytes().iter().copied().chain([0]) {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    hash
}

pub(super) fn evaluate(samples: &[Sample], holdout: f64, live: f64) -> Result<Report> {
    anyhow::ensure!(
        holdout.is_finite() && (0.05..=0.5).contains(&holdout),
        "invalid holdout fraction"
    );
    anyhow::ensure!(
        live.is_finite() && (0.0..=1.0).contains(&live),
        "invalid threshold"
    );
    let (corpus, mut groups) = groups(samples);
    groups.sort_by(|a, b| {
        split_key(&a.tokens)
            .cmp(&split_key(&b.tokens))
            .then(a.tokens.cmp(&b.tokens))
    });
    let step = (1.0 / holdout).round().max(2.0) as usize;
    let mut model = ModelState::default();
    let mut test = Vec::new();
    let mut class_counts = [0usize; 2];
    let (mut train_spam, mut train_ham, mut test_spam, mut test_ham) = (0, 0, 0, 0);
    for group in groups {
        let count = &mut class_counts[usize::from(group.spam)];
        let held_out = (*count).is_multiple_of(step);
        *count += 1;
        if held_out {
            if group.spam {
                test_spam += 1;
            } else {
                test_ham += 1;
            }
            test.push(group);
        } else {
            let tokens = if group.spam {
                train_spam += 1;
                model.spam_docs += 1;
                &mut model.spam_tokens
            } else {
                train_ham += 1;
                model.ham_docs += 1;
                &mut model.ham_tokens
            };
            for token in group.tokens {
                *tokens.entry(token).or_default() += 1;
            }
        }
    }
    let valid = corpus.usable_groups >= 20
        && train_spam > 0
        && train_ham > 0
        && test_spam > 0
        && test_ham > 0;
    let mut thresholds = vec![0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, live];
    thresholds.sort_by(f64::total_cmp);
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let scores: Vec<_> = if valid {
        test.iter()
            .map(|g| (g.spam, score_spam_from_tokens(&model, &g.tokens)))
            .collect()
    } else {
        Vec::new()
    };
    let thresholds = if valid {
        thresholds
            .into_iter()
            .map(|threshold| {
                let (mut tp, mut fp, mut fnn, mut tn) = (0, 0, 0, 0);
                for (spam, score) in &scores {
                    match (*spam, passes_threshold(*score, threshold)) {
                        (true, true) => tp += 1,
                        (false, true) => fp += 1,
                        (true, false) => fnn += 1,
                        (false, false) => tn += 1,
                    }
                }
                ThresholdResult {
                    threshold,
                    true_positive: tp,
                    false_positive: fp,
                    false_negative: fnn,
                    true_negative: tn,
                    precision: (tp + fp > 0).then(|| tp as f64 / (tp + fp) as f64),
                    recall: (tp + fnn > 0).then(|| tp as f64 / (tp + fnn) as f64),
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    Ok(Report { live_threshold: live, corpus, train_spam, train_ham, test_spam, test_ham, thresholds,
        limitation: "Retrospective evaluation of existing labels; not a live false-positive rate or a recommendation to change the threshold." })
}

pub(super) fn format_report(report: &Report, live: f64) -> String {
    let c = &report.corpus;
    let mut out = format!("<b>模型評估</b>\n原始 {} 筆 → 可用 {} 組\n排除：無有效詞 {}、無效標籤 {}、重複 {}、標籤衝突 {} 組（{} 筆）\n訓練：垃圾 {} / 正常 {}\n測試：垃圾 {} / 正常 {}\n目前門檻 {live:.2}",
        c.rows, c.usable_groups, c.empty_rows, c.invalid_labels, c.duplicate_rows, c.conflicting_groups, c.conflicting_rows,
        report.train_spam, report.train_ham, report.test_spam, report.test_ham);
    if report.thresholds.is_empty() {
        out.push_str("\n\n資料不足：至少需要 20 組，且訓練及測試均須有垃圾和正常樣本。");
        return out;
    }
    out.push_str("\n\n<code>門檻  精確率  召回率  漏放  誤判</code>");
    for row in &report.thresholds {
        let p = row
            .precision
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "  —  ".to_string());
        let r = row
            .recall
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "  —  ".to_string());
        let mark = if (row.threshold - live).abs() < 1e-9 {
            " ←"
        } else {
            ""
        };
        out.push_str(&format!(
            "\n<code>{:.2}   {p}   {r}   {:>3}   {:>3}</code>{mark}",
            row.threshold, row.false_negative, row.false_positive
        ));
    }
    out.push_str("\n\n相同詞袋只計一次，不跨訓練及測試集；衝突標籤暫不採用。\n漏放／誤判只指這批歷史標籤，不能當作實際群組誤封率。正式模型未改動。");
    out
}

pub(super) fn run_offline(path: &std::path::Path) -> Result<()> {
    #[derive(Deserialize)]
    struct Snapshot {
        threshold: f64,
        samples: Vec<Sample>,
    }
    anyhow::ensure!(
        std::fs::metadata(path)?.len() <= 32 * 1024 * 1024,
        "snapshot exceeds 32 MiB"
    );
    let snapshot: Snapshot = serde_json::from_slice(&std::fs::read(path)?)?;
    let report = evaluate(&snapshot.samples, 0.2, snapshot.threshold)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
