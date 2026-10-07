use super::*;
use crate::evaluation::{evaluate, Sample};

fn corpus() -> Vec<Sample> {
    (100..130)
        .flat_map(|i| {
            [
                Sample {
                    label: "spam".into(),
                    text: format!("buy promotion {i}"),
                },
                Sample {
                    label: "ham".into(),
                    text: format!("discuss article {i}"),
                },
            ]
        })
        .collect()
}

#[test]
fn duplicate_features_and_row_order_cannot_inflate_the_holdout() {
    let mut samples = corpus();
    let before = evaluate(&samples, 0.2, 0.9).unwrap();
    for i in 100..130 {
        samples.push(Sample {
            label: "spam".into(),
            text: format!("PROMOTION! {i} BUY"),
        });
    }
    samples.reverse();
    let after = evaluate(&samples, 0.2, 0.9).unwrap();
    assert_eq!(before.corpus.usable_groups, 60);
    assert_eq!(after.corpus.usable_groups, 60);
    assert_eq!(after.corpus.duplicate_rows, 30);
    assert_eq!(
        (before.train_spam, before.test_spam),
        (after.train_spam, after.test_spam)
    );
    assert_eq!(
        serde_json::to_value(before.thresholds).unwrap(),
        serde_json::to_value(after.thresholds).unwrap()
    );
}

#[test]
fn conflicting_labels_and_empty_features_are_not_silently_scored() {
    let mut samples = corpus();
    samples.extend([
        Sample {
            label: "spam".into(),
            text: "   ".into(),
        },
        Sample {
            label: "ham".into(),
            text: "! ? 1 a".into(),
        },
        Sample {
            label: "unknown".into(),
            text: "some content".into(),
        },
        Sample {
            label: "ham".into(),
            text: "promotion buy 100".into(),
        },
    ]);
    let report = evaluate(&samples, 0.2, 0.9).unwrap();
    assert_eq!(report.corpus.empty_rows, 2);
    assert_eq!(report.corpus.invalid_labels, 1);
    assert_eq!(report.corpus.conflicting_groups, 1);
    assert_eq!(report.corpus.conflicting_rows, 2);
    assert_eq!(report.corpus.usable_groups, 59);
    assert!(report.test_spam > 0 && report.test_ham > 0);
}

#[test]
fn insufficient_classes_and_invalid_parameters_do_not_produce_accuracy_claims() {
    let samples: Vec<_> = corpus().into_iter().filter(|s| s.label == "spam").collect();
    assert!(evaluate(&samples, 0.2, 0.9).unwrap().thresholds.is_empty());
    for holdout in [f64::NAN, f64::INFINITY, 0.0, 1.0] {
        assert!(evaluate(&samples, holdout, 0.9).is_err());
    }
    assert!(evaluate(&samples, 0.2, f64::NAN).is_err());
}

#[test]
fn token_scoring_keeps_the_live_classifiers_result() {
    let mut model = ModelState {
        spam_docs: 12,
        ham_docs: 10,
        ..Default::default()
    };
    model.spam_tokens.insert("buy".into(), 20);
    model.ham_tokens.insert("article".into(), 15);
    for text in [
        "",
        "BUY buy article",
        "討論條目 100",
        "https://example.org/",
    ] {
        let score = score_spam_from_text(&model, text);
        assert_eq!(score, score_debug_from_text(&model, text).score);
        assert!(score.is_finite());
    }
}
