use super::*;
use crate::host_model::{Outcome, Request};
use serde_json::{json, Value};

async fn read(runtime: &Runtime, value: Value) -> Value {
    for _ in 0..1000 {
        match runtime
            .host_model(HOST_ID, serde_json::from_value(value.clone()).unwrap())
            .await
            .unwrap()
        {
            Outcome::Ready(value) => return value,
            Outcome::Busy => tokio::time::sleep(Duration::from_millis(5)).await,
            _ => panic!("unexpected model response"),
        }
    }
    panic!("model inspection did not finish")
}

#[tokio::test]
async fn model_trials_use_the_live_model_and_never_write_or_train() {
    let runtime = test_runtime().await;
    runtime
        .train_atomic("spam", "casino offer", None)
        .await
        .unwrap();
    runtime
        .train_atomic("ham", "hello friends", None)
        .await
        .unwrap();
    runtime.set_threshold(0.9).await.unwrap();
    let before = serde_json::to_value(runtime.model.lock().await.clone()).unwrap();
    let db_before = runtime
        .with_conn(|conn| Ok(conn.total_changes()))
        .await
        .unwrap();
    let score = read(&runtime, json!({"action":"score","text":"casino offer"})).await;
    assert_eq!(
        score["score"],
        json!(score_spam_from_text(
            &*runtime.model.lock().await,
            "casino offer"
        ))
    );
    assert_eq!(score["global_threshold"], 0.9);
    assert_eq!(
        score["passes_global"],
        passes_threshold(score["score"].as_f64().unwrap(), 0.9)
    );
    assert!(score["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t["token"] == "casino" && t["spam_count"] == 1));
    assert_eq!(
        read(&runtime, json!({"action":"score","text":""})).await["token_count"],
        0
    );
    let summary = read(&runtime, json!({"action":"summary"})).await;
    assert_eq!(summary["samples"]["rows"], 2);
    assert_eq!(summary["model"]["spam_docs"], 1);
    assert_eq!(summary["training_reviews"]["rejected"], 0);
    let small = read(&runtime, json!({"action":"evaluate"})).await;
    assert_eq!(small["status"], "ready");
    assert!(small["report"]["thresholds"].as_array().unwrap().is_empty());
    assert_eq!(
        serde_json::to_value(runtime.model.lock().await.clone()).unwrap(),
        before
    );
    assert_eq!(
        runtime
            .with_conn(|conn| Ok(conn.total_changes()))
            .await
            .unwrap(),
        db_before
    );
    assert!(matches!(
        runtime.host_model(200, Request::Summary {}).await.unwrap(),
        Outcome::Forbidden
    ));
    assert!(matches!(
        runtime
            .host_model(
                HOST_ID,
                Request::Score {
                    text: "字".repeat(4001)
                }
            )
            .await
            .unwrap(),
        Outcome::Invalid
    ));
}

#[tokio::test]
async fn evaluations_use_one_complete_snapshot_and_report_conflicting_labels() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn|{
        for label in ["spam","ham"] { for n in 0..30 {
            conn.execute("INSERT INTO training_samples(label,text,created_at) VALUES (?1,?2,'2026-01-01')",params![label,format!("{label} word{n}")])?;
        }}
        for (label,text) in [("spam","repeat"),("spam","repeat"),("spam","conflict"),("ham","conflict"),("ham",""),("unknown","bad")]{
            conn.execute("INSERT INTO training_samples(label,text,created_at) VALUES (?1,?2,'2026-01-01')",params![label,text])?;
        }
        Ok(())
    }).await.unwrap();
    let result = read(&runtime, json!({"action":"evaluate"})).await;
    assert_eq!(result["report"]["corpus"]["rows"], 66);
    assert_eq!(result["report"]["corpus"]["conflicting_groups"], 1);
    assert_eq!(result["report"]["corpus"]["duplicate_rows"], 1);
    assert_eq!(result["report"]["corpus"]["invalid_labels"], 1);
    assert_eq!(result["report"]["corpus"]["empty_rows"], 1);
    assert!(!result["report"]["thresholds"]
        .as_array()
        .unwrap()
        .is_empty());
    let again = read(&runtime, json!({"action":"evaluate"})).await;
    assert_eq!(result["sample_fingerprint"], again["sample_fingerprint"]);
    assert_eq!(result["report"], again["report"]);
    runtime
        .train_atomic("ham", "another sample", None)
        .await
        .unwrap();
    let changed = read(&runtime, json!({"action":"evaluate"})).await;
    assert_ne!(changed["sample_fingerprint"], result["sample_fingerprint"]);
    assert_eq!(changed["report"]["corpus"]["rows"], 67);
}

#[tokio::test]
async fn invalid_thresholds_and_large_corpora_are_not_silently_evaluated() {
    let runtime = test_runtime().await;
    runtime.set_threshold(f64::NAN).await.unwrap();
    assert!(read(&runtime, json!({"action":"summary"})).await["global_threshold"].is_null());
    let score = read(&runtime, json!({"action":"score","text":"hello"})).await;
    assert!(score["global_threshold"].is_null());
    assert!(score["passes_global"].is_null());
    assert_eq!(
        read(&runtime, json!({"action":"evaluate"})).await["status"],
        "invalid_threshold"
    );
    runtime.set_threshold(0.9).await.unwrap();
    runtime.with_conn(|conn|{conn.execute("INSERT INTO training_samples(label,text,created_at) VALUES ('spam',?1,'2026-01-01')",["x".repeat(65537)])?;Ok(())}).await.unwrap();
    let result = read(&runtime, json!({"action":"evaluate"})).await;
    assert_eq!(result["status"], "too_large");
    assert!(result.get("report").is_none());
    runtime.with_conn(|conn|{conn.execute("DELETE FROM training_samples",[])?;conn.execute("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<20001) INSERT INTO training_samples(label,text,created_at) SELECT 'spam','x','2026-01-01' FROM n",[])?;Ok(())}).await.unwrap();
    assert_eq!(
        read(&runtime, json!({"action":"evaluate"})).await["status"],
        "too_large"
    );
}

#[tokio::test]
async fn model_override_counts_and_links_exclude_denied_groups() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn|{
        conn.execute_batch("INSERT INTO group_module_settings(chat_id,spam_threshold_override) VALUES (-100,0.6),(-200,0.7),(-300,NULL);
            INSERT INTO banned_groups(chat_id,reason,created_at) VALUES (-200,'test','2026-01-01');")?;Ok(())
    }).await.unwrap();
    let result = read(&runtime, json!({"action":"summary"})).await;
    assert_eq!(result["group_overrides"], 1);
    let query: crate::host_panel::Query =
        serde_json::from_value(json!({"view":"groups","filter":"overrides"})).unwrap();
    let groups = runtime.host_query(query).await.unwrap();
    assert_eq!(groups["items"].as_array().unwrap().len(), 1);
    assert_eq!(groups["items"][0]["chat_id"], -100);
    assert!(runtime
        .host_query(serde_json::from_value(json!({"view":"cases","filter":"overrides"})).unwrap())
        .await
        .is_err());
}
