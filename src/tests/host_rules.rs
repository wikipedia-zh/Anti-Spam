use super::*;
use crate::host_rules::{Outcome, Patch, Rule, Target, Trial};
use serde_json::{json, Value};

#[tokio::test]
async fn pending_rule_bans_use_the_current_pattern_before_retrying() {
    let runtime = test_runtime().await;
    let id = runtime.add_spam_rule("oldword", "test").await.unwrap();
    let mut old = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    old.model_score = None;
    old.evidence_text = "oldword".into();
    old.matched_rule_pattern = Some(format!("REGEX@{id}"));
    runtime
        .queue_origin_bans(vec![(old.clone(), "test".into())])
        .await
        .unwrap();
    runtime
        .update_spam_rule_pattern(id, "newword")
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let bot = TelegramStub::new(vec![]);
    crate::origin_retry::retry_origin_bans(&bot.bot, &restarted)
        .await
        .unwrap();
    assert!(bot
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|(m, _)| m != "banchatmember"));
    assert_eq!(
        restarted.load_case(&old.id).await.unwrap().unwrap().status,
        "ban_failed"
    );
    let mut current = old;
    current.id = Uuid::new_v4().to_string();
    current.target_user_id = 201;
    current.target_name = "newword".into();
    assert!(execute_auto_ban(&bot.bot, &restarted, current, "test")
        .await
        .unwrap());
}

async fn snapshot(runtime: &Runtime, id: Option<i64>) -> Value {
    match runtime
        .host_rule(HOST_ID, Target { rule_id: id })
        .await
        .unwrap()
    {
        Outcome::Saved(value) => value,
        _ => panic!("snapshot failed"),
    }
}
async fn patch(runtime: &Runtime, id: Option<i64>, pattern: Option<&str>) -> Patch {
    Patch {
        request_id: Uuid::new_v4().to_string(),
        rule_id: id,
        expected_revision: snapshot(runtime, id).await["revision"].as_i64().unwrap(),
        rule: pattern.map(|p| Rule {
            pattern: p.into(),
            description: "test".into(),
        }),
    }
}
async fn save(runtime: &Runtime, request: Patch) -> Outcome {
    loop {
        let result = runtime
            .save_host_rule(HOST_ID, request.clone())
            .await
            .unwrap();
        if !matches!(result, Outcome::Busy) {
            return result;
        }
        tokio::task::yield_now().await;
    }
}
async fn trial(runtime: &Runtime, pattern: &str, text: &str) -> Value {
    loop {
        match runtime
            .test_host_rule(
                HOST_ID,
                Trial {
                    pattern: pattern.into(),
                    text: text.into(),
                },
            )
            .await
            .unwrap()
        {
            Outcome::Saved(v) => return v,
            Outcome::Busy => tokio::task::yield_now().await,
            _ => panic!("trial failed"),
        }
    }
}

#[tokio::test]
async fn rule_trials_cover_unicode_empty_matches_invalid_syntax_and_work_limits_without_writes() {
    let runtime = test_runtime().await;
    let matched = trial(&runtime, "(?<=中)文", "中文測試").await;
    assert_eq!(
        matched,
        json!({"status":"match","matched":"文","start":1,"end":2})
    );
    assert_eq!(trial(&runtime, "^$", "").await["matched"], "");
    assert_eq!(
        trial(&runtime, "spam", "normal").await["status"],
        "no_match"
    );
    assert_eq!(trial(&runtime, "[", "normal").await["status"], "invalid");
    assert_eq!(
        trial(&runtime, "(?i)(a|b|ab)*(?=c)", &"ab".repeat(40)).await["status"],
        "limit"
    );
    assert!(matches!(
        runtime
            .test_host_rule(
                200,
                Trial {
                    pattern: "spam".into(),
                    text: "spam".into()
                }
            )
            .await
            .unwrap(),
        Outcome::Forbidden
    ));
    assert!(matches!(
        runtime
            .test_host_rule(
                HOST_ID,
                Trial {
                    pattern: "a".repeat(513),
                    text: "a".into()
                }
            )
            .await
            .unwrap(),
        Outcome::Invalid
    ));
    runtime
        .with_conn(|conn| {
            for table in [
                "spam_rules",
                "training_samples",
                "cases",
                "maintainer_actions",
                "host_rule_requests",
            ] {
                assert_eq!(
                    conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r
                        .get::<_, i64>(0))?,
                    0
                );
            }
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn rule_writes_replay_after_restart_and_do_not_restore_deleted_rules() {
    let runtime = test_runtime().await;
    let request = patch(&runtime, None, Some("casino")).await;
    let Outcome::Saved(created) = save(&runtime, request.clone()).await else {
        panic!("save failed")
    };
    let id = created["rule_id"].as_i64().unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let Outcome::Saved(replayed) = save(&restarted, request.clone()).await else {
        panic!("replay failed")
    };
    assert_eq!(created, replayed);
    let update = patch(&restarted, Some(id), Some("lottery")).await;
    assert!(matches!(save(&restarted, update).await, Outcome::Saved(_)));
    assert!(restarted
        .spam_rules
        .read()
        .await
        .iter()
        .any(|r| r.id == id && regex_is_match(&r.regex, "lottery")));
    let remove = patch(&restarted, Some(id), None).await;
    assert!(matches!(
        save(&restarted, remove.clone()).await,
        Outcome::Saved(_)
    ));
    assert!(matches!(save(&restarted, request).await, Outcome::Saved(_)));
    assert!(matches!(save(&restarted, remove).await, Outcome::Saved(_)));
    assert!(restarted.spam_rules.read().await.is_empty());
    restarted
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM maintainer_actions", [], |r| r
                    .get::<_, i64>(0))?,
                3
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn command_edits_invalidate_panel_changes_and_failed_receipts_roll_back_cache_and_audit() {
    let runtime = test_runtime().await;
    let id = runtime.add_spam_rule("before", "test").await.unwrap();
    let stale = patch(&runtime, Some(id), Some("panel")).await;
    runtime
        .update_spam_rule_pattern(id, "command")
        .await
        .unwrap();
    runtime
        .update_spam_rule_pattern(id, "before")
        .await
        .unwrap();
    assert!(matches!(save(&runtime, stale).await, Outcome::Conflict));
    let request = patch(&runtime, Some(id), Some("after")).await;
    assert!(matches!(
        runtime.save_host_rule(200, request.clone()).await.unwrap(),
        Outcome::Forbidden
    ));
    runtime.with_conn(|conn|{conn.execute_batch("CREATE TRIGGER fail_receipt BEFORE INSERT ON host_rule_requests BEGIN SELECT RAISE(ABORT,'receipt failure'); END;")?;Ok(())}).await.unwrap();
    loop {
        match runtime.save_host_rule(HOST_ID, request.clone()).await {
            Ok(Outcome::Busy) => tokio::task::yield_now().await,
            Err(_) => break,
            _ => panic!("failed receipt must roll back"),
        }
    }
    assert_eq!(
        snapshot(&runtime, Some(id)).await["rule"]["pattern"],
        "before"
    );
    assert_eq!(runtime.spam_rules.read().await[0].regex.as_str(), "before");
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM maintainer_actions", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            conn.execute_batch("DROP TRIGGER fail_receipt;")?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        save(&runtime, request.clone()).await,
        Outcome::Saved(_)
    ));
    let mut reused = request;
    reused.rule.as_mut().unwrap().pattern = "other".into();
    assert!(matches!(save(&runtime, reused).await, Outcome::Conflict));
}

#[tokio::test]
async fn rule_edit_undo_restores_both_fields_and_refuses_to_overwrite_a_later_change() {
    let runtime = test_runtime().await;
    let id = runtime.add_spam_rule("before", "old name").await.unwrap();
    let mut request = patch(&runtime, Some(id), Some("after")).await;
    request.rule.as_mut().unwrap().description = "new name".into();
    assert!(matches!(save(&runtime, request).await, Outcome::Saved(_)));
    let undo = runtime
        .with_conn(|conn| {
            Ok(serde_json::from_str::<UndoData>(&conn.query_row(
                "SELECT undo_data FROM maintainer_actions",
                [],
                |r| r.get::<_, String>(0),
            )?)?)
        })
        .await
        .unwrap();
    let UndoData::RuleUpdated {
        rule_id,
        before,
        after,
    } = undo
    else {
        panic!("missing undo")
    };
    runtime.update_spam_rule_pattern(id, "third").await.unwrap();
    assert!(!runtime
        .restore_host_rule(rule_id, before.clone(), after.clone())
        .await
        .unwrap());
    runtime.update_spam_rule_pattern(id, "after").await.unwrap();
    assert!(runtime
        .restore_host_rule(rule_id, before, after)
        .await
        .unwrap());
    assert_eq!(
        snapshot(&runtime, Some(id)).await["rule"],
        json!({"pattern":"before","description":"old name"})
    );
}

#[tokio::test]
async fn cancelled_panel_rule_update_still_publishes_the_committed_cache() {
    let runtime = Arc::new(test_runtime().await);
    let request = patch(&runtime, None, Some("committed")).await;
    let (started, ready) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let blocked = runtime.clone();
    let blocker = tokio::spawn(async move {
        blocked
            .with_conn(move |_| {
                let _ = started.send(());
                let _ = wait.recv();
                Ok(())
            })
            .await
            .unwrap();
    });
    ready.await.unwrap();
    let saving = runtime.clone();
    let task = tokio::spawn(async move {
        save(&saving, request).await;
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.spam_rules.try_read().is_ok() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    release.send(()).unwrap();
    blocker.await.unwrap();
    let rules = runtime.spam_rules.read().await;
    assert_eq!(rules.len(), 1);
    assert!(regex_is_match(&rules[0].regex, "committed"));
    drop(rules);
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM host_rule_requests", [], |r| r
                    .get::<_, i64>(0))?,
                1
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn rule_search_counts_saved_exact_ids_and_migration_preserves_rules() {
    let runtime = test_runtime().await;
    let id = runtime.add_spam_rule("hello", "名稱").await.unwrap();
    let mut case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    case.matched_rule_pattern = Some(format!("REGEX@{id}；CONTACT"));
    runtime.persist_case(&case).await.unwrap();
    let mut other = case.clone();
    other.id = Uuid::new_v4().to_string();
    other.matched_rule_pattern = Some(format!("REGEX@{id}1"));
    runtime.persist_case(&other).await.unwrap();
    let found = runtime
        .host_query(crate::host_panel::Query {
            view: "rules".into(),
            search: "名稱".into(),
            offset: 0,
            filter: String::new(),
            created_from: None,
            created_before: None,
        })
        .await
        .unwrap();
    assert_eq!(found["items"][0]["recorded_hits"], 1);
    runtime.with_conn(|conn|{conn.execute_batch("DROP TRIGGER rules_revision_INSERT; DROP TRIGGER rules_revision_UPDATE; DROP TRIGGER rules_revision_DELETE; DROP TABLE rule_revision; DROP TABLE host_rule_requests; PRAGMA user_version=33;")?;Ok(())}).await.unwrap();
    let result = crate::maintenance::check_upgrade(
        &runtime.config.sqlite_path,
        &runtime.config.data_dir.join("rules-upgrade"),
    )
    .unwrap();
    let result = serde_json::to_value(result).unwrap();
    assert_eq!(result["schema_after"], 40);
    assert_eq!(result["restore"], "ok");
}
