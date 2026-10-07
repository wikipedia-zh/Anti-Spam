use super::*;
use crate::group_settings::{Patch, SaveResult};

fn patch(revision: i64, changes: serde_json::Value) -> Patch {
    serde_json::from_value(serde_json::json!({"request_id":Uuid::new_v4().to_string(),"expected_revision":revision,"changes":changes})).unwrap()
}

#[tokio::test]
async fn panel_edits_existing_ot_text_without_changing_warning_policy() {
    let runtime = test_runtime().await;
    runtime
        .set_warn_config(-100, 7, "kick", None)
        .await
        .unwrap();
    runtime.set_ot_warn_count(-100, 2).await.unwrap();
    runtime
        .set_ot_template(-100, Some("{user} 舊文字 {count}"))
        .await
        .unwrap();
    let initial = runtime.group_settings_snapshot(-100).await.unwrap();
    assert_eq!(
        initial.ot_template.as_deref(),
        Some("{user} 舊文字 {count}")
    );
    assert_eq!(initial.default_ot_template, default_ot_template());
    let text =
        "<b>{user}</b> 請留意主題。\n目前 {count} 次。\n{button:群規}[https://example.com/rules]";
    let update = patch(initial.revision, serde_json::json!({"ot_template":text}));
    let saved = runtime
        .save_group_settings(-100, 200, update.clone(), false)
        .await
        .unwrap();
    let SaveResult::Saved(ref after) = saved else {
        panic!("not saved");
    };
    assert_eq!(after.ot_template.as_deref(), Some(text));
    assert_eq!(after.revision, initial.revision + 1);
    assert_eq!(
        runtime
            .save_group_settings(-100, 200, update, false)
            .await
            .unwrap(),
        saved
    );
    let policy = runtime.get_warn_settings(-100).await.unwrap();
    assert_eq!(policy.ot_template.as_deref(), Some(text));
    assert_eq!(
        (
            policy.threshold,
            policy.action.as_str(),
            policy.ot_warn_count
        ),
        (7, "kick", 2)
    );
    assert_eq!(
        runtime.get_warn_settings(-200).await.unwrap().ot_template,
        None
    );
    let SaveResult::Saved(reset) = runtime
        .save_group_settings(
            -100,
            200,
            patch(after.revision, serde_json::json!({"ot_template":null})),
            false,
        )
        .await
        .unwrap()
    else {
        panic!("not reset");
    };
    assert!(reset.ot_template.is_none());
    assert_eq!(
        runtime.get_warn_settings(-100).await.unwrap().ot_template,
        None
    );
}

#[tokio::test]
async fn command_text_changes_conflict_with_an_open_panel() {
    let runtime = test_runtime().await;
    let before = runtime.group_settings_snapshot(-100).await.unwrap();
    runtime
        .set_ot_template(-100, Some("New text {user}"))
        .await
        .unwrap();
    assert_eq!(
        runtime
            .save_group_settings(
                -100,
                200,
                patch(
                    before.revision,
                    serde_json::json!({"ot_template":"Old draft"})
                ),
                false
            )
            .await
            .unwrap(),
        SaveResult::Conflict
    );
    let current = runtime.group_settings_snapshot(-100).await.unwrap();
    runtime
        .set_ot_template(-100, Some("New text {user}"))
        .await
        .unwrap();
    assert_eq!(
        runtime
            .group_settings_snapshot(-100)
            .await
            .unwrap()
            .revision,
        current.revision
    );
    runtime.set_ot_template(-100, None).await.unwrap();
    assert_eq!(
        runtime
            .save_group_settings(
                -100,
                200,
                patch(current.revision, serde_json::json!({"flood":false})),
                false
            )
            .await
            .unwrap(),
        SaveResult::Conflict
    );
}

#[tokio::test]
async fn failed_text_audit_rolls_back_text_modules_and_revision() {
    let runtime = test_runtime().await;
    runtime
        .set_ot_template(-100, Some("Original"))
        .await
        .unwrap();
    let before = runtime.group_settings_snapshot(-100).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_text_audit BEFORE INSERT ON group_settings_audit BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    assert!(runtime
        .save_group_settings(
            -100,
            200,
            patch(
                before.revision,
                serde_json::json!({"ot_template":"Changed","flood":false})
            ),
            false
        )
        .await
        .is_err());
    assert_eq!(runtime.group_settings_snapshot(-100).await.unwrap(), before);
    assert_eq!(
        runtime
            .get_warn_settings(-100)
            .await
            .unwrap()
            .ot_template
            .as_deref(),
        Some("Original")
    );
}

#[tokio::test]
async fn panel_rejects_invalid_text_but_keeps_legacy_templates_on_unrelated_saves() {
    let runtime = test_runtime().await;
    for text in [
        serde_json::json!(true),
        serde_json::json!("  \n "),
        serde_json::json!("x".repeat(3501)),
        serde_json::json!("🙂".repeat(1751)),
        serde_json::json!("{user}".repeat(11)),
        serde_json::json!("Hi {button}[javascript:alert(1)]"),
        serde_json::json!("{button}[https://example.com]"),
        serde_json::json!("Hi {button}[broken"),
        serde_json::json!("Hi {button}[not a url]"),
    ] {
        assert_eq!(
            runtime
                .save_group_settings(
                    -100,
                    200,
                    patch(0, serde_json::json!({"ot_template":text})),
                    false
                )
                .await
                .unwrap(),
            SaveResult::InvalidTemplate
        );
    }
    runtime
        .with_conn(|conn| {
            conn.execute(
                "INSERT INTO group_warn_settings(chat_id,ot_template) VALUES (-100,?1)",
                ["x".repeat(5000)],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let before = runtime.group_settings_snapshot(-100).await.unwrap();
    let SaveResult::Saved(saved) = runtime
        .save_group_settings(
            -100,
            200,
            patch(before.revision, serde_json::json!({"flood":false})),
            false,
        )
        .await
        .unwrap()
    else {
        panic!("not saved");
    };
    assert_eq!(saved.ot_template, before.ot_template);
    assert!(!saved.modules["flood"]);
}

#[test]
fn older_audit_snapshots_can_still_be_replayed_after_text_settings_are_added() {
    let snapshot: crate::group_settings::Snapshot = serde_json::from_value(serde_json::json!({
        "chat_id":-100,"title":"Test","revision":3,"modules":{},"threshold_override":null
    }))
    .unwrap();
    assert!(snapshot.ot_template.is_none());
    assert_eq!(snapshot.default_ot_template, default_ot_template());
}

#[tokio::test]
async fn cancelled_requests_still_invalidate_cache_after_the_database_write() {
    for writer in 0..3 {
        let runtime = Arc::new(test_runtime().await);
        assert!(!runtime.get_group_modules(-100).await.unwrap().netban);
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let blocking = runtime.clone();
        let blocker = tokio::spawn(async move {
            blocking
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
            match writer {
                0 => {
                    saving
                        .save_group_settings(
                            -100,
                            200,
                            patch(0, serde_json::json!({"netban":true})),
                            false,
                        )
                        .await
                        .unwrap();
                }
                1 => saving.set_group_threshold(-100, Some(0.93)).await.unwrap(),
                _ => saving.set_group_module(-100, "netban", true).await.unwrap(),
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while runtime.group_module_cache.try_read().is_ok() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(runtime.group_module_cache.try_read().is_err());
        release.send(()).unwrap();
        blocker.await.unwrap();
        let cached = runtime.get_group_modules(-100).await.unwrap();
        if writer == 1 {
            assert_eq!(cached.spam_threshold_override, Some(0.93));
        } else {
            assert!(cached.netban);
        }
    }
}

#[tokio::test]
async fn panel_save_is_atomic_audited_and_invalidates_cached_modules() {
    let runtime = test_runtime().await;
    assert!(!runtime.get_group_modules(-100).await.unwrap().netban);
    let update = patch(
        0,
        serde_json::json!({"netban":true,"captcha":true,"threshold_override":0.93}),
    );
    let result = runtime
        .save_group_settings(-100, 200, update.clone(), true)
        .await
        .unwrap();
    let SaveResult::Saved(saved) = result else {
        panic!("save failed");
    };
    assert_eq!(saved.revision, 1);
    assert!(saved.modules["netban"] && saved.modules["captcha"]);
    let cached = runtime.get_group_modules(-100).await.unwrap();
    assert!(cached.netban && cached.captcha);
    assert_eq!(cached.spam_threshold_override, Some(0.93));
    assert_eq!(
        runtime
            .save_group_settings(-100, 200, update, true)
            .await
            .unwrap(),
        SaveResult::Saved(saved)
    );
    runtime
        .with_conn(|c| {
            assert_eq!(
                c.query_row("SELECT COUNT(*) FROM group_settings_audit", [], |r| r
                    .get::<_, i64>(0))?,
                1
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn stale_panel_cannot_overwrite_a_command_or_another_admin() {
    let runtime = Arc::new(test_runtime().await);
    runtime.get_group_modules(-100).await.unwrap();
    runtime
        .set_group_module(-100, "netban", true)
        .await
        .unwrap();
    assert_eq!(
        runtime
            .save_group_settings(
                -100,
                200,
                patch(0, serde_json::json!({"captcha":true})),
                false
            )
            .await
            .unwrap(),
        SaveResult::Conflict
    );
    let current = runtime.group_settings_snapshot(-100).await.unwrap();
    let (one, two) = tokio::join!(
        runtime.save_group_settings(
            -100,
            200,
            patch(current.revision, serde_json::json!({"captcha":true})),
            false
        ),
        runtime.save_group_settings(
            -100,
            201,
            patch(current.revision, serde_json::json!({"flood":false})),
            false
        )
    );
    assert_eq!(
        [one.unwrap(), two.unwrap()]
            .iter()
            .filter(|r| matches!(r, SaveResult::Saved(_)))
            .count(),
        1
    );
    runtime.set_group_threshold(-100, Some(0.91)).await.unwrap();
    assert_eq!(
        runtime
            .group_settings_snapshot(-100)
            .await
            .unwrap()
            .revision,
        3
    );
}

#[tokio::test]
async fn failed_audit_rolls_back_settings_revision_and_cache() {
    let runtime = test_runtime().await;
    let before = runtime.get_group_modules(-100).await.unwrap();
    runtime.with_conn(|c| { c.execute_batch("CREATE TRIGGER fail_settings_audit BEFORE INSERT ON group_settings_audit BEGIN SELECT RAISE(ABORT,'audit unavailable'); END;")?; Ok(()) }).await.unwrap();
    assert!(runtime
        .save_group_settings(
            -100,
            200,
            patch(0, serde_json::json!({"netban":true})),
            false
        )
        .await
        .is_err());
    let saved = runtime.group_settings_snapshot(-100).await.unwrap();
    assert_eq!(saved.revision, 0);
    assert!(!saved.modules["netban"]);
    assert_eq!(
        runtime.get_group_modules(-100).await.unwrap().netban,
        before.netban
    );
}

#[tokio::test]
async fn panel_rejects_hidden_modules_threshold_privilege_and_reused_request_ids() {
    let runtime = test_runtime().await;
    for changes in [
        serde_json::json!({"warn-pol":true}),
        serde_json::json!({"nohalal":"true"}),
        serde_json::json!({"threshold_override":0.4}),
    ] {
        assert_eq!(
            runtime
                .save_group_settings(-100, 200, patch(0, changes), true)
                .await
                .unwrap(),
            SaveResult::Invalid
        );
    }
    assert_eq!(
        runtime
            .save_group_settings(
                -100,
                200,
                patch(0, serde_json::json!({"threshold_override":0.8})),
                false
            )
            .await
            .unwrap(),
        SaveResult::Forbidden
    );
    let update = patch(0, serde_json::json!({"netban":true}));
    runtime
        .save_group_settings(-100, 200, update.clone(), false)
        .await
        .unwrap();
    assert_eq!(
        runtime
            .save_group_settings(-200, 200, update.clone(), false)
            .await
            .unwrap(),
        SaveResult::Conflict
    );
    assert_eq!(
        runtime
            .save_group_settings(-100, 201, update, false)
            .await
            .unwrap(),
        SaveResult::Conflict
    );
    assert!(!runtime
        .group_settings_snapshot(-100)
        .await
        .unwrap()
        .modules
        .contains_key("warn-pol"));
}

#[tokio::test]
async fn observing_a_new_group_preserves_default_protection_and_revision() {
    let runtime = test_runtime().await;
    runtime.record_group_seen(-100, Some("Test")).await;
    assert!(runtime.get_group_modules(-100).await.unwrap().no_contact);
    assert_eq!(
        runtime
            .group_settings_snapshot(-100)
            .await
            .unwrap()
            .revision,
        0
    );
    runtime
        .with_conn(|c| {
            Runtime::init_db(c)?;
            Ok(())
        })
        .await
        .unwrap();
    runtime
        .set_group_module(-100, "netban", true)
        .await
        .unwrap();
    assert_eq!(
        runtime
            .group_settings_snapshot(-100)
            .await
            .unwrap()
            .revision,
        1
    );
}
