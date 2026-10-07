use super::*;
use crate::group_settings::{Patch, SaveResult};

fn patch(revision: i64, changes: serde_json::Value) -> Patch {
    serde_json::from_value(serde_json::json!({"request_id":Uuid::new_v4().to_string(),"expected_revision":revision,"changes":changes})).unwrap()
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
