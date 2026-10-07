use super::*;
use crate::role_updates::{Patch, Role, SaveResult};

fn patch(revision: i64, user_id: i64, role: Role, enabled: bool) -> Patch {
    Patch {
        request_id: Uuid::new_v4().to_string(),
        expected_revision: revision,
        user_id,
        role,
        enabled,
    }
}

async fn audit_count(runtime: &Runtime) -> i64 {
    runtime
        .with_conn(|conn| {
            Ok(conn.query_row("SELECT COUNT(*) FROM host_role_requests", [], |r| r.get(0))?)
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn role_change_replay_is_durable_and_does_not_repeat_a_reversed_grant() {
    let runtime = test_runtime().await;
    let request = patch(0, 300, Role::Maintainer, true);
    let (a, b) = tokio::join!(
        runtime.save_host_role(HOST_ID, request.clone()),
        runtime.save_host_role(HOST_ID, request.clone())
    );
    assert!(matches!(a.unwrap(), SaveResult::Saved(_)));
    assert!(matches!(b.unwrap(), SaveResult::Saved(_)));
    assert!(runtime.is_maintainer(300).await);
    assert_eq!(audit_count(&runtime).await, 1);
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    restarted
        .set_maintainer(300, false, Some(HOST_ID))
        .await
        .unwrap();
    assert!(matches!(
        restarted
            .save_host_role(HOST_ID, request.clone())
            .await
            .unwrap(),
        SaveResult::Saved(_)
    ));
    assert!(!restarted.is_maintainer(300).await);
    let mut changed = request;
    changed.enabled = false;
    assert!(matches!(
        restarted.save_host_role(HOST_ID, changed).await.unwrap(),
        SaveResult::Conflict
    ));
    assert_eq!(audit_count(&restarted).await, 1);
    restarted
        .with_conn(|conn| {
            let undo: String =
                conn.query_row("SELECT undo_data FROM maintainer_actions", [], |r| r.get(0))?;
            assert!(matches!(
                serde_json::from_str::<UndoData>(&undo)?,
                UndoData::Maintainer {
                    user_id: 300,
                    old_enabled: false
                }
            ));
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn commands_and_other_panel_writes_invalidate_a_stale_role_snapshot() {
    let runtime = test_runtime().await;
    let old = runtime.role_snapshot(300).await.unwrap();
    runtime
        .set_reviewer(300, true, Some(HOST_ID))
        .await
        .unwrap();
    runtime
        .set_reviewer(300, false, Some(HOST_ID))
        .await
        .unwrap();
    assert!(matches!(
        runtime
            .save_host_role(HOST_ID, patch(old.revision, 300, Role::Reviewer, true))
            .await
            .unwrap(),
        SaveResult::Conflict
    ));
    let revision = runtime.role_snapshot(300).await.unwrap().revision;
    let (a, b) = tokio::join!(
        runtime.save_host_role(HOST_ID, patch(revision, 300, Role::Reviewer, true)),
        runtime.save_host_role(HOST_ID, patch(revision, 400, Role::Maintainer, true))
    );
    let successes = [a.unwrap(), b.unwrap()]
        .iter()
        .filter(|v| matches!(v, SaveResult::Saved(_)))
        .count();
    assert_eq!(successes, 1);
    assert_eq!(audit_count(&runtime).await, 1);
}

#[tokio::test]
async fn failed_role_receipt_rolls_back_role_audit_revision_and_cache() {
    let runtime = test_runtime().await;
    runtime.with_conn(|conn| {conn.execute_batch("CREATE TRIGGER fail_receipt BEFORE INSERT ON host_role_requests BEGIN SELECT RAISE(ABORT,'receipt failed'); END;")?;Ok(())}).await.unwrap();
    assert!(runtime
        .save_host_role(HOST_ID, patch(0, 300, Role::Maintainer, true))
        .await
        .is_err());
    assert!(!runtime.is_maintainer(300).await);
    assert_eq!(runtime.role_snapshot(300).await.unwrap().revision, 0);
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM maintainer_actions", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM maintainers", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn cancelled_role_writes_keep_the_cache_and_database_in_sync() {
    for panel in [false, true] {
        let runtime = Arc::new(test_runtime().await);
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
            if panel {
                saving
                    .save_host_role(HOST_ID, patch(0, 300, Role::Maintainer, true))
                    .await
                    .unwrap();
            } else {
                saving
                    .set_maintainer(300, true, Some(HOST_ID))
                    .await
                    .unwrap();
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while runtime.maintainers.try_write().is_ok() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let held = runtime.maintainers.try_read().is_err();
        release.send(()).unwrap();
        blocker.await.unwrap();
        assert!(held);
        assert!(runtime.is_maintainer(300).await);
        assert!(runtime.role_snapshot(300).await.unwrap().maintainer);
        assert_eq!(audit_count(&runtime).await, if panel { 1 } else { 0 });
    }
}

#[tokio::test]
async fn role_editor_cannot_grant_host_access_or_remove_the_host() {
    let runtime = test_runtime().await;
    assert!(matches!(
        runtime
            .save_host_role(123, patch(0, 300, Role::Maintainer, true))
            .await
            .unwrap(),
        SaveResult::Forbidden
    ));
    assert!(matches!(
        runtime
            .save_host_role(HOST_ID, patch(0, HOST_ID, Role::Maintainer, false))
            .await
            .unwrap(),
        SaveResult::Forbidden
    ));
    assert!(matches!(
        runtime
            .save_host_role(HOST_ID, patch(0, -100, Role::Maintainer, true))
            .await
            .unwrap(),
        SaveResult::Invalid
    ));
    assert!(serde_json::from_value::<Patch>(serde_json::json!({"request_id":Uuid::new_v4(),"user_id":300,"expected_revision":0,"role":"host","enabled":true})).is_err());
    assert_eq!(audit_count(&runtime).await, 0);
    assert!(runtime.is_maintainer(HOST_ID).await);
}
