use super::*;

#[tokio::test]
async fn cancelled_rule_writes_publish_before_releasing_the_cache() {
    for writer in 0..4 {
        let runtime = Arc::new(test_runtime().await);
        let id = runtime.add_spam_rule("before", "test").await.unwrap();
        if writer == 3 {
            runtime
                .with_conn(move |conn| {
                    conn.execute("UPDATE spam_rules SET pattern='after' WHERE id=?1", [id])?;
                    Ok(())
                })
                .await
                .unwrap();
        }
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
            match writer {
                0 => {
                    saving.add_spam_rule("after", "test").await.unwrap();
                }
                1 => {
                    saving.update_spam_rule_pattern(id, "after").await.unwrap();
                }
                2 => {
                    saving.delete_spam_rule(id).await.unwrap();
                }
                _ => {
                    saving.refresh_spam_rules().await.unwrap();
                }
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while runtime.spam_rules.try_write().is_ok() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let held = runtime.spam_rules.try_read().is_err();
        release.send(()).unwrap();
        blocker.await.unwrap();
        assert!(held, "writer {writer} released its cache before commit");
        let memory = runtime
            .spam_rules
            .read()
            .await
            .iter()
            .map(|r| (r.id, r.regex.as_str().to_string()))
            .collect::<Vec<_>>();
        let disk = runtime
            .with_conn(|conn| {
                Ok(Runtime::load_spam_rules(conn)?
                    .into_iter()
                    .map(|r| (r.id, r.regex.as_str().to_string()))
                    .collect::<Vec<_>>())
            })
            .await
            .unwrap();
        assert_eq!(memory, disk);
        assert_eq!(
            memory.len(),
            match writer {
                0 => 2,
                2 => 0,
                _ => 1,
            }
        );
        if writer != 2 {
            assert!(memory.iter().any(|(_, pattern)| pattern == "after"));
        }
    }
}

#[tokio::test]
async fn unreadable_rule_snapshot_rolls_back_the_mutation_and_keeps_the_old_cache() {
    let runtime = test_runtime().await;
    let id = runtime.add_spam_rule("before", "test").await.unwrap();
    runtime
        .with_conn(move |conn| {
            conn.execute("UPDATE spam_rules SET description=x'ff' WHERE id=?1", [id])?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(runtime.add_spam_rule("after", "test").await.is_err());
    assert_eq!(runtime.spam_rules.read().await.len(), 1);
    assert_eq!(runtime.spam_rules.read().await[0].regex.as_str(), "before");
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM spam_rules", [], |r| r
                    .get::<_, i64>(0))?,
                1
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn concurrent_rule_changes_leave_the_cache_equal_to_the_committed_database() {
    let runtime = test_runtime().await;
    let id = runtime.add_spam_rule("before", "test").await.unwrap();
    let (a, b, c) = tokio::join!(
        runtime.update_spam_rule_pattern(id, "after"),
        runtime.add_spam_rule("second", "test"),
        runtime.refresh_spam_rules()
    );
    a.unwrap();
    b.unwrap();
    c.unwrap();
    let memory = runtime
        .spam_rules
        .read()
        .await
        .iter()
        .map(|r| (r.id, r.regex.as_str().to_string()))
        .collect::<Vec<_>>();
    let disk = runtime
        .with_conn(|conn| {
            Ok(Runtime::load_spam_rules(conn)?
                .into_iter()
                .map(|r| (r.id, r.regex.as_str().to_string()))
                .collect::<Vec<_>>())
        })
        .await
        .unwrap();
    assert_eq!(memory, disk);
    assert_eq!(memory.len(), 2);
    assert_eq!(memory[0].1, "after");
}
