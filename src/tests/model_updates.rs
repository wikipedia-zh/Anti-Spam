use super::*;

#[tokio::test]
async fn cancelled_model_writes_finish_publishing_before_releasing_the_model() {
    for writer in 0..8 {
        let runtime = Arc::new(test_runtime().await);
        let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
        runtime.persist_case(&case).await.unwrap();
        match writer {
            2 | 3 | 6 | 7 => train_spam(&runtime, "casino", Some(&case.id))
                .await
                .unwrap(),
            4 => train_ham(&runtime, "article", Some(&case.id))
                .await
                .unwrap(),
            5 => {
                train_spam(&runtime, "casino", None).await.unwrap();
                train_spam(&runtime, "casino", None).await.unwrap();
            }
            _ => {}
        }
        if writer >= 6 {
            runtime
                .with_conn(|conn| {
                    conn.execute(
                        "UPDATE word_frequencies SET spam_count=1000 WHERE word='casino'",
                        [],
                    )?;
                    Ok(())
                })
                .await
                .unwrap();
        }
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
                0 => train_spam(&saving, "casino", Some(&case.id)).await.unwrap(),
                1 => {
                    saving
                        .decide_training_review(&case.id, "approve", HOST_ID)
                        .await
                        .unwrap();
                }
                2 => {
                    saving.purge_training_by_case(&case.id).await.unwrap();
                }
                3 => {
                    saving.purge_training_by_text("casino").await.unwrap();
                }
                4 => {
                    saving
                        .undo_clean_training_sample_by_text("article")
                        .await
                        .unwrap();
                }
                5 => {
                    saving.dedupe_training_samples().await.unwrap();
                }
                6 => {
                    saving.retrain_from_samples().await.unwrap();
                }
                _ => {
                    saving.rebuild_model().await.unwrap();
                }
            }
        });
        tokio::time::timeout(Duration::from_secs(5), async {
            while runtime.model.try_lock().is_ok() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let lock_retained = runtime.model.try_lock().is_err();
        release.send(()).unwrap();
        blocker.await.unwrap();
        assert!(
            lock_retained,
            "writer {writer} released the model before the blocked transaction finished"
        );
        let memory = runtime.model.lock().await.clone();
        let disk = runtime
            .with_conn(|conn| Runtime::load_model(conn))
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&memory).unwrap(),
            serde_json::to_value(&disk).unwrap(),
            "writer {writer}"
        );
        assert_eq!(memory.ham_docs, 0);
        assert_eq!(
            memory.spam_docs,
            if matches!(writer, 2..=4) { 0 } else { 1 }
        );
        if writer >= 6 {
            assert_eq!(
                memory.spam_tokens["casino"],
                if writer == 6 { 1 } else { 1000 }
            );
        }
    }
}

#[tokio::test]
async fn failed_model_snapshot_rolls_back_the_sample_and_training_decision() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    runtime
        .with_conn(|conn| {
            conn.execute(
                "INSERT INTO word_frequencies(word,spam_count,ham_count) VALUES ('broken',-1,0)",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(runtime
        .decide_training_review(&case.id, "approve", HOST_ID)
        .await
        .is_err());
    assert_eq!(runtime.model.lock().await.spam_docs, 0);
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM training_samples", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM training_reviews", [], |r| r
                    .get::<_, i64>(0))?,
                0
            );
            assert_eq!(
                conn.query_row(
                    "SELECT COUNT(*) FROM cases WHERE netban_eligible=1",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            Ok(())
        })
        .await
        .unwrap();
}
