use super::*;

const REPORTED_TEXT: &str = "带人有微信能行d我";

#[test]
fn single_character_words_survive_but_empty_and_symbol_only_text_do_not() {
    assert_eq!(
        tokenize(REPORTED_TEXT),
        ["带", "人", "有", "微", "信", "能", "行", "d", "我"]
    );
    assert_eq!(tokenize("我 I 7"), ["我", "i", "7"]);
    assert_eq!(tokenize("正常讨论 casino"), ["正常", "讨论", "casino"]);
    for text in ["", " \n\t", "🎉🎉", "！？", "/ - ."] {
        assert!(tokenize(text).is_empty(), "{text:?}");
    }
}

#[tokio::test]
async fn forced_chinese_sample_is_persisted_scored_and_reversible_after_restart() {
    let runtime = Arc::new(test_runtime().await);
    let telegram = TelegramStub::new(vec![]);
    handle_ban_mute_kick(
        telegram.bot.clone(),
        runtime.clone(),
        spam_ban_message(HOST_ID, "/sb -f", Some(REPORTED_TEXT)),
        parse_command("/sb -f"),
    )
    .await
    .unwrap();
    let case = runtime.find_active_network_ban(200).await.unwrap().unwrap();
    assert_eq!(case.status, "force_approved");
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    let report = restarted.score_debug("", REPORTED_TEXT).await.unwrap();
    assert_eq!(report.tokens.len(), 9);
    assert!(report
        .tokens
        .iter()
        .all(|t| t.spam_count == 1 && t.ham_count == 0));
    assert!(report.score > 0.5);
    assert!(restarted
        .export_training_data()
        .await
        .unwrap()
        .contains(&case.id));
    let InspectionResult::Ham { score } =
        restarted.inspect_message("", REPORTED_TEXT).await.unwrap()
    else {
        panic!("unexpected regex");
    };
    assert_eq!(score, report.score);
    reverse_ban_case(&telegram.bot, &restarted, case, HOST_ID, "Host")
        .await
        .unwrap();
    assert_eq!(restarted.model.lock().await.spam_docs, 0);
    assert!(restarted.model.lock().await.spam_tokens.is_empty());
}

#[tokio::test]
async fn regex_rules_still_apply_to_text_without_ml_tokens() {
    let runtime = test_runtime().await;
    runtime.spam_rules.write().await.push(SpamRule {
        id: 1,
        description: "Symbol rule".into(),
        regex: FancyRegex::new("^🎉+$").unwrap(),
    });
    assert!(matches!(
        runtime.inspect_message("", "🎉🎉").await.unwrap(),
        InspectionResult::Spam { score: 1.0, .. }
    ));
}

async fn seed_old_model(runtime: &Runtime) {
    runtime
        .with_conn(|conn| {
            conn.execute_batch(
                "PRAGMA user_version=40;
            INSERT INTO training_samples(label,text,case_id,created_at) VALUES
                ('spam','我 casino','existing-spam','2026-10-01'),
                ('ham','我 article','existing-ham','2026-10-01');
            INSERT INTO word_frequencies(word,spam_count,ham_count) VALUES
                ('casino',7,0),('article',0,1),('manualbias',99,0);",
            )?;
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn upgrade_recovers_only_skipped_reviewed_samples_and_preserves_existing_counts() {
    let runtime = test_runtime().await;
    seed_old_model(&runtime).await;
    for (id, action, status, text) in [
        (
            "forced",
            ActionKind::SpamBan,
            "force_approved",
            REPORTED_TEXT,
        ),
        ("reviewed", ActionKind::SpamBan, "done", REPORTED_TEXT),
        (
            "report",
            ActionKind::ReportApproved,
            "approved_and_banned",
            REPORTED_TEXT,
        ),
        (
            "ham",
            ActionKind::ReportRejected,
            "rejected_and_cleaned",
            "我有",
        ),
        ("ordinary", ActionKind::SpamBan, "done", REPORTED_TEXT),
        ("reversed", ActionKind::Unbanned, "reversed", REPORTED_TEXT),
        (
            "reversing",
            ActionKind::SpamBan,
            "reversal_pending",
            REPORTED_TEXT,
        ),
        ("failed", ActionKind::SpamBan, "ban_failed", REPORTED_TEXT),
        (
            "pending",
            ActionKind::ReportApproved,
            "ban_pending",
            REPORTED_TEXT,
        ),
        ("purged", ActionKind::SpamBan, "force_approved", "casino"),
        ("empty", ActionKind::SpamBan, "force_approved", "🎉🎉"),
        (
            "automatic",
            ActionKind::AutoBan,
            "auto_banned",
            REPORTED_TEXT,
        ),
    ] {
        let mut case = dummy_case(action, -100, 200, Utc::now());
        case.id = id.into();
        case.status = status.into();
        case.evidence_text = text.into();
        runtime.persist_case(&case).await.unwrap();
    }
    runtime.with_conn(|conn| {
        conn.execute("INSERT INTO training_reviews(case_id,decision,actor_id,decided_at) VALUES ('reviewed','approve',?1,'2026-10-01')", [HOST_ID])?;
        Ok(())
    }).await.unwrap();
    crate::maintenance::check_upgrade(
        &runtime.config.sqlite_path,
        &runtime.config.data_dir.join("repair-check"),
    )
    .unwrap();
    let upgraded = Runtime::load(runtime.config.clone()).await.unwrap();
    let model = upgraded.model.lock().await.clone();
    assert_eq!((model.spam_docs, model.ham_docs), (4, 2));
    assert_eq!(
        model.spam_tokens["casino"], 7,
        "preserve prior/manual word counts"
    );
    assert_eq!(model.spam_tokens["manualbias"], 99);
    assert_eq!(model.spam_tokens["我"], 4);
    assert_eq!(model.ham_tokens["我"], 2);
    assert_eq!(model.spam_tokens["微"], 3);
    upgraded
        .with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT case_id FROM training_samples ORDER BY case_id")?;
            let ids = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            assert_eq!(
                ids,
                [
                    "existing-ham",
                    "existing-spam",
                    "forced",
                    "ham",
                    "report",
                    "reviewed"
                ]
            );
            assert_eq!(
                conn.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))?,
                41
            );
            Ok(())
        })
        .await
        .unwrap();
    let restarted = Runtime::load(runtime.config.clone()).await.unwrap();
    assert_eq!(
        serde_json::to_value(&model).unwrap(),
        serde_json::to_value(restarted.model.lock().await.clone()).unwrap()
    );
}

#[tokio::test]
async fn failed_tokenizer_upgrade_rolls_back_counts_samples_and_schema() {
    let runtime = test_runtime().await;
    seed_old_model(&runtime).await;
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    case.status = "force_approved".into();
    case.evidence_text = REPORTED_TEXT.into();
    runtime.persist_case(&case).await.unwrap();
    runtime.with_conn(|conn| {
        conn.execute_batch("CREATE TRIGGER fail_recovery BEFORE INSERT ON training_samples BEGIN SELECT RAISE(ABORT,'injected'); END;")?;
        Ok(())
    }).await.unwrap();
    assert!(Runtime::load(runtime.config.clone()).await.is_err());
    runtime
        .with_conn(|conn| {
            assert_eq!(
                conn.query_row("PRAGMA user_version", [], |r| r.get::<_, i64>(0))?,
                40
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM training_samples", [], |r| r
                    .get::<_, i64>(0))?,
                2
            );
            assert_eq!(
                conn.query_row(
                    "SELECT COUNT(*) FROM word_frequencies WHERE word='我'",
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
