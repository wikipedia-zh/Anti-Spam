use super::*;
use crate::queue_admin::{Outcome, Retry, Target};
use serde_json::{json, Value};

async fn read(runtime: &Runtime, target: Target) -> Value {
    let Outcome::Ready(v) = runtime.host_queue_item(HOST_ID, target).await.unwrap() else {
        panic!("missing job")
    };
    v
}
async fn patch(runtime: &Runtime, target: Target) -> Retry {
    Retry {
        request_id: Uuid::new_v4().to_string(),
        expected_revision: read(runtime, target.clone()).await["revision"]
            .as_str()
            .unwrap()
            .into(),
        target,
    }
}
async fn network() -> (Runtime, CaseRecord, Target) {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    runtime.persist_case(&case).await.unwrap();
    for chat in [-300, -400] {
        runtime
            .set_group_module(chat, "netban", true)
            .await
            .unwrap();
    }
    runtime.enqueue_network_deliveries(&case.id).await.unwrap();
    runtime.with_conn(|c|{c.execute("UPDATE network_deliveries SET attempts=7,next_attempt_at=?1,last_error='injected error',outcome_unknown=1",[Utc::now().timestamp()+3600])?;Ok(())}).await.unwrap();
    let target = Target::Network {
        case_id: case.id.clone(),
        chat_id: -300,
    };
    (runtime, case, target)
}

#[tokio::test]
async fn retry_changes_only_the_selected_job_and_receipt_survives_completion() {
    let (runtime, case, target) = network().await;
    let p = patch(&runtime, target.clone()).await;
    let (a, b) = tokio::join!(
        runtime.retry_host_queue(HOST_ID, p.clone()),
        runtime.retry_host_queue(HOST_ID, p.clone())
    );
    let (Outcome::Ready(first), Outcome::Ready(second)) = (a.unwrap(), b.unwrap()) else {
        panic!()
    };
    assert_eq!(first, second);
    let id = case.id.clone();
    runtime.with_conn(move|c|{
        let a:(i64,bool,i64,String)=c.query_row("SELECT attempts,outcome_unknown,next_attempt_at,last_error FROM network_deliveries WHERE case_id=?1 AND chat_id=-300",[&id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)))?;
        assert_eq!(a.0,7);assert!(a.1);assert!(a.2<=Utc::now().timestamp());assert_eq!(a.3,"injected error");
        assert!(c.query_row("SELECT next_attempt_at FROM network_deliveries WHERE case_id=?1 AND chat_id=-400",[&id],|r|r.get::<_,i64>(0))?>Utc::now().timestamp()+3000);
        assert_eq!(c.query_row("SELECT COUNT(*) FROM queue_retry_requests",[],|r|r.get::<_,i64>(0))?,1);
        assert_eq!(c.query_row("SELECT COUNT(*) FROM maintainer_actions WHERE command='重試工作'",[],|r|r.get::<_,i64>(0))?,1);
        c.execute("UPDATE network_deliveries SET state='done' WHERE chat_id=-300",[])?;Ok(())
    }).await.unwrap();
    let restart = Runtime::load(runtime.config.clone()).await.unwrap();
    let Outcome::Ready(replayed) = restart.retry_host_queue(HOST_ID, p.clone()).await.unwrap()
    else {
        panic!()
    };
    assert_eq!(replayed, first);
    assert!(matches!(
        restart
            .host_queue_item(HOST_ID, target.clone())
            .await
            .unwrap(),
        Outcome::Missing
    ));
    let mut other = p.clone();
    other.target = Target::Network {
        case_id: case.id,
        chat_id: -400,
    };
    assert!(matches!(
        restart.retry_host_queue(HOST_ID, other).await.unwrap(),
        Outcome::Conflict
    ));
    let mut new = p;
    new.request_id = Uuid::new_v4().to_string();
    assert!(matches!(
        restart.retry_host_queue(HOST_ID, new).await.unwrap(),
        Outcome::Missing
    ));
}

#[tokio::test]
async fn retry_respects_cooldown_pauses_group_access_and_current_exemptions() {
    let (runtime, case, target) = network().await;
    super::operations::set(&runtime, false, false, true).await;
    let p = patch(&runtime, target.clone()).await;
    assert!(!read(&runtime, target.clone()).await["retryable"]
        .as_bool()
        .unwrap());
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Held
    ));
    super::operations::set(&runtime, false, false, false).await;
    let before = patch(&runtime, target.clone()).await;
    let obs = runtime.group_access_revision(-300).await.unwrap();
    runtime
        .record_group_access_check(-300, obs, "unavailable")
        .await
        .unwrap();
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, before).await.unwrap(),
        Outcome::Conflict
    ));
    let p = patch(&runtime, target.clone()).await;
    assert_eq!(
        read(&runtime, target.clone()).await["waiting_for_group"],
        true
    );
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Held
    ));
    let obs = runtime.group_access_revision(-300).await.unwrap();
    runtime
        .record_group_access_check(-300, obs, "present")
        .await
        .unwrap();
    runtime.delay_telegram_queue(120).await.unwrap();
    let p = patch(&runtime, target.clone()).await;
    let Outcome::Ready(v) = runtime.retry_host_queue(HOST_ID, p).await.unwrap() else {
        panic!()
    };
    assert!(v["next_attempt_at"].as_i64().unwrap() >= Utc::now().timestamp() + 110);
    let bot = TelegramStub::new(vec![]);
    deliver_network_bans(&bot.bot, &runtime, None)
        .await
        .unwrap();
    assert!(bot.requests.lock().unwrap().is_empty());
    runtime.with_conn(|c|{c.execute_batch("UPDATE telegram_retry_state SET not_before=0;UPDATE network_deliveries SET next_attempt_at=0 WHERE chat_id=-300;INSERT INTO group_whitelist(chat_id,user_id,created_at) VALUES(-300,200,'now');")?;Ok(())}).await.unwrap();
    deliver_network_bans(&bot.bot, &runtime, None)
        .await
        .unwrap();
    assert!(bot.requests.lock().unwrap().is_empty());
    let id = case.id;
    runtime
        .with_conn(move |c| {
            assert_eq!(
                c.query_row(
                    "SELECT state FROM network_deliveries WHERE case_id=?1 AND chat_id=-300",
                    [id],
                    |r| r.get::<_, String>(0)
                )?,
                "cancelled"
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn changed_or_running_work_and_rapid_retries_cannot_be_overwritten() {
    let (runtime, _, target) = network().await;
    let p = patch(&runtime, target.clone()).await;
    let guard = runtime.user_action_guard(200).await;
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p.clone()).await.unwrap(),
        Outcome::Busy
    ));
    drop(guard);
    runtime
        .with_conn(|c| {
            c.execute(
                "UPDATE network_deliveries SET attempts=attempts+1 WHERE chat_id=-300",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Conflict
    ));
    let p = patch(&runtime, target.clone()).await;
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Ready(_)
    ));
    let p = patch(&runtime, target).await;
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Busy
    ));
}

#[tokio::test]
async fn retry_audit_failure_rolls_back_schedule_and_untrusted_requests_are_rejected() {
    let (runtime, _, target) = network().await;
    let p = patch(&runtime, target.clone()).await;
    let old = read(&runtime, target.clone()).await;
    for user in [200, 555] {
        assert!(matches!(
            runtime.host_queue_item(user, target.clone()).await.unwrap(),
            Outcome::Forbidden
        ));
        assert!(matches!(
            runtime.retry_host_queue(user, p.clone()).await.unwrap(),
            Outcome::Forbidden
        ));
    }
    let mut invalid = p.clone();
    invalid.expected_revision = "z".repeat(64);
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, invalid).await.unwrap(),
        Outcome::Invalid
    ));
    assert!(serde_json::from_value::<Target>(
        json!({"kind":"network","case_id":"x","chat_id":-300,"table":"cases"})
    )
    .is_err());
    runtime.with_conn(|c|{c.execute_batch("CREATE TRIGGER fail_retry BEFORE INSERT ON queue_retry_requests BEGIN SELECT RAISE(ABORT,'injected');END;")?;Ok(())}).await.unwrap();
    assert!(runtime.retry_host_queue(HOST_ID, p.clone()).await.is_err());
    let after = read(&runtime, target).await;
    assert_eq!(old["revision"], after["revision"]);
    runtime
        .with_conn(|c| {
            assert_eq!(
                c.query_row(
                    "SELECT COUNT(*) FROM maintainer_actions WHERE command='重試工作'",
                    [],
                    |r| r.get::<_, i64>(0)
                )?,
                0
            );
            c.execute_batch("DROP TRIGGER fail_retry;")?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Ready(_)
    ));
}

#[tokio::test]
async fn retry_of_partial_origin_work_keeps_the_confirmed_ban() {
    let runtime = test_runtime().await;
    let case = dummy_case(ActionKind::AutoBan, -100, 200, Utc::now());
    let failed = TelegramStub::with_failures(vec![], vec![("sendmessage".into(), -1)]);
    execute_auto_ban(&failed.bot, &runtime, case.clone(), "test")
        .await
        .unwrap();
    let target = Target::Origin {
        case_id: case.id.clone(),
    };
    let p = patch(&runtime, target).await;
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Ready(_)
    ));
    let good = TelegramStub::new(vec![]);
    crate::origin_retry::retry_origin_bans(&good.bot, &runtime)
        .await
        .unwrap();
    assert!(!good
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(m, _)| m == "banchatmember"));
}

#[tokio::test]
async fn reversal_after_preview_never_requeues_a_ban() {
    let (runtime, case, target) = network().await;
    let p = patch(&runtime, target).await;
    let bot = TelegramStub::new(vec![]);
    reverse_ban_case(&bot.bot, &runtime, case, HOST_ID, "Host")
        .await
        .unwrap();
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Missing
    ));
    assert!(!bot
        .requests
        .lock()
        .unwrap()
        .iter()
        .any(|(m, _)| m == "banchatmember"));
}

#[tokio::test]
async fn every_queue_kind_has_a_unique_round_trippable_target() {
    let runtime = test_runtime().await;
    let mut case = dummy_case(ActionKind::SpamBan, -100, 200, Utc::now());
    case.status = "reversal_pending".into();
    runtime.persist_case(&case).await.unwrap();
    let id = case.id.clone();
    runtime.with_conn(move|c|{
        c.execute("INSERT INTO origin_ban_jobs(case_id,header,last_error) VALUES(?1,'header','failure')",[&id])?;
        c.execute("INSERT INTO network_deliveries(case_id,chat_id,last_error) VALUES(?1,-300,'failure')",[&id])?;
        c.execute("INSERT INTO network_catchups(chat_id,message_id,user_id,case_id,last_error) VALUES(-300,42,200,?1,'failure')",[&id])?;
        c.execute("INSERT INTO restriction_jobs(case_id,payload,last_error) VALUES(?1,'{}','failure')",[&id])?;
        c.execute("INSERT INTO reversal_retries(case_id,last_error) VALUES(?1,'failure')",[&id])?;
        c.execute("INSERT INTO report_deliveries(case_id,command_id,review_chat_id,last_error) VALUES(?1,42,-1,'failure')",[&id])?;
        c.execute("INSERT INTO review_updates(case_id,kind,decision,chat_id,message_id,last_error) VALUES(?1,'train','approve',-1,42,'failure')",[&id])?;
        c.execute("INSERT INTO rule_notice_jobs(id,case_id,chat_id,source_chat_id,payload,last_error) VALUES('rule-notice',?1,-1,-100,'[]','failure')",[&id])?;
        c.execute_batch("INSERT INTO warning_requests(chat_id,message_id,target_id,payload,last_error) VALUES(-100,42,200,'{}','failure');
            INSERT INTO captcha_jobs(chat_id,user_id,payload,last_error) VALUES(-100,200,'{}','failure');
            INSERT INTO group_departures(request_id,chat_id,actor_id,payload,reason,block_rejoin,action_id,created_at,last_error) VALUES('leave-request',-300,1,'{}','reason',0,1,'now','failure');")?;Ok(())
    }).await.unwrap();
    let keys = runtime
        .with_conn(|c| {
            let mut s = c.prepare(&format!(
                "SELECT job_key FROM ({})",
                crate::queue_status::WORK
            ))?;
            let rows = s.query_map([], |r| r.get::<_, String>(0))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .await
        .unwrap();
    assert_eq!(keys.len(), 11);
    let mut unique = std::collections::HashSet::new();
    for key in keys {
        assert!(unique.insert(key.clone()));
        let target: Target = serde_json::from_str(&key).unwrap();
        assert_eq!(serde_json::to_string(&target).unwrap(), key);
        let p = patch(&runtime, target.clone()).await;
        let snapshot = read(&runtime, target).await;
        assert!(snapshot.get("payload").is_none());
        assert!(snapshot.get("job_key").is_none());
        assert!(matches!(
            runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
            Outcome::Ready(_)
        ));
    }
}

#[test]
fn retry_migration_preserves_existing_data_and_restores() {
    let dir = std::env::temp_dir().join(format!("spb-retry-{}", Uuid::new_v4()));
    std::fs::create_dir(&dir).unwrap();
    let db = dir.join("bot.db");
    let mut conn = Connection::open(&db).unwrap();
    Runtime::init_db(&mut conn).unwrap();
    conn.execute_batch("DROP TABLE queue_retry_requests;PRAGMA user_version=39;")
        .unwrap();
    drop(conn);
    let result =
        serde_json::to_value(crate::maintenance::check_upgrade(&db, &dir.join("check")).unwrap())
            .unwrap();
    assert_eq!(result["schema_before"], 39);
    assert_eq!(result["schema_after"], 40);
    assert_eq!(result["restore"], "ok");
}

#[tokio::test]
async fn catchup_retry_waits_for_its_ban_but_not_for_unrelated_jobs() {
    let (runtime, case, _) = network().await;
    runtime
        .queue_network_catchup(&case.id, -300, 42, 200)
        .await
        .unwrap();
    runtime
        .with_conn(|c| {
            c.execute("UPDATE network_catchups SET last_error='old failure'", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let target = Target::Catchup {
        chat_id: -300,
        message_id: 42,
        user_id: 200,
    };
    let p = patch(&runtime, target.clone()).await;
    assert_eq!(
        read(&runtime, target.clone()).await["waiting_for_dependency"],
        true
    );
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p.clone()).await.unwrap(),
        Outcome::Held
    ));
    runtime
        .with_conn(|c| {
            c.execute(
                "UPDATE network_deliveries SET state='done' WHERE chat_id=-300",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Conflict
    ));
    let p = patch(&runtime, target).await;
    assert!(matches!(
        runtime.retry_host_queue(HOST_ID, p).await.unwrap(),
        Outcome::Ready(_)
    ));
}
